import glob
import os

import polars as pl

from .base import BaseLoader

__all__ = ['LogfmtLoader']

"""
LogfmtLoader Class

One loader for logfmt - the `key=value key2="value with spaces"` convention that came out of Heroku
and is what Grafana, Loki, Prometheus, Docker, Consul and most of the Go ecosystem emit. Every line
carries its own field names, so unlike an access log there is no layout to configure and unlike JSON
there is usually no mapping to configure either: the key names are conventional, so `ts`/`time`/
`timestamp`/`t` become m_timestamp, `msg`/`message` becomes m_message and `level`/`lvl`/`severity`
becomes level, with no spec file involved (docs/log-format-support.md section 5 item 2).

    LogfmtLoader(filename="app.log").execute()                          # zero configuration
    LogfmtLoader(filename="logs", filename_pattern="*.txt").execute()   # a tree of files
    LogfmtLoader(filename="app.log", message_field="event").execute()   # a log that names it oddly

Each distinct key becomes its own column, so a mixed set of files simply yields a wide frame with
nulls where a line did not carry that key - the same shape JsonLoader produces from heterogeneous
JSON, and for the same reason.

- filename (str): file, or directory to walk when filename_pattern is given.
- filename_pattern (str, optional): glob applied within each subdirectory, as in RawLoader. When
  given, a 'file_name' column is added.
- min_file_size (int, optional): skip files smaller than this many bytes.
- strip_full_data_path (str, optional): prefix removed from 'file_name'.
- timestamp_field (str, optional): key parsed into 'm_timestamp'. Omit to use whichever of
  ts/timestamp/time/t the data contains.
- timestamp_formats (list, optional): candidate chrono formats. The best one is chosen on a sample
  and then applied to the whole column in one vectorized pass. Omit to let Polars infer.
- timestamp_naive (bool): strip the time zone from 'm_timestamp'. On by default because every other
  LogLead loader produces naive datetimes, and Polars refuses to concat naive with tz-aware frames.
  logfmt timestamps are normally RFC3339 with a zone, so this is the common case, not a corner one.
- message_field (str, optional): key used as 'm_message'. Omit to use msg/message.
- level_field (str, optional): key copied to 'level'. Omit to use level/lvl/severity.
- seq_id_field (str, optional): key copied to 'seq_id'. No default - which key correlates records
  is the one thing the convention does not settle.
- extra_fields (str): 'keep' (default) leaves every key as its own column; 'drop' keeps only the
  mapped columns.

Where the automatic mapping is asked for and the data uses more than one of the candidate keys,
they are coalesced rather than the first one winning: one real file mixes `t=` and `ts=` lines from
two different Grafana components, and picking either alone leaves half of m_timestamp null.

A line with no key=value pairs at all is kept, with the line itself as m_message and nulls
elsewhere - plain-text banners and stack traces are interleaved into real logfmt output, and
dropping them would lose log content rather than noise. For the same reason m_message falls back to
the raw line whenever the message key is missing from a line, so it is never null.

Values are unquoted and unescaped (\\", \\n, \\t, \\\\); everything stays a string, since logfmt
carries no types.
"""

# A logfmt pair: a key, '=', then either a quoted value - which may hold spaces, '=' and escaped
# quotes - or a run of non-whitespace. The quoted branch is first so it wins, which is what keeps
# the '=' inside instance="datasource_uid=x, ref_id=A" from being read as another pair.
_PAIR = r'[A-Za-z_][A-Za-z0-9_./-]*=(?:"(?:[^"\\]|\\.)*"|[^\s]*)'

# The key names the convention settles on, taken from lnav's logfmt_log format.
_TIMESTAMP_KEYS = ("ts", "timestamp", "time", "t")
_MESSAGE_KEYS = ("msg", "message")
_LEVEL_KEYS = ("level", "lvl", "severity")

_RAW = "_logfmt_raw_line"
_ROW = "_logfmt_row"


class LogfmtLoader(BaseLoader):
    def __init__(self, filename, filename_pattern=None, min_file_size=0,
                 strip_full_data_path=None, timestamp_field=None, timestamp_formats=None,
                 timestamp_naive=True, message_field=None, level_field=None, seq_id_field=None,
                 extra_fields="keep"):

        if extra_fields not in ("keep", "drop"):
            raise ValueError(f"extra_fields must be 'keep' or 'drop', got {extra_fields!r}")

        self.filename_pattern = filename_pattern
        self.min_file_size = min_file_size
        self.strip_full_data_prefix = strip_full_data_path
        self.timestamp_field = timestamp_field
        self.timestamp_formats = timestamp_formats
        self.timestamp_naive = timestamp_naive
        self.message_field = message_field
        self.level_field = level_field
        self.seq_id_field = seq_id_field
        self.extra_fields = extra_fields
        # Filled in by preprocess(): whether m_timestamp is mandatory depends on whether the data
        # turned out to have a timestamp key at all.
        self._mandatory_columns = []
        super().__init__(filename)

    # Loading ---------------------------------------------------------------------------------

    def load(self):
        paths = self._collect_paths()
        add_file_col = bool(self.filename_pattern)
        frames = pl.collect_all([self._scan_lines(path, add_file_col) for path in paths])
        lines = frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical_relaxed")
        self.df = self._split_pairs(lines, add_file_col)

    def _collect_paths(self):
        if not self.filename_pattern:
            return [self.filename]

        paths = []
        for subdir, _, _ in os.walk(self.filename):
            for path in sorted(glob.glob(os.path.join(subdir, self.filename_pattern))):
                if os.path.getsize(path) <= self.min_file_size:
                    continue
                paths.append(path)
        if not paths:
            raise ValueError(f"No files matching pattern {self.filename_pattern} "
                             f"in directory {os.path.abspath(self.filename)}. "
                             f"Check the pattern and min_file_size.")
        return paths

    def _scan_lines(self, path, add_file_col=False):
        """Read a file as whole lines.

        This is RawLoader's read: an impossible separator and quote_char=None, so Polars hands back
        whole lines instead of trying to find fields in them. quote_char matters more here than
        elsewhere - logfmt lines are full of quotes, and letting the CSV reader interpret them
        would join lines together.
        """
        query = pl.scan_csv(path, has_header=False, schema={"column_1": pl.String},
                            infer_schema=False, quote_char=None, separator=self._csv_separator,
                            encoding="utf8-lossy", truncate_ragged_lines=True,
                            **({"include_file_paths": "file_name"} if add_file_col else {}))
        query = query.select(
            [pl.col("column_1").alias(_RAW)] + ([pl.col("file_name")] if add_file_col else []))
        if add_file_col and self.strip_full_data_prefix:
            query = query.with_columns(
                pl.col("file_name").str.strip_prefix(self.strip_full_data_prefix).alias("file_name"))
        return query

    def _split_pairs(self, lines, add_file_col):
        """Turn one column of lines into one column per key.

        Which keys exist is data, not schema, so the pairs are extracted into (row, key, value)
        rows and pivoted back out. That is what makes a file whose lines carry different keys - the
        normal case - land as a wide frame with nulls rather than needing a declared schema.
        """
        lines = lines.with_row_index(_ROW)
        pairs = (lines.select(_ROW, pl.col(_RAW).str.extract_all(_PAIR).alias("_pair"))
                      .explode("_pair")
                      .drop_nulls("_pair")
                      .with_columns(pl.col("_pair").str.splitn("=", 2)
                                      .struct.rename_fields(["_key", "_value"]).alias("_kv"))
                      .unnest("_kv")
                      .with_columns(self._unescape(pl.col("_value")).alias("_value")))
        if not pairs.height:
            raise ValueError(f"No key=value pairs found in {os.path.abspath(self.filename)}. "
                             f"This does not look like logfmt. First line: "
                             f"{str(lines.item(0, _RAW))[:200] if lines.height else '(empty)'}")

        wide = pairs.pivot(on="_key", index=_ROW, values="_value", aggregate_function="first")
        carried = lines.select([_ROW, _RAW] + (["file_name"] if add_file_col else []))
        return carried.join(wide, on=_ROW, how="left").drop(_ROW)

    @staticmethod
    def _unescape(value):
        """Drop the surrounding quotes of a quoted value and undo its escapes."""
        quoted = value.str.starts_with('"') & value.str.ends_with('"') & (value.str.len_chars() > 1)
        body = (value.str.slice(1, value.str.len_chars() - 2)
                     .str.replace_all('\\"', '"', literal=True)
                     .str.replace_all('\\n', '\n', literal=True)
                     .str.replace_all('\\t', '\t', literal=True)
                     .str.replace_all('\\\\', '\\', literal=True))
        return pl.when(quoted).then(body).otherwise(value)

    # Mapping ---------------------------------------------------------------------------------

    def preprocess(self):
        message = self._map(self.message_field, _MESSAGE_KEYS, "message_field")
        # The raw line is the last resort, so m_message is never null: a line that carried no msg
        # key still said something, and a null there breaks every enhancer downstream.
        mapped = [pl.coalesce([message, pl.col(_RAW)] if message is not None else [pl.col(_RAW)])
                    .alias("m_message")]

        level = self._map(self.level_field, _LEVEL_KEYS, "level_field")
        if level is not None:
            mapped.append(level.alias("level"))
        seq_id = self._map(self.seq_id_field, (), "seq_id_field")
        if seq_id is not None:
            mapped.append(seq_id.alias("seq_id"))
        self.df = self.df.with_columns(mapped)

        timestamp = self._map(self.timestamp_field, _TIMESTAMP_KEYS, "timestamp_field")
        if timestamp is not None:
            self._add_timestamp(timestamp)

        self._mandatory_columns = ["m_message"] + (["m_timestamp"] if timestamp is not None else [])

        if self.extra_fields == "drop":
            keep = [c for c in ("m_timestamp", "m_message", "level", "seq_id", "file_name")
                    if c in self.df.columns]
            self.df = self.df.select(keep)
        else:
            self.df = self.df.drop(_RAW)

    def _map(self, explicit, candidates, argument):
        """The expression for a mapped column: the named key, or the conventional ones present.

        Coalescing rather than taking the first match matters - one file can mix `t=` and `ts=`
        lines when it holds the output of two components, and either one alone is half null.
        """
        if explicit:
            if explicit not in self.df.columns:
                raise ValueError(f"{argument}={explicit!r} is not a key in the data. Keys found: "
                                 f"{', '.join(sorted(c for c in self.df.columns if c != _RAW)[:20])}")
            return pl.col(explicit)
        present = [key for key in candidates if key in self.df.columns]
        if not present:
            return None
        return pl.col(present[0]) if len(present) == 1 else pl.coalesce([pl.col(k) for k in present])

    def _add_timestamp(self, source):
        strings = self.df.select(source.alias("t")).to_series()
        chosen = self._pick_format(strings, self.timestamp_formats)
        parsed = strings.str.to_datetime(format=chosen, strict=False)
        if self.timestamp_naive and getattr(parsed.dtype, "time_zone", None):
            # logfmt timestamps are usually RFC3339 with a zone; every other LogLead loader yields
            # naive datetimes, and mixing the two breaks pl.concat.
            parsed = parsed.dt.replace_time_zone(None)
        if getattr(parsed.dtype, "time_unit", None) != "us":
            # Same reason: nanosecond timestamps parse to Datetime('ns'), which will not concat
            # with the Datetime('us') every other loader produces.
            parsed = parsed.cast(pl.Datetime("us", getattr(parsed.dtype, "time_zone", None)))
        unparsed = parsed.null_count() - strings.null_count()
        if unparsed > 0:
            print(f"WARNING! LogfmtLoader could not parse {unparsed} of {len(parsed)} timestamp "
                  f"values into m_timestamp. Check timestamp_formats.")
        self.df = self.df.with_columns(parsed.alias("m_timestamp"))

    @staticmethod
    def _pick_format(strings, formats):
        """Decide the format on a sample, so the whole column is parsed exactly once."""
        if not formats:
            return None  # let Polars infer
        if len(formats) == 1:
            return formats[0]
        sample = strings.drop_nulls().head(1000)
        if not len(sample):
            return formats[0]
        best, best_hits = formats[0], -1
        for candidate in formats:
            hits = sample.str.to_datetime(format=candidate, strict=False).is_not_null().sum()
            if hits > best_hits:
                best, best_hits = candidate, hits
            if hits == len(sample):
                break
        return best

    # Checks ----------------------------------------------------------------------------------

    def check_for_nulls_and_non_utf8(self):
        """
        Same reasoning as JsonLoader's override: BaseLoader prints a four-line warning per column
        that has nulls, and in logfmt a sparse column is the designed outcome, not a defect - each
        line carries only the keys that were relevant to it. Summarize instead, and keep the
        non-UTF-8 warning, which is a real problem rather than an expected shape.
        """
        sparse = [(c, n) for c, n in zip(self.df.columns, self.df.null_count().row(0)) if n]
        if sparse:
            worst = sorted(sparse, key=lambda item: -item[1])[:3]
            listed = ", ".join(f"{c} ({n})" for c, n in worst)
            print(f"LogfmtLoader: {len(sparse)} of {self.df.width} columns contain nulls out of "
                  f"{len(self.df)} rows - expected for logfmt, where keys vary per line. "
                  f"Most null: {listed}.")

        for column, dtype in self.df.schema.items():
            if dtype == pl.Utf8:
                bad = self.df.filter(pl.col(column).str.contains("�")).height
                if bad:
                    print(f"WARNING! Column '{column}' has {bad} non-UTF-8 encoded values out of "
                          f"{len(self.df)}. To investigate: "
                          f"<DF_NAME>.filter(pl.col('{column}').str.contains('�'))")
