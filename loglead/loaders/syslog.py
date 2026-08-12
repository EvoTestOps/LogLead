import datetime
import glob
import os
import re

import polars as pl

from .base import BaseLoader

__all__ = ['SyslogLoader']

"""
SyslogLoader Class

One loader for syslog - RFC 3164 (the old BSD format, still what almost every Linux box and network
device writes) and RFC 5424 (the structured successor). It is the substrate of Unix and appliance
logging and lnav treats it as a core format (docs/log-format-support.md section 5 item 5).

Unlike the web access logs of AccessLogLoader, syslog is not one layout per site: there are two,
both defined by an RFC, so they are built in and named rather than kept in spec files. Which one a
file uses is decided per file from its first lines, so a directory holding both reads in one call:

    SyslogLoader(filename="/var/log/syslog").execute()                     # detect the RFC
    SyslogLoader(filename="logs", filename_pattern="*.log").execute()      # a tree of files
    SyslogLoader(filename="app.log", format="rfc5424").execute()           # pin it
    SyslogLoader(filename="app.log", pattern=r"^(?P<timestamp>...)").execute()   # neither RFC

RFC 3164 gives m_timestamp, host, app_name, procid and m_message; RFC 5424 adds msgid and
structured_data. A leading priority is read in either case and becomes facility and level, which is
the one place syslog carries a severity at all - the BSD format's message text has no level in it.

- filename (str): file, or directory to walk when filename_pattern is given.
- filename_pattern (str, optional): glob applied within each subdirectory, as in RawLoader. When
  given, a 'file_name' column is added.
- min_file_size (int, optional): skip files smaller than this many bytes.
- strip_full_data_path (str, optional): prefix removed from 'file_name'.
- format (str): 'auto' (default), 'rfc3164' or 'rfc5424'. 'auto' tries both on the first lines of
  each file and keeps whichever matches more of them - per file, because a log directory holding
  two formats is normal rather than exceptional (docs/log-format-support.md section 5 item 6).
- pattern (str, optional): a regex with named captures, used instead of the built-in ones. The
  escape hatch for the vendor variants that call themselves syslog and are not. Recognized capture
  names: pri, timestamp, host, app_name, procid, msgid, structured_data, message.
- year (int, optional): RFC 3164 timestamps carry no year - 'Jun  9 06:06:20' is the whole of it -
  so one has to be supplied to build a datetime at all. Defaults to the current year, which keeps
  ordering inside a file right and the absolute date wrong for an archive; pass the year the log
  was written to fix that. Ignored for RFC 5424, whose timestamps are RFC 3339.
- timestamp_formats (list, optional): candidate chrono formats, if the timestamps deviate from the
  RFC. The best one is chosen on a sample and then applied to the whole column in one vectorized
  pass.
- timestamp_naive (bool): strip the time zone from 'm_timestamp'. On by default because every other
  LogLead loader produces naive datetimes, and Polars refuses to concat naive with tz-aware frames.
- seq_id_field (str, optional): capture copied to 'seq_id'. 'host' and 'app_name' are the usual
  candidates; there is no default, since which one groups a sequence is a property of the analysis
  rather than of the format.
- extra_fields (str): 'keep' (default) leaves every capture as its own column; 'drop' keeps only
  m_timestamp, m_message, level, seq_id and trace.
- multiline (str): what to do with lines that do not match the format. In syslog such a line is
  almost never garbage - it is the second and later lines of one event, a Java stack trace or a
  macOS constraint dump - so the question is not whether to tolerate it but where it belongs.
  'keep' and 'drop' are named as in RawLoader's missing_timestamp_action, which asks the same
  question; the two merges say where they put the text rather than borrowing that loader's plain
  'merge', because here there are two of them:
    'merge-message' (default) - one row per event, with the continuation lines appended to
        m_message itself. This is the reading that follows the format: m_message is the
        unstructured tail of the event, and the event does not end until the next timestamp does.
    'merge-add-column' - one row per event, the way RawLoader's 'merge' does it: m_message is the
        first line only, and the continuation lines are joined with newlines into an added 'trace'
        column.
    'keep' - each continuation line stays its own row, with the raw line as m_message and nulls
        elsewhere. Nothing is lost and nothing is assumed, but the event count is inflated and
        every fragment becomes its own parser template.
    'drop' - remove them. Loses log content rather than noise; there when the fragments are not
        wanted at all.
    'raise' - treat a non-matching line as an error and fail.
  The two merges differ less than they look, and not where you would expect. Every parse_* method
  reads e_message_normalized, and EventLogEnhancer.normalize() keeps only the first line of
  m_message, so template mining sees the same thing either way: measured on Mac.log, both give 647
  Drain templates, against 901 when the fragments stay their own rows. What does differ is the
  token and length columns - words(), trigrams(), alphanumerics() and length() read m_message
  whole, so under 'merge-message' the trace reaches e_words/e_trigrams/e_chars_len (mean 94 -> 100
  chars, max 1,264 -> 5,436 on that file) and under 'merge-add-column' it does not, sitting in
  'trace' where nothing reads it until asked.
  Merging groups by file, so a file whose first lines are continuations cannot attach them to the
  previous file's last event. A continuation with no event above it in its own file keeps its own
  first line as the message rather than being discarded.
- min_match_rate (float): raise if fewer than this fraction of lines match, whatever multiline
  says. Defaults to 0.5. This is what separates "a few continuation lines" from "the wrong format".

Not implemented: parsing structured_data into columns (it is kept as the raw '[id key="v"]' text),
and the syslog-over-TCP octet-counted framing, which is a transport concern rather than a file one.
"""

_UNSET = object()

# RFC 3164: an optional priority, a three-letter month with a space-padded day and no year, the
# host, then a tag that ends at '[', ':' or a space, an optional pid, and the message.
_RFC3164 = (r'^(?:<(?P<pri>\d{1,3})>)?'
            r'(?P<timestamp>[A-Z][a-z]{2}\s+\d{1,2} \d{2}:\d{2}:\d{2})\s+'
            r'(?P<host>\S+)\s+'
            r'(?P<app_name>[^\s:\[]+)(?:\[(?P<procid>\d+)\])?:?\s'
            r'(?P<message>.*)$')

# RFC 5424: priority and version, then six space-separated fields, of which structured_data is
# either '-' or one or more '[...]' elements, and everything after it is the message.
_RFC5424 = (r'^<(?P<pri>\d{1,3})>(?P<version>\d{1,2})\s+'
            r'(?P<timestamp>\S+)\s+(?P<host>\S+)\s+(?P<app_name>\S+)\s+'
            r'(?P<procid>\S+)\s+(?P<msgid>\S+)\s+'
            r'(?P<structured_data>-|\[[^\]]*\](?:\[[^\]]*\])*)'
            r'(?:\s(?P<message>.*))?$')

_PATTERNS = {"rfc3164": _RFC3164, "rfc5424": _RFC5424}

# RFC 3164 section 4.1.1, and unchanged in RFC 5424: the low three bits of the priority.
_SEVERITIES = ("emerg", "alert", "crit", "err", "warning", "notice", "info", "debug")

_RFC3164_FORMAT = "%Y %b %e %H:%M:%S"

# '-' is RFC 5424's NILVALUE: the field was not supplied.
_NIL = "-"

_RAW = "_syslog_raw_line"
_GROUP = "_syslog_event"
_TEXT = "_syslog_text"

_MULTILINE = ("merge-message", "merge-add-column", "keep", "drop", "raise")

# How many lines of a file are looked at to decide which RFC it is written in.
_SAMPLE_LINES = 1000


class SyslogLoader(BaseLoader):
    def __init__(self, filename, format="auto", filename_pattern=None, min_file_size=0,
                 strip_full_data_path=None, pattern=None, year=None, timestamp_formats=None,
                 timestamp_naive=True, seq_id_field=None, extra_fields="keep",
                 multiline="merge-message", min_match_rate=0.5):

        if format not in ("auto", "rfc3164", "rfc5424"):
            raise ValueError(f"format must be 'auto', 'rfc3164' or 'rfc5424', got {format!r}")
        if extra_fields not in ("keep", "drop"):
            raise ValueError(f"extra_fields must be 'keep' or 'drop', got {extra_fields!r}")
        if multiline not in _MULTILINE:
            raise ValueError(f"multiline must be one of {', '.join(_MULTILINE)}, got {multiline!r}")
        if pattern and "timestamp" not in re.compile(pattern).groupindex:
            # Every downstream step keys off it, and it is also how an unmatched line is spotted.
            raise ValueError("pattern needs a (?P<timestamp>...) capture; the rest are optional")

        self.format = format
        self.filename_pattern = filename_pattern
        self.min_file_size = min_file_size
        self.strip_full_data_prefix = strip_full_data_path
        self.pattern = pattern
        self.year = year
        self.timestamp_formats = timestamp_formats
        self.timestamp_naive = timestamp_naive
        self.seq_id_field = seq_id_field
        self.extra_fields = extra_fields
        self.multiline = multiline
        self.min_match_rate = min_match_rate
        # Which pattern each file turned out to be written in, for the error messages.
        self._formats_used = {}
        super().__init__(filename)

    # Loading ---------------------------------------------------------------------------------

    def load(self):
        paths = self._collect_paths()
        add_file_col = bool(self.filename_pattern)
        frames = pl.collect_all([self._scan(path, add_file_col) for path in paths])
        # Files parsed under different RFCs do not have the same captures, so a plain vertical
        # concat would raise; the format is per file precisely so that this case works.
        self.df = frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")
        self._handle_unmatched()

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
        whole lines instead of trying to find fields in them.
        """
        return pl.scan_csv(path, has_header=False, schema={"column_1": pl.String},
                           infer_schema=False, quote_char=None, separator=self._csv_separator,
                           encoding="utf8-lossy", truncate_ragged_lines=True,
                           **({"include_file_paths": "file_name"} if add_file_col else {}))

    def _scan(self, path, add_file_col):
        """One lazy query per file: read lines, then split each into captures."""
        regex = self._pattern_for(path)
        query = self._scan_lines(path, add_file_col)
        carried = [pl.col(_RAW)] + ([pl.col("file_name")] if add_file_col else [])
        query = query.with_columns(pl.col("column_1").alias(_RAW)).select(
            [pl.col("column_1").str.extract_groups(regex).alias("_fields")] + carried
        ).unnest("_fields")
        if add_file_col and self.strip_full_data_prefix:
            query = query.with_columns(
                pl.col("file_name").str.strip_prefix(self.strip_full_data_prefix).alias("file_name"))
        return query

    def _pattern_for(self, path):
        """Decide which RFC a file is written in, from its first lines.

        Trying both on a sample and keeping the better one is lnav's rule - match on the first few
        lines, then lock the format for the file - and it is cheap: one head() read per file.
        """
        if self.pattern:
            self._formats_used[path] = "pattern"
            return self.pattern
        if self.format != "auto":
            self._formats_used[path] = self.format
            return _PATTERNS[self.format]

        sample = self._scan_lines(path).head(_SAMPLE_LINES).collect()["column_1"].drop_nulls()
        if not len(sample):
            self._formats_used[path] = "rfc3164"
            return _RFC3164
        # RFC 5424 first: its lines also start with a priority, so a file that matches it would
        # partly match a priority-prefixed 3164 pattern too, but not the other way round.
        best, best_hits = "rfc3164", -1
        for name in ("rfc5424", "rfc3164"):
            hits = sample.str.contains(_PATTERNS[name]).sum()
            if hits > best_hits:
                best, best_hits = name, hits
            if hits == len(sample):
                break
        self._formats_used[path] = best
        return _PATTERNS[best]

    def _handle_unmatched(self):
        """A line that did not match leaves every capture null; 'timestamp' stands in for that,
        since it is the one capture both RFCs - and any usable custom pattern - always have."""
        mask = self.df["timestamp"].is_null()
        count = int(mask.sum())
        if not count:
            return
        total = len(self.df)
        rate = 1 - count / total
        example = str(self.df.filter(mask).select(_RAW).item(0, 0))[:200]
        used = ", ".join(sorted(set(self._formats_used.values())))

        if rate < self.min_match_rate:
            raise ValueError(
                f"Only {rate:.1%} of {total} lines matched syslog format(s) {used}, below "
                f"min_match_rate {self.min_match_rate:.0%}. This usually means the wrong format, "
                f"not bad data. First line that did not match: {example}")
        if self.multiline == "raise":
            raise ValueError(f"{count} of {total} lines did not match syslog format(s) {used}. "
                             f"Use multiline='merge-message', 'merge-add-column', 'keep' or "
                             f"'drop' to tolerate them, which is normal for multi-line messages. "
                             f"First one: {example}")
        if self.multiline == "drop":
            self.df = self.df.filter(~mask)
            outcome = "dropped"
        elif self.multiline in ("merge-message", "merge-add-column"):
            into_message = self.multiline == "merge-message"
            self._merge_continuations(into_message)
            target = "m_message" if into_message else "an added 'trace' column"
            outcome = f"merged into {target}, leaving {len(self.df)} events"
        else:
            outcome = "kept unparsed as their own rows"
        print(f"SyslogLoader: {count} of {total} lines ({count / total:.3%}) did not match "
              f"{used} and were {outcome} - usually continuation lines of multi-line messages. "
              f"First one: {example}")

    def _merge_continuations(self, into_message):
        """Fold each event's continuation lines back into the row that started it.

        The group is a running count of matched lines, so every unmatched line joins the event
        above it. It is counted **per file**: a plain cumulative sum would let a file whose first
        lines are continuations attach them to the last event of the previous file, which is the
        one thing a multi-file merge has to get right.
        """
        text = pl.coalesce([pl.col("message"), pl.col(_RAW)]) \
            if "message" in self.df.columns else pl.col(_RAW)
        event = pl.col("timestamp").is_not_null().cum_sum()
        partition = ["file_name"] if "file_name" in self.df.columns else []
        frame = self.df.with_columns(
            text.alias(_TEXT), (event.over("file_name") if partition else event).alias(_GROUP))

        carried = [c for c in self.df.columns if c not in partition + ["message"]]
        aggregations = [pl.col(c).first().alias(c) for c in carried]
        if into_message:
            # The event's text does not end until the next timestamp does, so all of it is the
            # message.
            aggregations.append(pl.col(_TEXT).str.join("\n").alias("message"))
        else:
            # first()/slice(1) rather than a filter on "did it match", so that a continuation with
            # no event above it keeps its own first line as the message instead of vanishing.
            aggregations.append(pl.col(_TEXT).first().alias("message"))
            aggregations.append(pl.col(_TEXT).slice(1).str.join("\n").alias("trace"))

        merged = frame.group_by(partition + [_GROUP], maintain_order=True).agg(aggregations)
        if not into_message:
            # An event with no continuation lines has no trace, rather than an empty one.
            merged = merged.with_columns(
                pl.when(pl.col("trace") == "").then(None).otherwise(pl.col("trace")).alias("trace"))
        columns = list(self.df.columns)
        if "message" not in columns:
            columns.append("message")
        self.df = merged.select(columns + ([] if into_message else ["trace"]))

    # Mapping ---------------------------------------------------------------------------------

    def preprocess(self):
        # RFC 5424 writes '-' for a field that was not supplied. Do this before anything reads the
        # captures, so a nil never reaches m_message or seq_id as the string "-".
        nillable = [c for c in ("procid", "msgid", "structured_data", "host", "app_name")
                    if c in self.df.columns]
        if nillable:
            self.df = self.df.with_columns(
                [pl.when(pl.col(c) == _NIL).then(None).otherwise(pl.col(c)).alias(c)
                 for c in nillable])

        # An unmatched line still said something, and a null m_message breaks every enhancer.
        message = pl.coalesce([pl.col("message"), pl.col(_RAW)]) \
            if "message" in self.df.columns else pl.col(_RAW)
        mapped = [message.alias("m_message")]
        if "pri" in self.df.columns:
            mapped.extend(self._priority_columns())
        if self.seq_id_field:
            if self.seq_id_field not in self.df.columns:
                raise ValueError(f"seq_id_field={self.seq_id_field!r} is not a capture of this "
                                 f"format. Captures: {', '.join(self.df.columns)}")
            mapped.append(pl.col(self.seq_id_field).cast(pl.String).alias("seq_id"))
        self.df = self.df.with_columns(mapped)

        self._add_timestamp()
        self._mandatory_columns = ["m_message", "m_timestamp"]

        if self.extra_fields == "drop":
            keep = [c for c in ("m_timestamp", "m_message", "level", "seq_id", "trace",
                                "file_name") if c in self.df.columns]
            self.df = self.df.select(keep)
        else:
            self.df = self.df.drop([c for c in (_RAW, "message") if c in self.df.columns])

    def _priority_columns(self):
        """Split the priority into its facility and severity, per RFC 3164 section 4.1.1.

        A priority is a single number holding both: facility * 8 + severity. Splitting it is the
        only way syslog yields a level at all, and 'level' is what the rest of LogLead expects a
        severity to be called.
        """
        priority = pl.col("pri").cast(pl.Int64, strict=False)
        severity = priority % 8
        return [(priority // 8).alias("facility"),
                pl.when(severity.is_null()).then(None)
                  .otherwise(severity.replace_strict(list(range(8)), list(_SEVERITIES),
                                                     default=None))
                  .alias("level")]

    def _add_timestamp(self):
        strings = self.df["timestamp"]
        used = set(self._formats_used.values())
        if self.timestamp_formats:
            parsed = self._parse(strings, self.timestamp_formats)
        elif used == {"rfc3164"}:
            parsed = self._parse(self._with_year(strings), [_RFC3164_FORMAT])
        elif used == {"rfc5424"}:
            parsed = self._parse(strings, None)
        else:
            # One load holding both RFCs, or a custom pattern of unknown shape. The two cannot
            # share a single strptime: RFC 5424's timestamps are RFC 3339 and parse as they are,
            # while RFC 3164's have no year until one is prepended. So both passes run over the
            # whole column - still two vectorized calls, not one per row - and the results are
            # coalesced, each row taking whichever pass understood it.
            parsed = pl.DataFrame({
                "rfc5424": self._parse(strings, None),
                "rfc3164": self._parse(self._with_year(strings), [_RFC3164_FORMAT]),
            }).select(pl.coalesce("rfc5424", "rfc3164")).to_series()
        unparsed = parsed.null_count() - strings.null_count()
        if unparsed > 0:
            print(f"WARNING! SyslogLoader could not parse {unparsed} of {len(parsed)} timestamps "
                  f"into m_timestamp. Check timestamp_formats, or year= for RFC 3164.")
        self.df = self.df.with_columns(parsed.alias("m_timestamp")).drop("timestamp")

    def _with_year(self, strings):
        """RFC 3164 timestamps have no year, so one is prepended rather than parsed out."""
        return f"{self.year or datetime.date.today().year} " + strings

    def _parse(self, strings, formats):
        parsed = strings.str.to_datetime(format=self._pick_format(strings, formats), strict=False)
        if self.timestamp_naive and getattr(parsed.dtype, "time_zone", None):
            # RFC 5424 timestamps carry an offset; every other LogLead loader yields naive
            # datetimes, and mixing the two breaks pl.concat.
            parsed = parsed.dt.replace_time_zone(None)
        if getattr(parsed.dtype, "time_unit", None) != "us":
            parsed = parsed.cast(pl.Datetime("us", getattr(parsed.dtype, "time_zone", None)))
        return parsed

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
        Same reasoning as JsonLoader's and LogfmtLoader's overrides: BaseLoader prints a four-line
        warning per column that has nulls, and here nulls are the expected shape rather than a
        defect - most syslog lines carry no pid, RFC 5424 writes '-' for anything not supplied, and
        the continuation lines that on_error='keep' preserves have no captures at all. Summarize
        instead, and keep the non-UTF-8 warning, which is a real problem.
        """
        sparse = [(c, n) for c, n in zip(self.df.columns, self.df.null_count().row(0)) if n]
        if sparse:
            worst = sorted(sparse, key=lambda item: -item[1])[:3]
            listed = ", ".join(f"{c} ({n})" for c, n in worst)
            print(f"SyslogLoader: {len(sparse)} of {self.df.width} columns contain nulls out of "
                  f"{len(self.df)} rows - expected, as syslog fields are optional. "
                  f"Most null: {listed}.")

        for column, dtype in self.df.schema.items():
            if dtype == pl.Utf8:
                bad = self.df.filter(pl.col(column).str.contains("�")).height
                if bad:
                    print(f"WARNING! Column '{column}' has {bad} non-UTF-8 encoded values out of "
                          f"{len(self.df)}. To investigate: "
                          f"<DF_NAME>.filter(pl.col('{column}').str.contains('�'))")
