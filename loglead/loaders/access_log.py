import fnmatch
import glob
import os
import re

import polars as pl
import yaml

from .base import BaseLoader

__all__ = ['AccessLogLoader']

"""
AccessLogLoader Class

One configurable loader for web access logs - Apache/nginx Common and Combined Log Format and the
endless site-specific variations of them - instead of a Python class per variant. These formats are
positional text, so each one is a single regex; what
differs between them is only *which* regex and which capture means what, so both are configuration.

This is the second spec-driven loader, after JsonLoader, and it deliberately reuses that loader's
field-semantics vocabulary (timestamp_field, message_field, line_format, level_field, seq_id_field,
extra_fields) so the two read the same way. As there, the keyword arguments below are also the keys
of a format spec, so a spec file is nothing more than a serialized call:

    AccessLogLoader(filename="access.log", format="combined").execute()        # shipped spec
    AccessLogLoader(filename="access.log", format="./my_format.yml").execute() # your own spec
    AccessLogLoader(filename="access.log",                                     # inline, no file
                    log_format='$remote_addr - - [$time_local] "$request" $status $bytes',
                    timestamp_field="time_local").execute()

Explicit keyword arguments always win over the values in a spec.

- filename (str): file, or directory to walk when filename_pattern is given.
- filename_pattern (str, optional): glob applied within each subdirectory, as in RawLoader. When
  given, a 'file_name' column is added - the normal shape for rotated logs (access.log.1, ...).
- min_file_size (int, optional): skip files smaller than this many bytes.
- strip_full_data_path (str, optional): prefix removed from 'file_name'.
- n_rows (int, optional): read at most this many lines *per file*. Access logs are routinely larger
  than memory - the Kaggle web-server log this was developed against is 3.5 GB / 10,365,152 lines
  and needs ~11 GB to hold as a frame - and no other LogLead loader has a way to bound that.
- log_format (str, optional): the layout, written as an nginx log_format string:
  '$remote_addr - $remote_user [$time_local] "$request" $status $body_bytes_sent'. Each $variable
  becomes a named capture, and the literal text between variables has to match literally. A variable
  followed by a delimiter captures up to it ("$request" -> [^"]*), one followed by whitespace or the
  end of the line captures a run of non-whitespace. Apache's LogFormat directives (%h %l %u %t) are
  not accepted; the shipped specs express Apache's formats in this same syntax, since the two
  servers' Common/Combined layouts are byte-identical.
- pattern (str, optional): a regex with named captures, used instead of log_format when a format
  cannot be written positionally. The escape hatch; log_format compiles down to exactly this.
- file_pattern (str, optional): only read files whose name matches this glob. Lets a spec state
  which files it applies to.
- timestamp_field (str, optional): capture parsed into 'm_timestamp'.
- timestamp_formats (list, optional): candidate chrono formats. The best one is chosen on a sample
  and then applied to the whole column in one vectorized pass. Omit to let Polars infer.
- timestamp_naive (bool): strip the time zone from 'm_timestamp'. On by default because every other
  LogLead loader produces naive datetimes, and Polars refuses to concat naive with tz-aware frames.
  Access logs make this the common case rather than a corner one: CLF's timestamp always carries an
  offset.
- message_field (str, optional): capture used as 'm_message'.
- line_format (str, optional): template rendering several captures into 'm_message',
  e.g. "{remote_addr} {request} {status}". Given neither, the raw log line is the message.
- level_field / seq_id_field (str, optional): captures copied to 'level' and 'seq_id'. Access logs
  have no level, but they do have a natural sequence id - the client address.
- request_field (str, optional): capture holding a "GET /path HTTP/1.1" request line, split into
  'method', 'path' and 'protocol'. Those three are what make an access log useful to
  AnomalyDetector: method and protocol are low-cardinality categoricals, and the undivided request
  is not - every path is its own value.
- numeric_fields (list, optional): captures cast to Int64. A regex yields strings, so unlike JSON
  this has to be declared; status and byte counts are the point of an access log as anomaly-
  detection input, and AnomalyDetector(numeric_cols=...) needs them typed.
- null_token (str): a capture exactly equal to this becomes null. Defaults to "-", which is how CLF
  writes "absent" for every field it has no value for.
- extra_fields (str): 'keep' (default) leaves every capture as its own column; 'drop' keeps only the
  mapped columns.
- on_error (str): what to do with lines the pattern does not match. 'drop' (default) removes them
  and reports the count with an example; 'keep' keeps them with null captures and the raw line as
  'm_message'; 'raise' fails. Unlike JsonLoader, which defaults to 'raise', the default here is
  lenient: a public web server logs whatever a client sends it, so a handful of unparseable lines
  per million is normal rather than a sign of a wrong format. What catches a genuinely wrong format
  is min_match_rate, not the first bad line.
- min_match_rate (float): raise if fewer than this fraction of lines match, whatever on_error says.
  Defaults to 0.5. This is what stops 'drop' from silently turning a mis-specified format into an
  empty DataFrame.

Not implemented yet: selecting a different spec per file within one tree (file_pattern only filters
which files this loader reads), labels, and error logs, whose layout is unrelated to CLF.
"""

_UNSET = object()

_SPEC_KEYS = ("log_format", "pattern", "file_pattern", "timestamp_field", "timestamp_formats",
              "timestamp_naive", "message_field", "line_format", "level_field", "seq_id_field",
              "request_field", "numeric_fields", "null_token", "extra_fields")

_DEFAULTS = {"log_format": None, "pattern": None, "file_pattern": None, "timestamp_field": None,
             "timestamp_formats": None, "timestamp_naive": True, "message_field": None,
             "line_format": None, "level_field": None, "seq_id_field": None,
             "request_field": None, "numeric_fields": None, "null_token": "-",
             "extra_fields": "keep"}

# Keys a spec may carry that describe the format rather than configure the loader.
_SPEC_METADATA = ("name", "description", "sample")

_FORMATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "access_log_formats")

_TEMPLATE_FIELD = re.compile(r"\{([^{}]+)\}")

# nginx writes $name or ${name}; the braces exist precisely for "$var followed by a word character".
_VARIABLE = re.compile(r"\$\{(\w+)\}|\$(\w+)")

_RAW = "_access_log_raw_line"
_UNMATCHED = "_access_log_unmatched"


class AccessLogLoader(BaseLoader):
    def __init__(self, filename, format=None, filename_pattern=None, min_file_size=0,
                 strip_full_data_path=None, n_rows=None, on_error="drop", min_match_rate=0.5,
                 log_format=_UNSET, pattern=_UNSET, file_pattern=_UNSET, timestamp_field=_UNSET,
                 timestamp_formats=_UNSET, timestamp_naive=_UNSET, message_field=_UNSET,
                 line_format=_UNSET, level_field=_UNSET, seq_id_field=_UNSET,
                 request_field=_UNSET, numeric_fields=_UNSET, null_token=_UNSET,
                 extra_fields=_UNSET):

        cfg = dict(_DEFAULTS)
        if format is not None:
            cfg.update(self._read_spec(format))
        for key, value in (("log_format", log_format), ("pattern", pattern),
                           ("file_pattern", file_pattern), ("timestamp_field", timestamp_field),
                           ("timestamp_formats", timestamp_formats),
                           ("timestamp_naive", timestamp_naive), ("message_field", message_field),
                           ("line_format", line_format), ("level_field", level_field),
                           ("seq_id_field", seq_id_field), ("request_field", request_field),
                           ("numeric_fields", numeric_fields), ("null_token", null_token),
                           ("extra_fields", extra_fields)):
            if value is not _UNSET:
                cfg[key] = value

        if not cfg["log_format"] and not cfg["pattern"]:
            raise ValueError("AccessLogLoader needs a layout: give log_format, pattern, or a "
                             f"format spec that has one. Shipped specs: "
                             f"{', '.join(self.available_formats()) or '(none installed)'}")
        if cfg["extra_fields"] not in ("keep", "drop"):
            raise ValueError(f"extra_fields must be 'keep' or 'drop', got {cfg['extra_fields']!r}")
        if on_error not in ("raise", "drop", "keep"):
            raise ValueError(f"on_error must be 'raise', 'drop' or 'keep', got {on_error!r}")
        if cfg["message_field"] and cfg["line_format"]:
            raise ValueError("give either message_field or line_format, not both")

        self.format = format
        self.filename_pattern = filename_pattern
        self.min_file_size = min_file_size
        self.strip_full_data_prefix = strip_full_data_path
        self.n_rows = n_rows
        self.on_error = on_error
        self.min_match_rate = min_match_rate
        for key in _SPEC_KEYS:
            setattr(self, key, cfg[key])
        self.numeric_fields = list(self.numeric_fields or [])
        # pattern wins when both are given, so an inline pattern= can override a spec's log_format.
        self.regex = self.pattern or self._compile_log_format(self.log_format)
        self.fields = list(re.compile(self.regex).groupindex)
        if not self.fields:
            # Unnamed groups would come back as columns called "1", "2", ... and nothing could then
            # be mapped to them, so say so here rather than producing that frame.
            raise ValueError(f"pattern has no named captures - write them as (?P<name>...) so the "
                             f"fields can be referred to by name. Got: {self.regex}")
        # Filled in by preprocess(), since which columns are mandatory depends on the mapping.
        self._mandatory_columns = []
        super().__init__(filename)

    # Spec handling ---------------------------------------------------------------------------

    @staticmethod
    def _read_spec(format):
        """Resolve a shipped spec name, or a path to a spec file, into a dict of loader options."""
        # isfile, not exists: a shipped spec name is frequently also the name of a directory in the
        # caller's working directory, and exists() would resolve to that and then fail on open().
        path = format if os.path.isfile(format) else os.path.join(_FORMATS_DIR, f"{format}.yml")
        if not os.path.isfile(path):
            available = sorted(f[:-4] for f in os.listdir(_FORMATS_DIR) if f.endswith(".yml")) \
                if os.path.isdir(_FORMATS_DIR) else []
            raise FileNotFoundError(
                f"No access log format spec {format!r}. Give a path to a .yml file, or one of: "
                f"{', '.join(available) if available else '(none installed)'}")
        with open(path, 'r') as file:
            spec = yaml.safe_load(file) or {}
        unknown = set(spec) - set(_SPEC_KEYS) - set(_SPEC_METADATA)
        if unknown:
            raise ValueError(f"Unknown key(s) in spec {path}: {', '.join(sorted(unknown))}")
        return {k: v for k, v in spec.items() if k in _SPEC_KEYS}

    @staticmethod
    def available_formats():
        """Names accepted by the format argument."""
        if not os.path.isdir(_FORMATS_DIR):
            return []
        return sorted(f[:-4] for f in os.listdir(_FORMATS_DIR) if f.endswith(".yml"))

    @staticmethod
    def _compile_log_format(log_format):
        """Compile an nginx log_format string into a regex with one named capture per $variable.

        How far a capture reaches is decided by what follows it in the template, which is the whole
        reason these formats are unambiguous despite their fields containing spaces: '"$request"'
        stops at the closing quote, '[$time_local]' at the bracket, and a variable followed by a
        space or the end of the line takes a run of non-whitespace.
        """
        pieces, names, cursor = [], [], 0
        for match in _VARIABLE.finditer(log_format):
            pieces.append(re.escape(log_format[cursor:match.start()]))
            name = match.group(1) or match.group(2)
            if name in names:
                raise ValueError(f"log_format names ${name} twice; captures must be unique")
            names.append(name)
            tail = log_format[match.end():]
            stop = tail[0] if tail else ""
            if not stop or stop.isspace():
                body = r"\S+"
            elif stop == "$":
                raise ValueError(f"${name} is followed directly by another variable in log_format, "
                                 f"so there is nothing to tell the two apart. Put a delimiter "
                                 f"between them, or give an explicit pattern=.")
            else:
                body = f"[^{re.escape(stop)}]*"
            pieces.append(f"(?P<{name}>{body})")
            cursor = match.end()
        pieces.append(re.escape(log_format[cursor:]))
        if not names:
            raise ValueError(f"log_format {log_format!r} names no $variables")
        return "^" + "".join(pieces) + "$"

    # Loading ---------------------------------------------------------------------------------

    def load(self):
        self._paths = self._collect_paths()
        add_file_col = bool(self.filename_pattern)
        # The raw line is only worth carrying when something downstream reads it, since on a big
        # access log it is the single largest column.
        keep_raw = self.on_error == "keep" or not (self.message_field or self.line_format)

        queries = [self._scan(path, add_file_col, keep_raw) for path in self._paths]
        frames = pl.collect_all(queries)
        self.df = frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical_relaxed")
        self._handle_unmatched(keep_raw)

    def _collect_paths(self):
        if not self.filename_pattern:
            if self.file_pattern and not fnmatch.fnmatch(os.path.basename(self.filename), self.file_pattern):
                raise ValueError(f"{self.filename} does not match the spec's file_pattern "
                                 f"{self.file_pattern!r}")
            return [self.filename]

        paths = []
        for subdir, _, _ in os.walk(self.filename):
            for path in sorted(glob.glob(os.path.join(subdir, self.filename_pattern))):
                if os.path.getsize(path) <= self.min_file_size:
                    continue
                if self.file_pattern and not fnmatch.fnmatch(os.path.basename(path), self.file_pattern):
                    continue
                paths.append(path)
        if not paths:
            raise ValueError(f"No files matching pattern {self.filename_pattern} "
                             f"in directory {os.path.abspath(self.filename)}. "
                             f"Check the pattern, min_file_size and file_pattern.")
        return paths

    def _scan_lines(self, path, add_file_col=False):
        """Read a file as whole lines.

        This is RawLoader's read: an impossible separator and quote_char=None, so Polars hands back
        whole lines instead of trying to find fields in them.
        """
        return pl.scan_csv(path, has_header=False, schema={"column_1": pl.String},
                           infer_schema=False, quote_char=None, separator=self._csv_separator,
                           encoding="utf8-lossy", truncate_ragged_lines=True, n_rows=self.n_rows,
                           **({"include_file_paths": "file_name"} if add_file_col else {}))

    def _scan(self, path, add_file_col, keep_raw):
        """One lazy query per file: read lines, then split each into captures."""
        query = self._scan_lines(path, add_file_col)
        carried = ([pl.col(_RAW)] if keep_raw else []) + \
                  ([pl.col("file_name")] if add_file_col else [])
        query = query.with_columns(pl.col("column_1").alias(_RAW)).select(
            [pl.col("column_1").str.extract_groups(self.regex).alias("_fields")] + carried
        ).unnest("_fields")
        if add_file_col and self.strip_full_data_prefix:
            query = query.with_columns(
                pl.col("file_name").str.strip_prefix(self.strip_full_data_prefix).alias("file_name"))
        return query

    def _handle_unmatched(self, keep_raw):
        """A line that did not match leaves every capture null - that is how extract_groups reports
        no match, and no real access log line matches while producing nothing."""
        mask = self.df.select(
            pl.all_horizontal([pl.col(name).is_null() for name in self.fields])).to_series()
        count = int(mask.sum())
        if not count:
            return
        total = len(self.df)
        rate = 1 - count / total
        example = self._first_unmatched(keep_raw, mask)

        if rate < self.min_match_rate:
            raise ValueError(
                f"Only {rate:.1%} of {total} lines matched the format"
                f"{f' {self.format!r}' if self.format else ''}, below min_match_rate "
                f"{self.min_match_rate:.0%}. This usually means the wrong format, not bad data. "
                f"First line that did not match: {example}")
        if self.on_error == "raise":
            raise ValueError(f"{count} of {total} lines did not match the format. Use "
                             f"on_error='drop' or 'keep' to tolerate them. First one: {example}")
        if self.on_error == "drop":
            self.df = self.df.filter(~mask)
        else:
            # preprocess() needs to know which rows to fall back to the raw line for, and cannot
            # recompute it: null_token is about to make nulls out of matched fields too.
            self.df = self.df.with_columns(mask.alias(_UNMATCHED))
        print(f"AccessLogLoader: {count} of {total} lines ({count / total:.3%}) did not match the "
              f"format and were {'dropped' if self.on_error == 'drop' else 'kept unparsed'}. "
              f"First one: {example}")

    def _first_unmatched(self, keep_raw, mask):
        """The first line that did not match, quoted in the error - which is the whole diagnostic.

        When the raw line was not carried through the load, re-scan for it rather than keeping a
        second copy of every line in memory just in case: this runs only when something failed, and
        the head(1) lets Polars stop at the first hit.
        """
        if keep_raw:
            return str(self.df.filter(mask).select(_RAW).item(0, 0))[:200]
        for path in self._paths:
            found = (self._scan_lines(path)
                     .filter(~pl.col("column_1").str.contains(self.regex))
                     .head(1).collect())
            if found.height:
                return str(found.item(0, 0))[:200]
        return "(could not be re-read)"

    # Mapping ---------------------------------------------------------------------------------

    def preprocess(self):
        if self.null_token is not None:
            # Do this before anything reads the captures, so "-" never reaches a numeric cast, a
            # request split or a rendered message.
            self.df = self.df.with_columns(
                [pl.when(pl.col(f) == self.null_token).then(None).otherwise(pl.col(f)).alias(f)
                 for f in self.fields])

        if self.request_field:
            self._split_request()
        if self.numeric_fields:
            missing = [f for f in self.numeric_fields if f not in self.df.columns]
            if missing:
                raise ValueError(f"numeric_fields names capture(s) the format does not produce: "
                                 f"{', '.join(missing)}. Captures: {', '.join(self.fields)}")
            self.df = self.df.with_columns([self._numeric(f) for f in self.numeric_fields])

        message = self._message_expr()
        if _UNMATCHED in self.df.columns:
            # on_error='keep'. Rendering a template over a row whose captures are all null yields
            # the template's punctuation and nothing else, so keep the line as it was logged.
            message = pl.when(pl.col(_UNMATCHED)).then(pl.col(_RAW)).otherwise(message)
        mapped = [message.alias("m_message")]
        if self.level_field:
            mapped.append(self._resolve(self.level_field).cast(pl.String).alias("level"))
        if self.seq_id_field:
            mapped.append(self._resolve(self.seq_id_field).cast(pl.String).alias("seq_id"))
        self.df = self.df.with_columns(mapped)

        if self.timestamp_field:
            self._add_timestamp()

        self._mandatory_columns = ["m_message"] + (["m_timestamp"] if self.timestamp_field else [])

        if self.extra_fields == "drop":
            keep = [c for c in ("m_timestamp", "m_message", "level", "seq_id", "method", "path",
                                "protocol", "file_name") + tuple(self.numeric_fields)
                    if c in self.df.columns]
            self.df = self.df.select(keep)
        else:
            self.df = self.df.drop([c for c in (_RAW, _UNMATCHED) if c in self.df.columns])

    def _numeric(self, field):
        """Cast a capture to a number, preferring Int64 and falling back to Float64.

        Which one it is cannot be declared per field without making every spec carry a type map,
        and guessing wrong is not harmless: nginx's $request_time and $upstream_response_time are
        decimals, and an Int64-only cast turns exactly the latency fields that make an access log
        worth scoring into a column of nulls. Counting the values lost decides it.
        """
        source = self.df[field]
        present = len(source) - source.null_count()
        integers = source.cast(pl.Int64, strict=False)
        if len(integers) - integers.null_count() == present:
            return integers.alias(field)
        floats = source.cast(pl.Float64, strict=False)
        lost = present - (len(floats) - floats.null_count())
        if lost > 0:
            print(f"WARNING! AccessLogLoader could not read {lost} of {present} values in "
                  f"'{field}' as a number; they are null. Drop it from numeric_fields if it is "
                  f"not one.")
        return floats.alias(field)

    def _resolve(self, field):
        if field not in self.df.columns:
            raise ValueError(f"Field {field!r} is not in the data. Available: "
                             f"{', '.join(self.df.columns)}")
        return pl.col(field)

    def _split_request(self):
        """Split "GET /path HTTP/1.1" into method, path and protocol."""
        source = self._resolve(self.request_field)
        clashes = [c for c in ("method", "path", "protocol") if c in self.df.columns]
        if clashes:
            raise ValueError(f"Splitting {self.request_field!r} would overwrite existing "
                             f"column(s): {', '.join(clashes)}. Unset request_field, or rename "
                             f"the capture in the format.")
        self.df = self.df.with_columns(
            source.str.splitn(" ", 3).struct.rename_fields(["method", "path", "protocol"])
                  .alias("_request")).unnest("_request")

    def _message_expr(self):
        if self.message_field:
            return self._resolve(self.message_field).cast(pl.String)
        if self.line_format:
            pieces, cursor = [], 0
            for match in _TEMPLATE_FIELD.finditer(self.line_format):
                if match.start() > cursor:
                    pieces.append(pl.lit(self.line_format[cursor:match.start()]))
                pieces.append(self._resolve(match.group(1)).cast(pl.String))
                cursor = match.end()
            if cursor < len(self.line_format):
                pieces.append(pl.lit(self.line_format[cursor:]))
            if not pieces:
                raise ValueError(f"line_format {self.line_format!r} names no fields")
            return pl.concat_str(pieces, ignore_nulls=True)
        # Nothing declared: the line itself is the message. Unlike a JSON record, a positional log
        # line already is text, so there is nothing to render and nothing lost by keeping it.
        return pl.col(_RAW)

    def _add_timestamp(self):
        strings = self.df.select(self._resolve(self.timestamp_field).cast(pl.String).alias("t")).to_series()
        chosen = self._pick_format(strings, self.timestamp_formats)
        parsed = strings.str.to_datetime(format=chosen, strict=False)
        if self.timestamp_naive and getattr(parsed.dtype, "time_zone", None):
            # CLF always carries an offset, and every other LogLead loader yields naive datetimes;
            # mixing the two breaks pl.concat.
            parsed = parsed.dt.replace_time_zone(None)
        if getattr(parsed.dtype, "time_unit", None) != "us":
            parsed = parsed.cast(pl.Datetime("us", getattr(parsed.dtype, "time_zone", None)))
        unparsed = parsed.null_count() - strings.null_count()
        if unparsed > 0:
            print(f"WARNING! AccessLogLoader could not parse {unparsed} of {len(parsed)} values in "
                  f"'{self.timestamp_field}' into m_timestamp. Check timestamp_formats.")
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
        that has nulls, and here nulls are the *designed* outcome. null_token turns CLF's "-" into
        null, and "-" is exactly what a web server writes for every field a request did not carry -
        remote_user is empty on almost every public site, referrer on direct traffic, and
        X-Forwarded-For on everything not behind a proxy. Summarize instead, and keep the non-UTF-8
        warning, which is a real problem rather than an expected shape.
        """
        sparse = [(c, n) for c, n in zip(self.df.columns, self.df.null_count().row(0)) if n]
        if sparse:
            worst = sorted(sparse, key=lambda item: -item[1])[:3]
            listed = ", ".join(f"{c} ({n})" for c, n in worst)
            print(f"AccessLogLoader: {len(sparse)} of {self.df.width} columns contain nulls out of "
                  f"{len(self.df)} rows - expected, as null_token={self.null_token!r} marks the "
                  f"fields a request did not carry. Most null: {listed}.")

        for column, dtype in self.df.schema.items():
            if dtype == pl.Utf8:
                bad = self.df.filter(pl.col(column).str.contains("�")).height
                if bad:
                    print(f"WARNING! Column '{column}' has {bad} non-UTF-8 encoded values out of "
                          f"{len(self.df)}. To investigate: "
                          f"<DF_NAME>.filter(pl.col('{column}').str.contains('�'))")
