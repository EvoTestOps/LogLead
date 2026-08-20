import fnmatch
import glob
import io
import os
import re

import polars as pl
import yaml

from .base import BaseLoader

__all__ = ['JsonLoader']

"""
JsonLoader Class

One configurable loader for JSON logs, instead of a Python class per dataset. Reading JSON is a
single Polars call; the part that differs between datasets is only the *mapping* - which key is the
timestamp, which is the message, which correlates records - so that mapping is configuration.

The keyword arguments below are also the keys of a format spec, so a spec file is nothing more than
a serialized call and the two forms cannot drift apart:

    JsonLoader(filename="access.json", timestamp_field="time", message_field="request").execute()
    JsonLoader(filename="access.json", format="nginx_json").execute()          # shipped spec
    JsonLoader(filename="access.json", format="./my_format.yml").execute()     # your own spec

Explicit keyword arguments always win over the values in a spec.

- filename (str): file, or directory to walk when filename_pattern is given.
- filename_pattern (str, optional): glob applied within each subdirectory, as in RawLoader. When
  given, a 'file_name' column is added and files are read in parallel.
- min_file_size (int, optional): skip files smaller than this many bytes.
- strip_full_data_path (str, optional): prefix removed from 'file_name'.
- container (str): 'ndjson' (one object per line), 'array' (a top-level [...]) or 'wrapped'
  (an object holding the records under records_path, e.g. CloudTrail's {"Records": [...]}).
- records_path (str): the key holding the record list, for container='wrapped'.
- file_pattern (str, optional): only read files whose name matches this glob. Lets a spec state
  which files it applies to.
- timestamp_field (str, optional): key parsed into 'm_timestamp'.
- timestamp_formats (list, optional): candidate chrono formats. The best one is chosen on a sample
  and then applied to the whole column in one vectorized pass. Omit to let Polars infer.
- timestamp_epoch (str, optional): 's', 'ms', 'us' or 'ns' when the field is an epoch number
  instead of a string. Mutually exclusive with timestamp_formats.
- timestamp_naive (bool): strip the time zone from 'm_timestamp'. On by default because every other
  LogLead loader produces naive datetimes, and Polars refuses to concat naive with tz-aware frames.
- message_field (str, optional): key used as 'm_message'.
- line_format (str, optional): template rendering several keys into 'm_message',
  e.g. "{remote_ip} {request} {response}". Used when no single key is the message.
- level_field / seq_id_field (str, optional): keys copied to 'level' and 'seq_id'.
- extra_fields (str): 'keep' (default) leaves every JSON key as its own typed column; 'drop' keeps
  only the mapped columns.
- flatten (bool): unnest nested objects into path-named columns.
- path_separator (str): separator for nested keys, JSON-pointer style: "log/logger". Defaults to
  "/" rather than "." because real logs contain dots inside key names.
- infer_schema_length (int or None): rows scanned to infer the schema. Defaults to None, meaning
  *all* rows. Polars' own default of 100 silently drops keys that first appear later in the file,
  which is exactly what happens with logs whose keys vary per record. Set an integer only if a file
  is too large to scan, and accept that late keys may be lost.
- on_error (str): 'raise' (default) or 'skip'. A single malformed line makes Polars fail the whole
  file, and ignore_errors does not help; 'skip' re-reads such a file as text, drops the lines that
  are not JSON objects, and reports how many were dropped. It is slower and holds the file in
  memory, so it is opt-in.

Not implemented yet: selecting a different spec per file within one tree (file_pattern only filters
which files this loader reads), unpivoting objects whose keys are data, and labels.
"""

_UNSET = object()

_SPEC_KEYS = ("container", "records_path", "file_pattern", "timestamp_field", "timestamp_formats",
              "timestamp_epoch", "timestamp_naive", "message_field", "line_format", "level_field",
              "seq_id_field", "extra_fields", "flatten", "path_separator")

_DEFAULTS = {"container": "ndjson", "records_path": None, "file_pattern": None,
             "timestamp_field": None, "timestamp_formats": None, "timestamp_epoch": None,
             "timestamp_naive": True, "message_field": None, "line_format": None,
             "level_field": None, "seq_id_field": None, "extra_fields": "keep",
             "flatten": False, "path_separator": "/"}

# Keys a spec may carry that describe the format rather than configure the loader.
_SPEC_METADATA = ("name", "description", "sample")

_FORMATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json_formats")

_TEMPLATE_FIELD = re.compile(r"\{([^{}]+)\}")


class JsonLoader(BaseLoader):
    def __init__(self, filename, format=None, filename_pattern=None, min_file_size=0,
                 strip_full_data_path=None, infer_schema_length=None, on_error="raise",
                 container=_UNSET, records_path=_UNSET, file_pattern=_UNSET,
                 timestamp_field=_UNSET, timestamp_formats=_UNSET, timestamp_epoch=_UNSET,
                 timestamp_naive=_UNSET, message_field=_UNSET, line_format=_UNSET,
                 level_field=_UNSET, seq_id_field=_UNSET, extra_fields=_UNSET,
                 flatten=_UNSET, path_separator=_UNSET):

        cfg = dict(_DEFAULTS)
        if format is not None:
            cfg.update(self._read_spec(format))
        for key, value in (("container", container), ("records_path", records_path),
                           ("file_pattern", file_pattern), ("timestamp_field", timestamp_field),
                           ("timestamp_formats", timestamp_formats),
                           ("timestamp_epoch", timestamp_epoch),
                           ("timestamp_naive", timestamp_naive), ("message_field", message_field),
                           ("line_format", line_format), ("level_field", level_field),
                           ("seq_id_field", seq_id_field), ("extra_fields", extra_fields),
                           ("flatten", flatten), ("path_separator", path_separator)):
            if value is not _UNSET:
                cfg[key] = value

        if cfg["container"] not in ("ndjson", "array", "wrapped"):
            raise ValueError(f"container must be 'ndjson', 'array' or 'wrapped', got {cfg['container']!r}")
        if cfg["container"] == "wrapped" and not cfg["records_path"]:
            raise ValueError("container='wrapped' needs records_path, e.g. records_path='Records'")
        if cfg["extra_fields"] not in ("keep", "drop"):
            raise ValueError(f"extra_fields must be 'keep' or 'drop', got {cfg['extra_fields']!r}")
        if on_error not in ("raise", "skip"):
            raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")
        if cfg["timestamp_epoch"] and cfg["timestamp_formats"]:
            raise ValueError("give either timestamp_epoch or timestamp_formats, not both")
        if cfg["message_field"] and cfg["line_format"]:
            raise ValueError("give either message_field or line_format, not both")

        self.format = format
        self.filename_pattern = filename_pattern
        self.min_file_size = min_file_size
        self.strip_full_data_prefix = strip_full_data_path
        self.infer_schema_length = infer_schema_length
        self.on_error = on_error
        for key in _SPEC_KEYS:
            setattr(self, key, cfg[key])
        # Filled in by preprocess(), since which columns are mandatory depends on the mapping.
        self._mandatory_columns = []
        self._dropped_lines = 0
        super().__init__(filename)

    # Spec handling ---------------------------------------------------------------------------

    @staticmethod
    def _read_spec(format):
        path = format if os.path.isfile(format) else os.path.join(_FORMATS_DIR, f"{format}.yml")
        if not os.path.isfile(path):
            available = sorted(f[:-4] for f in os.listdir(_FORMATS_DIR) if f.endswith(".yml")) \
                if os.path.isdir(_FORMATS_DIR) else []
            raise FileNotFoundError(
                f"No JSON format spec {format!r}. Give a path to a .yml file, or one of: "
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

    # Loading ---------------------------------------------------------------------------------

    def load(self):
        paths = self._collect_paths()
        add_file_col = bool(self.filename_pattern)

        if self.container == "ndjson" and self.on_error == "raise":
            queries = [self._scan_ndjson(path, add_file_col) for path in paths]
            try:
                frames = pl.collect_all(queries)
            except Exception:
                # collect_all does not say which file failed, so find it and name it.
                self._blame(paths, add_file_col)
                raise
        else:
            frames = [self._read_one(path, add_file_col) for path in paths]

        self.df = frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")
        if self._dropped_lines:
            print(f"JsonLoader: dropped {self._dropped_lines} line(s) that were not JSON objects "
                  f"(on_error='skip').")

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

    def _scan_ndjson(self, path, add_file_col):
        query = pl.scan_ndjson(path, infer_schema_length=self.infer_schema_length,
                               **({"include_file_paths": "file_name"} if add_file_col else {}))
        return self._strip_prefix(query, add_file_col)

    def _strip_prefix(self, frame, add_file_col):
        if add_file_col and self.strip_full_data_prefix:
            return frame.with_columns(
                pl.col("file_name").str.strip_prefix(self.strip_full_data_prefix).alias("file_name"))
        return frame

    def _blame(self, paths, add_file_col):
        """Re-read one file at a time so the error message names the offending file."""
        for path in paths:
            try:
                self._scan_ndjson(path, add_file_col).collect()
            except Exception as error:
                hint = " Use on_error='skip' to drop non-JSON lines." if self.on_error == "raise" else ""
                raise ValueError(f"Failed to read JSON from {path}: {error}.{hint}") from error

    def _read_one(self, path, add_file_col):
        if self.container == "ndjson":
            try:
                return self._scan_ndjson(path, add_file_col).collect()
            except Exception:
                if self.on_error != "skip":
                    raise
                return self._read_lenient(path, add_file_col)
        return self._read_container(path, add_file_col)

    def _read_container(self, path, add_file_col):
        frame = pl.read_json(path, infer_schema_length=self.infer_schema_length)
        if self.container == "wrapped":
            if self.records_path not in frame.columns:
                raise ValueError(f"records_path {self.records_path!r} is not a top-level key of "
                                 f"{path}. Found: {', '.join(frame.columns)}")
            frame = frame.explode(self.records_path).unnest(self.records_path)
        if add_file_col:
            frame = frame.with_columns(pl.lit(path).alias("file_name"))
            frame = self._strip_prefix(frame, add_file_col)
        return frame

    def _read_lenient(self, path, add_file_col):
        """Read a file that has non-JSON lines in it: take the lines, keep the JSON objects."""
        lines = pl.read_csv(path, has_header=False, schema={'column_1': pl.String},
                            infer_schema=False, quote_char=None, separator=self._csv_separator,
                            encoding="utf8-lossy", truncate_ragged_lines=True)
        objects = lines.filter(pl.col("column_1").str.strip_chars().str.starts_with("{"))
        self._dropped_lines += lines.height - objects.height
        payload = "\n".join(objects["column_1"].to_list()).encode("utf-8")
        frame = pl.read_ndjson(io.BytesIO(payload), infer_schema_length=self.infer_schema_length)
        if add_file_col:
            frame = frame.with_columns(pl.lit(path).alias("file_name"))
            frame = self._strip_prefix(frame, add_file_col)
        return frame

    # Mapping ---------------------------------------------------------------------------------

    def preprocess(self):
        if self.flatten:
            self.df = self._flatten(self.df, self.path_separator)

        source_columns = [c for c in self.df.columns if c != "file_name"]
        mapped = [self._message_expr(source_columns).alias("m_message")]
        if self.level_field:
            mapped.append(self._resolve(self.level_field).cast(pl.String).alias("level"))
        if self.seq_id_field:
            mapped.append(self._resolve(self.seq_id_field).cast(pl.String).alias("seq_id"))
        self.df = self.df.with_columns(mapped)

        if self.timestamp_field:
            self._add_timestamp()

        self._mandatory_columns = ["m_message"] + (["m_timestamp"] if self.timestamp_field else [])

        if self.extra_fields == "drop":
            keep = [c for c in ("m_timestamp", "m_message", "level", "seq_id", "file_name")
                    if c in self.df.columns]
            self.df = self.df.select(keep)

    def _resolve(self, path):
        """Turn a JSON-pointer-ish path ("log/logger") into an expression."""
        if path in self.df.columns:
            # Either a plain key, or a key that flatten() has already turned into a column of that
            # exact name. Checking this first is what lets a spec name the same field whether or
            # not flatten is on, and also handles keys that contain the separator themselves.
            return pl.col(path)
        parts = path.split(self.path_separator)
        if parts[0] not in self.df.columns:
            raise ValueError(f"Field {parts[0]!r} is not in the data. Available: "
                             f"{', '.join(sorted(self.df.columns)[:20])}"
                             f"{' ...' if self.df.width > 20 else ''}")
        expression = pl.col(parts[0])
        for part in parts[1:]:
            expression = expression.struct.field(part)
        return expression

    def _message_expr(self, columns):
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
        # Nothing declared: fold the record into "key=value" text, skipping the keys this record
        # does not have. Skipping matters - JSON logs are sparse, and encoding the nulls too would
        # make most of every message read '"SomeKey":null'.
        pieces = [pl.when(pl.col(c).is_not_null())
                    .then(pl.concat_str([pl.lit(f"{c}="), self._stringify(c), pl.lit(" ")]))
                    .otherwise(pl.lit(""))
                  for c in columns]
        return pl.concat_str(pieces).str.strip_chars()

    def _stringify(self, column):
        """Cast a column to String for the key=value fallback. Struct/List columns cannot be
        cast directly (Polars raises InvalidOperationError), which otherwise crashes the fallback
        on any JSON with nested objects/arrays that flatten() didn't fully unnest (e.g. a struct
        nested inside a list)."""
        return self._stringify_expr(pl.col(column), self.df.schema[column])

    @staticmethod
    def _stringify_expr(expr, dtype):
        if isinstance(dtype, pl.Struct):
            return expr.struct.json_encode()
        if isinstance(dtype, pl.List):
            inner = dtype.inner
            elem = JsonLoader._stringify_expr(pl.element(), inner) \
                if isinstance(inner, (pl.Struct, pl.List)) else pl.element().cast(pl.String)
            return expr.list.eval(elem).list.join(",")
        return expr.cast(pl.String)

    def _add_timestamp(self):
        source = self._resolve(self.timestamp_field)
        if self.timestamp_epoch:
            parsed = self.df.select(
                pl.from_epoch(source, time_unit=self.timestamp_epoch).alias("m_timestamp")
            ).to_series()
        else:
            strings = self.df.select(source.cast(pl.String).alias("t")).to_series()
            chosen = self._pick_format(strings, self.timestamp_formats)
            parsed = strings.str.to_datetime(format=chosen, strict=False)
        if self.timestamp_naive and getattr(parsed.dtype, "time_zone", None):
            # Every other LogLead loader yields naive datetimes; mixing the two breaks pl.concat.
            parsed = parsed.dt.replace_time_zone(None)
        if getattr(parsed.dtype, "time_unit", None) != "us":
            # Same reason: from_epoch(time_unit='ms') yields Datetime('ms'), which will not concat
            # with the Datetime('us') every other loader produces.
            parsed = parsed.cast(pl.Datetime("us", getattr(parsed.dtype, "time_zone", None)))
        unparsed = parsed.null_count()
        if unparsed:
            print(f"WARNING! JsonLoader could not parse {unparsed} of {len(parsed)} values in "
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

    @staticmethod
    def _flatten(df, separator):
        while True:
            structs = [c for c, dtype in df.schema.items() if isinstance(dtype, pl.Struct)]
            if not structs:
                return df
            for column in structs:
                names = [field.name for field in df.schema[column].fields]
                renamed = [f"{column}{separator}{name}" for name in names]
                clashes = [name for name in renamed if name in df.columns]
                if clashes:
                    raise ValueError(f"Flattening {column!r} would overwrite existing column(s): "
                                     f"{', '.join(clashes)}. Use flatten=False or a different "
                                     f"path_separator.")
                df = df.with_columns(
                    [pl.col(column).struct.field(name).alias(alias)
                     for name, alias in zip(names, renamed)]).drop(column)

    # Checks ----------------------------------------------------------------------------------

    def check_for_nulls_and_non_utf8(self):
        """
        BaseLoader prints a four-line warning per column that has nulls. For JSON that is noise,
        not signal: keys legitimately differ per record, so a file with 90 keys routinely produces
        80-odd sparse columns. Summarize instead, and keep the non-UTF-8 warning, which is a real
        problem rather than an expected shape.
        """
        sparse = [(c, n) for c, n in zip(self.df.columns, self.df.null_count().row(0)) if n]
        if sparse:
            worst = sorted(sparse, key=lambda item: -item[1])[:3]
            listed = ", ".join(f"{c} ({n})" for c, n in worst)
            print(f"JsonLoader: {len(sparse)} of {self.df.width} columns contain nulls out of "
                  f"{len(self.df)} rows - expected for JSON, where keys vary per record. "
                  f"Most null: {listed}.")

        for column, dtype in self.df.schema.items():
            if dtype == pl.Utf8:
                bad = self.df.filter(pl.col(column).str.contains("�")).height
                if bad:
                    print(f"WARNING! Column '{column}' has {bad} non-UTF-8 encoded values out of "
                          f"{len(self.df)}. To investigate: "
                          f"<DF_NAME>.filter(pl.col('{column}').str.contains('�'))")
