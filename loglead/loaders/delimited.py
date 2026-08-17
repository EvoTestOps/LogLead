import codecs
import fnmatch
import glob
import os
import re

import polars as pl
import yaml

from .base import BaseLoader

__all__ = ['DelimitedLoader']

"""
DelimitedLoader Class

One configurable loader for delimited text that names its own columns - CSV, TSV and the
self-describing header formats (docs/log-format-support.md section 5 item 3). This is how exported
observability data usually arrives: SIEM and audit exports, Zeek, W3C/IIS, and the
'*_structured.csv' files loghub-style datasets ship as their already-parsed form.

Polars reads the bytes, so the work here is the mapping layer, plus one thing the other loaders
never have to do: work out *where the column names come from*. There are three answers and they are
decided per file, so a directory holding more than one reads in a single call:

  'row'   a header line, as in any CSV. The separator is sniffed from that line.
  'zeek'  Zeek's TSV: '#separator', '#fields', '#types', '#empty_field' and '#unset_field' are all
          declared in the file, so the names, the delimiter, the null markers *and* the column
          types come out of the data and there is nothing to configure.
  'w3c'   W3C extended, which IIS writes: a '#Fields:' directive, space separated. The directive
          repeats at every log rotation, and a file can start with data before its first one.

As in JsonLoader and AccessLogLoader the keyword arguments below are also the keys of a format
spec, so a spec file is nothing more than a serialized call:

    DelimitedLoader(filename="conn.log", format="zeek").execute()             # shipped spec
    DelimitedLoader(filename="logs", filename_pattern="*.log",                # a tree of files
                    format="zeek").execute()
    DelimitedLoader(filename="audit.csv", message_field="Content").execute()  # a plain CSV
    DelimitedLoader(filename="audit.csv", format="./my_format.yml").execute() # your own spec

Explicit keyword arguments always win over the values in a spec.

- filename (str): file, or directory to walk when filename_pattern is given.
- filename_pattern (str, optional): glob applied within each subdirectory, as in RawLoader. When
  given, a 'file_name' column is added and every file is read with its own header.
- min_file_size (int, optional): skip files smaller than this many bytes.
- strip_full_data_path (str, optional): prefix removed from 'file_name'.
- n_rows (int, optional): read at most this many rows *per file*.
- file_pattern (str, optional): only read files whose name matches this glob. Lets a spec state
  which files it applies to.
- header (str): 'auto' (default), 'row', 'zeek', 'w3c' or 'none'. 'auto' decides per file from its
  first lines - '#separator'/'#fields' means Zeek, '#Fields:' means W3C, anything else is taken to
  have a header row, which is this item's premise. 'none' needs columns.
- columns (list, optional): the column names. Required for header='none', and used as the fallback
  for a W3C file that carries no '#Fields:' of its own - real IIS logs are routinely concatenated
  or rotated in a way that loses the directive, and the standard field set is then the only way to
  read them. A fallback that does not match the file's actual width is an error rather than a
  silent misalignment.
- separator (str, optional): the delimiter. Omit to take it from '#separator' (Zeek), to use a
  space (W3C), or to sniff it from the header line among ',', tab, ';' and '|'.
- comment_prefix (str, optional): lines to skip. Defaults to '#' for the self-describing styles -
  which is what makes a mid-file '#Fields:' redeclaration and Zeek's trailing '#close' harmless -
  and to None for a header row, where '#' is an ordinary character.
- null_values (list, optional): values read as null. Defaults to what the file declares
  ('#unset_field', '#empty_field') for Zeek, ['-'] for W3C, and nothing for a header row, where a
  bare '-' is more often data than a marker - loghub writes exactly that for "not an alert".
- declared_types (bool): cast columns to the types the file declares, i.e. Zeek's '#types' line.
  On by default: it is the second half of what "self-describing" is worth, and it is what puts
  duration/byte counts in front of AnomalyDetector(numeric_cols=...) as numbers rather than text.
- infer_schema_length (int or None): rows scanned to infer column types where they are not
  declared. Defaults to None, meaning all of them, for JsonLoader's reason - Polars' own default of
  100 mistypes any column whose first hundred values are unrepresentative. Set an integer for a
  file too large to scan twice.
- timestamp_field (str or list, optional): column parsed into 'm_timestamp'. A list is joined with
  a space first, which is the normal shape here rather than a corner case: W3C splits the timestamp
  into 'date' and 'time', and loghub splits it differently in every system.
- timestamp_formats (list, optional): candidate chrono formats. The best one is chosen on a sample
  and then applied to the whole column in one vectorized pass. Omit to let Polars infer.
- timestamp_epoch (str, optional): 's', 'ms', 'us' or 'ns' when the column is an epoch number
  instead of a string. Fractional seconds are kept - Zeek's 'ts' is 1521911721.255387 - so this is
  not the same as casting to an integer. Mutually exclusive with timestamp_formats, and takes a
  single column rather than a list.
- timestamp_naive (bool): strip the time zone from 'm_timestamp'. On by default because every other
  LogLead loader produces naive datetimes, and Polars refuses to concat naive with tz-aware frames.
- message_field (str, optional): column used as 'm_message'.
- line_format (str, optional): template rendering several columns into 'm_message', e.g.
  "{cs-method} {cs-uri-stem} {sc-status}". Given neither, the row is rendered as 'name=value' text,
  skipping the columns this row has no value for. That fallback is the usual case here: Zeek and
  W3C have no message column at all, and their rows only become a log line when the column names
  are put back in front of the values. label_field is never rendered into the message, whichever
  way it is built - see below.
- level_field / seq_id_field (str, optional): columns copied to 'level' and 'seq_id'.
- label_field (str, optional): column holding a ground-truth label, turned into the boolean
  'normal' column (and hence 'anomaly'). This is the one format family that routinely arrives
  labelled - loghub's structured CSVs keep the alert category in 'Label', IoT-23 appends 'label' to
  Zeek's conn.log - so reading it is part of reading the format.
- normal_values / anomaly_values (list, optional): which label values mean normal, or which mean
  anomalous; give one, not both. A label that is null - a row from a file that does not carry the
  column at all - stays null rather than being guessed at. The label column is kept out of the
  rendered m_message, since a message that states the answer makes every detector look perfect;
  a label that is *also* derivable from another column (IoT-23's det_label names the attack) is
  not something the loader can spot, so name the columns you want with line_format there.
- extra_fields (str): 'keep' (default) leaves every column as its own column; 'drop' keeps only the
  mapped ones.

Files are concatenated with how='diagonal_relaxed', so a directory whose files have different
columns lands wide-and-sparse the same way heterogeneous JSON does. That is the normal case for
this family, not an edge one: a Zeek output directory is 35 log types with 35 different '#fields'
lines, and a loghub download is one CSV per system with a different header in each.

Not implemented: a '#Fields:' redeclaration that changes the field set mid-file (the first one
found is applied to the whole file, which is lnav's "lock the format for the file" rule), Zeek's
JSON output (that is a JsonLoader spec), and reading '#types' beyond scalars - a set or vector
column is kept as the raw '1.2.3.4,5.6.7.8' text.
"""

_UNSET = object()

_SPEC_KEYS = ("header", "columns", "separator", "comment_prefix", "null_values", "declared_types",
              "file_pattern", "timestamp_field", "timestamp_formats", "timestamp_epoch",
              "timestamp_naive", "message_field", "line_format", "level_field", "seq_id_field",
              "label_field", "normal_values", "anomaly_values", "extra_fields")

_DEFAULTS = {"header": "auto", "columns": None, "separator": None, "comment_prefix": _UNSET,
             "null_values": _UNSET, "declared_types": True, "file_pattern": None,
             "timestamp_field": None, "timestamp_formats": None, "timestamp_epoch": None,
             "timestamp_naive": True, "message_field": None, "line_format": None,
             "level_field": None, "seq_id_field": None, "label_field": None,
             "normal_values": None, "anomaly_values": None, "extra_fields": "keep"}

# Keys a spec may carry that describe the format rather than configure the loader.
_SPEC_METADATA = ("name", "description", "sample")

_FORMATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "delimited_formats")

_TEMPLATE_FIELD = re.compile(r"\{([^{}]+)\}")

_HEADER_STYLES = ("auto", "row", "zeek", "w3c", "none")

# Sniffed in this order, so a file holding both commas and tabs is read as the one that splits it
# into more fields; ties keep the earlier, more common delimiter.
_SEPARATORS = (",", "\t", ";", "|")

# How many lines of a file are read to find its header. A W3C file does not have to start with its
# '#Fields:' directive - the Exchange log this was developed against opens with one data line, then
# declares its fields - so this cannot be "the first line".
_SAMPLE_LINES = 1000

# Zeek's own type names, from its log-formats documentation. Everything not listed - string, addr,
# enum, and the set/vector containers - stays text.
_ZEEK_TYPES = {"count": pl.Int64, "int": pl.Int64, "port": pl.Int64,
               "interval": pl.Float64, "double": pl.Float64, "time": pl.Float64,
               "bool": pl.Boolean}

# Zeek writes booleans as T/F, which no cast understands.
_ZEEK_TRUE, _ZEEK_FALSE = "T", "F"

_EPOCH_UNITS = {"s": 1_000_000, "ms": 1_000, "us": 1, "ns": 0.001}


class _FileFormat:
    """How one file is to be read: the answer to "where do the column names come from"."""

    def __init__(self, style, names, types, separator, null_values, comment_prefix):
        self.style = style
        self.names = names
        self.types = types
        self.separator = separator
        self.null_values = null_values
        self.comment_prefix = comment_prefix


class DelimitedLoader(BaseLoader):
    def __init__(self, filename, format=None, filename_pattern=None, min_file_size=0,
                 strip_full_data_path=None, n_rows=None, infer_schema_length=None,
                 header=_UNSET, columns=_UNSET, separator=_UNSET, comment_prefix=_UNSET,
                 null_values=_UNSET, declared_types=_UNSET, file_pattern=_UNSET,
                 timestamp_field=_UNSET, timestamp_formats=_UNSET, timestamp_epoch=_UNSET,
                 timestamp_naive=_UNSET, message_field=_UNSET, line_format=_UNSET,
                 level_field=_UNSET, seq_id_field=_UNSET, label_field=_UNSET,
                 normal_values=_UNSET, anomaly_values=_UNSET, extra_fields=_UNSET):

        cfg = dict(_DEFAULTS)
        if format is not None:
            cfg.update(self._read_spec(format))
        for key, value in (("header", header), ("columns", columns), ("separator", separator),
                           ("comment_prefix", comment_prefix), ("null_values", null_values),
                           ("declared_types", declared_types), ("file_pattern", file_pattern),
                           ("timestamp_field", timestamp_field),
                           ("timestamp_formats", timestamp_formats),
                           ("timestamp_epoch", timestamp_epoch),
                           ("timestamp_naive", timestamp_naive), ("message_field", message_field),
                           ("line_format", line_format), ("level_field", level_field),
                           ("seq_id_field", seq_id_field), ("label_field", label_field),
                           ("normal_values", normal_values), ("anomaly_values", anomaly_values),
                           ("extra_fields", extra_fields)):
            if value is not _UNSET:
                cfg[key] = value

        if cfg["header"] not in _HEADER_STYLES:
            raise ValueError(f"header must be one of {', '.join(_HEADER_STYLES)}, "
                             f"got {cfg['header']!r}")
        if cfg["header"] == "none" and not cfg["columns"]:
            raise ValueError("header='none' needs columns=[...] - with no header line and no names "
                             "given there is nothing to call the fields")
        if cfg["extra_fields"] not in ("keep", "drop"):
            raise ValueError(f"extra_fields must be 'keep' or 'drop', got {cfg['extra_fields']!r}")
        if cfg["timestamp_epoch"] and cfg["timestamp_formats"]:
            raise ValueError("give either timestamp_epoch or timestamp_formats, not both")
        if cfg["timestamp_epoch"] and cfg["timestamp_epoch"] not in _EPOCH_UNITS:
            raise ValueError(f"timestamp_epoch must be one of {', '.join(_EPOCH_UNITS)}, "
                             f"got {cfg['timestamp_epoch']!r}")
        if cfg["timestamp_epoch"] and not isinstance(cfg["timestamp_field"], str):
            raise ValueError("timestamp_epoch takes a single timestamp_field, not a list: an epoch "
                             "number is one column by definition")
        if cfg["message_field"] and cfg["line_format"]:
            raise ValueError("give either message_field or line_format, not both")
        if cfg["normal_values"] and cfg["anomaly_values"]:
            raise ValueError("give either normal_values or anomaly_values, not both - the other "
                             "side is everything else")
        if (cfg["normal_values"] or cfg["anomaly_values"]) and not cfg["label_field"]:
            raise ValueError("normal_values/anomaly_values need label_field to say which column "
                             "they are values of")

        self.format = format
        self.filename_pattern = filename_pattern
        self.min_file_size = min_file_size
        self.strip_full_data_prefix = strip_full_data_path
        self.n_rows = n_rows
        self.infer_schema_length = infer_schema_length
        for key in _SPEC_KEYS:
            setattr(self, key, cfg[key])
        # Which header style each file turned out to use, for the error messages and the warning.
        self._styles_used = {}
        # Filled in by preprocess(), since which columns are mandatory depends on the mapping.
        self._mandatory_columns = []
        super().__init__(filename)

    # Spec handling ---------------------------------------------------------------------------

    @staticmethod
    def _read_spec(format):
        """Resolve a shipped spec name, or a path to a spec file, into a dict of loader options."""
        # isfile, not exists: a shipped spec name is frequently also the name of a directory in the
        # caller's working directory ('zeek', 'loghub'), and exists() would resolve to that.
        path = format if os.path.isfile(format) else os.path.join(_FORMATS_DIR, f"{format}.yml")
        if not os.path.isfile(path):
            available = sorted(f[:-4] for f in os.listdir(_FORMATS_DIR) if f.endswith(".yml")) \
                if os.path.isdir(_FORMATS_DIR) else []
            raise FileNotFoundError(
                f"No delimited format spec {format!r}. Give a path to a .yml file, or one of: "
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
        frames = pl.collect_all([self._scan(path, add_file_col) for path in paths])
        # Files read under different headers do not have the same columns, and neither do two files
        # of the same style - a Zeek directory is 35 log types with 35 different '#fields' lines.
        self.df = frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")

    def _collect_paths(self):
        if not self.filename_pattern:
            if self.file_pattern and not fnmatch.fnmatch(os.path.basename(self.filename),
                                                         self.file_pattern):
                raise ValueError(f"{self.filename} does not match the spec's file_pattern "
                                 f"{self.file_pattern!r}")
            return [self.filename]

        paths = []
        for subdir, _, _ in os.walk(self.filename):
            for path in sorted(glob.glob(os.path.join(subdir, self.filename_pattern))):
                if os.path.getsize(path) <= self.min_file_size:
                    continue
                if self.file_pattern and not fnmatch.fnmatch(os.path.basename(path),
                                                             self.file_pattern):
                    continue
                paths.append(path)
        if not paths:
            raise ValueError(f"No files matching pattern {self.filename_pattern} "
                             f"in directory {os.path.abspath(self.filename)}. "
                             f"Check the pattern, min_file_size and file_pattern.")
        return paths

    def _scan_lines(self, path):
        """Read a file as whole lines.

        This is RawLoader's read: an impossible separator and quote_char=None, so Polars hands back
        whole lines instead of trying to find fields in them. Used only to find the header - the
        data itself is read by the CSV reader proper.
        """
        return pl.scan_csv(path, has_header=False, schema={"column_1": pl.String},
                           infer_schema=False, quote_char=None, separator=self._csv_separator,
                           encoding="utf8-lossy", truncate_ragged_lines=True)

    def _scan(self, path, add_file_col):
        """One lazy query per file: work out its header, then read it under that header."""
        file_format = self._describe(path)
        self._styles_used[path] = file_format.style

        if file_format.style == "row":
            query = pl.scan_csv(path, has_header=True, separator=file_format.separator,
                                comment_prefix=file_format.comment_prefix,
                                null_values=file_format.null_values,
                                infer_schema_length=self.infer_schema_length, n_rows=self.n_rows,
                                encoding="utf8-lossy", truncate_ragged_lines=True,
                                **self._file_path_option(add_file_col))
        else:
            # The names come from the file's own directives, or from columns=, so the first line is
            # data rather than a header. quote_char=None because neither Zeek nor W3C quotes: both
            # escape instead, and letting the reader treat a quote as an opening one joins rows.
            declared = file_format.types and self.declared_types
            query = pl.scan_csv(path, has_header=False, new_columns=file_format.names,
                                separator=file_format.separator,
                                comment_prefix=file_format.comment_prefix,
                                null_values=file_format.null_values, quote_char=None,
                                n_rows=self.n_rows, encoding="utf8-lossy",
                                truncate_ragged_lines=True,
                                **({"infer_schema": False} if declared
                                   else {"infer_schema_length": self.infer_schema_length}),
                                **self._file_path_option(add_file_col))
            if declared:
                query = query.with_columns(self._declared_casts(file_format))

        if add_file_col and self.strip_full_data_prefix:
            query = query.with_columns(
                pl.col("file_name").str.strip_prefix(self.strip_full_data_prefix).alias("file_name"))
        return query

    @staticmethod
    def _file_path_option(add_file_col):
        return {"include_file_paths": "file_name"} if add_file_col else {}

    @staticmethod
    def _declared_casts(file_format):
        """Cast each column to the type the file said it was.

        strict=False throughout: a declared type is what the writer intended, not a guarantee about
        every row, and one unparseable value should become null rather than fail the load.
        """
        casts = []
        for name, declared in zip(file_format.names, file_format.types):
            dtype = _ZEEK_TYPES.get(declared)
            if dtype is None:
                continue
            if dtype == pl.Boolean:
                casts.append(pl.when(pl.col(name) == _ZEEK_TRUE).then(True)
                               .when(pl.col(name) == _ZEEK_FALSE).then(False)
                               .otherwise(None).alias(name))
            else:
                casts.append(pl.col(name).cast(dtype, strict=False).alias(name))
        return casts

    # Header handling -------------------------------------------------------------------------

    def _describe(self, path):
        """Decide where this file's column names, delimiter and null markers come from."""
        sample = self._scan_lines(path).head(_SAMPLE_LINES).collect()["column_1"].drop_nulls()
        lines = sample.to_list()
        style = self.header if self.header != "auto" else self._detect_style(lines)

        if style == "zeek":
            return self._zeek_format(path, lines)
        if style == "w3c":
            return self._w3c_format(path, lines)
        if style == "none":
            data = next((line for line in lines if line), "")
            separator = self._separator_for(path, data, style)
            self._check_width(path, lines, self.columns, separator)
            return _FileFormat("none", list(self.columns), None, separator,
                               self._null_values(style), self._comment_prefix(style))
        return self._row_format(path, lines)

    @staticmethod
    def _detect_style(lines):
        """Zeek and W3C both announce themselves; anything else is taken to have a header row.

        Both markers are looked for anywhere in the sample rather than on the first line, because a
        real W3C log routinely starts with the tail of the previous rotation's data.
        """
        for line in lines:
            if line.startswith("#separator") or line.startswith("#fields\t"):
                return "zeek"
            if line[:8].lower() == "#fields:":
                return "w3c"
        return "row"

    def _zeek_format(self, path, lines):
        """Read Zeek's header block: it declares the delimiter, the names, the types and the two
        markers it writes for an unset and an empty field."""
        directives = {}
        separator = "\t"
        for line in lines:
            if not line.startswith("#"):
                continue
            if line.startswith("#separator"):
                # The one directive not written with the separator it is declaring, and its value
                # is an escape sequence rather than the character itself.
                value = line.split(None, 1)
                if len(value) == 2:
                    separator = codecs.decode(value[1].strip(), "unicode_escape")
                continue
            key, _, rest = line.partition(separator)
            directives[key] = rest
        separator = self.separator or separator

        names = directives.get("#fields", "").split(separator) if "#fields" in directives else None
        if not names:
            if not self.columns:
                raise ValueError(f"{path} looks like a Zeek log but has no '#fields' line, and no "
                                 f"columns= was given to stand in for it.")
            names = list(self.columns)
        types = directives.get("#types", "").split(separator) if "#types" in directives else None
        if types and len(types) != len(names):
            print(f"WARNING! DelimitedLoader: {os.path.basename(path)} declares {len(names)} "
                  f"'#fields' but {len(types)} '#types'; the types are ignored.")
            types = None

        declared_nulls = [directives[key] for key in ("#unset_field", "#empty_field")
                          if directives.get(key)]
        nulls = self.null_values if self.null_values is not _UNSET else (declared_nulls or None)
        return _FileFormat("zeek", names, types, separator, nulls, self._comment_prefix("zeek"))

    def _w3c_format(self, path, lines):
        """Read W3C's '#Fields:' directive, and check any redeclaration in the sample agrees."""
        declarations = [line.split(":", 1)[1].split() for line in lines
                        if line[:8].lower() == "#fields:"]
        distinct = {tuple(names) for names in declarations}
        if len(distinct) > 1:
            raise ValueError(f"{path} declares more than one set of '#Fields:' - "
                             f"{' | '.join(' '.join(names) for names in distinct)}. One file "
                             f"cannot be read under two field sets; split it, or pin one with "
                             f"columns=.")
        names = declarations[0] if declarations else None
        if not names:
            # A rotated or concatenated IIS log often loses the directive. The standard field set
            # is then the only way to read it, which is what a spec's columns= is for.
            if not self.columns:
                raise ValueError(f"{path} has no '#Fields:' directive, so its columns have no "
                                 f"names. Give columns=[...] (the format spec's fallback) or "
                                 f"header='row' if the first line is a header.")
            names = list(self.columns)
            self._check_width(path, lines, names, " ")
        return _FileFormat("w3c", names, None, self.separator or " ", self._null_values("w3c"),
                           self._comment_prefix("w3c"))

    def _row_format(self, path, lines):
        """The header is the first line the CSV reader will treat as one, i.e. the first line that
        is not skipped as a comment - which is also the line the delimiter is sniffed from."""
        prefix = self._comment_prefix("row")
        header = next((line for line in lines
                       if not (prefix and line.startswith(prefix))), None)
        if header is None:
            raise ValueError(f"{path} has no header row: it is empty, or every line is a comment.")
        return _FileFormat("row", None, None, self._separator_for(path, header, "row"),
                           self._null_values("row"), prefix)

    def _separator_for(self, path, line, style):
        if self.separator:
            return self.separator
        if style == "w3c":
            return " "
        counts = {sep: line.count(sep) for sep in _SEPARATORS}
        best = max(_SEPARATORS, key=lambda sep: counts[sep])
        if not counts[best]:
            raise ValueError(f"No delimiter found in the first line of {path}: it holds none of "
                             f"{', '.join(repr(s) for s in _SEPARATORS)}. Pass separator= if it "
                             f"uses another one, or use RawLoader if it is not delimited at all. "
                             f"First line: {line[:200]}")
        return best

    @staticmethod
    def _check_width(path, lines, names, separator):
        """Guard the columns= fallback: names that do not match the file's width would misalign
        every column rather than fail, which is the one failure mode worth spending a read on."""
        data = next((line for line in lines if line and not line.startswith("#")), None)
        if data is None:
            return
        width = len(data.split(separator))
        if width != len(names):
            raise ValueError(f"{path} has {width} fields per line but columns= names "
                             f"{len(names)}. Reading it that way would misalign every column. "
                             f"First line: {data[:200]}")

    def _comment_prefix(self, style):
        if self.comment_prefix is not _UNSET:
            return self.comment_prefix
        # '#' is structure in the self-describing styles - and skipping it is what makes a mid-file
        # '#Fields:' redeclaration and Zeek's '#close' harmless - but ordinary data in a CSV.
        return "#" if style in ("zeek", "w3c") else None

    def _null_values(self, style):
        if self.null_values is not _UNSET:
            return self.null_values
        if style == "w3c":
            return ["-"]  # W3C's own marker for a field the request did not carry
        # Not for a header row: there a bare '-' is more often data than a marker. loghub writes it
        # in Label to mean "not an alert", and nulling it would erase the labels.
        return None

    # Mapping ---------------------------------------------------------------------------------

    def preprocess(self):
        # The rendered message must not contain the label. m_message is what every detector reads,
        # so a row that says 'label=Malicious' in its own text makes each of them trivially right.
        # file_name is left out for the same kind of reason - it is provenance, not what was logged.
        source_columns = [c for c in self.df.columns
                          if c not in ("file_name", self.label_field)]
        mapped = [self._message_expr(source_columns).alias("m_message")]
        if self.level_field:
            mapped.append(self._resolve(self.level_field).cast(pl.String).alias("level"))
        if self.seq_id_field:
            mapped.append(self._resolve(self.seq_id_field).cast(pl.String).alias("seq_id"))
        if self.label_field:
            mapped.append(self._normal_expr().alias("normal"))
        self.df = self.df.with_columns(mapped)

        if self.timestamp_field:
            self._add_timestamp()

        self._mandatory_columns = ["m_message"] + (["m_timestamp"] if self.timestamp_field else [])

        if self.extra_fields == "drop":
            keep = [c for c in ("m_timestamp", "m_message", "level", "seq_id", "normal",
                                "file_name") if c in self.df.columns]
            self.df = self.df.select(keep)

    def _resolve(self, field):
        if field not in self.df.columns:
            raise ValueError(f"Column {field!r} is not in the data. Available: "
                             f"{', '.join(sorted(self.df.columns)[:20])}"
                             f"{' ...' if self.df.width > 20 else ''}")
        return pl.col(field)

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
        # Nothing declared: fold the row into "name=value" text, skipping the columns this row has
        # no value for. Unlike an access log line, a delimited row is not readable on its own - the
        # names live in the header - so putting them back is what turns it into a log message, and
        # skipping the nulls is what keeps a wide, sparse frame from rendering mostly '=' signs.
        pieces = [pl.when(pl.col(c).is_not_null())
                    .then(pl.concat_str([pl.lit(f"{c}="), pl.col(c).cast(pl.String), pl.lit(" ")]))
                    .otherwise(pl.lit(""))
                  for c in columns]
        return pl.concat_str(pieces).str.strip_chars()

    def _normal_expr(self):
        """Turn the label column into 'normal'. BaseLoader.add_ano_col() derives 'anomaly' from it.

        A null label stays null rather than becoming True: in a directory where only some files
        carry the column, "this row has no label" and "this row is normal" are different claims,
        and the second one would quietly invent ground truth.
        """
        label = self._resolve(self.label_field)
        if self.normal_values:
            return pl.when(label.is_null()).then(None) \
                     .otherwise(label.cast(pl.String).is_in([str(v) for v in self.normal_values]))
        if self.anomaly_values:
            return pl.when(label.is_null()).then(None) \
                     .otherwise(label.cast(pl.String).is_in([str(v) for v in self.anomaly_values])
                                .not_())
        raise ValueError(f"label_field={self.label_field!r} needs normal_values or anomaly_values "
                         f"to say which of its values mean what. Values found: "
                         f"{', '.join(str(v) for v in self.df[self.label_field].unique().head(10))}")

    def _timestamp_source(self):
        """The timestamp as one string expression, joining the columns it is spread across."""
        fields = [self.timestamp_field] if isinstance(self.timestamp_field, str) \
            else list(self.timestamp_field)
        pieces = [self._resolve(field).cast(pl.String) for field in fields]
        if len(pieces) == 1:
            return pieces[0]
        return pl.concat_str(pieces, separator=" ", ignore_nulls=False)

    def _add_timestamp(self):
        if self.timestamp_epoch:
            source = self.df.select(self._resolve(self.timestamp_field).alias("t")).to_series()
            parsed = self._epoch()
        else:
            source = self.df.select(self._timestamp_source().alias("t")).to_series()
            chosen = self._pick_format(source, self.timestamp_formats)
            parsed = source.str.to_datetime(format=chosen, strict=False)
        if self.timestamp_naive and getattr(parsed.dtype, "time_zone", None):
            # Every other LogLead loader yields naive datetimes; mixing the two breaks pl.concat.
            parsed = parsed.dt.replace_time_zone(None)
        if getattr(parsed.dtype, "time_unit", None) != "us":
            parsed = parsed.cast(pl.Datetime("us", getattr(parsed.dtype, "time_zone", None)))
        unparsed = parsed.null_count() - source.null_count()
        if unparsed > 0:
            print(f"WARNING! DelimitedLoader could not parse {unparsed} of {len(parsed)} values in "
                  f"'{self.timestamp_field}' into m_timestamp. Check timestamp_formats"
                  f"{' or timestamp_epoch' if self.timestamp_epoch else ''}.")
        self.df = self.df.with_columns(parsed.alias("m_timestamp"))

    def _epoch(self):
        """An epoch column, keeping its fraction.

        Zeek writes 1521911721.255387 - seconds with microseconds after the point - so casting to
        an integer first, which is what from_epoch(time_unit='s') needs, would throw away the part
        that orders events inside a second. Scaling to microseconds first keeps it.
        """
        scale = _EPOCH_UNITS[self.timestamp_epoch]
        micros = (self._resolve(self.timestamp_field).cast(pl.Float64, strict=False) * scale) \
            .round(0).cast(pl.Int64, strict=False)
        return self.df.select(pl.from_epoch(micros, time_unit="us").alias("t")).to_series()

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
        Same reasoning as JsonLoader's, AccessLogLoader's and SyslogLoader's overrides: BaseLoader
        prints a four-line warning per column that has nulls, and here nulls are the designed
        outcome twice over - Zeek and W3C write a marker for every field an event did not carry,
        and a directory of files with different headers is sparse by construction. Summarize
        instead, and keep the non-UTF-8 warning, which is a real problem rather than a shape.
        """
        sparse = [(c, n) for c, n in zip(self.df.columns, self.df.null_count().row(0)) if n]
        if sparse:
            worst = sorted(sparse, key=lambda item: -item[1])[:3]
            listed = ", ".join(f"{c} ({n})" for c, n in worst)
            styles = ", ".join(sorted(set(self._styles_used.values()))) or "delimited"
            print(f"DelimitedLoader: {len(sparse)} of {self.df.width} columns contain nulls out of "
                  f"{len(self.df)} rows - expected for {styles} data, where fields are optional "
                  f"and files differ in what they carry. Most null: {listed}.")

        for column, dtype in self.df.schema.items():
            if dtype == pl.Utf8:
                bad = self.df.filter(pl.col(column).str.contains("�")).height
                if bad:
                    print(f"WARNING! Column '{column}' has {bad} non-UTF-8 encoded values out of "
                          f"{len(self.df)}. To investigate: "
                          f"<DF_NAME>.filter(pl.col('{column}').str.contains('�'))")
