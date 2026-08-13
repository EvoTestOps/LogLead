import fnmatch
import glob
import os
import re

import polars as pl

from .access_log import AccessLogLoader
from .adfa import ADFALoader
from .awsctd import AWSCTDLoader
from .base import BaseLoader
from .bgl import BGLLoader
from .hadoop import HadoopLoader
from .hdfs import HDFSLoader
from .json import JsonLoader
# The logfmt pair grammar and its conventional key names are the detector's logfmt test and its
# fallback JSON field mapping. Importing them rather than restating them is the point: a detector
# that recognizes a format differently from the loader that reads it is worse than no detector.
from .logfmt import LogfmtLoader, _PAIR, _TIMESTAMP_KEYS, _MESSAGE_KEYS, _LEVEL_KEYS
from .nezha import NezhaLoader
from .pro import ProLoader
from .raw import RawLoader
from .supercomputers import ThuSpiLibLoader
from .syslog import SyslogLoader, _RFC3164, _RFC5424

__all__ = ['AutoLoader', 'Detection', 'detect_format']

"""
AutoLoader Class

The loader for when you do not know which loader you need. It looks at a file, decides what format
it is in, and builds the loader that reads that format - so `AutoLoader(filename=...).execute()`
works on anything, and tells you what it decided (docs/log-format-support.md section 5 item 6).

    AutoLoader(filename="mystery.log").execute()                     # one file
    AutoLoader(filename="logs", filename_pattern="*.log").execute()  # a tree, detected per file
    AutoLoader(filename="~/Datasets/hdfs").execute()                 # a known public dataset
    loader.detections()                                              # what it chose, and why

Detection returns a **loader class and its arguments**, never a parsed frame: the spec-driven
loaders already know how to read their formats, and detection's whole job is to pick one of them
and name a format spec (docs/log-format-json-loader.md section 2). That is also why detect_format()
is importable on its own - the decision is inspectable without loading anything.

Two stages, most specific first:

1. **Dataset probe.** Public datasets that have their own loader are recognized from what sits
   *next to* the log, not only from the log: HDFSLoader needs its anomaly_label.csv and
   HadoopLoader its abnormal_label.txt, and those live at a known place relative to the data. The
   sibling file both confirms the dataset and supplies the argument, so a dataset whose labels are
   missing is deliberately *not* claimed - it falls through to stage 2 and loads unlabeled rather
   than crashing or silently labelling everything normal.
2. **Format probe, per file.** JSON, then web access log, then syslog, then logfmt, then a generic
   timestamped-text pass, then plain text. Specific before generic, which is what stops a logfmt
   line whose value is an ISO timestamp from being read as generic timestamped text. Each candidate
   is scored as a match rate over a sample of the file and the best one above min_match_rate wins -
   the rule SyslogLoader already uses to choose between the two syslog RFCs, generalized.

Detection is per file because a log folder holding several formats is the normal case rather than
the exception. When every file agrees the whole tree is handed to one loader, which keeps the
parallel multi-file read the loaders already do; only a genuinely mixed tree pays for one loader
per file.

- filename (str): file, or directory to walk when filename_pattern is given. A directory with no
  filename_pattern is only looked at as a dataset (stage 1).
- filename_pattern (str, optional): glob applied within each subdirectory, as in RawLoader.
- min_file_size (int, optional): skip files smaller than this many bytes.
- strip_full_data_path (str, optional): prefix removed from 'file_name'.
- min_match_rate (float): the fraction of sampled lines a format must match to be chosen. Defaults
  to 0.5, as in AccessLogLoader and SyslogLoader. Below it for every candidate, the file is loaded
  as plain text and said so.
- sample_lines (int): how many lines to look at per file. Defaults to 1000, as in SyslogLoader.
- system (str, optional): 'TrainTicket' or 'WebShop', needed only when the Nezha dataset is
  detected - nothing on disk says which of the two a Nezha directory holds.

Not implemented: delimited text with a header (CSV/TSV), because there is no loader to delegate to
- docs/log-format-support.md section 5 item 3 is not built. Nor JSON containers that wrap their
records under a key ({"Records": [...]}), since the key name is not guessable.
"""

# How many lines of a file are looked at to decide its format. SyslogLoader's number, for the same
# reason: enough to be representative, small enough that it is one head() read.
_SAMPLE_LINES = 1000

# A sample taken only from the top of a file can misrepresent it - one of the logfmt test files
# changes shape after its first ~1500 lines - so a chunk from the middle is read too. It costs a
# seek and one buffer, milliseconds even on a multi-gigabyte log.
_MID_CHUNK_BYTES = 256 * 1024

# Line shapes that identify a public dataset which has its own loader. Verified mutually exclusive
# against the real corpora: each scores 1.000 on its own dataset and 0.000 on every other one,
# syslog and access logs included. Deliberately absent is Hadoop's '%Y-%m-%d %H:%M:%S,%3f' opening,
# which is log4j's default and says "a Java program wrote this", not "this is the Hadoop dataset" -
# that one is a generic timestamp below, and the Hadoop dataset is recognized by its directory.
_DATASET_PATTERNS = (
    ("hdfs", r'^\d{6} \d{6} \d+ [A-Z]+ '),
    ("bgl", r'^\S+ \d{9,10} \d{4}\.\d{2}\.\d{2} \S+ \d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+ '),
    ("thuspilib", r'^\S+ \d{9,10} \d{4}\.\d{2}\.\d{2} \S+ [A-Z][a-z]{2} +\d{1,2} \d{2}:\d{2}:\d{2} '),
    ("profilence", r'^\d+ \d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}\.\d{3} '),
)

# Timestamp shapes for text that is none of the named formats, as (regex capturing the timestamp,
# the chrono format that parses it). Ordered specific before generic, which is what settles ties:
# a log4j line matches both the ',%3f' entry and the plain second-resolution one at the same rate,
# and the first-listed winner is the one that keeps the milliseconds.
_GENERIC_TIMESTAMPS = (
    (r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', "%Y-%m-%d %H:%M:%S,%3f"),
    (r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})', "%Y-%m-%d %H:%M:%S%.3f"),
    (r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)', "%Y-%m-%dT%H:%M:%S%.fZ"),
    (r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)', "%Y-%m-%dT%H:%M:%SZ"),
    (r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', "%Y-%m-%dT%H:%M:%S"),
    (r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', "%Y-%m-%d %H:%M:%S"),
    (r'(\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d{6})', "%Y-%m-%d-%H.%M.%S%.6f"),
    (r'(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}\.\d{3})', "%d.%m.%Y %H:%M:%S%.3f"),
    (r'(\d{6} \d{6}) ', "%y%m%d %H%M%S"),
)

# The loaders that can be handed a whole tree themselves. Everything else - the dataset loaders -
# is built per file, so the fast path below must not try to give them a filename_pattern.
_TREE_CAPABLE = (JsonLoader, AccessLogLoader, LogfmtLoader, SyslogLoader, RawLoader)

# JSON keys worth trying as the message/timestamp when no shipped spec fits. The logfmt convention
# plus the two spellings the shipped JSON datasets use.
_JSON_TIMESTAMP_KEYS = _TIMESTAMP_KEYS + ("@timestamp", "TimeCreated", "eventTime")
_JSON_MESSAGE_KEYS = _MESSAGE_KEYS + ("short_message", "Message", "log", "event")
_JSON_LEVEL_KEYS = _LEVEL_KEYS + ("Level", "log_level")


class Detection:
    """What the detector decided about one file or directory.

    `loader` and `kwargs` are the answer; `filename` is not in `kwargs`, so the same Detection can
    be applied to one file or to a whole tree. The rest is the reasoning, kept because a wrong
    guess is only debuggable if the guess is visible.
    """

    def __init__(self, loader, kwargs=None, format=None, rate=1.0, lines=0, replacement_chars=0,
                 note=None):
        self.loader = loader
        self.kwargs = kwargs or {}
        self.format = format or loader.__name__
        self.rate = rate
        self.lines = lines
        self.replacement_chars = replacement_chars
        self.note = note

    def key(self):
        """What makes two detections the same choice, so a tree can be loaded in one go."""
        return (self.loader.__name__,
                tuple(sorted((k, repr(v)) for k, v in self.kwargs.items())))

    def __repr__(self):
        return f"Detection({self.loader.__name__}, format={self.format!r}, rate={self.rate:.3f})"


# Sampling ------------------------------------------------------------------------------------

def _scan_lines(path):
    """Read a file as whole lines.

    This is RawLoader's read, shared by every spec-driven loader: an impossible separator and
    quote_char=None, so Polars hands back whole lines instead of looking for fields in them.
    """
    return pl.scan_csv(path, has_header=False, schema={"column_1": pl.String}, infer_schema=False,
                       quote_char=None, separator=BaseLoader._csv_separator,
                       encoding="utf8-lossy", truncate_ragged_lines=True)


def _mid_chunk(path, limit):
    """Lines from the middle of a file, to catch files whose head is not representative."""
    size = os.path.getsize(path)
    if size <= 2 * _MID_CHUNK_BYTES:
        return None  # the head already covers most of it
    with open(path, "rb") as handle:
        handle.seek(size // 2)
        raw = handle.read(_MID_CHUNK_BYTES)
    # Both ends of the chunk are partial lines, since it was cut at a byte offset.
    lines = raw.decode("utf-8", errors="replace").splitlines()[1:-1]
    return pl.Series("column_1", lines[:limit], dtype=pl.String) if lines else None


def _sample_lines(path, limit=_SAMPLE_LINES):
    """The lines every candidate is scored against: the head, plus a chunk from the middle."""
    head = _scan_lines(path).head(limit).collect()["column_1"].drop_nulls()
    if path.endswith(".gz"):
        # Seeking into a compressed stream means decompressing from the start anyway, so the head
        # is all that is affordable here.
        return head
    mid = _mid_chunk(path, limit)
    return head if mid is None else pl.concat([head, mid])


def _rate(sample, regex):
    """The fraction of sampled lines a pattern matches - the score every candidate is ranked by."""
    if not len(sample):
        return 0.0
    return float(sample.str.contains(regex).sum()) / len(sample)


# Stage 1: is this a public dataset with its own loader? ----------------------------------------

def _first_existing(*paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def detect_dataset(path, system=None):
    """Recognize a directory that holds a public dataset one of the bespoke loaders owns.

    Every check needs the dataset's own artifacts, not just its log lines, because those loaders
    need them as arguments: the labels file *is* the evidence.
    """
    if not os.path.isdir(path):
        return None
    names = set(os.listdir(path))

    # Nezha: two sibling directories of logs, traces, metrics and fault lists.
    if {"construct_data", "rca_data"} <= names:
        if system not in ("TrainTicket", "WebShop"):
            raise ValueError(
                f"{path} looks like the Nezha dataset, which holds two systems in one directory. "
                f"Pass system='TrainTicket' or system='WebShop' to say which one to load.")
        return Detection(NezhaLoader, {"system": system}, format=f"nezha/{system}")

    # ADFA-LD: syscall id traces, split into attack and training directories. The loader wants the
    # ADFA-LD directory itself, which may be the one given or may be one below it.
    for root in (path, os.path.join(path, "ADFA-LD")):
        if os.path.isdir(root) and \
                {"Attack_Data_Master", "Training_Data_Master"} <= set(os.listdir(root)):
            return Detection(ADFALoader, {"_root": root}, format="adfa")

    # AWSCTD: one sequence of syscall names per CSV, in named subdirectories of CSV/. The directory
    # has to be the one called CSV, not merely one that contains CSVs somewhere - HDFS ships a
    # preprocessed/ folder full of .csv files and would otherwise match this.
    for root in (path, os.path.join(path, "CSV")):
        if os.path.basename(root) == "CSV" and os.path.isdir(root) and \
                glob.glob(os.path.join(root, "*", "*.csv")):
            return Detection(AWSCTDLoader, {"_root": root}, format="awsctd")

    # Hadoop: one directory per application, and the labels beside them.
    labels = _first_existing(os.path.join(path, "abnormal_label.txt"))
    if labels and glob.glob(os.path.join(path, "application_*")):
        return Detection(HadoopLoader,
                         {"filename_pattern": "*.log", "labels_file_name": labels},
                         format="hadoop")

    # Profilence: the label is the file name, so both spellings have to be present.
    if any(fnmatch.fnmatch(n, "success*.txt") for n in names) and \
            any(fnmatch.fnmatch(n, "fail*.txt") for n in names):
        return Detection(ProLoader, {"_glob": os.path.join(path, "*.txt")}, format="profilence")

    # HDFS: one big log plus its labels, which the downloader unpacks into preprocessed/.
    labels = _first_existing(os.path.join(path, "preprocessed", "anomaly_label.csv"),
                             os.path.join(path, "anomaly_label.csv"))
    log = _first_existing(os.path.join(path, "HDFS.log"))
    if labels and log:
        return Detection(HDFSLoader, {"labels_file_name": labels, "_file": log}, format="hdfs")

    return None


def _detect_dataset_file(path, sample):
    """Recognize a single file as a dataset whose loader reads exactly that file."""
    for name, pattern in _DATASET_PATTERNS:
        rate = _rate(sample, pattern)
        if rate < 0.5:
            continue
        base = os.path.basename(path)

        if name == "hdfs":
            # HDFSLoader cannot run without its labels, and a copy of HDFS.log with no
            # anomaly_label.csv beside it is not the HDFS dataset - it is an HDFS-shaped log.
            folder = os.path.dirname(os.path.abspath(path))
            labels = _first_existing(os.path.join(folder, "preprocessed", "anomaly_label.csv"),
                                     os.path.join(folder, "anomaly_label.csv"))
            if not labels:
                print(f"AutoLoader: {base} looks like the HDFS dataset but no anomaly_label.csv "
                      f"was found beside it, so it is read as plain timestamped text without "
                      f"labels or sequences.")
                return None
            return Detection(HDFSLoader, {"labels_file_name": labels}, format="hdfs", rate=rate,
                             lines=len(sample))

        if name == "bgl":
            return Detection(BGLLoader, {}, format="bgl", rate=rate, lines=len(sample))

        if name == "thuspilib":
            # Thunderbird, Spirit and Liberty share one line shape and one loader, but Liberty's
            # lines carry no component[pid] field and need split_component=False. The content does
            # not separate them - 85% of Liberty lines also look component-shaped - so the file
            # name is the evidence, and the majority case is the default when it says nothing.
            split = not base.startswith("liberty")
            if not (base.startswith(("liberty", "tbird", "spirit"))):
                print(f"AutoLoader: {base} has the Thunderbird/Spirit/Liberty line shape, but its "
                      f"name does not say which. Assuming split_component=True; pass "
                      f"ThuSpiLibLoader(split_component=False) directly if this is Liberty.")
            return Detection(ThuSpiLibLoader, {"split_component": split},
                             format="thunderbird/spirit/liberty", rate=rate, lines=len(sample))

        if name == "profilence":
            # ProLoader takes the anomaly label from the file name, so without that convention it
            # would silently mark every sequence anomalous.
            if not base.startswith(("success", "fail")):
                continue
            return Detection(ProLoader, {}, format="profilence", rate=rate, lines=len(sample))
    return None


# Stage 2: which format is this file written in? ------------------------------------------------

def _spec_fields(spec):
    """The data fields a spec insists on. JsonLoader raises when one is missing, so a spec is only
    a candidate if the data has all of them - there is no partial credit to be had."""
    fields = {spec[key] for key in ("timestamp_field", "message_field", "seq_id_field",
                                    "level_field") if spec.get(key)}
    if spec.get("line_format"):
        fields |= set(re.findall(r"\{([^{}]+)\}", spec["line_format"]))
    return fields


def _json_keys(path, container):
    try:
        if container == "ndjson":
            return set(pl.scan_ndjson(path, infer_schema_length=500).collect_schema().names())
        return set(pl.read_json(path, infer_schema_length=500).columns)
    except Exception:
        return set()


def _detect_json(path, sample, min_match_rate):
    stripped = sample.str.strip_chars()
    non_blank = stripped.filter(stripped != "")
    if not len(non_blank):
        return None
    brace_rate = float((non_blank.str.slice(0, 1) == "{").sum()) / len(non_blank)
    first = non_blank.item(0)[:1]

    if first == "[" or (first == "{" and brace_rate < min_match_rate):
        container, kwargs = "array", {"container": "array"}
    elif brace_rate >= min_match_rate:
        container, kwargs = "ndjson", {"container": "ndjson"}
        if brace_rate < 0.999:
            # Real log directories interleave plain-text banners into NDJSON, and one such line
            # fails the whole file otherwise. Dropping them with a count is the documented policy.
            kwargs["on_error"] = "skip"
    else:
        return None

    keys = _json_keys(path, container)
    if not keys:
        return None

    # Elect a shipped spec: every field it names must be present, ranked by how much it names.
    best = None
    for name in JsonLoader.available_formats():
        spec = JsonLoader._read_spec(name)
        pattern = spec.get("file_pattern")
        if pattern and not fnmatch.fnmatch(os.path.basename(path), pattern):
            continue
        fields = _spec_fields(spec)
        if fields and fields <= keys and (best is None or len(fields) > best[0]):
            best = (len(fields), name)
    if best:
        # file_pattern has served its purpose as a filter; left in, it makes the loader refuse any
        # file whose name does not match, which is not what a detected spec should do.
        return Detection(JsonLoader, {"format": best[1], "file_pattern": None, **kwargs},
                         format=f"json/{best[1]}", rate=brace_rate, lines=len(sample))

    # No spec fits: fall back to the conventional key names, which is what LogfmtLoader does.
    for argument, candidates in (("timestamp_field", _JSON_TIMESTAMP_KEYS),
                                 ("message_field", _JSON_MESSAGE_KEYS),
                                 ("level_field", _JSON_LEVEL_KEYS)):
        found = next((key for key in candidates if key in keys), None)
        if found:
            kwargs[argument] = found
    return Detection(JsonLoader, kwargs, format=f"json/{container}", rate=brace_rate,
                     lines=len(sample))


def _detect_access_log(sample, min_match_rate):
    best_rate, best_name = 0.0, None
    for name in AccessLogLoader.available_formats():
        spec = AccessLogLoader._read_spec(name)
        regex = spec.get("pattern") or (
            AccessLogLoader._compile_log_format(spec["log_format"]) if spec.get("log_format")
            else None)
        if not regex:
            continue
        rate = _rate(sample, regex)
        if rate > best_rate:
            best_rate, best_name = rate, name
    if best_name and best_rate >= min_match_rate:
        return Detection(AccessLogLoader, {"format": best_name, "file_pattern": None},
                         format=f"access_log/{best_name}", rate=best_rate, lines=len(sample))
    return None


def _detect_syslog(sample, min_match_rate):
    rate = max(_rate(sample, _RFC5424), _rate(sample, _RFC3164))
    if rate >= min_match_rate:
        # Which RFC, and which one per file, is SyslogLoader's own job - it already decides that
        # from the first lines of each file it reads.
        return Detection(SyslogLoader, {"format": "auto"}, format="syslog", rate=rate,
                         lines=len(sample))
    return None


def _detect_logfmt(sample, min_match_rate):
    """Two or more key=value pairs on a line, which is what tells logfmt from a log that merely
    mentions one.

    Requiring the line to *start* with a pair would be wrong: real Grafana output puts free text
    before the pairs, and the extremely common '<timestamp> level=info msg="..."' shape starts with
    neither. Counting pairs instead separates logfmt from syslog by a factor of four.
    """
    if not len(sample):
        return None
    rate = float((sample.str.count_matches(_PAIR) >= 2).sum()) / len(sample)
    if rate >= min_match_rate:
        return Detection(LogfmtLoader, {}, format="logfmt", rate=rate, lines=len(sample))
    return None


def _detect_generic(sample, min_match_rate):
    for pattern, chrono in _GENERIC_TIMESTAMPS:
        rate = _rate(sample, pattern)
        if rate >= min_match_rate:
            return Detection(RawLoader,
                             {"timestamp_pattern": pattern, "timestamp_format": chrono,
                              "strict": False, "missing_timestamp_action": "keep"},
                             format=f"text/{chrono}", rate=rate, lines=len(sample))
    return None


def detect_format(path, min_match_rate=0.5, sample_lines=_SAMPLE_LINES):
    """Decide which loader reads this file, and with what arguments.

    Candidates are tried specific before generic, and the first one that clears min_match_rate on a
    sample of the file wins - lnav's rule, and the one SyslogLoader already applies to its two RFCs.
    Nothing here reads the whole file, and nothing here parses anything: the answer is a loader and
    its arguments.
    """
    if os.path.getsize(path) == 0:
        return Detection(RawLoader, {}, format="empty", rate=0.0,
                         note="file is empty")

    sample = _sample_lines(path, sample_lines)
    if not len(sample):
        return Detection(RawLoader, {}, format="text", rate=0.0, note="no readable lines")
    replacements = int(sample.str.count_matches("�").sum())

    dataset = _detect_dataset_file(path, sample)
    detection = dataset or (
        _detect_json(path, sample, min_match_rate)
        or _detect_access_log(sample, min_match_rate)
        or _detect_syslog(sample, min_match_rate)
        or _detect_logfmt(sample, min_match_rate)
        or _detect_generic(sample, min_match_rate))

    if detection is None:
        print(f"AutoLoader: no format matched {os.path.basename(path)} at "
              f"{min_match_rate:.0%} of {len(sample)} sampled lines, so it is read as plain text "
              f"(m_message only). Name a loader explicitly if it should be something else.")
        detection = Detection(RawLoader, {}, format="text", rate=0.0, lines=len(sample),
                              note="no format matched")

    detection.replacement_chars = replacements
    if replacements:
        # A file that did not decode and a file whose format was guessed wrong look identical
        # downstream, so the two are told apart here rather than left to be confused later.
        print(f"AutoLoader: {os.path.basename(path)} has {replacements} undecodable character(s) "
              f"in its first {len(sample)} lines. It was read as {detection.format}, but check the "
              f"file's encoding before trusting that.")
    return detection


class AutoLoader(BaseLoader):
    def __init__(self, filename, filename_pattern=None, min_file_size=0,
                 strip_full_data_path=None, min_match_rate=0.5, sample_lines=_SAMPLE_LINES,
                 system=None):
        self.filename_pattern = filename_pattern
        self.min_file_size = min_file_size
        self.strip_full_data_prefix = strip_full_data_path
        self.min_match_rate = min_match_rate
        self.sample_lines = sample_lines
        self.system = system
        # What was chosen for each path, for detections() and for the error messages.
        self._detections = {}
        # A RawLoader fallback has no clock, and that is a legitimate outcome rather than a defect,
        # so only the message is required. Set per instance, as JsonLoader does.
        self._mandatory_columns = ["m_message"]
        super().__init__(os.path.expanduser(filename))

    # Loading ---------------------------------------------------------------------------------

    def load(self):
        dataset = detect_dataset(self.filename, self.system)
        if dataset is not None:
            self._detections = {self.filename: dataset}
            self._load_dataset(dataset)
            return

        paths = self._collect_paths()
        for path in paths:
            self._detections[path] = detect_format(path, self.min_match_rate, self.sample_lines)
        self._report()

        detections = [self._detections[path] for path in paths]
        first = detections[0]
        if len({d.key() for d in detections}) == 1 and issubclass(first.loader, _TREE_CAPABLE):
            self._load_one(first)
        else:
            self._load_per_file(paths)
        self._normalize_timestamp()

    def _normalize_timestamp(self):
        """Force m_timestamp to naive microseconds, whatever the chosen loader produced.

        Polars takes the time unit from the format string, so a '%3f' pattern yields
        Datetime('ms') while every other LogLead loader yields Datetime('us') - and the two will
        not pl.concat, which is the same trap timezone-awareness sets. A loader whose output
        depends on which format happened to be detected would export that trap to its caller.
        """
        if self.df is None or "m_timestamp" not in self.df.columns:
            return
        dtype = self.df.schema["m_timestamp"]
        if not isinstance(dtype, pl.Datetime):
            return
        column = pl.col("m_timestamp")
        if dtype.time_zone:
            column = column.dt.replace_time_zone(None)
        if dtype.time_unit != "us" or dtype.time_zone:
            self.df = self.df.with_columns(column.cast(pl.Datetime("us")).alias("m_timestamp"))

    def _collect_paths(self):
        if not self.filename_pattern:
            if not os.path.isfile(self.filename):
                raise ValueError(
                    f"{os.path.abspath(self.filename)} is not a file, and is not a dataset "
                    f"AutoLoader recognizes. Pass filename_pattern to load a tree of logs.")
            return [self.filename]

        paths = []
        for subdir, _, _ in os.walk(self.filename):
            for path in sorted(glob.glob(os.path.join(subdir, self.filename_pattern))):
                # No other loader checks this, but a pattern of "*" happily matches directories and
                # hands them to the CSV reader.
                if not os.path.isfile(path) or os.path.getsize(path) <= self.min_file_size:
                    continue
                paths.append(path)
        if not paths:
            raise ValueError(f"No files matching pattern {self.filename_pattern} "
                             f"in directory {os.path.abspath(self.filename)}. "
                             f"Check the pattern and min_file_size.")
        return paths

    def _load_dataset(self, detection):
        """A public dataset: one loader reads the whole directory, labels and all."""
        kwargs = dict(detection.kwargs)
        # A few dataset loaders want a path other than the directory itself.
        filename = kwargs.pop("_root", None) or kwargs.pop("_file", None) or \
            kwargs.pop("_glob", None) or self.filename
        child = detection.loader(filename=filename, **kwargs)
        print(f"AutoLoader: {self.filename} is the {detection.format} dataset, "
              f"loading it with {detection.loader.__name__}.")
        child.load()
        child.preprocess()
        self.df, self.df_seq = child.df, child.df_seq

    def _load_one(self, detection):
        """Every file agreed, so the chosen loader reads the tree itself and keeps its parallel
        multi-file read rather than being rebuilt once per file.

        Nulls are kept in the kwargs rather than filtered out: file_pattern=None is how a detected
        spec's own file_pattern is overridden, and dropping it would make the loader refuse every
        file whose name the spec did not anticipate.
        """
        kwargs = {k: v for k, v in detection.kwargs.items() if not k.startswith("_")}
        child = detection.loader(filename=self.filename,
                                 filename_pattern=self.filename_pattern,
                                 min_file_size=self.min_file_size,
                                 strip_full_data_path=self.strip_full_data_prefix, **kwargs)
        child.load()
        child.preprocess()
        self.df, self.df_seq = child.df, child.df_seq

    def _load_per_file(self, paths):
        """A genuinely mixed folder: one loader per file, then stack what they produced."""
        frames = []
        for path in paths:
            detection = self._detections[path]
            kwargs = {k: v for k, v in detection.kwargs.items() if not k.startswith("_")}
            child = detection.loader(filename=path, **kwargs)
            try:
                child.load()
                child.preprocess()
            except Exception as error:
                raise ValueError(f"Reading {path} as {detection.format} "
                                 f"({detection.loader.__name__}) failed: {error}. Detection said "
                                 f"{detection.rate:.1%} of {detection.lines} sampled lines "
                                 f"matched.") from error
            if child.df_seq is not None:
                print(f"AutoLoader: {os.path.basename(path)} produced a sequence-level frame, "
                      f"which is dropped when files of different formats are merged. Load it on "
                      f"its own to keep df_seq.")
            name = path
            if self.strip_full_data_prefix:
                name = name[len(self.strip_full_data_prefix):] \
                    if name.startswith(self.strip_full_data_prefix) else name
            frames.append(child.df.with_columns(pl.lit(name).alias("file_name")))

        if len(frames) == 1:
            self.df = frames[0]
            return
        try:
            # Files read under different formats do not have the same columns; that is the point of
            # detecting per file, so the concat has to be diagonal.
            self.df = pl.concat(frames, how="diagonal_relaxed")
        except Exception as error:
            formats = ", ".join(sorted({d.format for d in self._detections.values()}))
            raise ValueError(
                f"Could not merge the {len(frames)} files of this folder into one frame: {error}. "
                f"They were read as {formats}, and two of those give one column incompatible "
                f"types - a list against text is the usual case. Load the formats separately, or "
                f"narrow filename_pattern.") from error

    def _report(self):
        counts = {}
        for detection in self._detections.values():
            counts[detection.format] = counts.get(detection.format, 0) + 1
        listed = ", ".join(f"{name} ({n})" for name, n in
                           sorted(counts.items(), key=lambda item: -item[1]))
        print(f"AutoLoader: detected {len(counts)} format(s) across "
              f"{len(self._detections)} file(s): {listed}.")

    def detections(self):
        """What was chosen for each file, and how strong the evidence was."""
        rows = [{"path": path, "loader": d.loader.__name__, "format": d.format,
                 "rate": d.rate, "sampled_lines": d.lines,
                 "replacement_chars": d.replacement_chars, "note": d.note}
                for path, d in self._detections.items()]
        return pl.DataFrame(rows) if rows else pl.DataFrame(
            schema={"path": pl.String, "loader": pl.String, "format": pl.String,
                    "rate": pl.Float64, "sampled_lines": pl.Int64,
                    "replacement_chars": pl.Int64, "note": pl.String})

    def preprocess(self):
        # Every child ran its own preprocess() as it was loaded; there is nothing format-specific
        # left to do, which is the whole point of delegating rather than reimplementing.
        pass

    # Checks ----------------------------------------------------------------------------------

    def check_for_nulls_and_non_utf8(self):
        """
        Same reasoning as the overrides in JsonLoader, LogfmtLoader and SyslogLoader: BaseLoader
        prints a four-line warning per column that has nulls, and a frame merged from several
        formats is null wherever one format has a column another does not. That is the expected
        shape here rather than a defect, so summarize it instead - and keep the non-UTF-8 warning,
        which is a real problem.
        """
        sparse = [(c, n) for c, n in zip(self.df.columns, self.df.null_count().row(0)) if n]
        if sparse:
            worst = sorted(sparse, key=lambda item: -item[1])[:3]
            listed = ", ".join(f"{c} ({n})" for c, n in worst)
            print(f"AutoLoader: {len(sparse)} of {self.df.width} columns contain nulls out of "
                  f"{len(self.df)} rows - expected, as different formats carry different fields. "
                  f"Most null: {listed}.")
        for column in self.df.columns:
            if self.df.schema[column] == pl.Utf8:
                count = self.df.filter(pl.col(column).str.contains("�")).height
                if count:
                    print(f"WARNING! Column '{column}' has {count} non-UTF-8 encoded values out of "
                          f"{len(self.df)}. See detections() for which files did not decode.")
