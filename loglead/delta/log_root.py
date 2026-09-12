"""Loading and slicing a log root.

A *log root* is a directory holding **log folders**. A log folder is any set
of logs that belong together -- one test run, one day, one deployment, "last
release" -- so the term stays accurate whichever of those you happen to have.
A subdirectory of the log root is one log folder, and can hold several log
files; a log file sitting directly in the log root, with no subdirectory, is
a log folder of its own. A log root can hold both kinds at once. Files are
matched *by name across log folders*, which is what makes the comparison
possible.

The three object levels are nested: **log folder -> log file -> log line**.

Ported from LogDelta's ``logdelta/log_analysis_functions.py`` and
``logdelta/data_specific_preprocessing.py``. Unlike the originals these
functions have no module-level state, never ``os.chdir``, and never write
files.
"""

import fnmatch
import glob
import os
import re

import polars as pl

from ..enhancers import EventLogEnhancer
from ..loaders import (DEFAULT_MAX_DETECT_FILES, AccessLogLoader, AutoLoader, DelimitedLoader,
                       JsonLoader, LogfmtLoader, RawLoader, SyslogLoader, detect_format,
                       name_shape)

CONTENT_FORMATS = ("Words", "3grams", "Sklearn", "File")

#: Non-parser content formats plus the ``Parse-<Algorithm>`` family, which is
#: resolved dynamically against :class:`EventLogEnhancer`.
_PARSE_PREFIX = "Parse-"

#TODO Lot of stuff about the line comparer in logroot. Clearn up needed
#: The ``Prefix-<k>`` family: the first k words of a line, as one string. Built
#: by ``EventLogEnhancer.words(prefixes=[k])`` in the same pass as ``e_words``.
_PREFIX_PREFIX = "Prefix-"


def _prefix_width(content_format):
    """Parse the ``k`` out of ``Prefix-<k>``."""
    raw = content_format.split("-", 1)[1]
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError(
            f"Prefix content format needs a positive word count, got {content_format!r}"
        )
    return int(raw)


#: The ``Minhash-<tokenizer>`` family: one band of minhash values over a line's
#: tokens, as one string. Built by ``EventLogEnhancer.minhash()``. Accepts
#: optional ``-r<rows>`` and ``-s<seed>`` suffixes, e.g. ``Minhash-3gram-r6-s7``.
_MINHASH_PREFIX = "Minhash-"

#: Applied to a bare ``Minhash-<tokenizer>``. Fixed rather than random so a
#: signature means the same thing across runs and across cached sessions.
_MINHASH_DEFAULT_ROWS = 4
_MINHASH_DEFAULT_SEED = 0


def _minhash_config(content_format):
    """Parse ``Minhash-<tokenizer>[-r<rows>][-s<seed>]`` into its three parts."""
    parts = content_format.split("-")[1:]
    tokenizer = parts[0].lower() if parts else ""
    if tokenizer not in EventLogEnhancer.MINHASH_TOKENIZERS:
        raise ValueError(
            f"Unknown minhash tokenizer {tokenizer!r} in {content_format!r}. Valid "
            f"options: {', '.join(sorted(EventLogEnhancer.MINHASH_TOKENIZERS))}."
        )
    rows, seed = _MINHASH_DEFAULT_ROWS, _MINHASH_DEFAULT_SEED
    for part in parts[1:]:
        key, value = part[:1].lower(), part[1:]
        if key == "r" and value.isdigit() and int(value) >= 1:
            rows = int(value)
        elif key == "s" and value.isdigit():
            seed = int(value)
        else:
            raise ValueError(
                f"Unrecognized part {part!r} in {content_format!r}. Expected "
                f"Minhash-<tokenizer>[-r<rows>][-s<seed>], e.g. 'Minhash-3gram-r4-s0'."
            )
    return tokenizer, rows, seed


def _minhash_column(tokenizer, rows, seed):
    return f"e_minhash_{tokenizer}_r{rows}_s{seed}"


def minhash_formats(columns):
    """``Minhash-`` format names for every minhash column in ``columns``.

    The row count and seed live only in the column name, so this is how a
    reopened session recovers which signatures a cached frame is carrying.
    """
    return sorted(_MINHASH_PREFIX + "-".join(column.split("_")[2:])
                  for column in columns if column.startswith("e_minhash_"))


#: How a log root may be read, keyed by name. ``"auto"`` detects the format per
#: file; every other entry pins one format family for the whole log root.
#:
#: Name-keyed rather than "pass a loader class", for the same reason
#: :func:`masking.get_pattern` and :data:`FILE_NAME_NORMALIZERS` are: the value
#: arrives from a caller who is frequently a language model driving the MCP
#: server, and an allowlist is the only thing that makes that safe to accept.
LOG_ROOT_FORMATS = {
    "auto": AutoLoader,
    "raw": RawLoader,
    "json": JsonLoader,
    "access_log": AccessLogLoader,
    "delimited": DelimitedLoader,
    "logfmt": LogfmtLoader,
    "syslog": SyslogLoader,
}

#: Columns every analysis in this package needs, whichever loader produced the frame.
REQUIRED_COLUMNS = ("m_message", "file_name")


def available_formats():
    """Every accepted :func:`read_log_root` format name, sub-formats included."""
    names = []
    for family, loader in LOG_ROOT_FORMATS.items():
        names.append(family)
        if loader is SyslogLoader:
            names += [f"{family}/{rfc}" for rfc in ("rfc3164", "rfc5424")]
        elif hasattr(loader, "available_formats"):
            names += [f"{family}/{spec}" for spec in loader.available_formats()]
            if loader is DelimitedLoader:
                names += [f"{family}/{style}" for style in ("row", "w3c")]
    return sorted(names)


def resolve_format(format="auto"):
    """Turn a log root format name into ``(loader class, keyword arguments)``.

    Accepts a family name from :data:`LOG_ROOT_FORMATS`, or ``"family/sub"``
    naming one of that family's shipped format specs (``"json/nginx_json"``,
    ``"delimited/zeek"``), one of ``DelimitedLoader``'s header styles
    (``"delimited/w3c"``), or one syslog RFC (``"syslog/rfc5424"``).

    The vocabulary is deliberately :class:`AutoLoader`'s own: what its
    ``detections()`` reports as the format of a file is what you pass back here
    to pin that choice for every file in the log root.
    """
    name = str(format or "auto")
    family, _, sub = name.partition("/")
    loader = LOG_ROOT_FORMATS.get(family)
    if loader is None:
        raise ValueError(
            f"Unknown log root format {name!r}. Valid formats: "
            f"{', '.join(available_formats())}."
        )
    if not sub:
        return loader, {}

    if loader is AutoLoader:
        raise ValueError(
            f"{name!r} is not a format: 'auto' means detect per file, so it takes no sub-format. "
            f"Name the family you want to pin instead, e.g. 'json/nginx_json'."
        )
    if loader is RawLoader:
        raise ValueError(f"{name!r} is not a format: 'raw' reads any text file as one event per "
                         f"line and has nothing to select.")
    if loader is LogfmtLoader:
        raise ValueError(f"{name!r} is not a format: logfmt lines carry their own key names and "
                         f"those names are conventional, so there are no logfmt specs to choose "
                         f"between. Use 'logfmt'.")
    if loader is SyslogLoader:
        if sub not in ("auto", "rfc3164", "rfc5424"):
            raise ValueError(f"Unknown syslog format {sub!r}. Valid: auto, rfc3164, rfc5424 - or "
                             f"just 'syslog', which decides per file.")
        return loader, {"format": sub}

    specs = loader.available_formats()
    if sub in specs:
        # file_pattern is how a spec states which files it applies to, and the caller has already
        # chosen its files with filename_pattern. Left in, a spec would silently refuse every file
        # whose name it did not anticipate - the same override AutoLoader applies to a detected spec.
        return loader, {"format": sub, "file_pattern": None}
    # A Zeek or W3C file names its own columns, so DelimitedLoader can read one without a spec.
    # Checked after the specs because 'zeek' is both, and the spec is the richer of the two - it
    # adds the type casts and null markers on top of the header style.
    if loader is DelimitedLoader and sub in ("row", "w3c"):
        return loader, {"header": sub}
    valid = [option for option in available_formats() if option.startswith(f"{family}/")]
    raise ValueError(f"Unknown {family} format {sub!r}. "
                     f"Valid: {', '.join(valid) or '(none installed)'} - or just {family!r}.")


def read_log_root(root, filename_pattern="*.log", min_file_size=0, format="auto",
                  max_detect_files=DEFAULT_MAX_DETECT_FILES):
    """Load every matching log file under ``root`` into one event-level frame.

    :param root: the log root. Each subdirectory of it becomes a log folder; a
        log file sitting directly in ``root``, with no subdirectory, becomes a
        log folder of its own.
    :param filename_pattern: glob applied within each log folder.
    :param min_file_size: skip files of this size or smaller (bytes).
    :param format: how to read the files -- a name from :func:`available_formats`.
        ``"auto"`` detects per file, so a log root of JSON, syslog or CSV logs
        arrives parsed into columns instead of as one blob per line.
    :param max_detect_files: with ``format="auto"``, how many files to probe
        before applying their answer to the rest; 0 probes every file. A log
        root split one file per unit holds thousands of them and detection is a
        read each, so the default samples -- see :class:`AutoLoader`. Ignored
        when a format is pinned, since then nothing is detected at all.
    :returns: ``(df, info)``. ``df`` always has ``m_message``, ``file_name``
        (relative to its log folder), ``orig_file_name`` and ``folder``; every
        other column depends on what the chosen loader could read. ``info``
        reports the format asked for, what detection actually chose, how many
        files it looked at to choose it, and the counts, so a wrong guess is
        visible rather than silently analyzed.

    Rows with null messages or a U+FFFD replacement character are dropped, so
    ``df.height`` can be lower than the raw line count; ``info["dropped_rows"]``
    says how many.
    """
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Log root not found: {root}")
    root_posix = root.replace(os.sep, "/")
    loader_class, kwargs = resolve_format(format)
    if loader_class is AutoLoader:
        # Stage 1 of detection recognizes a public dataset from the label file beside the logs and
        # hands the whole directory to that dataset's own loader, which returns a frame shaped for
        # labels and sequences with no file_name column at all -- nothing this package can compare
        # log folders by. A log root is a set of log folders, not a dataset, so only the per-file
        # probe applies. Hadoop forces the issue rather than merely suggesting it: LogDelta's demo
        # log root keeps Hadoop's own abnormal_label.txt next to the application_* directories.
        kwargs["dataset_probe"] = False
        kwargs["max_detect_files"] = max_detect_files

    loader = loader_class(
        root,
        filename_pattern=filename_pattern,
        min_file_size=min_file_size,
        strip_full_data_path=root_posix,
        **kwargs,
    )
    df = loader.execute()

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Reading {root} as {format!r} produced no {', '.join(missing)} column, which every "
            f"log folder comparison needs. Columns present: {', '.join(df.columns[:15])}. "
            f"file_name comes from filename_pattern, so pass one; m_message means the format was "
            f"read but nothing in it was the message."
        )

    raw_height = df.height
    df = df.filter(pl.col("m_message").is_not_null())  # lose lines with nulls
    df = df.filter(~pl.col("m_message").str.contains("�"))  # lose non-utf8 lines

    df = df.with_columns([
        # Where the line came from on disk. Only RawLoader keeps this itself, and only when handed
        # a prefix to strip, so it is rebuilt here from the prefix every loader was given.
        pl.concat_str([pl.lit(root_posix), pl.col("file_name")]).alias("orig_file_name"),
        # First path segment is the log folder
        pl.col("file_name").str.extract(r"^/([^/]+)", 1).alias("folder"),
        # The rest stays as the file name relative to its log folder
        pl.col("file_name").str.replace(r"^/[^/]+/", "", literal=False).alias("file_name"),
    ])
    if df.select(pl.col("folder").is_null().any()).item():
        raise ValueError(
            f"Some rows of {root} have no log folder, meaning their file_name is not the path "
            f"below the log root that stripping {root!r} should have left. This is a loader "
            f"reporting file names differently, not bad data."
        )

    detected = {}
    probed_files = None
    if isinstance(loader, AutoLoader):
        table = loader.detections()
        if table.height:
            detected = dict(
                table.group_by("format").len().sort("len", descending=True).iter_rows()
            )
            # How each file was *read* is the count above; how many of them were looked at is this,
            # and the two differ whenever max_detect_files bit. A caller weighing whether a format
            # is right needs to know the answer came from a sample.
            probed_files = int(table["probed"].sum())
    info = {
        "format": str(format),
        "detected_formats": detected,
        "probed_files": probed_files,
        "n_folders": df.select("folder").n_unique(),
        "n_files": df.select("orig_file_name").n_unique(),
        "n_rows": df.height,
        "dropped_rows": raw_height - df.height,
    }
    return df, info


def read_folders(root, filename_pattern="*.log", min_file_size=0, format="auto",
                 max_detect_files=DEFAULT_MAX_DETECT_FILES):
    """:func:`read_log_root` returning only ``(df, n_folders)``.

    The two-value shape ``loglead.delta`` has always exported. Use
    :func:`read_log_root` when the format actually chosen matters.
    """
    df, info = read_log_root(root, filename_pattern, min_file_size, format, max_detect_files)
    return df, info["n_folders"]


def count_log_root_files(root, filename_pattern="*.log", min_file_size=0):
    """Fingerprint a log root on disk without reading any file contents.

    :returns: ``(n_files, total_bytes, max_mtime)``. Used as a cheap cache key
        so changed logs invalidate a stale parquet.
    """
    root = os.path.abspath(os.path.expanduser(root))
    n_files = 0
    total_bytes = 0
    max_mtime = 0.0
    for subdir, _, _ in os.walk(root):
        for path in glob.glob(os.path.join(subdir, filename_pattern)):
            try:
                stat = os.stat(path)
            except OSError:
                continue
            if stat.st_size <= min_file_size:
                continue
            n_files += 1
            total_bytes += stat.st_size
            max_mtime = max(max_mtime, stat.st_mtime)
    return n_files, total_bytes, max_mtime


# --------------------------------------------------------------------------- #
# Looking before loading
# --------------------------------------------------------------------------- #

#: Seek points and lines per point used to estimate a file's line count. Eight
#: points measured +1.3% against BGL's true 4,747,963 lines, where reading the
#: head alone is +5.8% (its opening lines are shorter than its average) and 32
#: shorter runs are -9.3%: each seek lands inside one locally-repetitive burst,
#: so a few longer reads spread widely beat many short ones.
_ESTIMATE_POINTS = 8
_ESTIMATE_LINES_PER_POINT = 125


def _estimate_lines(path, size):
    """Line count from a sample, without reading the file.

    Returned as an estimate and labelled one. A caller deciding whether a log
    root is worth opening needs the order of magnitude, not the exact number --
    and the exact number costs a full pass.
    """
    if size == 0:
        return 0
    sampled_bytes = sampled_lines = 0
    with open(path, "rb") as handle:
        for point in range(_ESTIMATE_POINTS):
            if point:
                handle.seek(size * point // _ESTIMATE_POINTS)
                handle.readline()  # discard the partial line the seek landed in
            for _ in range(_ESTIMATE_LINES_PER_POINT):
                line = handle.readline()
                if not line:
                    break
                sampled_bytes += len(line)
                sampled_lines += 1
    if not sampled_bytes:
        return 0
    return round(size / (sampled_bytes / sampled_lines))


def _head_lines(path, limit):
    """The first ``limit`` lines, decoded leniently. What the log actually says."""
    lines = []
    with open(path, "rb") as handle:
        for _ in range(limit):
            line = handle.readline()
            if not line:
                break
            lines.append(line.decode("utf-8", errors="replace").rstrip("\r\n"))
    return lines


def _probe_file(path, sample_lines):
    """Format, match rate and a few real lines for one file. Nothing is loaded."""
    size = os.path.getsize(path)
    entry = {
        "file": os.path.basename(path),
        # Which group of files this one was probed on behalf of. See file_names.
        "name_shape": name_shape(path),
        "bytes": size,
        "estimated_lines": _estimate_lines(path, size),
        "sample": _head_lines(path, sample_lines),
    }
    try:
        detection = detect_format(path)
        entry["format"] = detection.format
        entry["match_rate"] = round(detection.rate, 3)
    except Exception as error:  # a probe must never be the thing that fails
        entry["format"] = None
        entry["error"] = str(error)
    return entry


def _scale_estimate(probed, total_bytes):
    """Scale the probed files' bytes-per-line up to the whole log root."""
    sampled_bytes = sum(entry["bytes"] for entry in probed)
    sampled_lines = sum(entry["estimated_lines"] for entry in probed)
    if not sampled_bytes or not sampled_lines:
        return 0
    return round(total_bytes / (sampled_bytes / sampled_lines))


#: Files stat'ed before a peek stops counting and says so. A peek has to be
#: cheap enough to use for orientation, and an accurate count costs one stat per
#: file: ~/Datasets holds 765,416 of them, which is 17s. Stopping at this many
#: keeps every peek under a second, and the exact counts are read_log_root's job.
_PEEK_FILE_BUDGET = 20_000


def peek_log_root(path, filename_pattern="*.log", probe_files=5, sample_lines=5,
                  max_children=50, max_file_names=20, max_files=_PEEK_FILE_BUDGET):
    """Report what is at ``path`` without loading any of it.

    :func:`read_log_root` is the expensive call in this package: it reads and
    parses every file. This one stats them and reads a few hundred lines from a
    handful, so a caller can find out what it is pointing at -- how many log
    folders, how big, in what format, and what the lines actually say -- before
    paying for it. It is also how a directory of candidate log roots gets
    surveyed: point it at the parent and every child is listed with its size.

    :param path: a directory, or a single log file.
    :param filename_pattern: glob deciding which files count, as in
        :func:`read_log_root`.
    :param probe_files: how many files to detect the format of and sample. The
        largest file of each distinct file-name shape is taken first, so a log
        root holding two kinds of file gets one probe of each rather than five
        of whichever is biggest.
    :param sample_lines: log lines returned per probed file.
    :param max_children: subdirectories listed.
    :param max_file_names: file-name shapes listed in ``file_names``.
    :param max_files: stop stat'ing after this many files and report
        ``truncated``. See :data:`_PEEK_FILE_BUDGET`.
    :returns: a dict with ``kind`` (``"file"``, ``"log_root"``, ``"parent"`` or
        ``"empty"``), the counts and sizes, ``children``, ``file_names``,
        ``probed``, and ``notes`` saying what to do next.

    ``file_names`` groups the files by name with their digits collapsed
    (:func:`loglead.loaders.name_shape`), which is what says whether a log root
    is one kind of file or several: files of different formats nearly always
    have differently shaped names, and it is the same grouping
    :func:`read_log_root`'s format sampling spreads itself across. A log root of
    one shape can be opened on the default sample; several shapes, and it is
    worth probing every file or pinning a format.

    ``estimated_lines`` is sampled, not counted -- see :data:`_ESTIMATE_POINTS`.
    """
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nothing at {path}.")

    if os.path.isfile(path):
        # A single file is not a log root, and saying so is the whole point:
        # every analysis here compares log folders, and one file is one folder.
        return {
            "path": path,
            "kind": "file",
            "n_folders": 1,
            "n_files": 1,
            "total_bytes": os.path.getsize(path),
            "truncated": False,
            "children": [],
            "n_children": 0,
            "n_distinct_file_names": 1,
            "file_names": [{"name_shape": name_shape(path), "n_files": 1,
                            "total_bytes": os.path.getsize(path),
                            "example": os.path.basename(path)}],
            "probed": [_probe_file(path, sample_lines)],
            "notes": [
                f"{os.path.basename(path)} is a single log file, so there is nothing to compare "
                f"it against. Split it into slices first, then open the directory of slices as "
                f"the log root."
            ],
        }

    # One walk, stat'ing each matching file once. Every number below comes from
    # this pass; walking again per child is what made an early version 17s on a
    # directory of datasets.
    n_files = 0
    total_bytes = 0
    min_mtime = max_mtime = None
    folders = set()
    per_child = {}
    per_shape = {}
    distinct_names = set()
    biggest = []
    truncated = False
    # Which top-level children the walk actually entered, and the one it was
    # inside when the budget ran out. Without these, a child the walk never
    # reached is indistinguishable from an empty one.
    visited = []
    stopped_in = None
    for subdir, dirnames, filenames in os.walk(path):
        dirnames.sort()
        if subdir != path:
            top = os.path.relpath(subdir, path).split(os.sep)[0]
            if top not in visited:
                visited.append(top)
        for name in sorted(filenames):
            if not fnmatch.fnmatch(name, filename_pattern):
                continue
            if n_files >= max_files:
                truncated = True
                stopped_in = visited[-1] if visited else None
                break
            match = os.path.join(subdir, name)
            try:
                stat = os.stat(match)
            except OSError:
                continue
            n_files += 1
            total_bytes += stat.st_size
            min_mtime = stat.st_mtime if min_mtime is None else min(min_mtime, stat.st_mtime)
            max_mtime = stat.st_mtime if max_mtime is None else max(max_mtime, stat.st_mtime)
            # Same rule read_log_root applies: the first path segment is the log
            # folder, and a file directly in the root is its own log folder.
            top = os.path.relpath(match, path).split(os.sep)[0]
            folders.add(top)
            child = per_child.setdefault(top, {"n_files": 0, "total_bytes": 0})
            child["n_files"] += 1
            child["total_bytes"] += stat.st_size
            # What the files are *called*, grouped the way format detection samples them. One
            # shape means one kind of file; several mean this log root may hold several formats,
            # and that is the thing a caller cannot see from a count of files.
            shape = per_shape.setdefault(name_shape(name), {
                "n_files": 0, "total_bytes": 0, "example": name, "largest": (-1, None)})
            shape["n_files"] += 1
            shape["total_bytes"] += stat.st_size
            distinct_names.add(name)
            if stat.st_size > shape["largest"][0]:
                shape["largest"] = (stat.st_size, match)
            # Biggest first: a probe says most about a file with something in
            # it, and the large files are what opening this root would cost.
            biggest.append((stat.st_size, match))
            if len(biggest) > max(probe_files * 20, 200):
                biggest.sort(reverse=True)
                del biggest[probe_files:]
        if truncated:
            break

    # Subdirectories that matched nothing are still worth listing: on a survey
    # the pattern is exactly what tends to be wrong.
    children = []
    seen = set(visited)
    for entry in sorted(os.listdir(path))[:max_children]:
        if not os.path.isdir(os.path.join(path, entry)):
            continue
        if truncated and entry not in seen:
            # Never looked at, which is not the same as holding nothing.
            children.append({"name": entry, "n_files": None, "total_bytes": None,
                             "status": "not_counted"})
            continue
        counts = per_child.get(entry, {"n_files": 0, "total_bytes": 0})
        status = "partial" if entry == stopped_in else "counted"
        children.append({"name": entry, **counts, "status": status})

    # The largest file of each name shape, biggest shape group first, then the largest files left
    # over. A log root of one shape therefore gets the largest files, as it always did; one holding
    # a stray 'stderr.json' among 900 container logs gets that file probed rather than a fifth
    # container log, which is the case a probe of the five largest could never see.
    shapes_by_size = sorted(per_shape.values(), key=lambda group: -group["n_files"])
    chosen = [group["largest"][1] for group in shapes_by_size[:probe_files]]
    biggest.sort(reverse=True)
    for _, match in biggest:
        if len(chosen) >= probe_files:
            break
        if match not in chosen:
            chosen.append(match)
    probed = [_probe_file(match, sample_lines)
              for match in sorted(chosen, key=lambda m: -os.path.getsize(m))]

    file_names = [
        {"name_shape": shape, "n_files": group["n_files"], "total_bytes": group["total_bytes"],
         "example": group["example"]}
        for shape, group in sorted(per_shape.items(), key=lambda item: -item[1]["n_files"])
    ]

    kind = "log_root" if n_files else ("parent" if children else "empty")
    notes = []
    if truncated:
        where = f" inside {stopped_in!r}" if stopped_in else ""
        notes.append(
            f"More than {max_files:,} files here, so counting stopped{where} and the totals "
            f"below are partial. Children marked status='not_counted' were never reached -- that "
            f"is not the same as holding nothing. This is a big tree; peek at one subdirectory "
            f"instead."
        )
    if kind == "parent":
        notes.append(
            f"No files matching {filename_pattern!r} here, but {len(children)} subdirector"
            f"{'y' if len(children) == 1 else 'ies'} that may each be a log root. Peek at one "
            f"of them, or pass a filename_pattern matching the files these hold."
        )
    elif kind == "empty":
        notes.append(f"{path} holds nothing matching {filename_pattern!r} and no subdirectories.")
    else:
        if len(folders) < 2:
            only = next(iter(folders), path)
            notes.append(
                f"Only one log folder ({only}), so there is nothing to compare it against. "
                f"Every analysis judges a log folder against the others. If this is one big log "
                f"file, split it into slices first."
            )
        formats = {entry.get("format") for entry in probed}
        if len(formats) > 1:
            notes.append(
                f"The {len(probed)} probed files were detected as "
                f"{sorted(str(f) for f in formats)}. Files this different are usually separate "
                f"log roots sharing a parent directory rather than one log root -- peek at a "
                f"subdirectory to check. If it really is one, it is read one loader per file, "
                f"which is slower; pass format= to pin one."
            )
        if n_files > DEFAULT_MAX_DETECT_FILES:
            # The sample open_log_root takes is spread over the name shapes, so whether it can
            # speak for the whole log root is decided by how many shapes there are, not by how
            # many files: one shape, or fewer shapes than the sample size, and every kind of file
            # here gets looked at. More, and some kind of file will not be.
            shapes = len(per_shape)
            if shapes == 1:
                detail = (f"every file here is named like {file_names[0]['example']}, so one "
                          f"answer for all of them is a safe bet")
            elif shapes <= DEFAULT_MAX_DETECT_FILES:
                detail = (f"the sample is spread over all {shapes} file-name shapes here (see "
                          f"file_names), so each kind of file is looked at")
            else:
                detail = (f"there are {shapes} file-name shapes here (see file_names) and the "
                          f"sample reaches at most {DEFAULT_MAX_DETECT_FILES} of them, so a "
                          f"format used only by one of the rest would be missed")
            notes.append(
                f"open_log_root detects the format from {DEFAULT_MAX_DETECT_FILES} of these "
                f"{n_files:,} files and reads the rest the same way -- {detail}. Pass "
                f"max_detect_files=0 to detect every file, or format= to skip detection."
            )
        if "text" in formats:
            notes.append(
                "At least one file matched no known format and would be read as plain text, one "
                "event per line. Pass format= explicitly if it is structured."
            )

    return {
        "path": path,
        "kind": kind,
        "filename_pattern": filename_pattern,
        "n_folders": len(folders),
        "n_files": n_files,
        "total_bytes": total_bytes,
        # Scaled from the probed files' mean line length to the whole log root,
        # so the number is there whether 5 files were probed or all of them.
        "estimated_lines": _scale_estimate(probed, total_bytes),
        "truncated": truncated,
        "oldest_mtime": min_mtime,
        "newest_mtime": max_mtime,
        "children": children,
        "n_children": len(children),
        # Both counts, because they answer different questions: how many distinct names there are
        # says whether file-level analyses have anything to match across log folders, while the
        # shapes say whether one format can be assumed for all of them.
        "n_distinct_file_names": len(distinct_names),
        "file_names": file_names[:max_file_names],
        "probed": probed,
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# File-name normalization
# --------------------------------------------------------------------------- #

def strip_folder_id_from_file_names(df):
    """Remove the log folder's numeric id from every file name, so names match.

    Hadoop container logs embed the application id in the file name, e.g. in log
    folder ``application_1445062781478_0011`` the file
    ``container_1445062781478_0011_01_000001.log`` becomes
    ``container__01_000001.log``. Without this, file-content and line-content
    analyses (``distance_file_content``, ``anomaly_file_content``,
    ``distance_line_content``, ``anomaly_line_content``) find zero matching
    files between log folders.

    Ported from LogDelta's ``remove_run_name_from_file_names``.
    """
    df = df.with_columns(
        # Everything before the first digit is the folder's prefix; the rest is
        # the id shared with the file names. "My_folder_123_2" -> "123_2".
        pl.col("folder").str.replace_all(r"^[^\d]+", "").alias("_common_part")
    )
    for part in df["_common_part"].unique():
        if not part:
            continue
        df = df.with_columns(pl.col("file_name").str.replace(part, "", literal=True))
    return df.drop("_common_part")


def replace_in_file_names(df, pattern, replacement=""):
    """Regex-replace inside ``file_name``. Safe: uses Polars, never ``eval``."""
    return df.with_columns(pl.col("file_name").str.replace_all(pattern, replacement))


#: Allowlist of file-name normalizers selectable by name from untrusted input.
FILE_NAME_NORMALIZERS = {
    "none": lambda df: df,
    "strip_folder_id": strip_folder_id_from_file_names,
}


def normalize_file_names(df, normalizer="none"):
    """Apply an allowlisted file-name normalizer by name."""
    try:
        fn = FILE_NAME_NORMALIZERS[normalizer]
    except KeyError:
        raise ValueError(
            f"Unknown file_name_normalizer {normalizer!r}. "
            f"Valid options: {sorted(FILE_NAME_NORMALIZERS)}"
        ) from None
    return fn(df)


# --------------------------------------------------------------------------- #
# Log folder naming
# --------------------------------------------------------------------------- #

def validate_folder_names(names):
    """Check a log folder name mapping, returning it as a plain ``{str: str}`` dict.

    Raises :class:`ValueError` rather than silently dropping entries, because a
    typo in a folder name would otherwise leave that log folder unnamed and the
    mistake invisible until someone reads a plot legend.
    """
    if not names:
        return {}
    if not isinstance(names, dict):
        raise ValueError(
            f"folder_names must be a mapping of log folder to new name, "
            f"got {type(names).__name__}."
        )
    clean = {}
    for folder, name in names.items():
        if not isinstance(folder, str) or not isinstance(name, str):
            raise ValueError(
                f"folder_names entries must be strings, got {folder!r}: {name!r}."
            )
        name = name.strip()
        if not name:
            raise ValueError(f"Empty name for log folder {folder!r}.")
        # Folder names reach output file names (see export.build_file_name), and
        # a newline would corrupt every table and legend it lands in.
        bad = [ch for ch in "/\\\n\r\t" if ch in name]
        if bad:
            raise ValueError(
                f"Name {name!r} for log folder {folder!r} contains {bad!r}, "
                "which cannot appear in a log folder name."
            )
        clean[folder] = name
    return clean


def apply_folder_names(df, names, keep_original=True):
    """Give log folders meaningful names, in place of their directory names.

    A log folder is named after the directory it was read from, and that name is
    what every plot legend, result row and output file is labelled with -- so a
    tree of ``application_1445062781478_0012`` directories is hard to follow.
    The new name can be anything useful: a ground-truth label such as
    ``PageRank_MachineDown``, or simply something descriptive like
    ``FailingRunThu``. Supplying the mapping is the caller's job; datasets record
    this kind of thing in wildly different ways, if at all.

    Nothing on disk is renamed -- only the ``folder`` column changes. The
    directory name is kept in ``folder_original``, so results stay traceable to
    where they came from, and because names are always derived from the original,
    calling this again replaces the previous mapping instead of stacking onto it.

    :param names: ``{directory name: new name}``. A log folder missing from the
        mapping goes back to its directory name, and an empty mapping clears
        every name -- naming is always derived from the directory name, never
        from a name already applied.
    :param keep_original: append the directory name, giving
        ``PageRank_MachineDown_application_1445062781478_0012``. Keeps log
        folders traceable, and lets a multi-part name line up with
        ``group_by_indices`` (``PageRank_MachineDown`` occupies positions 0 and
        1) and with wildcards like ``"PageRank_Normal*"``. Pass ``False`` when
        the directory name is just noise and the log folder should simply *be*
        the name given.
    :returns: ``(df, info)`` where ``info`` reports what was applied.
    """
    names = validate_folder_names(names)
    if not names:
        if "folder_original" in df.columns:
            df = df.with_columns(pl.col("folder_original").alias("folder"))
        return df, {"named": 0, "unnamed": sorted(_unique_folders(df))}

    # Naming always starts from the original folder name, never from a name
    # already applied, so a second mapping replaces the first.
    source = "folder_original" if "folder_original" in df.columns else "folder"
    known = set(_unique_folders(df, column=source))

    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(
            f"folder_names names {len(unknown)} log folder(s) that are not in the log root: "
            f"{unknown[:5]}{' ...' if len(unknown) > 5 else ''}. "
            f"{len(known)} log folders available, e.g. {sorted(known)[:3]}."
        )

    # Every folder selector filters on the folder column, so two log folders
    # would be indistinguishable -- and silently so. Only reachable when the
    # original name is dropped; prefixing keeps names unique by construction.
    if not keep_original:
        resulting = [names.get(folder, folder) for folder in sorted(known)]
        duplicates = sorted({n for n in resulting if resulting.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"folder_names with keep_original=False would give {len(duplicates)} "
                f"name(s) to more than one log folder: {duplicates[:5]}"
                f"{' ...' if len(duplicates) > 5 else ''}. "
                "Log folder names must stay unique; make them distinct or keep the "
                "original name as a suffix."
            )

    df = df.with_columns(pl.col(source).alias("folder_original"))
    named = pl.col("folder_original").replace_strict(names, default=None).alias("_name")
    df = df.with_columns(named)
    combined = (
        pl.concat_str([pl.col("_name"), pl.col("folder_original")], separator="_")
        if keep_original
        else pl.col("_name")
    )
    df = df.with_columns(
        pl.when(pl.col("_name").is_null())
        .then(pl.col("folder_original"))
        .otherwise(combined)
        .alias("folder")
    ).drop("_name")

    return df, {
        "named": len(names),
        "unnamed": sorted(known - set(names)),
    }


def _unique_folders(df, column="folder"):
    return df.select(column).unique().to_series().to_list()


# --------------------------------------------------------------------------- #
# Selecting log folders and files
# --------------------------------------------------------------------------- #

def _match_wildcard(candidates, pattern):
    regex = re.compile(pattern.replace(".", r"\.").replace("*", ".*"))
    return [c for c in candidates if regex.match(c)]


def prepare_folders(df, target_folder, comparison_folders="ALL"):
    """Split ``df`` into the target log folder's rows and a validated comparison list.

    :param target_folder: exact log folder name. Must exist.
    :param comparison_folders: ``"ALL"``, a list of names, an int N (first N), or a
        ``"Prefix*"`` wildcard.
    :returns: ``(target_df, comparison_folder_names)``. The target log folder is always
        excluded from the comparison list, whichever form was used.
    :raises ValueError: on an unknown folder name or an out-of-range count.
    """
    unique_folders = df.select("folder").unique().sort("folder").to_series().to_list()

    if target_folder not in unique_folders:
        raise ValueError(
            f"Target log folder {target_folder!r} not found in the log_root. "
            f"{len(unique_folders)} log folders available, e.g. {unique_folders[:3]}"
        )

    target_df = df.filter(pl.col("folder") == target_folder)
    others = [folder for folder in unique_folders if folder != target_folder]

    if isinstance(comparison_folders, str) and comparison_folders == "ALL":
        validated = others
    elif isinstance(comparison_folders, bool):
        raise ValueError(f"Invalid comparison_folders: {comparison_folders!r}")
    elif isinstance(comparison_folders, int):
        if comparison_folders < 1 or comparison_folders > len(others):
            raise ValueError(
                f"Number of comparison log folders must be between 1 and {len(others)}."
            )
        validated = others[:comparison_folders]
    elif isinstance(comparison_folders, str) and "*" in comparison_folders:
        # Full-match wildcard: escape everything, then re-open the '*'
        regex = re.compile("^" + re.escape(comparison_folders).replace(r"\*", ".*") + "$")
        validated = [folder for folder in others if regex.match(folder)]
        if not validated:
            raise ValueError(f"No log folders match the wildcard pattern {comparison_folders!r}.")
    else:
        if isinstance(comparison_folders, str):
            comparison_folders = [comparison_folders]
        validated = [folder for folder in comparison_folders if folder != target_folder]
        invalid = [folder for folder in validated if folder not in unique_folders]
        if invalid:
            raise ValueError(f"Comparison log folder names {invalid} not found in the log_root.")

    return target_df, validated


def resolve_target_folders(df, target_folders):
    """Expand ``target_folder`` into a list. Accepts ``"ALL"``, int N, wildcard, name, list."""
    unique_folders = df.select("folder").unique().sort("folder").to_series().to_list()

    if isinstance(target_folders, str) and target_folders == "ALL":
        return unique_folders
    if isinstance(target_folders, bool):
        raise ValueError(f"Invalid target_folder: {target_folders!r}")
    if isinstance(target_folders, int):
        if target_folders < 1 or target_folders > len(unique_folders):
            raise ValueError(
                f"Number of target log folders must be between 1 and {len(unique_folders)}."
            )
        return unique_folders[:target_folders]
    if isinstance(target_folders, str) and "*" in target_folders:
        matched = _match_wildcard(unique_folders, target_folders)
        if not matched:
            raise ValueError(f"No log folders matched the pattern {target_folders!r}.")
        return matched
    if isinstance(target_folders, str):
        target_folders = [target_folders]
    invalid = [folder for folder in target_folders if folder not in unique_folders]
    if invalid:
        raise ValueError(f"Target log folder names {invalid} not found in the log_root.")
    return list(target_folders)


def prepare_files(target_df, files="ALL"):
    """Resolve ``target_files`` against the files present in the target log folder.

    Accepts ``"ALL"``, a list of names, an int N, or a ``"pattern*"`` wildcard.
    Names in a supplied list that are absent are dropped with a warning rather
    than raising, matching LogDelta.
    """
    available = target_df.select("file_name").unique().sort("file_name").to_series().to_list()

    if isinstance(files, list):
        missing = [f for f in files if f not in available]
        if missing:
            print(f"Warning: files not present in the target log folder, skipping: {missing}")
        files = [f for f in files if f in available]
        if not files:
            raise ValueError("No valid files found in the provided list for processing.")
        return files
    if isinstance(files, str) and files == "ALL":
        return available
    if isinstance(files, bool):
        raise ValueError(f"Invalid target_files: {files!r}")
    if isinstance(files, int):
        if files < 1 or files > len(available):
            raise ValueError(f"Number of files must be between 1 and {len(available)}.")
        return available[:files]
    if isinstance(files, str) and "*" in files:
        matched = _match_wildcard(available, files)
        if not matched:
            raise ValueError(f"No files matched the pattern: {files}")
        return matched
    if isinstance(files, str):
        if files not in available:
            raise ValueError(f"File {files!r} not present in the target log folder.")
        return [files]
    raise ValueError(
        f"Invalid type for 'target_files': {files!r}. "
        "Must be 'ALL', a list, an integer, or a wildcard pattern."
    )


# --------------------------------------------------------------------------- #
# Content representation
# --------------------------------------------------------------------------- #

def content_column(mask, content_format):
    """Name of the column ``prepare_content`` will produce, without computing it."""
    if content_format == "Words":
        return "e_words"
    if content_format == "3grams":
        return "e_trigrams"
    if content_format == "File":
        return "file_name"
    if content_format == "Sklearn":
        return "e_message_normalized" if mask else "m_message"
    if content_format.startswith(_PREFIX_PREFIX):
        return f"e_words_prefix_{_prefix_width(content_format)}"
    if content_format.startswith(_MINHASH_PREFIX):
        return _minhash_column(*_minhash_config(content_format))
    if content_format.startswith(_PARSE_PREFIX):
        return f"e_event_{content_format.split('-', 1)[1].lower()}_id"
    raise ValueError(
        f"Unrecognized content format: {content_format}. "
        f"Valid options: {', '.join(CONTENT_FORMATS)}, Prefix-<k>, "
        f"Minhash-<tokenizer>, Parse-<Algorithm>"
    )


def derived_columns(content_format):
    """Columns an enhancer creates for ``content_format``, so they can be dropped.

    ``Sklearn`` and ``File`` derive nothing -- they read a column that already
    exists -- so they return an empty list.
    """
    if content_format == "Words":
        return ["e_words", "e_words_len"]
    if content_format == "3grams":
        return ["e_trigrams", "e_trigrams_len"]
    if content_format in ("Sklearn", "File"):
        return []
    if content_format.startswith(_PREFIX_PREFIX):
        # words() emits e_words/e_words_len alongside the prefix, in one pass.
        return [f"e_words_prefix_{_prefix_width(content_format)}", "e_words", "e_words_len"]
    if content_format.startswith(_MINHASH_PREFIX):
        # minhash() builds the token column it reads, so that comes too.
        tokenizer, rows, seed = _minhash_config(content_format)
        token_column = EventLogEnhancer.MINHASH_TOKENIZERS[tokenizer]
        return [_minhash_column(tokenizer, rows, seed), token_column, f"{token_column}_len"]
    if content_format.startswith(_PARSE_PREFIX):
        parser = content_format.split("-", 1)[1].lower()
        return [
            f"e_event_{parser}_id",
            f"e_event_{parser}_template",
            f"e_template_{parser}",
        ]
    raise ValueError(f"Unrecognized content format: {content_format}")


def prepare_content(df, mask, content_format):
    """Ensure the column for ``content_format`` exists, computing it if needed.

    :param mask: read from ``e_message_normalized`` when True, ``m_message``
        otherwise. Masked input requires ``normalize()`` to have run already.
    :returns: ``(df, field)`` — the frame *including* any newly added column,
        and the name of the column to analyze.

    ``EventLogEnhancer`` short-circuits when its output column already exists,
    so repeat calls are cheap. Callers should keep the returned frame: that is
    what makes an interactive session avoid re-parsing.
    """
    field = "e_message_normalized" if mask else "m_message"
    if mask and field not in df.columns:
        raise ValueError(
            "mask=True requires the logs to have been normalized "
            "(no 'e_message_normalized' column). Open the log root with mask=True."
        )

    enhancer = EventLogEnhancer(df)
    if content_format == "Words":
        return enhancer.words(field), "e_words"
    if content_format == "3grams":
        return enhancer.trigrams(field), "e_trigrams"
    if content_format == "File":
        return df, "file_name"
    if content_format == "Sklearn":
        return df, field
    if content_format.startswith(_PREFIX_PREFIX):
        width = _prefix_width(content_format)
        return enhancer.words(field, prefixes=[width]), f"e_words_prefix_{width}"
    if content_format.startswith(_MINHASH_PREFIX):
        tokenizer, rows, seed = _minhash_config(content_format)
        return (enhancer.minhash(field, tokenizer=tokenizer, rows=rows, seed=seed),
                _minhash_column(tokenizer, rows, seed))
    if content_format.startswith(_PARSE_PREFIX):
        parse_type = content_format.split("-", 1)[1].lower()
        method_name = f"parse_{parse_type}"
        method = getattr(enhancer, method_name, None)
        if method is None:
            raise ValueError(
                f"No parse method found for {parse_type!r} "
                f"(EventLogEnhancer has no {method_name}())"
            )
        # parse_pliplom and parse_lenma consume e_words regardless of `field`
        if parse_type in ("pliplom", "lenma") and "e_words" not in df.columns:
            enhancer.words(field)
        return method(field), f"e_event_{parse_type}_id"
    raise ValueError(
        f"Unrecognized content format: {content_format}. "
        f"Valid options: {', '.join(CONTENT_FORMATS)}, Prefix-<k>, "
        f"Minhash-<tokenizer>, Parse-<Algorithm>"
    )


def aggregate_dataframe(df, group_by_col, field):
    """Collapse ``df`` to one row per ``group_by_col``, gathering ``field`` into a list.

    Handles both ``Utf8`` columns (parser ids, raw messages) and ``List[Utf8]``
    columns (words, trigrams), which get exploded first so the result is a flat
    list of tokens rather than a list of lists.
    """
    dtype = df.schema[field]
    if dtype == pl.List(pl.Utf8):
        return (
            df.select(group_by_col, field)
            .explode(field)
            .group_by(group_by_col)
            .agg(pl.col(field))
        )
    if dtype == pl.Utf8:
        # Projected first for the same reason as the List branch above: an eager
        # group_by drags every other column of the frame through the grouping.
        return (df.select(group_by_col, field)
                .group_by(group_by_col).agg(pl.col(field).alias(field)))
    raise ValueError(
        f"Unsupported datatype {dtype} in field {field}. Supported: Utf8, List[Utf8]"
    )


def create_vectorizer(vectorizer_type):
    """Map ``"Count"``/``"Tfidf"`` to the sklearn vectorizer *class*.

    A class, not an instance: ``LogDistance`` and ``AnomalyDetector`` both
    instantiate it themselves.

    sklearn is imported here rather than at module level: it costs ~150MB and
    ~1.1s, and this module is also the home of :func:`peek_log_root`, which
    vectorizes nothing. A peek-only process now never loads it.
    """
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    if vectorizer_type == "Count":
        return CountVectorizer
    if vectorizer_type == "Tfidf":
        return TfidfVectorizer
    raise ValueError(
        f"Unsupported vectorizer type: {vectorizer_type}. Valid options: Count, Tfidf"
    )


def group_folders_by_indices(df, group_by_indices):
    """Add a ``group`` column built from selected underscore-separated parts of ``folder``.

    ``group_by_indices=[0, 1]`` turns log folder ``PageRank_DiskFull_application_123``
    into group ``PageRank_DiskFull``. Used to colour plots.
    """
    if not group_by_indices:
        return df.with_columns(pl.lit("all").alias("group"))

    parts = df.select(pl.col("folder").str.split("_").alias("_parts"))
    selected = [
        parts.select(
            pl.col("_parts").list.get(i, null_on_oob=True).fill_null("").alias(f"_part_{i}")
        ).to_series()
        for i in group_by_indices
    ]
    group = pl.DataFrame(selected).select(
        pl.concat_str(pl.col("*"), separator="_").alias("group")
    )
    return df.with_columns(group.to_series())
