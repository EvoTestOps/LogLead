"""Loading and slicing a log root.

A *log root* is a directory whose immediate subdirectories are **log folders**
and whose leaves are log files. A log folder is any set of logs that belong together
-- one test run, one day, one deployment, "last release" -- so the term stays
accurate whichever of those you happen to have. Files are matched *by name
across log folders*, which is what makes the comparison possible.

The three object levels are nested: **log folder -> log file -> log line**.

Ported from LogDelta's ``logdelta/log_analysis_functions.py`` and
``logdelta/data_specific_preprocessing.py``. Unlike the originals these
functions have no module-level state, never ``os.chdir``, and never write
files.
"""

import glob
import os
import re

import polars as pl
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ..enhancers import EventLogEnhancer
from ..loaders import (AccessLogLoader, AutoLoader, DelimitedLoader, JsonLoader, LogfmtLoader,
                       RawLoader, SyslogLoader)

CONTENT_FORMATS = ("Words", "3grams", "Sklearn", "File")

#: Non-parser content formats plus the ``Parse-<Algorithm>`` family, which is
#: resolved dynamically against :class:`EventLogEnhancer`.
_PARSE_PREFIX = "Parse-"

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


def read_log_root(root, filename_pattern="*.log", min_file_size=0, format="auto"):
    """Load every matching log file under ``root`` into one event-level frame.

    :param root: the log root. Its immediate subdirectories become log folders.
    :param filename_pattern: glob applied within each subdirectory.
    :param min_file_size: skip files of this size or smaller (bytes).
    :param format: how to read the files -- a name from :func:`available_formats`.
        ``"auto"`` detects per file, so a log root of JSON, syslog or CSV logs
        arrives parsed into columns instead of as one blob per line.
    :returns: ``(df, info)``. ``df`` always has ``m_message``, ``file_name``
        (relative to its log folder), ``orig_file_name`` and ``folder``; every
        other column depends on what the chosen loader could read. ``info``
        reports the format asked for, what detection actually chose, and the
        counts, so a wrong guess is visible rather than silently analyzed.

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
    if isinstance(loader, AutoLoader):
        table = loader.detections()
        if table.height:
            detected = dict(
                table.group_by("format").len().sort("len", descending=True).iter_rows()
            )
    info = {
        "format": str(format),
        "detected_formats": detected,
        "n_folders": df.select("folder").n_unique(),
        "n_files": df.select("orig_file_name").n_unique(),
        "n_rows": df.height,
        "dropped_rows": raw_height - df.height,
    }
    return df, info


def read_folders(root, filename_pattern="*.log", min_file_size=0, format="auto"):
    """:func:`read_log_root` returning only ``(df, n_folders)``.

    The two-value shape ``loglead.delta`` has always exported. Use
    :func:`read_log_root` when the format actually chosen matters.
    """
    df, info = read_log_root(root, filename_pattern, min_file_size, format)
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
# File-name normalization
# --------------------------------------------------------------------------- #

def strip_folder_id_from_file_names(df):
    """Remove the log folder's numeric id from every file name, so names match.

    Hadoop container logs embed the application id in the file name, e.g. in log
    folder ``application_1445062781478_0011`` the file
    ``container_1445062781478_0011_01_000001.log`` becomes
    ``container__01_000001.log``. Without this, file-level (L3) and line-level
    (L4) analyses find zero matching files between log folders.

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

    :param names: ``{directory name: new name}``. Log folders missing from the
        mapping keep their current name.
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
    if content_format.startswith(_PARSE_PREFIX):
        return f"e_event_{content_format.split('-', 1)[1].lower()}_id"
    raise ValueError(
        f"Unrecognized content format: {content_format}. "
        f"Valid options: {', '.join(CONTENT_FORMATS)}, Parse-<Algorithm>"
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
        f"Valid options: {', '.join(CONTENT_FORMATS)}, Parse-<Algorithm>"
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
        return df.group_by(group_by_col).agg(pl.col(field).alias(field))
    raise ValueError(
        f"Unsupported datatype {dtype} in field {field}. Supported: Utf8, List[Utf8]"
    )


def create_vectorizer(vectorizer_type):
    """Map ``"Count"``/``"Tfidf"`` to the sklearn vectorizer *class*.

    A class, not an instance: ``LogDistance`` and ``AnomalyDetector`` both
    instantiate it themselves.
    """
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
