"""Loading and slicing a "folder of runs" log corpus.

A *corpus* is a directory whose immediate subdirectories are **runs** (one
execution, deployment, or test run) and whose leaves are log files. Files are
matched *by name across runs*, which is what makes run-vs-run comparison
possible.

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
from ..loaders import RawLoader

CONTENT_FORMATS = ("Words", "3grams", "Sklearn", "File")

#: Non-parser content formats plus the ``Parse-<Algorithm>`` family, which is
#: resolved dynamically against :class:`EventLogEnhancer`.
_PARSE_PREFIX = "Parse-"


def read_folders(folder, filename_pattern="*.log", min_file_size=0):
    """Load every matching log file under ``folder`` into one event-level frame.

    :param folder: corpus root. Its immediate subdirectories become runs.
    :param filename_pattern: glob applied within each subdirectory.
    :param min_file_size: skip files of this size or smaller (bytes).
    :returns: ``(df, n_runs)`` where ``df`` has columns ``m_message``,
        ``file_name`` (run-relative), ``orig_file_name`` and ``run``.

    Rows with null messages or a U+FFFD replacement character are dropped, so
    ``df.height`` can be lower than the raw line count.
    """
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Corpus folder not found: {folder}")

    loader = RawLoader(
        folder,
        filename_pattern=filename_pattern,
        min_file_size=min_file_size,
        strip_full_data_path=folder,
    )
    df = loader.execute()
    df = df.filter(pl.col("m_message").is_not_null())  # lose lines with nulls
    df = df.filter(~pl.col("m_message").str.contains("�"))  # lose non-utf8 lines

    df = df.with_columns([
        # First path segment is the run
        pl.col("file_name").str.extract(r"^/([^/]+)", 1).alias("run"),
        # The rest stays as the run-relative file name
        pl.col("file_name").str.replace(r"^/[^/]+/", "", literal=False).alias("file_name"),
    ])
    n_runs = df.select("run").n_unique()
    return df, n_runs


def count_corpus_files(folder, filename_pattern="*.log", min_file_size=0):
    """Fingerprint a corpus on disk without reading any file contents.

    :returns: ``(n_files, total_bytes, max_mtime)``. Used as a cheap cache key
        so a changed corpus invalidates a stale parquet.
    """
    folder = os.path.abspath(os.path.expanduser(folder))
    n_files = 0
    total_bytes = 0
    max_mtime = 0.0
    for subdir, _, _ in os.walk(folder):
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

def strip_run_id_from_file_names(df):
    """Remove the run's numeric id from every file name so names match across runs.

    Hadoop container logs embed the application id in the file name, e.g. under
    run ``application_1445062781478_0011`` the file
    ``container_1445062781478_0011_01_000001.log`` becomes
    ``container__01_000001.log``. Without this, file-level (L3) and line-level
    (L4) analyses find zero matching files between runs.

    Ported from LogDelta's ``remove_run_name_from_file_names``.
    """
    df = df.with_columns(
        # Everything before the first digit is the run's prefix; the rest is the
        # id shared with the file names. "My_run_123_2" -> "123_2".
        pl.col("run").str.replace_all(r"^[^\d]+", "").alias("_common_part")
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
    "strip_run_id": strip_run_id_from_file_names,
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
# Selecting runs and files
# --------------------------------------------------------------------------- #

def _match_wildcard(candidates, pattern):
    regex = re.compile(pattern.replace(".", r"\.").replace("*", ".*"))
    return [c for c in candidates if regex.match(c)]


def prepare_runs(df, target_run, comparison_runs="ALL"):
    """Split ``df`` into the target run's rows and a validated comparison list.

    :param target_run: exact run name. Must exist.
    :param comparison_runs: ``"ALL"``, a list of names, an int N (first N), or a
        ``"Prefix*"`` wildcard.
    :returns: ``(target_df, comparison_run_names)``. The target run is always
        excluded from the comparison list, whichever form was used.
    :raises ValueError: on an unknown run name or an out-of-range count.
    """
    unique_runs = df.select("run").unique().sort("run").to_series().to_list()

    if target_run not in unique_runs:
        raise ValueError(
            f"Target run {target_run!r} not found in the corpus. "
            f"{len(unique_runs)} runs available, e.g. {unique_runs[:3]}"
        )

    target_df = df.filter(pl.col("run") == target_run)
    others = [run for run in unique_runs if run != target_run]

    if isinstance(comparison_runs, str) and comparison_runs == "ALL":
        validated = others
    elif isinstance(comparison_runs, bool):
        raise ValueError(f"Invalid comparison_runs: {comparison_runs!r}")
    elif isinstance(comparison_runs, int):
        if comparison_runs < 1 or comparison_runs > len(others):
            raise ValueError(
                f"Number of comparison runs must be between 1 and {len(others)}."
            )
        validated = others[:comparison_runs]
    elif isinstance(comparison_runs, str) and "*" in comparison_runs:
        # Full-match wildcard: escape everything, then re-open the '*'
        regex = re.compile("^" + re.escape(comparison_runs).replace(r"\*", ".*") + "$")
        validated = [run for run in others if regex.match(run)]
        if not validated:
            raise ValueError(f"No runs match the wildcard pattern {comparison_runs!r}.")
    else:
        if isinstance(comparison_runs, str):
            comparison_runs = [comparison_runs]
        validated = [run for run in comparison_runs if run != target_run]
        invalid = [run for run in validated if run not in unique_runs]
        if invalid:
            raise ValueError(f"Comparison run names {invalid} not found in the corpus.")

    return target_df, validated


def resolve_target_runs(df, target_runs):
    """Expand ``target_run`` into a list. Accepts ``"ALL"``, int N, wildcard, name, list."""
    unique_runs = df.select("run").unique().sort("run").to_series().to_list()

    if isinstance(target_runs, str) and target_runs == "ALL":
        return unique_runs
    if isinstance(target_runs, bool):
        raise ValueError(f"Invalid target_run: {target_runs!r}")
    if isinstance(target_runs, int):
        if target_runs < 1 or target_runs > len(unique_runs):
            raise ValueError(
                f"Number of target runs must be between 1 and {len(unique_runs)}."
            )
        return unique_runs[:target_runs]
    if isinstance(target_runs, str) and "*" in target_runs:
        matched = _match_wildcard(unique_runs, target_runs)
        if not matched:
            raise ValueError(f"No runs matched the pattern {target_runs!r}.")
        return matched
    if isinstance(target_runs, str):
        target_runs = [target_runs]
    invalid = [run for run in target_runs if run not in unique_runs]
    if invalid:
        raise ValueError(f"Target run names {invalid} not found in the corpus.")
    return list(target_runs)


def prepare_files(target_df, files="ALL"):
    """Resolve ``target_files`` against the files present in the target run.

    Accepts ``"ALL"``, a list of names, an int N, or a ``"pattern*"`` wildcard.
    Names in a supplied list that are absent are dropped with a warning rather
    than raising, matching LogDelta.
    """
    available = target_df.select("file_name").unique().sort("file_name").to_series().to_list()

    if isinstance(files, list):
        missing = [f for f in files if f not in available]
        if missing:
            print(f"Warning: files not present in the target run, skipping: {missing}")
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
            raise ValueError(f"File {files!r} not present in the target run.")
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
            "mask=True requires the corpus to have been normalized "
            "(no 'e_message_normalized' column). Open the corpus with mask=True."
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


def group_runs_by_indices(df, group_by_indices):
    """Add a ``group`` column built from selected underscore-separated parts of ``run``.

    ``group_by_indices=[0, 1]`` turns run ``PageRank_DiskFull_application_123``
    into group ``PageRank_DiskFull``. Used to colour plots.
    """
    if not group_by_indices:
        return df.with_columns(pl.lit("all").alias("group"))

    parts = df.select(pl.col("run").str.split("_").alias("_parts"))
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
