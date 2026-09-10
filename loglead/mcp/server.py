#CLAUDE DO NOT TOUCH OR EDIT THESE TODO comments. 
 
# TODO: One should be able to supply own mask patterns also in openlog_root
# We also want away to for MCP client to inspect a sample of log lines
# max diversity of log lines to sample for mask pattern detection. 
# Also saving a mask is needed as it can be expensive to figure out
# a good mask and we do want to repeat

#TODO file splitting should support even splits (DONE)
#Timestamp splits NOT DONE
#Splits by block_ID as in HDFS and other custom splits. NOT DONE.abs
#The last two require reading in the the file



"""MCP server exposing LogLead's log folder comparison analyses.

Wraps :mod:`loglead.delta` in a session model so a log root is loaded, masked,
and parsed once and then interrogated repeatedly. The tool mimics
LogDelta's logic.

Run it with ``loglead-mcp`` (stdio, what MCP clients expect) or
``loglead-mcp --transport http --port 8000``.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Union

import polars as pl
import yaml

try:  # MCP SDK 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # MCP SDK 1.x, where the same class was called FastMCP
    from mcp.server.fastmcp import FastMCP as _Server

from ..delta import anomaly, distance, export, log_root, scoring, split, visualize
from ..loaders import DEFAULT_MAX_DETECT_FILES
from . import formatting
from .session import SessionStore

mcp = _Server("loglead", instructions="""\
Compares log folders (test runs, deployments, nodes -- any set of logs that
belong together) to find which one looks wrong, with no labels required.
Start with peek_log_root to see what is on disk without loading it -- it also
says when a path is one big log file rather than a set of log folders, which
split_log_file turns into slices you can compare. Then open_log_root, and drill
down: File names -> Whole log
text of all logs -> One log file text across folders -> Individual lines. distance_* pairs
performs pairwise distance measurement; anomaly_* trains on given set and 
scores on another (automatically avoids using train data in test) and ranks 
many folders/files/lines by anomaly score; plot_* draws them. Every result
keeps its full table server-side under a result_id -- use query_result to
filter or page through it instead of re-running the analysis. You can also
inspect raw log files directly: search_log_lines finds lines by regex or
substring with line numbers, and query_result filters/pages any result
table.

COST. These tools span milliseconds to hours. Every result reports its own
elapsed_seconds: make one narrow call.""")

#: Set by main(); tests and demos construct their own.
STORE = SessionStore()

#: A log folder selector: an exact name, "ALL", an int N, a "Prefix*" wildcard, or a list.
FolderSelector = Union[str, int, Sequence[str]]

#: A file selector: same forms, resolved against the target log folder's files.
FileSelector = Union[str, int, Sequence[str]]

#: Which figures a plot tool should build: any subset of ``visualize.PLOTS``.
PlotSelector = Sequence[str]


def tool(fn):
    """Register a function as an MCP tool, keep stdout clean, and time it.

    LogLead and the libraries it uses print a lot of messages while running --
    warnings, status notes, and so on. Normally that's fine, but the stdio
    connection to the MCP client also uses stdout to send its own messages, so
    any of these extra prints would corrupt that connection. This wrapper sends
    them to stderr instead, where they're harmless.

    It also puts ``elapsed_seconds`` on every result.

    The wrapper is what gets returned, so direct Python callers -- the demo,
    :func:`run_config`, the tests -- get the same behaviour an MCP client does.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            result = fn(*args, **kwargs)
        if isinstance(result, dict):
            result.setdefault("elapsed_seconds", round(time.perf_counter() - started, 2))
        return result

    mcp.tool()(wrapper)
    return wrapper


def _write(session, df, analysis, level, **name_parts):
    """Persist a full result table and return its path."""
    stem = export.build_file_name(analysis=analysis, level=level, **name_parts)
    return export.write_table(df, str(session.output_dir), stem, session.table_format)


# --------------------------------------------------------------------------- #
# Log root / session
# --------------------------------------------------------------------------- #

@tool
def peek_log_root(
    path: str,
    filename_pattern: str = "*.log",
    probe_files: int = 5,
    sample_lines: int = 5,
    max_children: int = 50,
    max_file_names: int = 20,
) -> dict:
    """Look at a directory without loading it. Call this before open_log_root.

    Reports how many log folders and files are there, how big they are, what
    they are called, what format they look like, and what a few of the actual
    log lines say -- by stat'ing the files and reading a few hundred lines,
    never by parsing them. open_log_root reads everything and can take minutes;
    this takes under a second and tells you whether it is worth it.

    `file_names` groups the files by name with the digits collapsed, e.g.
    978 files named `container_#_#_#_#.log`. One shape means one kind of file,
    and open_log_root's default format sampling can speak for all of them;
    several shapes mean the log root may hold several formats, so check what
    `probed` says about each and consider max_detect_files=0 or a pinned
    format.

    Point it at a directory holding several datasets and it lists each of them,
    so you can see what is available before choosing one. Point it at a single
    log file and it says so: one file is one log folder, and every analysis here
    compares log folders against each other, so a single file has to be cut into
    slices first with split_log_file.

    Read `notes` in the result -- it says what is wrong or what to do next.

    Args:
        path: A directory, or a single log file.
        filename_pattern: Glob deciding which files count. The same pattern you
            would pass to open_log_root, so a peek reporting zero files is
            telling you that open_log_root would find none either.
        probe_files: How many files to detect the format of and sample lines
            from. The largest file of each distinct file-name shape is taken
            first, so two kinds of file get one probe each.
        sample_lines: Raw log lines returned per probed file.
        max_children: Subdirectories listed.
        max_file_names: File-name shapes listed in `file_names`.
    """
    return log_root.peek_log_root(
        path,
        filename_pattern=filename_pattern,
        probe_files=probe_files,
        sample_lines=sample_lines,
        max_children=max_children,
        max_file_names=max_file_names,
    )


def _split_out_dir(path, n_slices, by):
    """Where a split goes when the caller does not say.

    Keyed on the source file's fingerprint and the split parameters, so asking
    for the same split twice reuses the slices instead of rewriting them -- the
    same bargain SessionStore's parquet cache makes, and worth more here, since
    the slices are a second copy of the log on disk.
    """
    stat = os.stat(path)
    payload = "|".join([
        os.path.abspath(path), str(stat.st_size), f"{stat.st_mtime:.0f}", str(n_slices), by,
    ])
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    stem = os.path.splitext(os.path.basename(path))[0]
    return STORE.cache_dir / "splits" / f"{stem}-{digest}"


@tool
def split_log_file(
    path: str,
    n_slices: int = 10,
    by: str = "lines",
    out_dir: Optional[str] = None,
    stem: Optional[str] = None,
    refresh: bool = False,
) -> dict:
    """Cut one big log file into slices, so it can be analysed as a log root.

    Every analysis here compares log folders against each other, so a single log
    file -- one long stream of lines -- has nothing to compare and cannot be
    analysed as it stands. Cutting it into slices gives it something: the slices
    become log folders, and asking which slice looks unlike the others is asking
    whether the log changed part way through.

    The slices are written side by side as `<name>_slice_000.log`,
    `<name>_slice_001.log`, ... in one directory, which is itself a log root --
    pass that directory to open_log_root next. Nothing is read into memory, so
    the file can be far larger than RAM.

    Splitting the same file the same way twice reuses the slices already on
    disk rather than writing them again.

    Because each slice is one file in its own log folder, the folder-level
    tools are the ones to use on a split file: distance_folder_content,
    anomaly_folder_content and plot_folder_content. The file-level and
    line-level tools (distance_file_content, anomaly_file_content,
    distance_line_content, anomaly_line_content) match files by name across log
    folders, and no two slices share a file name, so they find nothing here.

    Args:
        path: The log file to cut. A .gz is decompressed on the way in.
        n_slices: How many slices. More slices means finer resolution on where
            the log changed, and less text in each one to judge it by.
        by: "lines" gives every slice the same number of log lines, which is
            what makes slices comparable; "bytes" gives them the same size on
            disk in a single pass, which is faster but leaves the line counts
            uneven wherever line lengths vary.
        out_dir: Where to write the slices. Defaults to a directory beside the
            session cache, named after the file and the split.
        stem: Name the slices after this instead of the file's own name.
        refresh: Split again even if these slices already exist.
    """
    source = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(source):
        raise FileNotFoundError(
            f"Not a file: {source}. split_log_file cuts up one log file; a directory of logs is "
            f"already a log root, so pass it to open_log_root instead."
        )
    chosen = Path(os.path.expanduser(out_dir)) if out_dir else _split_out_dir(
        source, n_slices, by
    )
    # The manifest is written beside the slices so a reused split reports the
    # same thing a fresh one does, line counts included, without reading them
    # back. It also makes a half-written split visible: slices with no manifest
    # are not treated as a usable result.
    record = chosen / "split_manifest.json"
    reused = record.is_file() and not refresh
    if reused:
        manifest = json.loads(record.read_text())
        # The stored one is how long the original split took; this call did
        # not split anything, and @tool fills in what it actually cost.
        manifest.pop("elapsed_seconds", None)
    else:
        # Clearing the directory first is only safe when we picked it: it is
        # ours, keyed on this exact split. A caller-supplied out_dir that
        # already holds something raises instead, unless refresh says otherwise.
        manifest = split.split_log_file(
            source, str(chosen), n_slices=n_slices, by=by, stem=stem,
            overwrite=out_dir is None or refresh,
        )
        record.write_text(json.dumps(manifest, indent=2))
    manifest["reused_existing_slices"] = reused
    manifest["notes"] = [
        f"These slices are a log root. Open it with "
        f"open_log_root(path={str(chosen)!r}) to analyse them.",
        "Each slice is one file in its own log folder, so the log folders are named after the "
        "files, extension included (e.g. 'BGL_slice_000.log'). set_folder_names can rename them.",
        "Compare the slices with distance_folder_content, anomaly_folder_content or "
        "plot_folder_content. The file-level and line-level tools match files by name across log "
        "folders, and no two slices share a name, so those come back empty.",
    ]
    return manifest


@tool
def open_log_root(
    path: str,
    filename_pattern: str = "*.log",
    format: str = "auto",
    max_detect_files: int = DEFAULT_MAX_DETECT_FILES,
    mask: bool = True,
    mask_pattern: str = "myllari_extended",
    parsers: Optional[Sequence[str]] = None,
    file_name_normalizer: str = "none",
    min_file_size: int = 0,
    output_dir: Optional[str] = None,
    table_format: str = "csv",
    session_id: Optional[str] = None,
    refresh: bool = False,
    folder_names: Optional[dict] = None,
    keep_original_folder_name: bool = True,
) -> dict:
    """Load a log root -- a directory of log folders -- into a session.

    A **log folder** is any set of logs that belong together, be it a test run,
    a day, or a release -- it's the unit that gets compared against the rest.
    If `path` has subdirectories, each one is a log folder, and can hold
    several files (e.g. one folder per test run). If a file sits directly in
    `path` instead, with no subdirectory, that single file is its own log
    folder (e.g. one file per block id, compared file-to-file). A log root can
    have both kinds at once. Files are matched by name across log folders. Do
    this once, then run as many analyses against the returned `session_id` --
    nothing is re-read or re-parsed.

    Args:
        path: The log root directory. Its subdirectories are log files or log folders.
        filename_pattern: Glob applied inside each log folder.
        format: How to read the files. The default, "auto", looks at each file
            and guesses its format. Check the result's `detected_formats`
            field to see what it guessed, since that's the only place a wrong
            guess shows up. To skip guessing, name a format instead: "raw" (plain text,
            one line per log entry), "json", "syslog", "logfmt", "access_log",
            "delimited". For a more exact match, add a known layout after a
            slash, e.g. "json/nginx_json", "delimited/zeek",
            "access_log/combined", "syslog/rfc5424". These are the same names
            `detected_formats` reports, so you can take a guess it made and
            feed it back in to force every file to use it.
        max_detect_files: With format="auto", how many files to look at before
            reading the rest the same way. Detection costs a read per file, so
            a log root of thousands of files -- one per block, one per slice --
            would spend minutes on it; the files probed are spread over the
            distinct file-name shapes, since files of different formats are
            nearly always named differently. peek_log_root's `file_names` says
            how many shapes there are. Pass 0 to detect every file, which is
            worth it when one odd file among thousands would have to be read
            differently. Ignored unless format="auto".
        mask: Replace volatile tokens (ids, IPs, timestamps, hex) with
            placeholders. Almost always wanted.
        mask_pattern: One of "myllari_extended", "myllari", "drain_loglead",
            "drain_orig".
        parsers: Template log parsers to run up front, e.g. tipiing ["tip"] or ["drain"].
        file_name_normalizer: "none", or "strip_folder_id" when file names embed
            the folder id (Hadoop container logs do). Without it, file-level and
            line-level analyses find no files in common between log folders.
        min_file_size: Skip files this size or smaller, in bytes.
        output_dir: Where result tables and plots are written.
        table_format: "csv" (tab-separated, drops list columns) or "xlsx".
        session_id: Choose your own handle instead of a generated one.
        refresh: Ignore any cached parquet and re-read from disk.
        folder_names: Pass new folder names as dict. Often log folders names are ids
            dates which hard for humans to track. This allows renaming them so 
            they make sense: {folder name: meaningful name}, e.g.
            {"application_1445062781478_0012": "PageRank_MachineDown"} or
            {"logs_2024_11_04": "FailingRunThursday"} Where the meaningful names come from is up to
            you -- a ground-truth label file shipped with the dataset, a
            deployment log, or your own knowledge of what each one was. Can also
            be applied later with set_folder_names.
        keep_original_folder_name: append the folder name to the name you gave, so
            log folders stay traceable and multi-part names line up with
            group_by_indices. Pass False to use the given name.
    """
    session, info = STORE.open(
        path=path,
        filename_pattern=filename_pattern,
        format=format,
        max_detect_files=max_detect_files,
        mask=mask,
        mask_pattern=mask_pattern,
        parsers=parsers or (),
        file_name_normalizer=file_name_normalizer,
        min_file_size=min_file_size,
        output_dir=output_dir,
        table_format=table_format,
        session_id=session_id,
        refresh=refresh,
        folder_names=folder_names,
        keep_original_folder_name=keep_original_folder_name,
    )
    summary = session.summary()
    summary.update(info)
    folders = session.folders
    summary["folders"] = folders[:50]

    notes = []
    if len(folders) > 50:
        notes.append(f"{len(folders)} log folders total; first 50 listed. "
                     "Use describe_log_root for the rest.")
    # "text/<format>" is timestamped text and a good outcome; a bare "text" is the fallback that
    # matched nothing, which is the one case worth naming a format by hand for.
    probed = info.get("probed_files")
    n_read = sum(summary.get("detected_formats", {}).values())
    if probed is not None and probed < n_read:
        notes.append(f"The format was detected from {probed} of the {n_read} files and applied to "
                     f"all of them -- they agreed, but files that were not probed could still "
                     f"differ. Re-open with max_detect_files=0 to detect every file, or with "
                     f"format= to pin one.")
    unmatched = summary.get("detected_formats", {}).get("text", 0)
    if unmatched:
        notes.append(f"{unmatched} file(s) matched no known format and were read as plain text, "
                     f"one event per line. Pass format= explicitly if they are structured.")
    if info.get("dropped_rows"):
        notes.append(f"{info['dropped_rows']} row(s) dropped: null message or undecodable "
                     f"characters.")
    if notes:
        summary["notes"] = notes
    return summary


@tool
def list_log_roots() -> dict:
    """List every open session with its size and what has been computed."""
    return {"sessions": STORE.list()}


@tool
def describe_log_root(session_id: str, include_files: bool = False) -> dict:
    """Report the log folders, file counts, and line counts of an open log_root.

    Args:
        session_id: Handle from open_log_root.
        include_files: Also list every distinct file name and how many log folders
            contain it. Useful for picking a `target_files` value.
    """
    session = STORE.get(session_id)
    aggs = [pl.col("file_name").n_unique().alias("n_files"), pl.len().alias("n_lines")]
    columns = ["folder", "file_name"]
    if "folder_original" in session.df.columns:
        # Show what each log folder is called on disk, so a new name can still be
        # traced back to its folder.
        aggs.append(pl.col("folder_original").first().alias("folder_original"))
        columns.append("folder_original")
    
    per_folder = session.df.select(columns).group_by("folder").agg(aggs).sort("folder")
    out = session.summary()
    out["folders_detail"] = per_folder.to_dicts()

    if include_files:
        per_file = (
            session.df.select("file_name", "folder").group_by("file_name")
            .agg([pl.col("folder").n_unique().alias("n_folders"), pl.len().alias("n_lines")])
            .sort("n_folders", descending=True)
        )
        out["files_detail"] = per_file.to_dicts()
        out["notes"] = [
            "Files present in many log folders are the comparable ones; a file "
            "present in only one has nothing to compare against in "
            "distance_file_content, anomaly_file_content, distance_line_content, "
            "or anomaly_line_content, which all pair a file with its namesake in "
            "another log folder."
        ]
    return out


@tool
def set_folder_names(
    session_id: str, folder_names: dict, keep_original_folder_name: bool = True
) -> dict:
    """Give log folders meaningful names in place of their directory names.

    A log folder is named after the directory it was read from, and that is what
    every plot legend, result row and output file is labelled with -- so a tree
    of `application_1445062781478_0012` directories is hard to follow. The new
    name can be anything useful: a ground-truth label like `PageRank_MachineDown`
    if the dataset ships one, or simply something descriptive like
    `FailingRunThu`. Where those names come from is up to you; datasets record
    this kind of thing in wildly different ways, if at all.

    Nothing on disk is renamed. The folder name is kept in a `folder_original`
    column, and because names are always applied to the original, calling this
    again replaces the previous mapping rather than stacking onto it.

    Keeping the folder name as a suffix makes two other things work:
    `group_by_indices=[0, 1]` on the plot tools groups by `PageRank_MachineDown`,
    and `comparison_folders` accepts wildcards like `"PageRank_Normal*"`.

    Nothing is re-read or re-parsed -- every column computed so far is kept.

    Args:
        session_id: Handle from open_log_root.
        folder_names: {directory name: meaningful name}. Log folders left out keep their
            current name. Folder names must match the log root exactly; unknown
            ones are an error.
        keep_original_folder_name: append the folder name, giving
            `PageRank_MachineDown_application_1445062781478_0012`. Pass False to
            use the given name verbatim, e.g. just `FailingRunThu`; the resulting
            names must still be unique.
    """
    session, info = STORE.set_folder_names(session_id, folder_names, keep_original_folder_name)
    folders = session.folders
    return {
        "session_id": session_id,
        "named": info["named"],
        "unnamed": len(info["unnamed"]),
        "folders": folders[:50],
        "notes": (
            [f"{len(info['unnamed'])} log folder(s) kept their directory name, "
             f"e.g. {info['unnamed'][:3]}."] if info["unnamed"] else []
        ),
    }


@tool
def close_log_root(session_id: str) -> dict:
    """Close a session and free its memory. The parquet cache is kept."""
    return {"closed": STORE.close(session_id)}


@tool
def read_log_lines(
    session_id: str,
    folder: str,
    file_name: str,
    offset: int = 0,
    limit: int = 100,
    masked: bool = False,
) -> dict:
    """Read actual log lines. Use this to see the evidence behind a score.

    Args:
        session_id: Handle from open_log_root.
        folder: Log folder name.
        file_name: File name, relative to its log folder.
        offset: First line to return, 0-based.
        limit: How many lines (capped at 500).
        masked: Return the masked text instead of the raw message.
    """
    session = STORE.get(session_id)
    column = "e_message_normalized" if masked else "m_message"
    if column not in session.df.columns:
        raise ValueError(f"Column {column!r} is not available in this session.")

    selected = session.df.filter(
        (pl.col("folder") == folder) & (pl.col("file_name") == file_name)
    ).with_row_index("line_number")
    if selected.height == 0:
        raise ValueError(
            f"No lines for folder={folder!r} file={file_name!r}. "
            "Check describe_log_root for valid names."
        )

    limit = max(1, min(int(limit), 500))
    window = selected.slice(offset, limit).select(["line_number", column])
    return {
        "session_id": session_id,
        "folder": folder,
        "file_name": file_name,
        "total_lines": selected.height,
        "offset": offset,
        "returned": window.height,
        "lines": window.to_dicts(),
    }


@tool
def search_log_lines(
    session_id: str,
    pattern: str,
    folders: Optional[Sequence[str]] = None,
    files: Optional[Sequence[str]] = None,
    regex: bool = True,
    ignore_case: bool = False,
    limit: int = 50,
) -> dict:
    """Find log lines matching a pattern, and count matches per log folder.

    The per-folder counts are often the answer on their own: a message that
    appears only in the suspect log folder, or 50x more often there, explains its
    anomaly score.

    Args:
        session_id: Handle from open_log_root.
        pattern: Regex, or a literal substring when regex is False.
        folders: Restrict to these log folders. Defaults to all.
        files: Restrict to these file names. Defaults to all.
        regex: Treat pattern as a regular expression.
        ignore_case: Case-insensitive matching.
        limit: Maximum matching lines returned (capped at 200).
    """
    session = STORE.get(session_id)
    df = session.df
    if folders:
        df = df.filter(pl.col("folder").is_in(list(folders)))
    if files:
        df = df.filter(pl.col("file_name").is_in(list(files)))

    if regex:
        needle = f"(?i){pattern}" if ignore_case else pattern
        matches = df.filter(pl.col("m_message").str.contains(needle))
    elif ignore_case:
        matches = df.filter(
            pl.col("m_message").str.to_lowercase().str.contains(pattern.lower(), literal=True)
        )
    else:
        matches = df.filter(pl.col("m_message").str.contains(pattern, literal=True))

    per_folder = (
        matches.select("folder").group_by("folder").agg(pl.len().alias("matches"))
        .sort("matches", descending=True)
    )
    limit = max(1, min(int(limit), 200))
    sample = matches.select(["folder", "file_name", "m_message"]).head(limit)

    return {
        "session_id": session_id,
        "pattern": pattern,
        "total_matches": matches.height,
        "folders_with_matches": per_folder.height,
        "matches_per_folder": per_folder.to_dicts(),
        "sample": sample.to_dicts(),
        "truncated": matches.height > sample.height,
    }


# --------------------------------------------------------------------------- #
# Result tables
# --------------------------------------------------------------------------- #

#: What ``where`` clauses may say. Structured triples rather than an expression
#: string, because the clause arrives from a model and nothing here evaluates
#: what it is handed -- the same reason masking patterns resolve by name only.
#: Each lambda takes a Polars column expression ``col`` and a literal ``value``,
#: and returns a boolean Polars expression. ``_where_expr`` below builds ``col``
#: as ``pl.col(column)`` and passes the result to ``DataFrame.filter()``.
_QUERY_OPS = {
    "==": lambda col, value: col == value,
    "!=": lambda col, value: col != value,
    "<": lambda col, value: col < value,
    "<=": lambda col, value: col <= value,
    ">": lambda col, value: col > value,
    ">=": lambda col, value: col >= value,
    "in": lambda col, value: col.is_in(list(value)),
    "not_in": lambda col, value: ~col.is_in(list(value)),
    "contains": lambda col, value: col.cast(pl.Utf8).str.contains(str(value), literal=True),
    "is_null": lambda col, value: col.is_null(),
    "not_null": lambda col, value: col.is_not_null(),
}


def _where_expr(df, clause):
    """Turn one ``[column, op, value]`` triple into a Polars predicate."""
    if not isinstance(clause, (list, tuple)) or len(clause) != 3:
        raise ValueError(
            f"Each where clause is [column, operator, value]; got {clause!r}."
        )
    column, op, value = clause
    if column not in df.columns:
        raise ValueError(
            f"No column {column!r} in this result. Columns: {', '.join(df.columns)}."
        )
    if op not in _QUERY_OPS:
        raise ValueError(
            f"Unknown operator {op!r}. Use one of: {', '.join(_QUERY_OPS)}."
        )
    return _QUERY_OPS[op](pl.col(column), value)


@tool
def query_result(
    session_id: str,
    result_id: str,
    where: Optional[Sequence[Sequence]] = None,
    sort_by: Optional[str] = None,
    descending: bool = True,
    max_rows: int = 25,
    offset: int = 0,
) -> dict:
    """Filter the full table an earlier analysis produced.

    Every analysis returns `result_id`; the table
    itself stays in the session. Using `result_id` ask for
    the rows that answer your question instead of scrolling. 

    Examples:
        query_result(s, rid, where=[["lines", "<", 5]])
            log folders with fewer than 5 lines.
        query_result(s, rid, where=[["folder", "contains", "PageRank"]])
            one family of log folders, whatever their score.
        query_result(s, rid, where=[["rank_sum", ">", 12]], sort_by="rank_sum")
            every row with rank_sum over 12, sorted highest first.

    Args:
        session_id: Handle from open_log_root.
        result_id: From the result of any analysis tool, e.g. "anomaly_folder_content-1a2b3c4d".
            Results live in the server process only, and only the most recent
            few per session; re-run the analysis if the id has aged out.
        where: Clauses as [column, operator, value], combined with AND.
            Operators: ==, !=, <, <=, >, >=, in, not_in, contains (plain
            substring, no regex), is_null, not_null. The value for `in` /
            `not_in` is a list; for is_null / not_null it is ignored but the
            three-part shape stays, e.g. ["duration", "is_null", null].
        sort_by: Column to order by. Defaults to the order the analysis left,
            which for a ranking is already the meaningful one.
        descending: Sort direction.
        max_rows: Rows returned inline.
        offset: Skip this many matching rows first, to page through them.
    """
    session = STORE.get(session_id)
    analysis, df = session.get_result(result_id)

    matched = df
    for clause in where or []:
        matched = matched.filter(_where_expr(df, clause))
    if sort_by:
        if sort_by not in matched.columns:
            raise ValueError(
                f"No column {sort_by!r} in this result. Columns: {', '.join(df.columns)}."
            )
        matched = matched.sort(sort_by, descending=descending, nulls_last=True)

    offset = max(0, int(offset))
    page = matched.slice(offset, max(0, int(max_rows)))
    records = formatting.rows_to_records(page, page.height)

    notes = []
    if matched.height == 0:
        notes.append(
            "Nothing matched. 'summary' below is the whole table, so you can pick a "
            "threshold that does."
        )
    elif offset + len(records) < matched.height:
        notes.append(
            f"Rows {offset + 1}-{offset + len(records)} of {matched.height} matching "
            f"({df.height} in the table). Raise offset for the next page."
        )
    return {
        "session_id": session_id,
        "analysis": "query_result",
        "source_analysis": analysis,
        "result_id": result_id,
        "n_rows_total": df.height,
        "n_rows_matched": matched.height,
        "offset": offset,
        "sorted_by": sort_by,
        "columns": df.columns,
        "rows": records,
        "truncated": offset + len(records) < matched.height,
        "summary": formatting.numeric_summary(matched if matched.height else df),
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# Distance
# --------------------------------------------------------------------------- #

_DISTANCE_NOTE = (
    "All four measures are distances (larger = more different). rank_sum combines "
    "them scale-free; prefer it over zscore_sum."
)

_DISTANCE_SUBSET_NOTE = (
    "Only {count} of the 4 measures ran ({names}), so rank_sum here combines {count} "
    "of them instead of 4 and is a weaker, differently-scaled ranking -- not "
    "comparable with a 4-measure rank_sum. Re-run with measures unset to add "
    "{missing} unless you have a specific reason to isolate one."
)

#: One measure makes rank_sum a relabelling of that measure, not a combination.
_DISTANCE_SINGLE_MEASURE_NOTE = (
    "With one measure, rank_sum is simply that measure's rank, so it carries none "
    "of the cross-measure agreement it is there to provide."
)


def _distance_notes(measures, *extra):
    """Standing distance guidance, plus a warning if the caller narrowed ``measures``.

    Mirrors ``_anomaly_notes``: a model narrowing ``measures`` to save time is
    exactly what rank_sum exists to guard against, so the result says so rather
    than leaving it to a docstring the model saw once.
    """
    notes = [_DISTANCE_NOTE]
    used = distance.DEFAULT_MEASURES if measures is None else list(measures)
    missing = [name for name in distance.DEFAULT_MEASURES if name not in used]
    if missing:
        notes.append(_DISTANCE_SUBSET_NOTE.format(
            count=len(used), names=", ".join(used) or "none",
            missing=", ".join(missing),
        ))
        if len(used) == 1:
            notes.append(_DISTANCE_SINGLE_MEASURE_NOTE)
    notes.extend(extra)
    return notes


@tool
def distance_folder_filename(
    session_id: str,
    target_folder: str,
    comparison_folders: FolderSelector = "ALL",
    max_rows: int = 25,
) -> dict:
    """Compare log folders by which file names they contain. Never opens a file.

    The cheapest signal available. A log folder that
    crashed early might be missing files, one that retried might have extra ones.

    Args:
        session_id: Handle from open_log_root.
        target_folder: Exact log folder name to investigate.
        comparison_folders: "ALL", a list of names, an int N for the first N,
            or a "Prefix*" wildcard. The target is always excluded.
        max_rows: Rows returned inline, largest distance (least similar) first.
            For the closest matches instead, call query_result on the returned
            result_id with sort_by="jaccard distance", descending=False.
    """
    session = STORE.get(session_id)
    results = distance.distance_folder_filename(session.df, target_folder, comparison_folders)
    artifact = _write(session, results, "dis", 1, target_folder=target_folder, comparison_folder="Many")
    return formatting.result(
        session, "distance_folder_filename", 1,
        {"target_folder": target_folder, "comparison_folders": comparison_folders},
        results, artifact, max_rows, sort_by="jaccard distance",
        notes=["Distances: 1.0 means no file names in common, 0.0 means identical sets."],
    )


@tool
def distance_folder_content(
    session_id: str,
    target_folder: str,
    comparison_folders: FolderSelector = "ALL",
    mask: bool = True,
    content_format: str = "Words",
    vectorizer: str = "Count",
    measures: Optional[Sequence[str]] = None,
    max_rows: int = 25,
) -> dict:
    """Compare log folders by their log text content, with four distance measures.

    The four measures are cosine, jaccard, compression, and containment
    distance (larger = more different); rank_sum/zscore_sum in the result
    combine them scale-free -- prefer rank_sum. Running all four is a pairwise
    comparison per measure, so cost increase with comparison_folders. 
    Consider select only one when measuring distance between many logs. 
    Cosine,jaccard, and containment are equally cheap (matrix ops on vectors already
    built), while compression (a bz2 pass over the full text)
    gets expensive as the log folders grow large. Containment: unlike the other 
    three it is not 
    symmetric -- it scores how much of one side's text is contained in the
    other's, so target-vs-comparison and comparison-vs-target can differ
    sharply (e.g. 0 one way, 0.7 the other).
    Args:
        session_id: Handle from open_log_root.
        target_folder: Exact log folder name to investigate.
        comparison_folders: "ALL", a list, an int N, or a "Prefix*" wildcard.
        mask: Compare masked text. Requires a session opened with mask=True.
        content_format: "Words", "3grams", "Sklearn" (raw text), or
            "Parse-<Algorithm>" such as "Parse-Tip" or "Parse-Drain".
        vectorizer: "Count" or "Tfidf".
        measures: Leave unset. All four of ["cosine", "jaccard", "compression",
            "containment"] then run and rank_sum combines them, which is what
            makes the ranking trustworthy. Narrowing this weakens rank_sum; do
            it only to answer a question about one measure -- e.g. isolating
            "compression" (a bz2 pass over the full text) from the other three
            (matrix ops on the already-built vectors).
        max_rows: Rows returned inline, largest distance (least similar) first.
            For the closest matches instead, call query_result on the returned
            result_id with sort_by="rank_sum", descending=False.
    """
    session = STORE.get(session_id)
    session.ensure_content(mask, content_format)
    results, session.df = distance.distance_folder_content(
        session.df, target_folder, comparison_folders, mask, content_format, vectorizer,
        measures,
    )
    session.flush()
    artifact = _write(
        session, results, "dis", 2, target_folder=target_folder, comparison_folder="Many",
        mask=mask, content_format=content_format, vectorizer=vectorizer,
    )
    return formatting.result(
        session, "distance_folder_content", 2,
        {"target_folder": target_folder, "comparison_folders": comparison_folders, "mask": mask,
         "content_format": content_format, "vectorizer": vectorizer, "measures": measures},
        results, artifact, max_rows, sort_by=["rank_sum", "cosine"],
        notes=_distance_notes(measures),
    )


@tool
def distance_file_content(
    session_id: str,
    target_folder: str,
    comparison_folders: FolderSelector = "ALL",
    target_files: FileSelector = "ALL",
    mask: bool = True,
    content_format: str = "Words",
    vectorizer: str = "Count",
    measures: Optional[Sequence[str]] = None,
    max_rows: int = 25,
) -> dict:
    """Compare each file against the same-named file in other log folders.

    Only files present in both log folders can be compared -- use
    `describe_log_root(include_files=True)` to see which those are.
    The four measures are cosine, jaccard, compression, and containment
    distance (larger = more different); rank_sum/zscore_sum in the result
    combine them scale-free -- prefer rank_sum. Running all four is a pairwise
    comparison per measure, so cost increase with comparison_folders. 
    Consider select only one when measuring distance between many logs. 
    Cosine,jaccard, and containment are equally cheap (matrix ops on vectors already
    built), while compression (a bz2 pass over the full text)
    gets expensive as the log folders grow large. Containment: unlike the other 
    three it is not 
    symmetric -- it scores how much of one side's text is contained in the
    other's, so target-vs-comparison and comparison-vs-target can differ
    sharply (e.g. 0 one way, 0.7 the other).

    Args:
        session_id: Handle from open_log_root.
        target_folder: Exact log folder name to investigate.
        comparison_folders: "ALL", a list, an int N, or a "Prefix*" wildcard.
        target_files: "ALL", a list of file names, an int N, or a "name*" wildcard.
        mask: Compare masked text.
        content_format: "Words", "3grams", "Sklearn", or "Parse-<Algorithm>".
        vectorizer: "Count" or "Tfidf".
        measures: Leave unset. All four of ["cosine", "jaccard", "compression",
            "containment"] then run and rank_sum combines them, which is what
            makes the ranking trustworthy. Narrowing this weakens rank_sum; do
            it only to answer a question about one measure.
        max_rows: Rows returned inline, largest distance (least similar) first.
            For the closest matches instead, call query_result on the returned
            result_id with sort_by="zscore_sum", descending=False.
    """
    session = STORE.get(session_id)
    session.ensure_content(mask, content_format)
    results, session.df = distance.distance_file_content(
        session.df, target_folder, comparison_folders, target_files, mask,
        content_format, vectorizer, measures,
    )
    session.flush()
    artifact = _write(
        session, results, "dis", 3, target_folder=target_folder, comparison_folder="Many",
        mask=mask, content_format=content_format, vectorizer=vectorizer,
    )
    return formatting.result(
        session, "distance_file_content", 3,
        {"target_folder": target_folder, "comparison_folders": comparison_folders,
         "target_files": target_files, "mask": mask,
         "content_format": content_format, "vectorizer": vectorizer, "measures": measures},
        results, artifact, max_rows, sort_by=["zscore_sum", "cosine"],
        notes=_distance_notes(measures),
    )


@tool
def distance_line_content(
    session_id: str,
    target_folder: str,
    comparison_folders: FolderSelector = "ALL",
    target_files: FileSelector = "ALL",
    mask: bool = True,
    max_changed_lines: int = 40,
) -> dict:
    """Line-by-line diff of a file between the target log folder and others.

    Unlike the other distance_* tools, this does not vectorize and score --
    it runs a text diff, which only reads well between two specific log
    folders. Use it once the other measures have narrowed things down to a
    small comparison set, not as a first pass over many log folders.

    Returns change counts per comparison plus a sample of the differing lines;
    the complete diff for each pair is written to disk.

    Args:
        session_id: Handle from open_log_root.
        target_folder: Exact log folder name to investigate.
        comparison_folders: "ALL", a list, an int N, or a "Prefix*" wildcard.
            One diff is produced per comparison log folder per file, so narrow this.
        target_files: "ALL", a list of file names, an int N, or a "name*" wildcard.
        mask: Diff the masked text, which hides timestamp and id churn.
        max_changed_lines: Changed lines sampled per pair.
    """
    session = STORE.get(session_id)
    diffs = distance.distance_line_content(
        session.df, target_folder, comparison_folders, target_files, mask
    )

    comparisons = []
    for file_name, other_folder, diff_df in diffs:
        artifact = _write(
            session, diff_df, "dis", 4, target_folder=target_folder,
            comparison_folder=other_folder, mask=mask, file=file_name,
        )
        changed = diff_df.filter(pl.col("difference").is_in(["-", "+"]))
        comparisons.append({
            "file_name": file_name,
            "comparison_folder": other_folder,
            "summary": distance.summarize_diff(diff_df),
            "changed_sample": changed.head(max_changed_lines).to_dicts(),
            "changed_truncated": changed.height > max_changed_lines,
            "artifact": artifact,
        })

    return {
        "session_id": session_id,
        "analysis": "distance_line_content",
        "level": 4,
        "params": {"target_folder": target_folder, "comparison_folders": comparison_folders,
                   "target_files": target_files, "mask": mask},
        "n_comparisons": len(comparisons),
        "comparisons": comparisons,
        "notes": ["'-' is present only in the target log folder, '+' only in the comparison."],
    }


# --------------------------------------------------------------------------- #
# Anomaly
# --------------------------------------------------------------------------- #

_ANOMALY_NOTE = (
    "Rank by rank_sum of 4 detectors. Single detector, each is on its own scale and "
    "is weak evidence alone. rank_sum beats zscore_sum, which one "
    "distorted detector can dominate. No labels here, so this is suspicion, not a verdict."
)

_SUBSET_NOTE = (
    "Only {count} of the 4 detectors ran ({names}), so rank_sum here combines {count} of "
    "them instead of 4 and is a weaker, differently-scaled ranking -- not comparable with a "
    "4-detector rank_sum. "
    "Re-run with detectors unset to add {missing} unless you have a specific reason to "
    "exclude them."
)

#: One detector makes rank_sum a relabelling of that detector, not a combination.
_SINGLE_DETECTOR_NOTE = (
    "With one detector, rank_sum is simply that detector's rank, so it carries none of "
    "the cross-detector agreement it is there to provide."
)


def _anomaly_notes(detectors, *extra):
    """Standing anomaly guidance, plus a warning if the caller narrowed the detectors.

    Models driving these tools tend to read a single detector's score as a finding
    and to narrow ``detectors`` to save time, which is exactly what rank_sum exists
    to prevent -- so the result says so whenever it happens, rather than only in a
    docstring the model saw once.
    """
    notes = [_ANOMALY_NOTE]
    used = anomaly.DEFAULT_DETECTORS if detectors is None else list(detectors)
    missing = [name for name in anomaly.DEFAULT_DETECTORS if name not in used]
    if missing:
        notes.append(_SUBSET_NOTE.format(
            count=len(used), names=", ".join(used) or "none",
            missing=", ".join(missing),
        ))
        if len(used) == 1:
            notes.append(_SINGLE_DETECTOR_NOTE)
    notes.extend(extra)
    return notes


@tool
def anomaly_folder_filename(
    session_id: str,
    target_folder: FolderSelector = "ALL",
    comparison_folders: FolderSelector = "ALL",
    detectors: Optional[Sequence[str]] = None,
    detector_params: Optional[dict] = None,
    max_rows: int = 25,
) -> dict:
    """Train anomaly detection model on log file names.
    Score whole log folders, by their set of file names.

    Args:
        session_id: Handle from open_log_root.
        target_folder: Log folders to score -- "ALL", a name, an int N, or "Prefix*".
            Each target is scored against its own baseline of comparison folders.
        comparison_folders: The baseline. "ALL", a list, an int N, or "Prefix*".
        detectors: Leave unset. All four of ["KMeans", "IsolationForest",
            "RarityModel", "OOVDetector"] then run and rank_sum combines them,
            which is what makes the ranking trustworthy. Narrowing this weakens
            rank_sum; do it only to answer a question about one detector.
        detector_params: Per-detector overrides, e.g.
            {"KMeans": {"n_clusters": 3}, "RarityModel": {"threshold": 100}}.
        max_rows: Rows returned inline.
    """
    session = STORE.get(session_id)
    results, session.df = anomaly.anomaly_folder(
        session.df, target_folder, comparison_folders, file=True, detectors=detectors,
        mask=False, detector_params=detector_params,
    )
    session.flush()
    artifact = _write(session, results, "ano", 1, target_folder="Many", comparison_folder="Many")
    return formatting.result(
        session, "anomaly_folder_filename", 1,
        {"target_folder": target_folder, "comparison_folders": comparison_folders,
         "detectors": detectors, "detector_params": detector_params},
        results, artifact, max_rows, sort_by=["rank_sum", "zscore_sum"],
        notes=_anomaly_notes(detectors),
    )


@tool
def anomaly_folder_content(
    session_id: str,
    target_folder: FolderSelector = "ALL",
    comparison_folders: FolderSelector = "ALL",
    detectors: Optional[Sequence[str]] = None,
    mask: bool = True,
    content_format: str = "Words",
    vectorizer: str = "Count",
    detector_params: Optional[dict] = None,
    max_rows: int = 25,
) -> dict:
    """Train anomaly detection models on comparison_folders' log text, then
    score whole log folders by their log text.

    Args:
        session_id: Handle from open_log_root.
        target_folder: Log folders to score -- "ALL", a name, an int N, or "Prefix*".
        comparison_folders: The training baseline. Point this at known-good folders
            when you have them".
        detectors: Leave unset so all four run -- rank_sum is only trustworthy
            when it combines all of them. Narrowing this weakens the ranking.
        mask: Use masked text. Requires a session opened with mask=True.
        content_format: "Words", "3grams", "Sklearn", or "Parse-<Algorithm>".
        vectorizer: "Count" or "Tfidf".
        detector_params: Per-detector keyword overrides.
        max_rows: Rows returned inline.
    """
    session = STORE.get(session_id)
    session.ensure_content(mask, content_format)
    results, session.df = anomaly.anomaly_folder(
        session.df, target_folder, comparison_folders, file=False, detectors=detectors,
        mask=mask, content_format=content_format, vectorizer=vectorizer,
        detector_params=detector_params,
    )
    session.flush()
    artifact = _write(
        session, results, "ano", 2, target_folder="Many", comparison_folder="Many", mask=mask,
        content_format=content_format, vectorizer=vectorizer,
    )
    return formatting.result(
        session, "anomaly_folder_content", 2,
        {"target_folder": target_folder, "comparison_folders": comparison_folders,
         "detectors": detectors, "mask": mask, "content_format": content_format,
         "vectorizer": vectorizer, "detector_params": detector_params},
        results, artifact, max_rows, sort_by=["rank_sum", "zscore_sum"],
        notes=_anomaly_notes(detectors),
    )


@tool
def anomaly_file_content(
    session_id: str,
    target_folder: FolderSelector,
    comparison_folders: FolderSelector = "ALL",
    target_files: FileSelector = "ALL",
    detectors: Optional[Sequence[str]] = None,
    mask: bool = True,
    content_format: str = "Words",
    vectorizer: str = "Count",
    detector_params: Optional[dict] = None,
    max_rows: int = 25,
) -> dict:
    """Train anomaly detection models on comparison_folders' log text, per file,
    then score each file of the target log folder against the same file elsewhere.

    Narrows a suspicious log folder down to the file worth reading.

    Files are matched by name across log folders: the baseline for security.log
    is the other log folders' security.log, one document each. A target file
    that no comparison log folder has is skipped, so this level needs log
    folders that share file names -- if each log folder holds one uniquely-named
    file, use anomaly_folder_content instead.

    Args:
        session_id: Handle from open_log_root.
        target_folder: Log folders to score -- a name, "ALL", an int N, or "Prefix*".
        comparison_folders: The training baseline. Point this at known-good folders
            when you have them".
        target_files: "ALL", a list, an int N, or a "name*" wildcard.
        detectors: Leave unset so all four run -- rank_sum is only trustworthy
            when it combines all of them. Narrowing this weakens the ranking.
        mask: Use masked text.
        content_format: "Words", "3grams", "Sklearn", or "Parse-<Algorithm>".
        vectorizer: "Count" or "Tfidf".
        detector_params: Per-detector keyword overrides.
        max_rows: Rows returned inline.
    """
    session = STORE.get(session_id)
    session.ensure_content(mask, content_format)
    results, session.df = anomaly.anomaly_file_content(
        session.df, target_folder, comparison_folders, target_files, detectors, mask,
        content_format, vectorizer, detector_params,
    )
    session.flush()
    artifact = _write(
        session, results, "ano", 3, target_folder="Many", comparison_folder="Many", mask=mask,
        content_format=content_format, vectorizer=vectorizer,
    )
    return formatting.result(
        session, "anomaly_file_content", 3,
        {"target_folder": target_folder, "comparison_folders": comparison_folders,
         "target_files": target_files, "detectors": detectors, "mask": mask,
         "content_format": content_format, "vectorizer": vectorizer,
         "detector_params": detector_params},
        results, artifact, max_rows, sort_by=["rank_sum", "zscore_sum"],
        notes=_anomaly_notes(detectors),
    )


@tool
def anomaly_line_content(
    session_id: str,
    target_folder: FolderSelector,
    comparison_folders: FolderSelector = "ALL",
    target_files: FileSelector = "ALL",
    detectors: Optional[Sequence[str]] = None,
    mask: bool = True,
    content_format: str = "Words",
    vectorizer: str = "Count",
    detector_params: Optional[dict] = None,
    max_rows: int = 20,
    sort_by: str = "rank_sum",
) -> dict:
    """Train anomaly detection models on comparison_folders' log text, per line,
    then score every line of a target file, returning the worst with their text.

    The end of the drill-down. Each returned row is a real log line with its
    score, so you can read what actually made the log folder look wrong. Writes an
    interactive HTML plot of scores against line number per file.

    Args:
        session_id: Handle from open_log_root.
        target_folder: Log folders to score -- a name, "ALL", an int N, or "Prefix*".
        comparison_folders: The training baseline.
        target_files: Which files to score. Narrow this -- one plot and one
            table are produced per file.
        detectors: Leave unset so all four run -- rank_sum is only trustworthy
            when it combines all of them. Narrowing this weakens the ranking.
        mask: Use masked text for scoring; the returned text is always raw.
        content_format: "Words", "3grams", "Sklearn", or "Parse-<Algorithm>".
        vectorizer: "Count" or "Tfidf".
        detector_params: Per-detector keyword overrides.
        max_rows: Top-scoring lines returned per file.
        sort_by: Score column to rank lines by. Keep "rank_sum" -- a single
            detector column such as "RM_pred_ano_proba" ranks by that detector
            alone and is for investigating one detector, not for finding the
            worst lines. "moving_avg_100_RM_pred_ano_proba" and its siblings are
            the exception worth reaching for: they find suspicious *regions*
            rather than single lines.
    """
    session = STORE.get(session_id)
    session.ensure_content(mask, content_format)
    per_file, session.df = anomaly.anomaly_line_content(
        session.df, target_folder, comparison_folders, target_files, detectors, mask,
        content_format, vectorizer, detector_params,
    )
    session.flush()

    files = []
    for folder_name, file_name, scored in per_file:
        scored = scoring.add_combined_scores(scored, scoring.ANOMALY_COLUMNS)
        artifact = _write(
            session, scored, "ano", 4, target_folder=folder_name, comparison_folder="Many",
            mask=mask, content_format=content_format, vectorizer=vectorizer,
            file=file_name,
        )
        title = (
            f"Anomaly scores - mask:{mask}, {content_format}, {vectorizer}"
            f"<br>Target log folder: {folder_name}<br>Target file: {file_name}"
        )
        stem = export.build_file_name(
            analysis="ano_plot", level=4, target_folder=folder_name, comparison_folder="Many",
            mask=mask, content_format=content_format, vectorizer=vectorizer,
            file=file_name,
        )
        plot = export.write_figure(
            visualize.plot_line_scores(scored, title), str(session.output_dir), stem
        )
        ranked, sorted_by = formatting.sort_for_preview(scored, [sort_by, "rank_sum"])
        top_lines = formatting.rows_to_records(ranked, max_rows)
        # This tool builds its own per-file entries rather than going through
        # formatting.result, so it stashes its own tables -- one per file, which
        # is why Session bounds how many results it keeps.
        result_id = session.stash_result("anomaly_line_content", scored)
        entry_notes = []
        if scored.height > len(top_lines):
            entry_notes.append(
                f"Showing {len(top_lines)} of {scored.height} scored lines, sorted by "
                f"{sorted_by} descending."
            )
            entry_notes.append(
                formatting.query_hint(session_id, result_id, scored, sorted_by)
            )
        files.append({
            "target_folder": folder_name,
            "file_name": file_name,
            "n_lines": scored.height,
            "sorted_by": sorted_by,
            "result_id": result_id,
            "top_lines": top_lines,
            "artifact": artifact,
            "plot": plot,
            "notes": entry_notes,
        })

    return {
        "session_id": session_id,
        "analysis": "anomaly_line_content",
        "level": 4,
        "params": {"target_folder": target_folder, "comparison_folders": comparison_folders,
                   "target_files": target_files, "detectors": detectors, "mask": mask,
                   "content_format": content_format, "vectorizer": vectorizer,
                   "detector_params": detector_params},
        "n_files": len(files),
        "files": files,
        "notes": _anomaly_notes(
            detectors,
            "A single high line is often noise; a sustained rise in "
            "moving_avg_100_* marks the region where it went wrong.",
        ),
    }


# --------------------------------------------------------------------------- #
# Visualize
# --------------------------------------------------------------------------- #

#: The two axes every plot tool shares, and so what its `summary` describes --
#: these are what the picture shows, and a plot result sends no points.
PLOT_AXES = ("unique_terms", "lines")


def _target_point(points, folder):
    """The target log folder's own row, with its percentile on each axis.

    The target is the point the plot draws as a cross, so it is the one row a
    caller always wants -- and the percentiles are what make it readable.
    """
    if not folder or "folder" not in points.columns:
        return None
    row = points.filter(pl.col("folder") == folder)
    if row.height == 0:  # target filtered out, e.g. it lacks this file
        return None
    record = formatting.rows_to_records(row, 1)[0]
    for axis in PLOT_AXES:
        if axis in points.columns:
            record[f"{axis}_pct"] = formatting.percentile_of(points, axis, row[axis][0])
    return record


def _plot_result(session, analysis, level, params, points, figures):
    artifacts = {}
    for suffix, fig in figures.items():
        if fig is None:  # not requested via `plots`
            continue
        stem = export.build_file_name(
            analysis=f"{analysis}_{suffix}", level=level,
            target_folder=params.get("target_folder", ""), comparison_folder="Many",
            mask=params.get("mask", False),
            content_format=params.get("content_format", ""),
            vectorizer=params.get("vectorizer", ""),
            file=params.get("file", ""),
        )
        artifacts[suffix] = export.write_figure(fig, str(session.output_dir), stem)
    # The numbers are the result for a caller that cannot see the HTML, so the
    # note says what they mean rather than pointing at the picture.
    unit = "file names" if level == 1 else "terms"
    notes = [f"One point per log folder. unique_terms is how many distinct {unit} it "
             "uses (the x axis), lines is its line count (the y axis, log scale). "
             "A log folder far from the others on either is worth a look."]
    # A log root of one-file log folders makes the file-name plot degenerate: every
    # point shares an x, and a caller reading only the numbers sees a range of
    # zero width with nothing to say it was never going to differ. The
    # docstring says this too, but only this fires when it is actually happening.
    if level == 1 and points.height > 1 and points["unique_terms"].n_unique() == 1:
        count = points["unique_terms"][0]
        notes.append(
            f"CAUTION: every log folder here contains the same number of files ({count}), "
            "so the x axis is a single value and separates nothing. This plot needs log "
            "folders holding several files each. Use plot_folder_content, which reads the "
            "log text, or anomaly_folder_content for a ranking."
        )
    if "umap_x" in points.columns:
        notes.append("umap_x/umap_y place the same log folders in 2D: outliers sit away "
                     "from the cluster. The axes have no units -- only relative "
                     "positions mean anything.")
    else:
        notes.append('No UMAP was run, so there are no umap_x/umap_y columns. Pass '
                     'plots=["umap", "scatter"] if the positions above leave the answer '
                     'unclear; it sees which terms differ, not just how many, and costs '
                     'tens of seconds on a few thousand log folders.')

    # A plot result carries no rows, unlike every other tool here, because a
    # scatter has no top N: both ends of both axes matter, and so does the
    # target, which on a large log root is nowhere near the top of either (on
    # 5,000 HDFS log folders it ranks ~4,000th by unique_terms). Any first-N of
    # the points would be one arbitrary corner of the picture, so the shape of
    # it is described instead and the points themselves are a query away.
    result_id = session.stash_result(analysis, points)
    notes.append(
        "The points are not listed here -- 'summary' is the range of each axis and "
        "'target' is where the target log folder falls in it. Pick a threshold from "
        f'those and query_result(session_id="{session.session_id}", '
        f'result_id="{result_id}", where=[["lines", "<", <value>]]) returns exactly the '
        "log folders you mean."
    )
    payload = {
        "session_id": session.session_id,
        "analysis": analysis,
        "level": level,
        "params": params,
        "n_rows": points.height,
        "result_id": result_id,
        "summary": formatting.numeric_summary(points, PLOT_AXES),
        "plots": artifacts,
        "notes": notes,
    }
    target_point = _target_point(points, params.get("target_folder"))
    if target_point is not None:
        payload["target"] = target_point
    return payload


@tool
def plot_folder_filename(
    session_id: str,
    target_folder: str,
    comparison_folders: FolderSelector = "ALL",
    group_by_indices: Optional[Sequence[int]] = None,
    random_seed: Optional[int] = 42,
    plots: PlotSelector = visualize.DEFAULT_PLOTS,
) -> dict:
    """Scatter plot of every log folder, with an optional UMAP embedding. Scatter axes:
    X AXIS: distinct file names the log folder contains.
    Y AXIS: total log lines it contains, on a log scale.

    X and Y are the "unique_terms" and "lines" columns of the points,
    respectively, so you can read the plot without opening the HTML: "summary"
    gives each axis's range, "target" says where the target log folder sits,
    and query_result fetches any points you then want to see -- "every log
    folder under 5 lines", say. ("unique_terms" is the generic column name;
    here the terms are file names.)

    This plot is only useful when a log folder holds several files that recur
    by name across folders -- one file per container, service, task, or node.
    If every log folder holds one file, the X axis is the same for all of
    them and separates nothing; use plot_folder_content instead.

    Screening signals, not proof, since neither axis looks at file contents:
      - Fewer files can mean a component never started or died early; more
        can mean retries, since a restarted attempt writes under a new name.
      - Line count is roughly how much work happened: far fewer can mean an
        early crash/timeout/kill, far more can mean looping, retrying, or
        verbose stack traces.
    So a log folder with entirely ordinary counts can still hold one fatal
    line.

    UMAP cost: ~41s on 5,000 log folders vs under a second for the default --
    it stays off by default; ask for it only when the scatter leaves the
    answer unclear.

    Args:
        session_id: Handle from open_log_root.
        target_folder: Log folder to highlight with a cross marker.
        comparison_folders: Log folders to include. "ALL", a list, an int N, or "Prefix*".
        group_by_indices: Underscore-separated parts of the folder name to colour
            by, e.g. [0, 1] colours "PageRank_DiskFull_application_1" by
            "PageRank_DiskFull".
        random_seed: Makes the UMAP layout reproducible. Pass null for a fresh
            one; re-running with different layouts is a good stability check.
            Ignored unless you asked for "umap".
        plots: Which plots to build. One HTML file is written per entry.
            "scatter" (the default): the file-names-against-lines scatter
                described above. Cheap at any size.
            "umap": a different plot of the same log folders, based on which
                distinct file names each shares with the others, not just how
                many. Axes are "umap_x"/"umap_y" with no units -- only relative
                distance means anything, and outliers sit away from the cluster.
            Pass ["umap", "scatter"] for both; they share one vectorization, so
            both together cost no more than "umap" alone.
    """
    session = STORE.get(session_id)
    points, fig_umap, fig_scatter, session.df = visualize.plot_folder(
        session.df, target_folder, comparison_folders, file=True, random_seed=random_seed,
        group_by_indices=group_by_indices, mask=False, plots=plots,
    )
    session.flush()
    return _plot_result(
        session, "plot_folder_filename", 1,
        {"target_folder": target_folder, "comparison_folders": comparison_folders,
         "group_by_indices": group_by_indices, "random_seed": random_seed,
         "plots": list(plots)},
        points, {"umap": fig_umap, "scatter": fig_scatter},
    )


@tool
def plot_folder_content(
    session_id: str,
    target_folder: str,
    comparison_folders: FolderSelector = "ALL",
    group_by_indices: Optional[Sequence[int]] = None,
    mask: bool = True,
    content_format: str = "Words",
    vectorizer: str = "Count",
    random_seed: Optional[int] = 42,
    plots: PlotSelector = visualize.DEFAULT_PLOTS,
) -> dict:
    """Scatter plot of every log folder, with an optional UMAP embedding. Scatter axes:
    X AXIS: how many distinct terms the log folder's log text uses. A "term" is
        whatever `content_format` says -- a word by default, otherwise a
        3-gram or a parsed event template.
    Y AXIS: how many log lines it contains in total, on a log scale.

    X and Y are the "unique_terms" and "lines" columns of the points,
    respectively, so you can read the plot without opening the HTML: the
    result gives the range of each axis ("summary") and where the target log
    folder sits in it ("target"), and query_result returns any points you
    then want to see.

    Unlike plot_folder_filename this works whatever the folders hold, including
    one file each, because it reads the log text rather than the file layout.

    Screening signals, not proof, since neither axis looks at what the text says:
      - Distinct terms is vocabulary variety: more can mean it reached code
        paths the others didn't (error branches, stack traces); fewer means
        it never got far enough to say much.
      - Line count is roughly how much work happened: far fewer can mean an
        early crash/timeout/kill, far more can mean looping, retrying, or
        verbose stack traces.
    So a log folder can use exactly the usual number of words -- one of them
    just "OutOfMemoryError" -- and still look ordinary here.

    UMAP cost: ~41s on 5,000 log folders vs under a second for the default --
    it stays off by default; ask for it only when the scatter leaves the
    answer unclear.

    Args:
        session_id: Handle from open_log_root.
        target_folder: Log folder to highlight with a cross marker.
        comparison_folders: Log folders to include.
        group_by_indices: Folder-name parts to colour by, e.g. [0, 1].
        mask: Use masked text.
        content_format: "Words", "3grams", "Sklearn", or "Parse-<Algorithm>".
            Decides what counts as a term on the x axis.
        vectorizer: "Count" or "Tfidf".
        random_seed: Makes the UMAP layout reproducible; ignored unless you
            asked for "umap".
        plots: Which plots to build. One HTML file is written per entry.
            "scatter" (the default): the terms-against-lines scatter described
                above. Cheap at any size.
            "umap": a different plot of the same log folders, based on which
                distinct terms each shares with the others, not just how many.
                Axes are "umap_x"/"umap_y" with no units -- only relative
                distance means anything, and outliers sit away from the cluster.
            Pass ["umap", "scatter"] for both; they share one vectorization, so
            both together cost no more than "umap" alone.
    """
    session = STORE.get(session_id)
    session.ensure_content(mask, content_format)
    points, fig_umap, fig_scatter, session.df = visualize.plot_folder(
        session.df, target_folder, comparison_folders, file=False, random_seed=random_seed,
        group_by_indices=group_by_indices, mask=mask, content_format=content_format,
        vectorizer=vectorizer, plots=plots,
    )
    session.flush()
    return _plot_result(
        session, "plot_folder_content", 2,
        {"target_folder": target_folder, "comparison_folders": comparison_folders,
         "group_by_indices": group_by_indices, "mask": mask,
         "content_format": content_format, "vectorizer": vectorizer,
         "random_seed": random_seed, "plots": list(plots)},
        points, {"umap": fig_umap, "scatter": fig_scatter},
    )


@tool
def plot_file_content(
    session_id: str,
    target_folder: str,
    comparison_folders: FolderSelector = "ALL",
    target_files: FileSelector = "ALL",
    group_by_indices: Optional[Sequence[int]] = None,
    mask: bool = True,
    content_format: str = "Words",
    vectorizer: str = "Count",
    random_seed: Optional[int] = 42,
    plots: PlotSelector = visualize.DEFAULT_PLOTS,
) -> dict:
    """Scatter plot of one named file across log folders, each copy of it as one point.

    One plot per file you ask for. Within a plot, one point per log folder that
    has a file of that name:

    X AXIS: how many distinct terms that log folder's copy of the file uses. A
        "term" is whatever `content_format` says -- a word by default.
    Y AXIS: how many lines that copy has, on a log scale.

    X and Y are the "unique_terms" and "lines" columns of the points,
    respectively. Each file's entry carries the range of each axis
    ("summary"), where the target log folder sits in it ("target"), and a
    "result_id" -- query_result returns that file's points, one row per log
    folder holding a file of that name.

    This is the drill-down from the whole-folder plots: which log folder's
    copy of *this* file is the odd one out. A file only one log folder has is
    skipped -- there is nothing to compare it against.

    UMAP is slow, and one UMAP layout runs per file named in target_files --
    so cost increases the more files you ask for. Narrow target_files before adding "umap".

    Args:
        session_id: Handle from open_log_root.
        target_folder: Log folder to highlight with a cross marker.
        comparison_folders: Log folders to include.
        target_files: Which files to plot. "ALL", a list, an int N, or a
            wildcard, resolved against the files the target log folder has.
        group_by_indices: Folder-name parts to colour by, e.g. [0, 1].
        mask: Use masked text.
        content_format: "Words", "3grams", "Sklearn", or "Parse-<Algorithm>".
            Decides what counts as a term on the x axis.
        vectorizer: "Count" or "Tfidf".
        random_seed: Makes the UMAP layout reproducible; ignored unless you
            asked for "umap".
        plots: Which plots to build. One HTML file is written per entry, per file.
            "scatter" (the default): the terms-against-lines scatter described
                above. Cheap, linear in the files named.
            "umap": a different plot of the same points, based on which
                distinct terms each shares with the others. Axes are
                "umap_x"/"umap_y" with no units -- only relative distance
                means anything.
            Pass ["umap", "scatter"] for both; per file they share one
            vectorization, so both cost no more than "umap" alone.
    """
    session = STORE.get(session_id)
    session.ensure_content(mask, content_format)
    per_file, session.df = visualize.plot_file_content(
        session.df, target_folder, comparison_folders, target_files, random_seed,
        group_by_indices, mask, content_format, vectorizer, plots,
    )
    session.flush()

    files = []
    for file_name, points, fig_umap, fig_scatter in per_file:
        params = {"target_folder": target_folder, "mask": mask,
                  "content_format": content_format, "vectorizer": vectorizer,
                  "file": file_name}
        entry = _plot_result(
            session, "plot_file_content", 3, params,
            points, {"umap": fig_umap, "scatter": fig_scatter},
        )
        entry["file_name"] = file_name
        files.append(entry)

    return {
        "session_id": session_id,
        "analysis": "plot_file_content",
        "level": 3,
        "params": {"target_folder": target_folder, "comparison_folders": comparison_folders,
                   "target_files": target_files, "mask": mask,
                   "content_format": content_format, "vectorizer": vectorizer,
                   "plots": list(plots)},
        "n_files": len(files),
        "files": files,
    }


# --------------------------------------------------------------------------- #
# Reading LogDelta's YAML config format
# --------------------------------------------------------------------------- #
# TODO this LogDelta thing might not be needed here or at all. Delete?
#: LogDelta's step keys. These are a published *file format*, not an import --
#: nothing here depends on LogDelta -- so they keep its "run" vocabulary
#: verbatim. Only the values move with our renames.
_STEP_TOOLS = {
    "distance_run_file": distance_folder_filename,
    "distance_run_content": distance_folder_content,
    "distance_file_content": distance_file_content,
    "distance_line_content": distance_line_content,
    "anomaly_run_file": anomaly_folder_filename,
    "anomaly_run_content": anomaly_folder_content,
    "anomaly_file_content": anomaly_file_content,
    "anomaly_line_content": anomaly_line_content,
    "plot_run_file": plot_folder_filename,
    "plot_run_content": plot_folder_content,
    "plot_file_content": plot_file_content,
}

#: Likewise for step arguments. Without this the kwargs filter below would drop
#: a LogDelta ``target_run:`` *silently*, and the analysis would quietly run
#: against a default instead of the folder the config asked for.
_STEP_ARGS = {
    "target_run": "target_folder",
    "comparison_runs": "comparison_folders",
}

#: What a LogDelta step means but does not say. Its plot steps always draw both
#: the UMAP and the scatter view, and a config has no key to ask for either, so
#: reproducing one means requesting both here -- our own default is the cheap
#: half. Overridden by anything the config does state.
_STEP_DEFAULTS = {
    "plot_run_file": {"plots": visualize.PLOTS},
    "plot_run_content": {"plots": visualize.PLOTS},
    "plot_file_content": {"plots": visualize.PLOTS},
}

# LogDelta names preprocessing steps after its own functions; map to ours.
_PREPROCESSING = {"remove_run_name_from_file_names": "strip_folder_id"}


@tool
def run_config(config_path: str, session_id: Optional[str] = None,
               format: str = "auto") -> dict:
    """Run an existing LogDelta YAML config, then leave the logs open.

    Reproduces a batch config in one call and hands back the `session_id`, so
    you can follow up interactively without reloading anything.

    Args:
        config_path: Path to a LogDelta config.yml. Relative paths inside it
            resolve against the config file's own directory.
        session_id: Choose the handle for the session this opens.
        format: As in open_log_root. A LogDelta config says nothing about the
            format -- LogDelta reads every file as plain text -- so pass "raw"
            to reproduce its numbers exactly; the default detects per file.
    """
    path = os.path.abspath(os.path.expanduser(config_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    base = os.path.dirname(path)
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}

    def resolve(value):
        return value if os.path.isabs(value) else os.path.join(base, value)

    input_folder = config.get("input_data_folder") or os.environ.get("LOG_DATA_PATH")
    if not input_folder:
        raise ValueError(
            "Config has no 'input_data_folder' and LOG_DATA_PATH is not set."
        )

    regex_masking = config.get("regex_masking") or {}
    mask = bool(regex_masking.get("enabled", False))
    patterns = regex_masking.get("pattern") or []
    # LogDelta applies each pattern in turn but normalize() is idempotent, so
    # only the last one ever took effect. Use it directly.
    mask_pattern = patterns[-1]["name"] if patterns else "myllari_extended"

    pre_parse = config.get("pre_parse") or {}
    parsers = []
    if mask and pre_parse.get("enabled"):
        parsers = [
            p["name"].split("-", 1)[1].lower() for p in pre_parse.get("parsers", [])
        ]

    normalizer = "none"
    for step in config.get("preprocessing_steps") or []:
        mapped = _PREPROCESSING.get(step.get("name"))
        if mapped:
            normalizer = mapped

    output_folder = config.get("output_folder")
    session_info = open_log_root(
        path=resolve(input_folder),
        format=format,
        mask=mask,
        mask_pattern=mask_pattern,
        parsers=parsers,
        file_name_normalizer=normalizer,
        output_dir=resolve(output_folder) if output_folder else None,
        table_format=config.get("table_output", "csv"),
        session_id=session_id,
    )
    sid = session_info["session_id"]
    session = STORE.get(sid)

    executed, failed = [], []
    for step_name, items in (config.get("steps") or {}).items():
        tool = _STEP_TOOLS.get(step_name)
        if tool is None:
            failed.append({"step": step_name, "error": "unknown step"})
            continue
        for item in items or []:
            renamed = {_STEP_ARGS.get(k, k): v for k, v in item.items()}
            kwargs = {k: v for k, v in renamed.items() if k in tool.__annotations__}
            kwargs = {**_STEP_DEFAULTS.get(step_name, {}), **kwargs}
            try:
                tool(session_id=sid, **kwargs)
                executed.append({"step": step_name, "params": kwargs})
            except Exception as exc:  # keep going; report at the end
                failed.append({"step": step_name, "params": kwargs, "error": str(exc)})

    return {
        "session_id": sid,
        "config": path,
        "log_root": session_info,
        "output_dir": str(session.output_dir),
        "n_executed": len(executed),
        "executed": executed,
        "failed": failed,
        "notes": ["The session stays open -- pass this session_id to any tool to "
                  "drill in without reloading."],
    }


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="LogLead MCP server")
    parser.add_argument(
        "--transport", default="stdio", choices=["stdio", "sse", "streamable-http"],
        help="stdio (default) is what MCP clients launch.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP transports only.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP transports only.")
    parser.add_argument(
        "--cache-dir", default=None,
        help="Parquet cache location. Defaults to $LOGLEAD_MCP_CACHE or "
             "$XDG_CACHE_HOME/loglead-mcp.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Root for result tables and plots. Defaults to <cache-dir>/output.",
    )
    args = parser.parse_args()

    global STORE
    STORE = SessionStore(cache_dir=args.cache_dir, output_root=args.output_dir)

    kwargs = {} if args.transport == "stdio" else {"host": args.host, "port": args.port}
    mcp.run(transport=args.transport, **kwargs)


if __name__ == "__main__":
    main()
