"""MCP server exposing LogLead's run-comparison analyses.

Wraps :mod:`loglead.analysis` in a session model so a corpus is loaded, masked,
and parsed once and then interrogated repeatedly. The tool names mirror
LogDelta's YAML step names one-for-one, so an existing config translates
directly into a sequence of calls.

Run it with ``loglead-mcp`` (stdio, what MCP clients expect) or
``loglead-mcp --transport http --port 8000``.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import os
import sys
from typing import Optional, Sequence, Union

import polars as pl
import yaml

try:  # MCP SDK 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # MCP SDK 1.x, where the same class was called FastMCP
    from mcp.server.fastmcp import FastMCP as _Server

from ..analysis import anomaly, distance, export, scoring, visualize
from . import formatting
from .session import SessionStore

mcp = _Server("loglead")

#: Set by main(); tests and demos construct their own.
STORE = SessionStore()

#: A run selector: an exact name, "ALL", an int N, a "Prefix*" wildcard, or a list.
RunSelector = Union[str, int, Sequence[str]]

#: A file selector: same forms, resolved against the target run's files.
FileSelector = Union[str, int, Sequence[str]]


def tool(fn):
    """Register a function as an MCP tool, with stdout kept off the wire.

    LogLead and its dependencies print freely -- loader warnings, "e_words
    already found", Drain3's logger. Under the stdio transport stdout carries
    JSON-RPC frames, so a stray ``print`` would corrupt the stream. MCP SDK 2.x
    already diverts fd 1 to stderr for exactly this reason; redirecting here as
    well costs nothing there and keeps 1.x safe too.

    The wrapper is returned undecorated so the function stays directly callable
    from Python (which is how the demo and :func:`run_config` use these).
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(sys.stderr):
            return fn(*args, **kwargs)

    mcp.tool()(wrapper)
    return wrapper


def _write(session, df, analysis, level, **name_parts):
    """Persist a full result table and return its path."""
    stem = export.build_file_name(analysis=analysis, level=level, **name_parts)
    return export.write_table(df, str(session.output_dir), stem, session.table_format)


# --------------------------------------------------------------------------- #
# Corpus / session
# --------------------------------------------------------------------------- #

@tool
def open_corpus(
    path: str,
    filename_pattern: str = "*.log",
    mask: bool = True,
    mask_pattern: str = "myllari_extended",
    parsers: Optional[Sequence[str]] = None,
    file_name_normalizer: str = "none",
    min_file_size: int = 0,
    output_dir: Optional[str] = None,
    table_format: str = "csv",
    session_id: Optional[str] = None,
    refresh: bool = False,
) -> dict:
    """Load a folder-of-runs log corpus into a reusable session.

    Each immediate subdirectory of `path` is one **run**; files are matched by
    name across runs. Do this once, then run as many analyses as you like
    against the returned `session_id` -- nothing is re-read or re-parsed.

    Args:
        path: Corpus root directory.
        filename_pattern: Glob applied inside each run directory.
        mask: Replace volatile tokens (ids, IPs, timestamps, hex) with
            placeholders. Almost always wanted, and required for any parser.
        mask_pattern: One of "myllari_extended", "myllari", "drain_loglead",
            "drain_orig".
        parsers: Template parsers to run up front, e.g. ["tip"] or ["drain"].
            Optional -- analyses parse on demand -- but doing it here means the
            result lands in the cache.
        file_name_normalizer: "none", or "strip_run_id" when file names embed
            the run id (Hadoop container logs do). Without it, file-level and
            line-level analyses find no files in common between runs.
        min_file_size: Skip files this size or smaller, in bytes.
        output_dir: Where result tables and plots are written.
        table_format: "csv" (tab-separated, drops list columns) or "xlsx".
        session_id: Choose your own handle instead of a generated one.
        refresh: Ignore any cached parquet and re-read from disk.
    """
    session, info = STORE.open(
        path=path,
        filename_pattern=filename_pattern,
        mask=mask,
        mask_pattern=mask_pattern,
        parsers=parsers or (),
        file_name_normalizer=file_name_normalizer,
        min_file_size=min_file_size,
        output_dir=output_dir,
        table_format=table_format,
        session_id=session_id,
        refresh=refresh,
    )
    summary = session.summary()
    summary.update(info)
    runs = session.runs
    summary["runs"] = runs[:50]
    if len(runs) > 50:
        summary["notes"] = [f"{len(runs)} runs total; first 50 listed. "
                            "Use describe_corpus for the rest."]
    return summary


@tool
def list_corpora() -> dict:
    """List every open corpus session with its size and what has been computed."""
    return {"sessions": STORE.list()}


@tool
def describe_corpus(session_id: str, include_files: bool = False) -> dict:
    """Report the runs, file counts, and line counts of an open corpus.

    Args:
        session_id: Handle from open_corpus.
        include_files: Also list every distinct file name and how many runs
            contain it. Useful for picking a `target_files` value.
    """
    session = STORE.get(session_id)
    per_run = (
        session.df.group_by("run")
        .agg([pl.col("file_name").n_unique().alias("n_files"), pl.len().alias("n_lines")])
        .sort("run")
    )
    out = session.summary()
    out["runs_detail"] = per_run.to_dicts()

    if include_files:
        per_file = (
            session.df.group_by("file_name")
            .agg([pl.col("run").n_unique().alias("n_runs"), pl.len().alias("n_lines")])
            .sort("n_runs", descending=True)
        )
        out["files_detail"] = per_file.to_dicts()
        out["notes"] = [
            "Files present in many runs are the comparable ones; a file in only "
            "one run cannot be compared at L3/L4."
        ]
    return out


@tool
def close_corpus(session_id: str) -> dict:
    """Close a session and free its memory. The parquet cache is kept."""
    return {"closed": STORE.close(session_id)}


@tool
def read_log_lines(
    session_id: str,
    run: str,
    file_name: str,
    offset: int = 0,
    limit: int = 100,
    masked: bool = False,
) -> dict:
    """Read actual log lines. Use this to see the evidence behind a score.

    Args:
        session_id: Handle from open_corpus.
        run: Run name.
        file_name: Run-relative file name.
        offset: First line to return, 0-based.
        limit: How many lines (capped at 500).
        masked: Return the masked text instead of the raw message.
    """
    session = STORE.get(session_id)
    column = "e_message_normalized" if masked else "m_message"
    if column not in session.df.columns:
        raise ValueError(f"Column {column!r} is not available in this session.")

    selected = session.df.filter(
        (pl.col("run") == run) & (pl.col("file_name") == file_name)
    ).with_row_index("line_number")
    if selected.height == 0:
        raise ValueError(
            f"No lines for run={run!r} file={file_name!r}. "
            "Check describe_corpus for valid names."
        )

    limit = max(1, min(int(limit), 500))
    window = selected.slice(offset, limit).select(["line_number", column])
    return {
        "session_id": session_id,
        "run": run,
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
    runs: Optional[Sequence[str]] = None,
    files: Optional[Sequence[str]] = None,
    regex: bool = True,
    ignore_case: bool = False,
    limit: int = 50,
) -> dict:
    """Find log lines matching a pattern, and count matches per run.

    The per-run counts are often the answer on their own: a message that appears
    only in the suspect run, or 50x more often there, is the explanation for its
    anomaly score.

    Args:
        session_id: Handle from open_corpus.
        pattern: Regex, or a literal substring when regex is False.
        runs: Restrict to these runs. Defaults to all.
        files: Restrict to these file names. Defaults to all.
        regex: Treat pattern as a regular expression.
        ignore_case: Case-insensitive matching.
        limit: Maximum matching lines returned (capped at 200).
    """
    session = STORE.get(session_id)
    df = session.df
    if runs:
        df = df.filter(pl.col("run").is_in(list(runs)))
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

    per_run = (
        matches.group_by("run").agg(pl.len().alias("matches")).sort("matches", descending=True)
    )
    limit = max(1, min(int(limit), 200))
    sample = matches.select(["run", "file_name", "m_message"]).head(limit)

    return {
        "session_id": session_id,
        "pattern": pattern,
        "total_matches": matches.height,
        "runs_with_matches": per_run.height,
        "matches_per_run": per_run.to_dicts(),
        "sample": sample.to_dicts(),
        "truncated": matches.height > sample.height,
    }


# --------------------------------------------------------------------------- #
# Distance
# --------------------------------------------------------------------------- #

@tool
def distance_run_file(
    session_id: str,
    target_run: str,
    comparison_runs: RunSelector = "ALL",
    max_rows: int = 25,
) -> dict:
    """L1: compare runs by which file names they contain. Never opens a file.

    The cheapest signal available, and often enough on its own -- a run that
    crashed early is missing files, one that retried has extra ones.

    Args:
        session_id: Handle from open_corpus.
        target_run: Exact run name to investigate.
        comparison_runs: "ALL", a list of run names, an int N for the first N,
            or a "Prefix*" wildcard. The target is always excluded.
        max_rows: Rows returned inline.
    """
    session = STORE.get(session_id)
    results = distance.distance_run_file(session.df, target_run, comparison_runs)
    artifact = _write(session, results, "dis", 1, target_run=target_run, comparison_run="Many")
    return formatting.result(
        session, "distance_run_file", 1,
        {"target_run": target_run, "comparison_runs": comparison_runs},
        results, artifact, max_rows, sort_by="jaccard distance",
        notes=["Distances: 1.0 means no file names in common, 0.0 means identical sets."],
    )


@tool
def distance_run_content(
    session_id: str,
    target_run: str,
    comparison_runs: RunSelector = "ALL",
    mask: bool = True,
    content_format: str = "Words",
    vectorizer: str = "Count",
    max_rows: int = 25,
) -> dict:
    """L2: compare runs by their whole log text, with four distance measures.

    Args:
        session_id: Handle from open_corpus.
        target_run: Exact run name to investigate.
        comparison_runs: "ALL", a list, an int N, or a "Prefix*" wildcard.
        mask: Compare masked text. Requires a session opened with mask=True.
        content_format: "Words", "3grams", "Sklearn" (raw text), or
            "Parse-<Algorithm>" such as "Parse-Tip" or "Parse-Drain".
        vectorizer: "Count" or "Tfidf".
        max_rows: Rows returned inline.
    """
    session = STORE.get(session_id)
    session.ensure_content(mask, content_format)
    results, session.df = distance.distance_run_content(
        session.df, target_run, comparison_runs, mask, content_format, vectorizer
    )
    session.flush()
    artifact = _write(
        session, results, "dis", 2, target_run=target_run, comparison_run="Many",
        mask=mask, content_format=content_format, vectorizer=vectorizer,
    )
    return formatting.result(
        session, "distance_run_content", 2,
        {"target_run": target_run, "comparison_runs": comparison_runs, "mask": mask,
         "content_format": content_format, "vectorizer": vectorizer},
        results, artifact, max_rows, sort_by=["rank_sum", "cosine"],
        notes=["All four measures are distances (larger = more different). "
               "rank_sum combines them scale-free; prefer it over zscore_sum."],
    )


@tool
def distance_file_content(
    session_id: str,
    target_run: str,
    comparison_runs: RunSelector = "ALL",
    target_files: FileSelector = "ALL",
    mask: bool = True,
    content_format: str = "Words",
    vectorizer: str = "Count",
    max_rows: int = 25,
) -> dict:
    """L3: compare each file against the same-named file in other runs.

    Only files present in both runs can be compared -- use
    `describe_corpus(include_files=True)` to see which those are.

    Args:
        session_id: Handle from open_corpus.
        target_run: Exact run name to investigate.
        comparison_runs: "ALL", a list, an int N, or a "Prefix*" wildcard.
        target_files: "ALL", a list of file names, an int N, or a "name*" wildcard.
        mask: Compare masked text.
        content_format: "Words", "3grams", "Sklearn", or "Parse-<Algorithm>".
        vectorizer: "Count" or "Tfidf".
        max_rows: Rows returned inline.
    """
    session = STORE.get(session_id)
    session.ensure_content(mask, content_format)
    results, session.df = distance.distance_file_content(
        session.df, target_run, comparison_runs, target_files, mask,
        content_format, vectorizer,
    )
    session.flush()
    artifact = _write(
        session, results, "dis", 3, target_run=target_run, comparison_run="Many",
        mask=mask, content_format=content_format, vectorizer=vectorizer,
    )
    return formatting.result(
        session, "distance_file_content", 3,
        {"target_run": target_run, "comparison_runs": comparison_runs,
         "target_files": target_files, "mask": mask,
         "content_format": content_format, "vectorizer": vectorizer},
        results, artifact, max_rows, sort_by=["zscore_sum", "cosine"],
    )


@tool
def distance_line_content(
    session_id: str,
    target_run: str,
    comparison_runs: RunSelector = "ALL",
    target_files: FileSelector = "ALL",
    mask: bool = True,
    max_changed_lines: int = 40,
) -> dict:
    """L4: line-by-line diff of a file between the target run and other runs.

    Returns change counts per comparison plus a sample of the differing lines;
    the complete diff for each pair is written to disk.

    Args:
        session_id: Handle from open_corpus.
        target_run: Exact run name to investigate.
        comparison_runs: "ALL", a list, an int N, or a "Prefix*" wildcard.
            One diff is produced per comparison run per file, so narrow this.
        target_files: "ALL", a list of file names, an int N, or a "name*" wildcard.
        mask: Diff the masked text, which hides timestamp and id churn.
        max_changed_lines: Changed lines sampled per pair.
    """
    session = STORE.get(session_id)
    diffs = distance.distance_line_content(
        session.df, target_run, comparison_runs, target_files, mask
    )

    comparisons = []
    for file_name, other_run, diff_df in diffs:
        artifact = _write(
            session, diff_df, "dis", 4, target_run=target_run,
            comparison_run=other_run, mask=mask, file=file_name,
        )
        changed = diff_df.filter(pl.col("difference").is_in(["-", "+"]))
        comparisons.append({
            "file_name": file_name,
            "comparison_run": other_run,
            "summary": distance.summarize_diff(diff_df),
            "changed_sample": changed.head(max_changed_lines).to_dicts(),
            "changed_truncated": changed.height > max_changed_lines,
            "artifact": artifact,
        })

    return {
        "session_id": session_id,
        "analysis": "distance_line_content",
        "level": 4,
        "params": {"target_run": target_run, "comparison_runs": comparison_runs,
                   "target_files": target_files, "mask": mask},
        "n_comparisons": len(comparisons),
        "comparisons": comparisons,
        "notes": ["'-' is present only in the target run, '+' only in the comparison run."],
    }


# --------------------------------------------------------------------------- #
# Anomaly
# --------------------------------------------------------------------------- #

_ANOMALY_NOTE = (
    "Detector scores are on incomparable scales; sort by rank_sum. "
    "There are no labels here, so these are suspicion rankings, not verdicts."
)


@tool
def anomaly_run_file(
    session_id: str,
    target_run: RunSelector = "ALL",
    comparison_runs: RunSelector = "ALL",
    detectors: Optional[Sequence[str]] = None,
    detector_params: Optional[dict] = None,
    max_rows: int = 25,
) -> dict:
    """L1: score whole runs, describing each run by its set of file names.

    Args:
        session_id: Handle from open_corpus.
        target_run: Runs to score -- "ALL", a name, an int N, or "Prefix*".
            Each target is scored against its own baseline of comparison runs.
        comparison_runs: The baseline. "ALL", a list, an int N, or "Prefix*".
        detectors: Subset of ["KMeans", "IsolationForest", "RarityModel",
            "OOVDetector"]. Defaults to all four.
        detector_params: Per-detector overrides, e.g.
            {"KMeans": {"n_clusters": 3}, "RarityModel": {"threshold": 100}}.
        max_rows: Rows returned inline.
    """
    session = STORE.get(session_id)
    results, session.df = anomaly.anomaly_run(
        session.df, target_run, comparison_runs, file=True, detectors=detectors,
        mask=False, detector_params=detector_params,
    )
    session.flush()
    artifact = _write(session, results, "ano", 1, target_run="Many", comparison_run="Many")
    return formatting.result(
        session, "anomaly_run_file", 1,
        {"target_run": target_run, "comparison_runs": comparison_runs,
         "detectors": detectors, "detector_params": detector_params},
        results, artifact, max_rows, sort_by=["rank_sum", "zscore_sum"],
        notes=[_ANOMALY_NOTE],
    )


@tool
def anomaly_run_content(
    session_id: str,
    target_run: RunSelector = "ALL",
    comparison_runs: RunSelector = "ALL",
    detectors: Optional[Sequence[str]] = None,
    mask: bool = True,
    content_format: str = "Words",
    vectorizer: str = "Count",
    detector_params: Optional[dict] = None,
    max_rows: int = 25,
) -> dict:
    """L2: score whole runs, describing each run by its log text.

    The usual starting point of an investigation: score every run against the
    others, then drill into the top of the ranking.

    Args:
        session_id: Handle from open_corpus.
        target_run: Runs to score -- "ALL", a name, an int N, or "Prefix*".
        comparison_runs: The baseline. Point this at known-good runs when you
            have them, e.g. "PageRank_Normal*".
        detectors: Subset of the four detectors. Defaults to all.
        mask: Use masked text. Requires a session opened with mask=True.
        content_format: "Words", "3grams", "Sklearn", or "Parse-<Algorithm>".
        vectorizer: "Count" or "Tfidf".
        detector_params: Per-detector keyword overrides.
        max_rows: Rows returned inline.
    """
    session = STORE.get(session_id)
    session.ensure_content(mask, content_format)
    results, session.df = anomaly.anomaly_run(
        session.df, target_run, comparison_runs, file=False, detectors=detectors,
        mask=mask, content_format=content_format, vectorizer=vectorizer,
        detector_params=detector_params,
    )
    session.flush()
    artifact = _write(
        session, results, "ano", 2, target_run="Many", comparison_run="Many", mask=mask,
        content_format=content_format, vectorizer=vectorizer,
    )
    return formatting.result(
        session, "anomaly_run_content", 2,
        {"target_run": target_run, "comparison_runs": comparison_runs,
         "detectors": detectors, "mask": mask, "content_format": content_format,
         "vectorizer": vectorizer, "detector_params": detector_params},
        results, artifact, max_rows, sort_by=["rank_sum", "zscore_sum"],
        notes=[_ANOMALY_NOTE],
    )


@tool
def anomaly_file_content(
    session_id: str,
    target_run: RunSelector,
    comparison_runs: RunSelector = "ALL",
    target_files: FileSelector = "ALL",
    detectors: Optional[Sequence[str]] = None,
    mask: bool = True,
    content_format: str = "Words",
    vectorizer: str = "Count",
    detector_params: Optional[dict] = None,
    max_rows: int = 25,
) -> dict:
    """L3: score each file of the target run against the same file elsewhere.

    Narrows a suspicious run down to the file worth reading.

    Args:
        session_id: Handle from open_corpus.
        target_run: Runs to score -- a name, "ALL", an int N, or "Prefix*".
        comparison_runs: The baseline.
        target_files: "ALL", a list, an int N, or a "name*" wildcard.
        detectors: Subset of the four detectors. Defaults to all.
        mask: Use masked text.
        content_format: "Words", "3grams", "Sklearn", or "Parse-<Algorithm>".
        vectorizer: "Count" or "Tfidf".
        detector_params: Per-detector keyword overrides.
        max_rows: Rows returned inline.
    """
    session = STORE.get(session_id)
    session.ensure_content(mask, content_format)
    results, session.df = anomaly.anomaly_file_content(
        session.df, target_run, comparison_runs, target_files, detectors, mask,
        content_format, vectorizer, detector_params,
    )
    session.flush()
    artifact = _write(
        session, results, "ano", 3, target_run="Many", comparison_run="Many", mask=mask,
        content_format=content_format, vectorizer=vectorizer,
    )
    return formatting.result(
        session, "anomaly_file_content", 3,
        {"target_run": target_run, "comparison_runs": comparison_runs,
         "target_files": target_files, "detectors": detectors, "mask": mask,
         "content_format": content_format, "vectorizer": vectorizer,
         "detector_params": detector_params},
        results, artifact, max_rows, sort_by=["rank_sum", "zscore_sum"],
        notes=[_ANOMALY_NOTE],
    )


@tool
def anomaly_line_content(
    session_id: str,
    target_run: RunSelector,
    comparison_runs: RunSelector = "ALL",
    target_files: FileSelector = "ALL",
    detectors: Optional[Sequence[str]] = None,
    mask: bool = True,
    content_format: str = "Words",
    vectorizer: str = "Count",
    detector_params: Optional[dict] = None,
    max_rows: int = 20,
    sort_by: str = "rank_sum",
) -> dict:
    """L4: score every line of a target file, and return the worst with their text.

    The end of the drill-down. Each returned row is a real log line with its
    score, so you can read what actually made the run look wrong. Also writes an
    interactive HTML plot of scores against line number per file.

    Args:
        session_id: Handle from open_corpus.
        target_run: Runs to score -- a name, "ALL", an int N, or "Prefix*".
        comparison_runs: The baseline.
        target_files: Which files to score. Narrow this -- one plot and one
            table are produced per file.
        detectors: Subset of the four detectors. Defaults to all.
        mask: Use masked text for scoring; the returned text is always raw.
        content_format: "Words", "3grams", "Sklearn", or "Parse-<Algorithm>".
        vectorizer: "Count" or "Tfidf".
        detector_params: Per-detector keyword overrides.
        max_rows: Top-scoring lines returned per file.
        sort_by: Score column to rank lines by -- "rank_sum", or a single
            detector such as "RM_pred_ano_proba", or
            "moving_avg_100_RM_pred_ano_proba" to find suspicious *regions*.
    """
    session = STORE.get(session_id)
    session.ensure_content(mask, content_format)
    per_file, session.df = anomaly.anomaly_line_content(
        session.df, target_run, comparison_runs, target_files, detectors, mask,
        content_format, vectorizer, detector_params,
    )
    session.flush()

    files = []
    for run_name, file_name, scored in per_file:
        scored = scoring.add_combined_scores(scored, scoring.ANOMALY_COLUMNS)
        artifact = _write(
            session, scored, "ano", 4, target_run=run_name, comparison_run="Many",
            mask=mask, content_format=content_format, vectorizer=vectorizer,
            file=file_name,
        )
        title = (
            f"Anomaly scores - mask:{mask}, {content_format}, {vectorizer}"
            f"<br>Target run: {run_name}<br>Target file: {file_name}"
        )
        stem = export.build_file_name(
            analysis="ano_plot", level=4, target_run=run_name, comparison_run="Many",
            mask=mask, content_format=content_format, vectorizer=vectorizer,
            file=file_name,
        )
        plot = export.write_figure(
            visualize.plot_line_scores(scored, title), str(session.output_dir), stem
        )
        ranked, sorted_by = formatting.sort_for_preview(scored, [sort_by, "rank_sum"])
        files.append({
            "target_run": run_name,
            "file_name": file_name,
            "n_lines": scored.height,
            "sorted_by": sorted_by,
            "top_lines": formatting.rows_to_records(ranked, max_rows),
            "artifact": artifact,
            "plot": plot,
        })

    return {
        "session_id": session_id,
        "analysis": "anomaly_line_content",
        "level": 4,
        "params": {"target_run": target_run, "comparison_runs": comparison_runs,
                   "target_files": target_files, "detectors": detectors, "mask": mask,
                   "content_format": content_format, "vectorizer": vectorizer,
                   "detector_params": detector_params},
        "n_files": len(files),
        "files": files,
        "notes": [
            _ANOMALY_NOTE,
            "A single high line is often noise; a sustained rise in "
            "moving_avg_100_* marks the region where the run went wrong.",
        ],
    }


# --------------------------------------------------------------------------- #
# Visualize
# --------------------------------------------------------------------------- #

def _plot_result(session, analysis, level, params, points, figures, max_rows):
    artifacts = {}
    for suffix, fig in figures.items():
        stem = export.build_file_name(
            analysis=f"{analysis}_{suffix}", level=level,
            target_run=params.get("target_run", ""), comparison_run="Many",
            mask=params.get("mask", False),
            content_format=params.get("content_format", ""),
            vectorizer=params.get("vectorizer", ""),
            file=params.get("file", ""),
        )
        artifacts[suffix] = export.write_figure(fig, str(session.output_dir), stem)
    return formatting.result(
        session, analysis, level, params, points, artifacts.get("umap"), max_rows,
        sort_by=["unique_terms"],
        extra={"plots": artifacts},
        notes=["umap_x/umap_y place each run in 2D: outliers sit away from the "
               "cluster. unique_terms vs lines is the simpler, directly "
               "interpretable view."],
    )


@tool
def plot_run_file(
    session_id: str,
    target_run: str,
    comparison_runs: RunSelector = "ALL",
    group_by_indices: Optional[Sequence[int]] = None,
    random_seed: Optional[int] = 42,
    max_rows: int = 60,
) -> dict:
    """L1: plot every run as one point, described by its file names.

    Writes two interactive HTML plots and returns the coordinates, so the
    positions are readable without opening them.

    Args:
        session_id: Handle from open_corpus.
        target_run: Run to highlight with a cross marker.
        comparison_runs: Runs to include. "ALL", a list, an int N, or "Prefix*".
        group_by_indices: Underscore-separated parts of the run name to colour
            by, e.g. [0, 1] colours "PageRank_DiskFull_application_1" by
            "PageRank_DiskFull".
        random_seed: Makes UMAP reproducible. Pass null for a fresh layout;
            re-running with different layouts is a good stability check.
        max_rows: Runs returned inline.
    """
    session = STORE.get(session_id)
    points, fig_umap, fig_simple, session.df = visualize.plot_run(
        session.df, target_run, comparison_runs, file=True, random_seed=random_seed,
        group_by_indices=group_by_indices, mask=False,
    )
    session.flush()
    return _plot_result(
        session, "plot_run_file", 1,
        {"target_run": target_run, "comparison_runs": comparison_runs,
         "group_by_indices": group_by_indices, "random_seed": random_seed},
        points, {"umap": fig_umap, "simple": fig_simple}, max_rows,
    )


@tool
def plot_run_content(
    session_id: str,
    target_run: str,
    comparison_runs: RunSelector = "ALL",
    group_by_indices: Optional[Sequence[int]] = None,
    mask: bool = True,
    content_format: str = "Words",
    vectorizer: str = "Count",
    random_seed: Optional[int] = 42,
    max_rows: int = 60,
) -> dict:
    """L2: plot every run as one point, described by its log text.

    Args:
        session_id: Handle from open_corpus.
        target_run: Run to highlight with a cross marker.
        comparison_runs: Runs to include.
        group_by_indices: Run-name parts to colour by, e.g. [0, 1].
        mask: Use masked text.
        content_format: "Words", "3grams", "Sklearn", or "Parse-<Algorithm>".
        vectorizer: "Count" or "Tfidf".
        random_seed: Makes UMAP reproducible.
        max_rows: Runs returned inline.
    """
    session = STORE.get(session_id)
    session.ensure_content(mask, content_format)
    points, fig_umap, fig_simple, session.df = visualize.plot_run(
        session.df, target_run, comparison_runs, file=False, random_seed=random_seed,
        group_by_indices=group_by_indices, mask=mask, content_format=content_format,
        vectorizer=vectorizer,
    )
    session.flush()
    return _plot_result(
        session, "plot_run_content", 2,
        {"target_run": target_run, "comparison_runs": comparison_runs,
         "group_by_indices": group_by_indices, "mask": mask,
         "content_format": content_format, "vectorizer": vectorizer,
         "random_seed": random_seed},
        points, {"umap": fig_umap, "simple": fig_simple}, max_rows,
    )


@tool
def plot_file_content(
    session_id: str,
    target_run: str,
    comparison_runs: RunSelector = "ALL",
    target_files: FileSelector = "ALL",
    group_by_indices: Optional[Sequence[int]] = None,
    mask: bool = True,
    content_format: str = "Words",
    vectorizer: str = "Count",
    random_seed: Optional[int] = 42,
    max_rows: int = 60,
) -> dict:
    """L3: for each target file, plot each run's copy of that file as one point.

    Args:
        session_id: Handle from open_corpus.
        target_run: Run to highlight with a cross marker.
        comparison_runs: Runs to include.
        target_files: Which files to plot -- two plots are produced per file.
        group_by_indices: Run-name parts to colour by, e.g. [0, 1].
        mask: Use masked text.
        content_format: "Words", "3grams", "Sklearn", or "Parse-<Algorithm>".
        vectorizer: "Count" or "Tfidf".
        random_seed: Makes UMAP reproducible.
        max_rows: Runs returned inline per file.
    """
    session = STORE.get(session_id)
    session.ensure_content(mask, content_format)
    per_file, session.df = visualize.plot_file_content(
        session.df, target_run, comparison_runs, target_files, random_seed,
        group_by_indices, mask, content_format, vectorizer,
    )
    session.flush()

    files = []
    for file_name, points, fig_umap, fig_simple in per_file:
        params = {"target_run": target_run, "mask": mask,
                  "content_format": content_format, "vectorizer": vectorizer,
                  "file": file_name}
        entry = _plot_result(
            session, "plot_file_content", 3, params,
            points, {"umap": fig_umap, "simple": fig_simple}, max_rows,
        )
        entry["file_name"] = file_name
        files.append(entry)

    return {
        "session_id": session_id,
        "analysis": "plot_file_content",
        "level": 3,
        "params": {"target_run": target_run, "comparison_runs": comparison_runs,
                   "target_files": target_files, "mask": mask,
                   "content_format": content_format, "vectorizer": vectorizer},
        "n_files": len(files),
        "files": files,
    }


# --------------------------------------------------------------------------- #
# LogDelta config bridge
# --------------------------------------------------------------------------- #

_STEP_TOOLS = {
    "distance_run_file": distance_run_file,
    "distance_run_content": distance_run_content,
    "distance_file_content": distance_file_content,
    "distance_line_content": distance_line_content,
    "anomaly_run_file": anomaly_run_file,
    "anomaly_run_content": anomaly_run_content,
    "anomaly_file_content": anomaly_file_content,
    "anomaly_line_content": anomaly_line_content,
    "plot_run_file": plot_run_file,
    "plot_run_content": plot_run_content,
    "plot_file_content": plot_file_content,
}

# LogDelta names preprocessing steps after its own functions; map to ours.
_PREPROCESSING = {"remove_run_name_from_file_names": "strip_run_id"}


@tool
def run_config(config_path: str, session_id: Optional[str] = None) -> dict:
    """Run an existing LogDelta YAML config, then leave the corpus open.

    Reproduces a batch config in one call and hands back the `session_id`, so
    you can follow up interactively without reloading anything.

    Args:
        config_path: Path to a LogDelta config.yml. Relative paths inside it
            resolve against the config file's own directory.
        session_id: Choose the handle for the corpus this opens.
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
    session_info = open_corpus(
        path=resolve(input_folder),
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
            kwargs = {k: v for k, v in item.items() if k in tool.__annotations__}
            try:
                tool(session_id=sid, **kwargs)
                executed.append({"step": step_name, "params": kwargs})
            except Exception as exc:  # keep going; report at the end
                failed.append({"step": step_name, "params": kwargs, "error": str(exc)})

    return {
        "session_id": sid,
        "config": path,
        "corpus": session_info,
        "output_dir": str(session.output_dir),
        "n_executed": len(executed),
        "executed": executed,
        "failed": failed,
        "notes": ["The corpus stays open -- pass this session_id to any tool to "
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
