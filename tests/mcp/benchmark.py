"""What each MCP tool costs, measured rather than guessed.

The tools in ``loglead/mcp/server.py`` span five orders of magnitude in cost --
``describe_log_root`` is microseconds, ``anomaly_folder_content(target_folder="ALL")``
on 5,000 log folders is an hour -- and a model driving them over MCP cannot tell
which is which from a tool schema. This script produces the numbers that go into
their docstrings, and the scaling rules that let a caller predict a call it has
not made.

Run it with::

    uv run tests/mcp/benchmark.py                    # all three log roots
    uv run tests/mcp/benchmark.py --only hadoop      # one of them
    uv run tests/mcp/benchmark.py --repeat 5         # steadier medians
    uv run tests/mcp/benchmark.py --markdown cost.md # table to paste into docs

There are **two benchmarks here, asking different questions.** The default is a
sweep of *arguments*: one tool at a time, one argument moved, fitting
``cost = fixed + marginal * n`` -- what makes a call expensive. ``--grid`` is a
sweep of *data*: every tool at one canonical call, on each log root at 5, 10, 50
and 100% of it -- what the whole API costs on a log root of a given size. The
first writes ``COST.md``, the second ``PERFORMANCE.md``::

    uv run tests/mcp/benchmark.py --grid                     # the four fraction tables
    uv run tests/mcp/benchmark.py --grid --only hadoop       # one log root
    uv run tests/mcp/benchmark.py --grid --fractions 0.05    # one fraction
    uv run tests/mcp/benchmark.py --tables-only              # rebuild tables, measure nothing

The grid runs each (log root, fraction) in a **child process** and records one
JSON file per measured cell, because on bgl at 50% and 100% the anomaly cells
are killed by the OOM killer rather than raising -- the parent records that as
``OOM`` (which is the measurement) and relaunches the block to finish the rest.
That also makes the grid resumable: a re-run skips every cell already on disk.

**Two numbers per row, and the difference is the point.** ``first`` is a cold
call and ``warm`` the median of the repeats. Where they differ, the gap is a
column the session computed once and kept -- ``e_words`` for a ``Words`` call,
``e_event_drain_id`` for ``Parse-Drain`` -- so the first row of each content
format is the price of that representation and the rest is what the analysis
itself costs. That is the session model's entire justification, and it is only
visible as two numbers.

**The per-unit column is what makes this predictive.** Nearly every tool here is
linear in something a caller chooses: comparison log folders, target log folders,
files. The ``anomaly_*`` tools fit one model *per target*, so their per-target
cost is the number that matters -- ``target_folder="ALL"`` reads as innocuous and
is the most expensive thing in the API.

The log roots are the ones ``tests/mcp/server.py`` uses; see ``make_test_data.py``.
``bgl`` is the third shape and the odd one: a plain loghub download rather than a
derived corpus, and a single 743 MB file that ``split_log_file`` turns into ten
log folders of ~471,000 lines each. It is where a cost charged per log folder is
at its worst, and it is skipped if the download is not there.
Timings are hardware- and load-dependent, so treat the ratios as the durable part
and re-run for absolute numbers.
"""

import argparse
import fnmatch
import gc
import json
import resource
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import polars as pl
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_test_data  # noqa: E402  (sits next to this file)
from loglead import loaders  # noqa: E402
from loglead.delta import split  # noqa: E402

try:
    from loglead.mcp import server  # noqa: E402
    from loglead.mcp.session import SessionStore  # noqa: E402
except ImportError as exc:
    raise SystemExit(
        f"Cannot import loglead.mcp ({exc}). It is an optional install: uv sync --extra mcp"
    ) from exc

#: Cost bands, for the "class" column. A caller does not need three digits; it
#: needs to know whether a call is free, worth a pause, or worth avoiding.
BANDS = ((0.1, "instant"), (1.0, "fast"), (10.0, "slow"), (float("inf"), "very slow"))


def band(seconds):
    return next(name for limit, name in BANDS if seconds < limit)


class Report:
    """Collects one row per measurement and prints them as a table."""

    def __init__(self):
        self.rows = []
        self.models = []

    def add(self, tool, dataset, variant, first, warm, n=None, unit=None):
        self.rows.append({
            "tool": tool, "dataset": dataset, "variant": variant,
            "first_s": round(first, 3), "warm_s": round(warm, 3),
            "n": n, "unit": unit,
            "per_unit_s": round(warm / n, 4) if n else None,
            "class": band(warm),
        })
        per = f"  {warm / n:8.4f} s/{unit}" if n else ""
        print(f"  {tool:<28} {variant:<34} first {first:7.2f}s  warm {warm:7.2f}s"
              f"  {band(warm):<9}{per}")
        return self.rows[-1]

    def scaling(self, rows, unit, label=None):
        """Fit ``cost = fixed + marginal * n`` over a series of measurements.

        A single per-unit number is misleading wherever a tool does something
        once and then something per item -- ``distance_file_content`` aggregates
        every log folder's text before it looks at the first file, so its
        "per file" cost at one file is really the setup. Two points separate the
        two, and separating them is what lets a caller predict a call it has not
        made.
        """
        points = [(row["n"], row["warm_s"]) for row in rows if row["n"]]
        if len(points) < 2:
            return None
        mean_x = sum(x for x, _ in points) / len(points)
        mean_y = sum(y for _, y in points) / len(points)
        spread = sum((x - mean_x) ** 2 for x, _ in points)
        marginal = (sum((x - mean_x) * (y - mean_y) for x, y in points) / spread
                    if spread else 0.0)
        fixed = mean_y - marginal * mean_x
        model = {
            "tool": rows[0]["tool"], "dataset": rows[0]["dataset"],
            "label": label or rows[0]["tool"], "unit": unit,
            "fixed_s": round(max(fixed, 0.0), 2), "marginal_s": round(marginal, 4),
            "measured_at": [x for x, _ in points],
        }
        self.models.append(model)
        print(f"    -> cost ~ {model['fixed_s']:.2f}s + {model['marginal_s']:.3f}s per "
              f"{unit} (measured at {', '.join(str(x) for x in model['measured_at'])})")
        return model

    def note(self, text):
        print(f"    -> {text}")

    def section(self, title):
        print(f"\n{'=' * 100}\n {title}\n{'=' * 100}")

    def markdown(self):
        lines = ["| tool | log root | call | first (s) | warm (s) | class | per unit |",
                 "|---|---|---|---|---|---|---|"]
        for row in self.rows:
            per = f"{row['per_unit_s']} s/{row['unit']}" if row["n"] else ""
            lines.append(
                f"| `{row['tool']}` | {row['dataset']} | {row['variant']} | "
                f"{row['first_s']} | {row['warm_s']} | {row['class']} | {per} |"
            )
        return "\n".join(lines) + "\n"


def measure(call, repeat=3):
    """Time a call cold, then take the median of `repeat` warm runs."""
    started = time.perf_counter()
    call()
    first = time.perf_counter() - started

    warm = []
    for _ in range(max(1, repeat)):
        started = time.perf_counter()
        call()
        warm.append(time.perf_counter() - started)
    return first, statistics.median(warm)


# --------------------------------------------------------------------------- #

def bench_hadoop(report, log_root, repeat):
    """55 log folders, 978 files, 181k lines -- the multi-file shape."""
    report.section("hadoop_renamed -- 55 log folders, 978 files, 180,897 lines")
    sid = "bench-hadoop"

    # Opening is the one place a cold call is the *interesting* one: everything
    # else in the API is paid once here, and the cache is why it is paid once.
    started = time.perf_counter()
    info = server.open_log_root(path=str(log_root), format="auto", mask=True, parsers=["tip"],
                                file_name_normalizer="strip_folder_id", session_id=sid,
                                refresh=True)
    cold = time.perf_counter() - started
    started = time.perf_counter()
    server.open_log_root(path=str(log_root), format="auto", mask=True, parsers=["tip"],
                         file_name_normalizer="strip_folder_id", session_id=sid + "-2")
    cached = time.perf_counter() - started
    server.close_log_root(sid + "-2")
    report.add("open_log_root", "hadoop", "read + mask + parse tip (978 files)",
               cold, cached, n=info["n_files_on_disk"], unit="file")
    report.note(f"the parquet cache turns {cold:.0f}s into {cached:.2f}s -- "
                f"{cold / max(cached, 1e-6):.0f}x, and it is why sessions exist")

    # What the sampled format detection saves, which is the whole difference
    # between the two rows: same files, same frame, every file probed instead of
    # 50. Measured rather than argued, because it is the one knob a client would
    # otherwise have to guess the price of.
    started = time.perf_counter()
    server.open_log_root(path=str(log_root), format="auto", mask=True, parsers=["tip"],
                         file_name_normalizer="strip_folder_id", session_id=sid + "-every",
                         refresh=True, max_detect_files=0)
    every = time.perf_counter() - started
    server.close_log_root(sid + "-every")
    report.add("open_log_root", "hadoop", "same, max_detect_files=0 (probe all 978)",
               every, every, n=info["n_files_on_disk"], unit="file")
    report.note(f"probing every file instead of {loaders.DEFAULT_MAX_DETECT_FILES} costs "
                f"{every - cold:.0f}s more here, {every / max(cold, 1e-6):.1f}x, "
                f"for the same frame")

    folders = server.STORE.get(sid).folders
    target = folders[0]
    described = server.describe_log_root(sid, include_files=True)
    shared = max(described["files_detail"], key=lambda row: row["n_folders"])["file_name"]

    # -- session introspection: the free calls --
    report.add("list_log_roots", "hadoop", "1 open session",
               *measure(lambda: server.list_log_roots(), repeat))
    report.add("describe_log_root", "hadoop", "55 log folders",
               *measure(lambda: server.describe_log_root(sid), repeat))
    report.add("describe_log_root", "hadoop", "include_files=True",
               *measure(lambda: server.describe_log_root(sid, include_files=True), repeat))

    # -- reading raw text --
    report.add("read_log_lines", "hadoop", "100 lines",
               *measure(lambda: server.read_log_lines(sid, target, shared, limit=100), repeat))
    report.add("search_log_lines", "hadoop", "regex over 181k lines",
               *measure(lambda: server.search_log_lines(sid, r"Going to preempt"), repeat))
    report.add("search_log_lines", "hadoop", "literal over 181k lines",
               *measure(lambda: server.search_log_lines(sid, "org.apache", regex=False), repeat))

    # -- distance, L1 to L4 --
    report.scaling([
        report.add("distance_folder_filename", "hadoop", f"comparison_folders={n}",
                   *measure(lambda n=n: server.distance_folder_filename(
                       sid, target, comparison_folders=n), repeat),
                   n=n, unit="folder")
        for n in (5, 27, 54)], "comparison folder")
    report.scaling([
        report.add("distance_folder_content", "hadoop", f"comparison_folders={n}, Words",
                   *measure(lambda n=n: server.distance_folder_content(
                       sid, target, comparison_folders=n, content_format="Words"), repeat),
                   n=n, unit="folder")
        for n in (5, 27, 54)], "comparison folder")
    report.note("all four measures run per pair, and compression distance is three bz2 "
                "compressions of each pair (target, comparison, and the two concatenated) -- "
                "the target's own text is re-compressed once per comparison folder")
    report.scaling([
        report.add("distance_file_content", "hadoop", f"target_files={files}, comparison=ALL",
                   *measure(lambda files=files: server.distance_file_content(
                       sid, target, comparison_folders="ALL", target_files=files,
                       content_format="Words"), repeat),
                   n=files, unit="file")
        for files in (1, 3, 5)], "target file")
    report.note("the fixed part is aggregating every log folder's text before the first "
                "file is looked at, so asking for one file costs nearly what asking for five does")
    report.scaling([
        report.add("distance_line_content", "hadoop", f"1 file x {n} comparison folders",
                   *measure(lambda n=n: server.distance_line_content(
                       sid, target, comparison_folders=n, target_files=[shared]), repeat),
                   n=n, unit="diff")
        for n in (1, 5)], "diff")

    # -- anomaly: one model fit per target, which is the whole story --
    report.scaling([
        report.add("anomaly_folder_filename", "hadoop", f"target_folder={n}, comparison=ALL",
                   *measure(lambda n=n: server.anomaly_folder_filename(
                       sid, target_folder=n, comparison_folders="ALL"), repeat),
                   n=n, unit="target")
        for n in (1, 5, 55)], "target folder")
    model = report.scaling([
        report.add("anomaly_folder_content", "hadoop", f"target_folder={n}, comparison=ALL",
                   *measure(lambda n=n: server.anomaly_folder_content(
                       sid, target_folder=n, comparison_folders="ALL",
                       content_format="Words"), repeat),
                   n=n, unit="target")
        for n in (1, 5, 55)], "target folder")
    report.note(f"{model['marginal_s']:.2f}s per target, because the four detectors are "
                f'refitted for each one: target_folder="ALL" reads as innocuous and is the '
                "most expensive thing in the API")
    report.scaling([
        report.add("anomaly_file_content", "hadoop", f"target_files={files}, comparison=ALL",
                   *measure(lambda files=files: server.anomaly_file_content(
                       sid, target, comparison_folders="ALL", target_files=files,
                       content_format="Words"), repeat),
                   n=files, unit="file")
        for files in (1, 3, 5)], "target file")
    report.add("anomaly_line_content", "hadoop", "1 file, scores every line + writes a plot",
               *measure(lambda: server.anomaly_line_content(
                   sid, target, comparison_folders="ALL", target_files=[shared],
                   content_format="Words"), repeat))

    # -- content formats: the first call buys the column, the rest reuse it --
    report.section("hadoop_renamed -- what each content_format costs")
    report.note("first - warm is the representation being computed over 181k lines and kept; "
                "the comparison set is held at 10 folders so this measures the format, "
                "not the O(n) distance above")
    for content_format in ("Words", "3grams", "Sklearn", "Parse-Tip", "Parse-Drain"):
        report.add("distance_folder_content", "hadoop", f"comparison=10, {content_format}",
                   *measure(lambda cf=content_format: server.distance_folder_content(
                       sid, target, comparison_folders=10, content_format=cf), repeat))

    # -- plots: the UMAP is the opt-in half --
    report.section("hadoop_renamed -- plots")
    report.add("plot_folder_filename", "hadoop", 'plots=["scatter"] (default)',
               *measure(lambda: server.plot_folder_filename(sid, target), repeat))
    # The scatter alone is cheap enough to sweep like distance_folder_content, so
    # it gets a fitted cost model too rather than one number at one size.
    report.scaling([
        report.add("plot_folder_content", "hadoop",
                   f"comparison_folders={n}, plots=[scatter]",
                   *measure(lambda n=n: server.plot_folder_content(
                       sid, target, comparison_folders=n, content_format="Words"), repeat),
                   n=n, unit="folder")
        for n in (5, 27, 54)], "comparison folder")
    umap = report.add("plot_folder_content", "hadoop", 'plots=["umap","scatter"]',
                      *measure(lambda: server.plot_folder_content(
                          sid, target, content_format="Words", random_seed=42,
                          plots=["umap", "scatter"]), min(repeat, 2)))
    report.note(f"the first UMAP in a process also pays numba's JIT: {umap['first_s']}s "
                f"against {umap['warm_s']}s once warm")
    report.add("plot_file_content", "hadoop", '1 file, plots=["scatter"]',
               *measure(lambda: server.plot_file_content(
                   sid, target, target_files=[shared], content_format="Words"), repeat))
    report.add("plot_file_content", "hadoop", '1 file, plots=["umap","scatter"]',
               *measure(lambda: server.plot_file_content(
                   sid, target, target_files=[shared], content_format="Words",
                   random_seed=42, plots=["umap", "scatter"]), min(repeat, 2)))

    # -- query_result: the alternative to re-running any of the above --
    scored = server.anomaly_folder_content(sid, target_folder="ALL", comparison_folders="ALL",
                                           content_format="Words")
    query = report.add("query_result", "hadoop", "filter + sort a 55-row result",
                       *measure(lambda: server.query_result(
                           sid, scored["result_id"], where=[["rank_sum", ">", 10]],
                           sort_by="rank_sum"), repeat))
    report.note("asking a different question of the same table costs "
                f"{query['warm_s']:.4f}s; re-running the analysis to ask it costs the "
                "anomaly_folder_content row above")

    server.close_log_root(sid)


def bench_hdfs(report, log_root, repeat):
    """5,000 single-file log folders -- where the selectors get dangerous."""
    report.section(f"hdfs_balanced_5k -- 5,000 log folders, "
                   f"{make_test_data.EXPECTED_LINES:,} lines")
    sid = "bench-hdfs"

    started = time.perf_counter()
    info = server.open_log_root(path=str(log_root), format="auto", mask=True,
                                parsers=["tip"], session_id=sid, refresh=True)
    cold = time.perf_counter() - started
    started = time.perf_counter()
    server.open_log_root(path=str(log_root), format="auto", mask=True, parsers=["tip"],
                         session_id=sid + "-2")
    cached = time.perf_counter() - started
    server.close_log_root(sid + "-2")
    report.add("open_log_root", "hdfs", "read + mask + parse tip (5,000 files)",
               cold, cached, n=info["n_files_on_disk"], unit="file")
    report.note(f"the format is detected from {loaders.DEFAULT_MAX_DETECT_FILES} of the 5,000 "
                f"files and applied to the rest; the cached open is {cached:.2f}s")

    started = time.perf_counter()
    server.open_log_root(path=str(log_root), format="auto", mask=True, parsers=["tip"],
                         session_id=sid + "-every", refresh=True, max_detect_files=0)
    every = time.perf_counter() - started
    server.close_log_root(sid + "-every")
    report.add("open_log_root", "hdfs", "same, max_detect_files=0 (probe all 5,000)",
               every, every, n=info["n_files_on_disk"], unit="file")
    report.note(f"a format probe per file is {every - cold:.0f}s of it -- "
                f"{every / max(cold, 1e-6):.0f}x the sampled open, for the same frame. This is "
                f"the log root shape that makes the sampling worth having: one file per unit")

    folders = server.STORE.get(sid).folders
    anomalies = [name for name in folders if name.startswith("Anomaly_")]
    target = anomalies[0]

    report.add("describe_log_root", "hdfs", "5,000 log folders",
               *measure(lambda: server.describe_log_root(sid), repeat))
    report.add("describe_log_root", "hdfs", "include_files=True (5,000 files)",
               *measure(lambda: server.describe_log_root(sid, include_files=True), repeat))
    report.add("search_log_lines", "hdfs", "regex over 91k lines",
               *measure(lambda: server.search_log_lines(sid, r"Exception"), repeat))

    report.add("distance_folder_filename", "hdfs", "comparison_folders=ALL (4,999)",
               *measure(lambda: server.distance_folder_filename(sid, target), repeat),
               n=4999, unit="folder")
    report.add("distance_folder_content", "hdfs", "comparison_folders=ALL (4,999), Words",
               *measure(lambda: server.distance_folder_content(
                   sid, target, content_format="Words"), repeat),
               n=4999, unit="folder")

    # The scaling that matters most in the whole API.
    model = report.scaling([
        report.add("anomaly_folder_content", "hdfs",
                   f'target_folder={n}, comparison="Normal_*"',
                   *measure(lambda n=n: server.anomaly_folder_content(
                       sid, target_folder=anomalies[:n], comparison_folders="Normal_*",
                       content_format="Words"), repeat),
                   n=n, unit="target")
        for n in (1, 5, 20)], "target folder")
    if model:
        whole = model["fixed_s"] + 5000 * model["marginal_s"]
        report.note(f'target_folder="ALL" here is 5,000 targets = {whole / 60:.0f} minutes. '
                    "Nothing in the call says so, and the default is ALL.")

    report.section("hdfs_balanced_5k -- plots at 5,000 points")
    report.add("plot_folder_filename", "hdfs", 'plots=["scatter"] (default)',
               *measure(lambda: server.plot_folder_filename(sid, target), repeat))
    # 4999 is comparison_folders="ALL" here, so this sweep's last point is also
    # the default-arguments measurement -- no need to run it a second time.
    swept = [
        report.add("plot_folder_content", "hdfs",
                   f"comparison_folders={n}, plots=[scatter]" + (" (= ALL)" if n == 4999 else ""),
                   *measure(lambda n=n: server.plot_folder_content(
                       sid, target, comparison_folders=n, content_format="Words"), repeat),
                   n=n, unit="folder")
        for n in (50, 500, 4999)
    ]
    report.scaling(swept, "comparison folder")
    scatter = swept[-1]
    # One repeat: this is the most expensive call in the benchmark and the point
    # of measuring it is the order of magnitude, not the third digit.
    umap = report.add("plot_folder_content", "hdfs", 'plots=["umap","scatter"]',
                      *measure(lambda: server.plot_folder_content(
                          sid, target, content_format="Words", random_seed=42,
                          plots=["umap", "scatter"]), 1))
    report.note(f"the UMAP is {umap['warm_s'] / max(scatter['warm_s'], 1e-6):.0f}x the default "
                f"view ({umap['warm_s']:.1f}s vs {scatter['warm_s']:.1f}s), and the default "
                f"view never reads its output -- which is why it is opt-in")
    no_seed = report.add("plot_folder_content", "hdfs",
                         'plots=["umap"], random_seed=None (threaded)',
                         *measure(lambda: server.plot_folder_content(
                             sid, target, content_format="Words", random_seed=None,
                             plots=["umap"]), 1))
    report.note(f"random_seed makes umap-learn single-threaded: {umap['warm_s']:.1f}s seeded "
                f"vs {no_seed['warm_s']:.1f}s unseeded -- the price of a reproducible layout")

    points = server.plot_folder_content(sid, target, content_format="Words")
    report.add("query_result", "hdfs", "filter + sort a 5,000-row result",
               *measure(lambda: server.query_result(
                   sid, points["result_id"], where=[["lines", "<", 5]], sort_by="lines"), repeat))
    report.note("this is what replaces re-running the analysis with different arguments")

    server.close_log_root(sid)


def peak_rss_gb():
    """Process high-water memory, in GB. Linux reports ru_maxrss in kilobytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def current_rss_gb():
    """Resident memory *now*, in GB.

    Distinct from :func:`peak_rss_gb` on purpose: the peak is a high-water mark
    that never comes down, so it reports a transient double-allocation forever.
    What a caller wants to know about a session is what it is still holding.
    """
    try:
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (OSError, ValueError, IndexError):
        return peak_rss_gb()


def bench_bgl(report, source, workdir, repeat):
    """One 743 MB log file, split into 10 slices -- the third shape, and the one
    where memory rather than the clock is what stops a call.

    Hadoop and HDFS differ in how *many* log folders there are. BGL differs in
    how big one is: ten slices of ~471,000 lines each, against Hadoop's ~3,300
    and HDFS's ~18. Every cost charged per log folder is at its worst here, and
    two of them stop fitting in a 16 GB machine rather than merely taking a
    while -- which is why this function reports peak RSS and the other two do
    not.

    Heavy rows are measured once rather than three times: one ``Words`` call at
    nine comparison folders is two and a half minutes, and its third digit is
    not what a caller needs.
    """
    report.section("bgl -- one 743 MB log file, split into 10 slices")
    sid = "bench-bgl"

    # -- the two tools that come before a session exists --
    report.add("peek_log_root", "bgl", "the unsplit 743 MB file",
               *measure(lambda: server.peek_log_root(str(source)), repeat))
    report.note("peek stats the file and samples a few hundred lines; it never reads it, so a "
                "743 MB log costs what a small one does")

    out_dir = os.path.join(workdir, "bgl-slices")
    started = time.perf_counter()
    server.split_log_file(str(source), n_slices=10, out_dir=out_dir)
    first_split = time.perf_counter() - started
    # Every later call finds the manifest beside the slices and reuses them, so
    # the cold number has to be taken by hand rather than from measure().
    _, reused = measure(lambda: server.split_log_file(
        str(source), n_slices=10, out_dir=out_dir), repeat)
    report.add("split_log_file", "bgl", "743 MB -> 10 slices, by=lines",
               first_split, reused, n=743, unit="MB")
    report.note(f"{first_split:.1f}s to write 743 MB of slices, and {reused:.2f}s to be handed "
                "the same split again -- the slices are on disk and a manifest beside them says "
                "what they are")

    # The other two log roots measure the cached open by holding a second session
    # open beside the first. Here that would mean two 3 GB frames at once, which
    # is 6.5 GB of floor before a single analysis runs -- and this is the log
    # root with no headroom to spare. So the cold session is closed and reopened
    # instead: same measurement, one frame at a time.
    started = time.perf_counter()
    info = server.open_log_root(path=out_dir, format="auto", mask=True, parsers=["tip"],
                                session_id=sid, refresh=True)
    cold = time.perf_counter() - started
    server.close_log_root(sid)
    gc.collect()
    started = time.perf_counter()
    server.open_log_root(path=out_dir, format="auto", mask=True, parsers=["tip"],
                         session_id=sid)
    cached = time.perf_counter() - started
    report.add("open_log_root", "bgl", "read + mask + parse tip (10 files, 4.7M lines)",
               cold, cached, n=info["n_rows"], unit="line")
    report.note(f"there are only 10 files to probe, so unlike the other two log roots almost "
                f"none of the {cold:.0f}s is format detection -- it is masking and parsing 4.7M "
                f"lines. The frame it leaves behind is {current_rss_gb():.1f} GB resident, which "
                f"is the floor every row below sits on")

    folders = server.STORE.get(sid).folders
    target = folders[0]
    # Every log folder here holds one file with a name of its own, so the file to
    # read has to come from the target's own row rather than from a list of file
    # names shared across folders -- there are none.
    file_of = dict(server.STORE.get(sid).df
                   .select(["folder", "file_name"]).unique().iter_rows())
    only_file = file_of[target]

    report.add("describe_log_root", "bgl", "10 log folders",
               *measure(lambda: server.describe_log_root(sid), repeat))
    report.add("read_log_lines", "bgl", "100 lines out of 471k",
               *measure(lambda: server.read_log_lines(sid, target, only_file, limit=100), repeat))
    report.add("search_log_lines", "bgl", "regex over 4.7M lines",
               *measure(lambda: server.search_log_lines(sid, r"kernel panic"), repeat))
    report.note("scanning every line is the one cost here that scales with the log rather than "
                "with the number of log folders, and it is still the cheapest row in the table")

    # -- L1: degenerate on this shape, one file per log folder --
    report.add("distance_folder_filename", "bgl", "comparison_folders=ALL (9)",
               *measure(lambda: server.distance_folder_filename(
                   sid, target, comparison_folders="ALL"), repeat),
               n=9, unit="folder")

    # -- L3/L4: not applicable to this shape, and cheap enough to find out --
    empty = report.add("distance_file_content", "bgl", "target_files=ALL, comparison=ALL",
                       *measure(lambda: server.distance_file_content(
                           sid, target, comparison_folders="ALL", target_files="ALL",
                           content_format="Parse-Tip"), repeat))
    report.note(f"{empty['warm_s']:.1f}s to return zero rows: every slice is one uniquely-named "
                "file in its own log folder, so no file name is shared and the file- and "
                "line-level tools have nothing to match on. Use the folder-level tools instead")

    # -- anomaly first, not last: it is the heaviest thing here in memory, and
    # -- on this log root that is what decides whether a call runs at all.
    report.section("bgl -- anomaly, where this log root stops fitting in memory")
    # Deliberately one point, not a sweep. Every other cost model in this file
    # comes from calling a tool at two or three sizes in one process; that is not
    # possible here. anomaly_folder_content holds roughly a gigabyte per target
    # log folder on top of a 3 GB session, and a sweep to 4 targets was killed by
    # the OOM killer three times before this became the measurement. The rate
    # below is the honest one: two points from separate processes.
    single = report.add("anomaly_folder_content", "bgl", "target_folder=1, comparison=ALL",
                        *measure(lambda: server.anomaly_folder_content(
                            sid, target_folder=[folders[0]], comparison_folders="ALL",
                            content_format="Words"), 1),
                        n=1, unit="target")
    report.note(f"one target is {single['warm_s']:.0f}s. Measured on its own in a fresh process, "
                'target_folder="ALL" -- the default -- is 76s and peaks at 14.6 GB, so the rate '
                "is about 7s per target log folder. There is no fitted model for this row "
                "because a sweep cannot be run: 4 targets in the same process as anything else "
                "is an out-of-memory kill on a 16 GB machine. On this log root the default "
                "argument is not merely slow")
    report.note(f"RSS here: {current_rss_gb():.1f} GB held, {peak_rss_gb():.1f} GB peak")

    report.section("bgl -- distance")
    # -- L2: the level a split file is actually compared at --
    report.scaling([
        report.add("distance_folder_content", "bgl",
                   f"comparison_folders={n}, Words" + (" (= ALL)" if n == 9 else ""),
                   *measure(lambda n=n: server.distance_folder_content(
                       sid, target, comparison_folders=n, content_format="Words"), 1),
                   n=n, unit="folder")
        for n in (1, 3, 9)], "comparison folder")
    report.note("each comparison folder is ~471k lines to vectorize and compress, which is what "
                "puts the marginal cost two orders of magnitude above Hadoop's 0.285s per folder")

    # -- content formats, at a comparison count this log root can afford --
    report.section("bgl -- what each content_format costs")
    report.note("held at 3 comparison folders, not Hadoop's 10, because Words alone is 38s at "
                "that size here. 3grams is not in this sweep: measured on its own it is 187s and "
                "11.6 GB at these same 3 comparison folders, which runs the benchmark out of "
                "memory when anything else has run first. It fits in a fresh process and "
                "nowhere else")
    for content_format in ("Parse-Tip", "Words", "Sklearn"):
        report.add("distance_folder_content", "bgl", f"comparison=3, {content_format}",
                   *measure(lambda cf=content_format: server.distance_folder_content(
                       sid, target, comparison_folders=3, content_format=cf), 1))

    # -- plots --
    report.section("bgl -- plots")
    scatter = report.add("plot_folder_content", "bgl",
                         'comparison_folders=ALL (9), plots=["scatter"]',
                         *measure(lambda: server.plot_folder_content(
                             sid, target, comparison_folders="ALL", content_format="Words"), 1),
                         n=9, unit="folder")
    umap = report.add("plot_folder_content", "bgl", 'plots=["umap","scatter"]',
                      *measure(lambda: server.plot_folder_content(
                          sid, target, comparison_folders="ALL", content_format="Words",
                          random_seed=42, plots=["umap", "scatter"]), 1))
    report.note(f"the UMAP is free here ({umap['warm_s']:.1f}s against the scatter's "
                f"{scatter['warm_s']:.1f}s) -- the opposite of HDFS, where it is 32x the "
                "scatter. It lays out one point per log folder and there are ten of them, so "
                "what both calls actually pay for is the term matrix over 4.7M lines. The gap "
                f"between its first call ({umap['first_s']:.1f}s) and its warm one is numba "
                "compiling the layout code")

    scored = server.plot_folder_content(sid, target, comparison_folders="ALL",
                                        content_format="Words")
    report.add("query_result", "bgl", "filter + sort a 10-row result",
               *measure(lambda: server.query_result(
                   sid, scored["result_id"], where=[["lines", ">", 1]], sort_by="lines"), repeat))

    server.close_log_root(sid)


# --------------------------------------------------------------------------- #
# The fraction grid: every tool, every log root, at 5 / 10 / 50 / 100% of data
# --------------------------------------------------------------------------- #
#
# The sweeps above answer "what is this call proportional to" by moving one
# argument at a time. This second half answers a different question -- "what
# does the whole API cost on a log root of this size" -- by holding the
# arguments at one canonical call per tool and moving the *data* instead. It
# produces tests/mcp/PERFORMANCE.md.
#
# What a fraction means differs by log root, because the two shapes differ in
# what makes them big. hadoop_renamed and hdfs_balanced_5k are many log folders,
# so a fraction of them is a fraction of the *log folders*. bgl_split_10 is ten
# large ones, so a fraction there is a fraction of BGL.log's *lines*, taken
# before the split -- the slice count stays 10 and what shrinks is the text
# inside each. Reducing bgl by folders instead would leave one or two slices at
# 5%, which is not a smaller version of the same shape.

#: The fractions every tool is measured at.
FRACTIONS = (0.05, 0.10, 0.50, 1.00)

#: A cold call slower than this gets one warm repeat instead of ``--repeat``.
#: One rule instead of a hand-maintained list of heavy cells: the third digit of
#: a two-minute call is not what anyone reads, and on bgl the repeats are what
#: turn a slow grid into an overnight one.
HEAVY_SECONDS = 20.0

#: Log root names as they appear in PERFORMANCE.md.
ROOT_LABELS = {"hadoop": "hadoop_renamed", "hdfs": "hdfs_balanced_5k",
               "bgl": "bgl_split_10"}


def fraction_tag(fraction):
    """``0.05`` -> ``"005pct"``, for directory and file names."""
    return f"{round(fraction * 100):03d}pct"


def log_folders_on_disk(root, filename_pattern="*.log"):
    """``{log folder name: [file paths]}`` for one log root, without loading it.

    Mirrors what ``read_log_root`` treats as a log folder: a subdirectory of the
    log root, or a log file sitting directly in it. Anything else at the top
    level -- Hadoop's ``abnormal_label.txt`` -- is not a log folder and is left
    out of the reduced copy.
    """
    folders = {}
    for entry in sorted(os.scandir(root), key=lambda item: item.name):
        if entry.is_dir():
            files = sorted(str(path) for path in Path(entry.path).rglob(filename_pattern))
            if files:
                folders[entry.name] = files
        elif entry.is_file() and fnmatch.fnmatch(entry.name, filename_pattern):
            folders[entry.name] = [entry.path]
    return folders


def select_fraction(names, fraction):
    """A *spread* of ``fraction`` of ``names``, not the first N of them.

    hdfs_balanced_5k's log folders are called ``Anomaly_blk_…`` and
    ``Normal_blk_…``, so the first 5% of the sorted names is 250 anomalies and no
    normal log folder at all -- a reduced log root with nothing to compare
    against. Striding keeps the class balance, and being deterministic keeps the
    same 5% across runs.
    """
    names = list(names)
    take = max(1, round(len(names) * fraction))
    if take >= len(names):
        return names
    step = len(names) / take
    return [names[min(len(names) - 1, int(index * step))] for index in range(take)]


def _reduction_marker(dest):
    """Beside the reduced log root, not inside it: a stray file in a log root is
    a file some later walk has to know to ignore."""
    return str(dest).rstrip("/") + ".reduction.json"


def reduce_folder_root(source, dest, fraction, filename_pattern="*.log"):
    """Build a smaller log root holding a spread of ``fraction`` of the folders.

    Files are **hard-linked**, not copied: a 5,000-folder reduction costs
    directory entries rather than bytes, and ``stat`` still reports the real
    size, so the session fingerprint and ``peek_log_root``'s byte counts are the
    same numbers the full log root would give. Falls back to a copy across
    devices, as ``make_test_data.link_or_copy`` does.

    Idempotent: a marker file records what was built, so a re-run -- or the
    relaunch after an out-of-memory kill -- reuses the reduction instead of
    rebuilding it.
    """
    folders = log_folders_on_disk(source, filename_pattern)
    if not folders:
        raise SystemExit(f"No log folders matching {filename_pattern} under {source}")
    chosen = select_fraction(sorted(folders), fraction)
    n_files = sum(len(folders[name]) for name in chosen)
    marker = {"source": str(source), "fraction": fraction,
              "n_folders": len(chosen), "n_files": n_files}

    existing = _reduction_marker(dest)
    if os.path.isfile(existing):
        with open(existing) as handle:
            if json.load(handle) == marker:
                return dest, len(chosen), n_files
    shutil.rmtree(dest, ignore_errors=True)

    for name in chosen:
        for path in folders[name]:
            relative = os.path.relpath(path, source)
            target = os.path.join(dest, relative)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            make_test_data.link_or_copy(path, target)
    with open(existing, "w") as handle:
        json.dump(marker, handle)
    return dest, len(chosen), n_files


def count_file_lines(path):
    """Lines in one file, read in 8 MB blocks. ~2.6s on BGL's 743 MB."""
    total = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 23)
            if not block:
                return total
            total += block.count(b"\n")


def reduce_line_file(source, dest, fraction, total_lines):
    """Write the first ``fraction`` of ``source``'s lines to ``dest``, streaming.

    Never holds the file in memory -- the same reason ``delta/split.py``
    streams -- so reducing a 70 GB log would cost what reducing this one does.
    At ``fraction == 1.0`` the source is used as it stands rather than copied.
    """
    take = max(1, round(total_lines * fraction))
    if take >= total_lines:
        return source, total_lines
    marker = f"{dest}.lines.json"
    if os.path.isfile(dest) and os.path.isfile(marker):
        with open(marker) as handle:
            if json.load(handle) == {"source": str(source), "n_lines": take}:
                return dest, take

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    written = 0
    with open(source, "rb") as reader, open(dest, "wb") as writer:
        for line in reader:
            writer.write(line)
            written += 1
            if written >= take:
                break
    with open(marker, "w") as handle:
        json.dump({"source": str(source), "n_lines": take}, handle)
    return dest, written


def prepare_grid_root(kind, fraction, datasets_folder, paths, reduced_folder):
    """The log root to benchmark ``kind`` at ``fraction``, built if need be.

    Returns ``(path, shape)`` where shape is what the log root turned out to
    be -- the numbers that fill PERFORMANCE.md's first table.
    """
    os.makedirs(reduced_folder, exist_ok=True)
    tag = fraction_tag(fraction)

    if kind in ("hadoop", "hdfs"):
        name = make_test_data.HADOOP_RENAMED if kind == "hadoop" else make_test_data.HDFS_BALANCED_5K
        source = paths[name]
        if fraction >= 1.0:
            folders = log_folders_on_disk(source)
            return str(source), {"n_folders": len(folders),
                                 "n_files": sum(len(f) for f in folders.values())}
        path, n_folders, n_files = reduce_folder_root(
            str(source), os.path.join(reduced_folder, f"{name}_{tag}"), fraction)
        return path, {"n_folders": n_folders, "n_files": n_files}

    # bgl: a fraction of the lines, then split into ten slices -- the split is
    # what makes a single file into a log root at all.
    source = Path(datasets_folder) / "bgl" / "BGL.log"
    if not source.is_file():
        return None, {}
    counted = os.path.join(reduced_folder, "bgl_total_lines.json")
    if os.path.isfile(counted):
        with open(counted) as handle:
            total = json.load(handle)["n_lines"]
    else:
        total = count_file_lines(str(source))
        with open(counted, "w") as handle:
            json.dump({"n_lines": total}, handle)

    reduced, n_lines = reduce_line_file(
        str(source), os.path.join(reduced_folder, f"bgl_{tag}.log"), fraction, total)
    slices = os.path.join(reduced_folder, f"bgl_{tag}_slices")
    # delta.split returns its manifest rather than writing one -- only the MCP
    # tool writes split_manifest.json -- so the marker that makes this
    # idempotent has to be ours. Without it every OOM relaunch would rewrite
    # 743 MB of slices before measuring anything.
    marker, stamp = slices + ".split.json", {"source": str(reduced),
                                             "n_lines": n_lines, "n_slices": 10}
    built = os.path.isfile(marker) and json.load(open(marker)) == stamp
    if not built:
        shutil.rmtree(slices, ignore_errors=True)
        split.split_log_file(str(reduced), out_dir=slices, n_slices=10, by="lines")
        with open(marker, "w") as handle:
            json.dump(stamp, handle)
    return slices, {"n_folders": 10, "n_files": 10, "n_lines": n_lines,
                    "source_file": str(reduced)}


def measure_adaptive(call, repeat=3, heavy_seconds=HEAVY_SECONDS):
    """:func:`measure`, but a call that turns out to be slow is repeated once.

    Which cells are heavy is a property of the log root, not of the tool -- the
    same ``distance_folder_content`` is 0.3s on Hadoop and minutes on bgl -- so
    the cold call decides, rather than a list maintained by hand.
    """
    started = time.perf_counter()
    call()
    first = time.perf_counter() - started
    times = []
    for _ in range(1 if first > heavy_seconds else max(1, repeat)):
        started = time.perf_counter()
        call()
        times.append(time.perf_counter() - started)
    return first, statistics.median(times)


#: One canonical call per tool, as ``(table, label)`` in the order
#: PERFORMANCE.md prints them. The grid holds the *arguments* still and moves
#: the data, so every row here is one call shape measured at four sizes.
GRID_ROWS = (
    ("aux", "peek_log_root"),
    ("aux", "open_log_root"),
    ("aux", "list_log_roots"),
    ("aux", "describe_log_root"),
    ("aux", "set_folder_names"),
    ("aux", "read_log_lines"),
    ("aux", "search_log_lines"),
    ("aux", "query_result"),
    ("aux", "split_log_file"),
    ("aux", "close_log_root"),
    ("aux", "run_config"),
    ("distance", "distance_folder_filename"),
    ("distance", "distance_folder_content"),
    ("distance", "distance_file_content"),
    ("distance", "distance_line_content"),
    ("anomaly", "anomaly_folder_filename"),
    ("anomaly", "anomaly_folder_content"),
    ("anomaly", "anomaly_file_content"),
    ("anomaly", "anomaly_line_content"),
    ("plot", "plot_folder_filename"),
    ("plot", "plot_folder_content (scatter)"),
    ("plot", "plot_folder_content (scatter+umap)"),
    ("plot", "plot_file_content (scatter)"),
    ("plot", "plot_file_content (scatter+umap)"),
)

GRID_TABLE_TITLES = (("aux", "Auxiliary tools"), ("distance", "Distance tools"),
                     ("anomaly", "Anomaly tools"), ("plot", "Plot tools"))


def open_kwargs(kind, path, session_id):
    """The open call every cell of one log root sits on.

    Hadoop's container logs repeat the log folder id in every file name, so it
    gets ``strip_folder_id`` -- without it no file name is shared between log
    folders and the file-level tools have nothing to match on. The other two log
    roots are one file per log folder and have nothing to strip.
    """
    kwargs = {"path": path, "format": "auto", "mask": True, "parsers": ["tip"],
              "session_id": session_id}
    if kind == "hadoop":
        kwargs["file_name_normalizer"] = "strip_folder_id"
    return kwargs


class GridContext:
    """One log root at one fraction, open, with the cells' arguments chosen.

    The session is opened untimed here and re-opened untimed after an
    out-of-memory relaunch -- the *timed* open is a cell like any other. Cells
    are named rather than positional so a relaunch can skip the ones already
    recorded.
    """

    def __init__(self, kind, fraction, path, workdir, repeat):
        self.kind, self.fraction, self.path = kind, fraction, path
        self.workdir, self.repeat = workdir, repeat
        self.sid = f"grid-{kind}-{fraction_tag(fraction)}"
        self.session = None

    def open(self):
        server.open_log_root(**open_kwargs(self.kind, self.path, self.sid))
        self.session = server.STORE.get(self.sid)
        folders = self.session.folders
        # hdfs_balanced_5k is half anomalies by construction and they sort first;
        # pick one so the anomaly rows score something the detectors can see.
        anomalies = [name for name in folders if name.startswith("Anomaly_")]
        self.target = anomalies[0] if anomalies else folders[0]
        described = server.describe_log_root(self.sid, include_files=True)
        # The file the file-level tools get: the most widely shared name where
        # there is one (Hadoop), and the target's own only file where every log
        # folder holds a uniquely named file (hdfs, bgl).
        shared = max(described["files_detail"], key=lambda row: row["n_folders"])
        if shared["n_folders"] > 1:
            self.file_name = shared["file_name"]
        else:
            self.file_name = (self.session.df
                              .filter(pl.col("folder") == self.target)
                              .select("file_name").row(0)[0])
        self.n_rows = self.session.df.height
        self.n_folders = len(folders)
        return self

    # -- the cells that are not a plain "call it four times" ---------------- #

    def measured_open(self):
        """Cold read against cached re-attach -- the one row where the cold
        number is the interesting one, and the justification for sessions."""
        sid = self.sid + "-cold"
        started = time.perf_counter()
        server.open_log_root(**open_kwargs(self.kind, self.path, sid), refresh=True)
        cold = time.perf_counter() - started
        server.close_log_root(sid)
        gc.collect()
        started = time.perf_counter()
        server.open_log_root(**open_kwargs(self.kind, self.path, sid))
        cached = time.perf_counter() - started
        server.close_log_root(sid)
        gc.collect()
        return cold, cached

    def measured_split(self):
        """Cut the log root's largest file into ten slices.

        Cold writes the slices; warm is handed the manifest beside them, which
        is the reuse path a client hits when it asks for the same split twice.
        """
        files = [path for folder in log_folders_on_disk(self.path).values()
                 for path in folder]
        biggest = max(files, key=os.path.getsize)
        out_dir = os.path.join(self.workdir, "split-cell")
        shutil.rmtree(out_dir, ignore_errors=True)
        started = time.perf_counter()
        server.split_log_file(biggest, n_slices=10, out_dir=out_dir)
        cold = time.perf_counter() - started
        _, warm = measure_adaptive(
            lambda: server.split_log_file(biggest, n_slices=10, out_dir=out_dir),
            self.repeat)
        shutil.rmtree(out_dir, ignore_errors=True)
        return cold, warm

    def measured_close(self):
        """A close is a one-shot: closing an already-closed session is a no-op,
        so this is measured once and the same number stands as both."""
        started = time.perf_counter()
        server.close_log_root(self.sid)
        elapsed = time.perf_counter() - started
        self.session = None
        return elapsed, elapsed

    def measured_run_config(self):
        """A minimal LogDelta config over this log root: one content-distance
        step. Every call re-opens the log root (from cache) and runs the step,
        which is what a config costs a client that has nothing open yet."""
        config_path = os.path.join(self.workdir, f"{self.sid}-config.yml")
        with open(config_path, "w") as handle:
            yaml.safe_dump({
                "input_data_folder": self.path,
                "output_folder": os.path.join(self.workdir, "run-config-output"),
                "regex_masking": {"enabled": True,
                                  "pattern": [{"name": "myllari_extended"}]},
                "steps": {"distance_run_content": [{"target_run": self.target}]},
            }, handle)
        sid = self.sid + "-cfg"

        def call():
            server.run_config(config_path, session_id=sid)
            server.close_log_root(sid)
            gc.collect()

        return measure_adaptive(call, self.repeat)

    def measured_folder_names(self):
        """Naming is applied to the original names every time, so repeating it
        is idempotent -- but it does rewrite the ``folder`` column, so this cell
        runs after every analysis cell and puts the names back afterwards."""
        names = {name: f"Bench{index:04d}"
                 for index, name in enumerate(self.session.folders)}
        first, warm = measure_adaptive(
            lambda: server.set_folder_names(self.sid, names), self.repeat)
        server.set_folder_names(self.sid, {})
        return first, warm

    def query_target(self):
        """A stashed table to query, produced by the cheapest tool that makes
        one. Untimed: the cell measures the query, not the analysis."""
        result = server.plot_folder_filename(self.sid, self.target)
        return result["result_id"]


def grid_cells(ctx):
    """``[(table, label, run)]`` -- the canonical call per tool for one cell block.

    ``run`` returns ``(cold, warm)``. Order matters in three places:
    ``query_result`` needs a stashed table, ``set_folder_names`` rewrites the
    ``folder`` column every selector filters on, and ``close_log_root`` ends the
    session -- so those come last, in that order.
    """
    call = lambda fn: measure_adaptive(fn, ctx.repeat)  # noqa: E731
    sid, target, file_name = ctx.sid, ctx.target, ctx.file_name

    def query_cell():
        # The table to query is built when this cell runs, not when the list is
        # built -- otherwise every relaunch pays for it even if the cell is
        # already recorded.
        rid = ctx.query_target()
        return call(lambda: server.query_result(
            sid, rid, where=[["lines", ">", 0]], sort_by="lines"))

    cells = [
        ("aux", "peek_log_root", lambda: call(lambda: server.peek_log_root(ctx.path))),
        ("aux", "open_log_root", ctx.measured_open),
        ("aux", "list_log_roots", lambda: call(lambda: server.list_log_roots())),
        ("aux", "describe_log_root", lambda: call(lambda: server.describe_log_root(sid))),
        ("aux", "read_log_lines",
         lambda: call(lambda: server.read_log_lines(sid, target, file_name, limit=100))),
        ("aux", "search_log_lines",
         lambda: call(lambda: server.search_log_lines(sid, r"[Ee]rror"))),

        ("distance", "distance_folder_filename",
         lambda: call(lambda: server.distance_folder_filename(sid, target))),
        ("distance", "distance_folder_content",
         lambda: call(lambda: server.distance_folder_content(sid, target))),
        ("distance", "distance_file_content",
         lambda: call(lambda: server.distance_file_content(sid, target))),
        ("distance", "distance_line_content",
         lambda: call(lambda: server.distance_line_content(
             sid, target, target_files=[file_name]))),

        ("anomaly", "anomaly_folder_filename",
         lambda: call(lambda: server.anomaly_folder_filename(sid, target_folder=[target]))),
        ("anomaly", "anomaly_folder_content",
         lambda: call(lambda: server.anomaly_folder_content(sid, target_folder=[target]))),
        ("anomaly", "anomaly_file_content",
         lambda: call(lambda: server.anomaly_file_content(sid, target))),
        ("anomaly", "anomaly_line_content",
         lambda: call(lambda: server.anomaly_line_content(
             sid, target, target_files=[file_name]))),

        ("plot", "plot_folder_filename",
         lambda: call(lambda: server.plot_folder_filename(sid, target))),
        ("plot", "plot_folder_content (scatter)",
         lambda: call(lambda: server.plot_folder_content(sid, target))),
        ("plot", "plot_folder_content (scatter+umap)",
         lambda: call(lambda: server.plot_folder_content(
             sid, target, random_seed=42, plots=["umap", "scatter"]))),
        ("plot", "plot_file_content (scatter)",
         lambda: call(lambda: server.plot_file_content(
             sid, target, target_files=[file_name]))),
        ("plot", "plot_file_content (scatter+umap)",
         lambda: call(lambda: server.plot_file_content(
             sid, target, target_files=[file_name], random_seed=42,
             plots=["umap", "scatter"]))),

        ("aux", "query_result", query_cell),
        ("aux", "split_log_file", ctx.measured_split),
        ("aux", "run_config", ctx.measured_run_config),
        ("aux", "set_folder_names", ctx.measured_folder_names),
        ("aux", "close_log_root", ctx.measured_close),
    ]
    return cells


def cell_slug(kind, fraction, label):
    keep = "".join(char if char.isalnum() else "-" for char in label)
    return f"{kind}-{fraction_tag(fraction)}-{keep}"


def run_grid_block(kind, fraction, path, workdir, repeat, cell_dir, shape):
    """Measure every cell of one (log root, fraction), one JSON file per cell.

    Written per cell rather than per block because the block can *die*: on bgl
    at 50% and 100% the anomaly cells peak well past what a 16 GB machine has,
    and the OOM killer takes the whole process with no chance to record
    anything. The parent relaunches, this function skips what is already on
    disk, and the cell that was in flight is the one named in ``inflight.json``.
    """
    os.makedirs(cell_dir, exist_ok=True)
    inflight_path = os.path.join(cell_dir, "inflight.json")
    ctx = GridContext(kind, fraction, path, workdir, repeat).open()
    shape = {**shape, "n_lines": shape.get("n_lines") or ctx.n_rows,
             "n_folders": shape.get("n_folders") or ctx.n_folders}
    with open(os.path.join(cell_dir, f"shape-{kind}-{fraction_tag(fraction)}.json"),
              "w") as handle:
        json.dump({"root": kind, "fraction": fraction, **shape}, handle)

    print(f"\n{'=' * 100}\n {ROOT_LABELS[kind]} at {fraction:.0%} -- "
          f"{shape['n_folders']} log folders, {shape['n_lines']:,} lines\n{'=' * 100}")

    for table, label, run in grid_cells(ctx):
        out_path = os.path.join(cell_dir, cell_slug(kind, fraction, label) + ".json")
        if os.path.isfile(out_path):
            continue
        with open(inflight_path, "w") as handle:
            json.dump({"root": kind, "fraction": fraction, "table": table,
                       "tool": label, "path": out_path}, handle)
        record = {"root": kind, "fraction": fraction, "table": table, "tool": label,
                  "status": "ok"}
        started = time.perf_counter()
        try:
            cold, warm = run()
            record.update(cold_s=round(cold, 4), warm_s=round(warm, 4),
                          band=band(warm))
            print(f"  {label:<38} cold {cold:8.3f}s  warm {warm:8.3f}s  {band(warm)}")
        except Exception as exc:  # a tool that cannot run on this shape is data
            record.update(status="error", error=f"{type(exc).__name__}: {exc}",
                          cold_s=round(time.perf_counter() - started, 4))
            print(f"  {label:<38} ERROR {type(exc).__name__}: {exc}")
        with open(out_path, "w") as handle:
            json.dump(record, handle)
        os.remove(inflight_path)

    if ctx.session is not None:
        server.close_log_root(ctx.sid)
    gc.collect()


def collect_grid(cell_dir):
    """Every recorded cell and log-root shape, keyed for the writer."""
    records, shapes = {}, {}
    if not os.path.isdir(cell_dir):
        return records, shapes
    for name in sorted(os.listdir(cell_dir)):
        if not name.endswith(".json") or name == "inflight.json":
            continue
        with open(os.path.join(cell_dir, name)) as handle:
            row = json.load(handle)
        if name.startswith("shape-"):
            shapes[(row["root"], row["fraction"])] = row
        else:
            records[(row["root"], row["fraction"], row["tool"])] = row
    return records, shapes


def _fmt_seconds(record, key):
    if record is None:
        return ""
    if record["status"] == "oom":
        return "OOM"
    if record["status"] == "error":
        return "err"
    value = record.get(key)
    if value is None:
        return ""
    return f"{value:.3f}" if value < 10 else f"{value:.1f}"


def performance_markdown(records, shapes, fractions=FRACTIONS,
                         roots=("hadoop", "hdfs", "bgl")):
    """Fill PERFORMANCE.md's eight grids from the recorded cells."""
    def shape_cell(kind, fraction):
        shape = shapes.get((kind, fraction))
        if not shape:
            return ""
        if kind == "bgl":
            per = shape["n_lines"] // shape["n_folders"]
            return f"{shape['n_lines']:,} lines (10 x {per:,})"
        return f"{shape['n_folders']:,} folders / {shape['n_files']:,} files"

    lines = [
        "# MCP tool performance",
        "",
        "Cells: seconds. Columns: the four data fractions.",
        "",
        'What "5%" means differs by log root: for `hadoop_renamed` and '
        "`hdfs_balanced_5k` it is 5% of the",
        "**log folders** (and their files); for `bgl_split_10` it is 5% of "
        "`BGL.log`'s **log lines**, taken",
        "first and then split into 10 slices, so the folder count is always 10 "
        "and what varies is the text",
        "inside each one.",
        "",
        "Tables A1-A4 are **cold** (first call, nothing cached). Tables B1-B4 "
        "are the same grid **warm**",
        "(repeated call, same process).",
        "",
        "## Log roots at each fraction",
        "",
        "| log root | unit varied | " + " | ".join(f"{f:.0%}" for f in fractions) + " |",
        "|---|---|" + "---|" * len(fractions),
    ]
    for kind in roots:
        varied = "log lines" if kind == "bgl" else "log folders"
        cells = " | ".join(shape_cell(kind, f) for f in fractions)
        lines.append(f"| {ROOT_LABELS[kind]} | {varied} | {cells} |")

    for part, key, title in (("A", "cold_s", "cold (first call)"),
                             ("B", "warm_s", "warm (repeated call)")):
        lines += ["", f"# Part {part} -- {title}"]
        for index, (table, table_title) in enumerate(GRID_TABLE_TITLES, start=1):
            lines += ["", f"## Table {part}{index} -- {table_title}", "",
                      "| tool | log root | " + " | ".join(f"{f:.0%}" for f in fractions) + " |",
                      "|---|---|" + "---|" * len(fractions)]
            for row_table, label in GRID_ROWS:
                if row_table != table:
                    continue
                for kind in roots:
                    cells = " | ".join(
                        _fmt_seconds(records.get((kind, f, label)), key) for f in fractions)
                    lines.append(f"| {label} | {ROOT_LABELS[kind]} | {cells} |")
    return "\n".join(lines) + "\n"


def grid_main(args, datasets_folder, paths, cache_dir, workdir):
    """Drive the fraction grid, relaunching a block the OOM killer took.

    Each (log root, fraction) block runs in a child process. A block that dies
    is not a failed run: the cell it died on is recorded as ``OOM`` -- which is
    the measurement, on bgl -- and the block is relaunched to finish the rest.
    """
    reduced_folder = os.path.join(datasets_folder, "test_data", "mcp_bench_reduced")
    cell_dir = args.cell_dir or os.path.join(datasets_folder, "test_data", "mcp_bench_cells")
    os.makedirs(cell_dir, exist_ok=True)
    roots = tuple(args.roots or ("hadoop", "hdfs", "bgl"))
    fractions = tuple(args.fractions or FRACTIONS)

    for kind in roots:
        for fraction in fractions:
            path, shape = prepare_grid_root(kind, fraction, datasets_folder, paths,
                                            reduced_folder)
            if path is None:
                print(f"Skipping {kind} at {fraction:.0%}: BGL.log not found.")
                continue
            for attempt in range(len(GRID_ROWS) + 2):
                command = [sys.executable, os.path.abspath(__file__),
                           "--grid-block", f"{kind}:{fraction}", "--grid-block-path", path,
                           "--cell-dir", cell_dir, "--datasets", datasets_folder,
                           "--cache-dir", cache_dir, "--repeat", str(args.repeat),
                           "--grid-block-shape", json.dumps(shape)]
                completed = subprocess.run(command)
                if completed.returncode == 0:
                    break
                inflight_path = os.path.join(cell_dir, "inflight.json")
                if not os.path.isfile(inflight_path):
                    print(f"  {kind} {fraction:.0%} block failed with no cell in flight "
                          f"(exit {completed.returncode}); not retrying")
                    break
                with open(inflight_path) as handle:
                    inflight = json.load(handle)
                # Killed rather than raised: a MemoryError would have been caught
                # in the child and recorded as an error. Being killed by the OOM
                # killer is the finding on a log root this size.
                print(f"  {inflight['tool']} on {kind} at {fraction:.0%} was killed "
                      f"(exit {completed.returncode}) -- recording OOM and continuing")
                with open(inflight["path"], "w") as handle:
                    json.dump({**{k: v for k, v in inflight.items() if k != "path"},
                               "status": "oom", "exit_code": completed.returncode}, handle)
                os.remove(inflight_path)

    records, shapes = collect_grid(cell_dir)
    out_path = args.performance or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "PERFORMANCE.md")
    Path(out_path).write_text(performance_markdown(records, shapes, fractions, roots))
    print(f"\nCells: {len(records)} recorded in {cell_dir}\nTables -> {out_path}")


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default=None,
                        help="Where the log roots live. Defaults to root_folder in "
                             "downloader/datasets.yml.")
    parser.add_argument("--only", choices=("hadoop", "hdfs", "bgl"), action="append",
                        dest="roots", help="Benchmark one log root; repeatable. 'bgl' needs "
                                           "<datasets>/bgl/BGL.log and writes a 743 MB copy of "
                                           "it into the workdir.")
    parser.add_argument("--repeat", type=int, default=3,
                        help="Warm runs per measurement; the median is reported (default 3).")
    parser.add_argument("--markdown", default=None, help="Write the table to this file.")
    parser.add_argument("--json", default=None, help="Write the raw rows to this file.")
    parser.add_argument("--cache-dir", default=None,
                        help="Parquet cache. Defaults to <datasets>/test_data/mcp_cache.")
    parser.add_argument("--grid", action="store_true",
                        help="Run the fraction grid instead of the argument sweeps: every "
                             "tool, every log root, at 5/10/50/100%% of the data. Writes "
                             "tests/mcp/PERFORMANCE.md.")
    parser.add_argument("--fractions", type=float, nargs="+", default=None,
                        help=f"Fractions for --grid (default {' '.join(str(f) for f in FRACTIONS)}).")
    parser.add_argument("--performance", default=None,
                        help="Where --grid writes its tables (default tests/mcp/PERFORMANCE.md).")
    parser.add_argument("--cell-dir", default=None,
                        help="Where --grid keeps one JSON per measured cell, so a killed "
                             "block resumes. Defaults to <datasets>/test_data/mcp_bench_cells.")
    parser.add_argument("--tables-only", action="store_true",
                        help="Rebuild the --grid tables from the recorded cells without "
                             "measuring anything.")
    # Set by --grid when it launches a child for one (log root, fraction). Not
    # for hand use: the parent has already built the reduced log root, and the
    # child must not spend minutes checking that the datasets exist.
    parser.add_argument("--grid-block", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--grid-block-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--grid-block-shape", default="{}", help=argparse.SUPPRESS)
    args = parser.parse_args()

    datasets_folder = os.path.expanduser(
        args.datasets or make_test_data.default_source_folder())
    roots = tuple(args.roots or ("hadoop", "hdfs", "bgl"))
    cache_dir = args.cache_dir or os.path.join(datasets_folder, "test_data", "mcp_cache")

    if args.tables_only:
        cell_dir = args.cell_dir or os.path.join(datasets_folder, "test_data", "mcp_bench_cells")
        records, shapes = collect_grid(cell_dir)
        out_path = args.performance or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "PERFORMANCE.md")
        Path(out_path).write_text(performance_markdown(
            records, shapes, tuple(args.fractions or FRACTIONS), roots))
        print(f"Cells: {len(records)} recorded in {cell_dir}\nTables -> {out_path}")
        return

    # Building the two derived log roots is minutes of work, and neither 'bgl'
    # nor a grid child needs it -- the child is handed its log root.
    paths = (make_test_data.ensure_datasets(dest_folder=datasets_folder)
             if {"hadoop", "hdfs"} & set(roots) and not args.grid_block else {})

    workdir = tempfile.mkdtemp(prefix="loglead-mcp-bench-")
    server.STORE = SessionStore(cache_dir=cache_dir,
                                output_root=os.path.join(workdir, "output"))
    print(f"Cache:   {cache_dir}\nWorkdir: {workdir}\nRepeats: {args.repeat}")

    if args.grid_block:
        kind, fraction = args.grid_block.split(":")
        try:
            run_grid_block(kind, float(fraction), args.grid_block_path, workdir,
                           args.repeat, args.cell_dir,
                           json.loads(args.grid_block_shape))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        return

    if args.grid:
        try:
            grid_main(args, datasets_folder, paths, cache_dir, workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        return

    report = Report()
    started = time.time()
    try:
        if "hadoop" in roots:
            bench_hadoop(report, paths[make_test_data.HADOOP_RENAMED], args.repeat)
        if "hdfs" in roots:
            bench_hdfs(report, paths[make_test_data.HDFS_BALANCED_5K], args.repeat)
        if "bgl" in roots:
            # Not one of make_test_data's derived log roots: a plain loghub
            # download, skipped rather than built if it is not there.
            bgl = Path(datasets_folder) / "bgl" / "BGL.log"
            if bgl.is_file():
                bench_bgl(report, bgl, workdir, args.repeat)
            else:
                print(f"\nSkipping bgl: {bgl} not found. Get it with "
                      f"'uv run downloader/download_data.py'.")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    report.section("cost models -- what a call costs before you make it")
    for model in report.models:
        print(f"  {model['tool']:<28} ({model['dataset']:<6}) "
              f"{model['fixed_s']:7.2f}s + {model['marginal_s']:7.3f}s per {model['unit']}")

    report.section("summary -- what to avoid, by cost band")
    for name in ("very slow", "slow"):
        rows = [r for r in report.rows if r["class"] == name]
        if rows:
            print(f"\n {name}:")
            for row in sorted(rows, key=lambda r: -r["warm_s"]):
                print(f"   {row['warm_s']:7.1f}s  {row['tool']} ({row['dataset']}) "
                      f"{row['variant']}")
    print(f"\nBenchmark took {time.time() - started:.0f}s")

    if args.markdown:
        Path(args.markdown).write_text(report.markdown())
        print(f"Markdown table -> {args.markdown}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"measurements": report.rows, "cost_models": report.models}, indent=2))
        print(f"Raw rows       -> {args.json}")


if __name__ == "__main__":
    main()
