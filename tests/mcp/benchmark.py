"""What each MCP tool costs, measured rather than guessed.

The tools in ``loglead/mcp/server.py`` span five orders of magnitude in cost --
``describe_log_root`` is microseconds, ``anomaly_folder_content(target_folder="ALL")``
on 5,000 log folders is an hour -- and a model driving them over MCP cannot tell
which is which from a tool schema. This script produces the numbers that go into
their docstrings, and the scaling rules that let a caller predict a call it has
not made.

Run it with::

    uv run tests/mcp/benchmark.py                    # both log roots
    uv run tests/mcp/benchmark.py --only hadoop      # one of them
    uv run tests/mcp/benchmark.py --repeat 5         # steadier medians
    uv run tests/mcp/benchmark.py --markdown cost.md # table to paste into docs

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
Timings are hardware- and load-dependent, so treat the ratios as the durable part
and re-run for absolute numbers.
"""

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_test_data  # noqa: E402  (sits next to this file)

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
    report.note(f"detection runs per file, so {cold:.0f}s of this is 5,000 format probes; "
                f"the cached open is {cached:.2f}s")

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


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default=None,
                        help="Where the log roots live. Defaults to root_folder in "
                             "downloader/datasets.yml.")
    parser.add_argument("--only", choices=("hadoop", "hdfs"), action="append", dest="roots",
                        help="Benchmark one log root; repeatable.")
    parser.add_argument("--repeat", type=int, default=3,
                        help="Warm runs per measurement; the median is reported (default 3).")
    parser.add_argument("--markdown", default=None, help="Write the table to this file.")
    parser.add_argument("--json", default=None, help="Write the raw rows to this file.")
    parser.add_argument("--cache-dir", default=None,
                        help="Parquet cache. Defaults to <datasets>/test_data/mcp_cache.")
    args = parser.parse_args()

    datasets_folder = os.path.expanduser(
        args.datasets or make_test_data.default_source_folder())
    paths = make_test_data.ensure_datasets(dest_folder=datasets_folder)
    roots = tuple(args.roots or ("hadoop", "hdfs"))

    cache_dir = args.cache_dir or os.path.join(datasets_folder, "test_data", "mcp_cache")
    workdir = tempfile.mkdtemp(prefix="loglead-mcp-bench-")
    server.STORE = SessionStore(cache_dir=cache_dir,
                                output_root=os.path.join(workdir, "output"))
    print(f"Cache:   {cache_dir}\nWorkdir: {workdir}\nRepeats: {args.repeat}")

    report = Report()
    started = time.time()
    try:
        if "hadoop" in roots:
            bench_hadoop(report, paths[make_test_data.HADOOP_RENAMED], args.repeat)
        if "hdfs" in roots:
            bench_hdfs(report, paths[make_test_data.HDFS_BALANCED_5K], args.repeat)
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
