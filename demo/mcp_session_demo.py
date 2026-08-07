"""Exercise every LogLead MCP tool against a real corpus, without MCP transport.

Calls the tool functions directly, which is what the server does once a request
has been decoded -- so this covers the analysis logic and the session model, and
runs in a few minutes without an MCP client attached.

What it demonstrates, beyond "nothing crashes":

* the corpus is read, masked, and parsed **once**; the second open is a cache hit;
* asking for a second parser adds only the missing column instead of redoing the
  first one;
* every analysis returns real numbers, not just a path to a file.

Usage::

    uv run demo/mcp_session_demo.py [--corpus /path/to/corpus] [--keep-cache]

The default corpus is LogDelta's Hadoop demo data. Get it with::

    cd <LogDelta>/demo
    wget -O Hadoop.zip 'https://zenodo.org/records/8196385/files/Hadoop.zip?download=1'
    unzip Hadoop.zip -d Hadoop
"""

import argparse
import os
import shutil
import tempfile
import time

from loglead.mcp import server
from loglead.mcp.session import SessionStore

DEFAULT_CORPUS = os.path.expanduser("~/LogDelta/demo/Hadoop")


def banner(text):
    print(f"\n{'=' * 78}\n {text}\n{'=' * 78}")


def show(result, keys, limit=5):
    """Print the interesting part of a tool result."""
    for row in result.get("rows", [])[:limit]:
        print("   " + "  ".join(f"{k}={row.get(k)}" for k in keys if k in row))
    if result.get("truncated"):
        print(f"   ... {result['n_rows']} rows total -> {result.get('artifact')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument(
        "--keep-cache", action="store_true",
        help="Keep the scratch cache/output dir instead of deleting it.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.corpus):
        raise SystemExit(
            f"Corpus not found: {args.corpus}\n"
            "Pass --corpus, or fetch LogDelta's Hadoop demo data (see module docstring)."
        )

    workdir = tempfile.mkdtemp(prefix="loglead-mcp-demo-")
    server.STORE = SessionStore(cache_dir=os.path.join(workdir, "cache"),
                                output_root=os.path.join(workdir, "output"))
    print(f"Corpus:  {args.corpus}\nWorkdir: {workdir}")

    try:
        run_demo(args.corpus)
    finally:
        if args.keep_cache:
            print(f"\nArtifacts kept in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def run_demo(corpus_path):
    # ---------------------------------------------------------------- load --
    banner("open_corpus -- read, mask, and parse once")
    started = time.time()
    info = server.open_corpus(
        path=corpus_path,
        mask=True,
        mask_pattern="myllari_extended",
        parsers=["tip"],
        # Hadoop file names embed the run id, so without this no file appears
        # in more than one run and L3/L4 have nothing to compare.
        file_name_normalizer="strip_run_id",
        session_id="demo",
    )
    cold = time.time() - started
    print(f" {info['n_runs']} runs, {info['n_files']} distinct file names, "
          f"{info['n_rows']:,} lines in {cold:.1f}s")
    print(f" parsers={info['parsers']}  enhanced={info['enhanced_columns']}")
    print(f" cache_hit={info['cache_hit']}")

    target = info["runs"][0]
    print(f" target run: {target}")

    banner("open_corpus again -- served from the parquet cache")
    started = time.time()
    again = server.open_corpus(
        path=corpus_path, mask=True, mask_pattern="myllari_extended",
        parsers=["tip"], file_name_normalizer="strip_run_id", session_id="demo2",
    )
    warm = time.time() - started
    print(f" cache_hit={again['cache_hit']}  {warm:.1f}s  (cold was {cold:.1f}s)")
    assert again["cache_hit"], "second open should have hit the parquet cache"
    server.close_corpus("demo2")

    # ------------------------------------------------------------ distance --
    banner("L1 distance_run_file -- which runs have different file sets?")
    res = server.distance_run_file("demo", target, comparison_runs=5)
    show(res, ["comparison_run", "intersection", "jaccard distance", "overlap distance"])

    banner("L2 distance_run_content -- which run's text differs most?")
    res = server.distance_run_content("demo", target, comparison_runs=5,
                                      content_format="Words")
    show(res, ["comparison_run", "cosine", "jaccard", "rank_sum"])

    banner("L3 distance_file_content -- which file differs most?")
    res = server.distance_file_content("demo", target, comparison_runs=3,
                                       target_files=2, content_format="Words")
    show(res, ["file_name", "comparison_run", "cosine", "zscore_sum"])

    banner("L4 distance_line_content -- the actual diff")
    res = server.distance_line_content("demo", target, comparison_runs=1,
                                       target_files=1, max_changed_lines=3)
    for comp in res["comparisons"]:
        print(f"   {comp['file_name']} vs {comp['comparison_run']}: {comp['summary']}")
        for line in comp["changed_sample"]:
            print(f"     {line['difference']} {line['content'][:80]}")

    # ------------------------------------------------------------- anomaly --
    banner("L1 anomaly_run_file -- score runs by their file sets")
    res = server.anomaly_run_file("demo", target_run=3, comparison_runs=10)
    show(res, ["run", "rank_sum", "zscore_sum"])

    banner("L2 anomaly_run_content -- score runs by their text")
    res = server.anomaly_run_content("demo", target_run=3, comparison_runs=10,
                                     content_format="Words")
    show(res, ["run", "rank_sum", "zscore_sum"])

    banner("L3 anomaly_file_content -- which file of the target looks worst?")
    res = server.anomaly_file_content("demo", target, comparison_runs=10,
                                      target_files=3, content_format="Words")
    show(res, ["file_name", "rank_sum", "zscore_sum"])
    worst_file = res["rows"][0]["file_name"] if res["rows"] else "container__01_000001.log"

    banner(f"L4 anomaly_line_content -- worst lines of {worst_file}, with their text")
    res = server.anomaly_line_content("demo", target, comparison_runs="ALL",
                                      target_files=[worst_file],
                                      content_format="Words", max_rows=5)
    for entry in res["files"]:
        print(f"   {entry['file_name']}: {entry['n_lines']} lines, "
              f"ranked by {entry['sorted_by']}")
        for line in entry["top_lines"]:
            print(f"     L{line['line_number']:<5} rank_sum={line.get('rank_sum')}"
                  f" :: {str(line.get('m_message'))[:70]}")
        print(f"     plot: {entry['plot']}")

    banner("detector subset + hyperparameters (LogDelta hardcoded these)")
    res = server.anomaly_run_content(
        "demo", target_run=2, comparison_runs=5,
        detectors=["KMeans", "RarityModel"],
        detector_params={"KMeans": {"n_clusters": 3}, "RarityModel": {"threshold": 100}},
    )
    show(res, ["run", "kmeans_pred_ano_proba", "RM_pred_ano_proba", "rank_sum"])

    # ------------------------------------ incremental enhancement, the point --
    banner("switching parser -- only the missing column gets computed")
    session = server.STORE.get("demo")
    print(f" before: parsers={session.parsers}")
    started = time.time()
    server.anomaly_run_content("demo", target_run=2, comparison_runs=5,
                               content_format="Parse-Drain")
    drain_time = time.time() - started
    print(f" after Parse-Drain: parsers={session.parsers}  ({drain_time:.1f}s)")

    started = time.time()
    server.anomaly_run_content("demo", target_run=2, comparison_runs=5,
                               content_format="Parse-Tip")
    tip_time = time.time() - started
    print(f" reusing Parse-Tip from open time: {tip_time:.1f}s "
          f"(vs {drain_time:.1f}s to add a new parser)")
    assert "tip" in session.parsers and "drain" in session.parsers

    # ----------------------------------------------------------- drill-down --
    banner("search_log_lines -- which runs mention preemption?")
    res = server.search_log_lines("demo", r"Going to preempt", limit=2)
    print(f"   {res['total_matches']} matches across {res['runs_with_matches']} runs")
    for row in res["matches_per_run"][:5]:
        print(f"     {row['run']}: {row['matches']}")

    banner("read_log_lines -- read the raw text")
    res = server.read_log_lines("demo", target, worst_file, offset=0, limit=3)
    print(f"   {res['total_lines']} lines in {res['file_name']}")
    for line in res["lines"]:
        print(f"     L{line['line_number']:<4} {line['m_message'][:80]}")

    # ------------------------------------------------------------ visualize --
    banner("L1 plot_run_file -- coordinates come back, not just an HTML file")
    res = server.plot_run_file("demo", target, comparison_runs=8,
                               group_by_indices=[0, 1], random_seed=42)
    show(res, ["run", "umap_x", "umap_y", "unique_terms", "lines"], limit=4)
    print(f"   plots: {res['plots']}")

    banner("L2 plot_run_content")
    res = server.plot_run_content("demo", target, comparison_runs=8,
                                  content_format="Words", random_seed=42)
    show(res, ["run", "umap_x", "umap_y", "unique_terms", "lines"], limit=4)

    banner("L3 plot_file_content")
    res = server.plot_file_content("demo", target, comparison_runs=8,
                                   target_files=[worst_file],
                                   content_format="Words", random_seed=42)
    for entry in res["files"]:
        print(f"   {entry['file_name']}: {entry['n_rows']} runs plotted")

    # ----------------------------------------------------------------- wrap --
    banner("final session state")
    summary = server.STORE.get("demo").summary()
    for key in ("n_runs", "n_files", "n_rows", "parsers", "enhanced_columns"):
        print(f"   {key}: {summary[key]}")
    print(f"   output_dir: {summary['output_dir']}")
    print(f"   artifacts written: {len(os.listdir(summary['output_dir']))}")
    server.close_corpus("demo")
    print("\nAll tools exercised successfully.")


if __name__ == "__main__":
    main()
