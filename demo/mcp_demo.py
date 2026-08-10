"""Exercise every LogLead MCP tool against real logs, without MCP transport.

Calls the tool functions directly, which is what the server does once a request
has been decoded -- so this covers the analysis logic and the session model, and
runs in a few minutes without an MCP client attached.

What it demonstrates, beyond "nothing crashes":

* the logs are read, masked, and parsed **once**; the second open is a cache hit;
* asking for a second parser adds only the missing column instead of redoing the
  first one;
* every analysis returns real numbers, not just a path to a file.

Usage::

    uv run demo/mcp_demo.py [--log-root /path/to/logs] [--keep-cache]
                                    [--folder-names names.json]

The default log root is LogDelta's Hadoop demo data. Get it with::

    cd <LogDelta>/demo
    wget -O Hadoop.zip 'https://zenodo.org/records/8196385/files/Hadoop.zip?download=1'
    unzip Hadoop.zip -d Hadoop

**Naming the log folders.** A log folder -- any set of logs that belong
together, be it a test run, a day, or a release -- is named after the directory
it was read from, and that name is what every plot legend, result row and output
file is labelled with. Hadoop's directories are opaque ids, so this demo passes a
mapping from ``--folder-names``, a flat JSON object of
``{directory name: meaningful name}``::

    {
      "application_1445062781478_0012": "PageRank_MachineDown",
      "application_1445087491445_0005": "WordCount_Normal"
    }

It defaults to ``demo/mcp_demo_hadoop_folder_names.json``, shipped here and derived from the
labels the Hadoop dataset publishes in its own ``abnormal_label.txt``. Supplying
this mapping is the caller's job -- datasets record this metadata in wildly
different ways, if at all -- so write the JSON however suits your data. The names
need not be ground-truth labels: ``WorkingRunTue``/``FailingRunThu`` is just as
useful. Log folders left out of the mapping keep their directory name.
"""

import argparse
import json
import os
import shutil
import tempfile
import time

from loglead.mcp import server
from loglead.mcp.session import SessionStore

DEFAULT_LOG_ROOT = os.path.expanduser("~/LogDelta/demo/Hadoop")
DEFAULT_FOLDER_NAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "mcp_demo_hadoop_folder_names.json")


def banner(text):
    print(f"\n{'=' * 78}\n {text}\n{'=' * 78}")


def load_folder_names(path, log_root_path):
    """Read a ``{folder name: meaningful name}`` JSON mapping for this log_root.

    The shipped default describes Hadoop, so ``--log-root`` pointing anywhere else
    would name log folders that do not exist -- which open_log_root rightly
    rejects. Drop
    the mapping in that case rather than failing the demo.
    """
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as handle:
        names = json.load(handle)

    folders = {entry.name for entry in os.scandir(log_root_path) if entry.is_dir()}
    if not folders & set(names):
        print(f" no log folder in {log_root_path} appears in "
              f"{os.path.basename(path)}; keeping directory names")
        return {}
    return {folder: name for folder, name in names.items() if folder in folders}


def show(result, keys, limit=5):
    """Print the interesting part of a tool result."""
    for row in result.get("rows", [])[:limit]:
        print("   " + "  ".join(f"{k}={row.get(k)}" for k in keys if k in row))
    if result.get("truncated"):
        print(f"   ... {result['n_rows']} rows total -> {result.get('artifact')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument(
        "--keep-cache", action="store_true",
        help="Keep the scratch cache/output dir instead of deleting it.",
    )
    parser.add_argument(
        "--folder-names", default=DEFAULT_FOLDER_NAMES,
        help="JSON object of {folder name: meaningful name} for this log_root.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.log_root):
        raise SystemExit(
            f"Log root not found: {args.log_root}\n"
            "Pass --log-root, or fetch LogDelta's Hadoop demo data (see module docstring)."
        )

    workdir = tempfile.mkdtemp(prefix="loglead-mcp-demo-")
    server.STORE = SessionStore(cache_dir=os.path.join(workdir, "cache"),
                                output_root=os.path.join(workdir, "output"))
    print(f"Log root: {args.log_root}\nWorkdir:  {workdir}")

    try:
        run_demo(args.log_root, keep_cache=args.keep_cache,
                 folder_names_path=args.folder_names)
    finally:
        if args.keep_cache:
            print(f"\nArtifacts kept in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def run_demo(log_root_path, keep_cache=False, folder_names_path=None):
    # ---------------------------------------------------------------- load --
    banner("open_log_root -- read, mask, and parse once")
    names = load_folder_names(folder_names_path, log_root_path)
    if names:
        print(f" naming {len(names)} log folders from {os.path.basename(folder_names_path)}, "
              "so output means something")
    started = time.time()
    info = server.open_log_root(
        path=log_root_path,
        mask=True,
        mask_pattern="myllari_extended",
        parsers=["tip"],
        # Hadoop file names embed the folder id, so without this no file appears
        # in more than one log folder and L3/L4 have nothing to compare.
        file_name_normalizer="strip_folder_id",
        # application_1445062781478_0011 -> PageRank_MachineDown_application_...
        folder_names=names,
        session_id="demo",
    )
    cold = time.time() - started
    print(f" {info['n_folders']} log folders, {info['n_files']} distinct file names, "
          f"{info['n_rows']:,} lines in {cold:.1f}s")
    print(f" parsers={info['parsers']}  enhanced={info['enhanced_columns']}")
    print(f" cache_hit={info['cache_hit']}")

    target = info["folders"][0]
    print(f" target log folder: {target}")

    banner("open_log_root again -- served from the parquet cache")
    started = time.time()
    again = server.open_log_root(
        path=log_root_path, mask=True, mask_pattern="myllari_extended",
        parsers=["tip"], file_name_normalizer="strip_folder_id", folder_names=names,
        session_id="demo2",
    )
    warm = time.time() - started
    print(f" cache_hit={again['cache_hit']}  {warm:.1f}s  (cold was {cold:.1f}s)")
    assert again["cache_hit"], "second open should have hit the parquet cache"

    if names:
        banner("set_folder_names -- rename after opening, nothing re-read")
        before = again["folders"][0]
        started = time.time()
        # keep_original_folder_name=False: the log folder simply becomes the
        # given name, for when the directory name carries nothing worth keeping.
        renamed = server.set_folder_names(
            "demo2",
            {folder: f"Folder{i}" for i, folder in enumerate(sorted(names)[:3])},
            keep_original_folder_name=False,
        )
        print(f" {renamed['named']} named, {renamed['unnamed']} left alone"
              f" in {time.time() - started:.2f}s")
        print(f"   before: {before}")
        print(f"   after:  {[f for f in renamed['folders'] if f.startswith('Folder')][:3]}")
        # Names always apply to the directory name, so this replaces the mapping
        # from the JSON file rather than stacking onto it.
        assert "Folder0" in renamed["folders"], "keep_original=False should give a bare name"

    server.close_log_root("demo2")

    # ------------------------------------------------------------ distance --
    banner("L1 distance_folder_filename -- which log folders differ in file sets?")
    res = server.distance_folder_filename("demo", target, comparison_folders=5)
    show(res, ["comparison_folder", "intersection", "jaccard distance", "overlap distance"])

    banner("L2 distance_folder_content -- which log folder's text differs most?")
    res = server.distance_folder_content("demo", target, comparison_folders=5,
                                      content_format="Words")
    show(res, ["comparison_folder", "cosine", "jaccard", "rank_sum"])

    banner("L3 distance_file_content -- which file differs most?")
    res = server.distance_file_content("demo", target, comparison_folders=3,
                                       target_files=2, content_format="Words")
    show(res, ["file_name", "comparison_folder", "cosine", "zscore_sum"])

    banner("L4 distance_line_content -- the actual diff")
    res = server.distance_line_content("demo", target, comparison_folders=1,
                                       target_files=1, max_changed_lines=3)
    for comp in res["comparisons"]:
        print(f"   {comp['file_name']} vs {comp['comparison_folder']}: {comp['summary']}")
        for line in comp["changed_sample"]:
            print(f"     {line['difference']} {line['content'][:80]}")

    # ------------------------------------------------------------- anomaly --
    banner("L1 anomaly_folder_filename -- score log folders by their file sets")
    res = server.anomaly_folder_filename("demo", target_folder=3, comparison_folders=10)
    show(res, ["folder", "rank_sum", "zscore_sum"])

    banner("L2 anomaly_folder_content -- score log folders by their text")
    res = server.anomaly_folder_content("demo", target_folder=3, comparison_folders=10,
                                     content_format="Words")
    show(res, ["folder", "rank_sum", "zscore_sum"])

    banner("L3 anomaly_file_content -- which file of the target looks worst?")
    res = server.anomaly_file_content("demo", target, comparison_folders=10,
                                      target_files=3, content_format="Words")
    show(res, ["file_name", "rank_sum", "zscore_sum"])
    worst_file = res["rows"][0]["file_name"] if res["rows"] else "container__01_000001.log"

    banner(f"L4 anomaly_line_content -- worst lines of {worst_file}, with their text")
    res = server.anomaly_line_content("demo", target, comparison_folders="ALL",
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
    res = server.anomaly_folder_content(
        "demo", target_folder=2, comparison_folders=5,
        detectors=["KMeans", "RarityModel"],
        detector_params={"KMeans": {"n_clusters": 3}, "RarityModel": {"threshold": 100}},
    )
    show(res, ["folder", "kmeans_pred_ano_proba", "RM_pred_ano_proba", "rank_sum"])

    # ------------------------------------ incremental enhancement, the point --
    banner("switching parser -- only the missing column gets computed")
    session = server.STORE.get("demo")
    print(f" before: parsers={session.parsers}")
    started = time.time()
    server.anomaly_folder_content("demo", target_folder=2, comparison_folders=5,
                               content_format="Parse-Drain")
    drain_time = time.time() - started
    print(f" after Parse-Drain: parsers={session.parsers}  ({drain_time:.1f}s)")

    started = time.time()
    server.anomaly_folder_content("demo", target_folder=2, comparison_folders=5,
                               content_format="Parse-Tip")
    tip_time = time.time() - started
    print(f" reusing Parse-Tip from open time: {tip_time:.1f}s "
          f"(vs {drain_time:.1f}s to add a new parser)")
    assert "tip" in session.parsers and "drain" in session.parsers

    # ----------------------------------------------------------- drill-down --
    banner("search_log_lines -- which log folders mention preemption?")
    res = server.search_log_lines("demo", r"Going to preempt", limit=2)
    print(f"   {res['total_matches']} matches across "
          f"{res['folders_with_matches']} log folders")
    for row in res["matches_per_folder"][:5]:
        print(f"     {row['folder']}: {row['matches']}")

    banner("read_log_lines -- read the raw text")
    res = server.read_log_lines("demo", target, worst_file, offset=0, limit=3)
    print(f"   {res['total_lines']} lines in {res['file_name']}")
    for line in res["lines"]:
        print(f"     L{line['line_number']:<4} {line['m_message'][:80]}")

    # ------------------------------------------------------------ visualize --
    banner("L1 plot_folder_filename -- coordinates come back, not just an HTML file")
    res = server.plot_folder_filename("demo", target, comparison_folders=8,
                               group_by_indices=[0, 1], random_seed=42)
    show(res, ["folder", "umap_x", "umap_y", "unique_terms", "lines"], limit=4)
    print(f"   plots: {res['plots']}")

    banner("L2 plot_folder_content")
    res = server.plot_folder_content("demo", target, comparison_folders=8,
                                  content_format="Words", random_seed=42)
    show(res, ["folder", "umap_x", "umap_y", "unique_terms", "lines"], limit=4)

    banner("L3 plot_file_content")
    res = server.plot_file_content("demo", target, comparison_folders=8,
                                   target_files=[worst_file],
                                   content_format="Words", random_seed=42)
    for entry in res["files"]:
        print(f"   {entry['file_name']}: {entry['n_rows']} log folders plotted")

    # ----------------------------------------------------------------- wrap --
    banner("final session state")
    summary = server.STORE.get("demo").summary()
    for key in ("n_folders", "n_files", "n_rows", "parsers", "enhanced_columns"):
        print(f"   {key}: {summary[key]}")
    n_artifacts = len(os.listdir(summary['output_dir']))
    if keep_cache:
        print(f"   output_dir: {summary['output_dir']}  ({n_artifacts} artifacts, kept on exit)")
    else:
        print(f"   output_dir: {summary['output_dir']}  ({n_artifacts} artifacts, "
              "deleted on exit -- rerun with --keep-cache to inspect them)")
    server.close_log_root("demo")
    print("\nAll tools exercised successfully.")


if __name__ == "__main__":
    main()
