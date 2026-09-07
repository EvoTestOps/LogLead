"""Exercise every LogLead MCP tool against real logs, without MCP transport.

Calls the tool functions directly, which is what the server does once a request
has been decoded -- so this covers the analysis logic and the session model, and
runs in a few minutes without an MCP client attached.

What it demonstrates, beyond "nothing crashes":

* the logs are read, masked, and parsed **once**; the second open is a cache hit;
* asking for a second parser adds only the missing column instead of redoing the
  first one;
* every analysis returns real numbers, not just a path to a file;
* a truncated preview is not the end of the table -- ``query_result`` filters the
  rest of it from the session, without recomputing anything.

Usage::

    uv run demo/mcp_demo.py [--log-root /path/to/logs] [--keep-cache]
                                    [--folder-names names.json] [--format auto]

``--format`` is which loader reads the files: ``auto`` (the default) samples each file and picks
one, ``raw`` reads every line as one event the way LogDelta does, and a family or ``family/spec``
(``json``, ``syslog``, ``json/nginx_json``, ``delimited/zeek``, ...) pins one. On Hadoop, ``auto``
detects log4j-timestamped text and so yields an ``m_timestamp`` column that ``raw`` does not.

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
        # query_result counts matching rows rather than table rows, and pages
        # with offset rather than pointing at a file.
        total = result.get("n_rows", result.get("n_rows_matched"))
        shown = len(result.get("rows", []))
        rest = result.get("artifact") or (
            f"query_result(result_id={result['result_id']!r}, "
            f"offset={result.get('offset', 0) + shown})"
        )
        print(f"   ... {total} rows total -> {rest}")


def show_plot(result):
    """Print a plot result, which carries no rows -- a scatter has no top N."""
    for axis, stats in result["summary"].items():
        print(f"   {axis}: min={stats['min']}  median={stats['median']}  max={stats['max']}")
    target = result.get("target")
    if target:
        print(f"   target {target['folder']}: "
              f"unique_terms={target['unique_terms']} (p{target['unique_terms_pct']}), "
              f"lines={target['lines']} (p{target['lines_pct']})")
    print(f"   {result['n_rows']} points -> query_result(result_id="
          f"{result['result_id']!r})")


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
    parser.add_argument(
        "--format", default="auto",
        help="How to read the files: 'auto' (default, detect per file), 'raw', or a family or "
             "family/spec such as 'json/nginx_json'. See loglead.delta.log_root.available_formats().",
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
                 folder_names_path=args.folder_names, format=args.format)
    finally:
        if args.keep_cache:
            print(f"\nArtifacts kept in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def run_demo(log_root_path, keep_cache=False, folder_names_path=None, format="auto"):
    # ---------------------------------------------------------------- load --
    banner("open_log_root -- read, mask, and parse once")
    names = load_folder_names(folder_names_path, log_root_path)
    if names:
        print(f" naming {len(names)} log folders from {os.path.basename(folder_names_path)}, "
              "so output means something")
    started = time.time()
    info = server.open_log_root(
        path=log_root_path,
        # "auto" samples each file and builds the loader that reads it, so a log root of JSON,
        # syslog or CSV logs arrives in columns rather than as one blob per line. Pass --format raw
        # for the plain-text reading LogDelta does.
        format=format,
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
    # What detection chose, per format. The only place a wrong guess is visible.
    print(f" format={info['format']}  detected={info['detected_formats']}")

    target = info["folders"][0]
    print(f" target log folder: {target}")

    banner("open_log_root again -- served from the parquet cache")
    started = time.time()
    again = server.open_log_root(
        # Every argument that shapes the frame has to match for the cache to be hit, format
        # included -- it decides which loader ran and therefore every column.
        path=log_root_path, format=format, mask=True, mask_pattern="myllari_extended",
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
        # keep_original_folder_name=False: the log folder simply *is* the given name, for when the
        # directory id is noise in a plot legend rather than information. It replaces the directory
        # name outright, so the names have to be unique - and the JSON's are not, since four of its
        # folders are all "PageRank_MachineDown". Disambiguating them is the caller's job, which is
        # the whole reason this flag defaults to True.
        short_names = ("PageRank_Normal", "PageRank_MachineDown_A", "PageRank_MachineDown_B")
        renamed = server.set_folder_names(
            "demo2",
            dict(zip(sorted(names)[:3], short_names)),
            keep_original_folder_name=False,
        )
        print(f" {renamed['named']} named, {renamed['unnamed']} left alone"
              f" in {time.time() - started:.2f}s")
        print(f"   before: {before}")
        print(f"   after:  {[f for f in renamed['folders'] if f in short_names]}")
        # Names always apply to the directory name, so this replaces the mapping
        # from the JSON file rather than stacking onto it.
        assert short_names[0] in renamed["folders"], "keep_original=False should give a bare name"

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
    banner("L1 plot_folder_filename -- the axes come back, not just an HTML file")
    res = server.plot_folder_filename("demo", target, comparison_folders=8,
                               group_by_indices=[0, 1])
    show_plot(res)
    print(f"   plots: {res['plots']}")

    banner("L2 plot_folder_content")
    res = server.plot_folder_content("demo", target, comparison_folders=8,
                                  content_format="Words")
    show_plot(res)

    # The UMAP layout is essentially the whole cost of these tools, and the
    # default view -- unique terms against lines -- does not use it, so it is
    # opt-in: the difference is ~42s against under a second on 5,000 log
    # folders. Ask for it when the numbers alone leave the answer unclear.
    banner('L2 plot_folder_content again, plots=["umap", "scatter"] -- the embedding too')
    started = time.perf_counter()
    res = server.plot_folder_content("demo", target, comparison_folders=8,
                                     content_format="Words", random_seed=42,
                                     plots=["umap", "scatter"])
    print(f"   {time.perf_counter() - started:.2f}s, "
          f"figures written: {sorted(res['plots'])}")
    show_plot(res)
    # The layout's coordinates are columns of the points like any other, so
    # reading them is a query rather than a bigger result.
    q = server.query_result("demo", res["result_id"], sort_by="umap_x", max_rows=3)
    show(q, ["folder", "umap_x", "umap_y", "unique_terms", "lines"], limit=3)

    banner("query_result -- ask the whole table, instead of reading a preview of it")
    # Scoring every log folder produces one row each, of which a preview shows a
    # handful. The table stays in the session, so the follow-up question is a
    # filter rather than another analysis run.
    res = server.anomaly_folder_content("demo", target_folder="ALL", comparison_folders="ALL",
                                        content_format="Words", max_rows=3)
    print(f"   {res['n_rows']} log folders scored, {len(res['rows'])} previewed"
          f" -> result_id {res['result_id']}")
    q = server.query_result("demo", res["result_id"],
                            where=[["folder", "contains", "MachineDown"]],
                            sort_by="rank_sum", max_rows=4)
    print(f"   of those, {q['n_rows_matched']} are MachineDown log folders:")
    show(q, ["folder", "rank_sum"], limit=4)

    banner("L3 plot_file_content")
    res = server.plot_file_content("demo", target, comparison_folders=8,
                                   target_files=[worst_file],
                                   content_format="Words")
    for entry in res["files"]:
        print(f"   {entry['file_name']}: {entry['n_rows']} log folders plotted")
        show_plot(entry)

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
