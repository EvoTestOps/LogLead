"""What each MCP tool costs, measured rather than guessed.

The tools in ``loglead/mcp/server.py`` span five orders of magnitude in cost --
``describe_log_root`` is microseconds, ``anomaly_folder_content(target_folder="ALL")``
on 5,000 log folders is an hour and a gigabyte-scale memory spike -- and a model
driving them over MCP cannot tell which is which from a tool schema. This script
measures every tool at one canonical call on each log root at 5, 10, 50 and 100%
of its data, which is what the whole API costs on a log root of a given size. It
writes ``tests/mcp/PERFORMANCE.md`` (time) and ``tests/mcp/PERF_MEMORY.md``
(peak resident memory) from the same recorded cells::

    uv run tests/mcp/benchmark.py                     # all three log roots
    uv run tests/mcp/benchmark.py --only hadoop       # one of them
    uv run tests/mcp/benchmark.py --fractions 0.05    # one fraction
    uv run tests/mcp/benchmark.py --repeat 5          # steadier medians
    uv run tests/mcp/benchmark.py --tables-only       # rebuild tables, measure nothing

Each (log root, fraction) runs in a **child process** and one JSON file is
recorded per measured cell, because on bgl at 50% and 100% the anomaly cells are
killed by the OOM killer rather than raising -- the parent records that as
``OOM`` (which is the measurement) and relaunches the block to finish the rest.
That also makes the grid resumable: a re-run skips every cell already on disk.

**Two numbers per cell, and the difference is the point.** The cold tables are a
first call with nothing cached; the warm ones are the median time (max memory)
of the repeats. Where they differ, the gap is a column the session computed once
and kept -- ``e_words``, ``e_event_drain_id`` -- so a cold number is the price
of that representation and the warm one is what the analysis itself costs. That
is the session model's entire justification, and it is only visible as two
numbers. Memory is sampled from a background thread while each call runs (see
``_RSSMonitor``), since ``ru_maxrss`` never resets and would otherwise let one
memory-hungry cell's peak bleed into every cell measured after it in the same
process.

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
import os
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import NamedTuple

import polars as pl
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_test_data  # noqa: E402  (sits next to this file)
from loglead.delta import anomaly, distance, split  # noqa: E402

try:
    from loglead.mcp import server  # noqa: E402
    from loglead.mcp.session import SessionStore  # noqa: E402
except ImportError as exc:
    raise SystemExit(
        f"Cannot import loglead.mcp ({exc}). It is an optional install: uv sync --extra mcp"
    ) from exc

#: Cost bands, printed beside each cell and recorded with it. A caller does not
#: need three digits; it needs to know whether a call is free, worth a pause, or
#: worth avoiding.
BANDS = ((0.1, "instant"), (1.0, "fast"), (10.0, "slow"), (float("inf"), "very slow"))


def band(seconds):
    return next(name for limit, name in BANDS if seconds < limit)


# --------------------------------------------------------------------------- #
# The fraction grid: every tool, every log root, at 5 / 10 / 50 / 100% of data
# --------------------------------------------------------------------------- #
#
# Every tool is held at one canonical call and the *data* is moved instead, so a
# row answers "what does this call cost on a log root of this size" rather than
# "what is it proportional to". It produces tests/mcp/PERFORMANCE.md.
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


def _rss_gb():
    """Resident memory *right now*, in GB.

    Reads ``/proc/self/statm`` -- Linux only, which the grid already assumes
    elsewhere (OOM-kill detection is Linux-specific too). Falls back to the
    ``getrusage`` high-water mark if the read fails for any reason.
    """
    try:
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (OSError, ValueError, IndexError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


class _RSSMonitor:
    """Samples resident memory in a background thread while one call runs.

    ``ru_maxrss`` is a *process* high-water mark that never resets, so a cell
    run after a memory-hungry one would inherit its peak forever -- which is
    exactly wrong for attributing memory to a single cell. Sampling instead
    gives each call its own peak, at the cost of missing a spike narrower than
    ``interval``; 20ms is short enough to catch what OOM-kills a 16 GB machine
    without the polling thread itself costing anything a caller would notice.
    """

    def __init__(self, interval=0.02):
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self.peak = _rss_gb()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, _rss_gb())
            self._stop.wait(self.interval)

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, _rss_gb())
        return False


class Measure(NamedTuple):
    """What one call cost: how long, the process peak, and the floor it started from.

    ``peak_gb`` is what the tables print: what has to fit in RAM while the call
    runs, which is the number that OOM-kills a machine.

    ``floor_gb`` is resident memory the instant before the call -- the frames
    already open, the libraries already imported. It is recorded in the cell
    JSON but **not** rendered: it is there to keep a peak interpretable when one
    looks wrong, since most of a bgl cell's peak is the 3GB frame sitting beside
    it rather than the call. That is how ``peek_log_root`` came to read 5.14GB
    in the first grid that measured memory, ~0.02 of which was peek -- the fix
    for which was to measure it before anything is open, not to print a second
    number in every cell.
    """

    seconds: float
    peak_gb: float
    floor_gb: float


def time_and_mem(call):
    """One call's elapsed time, peak resident memory and starting floor."""
    with _RSSMonitor() as monitor:
        floor = monitor.peak  # seeded with the RSS at entry
        started = time.perf_counter()
        call()
        elapsed = time.perf_counter() - started
    return Measure(elapsed, monitor.peak, floor)


def measure_adaptive(call, repeat=3, heavy_seconds=HEAVY_SECONDS):
    """Time and memory-profile a call cold, then take the median time and max
    memory of ``repeat`` warm runs -- except that a call which turns out to be
    slow gets one warm run instead.

    Which cells are heavy is a property of the log root, not of the tool -- the
    same ``distance_folder_content`` is 0.3s on Hadoop and minutes on bgl -- so
    the cold call decides, rather than a list maintained by hand. Memory takes
    the max across repeats rather than the median that time uses: a peak that
    only showed up once is still the one a client should plan for.
    """
    cold = time_and_mem(call)
    warm = [time_and_mem(call)
            for _ in range(1 if cold.seconds > heavy_seconds else max(1, repeat))]
    return cold, Measure(statistics.median(run.seconds for run in warm),
                         max(run.peak_gb for run in warm),
                         max(run.floor_gb for run in warm))


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
    ("aux", "read_log_lines (new tokens)"),
    ("aux", "new_tokens"),
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

#: The four anomaly tools and the two content-based distance tools, each run
#: with a single detector/measure instead of the default all-four -- Part C/D
#: of PERFORMANCE.md and PERF_MEMORY.md. Built from ``anomaly.DEFAULT_DETECTORS``
#: / ``distance.DEFAULT_MEASURES`` rather than hardcoded, so a detector or
#: measure added there shows up here without a second edit.
#: ``distance_folder_filename`` (jaccard/overlap distance over file names only)
#: and ``distance_line_content`` (a text diff, no measures) have nothing to
#: isolate -- neither computes multiple vectorized measures in one pass.
DETAIL_ANOMALY_TOOLS = ("anomaly_folder_filename", "anomaly_folder_content",
                        "anomaly_file_content", "anomaly_line_content")
DETAIL_DISTANCE_TOOLS = ("distance_folder_content", "distance_file_content")

DETAIL_GRID_ROWS = tuple(
    ("anomaly_detail", f"{tool} ({detector})")
    for tool in DETAIL_ANOMALY_TOOLS for detector in anomaly.DEFAULT_DETECTORS
) + tuple(
    ("distance_detail", f"{tool} ({measure})")
    for tool in DETAIL_DISTANCE_TOOLS for measure in distance.DEFAULT_MEASURES
)

DETAIL_TABLE_TITLES = (("anomaly_detail", "Anomaly tools detailed"),
                       ("distance_detail", "Distance tools detailed"))


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
        cold = time_and_mem(
            lambda: server.open_log_root(**open_kwargs(self.kind, self.path, sid),
                                         refresh=True))
        server.close_log_root(sid)
        gc.collect()
        cached = time_and_mem(
            lambda: server.open_log_root(**open_kwargs(self.kind, self.path, sid)))
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
        cold = time_and_mem(
            lambda: server.split_log_file(biggest, n_slices=10, out_dir=out_dir))
        _, warm = measure_adaptive(
            lambda: server.split_log_file(biggest, n_slices=10, out_dir=out_dir),
            self.repeat)
        shutil.rmtree(out_dir, ignore_errors=True)
        return cold, warm

    def measured_close(self):
        """A close is a one-shot: closing an already-closed session is a no-op,
        so this is measured once and the same numbers stand as both."""
        closed = time_and_mem(lambda: server.close_log_root(self.sid))
        self.session = None
        return closed, closed

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
        cold, warm = measure_adaptive(
            lambda: server.set_folder_names(self.sid, names), self.repeat)
        server.set_folder_names(self.sid, {})
        return cold, warm

    def query_target(self):
        """A stashed table to query, produced by the cheapest tool that makes
        one. Untimed: the cell measures the query, not the analysis."""
        result = server.plot_folder_filename(self.sid, self.target)
        return result["result_id"]


def session_free_cells(ctx):
    """``[(table, label, run)]`` for the tools that need no session open.

    Measured **before** the block opens the log root, and that ordering is the
    whole point. Both of these are questions about what is on disk -- peek
    stats files and reads a few hundred lines, split streams a file through an
    8MB buffer -- so neither should be reported carrying the weight of a frame
    it never touches. Run after the open, ``peek_log_root`` on bgl at 100%
    measured 5.14GB, all but ~0.02 of it the session sitting beside it.
    """
    return [
        ("aux", "peek_log_root",
         lambda: measure_adaptive(lambda: server.peek_log_root(ctx.path), ctx.repeat)),
        ("aux", "split_log_file", ctx.measured_split),
    ]


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

    def fresh_vocabulary(fn):
        # The cold call has to build the baseline vocabulary, so drop any an
        # earlier cell left in the session; the warm repeats then reuse it.
        def run():
            ctx.session.vocabularies.clear()
            return call(fn)
        return run

    cells = [
        ("aux", "open_log_root", ctx.measured_open),
        ("aux", "list_log_roots", lambda: call(lambda: server.list_log_roots())),
        ("aux", "describe_log_root", lambda: call(lambda: server.describe_log_root(sid))),
        ("aux", "read_log_lines",
         lambda: call(lambda: server.read_log_lines(sid, target, file_name, limit=100))),
        ("aux", "search_log_lines",
         lambda: call(lambda: server.search_log_lines(sid, r"[Ee]rror"))),
        ("aux", "read_log_lines (new tokens)",
         fresh_vocabulary(lambda: server.read_log_lines(
             sid, target, file_name, limit=100, new_tokens_vs="ALL"))),
        ("aux", "new_tokens",
         fresh_vocabulary(lambda: server.new_tokens(sid, target))),

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
        ("aux", "run_config", ctx.measured_run_config),
        ("aux", "set_folder_names", ctx.measured_folder_names),
        ("aux", "close_log_root", ctx.measured_close),
    ]
    return cells


def grid_detail_cells(ctx):
    """``[(table, label, run)]`` -- one detector/measure isolated per cell.

    Same target/target_files as ``grid_cells``: the combined-call cost is
    measured there, this isolates one component's own share of it. Default
    parameters bind the loop variable at definition time, the usual fix for a
    lambda otherwise closing over the loop's final value.
    """
    call = lambda fn: measure_adaptive(fn, ctx.repeat)  # noqa: E731
    sid, target, file_name = ctx.sid, ctx.target, ctx.file_name

    cells = []
    for detector in anomaly.DEFAULT_DETECTORS:
        cells.append((
            "anomaly_detail", f"anomaly_folder_filename ({detector})",
            lambda detector=detector: call(lambda: server.anomaly_folder_filename(
                sid, target_folder=[target], detectors=[detector]))))
        cells.append((
            "anomaly_detail", f"anomaly_folder_content ({detector})",
            lambda detector=detector: call(lambda: server.anomaly_folder_content(
                sid, target_folder=[target], detectors=[detector]))))
        cells.append((
            "anomaly_detail", f"anomaly_file_content ({detector})",
            lambda detector=detector: call(lambda: server.anomaly_file_content(
                sid, target, detectors=[detector]))))
        cells.append((
            "anomaly_detail", f"anomaly_line_content ({detector})",
            lambda detector=detector: call(lambda: server.anomaly_line_content(
                sid, target, target_files=[file_name], detectors=[detector]))))

    for measure in distance.DEFAULT_MEASURES:
        cells.append((
            "distance_detail", f"distance_folder_content ({measure})",
            lambda measure=measure: call(lambda: server.distance_folder_content(
                sid, target, measures=[measure]))))
        cells.append((
            "distance_detail", f"distance_file_content ({measure})",
            lambda measure=measure: call(lambda: server.distance_file_content(
                sid, target, measures=[measure]))))
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
    ctx = GridContext(kind, fraction, path, workdir, repeat)

    print(f"\n{'=' * 100}\n {ROOT_LABELS[kind]} at {fraction:.0%} -- "
          f"{shape['n_folders']} log folders\n{'=' * 100}")

    # The tools that ask about files on disk, measured while nothing is open:
    # a floor of bare imports is the one they should be read against.
    session_free = session_free_cells(ctx)
    record_cells(session_free, kind, fraction, cell_dir, inflight_path)

    # Opening is the expensive thing in the block -- minutes and gigabytes on
    # bgl -- so it is skipped when nothing left to measure needs a session. That
    # is not a rare case: it is every relaunch that only has session-free cells
    # left, and every targeted re-measure of one row.
    session_labels = {label for _, label, _ in session_free}
    all_rows = GRID_ROWS + DETAIL_GRID_ROWS
    pending = [label for _, label in all_rows if label not in session_labels
               and cell_pending(os.path.join(cell_dir,
                                             cell_slug(kind, fraction, label) + ".json"))]
    if not pending:
        print(" ...every cell needing a session is already recorded; not opening")
        return

    ctx.open()
    shape = {**shape, "n_lines": shape.get("n_lines") or ctx.n_rows,
             "n_folders": shape.get("n_folders") or ctx.n_folders}
    with open(os.path.join(cell_dir, f"shape-{kind}-{fraction_tag(fraction)}.json"),
              "w") as handle:
        json.dump({"root": kind, "fraction": fraction, **shape}, handle)
    print(f" ...open: {shape['n_folders']} log folders, {shape['n_lines']:,} lines")

    # Detail cells first: grid_cells' own list ends with set_folder_names and
    # close_log_root (session must go last), and detail cells need the same
    # still-open, still-original-names session grid_cells' other rows do.
    record_cells(grid_detail_cells(ctx), kind, fraction, cell_dir, inflight_path)
    record_cells(grid_cells(ctx), kind, fraction, cell_dir, inflight_path)

    if ctx.session is not None:
        server.close_log_root(ctx.sid)
    gc.collect()


def cell_pending(out_path):
    """Whether this cell still has to be measured.

    A cell recorded before memory instrumentation existed has a status but no
    memory fields -- redo it rather than skip it, so an old PERFORMANCE.md-only
    cache grows PERF_MEMORY.md data in place. An "error"/"oom" cell never called
    ``run()`` and never will produce memory either, so those stay skipped.

    A cell that has ``*_mem_gb`` but no ``*_floor_gb`` predates the floor
    measurement and is **kept**: the floor is not rendered anyway, so the cell
    prints exactly what a fresh one would. Forcing those to be redone would
    mean re-measuring the whole grid -- hours, and an OOM relaunch or two --
    every time a field is added, which is a decision for whoever is running it.
    Clear the cell directory to ask for that deliberately.
    """
    if not os.path.isfile(out_path):
        return True
    with open(out_path) as handle:
        existing = json.load(handle)
    return existing["status"] == "ok" and "cold_mem_gb" not in existing


def record_cells(cells, kind, fraction, cell_dir, inflight_path):
    """Measure and record each cell, skipping the ones already on disk."""
    for table, label, run in cells:
        out_path = os.path.join(cell_dir, cell_slug(kind, fraction, label) + ".json")
        if not cell_pending(out_path):
            continue
        with open(inflight_path, "w") as handle:
            json.dump({"root": kind, "fraction": fraction, "table": table,
                       "tool": label, "path": out_path}, handle)
        record = {"root": kind, "fraction": fraction, "table": table, "tool": label,
                  "status": "ok"}
        started = time.perf_counter()
        try:
            cold, warm = run()
            record.update(cold_s=round(cold.seconds, 4), warm_s=round(warm.seconds, 4),
                          cold_mem_gb=round(cold.peak_gb, 3),
                          warm_mem_gb=round(warm.peak_gb, 3),
                          cold_floor_gb=round(cold.floor_gb, 3),
                          warm_floor_gb=round(warm.floor_gb, 3),
                          band=band(warm.seconds))
            print(f"  {label:<38} cold {cold.seconds:8.3f}s {cold.peak_gb:6.2f}GB  "
                  f"warm {warm.seconds:8.3f}s {warm.peak_gb:6.2f}GB  {band(warm.seconds)}")
        except Exception as exc:  # a tool that cannot run on this shape is data
            record.update(status="error", error=f"{type(exc).__name__}: {exc}",
                          cold_s=round(time.perf_counter() - started, 4))
            print(f"  {label:<38} ERROR {type(exc).__name__}: {exc}")
        with open(out_path, "w") as handle:
            json.dump(record, handle)
        os.remove(inflight_path)


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


def _fmt_gb(record, key):
    """One number per cell: the peak.

    The floor each call started from is recorded too (see :class:`Measure`) and
    stays in the cell JSON, but it is deliberately **not** rendered here. A
    grid is read by scanning a column for the number that stands out, and a
    second figure in every cell is what stops that working. Do not put it back.
    """
    if record is None:
        return ""
    if record["status"] == "oom":
        return "OOM"
    if record["status"] == "error":
        return "err"
    value = record.get(key)
    if value is None:
        return ""
    return f"{value:.3f}" if value < 1 else f"{value:.2f}"


def _grid_header(shapes, fractions, roots, title, intro_lines):
    """Title, prose, and the "log roots at each fraction" table -- printed once
    per file, ahead of however many ``_grid_parts`` sections follow it."""
    def shape_cell(kind, fraction):
        shape = shapes.get((kind, fraction))
        if not shape:
            return ""
        if kind == "bgl":
            per = shape["n_lines"] // shape["n_folders"]
            return f"{shape['n_lines']:,} lines (10 x {per:,})"
        return f"{shape['n_folders']:,} folders / {shape['n_files']:,} files"

    lines = [f"# {title}", "", *intro_lines, "",
             "## Log roots at each fraction", "",
             "| log root | unit varied | " + " | ".join(f"{f:.0%}" for f in fractions) + " |",
             "|---|---|" + "---|" * len(fractions)]
    for kind in roots:
        varied = "log lines" if kind == "bgl" else "log folders"
        cells = " | ".join(shape_cell(kind, f) for f in fractions)
        lines.append(f"| {ROOT_LABELS[kind]} | {varied} | {cells} |")
    return lines


def _grid_parts(records, fractions, roots, keys, table_titles, grid_rows, fmt):
    """One or more "Part X" sections -- each a set of tables, one per entry in
    ``table_titles``, over the rows in ``grid_rows`` that belong to it.

    Shared by the main grid (``GRID_TABLE_TITLES``/``GRID_ROWS``, Part A/B) and
    the per-detector/per-measure detail grid (``DETAIL_TABLE_TITLES``/
    ``DETAIL_GRID_ROWS``, Part C/D) -- same recorded cells, same lookup, only
    which rows and which tables differ.
    """
    lines = []
    for part, key, part_title in keys:
        lines += ["", f"# Part {part} -- {part_title}"]
        for index, (table, table_title) in enumerate(table_titles, start=1):
            lines += ["", f"## Table {part}{index} -- {table_title}", "",
                      "| tool | log root | " + " | ".join(f"{f:.0%}" for f in fractions) + " |",
                      "|---|---|" + "---|" * len(fractions)]
            for row_table, label in grid_rows:
                if row_table != table:
                    continue
                for kind in roots:
                    cells = " | ".join(
                        fmt(records.get((kind, f, label)), key) for f in fractions)
                    lines.append(f"| {label} | {ROOT_LABELS[kind]} | {cells} |")
    return lines


#: Prose introducing Part C/D, printed once between the main grid (Part A/B)
#: and the detail grid -- what a fraction means (``_FRACTION_INTRO``) does not
#: need repeating, but what "detailed" narrows down does.
_DETAIL_INTRO = [
    "# Detailed breakdowns (per detector / per measure)",
    "",
    "The tables above run every anomaly tool with all four detectors, and "
    "`distance_folder_content`/`distance_file_content` with all four measures, "
    "at once. Part C/D below break the same figure down per detector / per "
    "measure run in isolation (`detectors=[\"<name>\"]` / "
    "`measures=[\"<name>\"]`), so the cost of narrowing either is visible on "
    "its own rather than folded into the combined call. `distance_folder_filename` "
    "(jaccard/overlap distance over file names only) and `distance_line_content` "
    "(a text diff, no measures) are not broken down further -- neither computes "
    "multiple vectorized measures in one pass.",
]


#: Prose shared by both markdown writers -- what a fraction means is a property
#: of the grid, not of the metric being read off it.
_FRACTION_INTRO = [
    'What "5%" means differs by log root: for `hadoop_renamed` and '
    "`hdfs_balanced_5k` it is 5% of the",
    "**log folders** (and their files); for `bgl_split_10` it is 5% of "
    "`BGL.log`'s **log lines**, taken",
    "first and then split into 10 slices, so the folder count is always 10 "
    "and what varies is the text",
    "inside each one.",
]


def performance_markdown(records, shapes, fractions=FRACTIONS,
                         roots=("hadoop", "hdfs", "bgl")):
    """Fill PERFORMANCE.md's eight grids from the recorded cells."""
    intro = [
        "Cells: seconds. Columns: the four data fractions.",
        "",
        *_FRACTION_INTRO,
        "",
        "Tables A1-A4 are **cold** (first call, nothing cached). Tables B1-B4 "
        "are the same grid **warm**",
        "(repeated call, same process).",
    ]
    lines = _grid_header(shapes, fractions, roots, "MCP tool performance", intro)
    lines += _grid_parts(records, fractions, roots,
                         (("A", "cold_s", "cold (first call)"),
                          ("B", "warm_s", "warm (repeated call)")),
                         GRID_TABLE_TITLES, GRID_ROWS, _fmt_seconds)
    lines += ["", *_DETAIL_INTRO]
    lines += _grid_parts(records, fractions, roots,
                         (("C", "cold_s", "cold (first call)"),
                          ("D", "warm_s", "warm (repeated call)")),
                         DETAIL_TABLE_TITLES, DETAIL_GRID_ROWS, _fmt_seconds)
    return "\n".join(lines) + "\n"


def memory_markdown(records, shapes, fractions=FRACTIONS,
                    roots=("hadoop", "hdfs", "bgl")):
    """Fill PERF_MEMORY.md's eight grids from the recorded cells.

    Same grid as :func:`performance_markdown`, same recorded cells -- each cell
    already carries both its timing and its peak resident memory (see
    :func:`measure_adaptive`), so this reads `*_mem_gb` where that one reads
    `*_s` rather than measuring anything new.
    """
    intro = [
        "Cells: GB, peak resident memory sampled every 20ms while the call ran. "
        "Columns: the four",
        "data fractions.",
        "",
        *_FRACTION_INTRO,
        "",
        "Tables A1-A4 are **cold** (first call, nothing cached). Tables B1-B4 "
        "are the same grid",
        "**warm** -- the max across the repeats, same process, rather than the "
        "median: a peak that",
        "only showed up once is still the one a client should plan for.",
        "",
        "**A cell is the whole process's peak while the call ran, not the "
        "call's own allocation.**",
        "Every cell of a block runs with the log root already open, so the "
        "session's frame is",
        "resident underneath -- on bgl at 100% that is ~3GB before any tool is "
        "called, and it is why",
        "the aux column climbs down a bgl column. Read a cell as what a machine "
        "running this call on",
        "this log root needs, which is the number that OOM-kills.",
        "",
        "`peek_log_root` and `split_log_file` are the exception: they are "
        "measured **before** the",
        "block opens anything, since neither needs a session. Peek stats the "
        "files and reads a few",
        "hundred lines; split streams its file through an 8MB buffer. Their "
        "floor is the server's own",
        "imports (~0.31GB: polars, sklearn, plotly, the MCP SDK), paid once at "
        "startup and shared by",
        "every tool. Measured *after* the open, as they were in the first grid "
        "to record memory, peek",
        "read 5.14GB on bgl at 100% -- all but ~0.02 of it the frame sitting "
        "beside it. Neither",
        "scales with the data: peek is ~20-30MB from 10 log folders to 5,000 "
        "and from 0.01GB to",
        "0.74GB of logs, because it counts files and samples lines rather than "
        "reading them.",
        "",
        "Cells recorded before `loglead.delta` stopped importing umap eagerly "
        "read ~0.2GB high. Clear",
        "the cell directory to re-measure the grid from scratch.",
    ]
    lines = _grid_header(shapes, fractions, roots, "MCP tool memory", intro)
    lines += _grid_parts(records, fractions, roots,
                         (("A", "cold_mem_gb", "cold (first call)"),
                          ("B", "warm_mem_gb", "warm (repeated call)")),
                         GRID_TABLE_TITLES, GRID_ROWS, _fmt_gb)
    lines += ["", *_DETAIL_INTRO]
    lines += _grid_parts(records, fractions, roots,
                         (("C", "cold_mem_gb", "cold (first call)"),
                          ("D", "warm_mem_gb", "warm (repeated call)")),
                         DETAIL_TABLE_TITLES, DETAIL_GRID_ROWS, _fmt_gb)
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
            for attempt in range(len(GRID_ROWS) + len(DETAIL_GRID_ROWS) + 2):
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
    here = os.path.dirname(os.path.abspath(__file__))
    perf_path = args.performance or os.path.join(here, "PERFORMANCE.md")
    mem_path = args.memory or os.path.join(here, "PERF_MEMORY.md")
    Path(perf_path).write_text(performance_markdown(records, shapes, fractions, roots))
    Path(mem_path).write_text(memory_markdown(records, shapes, fractions, roots))
    print(f"\nCells: {len(records)} recorded in {cell_dir}"
          f"\nTables -> {perf_path}\nTables -> {mem_path}")


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default=None,
                        help="Where the log roots live. Defaults to root_folder in "
                             "downloader/datasets.yml.")
    parser.add_argument("--only", choices=("hadoop", "hdfs", "bgl"), action="append",
                        dest="roots", help="Benchmark one log root; repeatable. 'bgl' needs "
                                           "<datasets>/bgl/BGL.log and writes its reduced copies "
                                           "and slices -- 743 MB at 100%% -- into "
                                           "<datasets>/test_data/mcp_bench_reduced.")
    parser.add_argument("--repeat", type=int, default=3,
                        help="Warm runs per measurement; the median is reported (default 3).")
    parser.add_argument("--cache-dir", default=None,
                        help="Parquet cache. Defaults to <datasets>/test_data/mcp_cache.")
    parser.add_argument("--fractions", type=float, nargs="+", default=None,
                        help=f"Fractions to measure at (default "
                             f"{' '.join(str(f) for f in FRACTIONS)}).")
    parser.add_argument("--performance", default=None,
                        help="Where the timing tables are written "
                             "(default tests/mcp/PERFORMANCE.md).")
    parser.add_argument("--memory", default=None,
                        help="Where the memory tables are written "
                             "(default tests/mcp/PERF_MEMORY.md).")
    parser.add_argument("--cell-dir", default=None,
                        help="Where one JSON per measured cell is kept, so a killed "
                             "block resumes. Defaults to <datasets>/test_data/mcp_bench_cells.")
    parser.add_argument("--tables-only", action="store_true",
                        help="Rebuild the tables from the recorded cells without "
                             "measuring anything.")
    # Set when the parent launches a child for one (log root, fraction). Not
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
        here = os.path.dirname(os.path.abspath(__file__))
        perf_path = args.performance or os.path.join(here, "PERFORMANCE.md")
        mem_path = args.memory or os.path.join(here, "PERF_MEMORY.md")
        fractions = tuple(args.fractions or FRACTIONS)
        Path(perf_path).write_text(performance_markdown(records, shapes, fractions, roots))
        Path(mem_path).write_text(memory_markdown(records, shapes, fractions, roots))
        print(f"Cells: {len(records)} recorded in {cell_dir}"
              f"\nTables -> {perf_path}\nTables -> {mem_path}")
        return

    # Building the two derived log roots is minutes of work, and neither 'bgl'
    # nor a grid child needs it -- the child is handed its log root.
    paths = (make_test_data.ensure_datasets(dest_folder=datasets_folder)
             if {"hadoop", "hdfs"} & set(roots) and not args.grid_block else {})

    workdir = tempfile.mkdtemp(prefix="loglead-mcp-bench-")
    server.STORE = SessionStore(cache_dir=cache_dir,
                                output_root=os.path.join(workdir, "output"))
    print(f"Cache:   {cache_dir}\nWorkdir: {workdir}\nRepeats: {args.repeat}")

    try:
        if args.grid_block:
            kind, fraction = args.grid_block.split(":")
            run_grid_block(kind, float(fraction), args.grid_block_path, workdir,
                           args.repeat, args.cell_dir,
                           json.loads(args.grid_block_shape))
        else:
            grid_main(args, datasets_folder, paths, cache_dir, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
