"""Tests for ``loglead/mcp/server.py``: every tool, against two real log roots.

Like the rest of ``tests/``, this is a plain script rather than a pytest suite --
run it and read the output. Every check prints ``ok`` or ``FAIL``, sections keep
going after a failure so one broken tool does not hide the other nineteen, and
the exit code is non-zero if anything failed.

Run it with::

    uv run tests/mcp/server.py                     # everything
    uv run tests/mcp/server.py --only hadoop       # one stage: data, hadoop, hdfs, split, detect, crash
    uv run tests/mcp/server.py --regenerate        # rebuild the log roots first
    uv run tests/mcp/server.py --keep-artifacts    # keep the tables and plots written

**The data comes first.** The two log roots are derived from public loghub
datasets, not downloadable as such, so ``make_test_data.py`` next to this file
rebuilds them from ``~/Datasets/hadoop`` and ``~/Datasets/hdfs`` (downloading
those if needed), and stage 1 here runs it and checks what it produced. Nothing
else runs until that passes.

**Why these two.** They are the two shapes a log root comes in, and most of what
this file checks only shows up in one of them:

* ``hadoop_renamed`` -- 55 log folders, 978 files, one folder per test run. The
  file-level and line-level tools need same-named files across folders (which
  is what ``file_name_normalizer="strip_folder_id"`` provides), and the folder
  names carry a ground-truth label, so an unsupervised ranking can be scored
  against something.
* ``hdfs_balanced_5k`` -- 5,000 log folders of one file each. This is where
  scale-dependent behaviour lives: the plot tools returning a summary instead of
  5,000 rows, the caution the file-name plot emits when every folder holds one
  file, and paging a 5,000-row result. Its names are labelled too
  (``Anomaly_``/``Normal_``).

**What is asserted and what is only reported.** Every count is asserted exactly,
because both log roots are byte-identical wherever ``make_test_data.py`` built
them: 55 log folders / 978 files / 180,897 lines for Hadoop, and 5,000 log
folders / 90,862 lines for HDFS, the latter pinned in ``make_test_data.py``
alongside a digest of *which* 5,000 blocks they are, so a differently-sampled
copy is reported as such instead of failing a line count with no explanation.
Detector output is the one thing checked as a ranking rather than a value: two
identical runs differ in the third significant figure, so the labelled checks ask
whether anomalous log folders rank above normal ones (``rank_auc``, with margin
against the observed spread) and never what they scored. Timings are printed, not
asserted.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy  # noqa: E402
import polars as pl  # noqa: E402
import make_test_data  # noqa: E402  (sits next to this file)
from loglead import loaders  # noqa: E402
from loglead.delta import log_root, split, visualize, vocabulary  # noqa: E402

try:  # the MCP server is an optional extra, and its absence is not a test failure
    from loglead.mcp import server  # noqa: E402
    from loglead.mcp.session import SessionStore  # noqa: E402
except ImportError as exc:
    raise SystemExit(
        f"Cannot import loglead.mcp ({exc}). It is an optional install: uv sync --extra mcp"
    ) from exc

# --------------------------------------------------------------------------- #
# Expected values that follow from the public datasets, so a change here means
# either the data is wrong or LogLead reads it differently than it used to.
# --------------------------------------------------------------------------- #

#: Hadoop: 55 labelled applications, 978 container logs between them.
HADOOP_FOLDERS = 55
HADOOP_FILES_ON_DISK = 978
#: Lines after AutoLoader folds stack-trace continuations into the event that
#: printed them. Moves only when the reading changes -- see
#: SessionStore._PREPROCESSING_VERSION.
HADOOP_ROWS = 180897
#: Distinct file names once strip_folder_id has removed the application id, i.e.
#: how many files can be compared across log folders. Without the normalizer
#: this would be 978, one per folder, and every L3/L4 result would be empty.
HADOOP_SHARED_FILES = 47
#: What AutoLoader should make of Hadoop's log4j lines and HDFS's.
HADOOP_FORMAT = "text/%Y-%m-%d %H:%M:%S,%3f"
HDFS_FORMAT = "text/%y%m%d %H%M%S"

#: HDFS is a sample, but a pinned one -- the generator records which 5,000 blocks
#: it builds and how many lines they hold, and stage 1 verifies both, so these are
#: as exact as Hadoop's.
HDFS_FOLDERS = 2 * make_test_data.BLOCKS_PER_CLASS
HDFS_ROWS = make_test_data.EXPECTED_LINES

#: BGL is one 743 MB file of 4,747,963 lines -- the shape split_log_file exists
#: for. Unlike the two derived log roots it is a plain loghub download, so this
#: count is a property of the dataset rather than of a generator.
BGL_LINES = 4747963

#: The tools an MCP client should see. A function here that lost its @tool
#: decorator still works from Python and simply disappears from every client,
#: which is the one failure nothing else in this file would notice.
TOOLS = (
    "open_log_root", "list_log_roots", "describe_log_root", "set_folder_names",
    "close_log_root", "read_log_lines", "search_log_lines", "query_result",
    "distance_folder_filename", "distance_folder_content", "distance_file_content",
    "distance_line_content", "anomaly_folder_filename", "anomaly_folder_content",
    "anomaly_file_content", "anomaly_line_content", "plot_folder_filename",
    "plot_folder_content", "plot_file_content", "run_config",
    "peek_log_root", "split_log_file",
    "register_mask_pattern", "list_mask_patterns", "remask_log_root", "new_tokens",
)


# --------------------------------------------------------------------------- #
# A very small check harness
# --------------------------------------------------------------------------- #

class Checks:
    """Counts passes and failures and prints one line per check."""

    def __init__(self, verbose=True):
        self.passed = 0
        self.failures = []
        self.verbose = verbose
        self.section_name = "-"

    def section(self, name):
        self.section_name = name
        print(f"\n{'=' * 78}\n {name}\n{'=' * 78}")

    def ok(self, name, condition, detail=""):
        detail = detail if len(str(detail)) <= 120 else str(detail)[:117] + "..."
        if condition:
            self.passed += 1
            if self.verbose:
                print(f"  ok   {name}" + (f"  [{detail}]" if detail else ""))
        else:
            self.failures.append((self.section_name, name, detail))
            print(f"  FAIL {name}" + (f"  [{detail}]" if detail else ""))
        return bool(condition)

    def eq(self, name, actual, expected):
        # Only say what was expected when it did not happen: several of these
        # compare 55-element lists, and printing both on success is unreadable.
        same = actual == expected
        return self.ok(name, same, "" if same else f"got {actual!r}, expected {expected!r}")

    def raises(self, name, exception, call, *args, **kwargs):
        """Check that a tool rejects bad input instead of doing something odd."""
        try:
            call(*args, **kwargs)
        except exception as exc:
            return self.ok(name, True, str(exc).split("\n")[0][:70])
        except Exception as exc:  # wrong type is still a failure worth seeing
            return self.ok(name, False, f"raised {type(exc).__name__}: {exc}")
        return self.ok(name, False, "no exception raised")

    def info(self, message):
        print(f"       {message}")

    def report(self):
        print(f"\n{'=' * 78}")
        if self.failures:
            print(f" {len(self.failures)} FAILED, {self.passed} passed")
            for section, name, detail in self.failures:
                print(f"   {section}: {name}" + (f"  [{detail}]" if detail else ""))
        else:
            print(f" All {self.passed} checks passed")
        print("=" * 78)
        return 1 if self.failures else 0


def timed(label, call, *args, **kwargs):
    """Run a tool, printing what it cost. Timings are reported, never asserted."""
    started = time.time()
    result = call(*args, **kwargs)
    print(f"       {label}: {time.time() - started:.1f}s")
    return result


def rank_auc(labelled, ordinary):
    """How well two scored groups separate: 1.0 is perfect, 0.5 is chance.

    The detectors are unsupervised and their scores are not reproducible to the
    digit (KMeans and IsolationForest are built without a random_state, and
    sklearn's threaded KMeans is not bit-stable anyway), so the labelled checks
    ask whether the *ranking* puts anomalies above normals. This is the
    Mann-Whitney statistic -- the share of (anomalous, normal) pairs in the right
    order -- which is far steadier than "how many of the top ten": on
    hdfs_balanced_5k that count moves between 7 and 10 across runs while this
    stays within 0.84-0.91, and on hadoop_renamed, where 44 of 55 log folders
    are failures, a top-ten count of 8 is what chance alone produces.
    """
    pairs = [(one > other) + 0.5 * (one == other)
             for one in labelled for other in ordinary]
    return sum(pairs) / len(pairs) if pairs else float("nan")


# --------------------------------------------------------------------------- #
# Stage 0 -- what a client actually sees
# --------------------------------------------------------------------------- #

def stage_tools(check):
    """Every tool is callable from Python *and* registered with the MCP server.

    The two are separate: ``@tool`` returns the undecorated function, so calling
    one directly (as the rest of this file, ``run_config`` and the demo do) works
    whether or not it was ever registered. Only a client would notice the
    difference, and there is no client here.
    """
    check.section("0. tool surface")

    missing = [name for name in TOOLS if not callable(getattr(server, name, None))]
    check.ok(f"all {len(TOOLS)} tools are importable", not missing,
             str(missing) if missing else "")

    # Every tool goes through the same decorator, which is what guarantees the
    # stdout redirect and the elapsed_seconds every result carries. A tool
    # defined without it would still work and would silently have neither.
    undecorated = [name for name in TOOLS
                   if getattr(getattr(server, name, None), "__wrapped__", None) is None]
    check.ok("all of them go through the @tool wrapper", not undecorated,
             str(undecorated) if undecorated else "")

    # SDK 2.x keeps them in a tool manager; older ones may not expose it at all,
    # in which case there is nothing to check rather than something to fail.
    manager = getattr(server.mcp, "_tool_manager", None)
    if manager is None or not hasattr(manager, "list_tools"):
        check.info(f"{type(server.mcp).__name__} exposes no tool registry; "
                   "registration not checked")
        return
    registered = {tool.name for tool in manager.list_tools()}
    check.eq("...and all of them are registered with the MCP server",
             sorted(registered), sorted(TOOLS))
    described = [tool.name for tool in manager.list_tools() if not tool.description]
    check.ok("every tool has a description, which is all a client gets",
             not described, str(described) if described else "")


# --------------------------------------------------------------------------- #
# Stage 1 -- the test data itself
# --------------------------------------------------------------------------- #

def stage_data(check, datasets_folder, regenerate):
    """Rebuild the two log roots if needed, then verify what is on disk.

    This runs first because everything below reads these directories; a missing
    or half-built one would otherwise surface as a confusing analysis failure
    twenty checks later.
    """
    check.section("1. test data")
    paths = make_test_data.ensure_datasets(dest_folder=datasets_folder, force=regenerate)
    hadoop = paths[make_test_data.HADOOP_RENAMED]
    hdfs = paths[make_test_data.HDFS_BALANCED_5K]

    # -- hadoop_renamed: fully determined by the public dataset --
    folders = sorted(entry.name for entry in os.scandir(hadoop) if entry.is_dir())
    check.eq("hadoop_renamed log folders", len(folders), HADOOP_FOLDERS)
    naming = re.compile(r"^(WordCount|PageRank)_"
                        r"(Normal|MachineDown|NetworkDisconnection|DiskFull)_"
                        r"application_\d+_\d+$")
    check.ok("every folder name carries its application and failure",
             all(naming.match(name) for name in folders),
             f"e.g. {folders[0]}")
    # The names have to agree with the label file they were built from, or every
    # labelled check further down is scoring against fiction. The label file is
    # copied in beside the log folders for exactly this.
    labels = make_test_data.read_hadoop_labels(hadoop / "abnormal_label.txt")
    check.eq("folder names match abnormal_label.txt",
             folders, sorted(f"{label}_{app}" for app, label in labels.items()))
    n_files = sum(len([f for f in os.listdir(hadoop / name) if f.endswith(".log")])
                  for name in folders)
    check.eq("hadoop_renamed .log files", n_files, HADOOP_FILES_ON_DISK)
    normals = [name for name in folders if "_Normal_" in name]
    check.eq("hadoop_renamed normal log folders", len(normals), 11)

    # -- hdfs_balanced_5k: a sample, but a pinned one --
    files = sorted(entry.name for entry in os.scandir(hdfs) if entry.is_file())
    check.eq("hdfs_balanced_5k files", len(files), HDFS_FOLDERS)
    for label in ("Anomaly", "Normal"):
        check.eq(f"hdfs_balanced_5k {label} files",
                 sum(1 for name in files if name.startswith(f"{label}_blk_")),
                 make_test_data.BLOCKS_PER_CLASS)
    # Which 5,000 blocks, not just how many: this is what makes the exact line
    # counts below meaningful on a machine that built its own copy.
    check.eq("...and they are the pinned 5,000 blocks",
             make_test_data.sample_digest(files), make_test_data.EXPECTED_SAMPLE_DIGEST)
    check.eq("...holding the expected number of lines",
             make_test_data.count_lines(hdfs / name for name in files), HDFS_ROWS)
    check.ok("no empty log files",
             all(os.path.getsize(hdfs / name) > 0 for name in files))
    # Every line of a block's file must actually name that block -- the whole
    # basis of the split. Checked on a sample; reading all 5,000 is stage 2's job.
    sample = files[::200]
    wrong = []
    for name in sample:
        block = name.split("_", 1)[1][:-len(".log")]
        with open(hdfs / name) as handle:
            if any(block not in line for line in handle):
                wrong.append(name)
    check.ok(f"every line names its own block ({len(sample)} files sampled)",
             not wrong, f"bad: {wrong[:3]}" if wrong else "")

    return hadoop, hdfs


# --------------------------------------------------------------------------- #
# Stage 2 -- hadoop_renamed: sessions, drill-down, and all 20 tools
# --------------------------------------------------------------------------- #

def stage_hadoop_open(check, log_root, session_id):
    """open_log_root, its cache, and the arguments it refuses."""
    check.section("2. open_log_root (hadoop_renamed)")

    # The cheap call that comes before the expensive one. What a client reads off
    # it here is that 978 files share one name shape, which is what makes reading
    # them all as one format a safe default.
    peeked = timed("peek_log_root", server.peek_log_root, str(log_root))
    check.eq("peek counts the files without reading them",
             (peeked["n_files"], peeked["n_folders"]), (HADOOP_FILES_ON_DISK, HADOOP_FOLDERS))
    check.eq("...and finds one file-name shape behind all of them",
             [(entry["name_shape"], entry["n_files"]) for entry in peeked["file_names"]],
             [("container_#_#_#_#.log", HADOOP_FILES_ON_DISK)])
    check.eq("...though every file is named differently",
             peeked["n_distinct_file_names"], HADOOP_FILES_ON_DISK)
    check.ok("...so it says one sample can speak for the log root",
             any("named like" in note for note in peeked["notes"]),
             str(peeked["notes"])[:160])

    # refresh=True forces the read even if a cache is lying around, so the cold
    # path is exercised on every run and the cache check below means something.
    info = timed("cold read", server.open_log_root,
                 path=str(log_root), format="auto", mask=True, parsers=["tip"],
                 file_name_normalizer="strip_folder_id", session_id=session_id,
                 refresh=True)
    check.eq("cache_hit on a forced read", info["cache_hit"], False)
    check.eq("n_folders", info["n_folders"], HADOOP_FOLDERS)
    check.eq("n_files_on_disk", info["n_files_on_disk"], HADOOP_FILES_ON_DISK)
    check.eq("n_rows", info["n_rows"], HADOOP_ROWS)
    check.eq("file names shared across log folders (strip_folder_id)",
             info["n_files"], HADOOP_SHARED_FILES)
    check.eq("dropped_rows", info["dropped_rows"], 0)
    check.eq("AutoLoader detected log4j text",
             info["detected_formats"], {HADOOP_FORMAT: HADOOP_FILES_ON_DISK})
    # 978 files agreeing is what the count above says; 50 of them being looked at
    # is what made the call seconds rather than twenty of them.
    check.eq("...from a sample of the files, not all of them",
             info["probed_files"], loaders.DEFAULT_MAX_DETECT_FILES)
    check.ok("...and says which of the two it did",
             any("detected from 50 of the 978" in note for note in info.get("notes", [])),
             str(info.get("notes"))[:160])
    check.eq("parsers ran at open time", info["parsers"], ["tip"])
    check.ok("masking produced e_message_normalized",
             "e_message_normalized" in info["enhanced_columns"],
             str(info["enhanced_columns"]))
    check.ok("folders listing is capped at 50", len(info["folders"]) == 50)
    check.ok("...and says so in a note",
             any("first 50" in note for note in info.get("notes", [])))
    # Every result says what it cost, since a caller over MCP has no clock of its
    # own and these tools range from 0.02s to tens of minutes.
    check.ok("the result reports what the call cost",
             isinstance(info.get("elapsed_seconds"), (int, float))
             and info["elapsed_seconds"] > 0,
             f"elapsed_seconds={info.get('elapsed_seconds')!r}")

    # A second open with the same arguments must come off the parquet, and say
    # it did -- this is the entire reason SessionStore exists.
    warm = timed("cached read", server.open_log_root,
                 path=str(log_root), format="auto", mask=True, parsers=["tip"],
                 file_name_normalizer="strip_folder_id", session_id=session_id + "-warm")
    check.eq("second open hits the cache", warm["cache_hit"], True)
    check.eq("cached frame is the same size", warm["n_rows"], HADOOP_ROWS)
    check.eq("nothing was read, so nothing was detected", warm["detected_formats"], {})

    # Any argument that changes the frame has to change the cache key.
    other = timed("different format", server.open_log_root,
                  path=str(log_root), format="raw", mask=True, parsers=["tip"],
                  file_name_normalizer="strip_folder_id", session_id=session_id + "-raw")
    # The key, not the hit: this entry survives between runs, so a cache hit here
    # is expected on the second run and says nothing either way.
    check.ok("a different format is a different cache entry",
             other["cache_path"] != info["cache_path"],
             f"both at {other['cache_path']}")
    check.ok("raw reads every line as its own event, so it has more rows",
             other["n_rows"] > HADOOP_ROWS, f"raw {other['n_rows']} vs auto {HADOOP_ROWS}")
    server.close_log_root(session_id + "-raw")
    server.close_log_root(session_id + "-warm")

    # Everything cheap is validated before a single log file is read.
    check.raises("missing log root is rejected", FileNotFoundError,
                 server.open_log_root, path="/nonexistent/log/root")
    check.raises("duplicate session_id is rejected", ValueError,
                 server.open_log_root, path=str(log_root), session_id=session_id)
    check.raises("unknown mask pattern is rejected", ValueError,
                 server.open_log_root, path=str(log_root), mask_pattern="no_such_pattern")
    check.raises("unknown file_name_normalizer is rejected", ValueError,
                 server.open_log_root, path=str(log_root), file_name_normalizer="nope")
    check.raises("unknown format is rejected", ValueError,
                 server.open_log_root, path=str(log_root), format="json/no_such_spec")
    check.raises("unknown table_format is rejected", ValueError,
                 server.open_log_root, path=str(log_root), table_format="parquet")
    check.raises("a pattern matching no file is rejected", FileNotFoundError,
                 server.open_log_root, path=str(log_root), filename_pattern="*.nosuchext")


def stage_hadoop_describe(check, session_id, log_root):
    """list_log_roots, describe_log_root, set_folder_names."""
    check.section("3. describe_log_root / list_log_roots / set_folder_names")

    listed = server.list_log_roots()["sessions"]
    check.ok("the open session is listed",
             session_id in [s["session_id"] for s in listed],
             f"{len(listed)} session(s)")

    described = server.describe_log_root(session_id)
    check.eq("one detail row per log folder",
             len(described["folders_detail"]), HADOOP_FOLDERS)
    check.eq("per-folder line counts sum to n_rows",
             sum(row["n_lines"] for row in described["folders_detail"]), HADOOP_ROWS)

    with_files = server.describe_log_root(session_id, include_files=True)
    check.eq("one detail row per distinct file name",
             len(with_files["files_detail"]), HADOOP_SHARED_FILES)
    shared = [row for row in with_files["files_detail"] if row["n_folders"] > 1]
    check.ok("most file names appear in several log folders",
             len(shared) > HADOOP_SHARED_FILES / 2,
             f"{len(shared)} of {HADOOP_SHARED_FILES} shared")
    check.ok("include_files explains what a shared file is for",
             any("nothing to compare" in note for note in with_files.get("notes", [])))

    check.raises("an unknown session is rejected", ValueError,
                 server.describe_log_root, "no-such-session")

    # Renaming happens in a session of its own: it rewrites the folder column,
    # and every later stage here selects log folders by name.
    naming_id = session_id + "-naming"
    victim, second = server.STORE.get(session_id).folders[:2]
    opened = server.open_log_root(path=str(log_root), format="auto", mask=True,
                                  file_name_normalizer="strip_folder_id",
                                  folder_names={victim: "Named"}, session_id=naming_id)
    check.eq("folder_names at open time is applied", opened["n_named_folders"], 1)
    check.ok("...to the folder it named",
             f"Named_{victim}" in opened["folders"], str(opened["folders"][:2]))

    renamed = server.set_folder_names(naming_id, {second: "Renamed"},
                                      keep_original_folder_name=True)
    check.eq("one log folder named", renamed["named"], 1)
    check.eq("the rest kept their names", renamed["unnamed"], HADOOP_FOLDERS - 1)
    check.ok("the new name keeps the directory name as a suffix",
             f"Renamed_{second}" in renamed["folders"])
    check.ok("names are applied to the directory name, so the open-time one is replaced",
             f"Named_{victim}" not in renamed["folders"] and victim in renamed["folders"])
    check.ok("the note says how many kept their directory name",
             any("kept their directory name" in note for note in renamed["notes"]))

    bare = server.set_folder_names(naming_id, {second: "Renamed2"},
                                   keep_original_folder_name=False)
    check.ok("keep_original=False gives the bare name, and does not stack",
             "Renamed2" in bare["folders"]
             and not any(name.startswith("Renamed_") for name in bare["folders"]))
    detail = server.describe_log_root(naming_id)["folders_detail"]
    check.ok("folder_original still points at the directory it was read from",
             any(row.get("folder_original") == second for row in detail))

    # An empty mapping is the reset. Reading it as a no-op would leave the
    # previous names on a frame that is then cached under the key for *no*
    # names, so the next plain open of these logs is served renamed folders.
    cleared = server.set_folder_names(naming_id, {})
    check.eq("clearing the mapping names nothing", cleared["named"], 0)
    check.ok("...and puts every directory name back",
             second in cleared["folders"] and victim in cleared["folders"]
             and not any(name.startswith(("Named", "Renamed")) for name in cleared["folders"]),
             str([n for n in cleared["folders"] if n.startswith(("Named", "Renamed"))])[:120]
             if any(n.startswith(("Named", "Renamed")) for n in cleared["folders"]) else "")

    check.raises("naming an unknown log folder is rejected", ValueError,
                 server.set_folder_names, naming_id, {"no_such_folder": "X"})
    check.raises("a name that would collide with another is rejected", ValueError,
                 server.set_folder_names, naming_id,
                 {victim: "Same", second: "Same"}, False)
    server.close_log_root(naming_id)


def stage_hadoop_read(check, session_id, target):
    """read_log_lines and search_log_lines -- the raw evidence behind a score."""
    check.section("4. read_log_lines / search_log_lines")

    described = server.describe_log_root(session_id, include_files=True)
    file_name = max(described["files_detail"], key=lambda row: row["n_folders"])["file_name"]
    check.info(f"target log folder {target}, file {file_name}")

    lines = server.read_log_lines(session_id, target, file_name, offset=0, limit=5)
    check.eq("5 lines returned", lines["returned"], 5)
    check.ok("line_number starts at the offset",
             lines["lines"][0]["line_number"] == 0)
    check.ok("raw text comes back in m_message",
             all("m_message" in line for line in lines["lines"]))
    later = server.read_log_lines(session_id, target, file_name, offset=3, limit=2)
    check.ok("offset pages through the same file",
             later["lines"][0]["m_message"] == lines["lines"][3]["m_message"])
    masked = server.read_log_lines(session_id, target, file_name, limit=5, masked=True)
    check.ok("masked=True returns the normalized text",
             all("e_message_normalized" in line for line in masked["lines"]))
    capped = server.read_log_lines(session_id, target, file_name, limit=10_000)
    check.ok("limit is capped at 500", capped["returned"] <= 500,
             f"returned {capped['returned']} of {capped['total_lines']}")
    check.raises("an unknown file is rejected", ValueError,
                 server.read_log_lines, session_id, target, "no_such_file.log")

    found = server.search_log_lines(session_id, r"Going to preempt", limit=5)
    check.ok("the preemption message is found",
             found["total_matches"] > 0, f"{found['total_matches']} matches")
    check.eq("per-folder counts sum to the total",
             sum(row["matches"] for row in found["matches_per_folder"]),
             found["total_matches"])
    check.eq("folders_with_matches counts the rows",
             found["folders_with_matches"], len(found["matches_per_folder"]))
    check.ok("the sample is capped by limit", len(found["sample"]) <= 5)
    check.ok("truncated says the sample is not the whole thing",
             found["truncated"] == (found["total_matches"] > len(found["sample"])))

    # A regex metacharacter must be inert with regex=False, which is the only
    # thing separating a literal search from a pattern here.
    literal = server.search_log_lines(session_id, "org.apache.hadoop", regex=False, limit=1)
    as_regex = server.search_log_lines(session_id, "org.apache.hadoop", regex=True, limit=1)
    check.ok("literal search finds the dotted class name",
             literal["total_matches"] > 0, f"{literal['total_matches']} matches")
    check.ok("regex search matches at least as much (. is any character)",
             as_regex["total_matches"] >= literal["total_matches"])
    upper = server.search_log_lines(session_id, "ERROR", ignore_case=False, limit=1)
    mixed = server.search_log_lines(session_id, "error", ignore_case=True, limit=1)
    check.ok("ignore_case widens the match",
             mixed["total_matches"] >= upper["total_matches"],
             f"{mixed['total_matches']} vs {upper['total_matches']}")
    scoped = server.search_log_lines(session_id, "org.apache", regex=False,
                                     folders=[target], limit=1)
    check.ok("folders= restricts the search to one log folder",
             scoped["folders_with_matches"] <= 1
             and scoped["total_matches"] <= literal["total_matches"])
    return file_name


def stage_hadoop_new_tokens(check, session_id, target):
    """new_tokens and read_log_lines(new_tokens_vs=...): what the target has that the rest lack.

    The baseline is a set of tokens, so every count here is exact -- including
    against OOVDetector, whose per-line score is the same count. The target is
    a failure run and the baseline the eight PageRank_Normal runs; against all
    54 other log folders it has almost nothing new, which would leave the
    per-line checks comparing zeros.
    """
    check.section("4b. new_tokens / read_log_lines(new_tokens_vs)")
    session = server.STORE.get(session_id)
    session.vocabularies.clear()
    normal = "PageRank_Normal*"

    everything = server.new_tokens(session_id, target)
    check.eq("the default baseline is every other log folder",
             everything["n_comparison_folders"], HADOOP_FOLDERS - 1)
    session.vocabularies.clear()

    result = timed("new_tokens (builds the baseline)", server.new_tokens,
                   session_id, target, comparison_folders=normal)
    check.eq("a wildcard baseline is the 8 normal runs", result["n_comparison_folders"], 8)
    check.ok("a smaller baseline leaves more new tokens",
             result["new_token_occurrences"] >= everything["new_token_occurrences"],
             f"{result['new_token_occurrences']} vs {everything['new_token_occurrences']}")
    check.ok("new tokens are found", result["n_rows"] > 0,
             f"{result['n_rows']} tokens on {result['lines_with_new_tokens']} lines")
    counts = [row["count"] for row in result["rows"]]
    check.ok("rows are sorted by count", counts == sorted(counts, reverse=True))
    table = session.get_result(result["result_id"])[1]
    check.eq("counts add up to the occurrences",
             table["count"].sum(), result["new_token_occurrences"])
    check.ok("a token is never on more lines than it occurs",
             (table["n_lines"] <= table["count"]).all())

    # Recount from the frame itself, independently of the vocabulary module.
    def occurrences(frame):
        return (frame.select(pl.col("e_words").explode().alias("token"))
                     .join(table.select("token"), on="token", how="semi").height)
    check.eq("...none of them occurs in any comparison folder",
             occurrences(session.df.filter(pl.col("folder").str.starts_with("PageRank_Normal"))),
             0)
    check.eq("...and each occurs in the target as often as counted",
             occurrences(session.df.filter(pl.col("folder") == target)),
             result["new_token_occurrences"])
    first = result["rows"][0]
    at = server.read_log_lines(session_id, target, first["file_name"],
                               offset=first["line_number"], limit=1)
    check.eq("sample_line is the line read_log_lines has at that line_number",
             at["lines"][0]["m_message"], first["sample_line"])

    check.eq("the baseline vocabulary is kept in the session", len(session.vocabularies), 1)
    again = timed("new_tokens (reuses it)", server.new_tokens,
                  session_id, target, comparison_folders=normal)
    check.eq("...and a repeat gives the same table", again["n_rows"], result["n_rows"])

    # The per-line checks read the file the most common new token first shows up
    # in, so there is something on its lines to compare.
    file_name = first["file_name"]
    check.info(f"target log folder {target}, file {file_name}")
    lines = server.read_log_lines(session_id, target, file_name, limit=500, new_tokens_vs=normal)
    check.ok("read_log_lines marks every line with its new tokens",
             all("new_tokens" in line for line in lines["lines"]))
    check.eq("...from the same kept baseline", len(session.vocabularies), 1)
    only = server.read_log_lines(session_id, target, file_name, limit=500,
                                 new_tokens_vs=normal, only_new=True)
    numbers = [line["line_number"] for line in only["lines"]]
    check.ok("only_new returns only lines with new tokens",
             only["returned"] > 0 and all(line["new_tokens"] for line in only["lines"]),
             f"{only['lines_with_new_tokens']} of {only['total_lines']} lines")
    check.eq("...all of them, up to the limit",
             only["returned"], min(500, only["lines_with_new_tokens"]))
    check.ok("...in file order, with their real line numbers", numbers == sorted(set(numbers)))
    marked = [line["line_number"] for line in lines["lines"] if line["new_tokens"]]
    check.eq("...the same lines a plain read marks", numbers[:len(marked)], marked)

    narrow = server.read_log_lines(session_id, target, file_name, limit=1,
                                   new_tokens_vs=normal, match_file_name=True)
    check.ok("a same-named-file baseline is narrower, so no fewer lines have new tokens",
             narrow["lines_with_new_tokens"] >= lines["lines_with_new_tokens"],
             f"{narrow['lines_with_new_tokens']} vs {lines['lines_with_new_tokens']}")

    # anomaly_line_content trains on the same file in the comparison log folders
    # -- the match_file_name baseline -- and hadoop has no label column, so
    # OOVDetector's vocabulary is every baseline line.
    scored = server.anomaly_line_content(session_id, target, comparison_folders=normal,
                                         target_files=[file_name],
                                         detectors=["OOVDetector"], max_rows=1)
    oovd = session.get_result(scored["files"][0]["result_id"])[1]["OOVD_pred_ano_proba"]
    vocab = vocabulary.baseline_vocabulary(
        session.df, [name for name in session.folders if name.startswith("PageRank_Normal")],
        by_file_name=True)
    file_lines = session.df.filter((pl.col("folder") == target)
                                   & (pl.col("file_name") == file_name))
    per_line = vocabulary.annotate(file_lines, vocab)["new_tokens"].list.len().to_list()
    mismatched = sum(int(score) != count for score, count in zip(oovd.to_list(), per_line))
    check.ok("per line, the new-token count is OOVDetector's score",
             len(per_line) == oovd.len() and mismatched == 0 and sum(per_line) > 0,
             f"{len(per_line)} lines, {sum(per_line)} new tokens, {mismatched} differ")

    if session.parsers:
        parser = session.parsers[0]
        events = server.new_tokens(session_id, target, comparison_folders=normal,
                                   content_format=f"Parse-{parser}")
        check.ok(f"with Parse-{parser} each token is an event id, one per line",
                 all(row["count"] == row["n_lines"] for row in events["rows"]),
                 f"{events['n_rows']} new event types")

    check.raises("only_new needs new_tokens_vs", ValueError,
                 server.read_log_lines, session_id, target, file_name, only_new=True)
    check.raises("the target alone is no baseline", ValueError,
                 server.new_tokens, session_id, target, comparison_folders=[target])
    check.raises("raw-text content has no tokens", ValueError,
                 server.new_tokens, session_id, target, content_format="Sklearn")

    session.ensure_content(False, "Words")
    check.eq("recomputing the words from other text drops the kept baselines",
             len(session.vocabularies), 0)
    session.ensure_content(True, "Words")


def stage_hadoop_distance(check, session_id, target, file_name):
    """The four distance_* tools -- pairwise comparison, L1 to L4."""
    check.section("5. distance_folder_filename / _folder_content / _file_content / _line_content")

    l1 = server.distance_folder_filename(session_id, target, comparison_folders="ALL")
    check.eq("every other log folder is compared", l1["n_rows"], HADOOP_FOLDERS - 1)
    check.ok("the target is not compared with itself",
             all(row["comparison_folder"] != target for row in l1["rows"]))
    check.ok("jaccard distance is a distance in [0, 1]",
             all(0.0 <= row["jaccard distance"] <= 1.0 for row in l1["rows"]))
    check.ok("the preview is sorted by jaccard distance, least similar first",
             [row["jaccard distance"] for row in l1["rows"]]
             == sorted((row["jaccard distance"] for row in l1["rows"]), reverse=True))
    check.eq("sorted_by says so", l1["sorted_by"], "jaccard distance")
    check.ok("the whole table was written", os.path.isfile(l1["artifact"]))

    subset = server.distance_folder_filename(session_id, target, comparison_folders=5)
    check.eq("an int selector takes the first N", subset["n_rows"], 5)
    wildcard = server.distance_folder_filename(session_id, target,
                                               comparison_folders="PageRank_Normal*")
    check.eq("a wildcard selector matches by prefix (8 PageRank_Normal runs)",
             wildcard["n_rows"], 8)
    check.raises("an unknown target log folder is rejected", ValueError,
                 server.distance_folder_filename, session_id, "no_such_folder")
    check.raises("an out-of-range int selector is rejected", ValueError,
                 server.distance_folder_filename, session_id, target,
                 comparison_folders=10_000)

    l2 = timed("distance_folder_content", server.distance_folder_content,
               session_id, target, comparison_folders=10, content_format="Words")
    check.eq("one row per comparison log folder", l2["n_rows"], 10)
    measures = ["cosine", "jaccard", "compression", "containment"]
    check.ok("all four distance measures are present",
             all(m in l2["rows"][0] for m in measures), str(list(l2["rows"][0])))
    check.ok("rank_sum combines them", "rank_sum" in l2["rows"][0])
    check.ok("rank_sum is within [n_measures, n_measures * n_rows]",
             all(len(measures) <= row["rank_sum"] <= len(measures) * l2["n_rows"]
                 for row in l2["rows"]))
    check.ok("the note says larger means more different",
             any("more different" in note for note in l2["notes"]))

    l3 = timed("distance_file_content", server.distance_file_content,
               session_id, target, comparison_folders=5, target_files=2,
               content_format="Words")
    check.ok("files x comparison folders rows", l3["n_rows"] > 0, f"{l3['n_rows']} rows")
    check.ok("each row names the file it compared",
             all("file_name" in row for row in l3["rows"]))

    l4 = timed("distance_line_content", server.distance_line_content,
               session_id, target, comparison_folders=2, target_files=[file_name],
               max_changed_lines=3)
    check.eq("one diff per (file, comparison folder)", l4["n_comparisons"], 2)
    comparison = l4["comparisons"][0]
    check.eq("the diff summary adds up",
             comparison["summary"]["unchanged"] + comparison["summary"]["only_in_target"]
             + comparison["summary"]["only_in_comparison"] + comparison["summary"]["hints"],
             comparison["summary"]["total"])
    check.ok("the changed sample is capped", len(comparison["changed_sample"]) <= 3)
    check.ok("sampled lines are marked - or +",
             all(line["difference"] in ("-", "+") for line in comparison["changed_sample"]))
    check.ok("each diff was written out", os.path.isfile(comparison["artifact"]))
    check.ok("the note explains the markers",
             any("only in the target" in note for note in l4["notes"]))
    return l2


def stage_hadoop_anomaly(check, session_id, target, file_name):
    """The four anomaly_* tools, their guidance notes, and whether they rank."""
    check.section("6. anomaly_folder_filename / _folder_content / _file_content / _line_content")

    l1 = timed("anomaly_folder_filename", server.anomaly_folder_filename,
               session_id, target_folder=5, comparison_folders="ALL")
    check.eq("one row per target log folder", l1["n_rows"], 5)
    check.ok("rank_sum and zscore_sum are both there",
             all(key in l1["rows"][0] for key in ("rank_sum", "zscore_sum")))
    check.ok("the standing anomaly guidance rides on the result",
             any("rank_sum" in note and "suspicion, not a verdict" in note
                 for note in l1["notes"]))

    # Narrowing the detectors is the documented failure mode, so the result has
    # to say so at the call site, and name what is missing.
    narrowed = server.anomaly_folder_filename(session_id, target_folder=3,
                                              detectors=["KMeans", "RarityModel"])
    check.ok("narrowing the detectors warns",
             any("Only 2 of the 4 detectors ran" in note for note in narrowed["notes"]))
    check.ok("...and names the missing ones",
             any("IsolationForest" in note and "OOVDetector" in note
                 for note in narrowed["notes"]))
    single = server.anomaly_folder_filename(session_id, target_folder=2,
                                            detectors=["KMeans"])
    check.ok("one detector gets its own warning",
             any("rank_sum is simply that detector's rank" in note
                 for note in single["notes"]))
    check.raises("an unknown detector is rejected", ValueError,
                 server.anomaly_folder_filename, session_id, target_folder=2,
                 detectors=["NoSuchDetector"])

    # The labelled check: score every log folder against a baseline of known-good
    # ones. Ranking, not exact scores -- the detectors are unsupervised.
    l2 = timed("anomaly_folder_content (all 55 vs PageRank_Normal*)",
               server.anomaly_folder_content, session_id, target_folder="ALL",
               comparison_folders="PageRank_Normal*", content_format="Words", max_rows=5)
    check.eq("every log folder scored", l2["n_rows"], HADOOP_FOLDERS)
    check.ok("the preview is the worst rows, not the first ones",
             [row["rank_sum"] for row in l2["rows"]]
             == sorted((row["rank_sum"] for row in l2["rows"]), reverse=True))
    every = server.query_result(session_id, l2["result_id"], sort_by="rank_sum",
                                max_rows=HADOOP_FOLDERS)["rows"]
    normal = [row["rank_sum"] for row in every if "_Normal_" in row["folder"]]
    failed = [row["rank_sum"] for row in every if "_Normal_" not in row["folder"]]
    auc = rank_auc(failed, normal)
    check.info(f"mean rank_sum: normal {sum(normal) / len(normal):.1f}, "
               f"failure {sum(failed) / len(failed):.1f}; AUC {auc:.3f}")
    check.ok("failure log folders rank as more anomalous than normal ones",
             sum(failed) / len(failed) > sum(normal) / len(normal))
    # Observed 0.839-0.841 over repeated runs; 0.5 would be chance. The margin is
    # for the detectors' run-to-run wobble, not for a real regression.
    check.ok("...and the ranking separates the two, not just their means",
             auc >= 0.75, f"AUC {auc:.3f}, expected >= 0.75")

    params = server.anomaly_folder_content(
        session_id, target_folder=2, comparison_folders=5,
        detectors=["KMeans", "RarityModel"],
        detector_params={"KMeans": {"n_clusters": 3}, "RarityModel": {"threshold": 100}})
    check.ok("detector_params reach the detectors",
             "kmeans_pred_ano_proba" in params["rows"][0]
             and "RM_pred_ano_proba" in params["rows"][0])

    l3 = timed("anomaly_file_content", server.anomaly_file_content,
               session_id, target, comparison_folders="ALL", target_files=3,
               content_format="Words")
    check.eq("one row per scored file", l3["n_rows"], 3)
    check.ok("each row names its file", all("file_name" in row for row in l3["rows"]))
    worst_file = l3["rows"][0]["file_name"]

    # anomaly_file_content compares a file with its namesake in the other log
    # folders -- security.log against the other runs' security.log -- which is
    # the comparison the whole level exists to make. The ported original built
    # its baseline outside the per-file loop and so never filtered it by name,
    # scoring each file against the other *kinds* of file instead. That is
    # invisible in the output (it still returns a row per file with plausible
    # scores) and shows up only as this: a file no comparison log folder has
    # must be skipped, not scored against whatever else is lying around.
    unmatched = "no_other_folder_has_this_file.log"
    session = server.STORE.get(session_id)
    server.STORE._sessions[session_id].df = session.df.with_columns(
        pl.when((pl.col("folder") == target) & (pl.col("file_name") == worst_file))
        .then(pl.lit(unmatched)).otherwise(pl.col("file_name")).alias("file_name")
    )
    try:
        renamed = server.anomaly_file_content(
            session_id, target, comparison_folders="ALL", target_files=[unmatched],
            content_format="Words")
        check.eq("a file no other log folder has is skipped, not scored",
                 renamed["n_rows"], 0)
    finally:
        server.STORE._sessions[session_id].df = session.df.with_columns(
            pl.when((pl.col("folder") == target) & (pl.col("file_name") == unmatched))
            .then(pl.lit(worst_file)).otherwise(pl.col("file_name")).alias("file_name")
        )
    check.eq("...and the log root is back as it was",
             server.STORE.get(session_id).df.filter(
                 pl.col("file_name") == unmatched).height, 0)

    l4 = timed("anomaly_line_content", server.anomaly_line_content,
               session_id, target, comparison_folders="ALL", target_files=[worst_file],
               content_format="Words", max_rows=5)
    check.eq("one entry for the file asked for", l4["n_files"], 1)
    entry = l4["files"][0]
    check.eq("it is the file asked for", entry["file_name"], worst_file)
    check.eq("sorted by rank_sum", entry["sorted_by"], "rank_sum")
    check.ok("every returned line carries its text and line number",
             all("m_message" in line and "line_number" in line
                 for line in entry["top_lines"]))
    check.ok("at most max_rows lines come back", len(entry["top_lines"]) <= 5)
    check.ok("the score plot was written",
             entry["plot"].endswith(".html") and os.path.isfile(entry["plot"]))
    check.ok("the scored table was written", os.path.isfile(entry["artifact"]))
    check.ok("moving averages are in the stashed table, for finding regions",
             any(col.startswith("moving_avg_100_")
                 for col in server.STORE.get(session_id).get_result(entry["result_id"])[1].columns))
    check.ok("the note points at them",
             any("moving_avg_100_" in note for note in l4["notes"]))
    return l2, entry


def stage_hadoop_query(check, session_id, previous, line_entry):
    """query_result -- the follow-up question, without re-running the analysis."""
    check.section("7. query_result")

    result_id = previous["result_id"]
    everything = server.query_result(session_id, result_id, max_rows=HADOOP_FOLDERS)
    check.eq("the whole table is still there",
             everything["n_rows_total"], HADOOP_FOLDERS)
    check.eq("...and it is the table the analysis previewed",
             everything["source_analysis"], "anomaly_folder_content")

    filtered = server.query_result(session_id, result_id,
                                   where=[["folder", "contains", "MachineDown"]],
                                   sort_by="rank_sum")
    check.eq("contains filters to one family of log folders",
             filtered["n_rows_matched"],
             sum(1 for row in everything["rows"] if "MachineDown" in row["folder"]))
    check.ok("the filtered rows really are that family",
             all("MachineDown" in row["folder"] for row in filtered["rows"]))

    threshold = sorted(row["rank_sum"] for row in everything["rows"])[HADOOP_FOLDERS // 2]
    above = server.query_result(session_id, result_id,
                                where=[["rank_sum", ">", threshold]], sort_by="rank_sum")
    check.eq("a numeric threshold matches the rows it should",
             above["n_rows_matched"],
             sum(1 for row in everything["rows"] if row["rank_sum"] > threshold))
    check.ok("descending sort really descends",
             [row["rank_sum"] for row in above["rows"]]
             == sorted((row["rank_sum"] for row in above["rows"]), reverse=True))
    ascending = server.query_result(session_id, result_id, sort_by="rank_sum",
                                    descending=False, max_rows=3)
    check.ok("descending=False gives the other end",
             ascending["rows"][0]["rank_sum"] <= everything["rows"][0]["rank_sum"])

    page1 = server.query_result(session_id, result_id, sort_by="rank_sum", max_rows=10)
    page2 = server.query_result(session_id, result_id, sort_by="rank_sum", max_rows=10,
                                offset=10)
    check.eq("a page holds max_rows rows", len(page1["rows"]), 10)
    check.ok("offset moves on without repeating",
             not {row["folder"] for row in page1["rows"]}
             & {row["folder"] for row in page2["rows"]})
    check.ok("truncated marks that there is more",
             page1["truncated"] and any("Raise offset" in n for n in page1["notes"]))

    nothing = server.query_result(session_id, result_id,
                                  where=[["rank_sum", ">", 10 ** 9]])
    check.eq("an over-tight filter matches nothing", nothing["n_rows_matched"], 0)
    check.ok("...and the summary then describes the whole table so a threshold can be picked",
             "rank_sum" in nothing["summary"]
             and any("whole table" in note for note in nothing["notes"]))

    # A where clause arrives from a model, so nothing here evaluates it: every
    # malformed shape has to come back as an error, not as a surprise.
    check.raises("an unknown result_id is rejected", ValueError,
                 server.query_result, session_id, "anomaly_folder_content-deadbeef")
    check.raises("an unknown column is rejected", ValueError,
                 server.query_result, session_id, result_id,
                 where=[["no_such_column", "==", 1]])
    check.raises("an unknown operator is rejected", ValueError,
                 server.query_result, session_id, result_id,
                 where=[["rank_sum", "=~", 1]])
    check.raises("a malformed clause is rejected", ValueError,
                 server.query_result, session_id, result_id, where=[["rank_sum", ">"]])
    check.raises("an unknown sort column is rejected", ValueError,
                 server.query_result, session_id, result_id, sort_by="no_such_column")

    # anomaly_line_content stashes one table per file, of one row per log line.
    scored = server.query_result(session_id, line_entry["result_id"],
                                 sort_by="rank_sum", max_rows=3)
    check.eq("the line-level table is queryable too",
             scored["n_rows_total"], line_entry["n_lines"])


def stage_hadoop_plots(check, session_id, target, file_name):
    """The three plot tools: a summary and a queryable table, not rows."""
    check.section("8. plot_folder_filename / plot_folder_content / plot_file_content")

    l1 = timed("plot_folder_filename", server.plot_folder_filename,
               session_id, target, comparison_folders="ALL", group_by_indices=[0, 1])
    check.ok("a plot result carries no rows", "rows" not in l1)
    check.eq("one point per log folder", l1["n_rows"], HADOOP_FOLDERS)
    check.ok("both axes are summarized",
             set(l1["summary"]) == {"unique_terms", "lines"}, str(list(l1["summary"])))
    check.ok("the summary carries the tails, not just the quartiles",
             all(key in l1["summary"]["lines"]
                 for key in ("min", "p10", "p25", "median", "p75", "p90", "max")))
    check.ok("the target's own row comes back with its percentiles",
             l1["target"]["folder"] == target
             and 0 <= l1["target"]["unique_terms_pct"] <= 100)
    check.eq("only the scatter is built by default", list(l1["plots"]), ["scatter"])
    check.ok("the scatter was written", os.path.isfile(l1["plots"]["scatter"]))
    check.ok("no UMAP means no umap columns, and a note saying so",
             "umap_x" not in l1["summary"]
             and any("No UMAP was run" in note for note in l1["notes"]))
    check.ok("the points are a query away",
             any("query_result" in note for note in l1["notes"]))
    points = server.query_result(session_id, l1["result_id"], max_rows=5)
    check.ok("group_by_indices colours by the first two name parts",
             all(row["group"].count("_") == 1 for row in points["rows"]),
             str({row["group"] for row in points["rows"]}))
    check.ok("no umap columns in the stashed table either",
             "umap_x" not in points["columns"], str(points["columns"]))

    l2 = timed("plot_folder_content", server.plot_folder_content,
               session_id, target, comparison_folders="ALL", content_format="Words")
    check.eq("one point per log folder", l2["n_rows"], HADOOP_FOLDERS)
    check.ok("x is distinct terms, y is lines",
             l2["summary"]["unique_terms"]["max"] > 0
             and l2["summary"]["lines"]["max"] > 0)
    check.ok("the note names both axes",
             any("unique_terms" in note and "lines" in note for note in l2["notes"]))

    # The UMAP is the opt-in half: it is essentially the whole cost of the tool,
    # and the default view never uses its output.
    umap = timed("plot_folder_content with UMAP (first one pays numba's JIT)",
                 server.plot_folder_content, session_id, target,
                 comparison_folders="ALL", content_format="Words",
                 random_seed=42, plots=["umap", "scatter"])
    check.eq("both figures written", sorted(umap["plots"]), ["scatter", "umap"])
    check.ok("both files exist",
             all(os.path.isfile(path) for path in umap["plots"].values()))
    laid_out = server.query_result(session_id, umap["result_id"], sort_by="umap_x",
                                   max_rows=5)
    check.ok("the layout is columns of the points, readable by query",
             "umap_x" in laid_out["columns"] and "umap_y" in laid_out["columns"])
    check.ok("every point got coordinates",
             all(row["umap_x"] is not None for row in laid_out["rows"]))
    check.ok("the note explains what the coordinates mean",
             any("umap_x/umap_y" in note for note in umap["notes"]))
    check.raises("an unknown plot name is rejected", ValueError,
                 server.plot_folder_content, session_id, target, plots=["barchart"])

    l3 = timed("plot_file_content", server.plot_file_content,
               session_id, target, comparison_folders="ALL", target_files=[file_name],
               content_format="Words")
    check.eq("one plot per file asked for", l3["n_files"], 1)
    entry = l3["files"][0]
    check.eq("it is the file asked for", entry["file_name"], file_name)
    check.ok("one point per log folder holding that file",
             1 < entry["n_rows"] <= HADOOP_FOLDERS, f"{entry['n_rows']} log folders")
    check.ok("each file entry has its own summary and result_id",
             "summary" in entry and "result_id" in entry and "rows" not in entry)


def stage_hadoop_config(check, log_root, workdir):
    """run_config -- LogDelta's YAML vocabulary, read without importing LogDelta."""
    check.section("9. run_config")

    config_dir = Path(workdir) / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    target = sorted(entry.name for entry in os.scandir(log_root) if entry.is_dir())[0]
    # Deliberately in LogDelta's vocabulary: `target_run`, `comparison_runs`,
    # `distance_run_content`, `remove_run_name_from_file_names`. Those names are
    # a file format we read, not ours, and are what _STEP_TOOLS/_STEP_ARGS exist
    # to translate -- a config using our names would test nothing.
    config = f"""
input_data_folder: {log_root}
output_folder: output
regex_masking:
  enabled: true
  pattern:
    - name: myllari_extended
pre_parse:
  enabled: true
  parsers:
    - name: Parse-Tip
preprocessing_steps:
  - name: remove_run_name_from_file_names
steps:
  distance_run_content:
    - target_run: {target}
      comparison_runs: 3
  anomaly_run_file:
    - target_run: {target}
      comparison_runs: 5
  plot_run_content:
    - target_run: {target}
      comparison_runs: 5
  no_such_step:
    - target_run: {target}
"""
    config_path = config_dir / "logdelta_config.yml"
    config_path.write_text(config)

    result = timed("run_config", server.run_config, str(config_path),
                   session_id="config-session", format="auto")
    executed = {step["step"] for step in result["executed"]}
    check.eq("the three known steps ran", result["n_executed"], 3)
    check.eq("LogDelta's step names map to our tools", executed,
             {"distance_run_content", "anomaly_run_file", "plot_run_content"})
    unexpected = [step for step in result["failed"] if step["step"] != "no_such_step"]
    check.ok("no known step failed", not unexpected,
             str(unexpected)[:200] if unexpected else "")
    check.eq("the unknown step is reported, not raised", len(result["failed"]), 1)
    check.eq("...by name", result["failed"][0]["step"], "no_such_step")
    check.ok("target_run was translated, not dropped",
             result["executed"] and all(step["params"].get("target_folder") == target
                                        for step in result["executed"]))
    check.ok("comparison_runs was translated too",
             result["executed"] and all("comparison_folders" in step["params"]
                                        for step in result["executed"]))
    plot_step = [s for s in result["executed"] if s["step"] == "plot_run_content"]
    check.eq("a LogDelta plot step means both figures, which its config cannot say",
             sorted(plot_step[0]["params"]["plots"]) if plot_step else None,
             sorted(visualize.PLOTS))
    check.eq("the config's masking pattern was used",
             result["log_root"]["mask_pattern"], "myllari_extended")
    check.eq("its pre_parse parser ran", result["log_root"]["parsers"], ["tip"])
    check.eq("its preprocessing step became our normalizer",
             result["log_root"]["file_name_normalizer"], "strip_folder_id")
    check.eq("its output_folder is relative to the config file",
             Path(result["output_dir"]), config_dir / "output")
    check.ok("the output really was written there",
             any(name.endswith(".html") for name in os.listdir(result["output_dir"])))
    check.ok("the session stays open for follow-up questions",
             "config-session" in [s["session_id"] for s in
                                  server.list_log_roots()["sessions"]])
    server.close_log_root("config-session")
    check.raises("a missing config file is rejected", FileNotFoundError,
                 server.run_config, str(config_dir / "no_such_config.yml"))


def stage_hadoop_incremental(check, session_id):
    """Incremental enhancement and close_log_root -- the session contract."""
    check.section("10. incremental enhancement / close_log_root")

    session = server.STORE.get(session_id)
    check.ok("tip was parsed at open time and kept", "tip" in session.parsers,
             str(session.parsers))
    before = set(session.df.columns)
    reused = timed("re-using Parse-Tip", server.anomaly_folder_content,
                   session_id, target_folder=2, comparison_folders=5,
                   content_format="Parse-Tip")
    check.eq("no new columns for a parser already there",
             set(server.STORE.get(session_id).df.columns), before)
    added = timed("adding Parse-Drain", server.anomaly_folder_content,
                  session_id, target_folder=2, comparison_folders=5,
                  content_format="Parse-Drain")
    session = server.STORE.get(session_id)
    check.ok("a new parser adds only its own column",
             "drain" in session.parsers and "tip" in session.parsers,
             str(session.parsers))
    check.ok("both results scored the same log folders",
             reused["n_rows"] == added["n_rows"] == 2)

    # e_words computed from masked text must not be handed back for an unmasked
    # request: EventLogEnhancer short-circuits on the output column alone, and
    # Session.content_source is what stops that.
    masked_first = server.distance_folder_content(session_id, session.folders[0],
                                                  comparison_folders=2, mask=True,
                                                  content_format="Words")
    unmasked = server.distance_folder_content(session_id, session.folders[0],
                                              comparison_folders=2, mask=False,
                                              content_format="Words")
    check.ok("switching mask recomputes rather than reusing the wrong column",
             any(a["cosine"] != b["cosine"]
                 for a, b in zip(masked_first["rows"], unmasked["rows"])),
             f"masked {[r['cosine'] for r in masked_first['rows']]} vs "
             f"unmasked {[r['cosine'] for r in unmasked['rows']]}")

    closed = server.close_log_root(session_id)
    check.eq("closing returns the final summary",
             closed["closed"]["session_id"], session_id)
    check.raises("a closed session is gone", ValueError,
                 server.describe_log_root, session_id)


def stage_hadoop_mask_off(check, log_root):
    """A session opened with mask=False must refuse masked analyses, clearly."""
    check.section("11. mask=False")

    info = timed("open unmasked", server.open_log_root,
                 path=str(log_root), format="auto", mask=False,
                 file_name_normalizer="strip_folder_id", session_id="unmasked")
    check.eq("no masked column exists", info["masked"], False)
    check.ok("...so nothing was normalized",
             "e_message_normalized" not in info["enhanced_columns"],
             "" if not info["enhanced_columns"] else str(info["enhanced_columns"]))
    check.raises("a masked analysis says how to fix it", ValueError,
                 server.distance_folder_content, "unmasked",
                 server.STORE.get("unmasked").folders[0], comparison_folders=2, mask=True)
    folder = server.STORE.get("unmasked").folders[0]
    file_name = (server.STORE.get("unmasked").df
                 .filter(pl.col("folder") == folder)["file_name"][0])
    check.raises("reading masked text is refused too", ValueError,
                 server.read_log_lines, "unmasked", folder, file_name, masked=True)
    unmasked = server.distance_folder_content("unmasked",
                                              server.STORE.get("unmasked").folders[0],
                                              comparison_folders=2, mask=False)
    check.eq("mask=False works on the same session", unmasked["n_rows"], 2)
    server.close_log_root("unmasked")


# --------------------------------------------------------------------------- #
# Stage 3 -- hdfs_balanced_5k: 5,000 single-file log folders
# --------------------------------------------------------------------------- #

def stage_hdfs(check, log_root, session_id):
    """What only shows up at scale, and on a log root of one-file log folders."""
    check.section("12. hdfs_balanced_5k (5,000 log folders)")

    info = timed("open (cached after the first run)", server.open_log_root,
                 path=str(log_root), format="auto", mask=True, parsers=["tip"],
                 session_id=session_id)
    check.eq("one log folder per file", info["n_folders"], HDFS_FOLDERS)
    check.eq("each is its own file", info["n_files"], HDFS_FOLDERS)
    # Exact, because stage 1 has already established that these are the 5,000
    # blocks the generator builds and that they hold this many lines.
    check.eq("n_rows", info["n_rows"], HDFS_ROWS)
    check.eq("dropped_rows", info["dropped_rows"], 0)
    check.info(f"{info['n_rows']:,} lines, cache_hit={info['cache_hit']}")
    if not info["cache_hit"]:
        check.eq("AutoLoader detected HDFS's timestamp format",
                 info["detected_formats"], {HDFS_FORMAT: HDFS_FOLDERS})
        check.eq("...from a sample, which is what keeps 5,000 files off two minutes",
                 info["probed_files"], loaders.DEFAULT_MAX_DETECT_FILES)

    # Four name shapes here rather than Hadoop's one -- the sign of the block id
    # is part of the name -- and the sample is spread over all four.
    peeked = timed("peek_log_root", server.peek_log_root, str(log_root))
    check.eq("peek counts 5,000 files without reading them", peeked["n_files"], HDFS_FOLDERS)
    check.eq("...in four name shapes, one per label and sign",
             sorted(entry["name_shape"] for entry in peeked["file_names"]),
             ["Anomaly_blk_#.log", "Anomaly_blk_-#.log",
              "Normal_blk_#.log", "Normal_blk_-#.log"])
    check.eq("...covering every file between them",
             sum(entry["n_files"] for entry in peeked["file_names"]), HDFS_FOLDERS)
    check.ok("...and the sample reaches all four",
             any("all 4 file-name shapes" in note for note in peeked["notes"]),
             str(peeked["notes"])[:160])

    folders = server.STORE.get(session_id).folders
    anomalies = [name for name in folders if name.startswith("Anomaly_")]
    normals = [name for name in folders if name.startswith("Normal_")]
    check.eq("the labels survive as log folder names",
             (len(anomalies), len(normals)),
             (make_test_data.BLOCKS_PER_CLASS, make_test_data.BLOCKS_PER_CLASS))

    # The file-name plot cannot work here: one file per log folder means the x
    # axis is a single value. It has to say so rather than draw a useless plot.
    l1 = timed("plot_folder_filename", server.plot_folder_filename,
               session_id, anomalies[0], comparison_folders="ALL")
    check.eq("a point per log folder", l1["n_rows"], HDFS_FOLDERS)
    check.ok("the degenerate x axis is called out",
             any(note.startswith("CAUTION") and "same number of files" in note
                 for note in l1["notes"]))

    # The default plot is the cheap half: no UMAP, and a summary instead of
    # 5,000 rows. Both are the point of the tool at this size.
    l2 = timed("plot_folder_content (default: scatter only)",
               server.plot_folder_content, session_id, anomalies[0],
               comparison_folders="ALL", content_format="Words")
    check.eq("all 5,000 log folders are in the table", l2["n_rows"], HDFS_FOLDERS)
    check.ok("but none of them are in the result", "rows" not in l2)
    check.eq("only the scatter was built", list(l2["plots"]), ["scatter"])
    check.ok("the summary spans the axes",
             l2["summary"]["lines"]["min"] < l2["summary"]["lines"]["max"])
    check.ok("the target is placed in that range",
             l2["target"]["folder"] == anomalies[0]
             and 0 <= l2["target"]["lines_pct"] <= 100)

    # 5,000 points is exactly the case a preview cannot serve, so paging the
    # stashed table is the only way to read them.
    page1 = server.query_result(session_id, l2["result_id"], sort_by="lines",
                                max_rows=50)
    page2 = server.query_result(session_id, l2["result_id"], sort_by="lines",
                                max_rows=50, offset=50)
    check.eq("the whole table is stashed", page1["n_rows_total"], HDFS_FOLDERS)
    check.eq("a page is max_rows long", len(page1["rows"]), 50)
    check.ok("paging does not repeat itself",
             not {row["folder"] for row in page1["rows"]}
             & {row["folder"] for row in page2["rows"]})
    small = server.query_result(session_id, l2["result_id"],
                                where=[["lines", "<", 5]], sort_by="lines")
    check.ok("a threshold picked from the summary selects real rows",
             0 < small["n_rows_matched"] < HDFS_FOLDERS,
             f"{small['n_rows_matched']} log folders under 5 lines")
    check.ok("...and they satisfy it",
             all(row["lines"] < 5 for row in small["rows"]))

    # The labelled check at this scale: score a mixed handful against a baseline
    # of normal log folders only. One model is fitted per target, so this stays
    # a handful deliberately.
    targets = anomalies[:10] + normals[:10]
    scored = timed("anomaly_folder_content (20 targets vs Normal_*)",
                   server.anomaly_folder_content, session_id, target_folder=targets,
                   comparison_folders="Normal_*", content_format="Words", max_rows=20)
    check.eq("one row per target", scored["n_rows"], len(targets))
    rows = server.query_result(session_id, scored["result_id"], sort_by="rank_sum",
                               max_rows=len(targets))["rows"]
    anomalous = [row["rank_sum"] for row in rows if row["folder"].startswith("Anomaly_")]
    ordinary = [row["rank_sum"] for row in rows if row["folder"].startswith("Normal_")]
    auc = rank_auc(anomalous, ordinary)
    check.info(f"mean rank_sum: anomaly {sum(anomalous) / len(anomalous):.1f}, "
               f"normal {sum(ordinary) / len(ordinary):.1f}; AUC {auc:.3f}")
    check.ok("anomalous blocks rank above normal ones",
             sum(anomalous) / len(anomalous) > sum(ordinary) / len(ordinary))
    # Observed 0.835-0.910 over repeated runs, on 10 blocks per class -- a small
    # sample, hence the wider margin than Hadoop's.
    check.ok("...and the ranking separates the two, not just their means",
             auc >= 0.70, f"AUC {auc:.3f}, expected >= 0.70")

    # One file per log folder means no file name recurs, so the file level has
    # nothing to compare. It has to say so rather than raise: distance_file_content
    # produces no pairs at all, and anomaly_file_content can only score the
    # target's single file against files that are not the same file, which is one
    # row and so a ranking over nothing.
    files_detail = server.describe_log_root(session_id, include_files=True)["files_detail"]
    check.ok("no file name recurs across log folders",
             all(row["n_folders"] == 1 for row in files_detail))
    pairs = server.distance_file_content(session_id, anomalies[0], comparison_folders=20,
                                         content_format="Words")
    check.eq("so L3 distance finds no pairs", pairs["n_rows"], 0)
    scored_file = server.anomaly_file_content(session_id, anomalies[0],
                                              comparison_folders=20, content_format="Words")
    check.ok("and L3 anomaly has at most the target's own file to score",
             scored_file["n_rows"] <= 1, f"{scored_file['n_rows']} row(s)")

    server.close_log_root(session_id)


# --------------------------------------------------------------------------- #
# Stage 13 -- splitting a single log file, on a log that fits in this file
# --------------------------------------------------------------------------- #

def stage_split(check, workdir):
    """Cutting one file into slices, checked on a synthetic log.

    Needs no dataset, which is the point: the invariant being checked -- that the
    slices are the source file and nothing else -- is a property of the splitter,
    not of any corpus, and a check that only runs when ~/Datasets/bgl exists is a
    check that mostly does not run.
    """
    check.section("13. split_log_file (synthetic)")

    source = os.path.join(workdir, "synthetic.log")
    # Deliberately uneven line lengths: equal-byte and equal-line splitting only
    # differ on a file whose lines vary, and a fixed-width one would hide it.
    lines = [f"2024-01-01 00:00:{index % 60:02d} event {index} {'x' * (index % 40)}"
             for index in range(1000)]
    with open(source, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    original = open(source, "rb").read()

    for mode in ("lines", "bytes"):
        out = os.path.join(workdir, f"split-{mode}")
        manifest = split.split_log_file(source, out, n_slices=10, by=mode)
        counts = [entry["lines"] for entry in manifest["slices"]]

        check.eq(f"{mode}: ten slices", manifest["n_slices"], 10)
        check.eq(f"{mode}: every line accounted for", manifest["total_lines"], len(lines))
        check.eq(f"{mode}: named after the source",
                 manifest["slices"][0]["file"], "synthetic_slice_000.log")
        # The one thing that must never fail: the slices are the file, in order,
        # byte for byte. Anything else is a splitter that drops or duplicates log
        # lines, which no later analysis could detect.
        rejoined = b"".join(
            open(os.path.join(out, entry["file"]), "rb").read()
            for entry in manifest["slices"]
        )
        check.ok(f"{mode}: the slices rejoin into the original file",
                 rejoined == original,
                 "" if rejoined == original else f"{len(rejoined)} vs {len(original)} bytes")
        check.info(f"{mode}: lines per slice {min(counts)}-{max(counts)}")

        if mode == "lines":
            check.eq("lines: every slice holds the same count", max(counts) - min(counts), 0)
        else:
            # Equal bytes on uneven lines cannot also be equal lines; if it ever
            # were, the two modes would be the same code and one should go.
            check.ok("bytes: slices differ in line count, as equal bytes implies",
                     max(counts) > min(counts), f"{min(counts)}-{max(counts)}")

    # A split root is a log root: this is the whole reason for the flat layout.
    out = os.path.join(workdir, "split-lines")
    df, info = log_root.read_log_root(out)
    check.eq("a split directory reads back as a log root of 10 log folders",
             info["n_folders"], 10)
    check.eq("...and gives back every line", info["n_rows"], len(lines))
    check.eq("...with the log folder named after the slice file",
             sorted(df["folder"].unique())[0], "synthetic_slice_000.log")

    check.raises("one slice is refused, having nothing to compare against",
                 ValueError, split.split_log_file, source,
                 os.path.join(workdir, "split-1"), n_slices=1)
    check.raises("a non-empty output directory is refused without overwrite",
                 FileExistsError, split.split_log_file, source, out, n_slices=10)
    check.raises("a directory is refused: it is already a log root",
                 FileNotFoundError, split.split_log_file, workdir,
                 os.path.join(workdir, "split-dir"), n_slices=10)

    # peek on the same two shapes, which is what a client meets first.
    peeked = server.peek_log_root(source)
    check.eq("peek calls a single file a file", peeked["kind"], "file")
    check.ok("...and says it must be split before it can be compared",
             any("split" in note.lower() for note in peeked["notes"]),
             str(peeked["notes"])[:100])
    check.eq("peek reads the first lines of it",
             peeked["probed"][0]["sample"][0], lines[0])

    peeked = server.peek_log_root(out)
    check.eq("peek calls a split directory a log root", peeked["kind"], "log_root")
    check.eq("...with a log folder per slice", peeked["n_folders"], 10)
    check.ok("...and nothing to warn about", not peeked["notes"], str(peeked["notes"])[:100])

    empty = os.path.join(workdir, "empty-peek")
    os.makedirs(os.path.join(empty, "child"), exist_ok=True)
    peeked = server.peek_log_root(empty)
    check.eq("a directory of directories with no logs is a 'parent'",
             peeked["kind"], "parent")


# --------------------------------------------------------------------------- #
# Stage 15 -- format detection sampling, on a synthetic log root
# --------------------------------------------------------------------------- #

def stage_detect(check, workdir):
    """Detecting the format from a sample of the files rather than all of them.

    Synthetic and in the default set for the same reason as the split stage: what
    is checked -- that reading a log root gives the same frame whether 50 files
    were probed or every one of them -- is a property of AutoLoader, not of any
    corpus, and the two built log roots are both single-format, so neither can
    show what happens when the sample meets a file that disagrees.
    """
    check.section("15. format detection sampling (synthetic)")

    # 120 log folders of log4j text, plus 3 of NDJSON. The odd ones are named
    # differently, which is the case the sampling leans on: files of different
    # formats are nearly always named differently too.
    root = os.path.join(workdir, "mixed-root")
    for index in range(120):
        folder = os.path.join(root, f"folder_{index:03d}")
        os.makedirs(folder)
        with open(os.path.join(folder, f"app_{index:03d}.log"), "w") as handle:
            for line in range(20):
                handle.write(f"2024-03-0{line % 9 + 1} 10:11:12,345 INFO task {line} started\n")
    for index in range(3):
        folder = os.path.join(root, f"json_{index}")
        os.makedirs(folder)
        with open(os.path.join(folder, f"events_{index}.log"), "w") as handle:
            for line in range(20):
                handle.write(json.dumps({"timestamp": f"2024-03-01T10:11:{line:02d}",
                                         "message": f"event {line}", "level": "INFO"}) + "\n")

    # The sampler on its own: a budget is a cap, and every name shape is covered
    # before any shape is covered twice.
    paths = ([f"/root/f{i}/app_{i:03d}.log" for i in range(120)]
             + [f"/root/json_{i}/events_{i}.log" for i in range(3)])
    picked = loaders.sample_paths(paths, 50)
    check.eq("the sample honours its budget", len(picked), 50)
    check.eq("...and covers every file-name shape",
             sorted({loaders.name_shape(p) for p in picked}),
             ["app_#.log", "events_#.log"])
    check.eq("a budget of 0 means every file", loaders.sample_paths(paths, 0), paths)
    check.eq("so does a budget bigger than the log root",
             loaders.sample_paths(paths, 500), paths)

    peeked = server.peek_log_root(root)
    check.eq("peek groups the files by name shape",
             [(entry["name_shape"], entry["n_files"]) for entry in peeked["file_names"]],
             [("app_#.log", 120), ("events_#.log", 3)])
    check.eq("...and counts the distinct names too", peeked["n_distinct_file_names"], 123)
    check.eq("...and probes each shape rather than the largest files only",
             sorted({entry["name_shape"] for entry in peeked["probed"]}),
             ["app_#.log", "events_#.log"])
    check.ok("...so it reports both formats",
             {entry["format"] for entry in peeked["probed"]} ==
             {"json/ndjson", "text/%Y-%m-%d %H:%M:%S,%3f"},
             str({entry["format"] for entry in peeked["probed"]}))
    check.ok("...and says the sample will cover every shape",
             any("all 2 file-name shapes" in note for note in peeked["notes"]),
             str(peeked["notes"])[:160])

    # The claim the default rests on: a sample that disagrees is not extrapolated
    # from. It probes the rest instead, so a mixed log root reads identically.
    sampled, sampled_info = log_root.read_log_root(root, max_detect_files=50)
    every, every_info = log_root.read_log_root(root, max_detect_files=0)
    check.eq("a disagreeing sample falls back to probing every file",
             sampled_info["probed_files"], 123)
    check.eq("sampling reads the same rows as probing every file",
             sampled.height, every.height)
    check.eq("...and reads them as the same formats",
             sampled_info["detected_formats"], every_info["detected_formats"])
    check.eq("...which is one loader per format, not one per file",
             sorted(sampled_info["detected_formats"].items()),
             [("json/ndjson", 3), ("text/%Y-%m-%d %H:%M:%S,%3f", 120)])

    # Same again with the odd files removed: now the sample agrees and speaks for
    # the other 70, which is the whole point of the default.
    shutil.rmtree(os.path.join(root, "json_0"))
    shutil.rmtree(os.path.join(root, "json_1"))
    shutil.rmtree(os.path.join(root, "json_2"))
    sampled, sampled_info = log_root.read_log_root(root, max_detect_files=50)
    every, every_info = log_root.read_log_root(root, max_detect_files=0)
    check.eq("a unanimous sample is not extended into a full probe",
             sampled_info["probed_files"], 50)
    check.eq("...but the format is reported for every file it was applied to",
             sampled_info["detected_formats"], {"text/%Y-%m-%d %H:%M:%S,%3f": 120})
    check.eq("...and the frame is the one a full probe produces",
             (sampled.height, sampled_info["detected_formats"]),
             (every.height, every_info["detected_formats"]))

    # detections() must not invent evidence for a file nobody looked at.
    loader = loaders.AutoLoader(root, filename_pattern="*.log", max_detect_files=50)
    loader.load()
    table = loader.detections()
    check.eq("one detection row per file", table.height, 120)
    check.eq("...of which the sampled ones are marked probed",
             int(table["probed"].sum()), 50)
    check.eq("...and the rest carry no match rate of their own",
             table.filter(~pl.col("probed"))["rate"].null_count(), 70)

    # The session cache is keyed on it, because 0 and 50 can read a log root
    # differently and a cache hit skips the reading entirely.
    previous = server.STORE
    server.STORE = SessionStore(cache_dir=os.path.join(workdir, "detect-cache"),
                                output_root=os.path.join(workdir, "detect-output"))
    try:
        opened = server.open_log_root(path=root, session_id="detect-sampled", mask=False)
        check.eq("open_log_root reports how many files it probed",
                 opened["probed_files"], 50)
        check.ok("...and says the answer came from a sample",
                 any("detected from 50 of the 120" in note for note in opened.get("notes", [])),
                 str(opened.get("notes"))[:160])
        exhaustive = server.open_log_root(path=root, session_id="detect-every", mask=False,
                                          max_detect_files=0)
        check.ok("a different max_detect_files is a different cache entry",
                 exhaustive["cache_path"] != opened["cache_path"],
                 f"both at {opened['cache_path']}")
        check.ok("...and says nothing about sampling, having probed everything",
                 not any("came from a sample" in note or "detected from" in note
                         for note in exhaustive.get("notes", [])),
                 str(exhaustive.get("notes"))[:160])
    finally:
        server.STORE = previous


# --------------------------------------------------------------------------- #
# Stage 16 -- surviving a kill that cannot be caught
# --------------------------------------------------------------------------- #

CRASH_CHILD = """
import os, sys
from loglead.mcp import server
from loglead.mcp.session import SessionStore

cache, root = sys.argv[1], sys.argv[2]
server.STORE = SessionStore(cache_dir=cache, output_root=os.path.join(cache, "out"))
server.open_log_root(path=root, session_id="crash-demo",
                     file_name_normalizer="strip_folder_id")
# The OOM killer does not raise, does not unwind, and lets nothing run
# afterwards. SIGKILL to self is that same event, on demand.
server.anomaly.anomaly_folder = lambda *a, **k: os.kill(os.getpid(), 9)
server.anomaly_folder_content("crash-demo", content_format="3grams")
print("UNREACHABLE")
"""


def stage_crash(check, workdir):
    """A killed process names the call that killed it, in the next process.

    Synthetic and in the default set for the same reason as the split stage: what
    is checked is a property of the server, not of any corpus. It needs a real
    ``SIGKILL`` in a real child rather than a hand-written breadcrumb, because
    the thing being checked is precisely that nothing gets to run at the end --
    no ``finally``, no ``atexit``, no last line on stderr. A test that wrote the
    breadcrumb itself would pass while the server wrote none.

    Nothing here asserts *why* the child died: the server cannot know that, and
    its report says so. What it has to get right is the call, its arguments, the
    session, the advice, and not saying it twice.
    """
    check.section("16. crash reporting (synthetic, real SIGKILL)")

    root = os.path.join(workdir, "crash-root")
    for folder in ("run_a", "run_b", "run_c"):
        os.makedirs(os.path.join(root, folder), exist_ok=True)
        for index in range(3):
            with open(os.path.join(root, folder, f"{folder}_service_{index}.log"), "w") as handle:
                handle.write(f"2024-01-01 00:0{index}:00 INFO {folder} started job {index}\n"
                             f"2024-01-01 00:0{index}:01 ERROR {folder} failed job {index}\n")

    # Its own cache directory: a ledger belongs to a cache, and the other stages'
    # sessions have no business in this one.
    cache = os.path.join(workdir, "crash-cache")
    script = os.path.join(workdir, "crash_child.py")
    with open(script, "w") as handle:
        handle.write(CRASH_CHILD)

    previous = server.STORE
    server.STORE = SessionStore(cache_dir=cache, output_root=os.path.join(cache, "out"))
    try:
        completed = subprocess.run([sys.executable, script, cache, root],
                                   capture_output=True, text=True)
        check.eq("the child was killed rather than raising", completed.returncode, -9)
        check.ok("nothing ran after the kill", "UNREACHABLE" not in completed.stdout)

        found = server.crash_log().sweep()
        check.eq("one breadcrumb left behind", len(found), 1)
        if not found:
            return
        record = found[0]
        check.eq("it names the call", record["tool"], "anomaly_folder_content")
        # Why effective arguments are recorded rather than the ones spelled out:
        # the expensive default is the one nobody passes.
        check.eq("including arguments the caller never passed",
                 record["args"].get("target_folder"), "ALL")
        check.eq("and the ones it did", record["args"].get("content_format"), "3grams")
        check.eq("the session it was running against", record.get("session_id"), "crash-demo")
        check.eq("the log root", record.get("root"), root)
        check.ok("and how much memory was already held",
                 (record.get("memory") or {}).get("rss_gb", 0) > 0)

        # The client hears about it on the next result, whichever tool that is.
        peeked = server.peek_log_root(root)
        notes = " ".join(peeked.get("notes", []))
        check.ok("reported on the next tool result", "anomaly_folder_content" in notes)
        check.ok("with advice on what to make smaller", "target_folder" in notes)
        check.ok("and as a structured record", bool(peeked.get("server_crash")))
        # Once: a restart loop must not bury every later result under the same news.
        again = server.peek_log_root(root)
        check.ok("and only once",
                 not any("died" in note for note in again.get("notes", [])))

        # The ledger outlives the restart that found it.
        opened = server.open_log_root(path=root, session_id="after-crash",
                                      file_name_normalizer="strip_folder_id")
        history = " ".join(opened.get("notes", []))
        check.ok("open_log_root warns about this log root's history",
                 "killed a server process before" in history)
        check.ok("with the structured record", bool(opened.get("previous_crashes")))

        # The recovery advice is a call, not a description of one.
        recovered = server.open_log_root(**record["open_args"])
        check.eq("the recorded open_args re-open the dead session",
                 recovered["session_id"], "crash-demo")
        check.eq("with the same preprocessing", recovered["n_rows"], opened["n_rows"])

        # A raised exception is not a crash: the client was told about it, so
        # there is nothing left behind to report on the next call.
        check.raises("an unknown session still raises", Exception,
                     server.describe_log_root, "no-such-session")
        check.eq("and leaves no breadcrumb", len(server.crash_log().sweep()), 0)
        clean = server.peek_log_root(root)
        check.ok("so nothing is reported afterwards",
                 not any("died" in note for note in clean.get("notes", [])))
    finally:
        server.STORE = previous


# --------------------------------------------------------------------------- #
# Stage 14 -- BGL: the real single-file case, opt-in
# --------------------------------------------------------------------------- #

def stage_bgl(check, datasets_folder, workdir):
    """Split the real 743 MB BGL.log and analyse the slices.

    Opt-in (``--only bgl``) because it needs the loghub BGL download and writes a
    second copy of it. Everything asserted here is exact: BGL is a plain download,
    so its line count is a property of the dataset.
    """
    check.section("14. BGL (single 743 MB log file)")

    source = os.path.join(datasets_folder, "bgl", "BGL.log")
    if not os.path.isfile(source):
        check.info(f"{source} not found -- skipped. "
                   f"Get it with: uv run downloader/download_data.py --config "
                   f"downloader/datasets.yml")
        return

    peeked = timed("peek_log_root", server.peek_log_root,
                   os.path.join(datasets_folder, "bgl"))
    check.eq("peek finds one log folder, so nothing to compare", peeked["n_folders"], 1)
    check.eq("...and detects BGL without reading it", peeked["probed"][0]["format"], "bgl")
    check.info(f"estimated {peeked['probed'][0]['estimated_lines']:,} lines "
               f"(true {BGL_LINES:,})")

    out = os.path.join(workdir, "bgl-slices")
    manifest = timed("split_log_file (10 slices)", split.split_log_file,
                     source, out, n_slices=10)
    check.eq("ten slices", manifest["n_slices"], 10)
    check.eq("every line of BGL is in one of them", manifest["total_lines"], BGL_LINES)
    counts = [entry["lines"] for entry in manifest["slices"]]
    check.ok("the slices hold the same number of lines to within one",
             max(counts) - min(counts) <= 1, f"{min(counts)}-{max(counts)}")
    check.eq("no bytes gained or lost",
             manifest["total_bytes"], os.path.getsize(source))

    session_id = "bgl-test"
    info = timed("open_log_root", server.open_log_root, path=out, session_id=session_id)
    check.eq("ten log folders", info["n_folders"], 10)
    check.eq("each read as the BGL dataset", info["detected_formats"], {"bgl": 10})
    check.eq("rows plus dropped rows account for every line",
             info["n_rows"] + info["dropped_rows"], BGL_LINES)

    # Every slice is one uniquely named file, so the baseline has to be the
    # other slices' whole content: there is no same-named file to match.
    target = server.STORE.get(session_id).folders[0]
    fresh = timed("new_tokens", server.new_tokens, session_id, target)
    check.eq("the baseline is the other nine slices", fresh["n_comparison_folders"], 9)
    check.ok("new tokens are found", fresh["n_rows"] > 0,
             f"{fresh['n_rows']} tokens on {fresh['lines_with_new_tokens']:,} lines")
    unmatched = server.new_tokens(session_id, target, match_file_name=True)
    check.ok("with match_file_name a slice has nothing to compare with, and says so",
             unmatched["n_rows"] == 0 and len(unmatched["skipped_files"]) == 1)
    only = server.read_log_lines(session_id, target, fresh["rows"][0]["file_name"],
                                 limit=5, new_tokens_vs="ALL", only_new=True)
    check.ok("read_log_lines(only_new) finds those lines in the slice",
             only["returned"] > 0 and all(line["new_tokens"] for line in only["lines"]))

    # L2 is the level a split file can be compared at: one file per log folder
    # means L1 has a single-valued axis and L3/L4 share no file names.
    result = timed("anomaly_folder_content", server.anomaly_folder_content,
                   session_id, target_folder="ALL", content_format="Words")
    check.eq("every slice is scored", len(result["rows"]), 10)
    check.ok("rank_sum is in range", all(4 <= row["rank_sum"] <= 40 for row in result["rows"]),
             str([row["rank_sum"] for row in result["rows"]]))

    # BGL marks alert lines in its first column, and that column survives the
    # split, so the slices can be described even though nothing here used it.
    df = server.STORE.get(session_id).df
    alerts = (df.group_by("folder")
                .agg((pl.col("label") != "-").sum().alias("alerts"))
                .sort("folder"))
    check.eq("the BGL label column survives the split", alerts.height, 10)
    check.info("alerts per slice: " + ", ".join(str(row["alerts"]) for row in alerts.to_dicts()))

    server.close_log_root(session_id)


# --------------------------------------------------------------------------- #

STAGES = ("data", "hadoop", "hdfs", "split", "detect", "crash", "bgl")

#: What runs when no --only is given. 'bgl' is out because it needs the 743 MB
#: loghub download and writes a second copy of it; everything else here runs on
#: data this suite builds for itself.
DEFAULT_STAGES = ("data", "hadoop", "hdfs", "split", "detect", "crash")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default=None,
                        help="Where the log roots live. Defaults to root_folder in "
                             "downloader/datasets.yml.")
    parser.add_argument("--only", choices=STAGES, action="append", dest="stages",
                        help="Run just this stage; repeatable. 'data' always runs. 'bgl' needs "
                             "the loghub BGL download and is not in the default set.")
    parser.add_argument("--regenerate", action="store_true",
                        help="Rebuild the log roots before testing. Note this redraws "
                             "hdfs_balanced_5k's sample.")
    parser.add_argument("--cache-dir", default=None,
                        help="Parquet cache. Kept between runs, because a cold read of "
                             "hdfs_balanced_5k is two minutes. Defaults to "
                             "<datasets>/test_data/mcp_cache.")
    parser.add_argument("--fresh-cache", action="store_true",
                        help="Delete the parquet cache first, so everything is read cold.")
    parser.add_argument("--keep-artifacts", action="store_true",
                        help="Keep the tables and plots the tools wrote.")
    args = parser.parse_args()

    datasets_folder = os.path.expanduser(
        args.datasets or make_test_data.default_source_folder())
    stages = tuple(args.stages or DEFAULT_STAGES)

    # Two of the four detectors are stochastic and take no seed: KMeans and
    # IsolationForest are built without a random_state, so sklearn draws from
    # numpy's global RNG. Seeding it removes that one source of variation. It
    # does *not* make the scores reproducible -- two identical invocations still
    # differ in the third significant figure, because sklearn's threaded KMeans
    # sums in whatever order the threads finish -- which is why no detector score
    # is asserted as a value and the labelled checks use rank_auc with margin.
    numpy.random.seed(make_test_data.SEED)

    # The parquet cache is expensive to rebuild and safe to keep (it is keyed on
    # the files and the preprocessing); the artifacts are neither, so they go to
    # a scratch directory that is deleted unless asked for.
    cache_dir = args.cache_dir or os.path.join(datasets_folder, "test_data", "mcp_cache")
    if args.fresh_cache and os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    workdir = tempfile.mkdtemp(prefix="loglead-mcp-test-")
    server.STORE = SessionStore(cache_dir=cache_dir,
                                output_root=os.path.join(workdir, "output"))

    print(f"Datasets: {datasets_folder}\nCache:    {cache_dir}\nWorkdir:  {workdir}")
    check = Checks()
    started = time.time()

    try:
        run_stage(check, stage_tools, check)

        # The two log roots are only built when something is going to read them:
        # the split, detect and crash stages need no corpus, and `--only split` should
        # not go looking for one.
        if {"data", "hadoop", "hdfs"} & set(stages):
            # Stage 1 is not optional for the stages below it: they read what it
            # verifies, and a crash here is not a bug in the server.
            hadoop, hdfs = stage_data(check, datasets_folder, args.regenerate)
            run_dataset_stages(check, stages, hadoop, hdfs, workdir)

        if "split" in stages:
            run_stage(check, stage_split, check, workdir)
        if "detect" in stages:
            run_stage(check, stage_detect, check, workdir)
        if "crash" in stages:
            run_stage(check, stage_crash, check, workdir)
        if "bgl" in stages:
            run_stage(check, stage_bgl, check, datasets_folder, workdir)
    finally:
        if args.keep_artifacts:
            print(f"\nArtifacts kept in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    print(f"\nTotal time: {time.time() - started:.0f}s")
    return check.report()


def run_dataset_stages(check, stages, hadoop, hdfs, workdir):
    """Everything that reads the two built log roots, in stage order."""
    if "hadoop" in stages:
        session_id = "hadoop-test"
        run_stage(check, stage_hadoop_open, check, hadoop, session_id)
        open_sessions = [s["session_id"] for s in server.list_log_roots()["sessions"]]
        if session_id not in open_sessions:
            check.ok("hadoop_renamed opened", False, "later stages skipped")
        else:
            run_stage(check, stage_hadoop_describe, check, session_id, hadoop)
            # Every later hadoop stage works on the same target log folder and
            # the same widely-shared file, so both are picked once here.
            target = server.STORE.get(session_id).folders[0]
            file_name = run_stage(check, stage_hadoop_read, check, session_id, target)
            if file_name:
                run_stage(check, stage_hadoop_new_tokens, check, session_id, target)
                run_stage(check, stage_hadoop_distance,
                          check, session_id, target, file_name)
                scored = run_stage(check, stage_hadoop_anomaly,
                                   check, session_id, target, file_name)
                if scored:
                    run_stage(check, stage_hadoop_query, check, session_id, *scored)
                run_stage(check, stage_hadoop_plots,
                          check, session_id, target, file_name)
            run_stage(check, stage_hadoop_config, check, hadoop, workdir)
            run_stage(check, stage_hadoop_incremental, check, session_id)
            run_stage(check, stage_hadoop_mask_off, check, hadoop)

    if "hdfs" in stages:
        run_stage(check, stage_hdfs, check, hdfs, "hdfs-test")


def run_stage(check, stage, *args):
    """Run one section, turning a crash into a failure so the rest still runs."""
    try:
        return stage(*args)
    except Exception:
        traceback.print_exc()
        check.ok(f"{stage.__name__} raised", False,
                 traceback.format_exc().strip().split("\n")[-1][:90])
        return None


if __name__ == "__main__":
    sys.exit(main())
