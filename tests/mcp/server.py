"""Tests for ``loglead/mcp/server.py``: every tool, against two real log roots.

Like the rest of ``tests/``, this is a plain script rather than a pytest suite --
run it and read the output. Every check prints ``ok`` or ``FAIL``, sections keep
going after a failure so one broken tool does not hide the other nineteen, and
the exit code is non-zero if anything failed.

Run it with::

    uv run tests/mcp/server.py                     # everything
    uv run tests/mcp/server.py --only hadoop       # one stage: data, hadoop, hdfs
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
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy  # noqa: E402
import polars as pl  # noqa: E402
import make_test_data  # noqa: E402  (sits next to this file)
from loglead.delta import visualize  # noqa: E402

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

STAGES = ("data", "hadoop", "hdfs")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default=None,
                        help="Where the log roots live. Defaults to root_folder in "
                             "downloader/datasets.yml.")
    parser.add_argument("--only", choices=STAGES, action="append", dest="stages",
                        help="Run just this stage; repeatable. 'data' always runs.")
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
    stages = tuple(args.stages or STAGES)

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
        # Stage 1 is not optional: everything below reads what it verifies, and
        # a crash here is not a bug in the server.
        hadoop, hdfs = stage_data(check, datasets_folder, args.regenerate)

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
    finally:
        if args.keep_artifacts:
            print(f"\nArtifacts kept in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    print(f"\nTotal time: {time.time() - started:.0f}s")
    return check.report()


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
