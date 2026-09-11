"""Rebuild the two log roots that ``tests/mcp/server.py`` runs against.

Both are *derived* from public loghub datasets rather than downloadable as such,
which is why this script exists: without it the MCP tests would depend on two
directories that happen to sit on one machine.

``hadoop_renamed``
    loghub's Hadoop, with every ``application_<id>`` directory renamed to
    ``<Application>_<Failure>_application_<id>`` using the labels the dataset
    publishes in its own ``abnormal_label.txt``. Nothing inside a folder
    changes. This is the multi-file-per-log-folder case: 55 log folders holding
    978 files between them, so the file-level (L3) and line-level (L4) analyses
    have same-named files to compare across folders, and ``group_by_indices``
    and ``"Prefix*"`` selectors have something to group by. The same mapping
    ships as data in ``demo/mcp_demo_hadoop_folder_names.json``; here it is
    baked into the directory names instead, so the log root is self-describing
    and a test needs no side file to know which log folder is which.

``hdfs_balanced_5k``
    loghub's HDFS_v1, split into one file per block id, named
    ``<Label>_<block id>.log`` from ``preprocessed/anomaly_label.csv``, and
    sampled down to 2,500 anomalous and 2,500 normal blocks -- the same 5,000
    everywhere, see reproducibility below. This is the opposite shape: 5,000
    single-file log folders, which is what makes it worth testing -- it is where
    the plot tools' UMAP cost, the degenerate L1 plot (every log folder holds
    exactly one file), and result paging actually bite.

Both are built from ``~/Datasets/hadoop`` and ``~/Datasets/hdfs``, downloading
them through ``downloader/download_data.py`` if they are not there yet.

Usage::

    uv run tests/mcp/make_test_data.py                 # build whatever is missing
    uv run tests/mcp/make_test_data.py --force         # rebuild both from scratch
    uv run tests/mcp/make_test_data.py --only hdfs_balanced_5k
    uv run tests/mcp/make_test_data.py --check         # verify, build nothing

**On reproducibility.** Both log roots are byte-identical wherever this runs,
which is what lets ``tests/mcp/server.py`` assert exact line counts instead of
whatever happens to be on the machine. ``hadoop_renamed`` is fully determined by
the public dataset. ``hdfs_balanced_5k`` is a *sample*, so it is pinned three
ways: the selection is a hash ordering of the block ids rather than a draw (see
:func:`choose_blocks` -- an RNG would depend on both directory order and
CPython's sampling internals), and the resulting sample is recorded here as
:data:`EXPECTED_SAMPLE_DIGEST` and :data:`EXPECTED_LINES`, which
:func:`check_hdfs_balanced_5k` verifies. A copy holding some other 5,000 blocks
is reported as wrong and rebuilt, rather than quietly failing a line count later.
"""

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Source datasets, as named in downloader/datasets.yml.
SOURCE_HADOOP = "hadoop"
SOURCE_HDFS = "hdfs"

#: What this script builds.
HADOOP_RENAMED = "hadoop_renamed"
HDFS_BALANCED_5K = "hdfs_balanced_5k"
DATASETS = (HADOOP_RENAMED, HDFS_BALANCED_5K)

#: Hadoop ships 55 labelled applications; every one of them becomes a log folder.
N_HADOOP_FOLDERS = 55

#: Blocks sampled per class, and the seed that fixes which ones. See
#: choose_blocks: the selection is a hash ordering, not a draw, so it is the same
#: 5,000 blocks on every machine and every Python version.
BLOCKS_PER_CLASS = 2500
SEED = 42

#: What that sample *is*, so a drifted selection is caught rather than silently
#: tested against. The digest covers the sorted file names; the line count is the
#: total across them, and is what tests/mcp/server.py asserts the loader reads.
#: Both change only if BLOCKS_PER_CLASS, SEED, choose_blocks or the source
#: dataset changes -- in which case run --force and update them together.
EXPECTED_SAMPLE_DIGEST = "07f2094b99d06c6b8957c4ca4c563f48b090e591b62dbfed60fda838aa416793"
EXPECTED_LINES = 90862

#: Every id in an HDFS line. A line naming two blocks belongs to both.
BLOCK_ID = re.compile(rb"blk_-?\d+")


# --------------------------------------------------------------------------- #
# Source datasets
# --------------------------------------------------------------------------- #

def default_source_folder():
    """``root_folder`` from downloader/datasets.yml -- where the public sets land."""
    with open(REPO_ROOT / "downloader" / "datasets.yml") as handle:
        return os.path.expanduser(yaml.safe_load(handle)["root_folder"])


def download_sources(source_folder, names, allow_download=True):
    """Make sure the public datasets are unpacked under `source_folder`.

    Reuses ``downloader/download_data.py`` with a config holding only the
    entries we need, so the URLs stay in the one file that owns them and this
    never triggers the 104 GB full download.
    """
    missing = [name for name in names if not os.path.isdir(os.path.join(source_folder, name))]
    if not missing:
        return
    if not allow_download:
        raise SystemExit(
            f"Missing source dataset(s) {missing} under {source_folder}, and --no-download "
            f"was given. Fetch them with:\n"
            f"  uv run downloader/download_data.py --config downloader/datasets.yml"
        )

    with open(REPO_ROOT / "downloader" / "datasets.yml") as handle:
        catalogue = yaml.safe_load(handle)
    entries = [d for d in catalogue["datasets"] if d["name"] in missing]
    found = {entry["name"] for entry in entries}
    if found != set(missing):
        raise SystemExit(f"downloader/datasets.yml has no entry for {sorted(set(missing) - found)}")

    print(f"Downloading {sorted(found)} into {source_folder} (Hadoop ~200 MB, HDFS ~1.5 GB) ...")
    sys.path.insert(0, str(REPO_ROOT / "downloader"))
    from download_data import main as download_data_main

    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as handle:
        yaml.safe_dump({"root_folder": source_folder,
                        "datasets": [dict(entry, download=True) for entry in entries]}, handle)
        config_path = handle.name
    try:
        download_data_main(source_folder, config_path)
    finally:
        os.unlink(config_path)


def link_or_copy(src, dst):
    """Hard-link a file into place, falling back to a copy across devices."""
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copyfile(src, dst)


# --------------------------------------------------------------------------- #
# hadoop_renamed
# --------------------------------------------------------------------------- #

def read_hadoop_labels(label_file):
    """Parse ``abnormal_label.txt`` into ``{application id: "App_Failure"}``.

    The file is prose with structure: ``### WordCount`` names the application,
    ``Machine down:`` the failure, and ``+ application_...`` lines list the runs
    it applies to. Failure names become one CamelCase token so the folder name
    splits on underscores the way ``group_by_indices=[0, 1]`` expects.
    """
    labels = {}
    application = failure = None
    with open(label_file) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("###"):
                application, failure = line.lstrip("#").strip(), None
            elif application and line.endswith(":"):
                failure = "".join(word.capitalize() for word in line[:-1].split())
            elif application and failure and line.startswith("+"):
                labels[line[1:].strip()] = f"{application}_{failure}"
    if not labels:
        raise SystemExit(f"No labels parsed from {label_file}; is it the loghub Hadoop file?")
    return labels


def build_hadoop_renamed(source_folder, dest_folder):
    """Copy each Hadoop application folder to a name carrying its label."""
    source = Path(source_folder) / SOURCE_HADOOP
    dest = Path(dest_folder) / HADOOP_RENAMED
    labels = read_hadoop_labels(source / "abnormal_label.txt")

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    n_files = 0
    for application, label in sorted(labels.items()):
        src_dir = source / application
        if not src_dir.is_dir():
            raise SystemExit(f"{src_dir} is missing; is {source} the loghub Hadoop dataset?")
        dst_dir = dest / f"{label}_{application}"
        dst_dir.mkdir()
        for entry in sorted(os.listdir(src_dir)):
            link_or_copy(src_dir / entry, dst_dir / entry)
            n_files += 1

    # Not read by anything -- the labels are in the directory names now -- but
    # kept so the copy still says where it came from and what its licence is.
    for extra in ("README.md", "abnormal_label.txt"):
        if (source / extra).is_file():
            link_or_copy(source / extra, dest / extra)

    print(f"  {dest}: {len(labels)} log folders, {n_files} files")
    return dest


def check_hadoop_renamed(dest_folder):
    """Report what is wrong with an existing hadoop_renamed, or nothing."""
    dest = Path(dest_folder) / HADOOP_RENAMED
    if not dest.is_dir():
        return [f"{dest} does not exist"]
    folders = sorted(entry.name for entry in os.scandir(dest) if entry.is_dir())
    problems = []
    if len(folders) != N_HADOOP_FOLDERS:
        problems.append(f"{len(folders)} log folders, expected {N_HADOOP_FOLDERS}")
    naming = re.compile(r"^(WordCount|PageRank)_"
                        r"(Normal|MachineDown|NetworkDisconnection|DiskFull)_"
                        r"application_\d+_\d+$")
    unnamed = [name for name in folders if not naming.match(name)]
    if unnamed:
        problems.append(f"{len(unnamed)} log folder(s) not named <App>_<Failure>_<id>, "
                        f"e.g. {unnamed[:3]}")
    empty = [name for name in folders
             if not any(child.endswith(".log") for child in os.listdir(dest / name))]
    if empty:
        problems.append(f"{len(empty)} log folder(s) hold no .log file, e.g. {empty[:3]}")
    return problems


# --------------------------------------------------------------------------- #
# hdfs_balanced_5k
# --------------------------------------------------------------------------- #

def choose_blocks(label_file, per_class=BLOCKS_PER_CLASS, seed=SEED):
    """Pick a balanced, reproducible sample of block ids as ``{block id: label}``.

    The sample has to be *the same 5,000 blocks on every machine*, since the
    tests assert exact line counts against it. So it is not a draw at all: each
    block id is hashed with the seed and the lowest ``per_class`` hashes per
    label win. That is deterministic in a way an RNG is not -- ``random.sample``
    would also depend on the filesystem order it was handed (which is what the
    ad-hoc script this replaces did, and why its output could not be
    reproduced), and on CPython's sampling internals, which are an
    implementation detail rather than a promise.

    Changing ``seed`` picks a different, equally reproducible 5,000 -- and
    invalidates :data:`EXPECTED_LINES` and :data:`EXPECTED_SAMPLE_DIGEST`.
    """
    by_label = {"Anomaly": [], "Normal": []}
    with open(label_file, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["BlockId", "Label"]:
            raise SystemExit(f"{label_file} does not look like HDFS anomaly_label.csv "
                             f"(header was {header})")
        for row in reader:
            if len(row) >= 2 and row[1].strip() in by_label:
                by_label[row[1].strip()].append(row[0].strip())

    def rank(block):
        return hashlib.sha256(f"{seed}:{block}".encode()).hexdigest()

    chosen = {}
    for label, blocks in by_label.items():
        if len(blocks) < per_class:
            raise SystemExit(f"Only {len(blocks)} {label} blocks, need {per_class}")
        for block in sorted(blocks, key=rank)[:per_class]:
            chosen[block] = label
    return chosen


def build_hdfs_balanced_5k(source_folder, dest_folder):
    """Split HDFS.log into one file per sampled block, labelled in its name.

    One pass over the 1.5 GB log, keeping only the lines of the 5,000 sampled
    blocks. A line naming two block ids belongs to both, which is how HDFS is
    normally grouped; CRLF becomes LF so the files are ordinary text.
    """
    source = Path(source_folder) / SOURCE_HDFS
    log_file = source / "HDFS.log"
    label_file = source / "preprocessed" / "anomaly_label.csv"
    for path in (log_file, label_file):
        if not path.is_file():
            raise SystemExit(f"{path} is missing; is {source} the loghub HDFS_v1 dataset?")

    chosen = choose_blocks(label_file)
    print(f"  reading {log_file} ({log_file.stat().st_size / 1e9:.1f} GB) for "
          f"{len(chosen)} block ids ...")

    lines = {block: [] for block in chosen}
    with open(log_file, "rb") as handle:
        for raw in handle:
            line = raw.rstrip(b"\r\n")
            for match in set(BLOCK_ID.findall(line)):
                block = match.decode()
                if block in lines:
                    lines[block].append(line)

    empty = [block for block, found in lines.items() if not found]
    if empty:
        raise SystemExit(f"{len(empty)} sampled block(s) have no lines in {log_file}, "
                         f"e.g. {empty[:3]}; the log and the label file disagree.")

    dest = Path(dest_folder) / HDFS_BALANCED_5K
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    total = 0
    for block, label in chosen.items():
        with open(dest / f"{label}_{block}.log", "wb") as handle:
            handle.write(b"".join(line + b"\n" for line in lines[block]))
        total += len(lines[block])

    # Printed so a deliberate change to the sample (a new seed, a different
    # per-class count) can be pasted straight into the two constants above.
    print(f"  {dest}: {len(chosen)} files ({BLOCKS_PER_CLASS} per class), {total} lines")
    print(f"  EXPECTED_LINES = {total}\n"
          f"  EXPECTED_SAMPLE_DIGEST = "
          f'"{sample_digest(f"{label}_{block}.log" for block, label in chosen.items())}"')
    return dest


def sample_digest(file_names):
    """Fingerprint of *which* blocks a copy of hdfs_balanced_5k holds."""
    return hashlib.sha256("\n".join(sorted(file_names)).encode()).hexdigest()


def count_lines(paths):
    return sum(sum(1 for _ in open(path, "rb")) for path in paths)


def check_hdfs_balanced_5k(dest_folder):
    """Report what is wrong with an existing hdfs_balanced_5k, or nothing.

    This is where the sample is pinned. A copy holding 5,000 valid-looking files
    that are not *these* 5,000 would pass every structural check and then fail
    the exact line counts in tests/mcp/server.py with nothing to explain it, so
    the digest is checked here, next to the code that produces it.
    """
    dest = Path(dest_folder) / HDFS_BALANCED_5K
    if not dest.is_dir():
        return [f"{dest} does not exist"]
    files = [entry.name for entry in os.scandir(dest) if entry.is_file()]
    problems = []
    per_class = {label: sum(1 for name in files if name.startswith(f"{label}_blk_"))
                 for label in ("Anomaly", "Normal")}
    for label, count in per_class.items():
        if count != BLOCKS_PER_CLASS:
            problems.append(f"{count} {label} files, expected {BLOCKS_PER_CLASS}")
    stray = [name for name in files
             if not name.endswith(".log") or not name.startswith(("Anomaly_blk_", "Normal_blk_"))]
    if stray:
        problems.append(f"{len(stray)} file(s) not named <Label>_blk_<id>.log, e.g. {stray[:3]}")
    empty = [name for name in files if os.path.getsize(dest / name) == 0]
    if empty:
        problems.append(f"{len(empty)} empty file(s), e.g. {empty[:3]}")
    if problems:  # the digest would only add noise to an already-wrong copy
        return problems

    digest = sample_digest(files)
    if digest != EXPECTED_SAMPLE_DIGEST:
        problems.append(
            f"holds a different 5,000 blocks than this script builds (digest {digest[:16]}, "
            f"expected {EXPECTED_SAMPLE_DIGEST[:16]}) -- rebuild it with --force")
        return problems  # a different sample has a different line count by construction
    lines = count_lines(dest / name for name in files)
    if lines != EXPECTED_LINES:
        problems.append(f"{lines} lines, expected {EXPECTED_LINES}")
    return problems


# --------------------------------------------------------------------------- #

BUILDERS = {
    HADOOP_RENAMED: (SOURCE_HADOOP, build_hadoop_renamed, check_hadoop_renamed),
    HDFS_BALANCED_5K: (SOURCE_HDFS, build_hdfs_balanced_5k, check_hdfs_balanced_5k),
}


def ensure_datasets(dest_folder=None, source_folder=None, datasets=DATASETS,
                    force=False, allow_download=True, check_only=False):
    """Build any of `datasets` that is missing or broken; return ``{name: path}``.

    An existing directory that passes its check is left alone, because rebuilding
    ``hdfs_balanced_5k`` draws a different (valid) sample -- see the module
    docstring. `force` rebuilds regardless; `check_only` builds nothing and
    raises if anything is missing.
    """
    source_folder = os.path.expanduser(source_folder or default_source_folder())
    dest_folder = os.path.expanduser(dest_folder or source_folder)
    built = {}

    for name in datasets:
        source_name, build, check = BUILDERS[name]
        problems = check(dest_folder)
        if not problems and not force:
            print(f"{name}: present and complete, keeping it")
            built[name] = Path(dest_folder) / name
            continue
        if check_only:
            raise SystemExit(f"{name}: " + "; ".join(problems))
        print(f"{name}: building ({'--force' if not problems else '; '.join(problems)})")
        download_sources(source_folder, [source_name], allow_download)
        built[name] = build(source_folder, dest_folder)
        remaining = check(dest_folder)
        if remaining:
            raise SystemExit(f"{name} is still wrong after building: " + "; ".join(remaining))

    return built


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=None,
                        help="Where the public loghub datasets live (and are downloaded to). "
                             "Defaults to root_folder in downloader/datasets.yml.")
    parser.add_argument("--dest", default=None,
                        help="Where to write the derived datasets. Defaults to --source.")
    parser.add_argument("--only", choices=DATASETS, action="append", dest="datasets",
                        help="Build just this one; repeatable.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if the dataset is already there and complete. "
                             "Note this redraws hdfs_balanced_5k's sample.")
    parser.add_argument("--check", action="store_true",
                        help="Verify what is on disk and build nothing.")
    parser.add_argument("--no-download", action="store_true",
                        help="Fail instead of downloading a missing source dataset.")
    args = parser.parse_args()

    built = ensure_datasets(
        dest_folder=args.dest, source_folder=args.source,
        datasets=tuple(args.datasets or DATASETS), force=args.force,
        allow_download=not args.no_download, check_only=args.check,
    )
    print("\nReady:")
    for name, path in built.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
