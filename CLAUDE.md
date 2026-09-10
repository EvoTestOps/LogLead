# CLAUDE.md

This file provides guidance to coding agents when working with code in this repository.

## Project overview

LogLead (Log Loader, Enhancer, Anomaly Detector) is a Python library for benchmarking log anomaly
detection algorithms and log representations. It provides custom loaders for ~10 public log datasets,
~11 log representation "enhancers" (parsers, tokenizers, n-grams, embeddings), and ~11 anomaly detection
classifiers, so a given dataset/representation/classifier combination can be swapped independently. It is
also used as a backend library by the sibling projects LogDelta and VisualLogAnalyzer, so changes to
public APIs in `loglead/` can affect those consumers.

Data is represented as [Polars](https://www.pola.rs/) DataFrames throughout (not Pandas), chosen for speed.

## Environment setup

- Python 3.9–3.12 (`.python-version` pins 3.11 for local dev). Dependency/venv management is via
  [`uv`](https://docs.astral.sh/uv/); `uv run <script>` syncs the environment from `pyproject.toml`/`uv.lock`
  automatically, installing `loglead` itself editable into `.venv` — there is no separate install step, and
  no need to fiddle with `sys.path` to make `import loglead` work.
- A `.env` file is **not** required for the normal `uv` workflow (there isn't one checked in, and none is
  needed for the smoke demos or `tests/main.py`). It only matters if you're pointing scripts at your own
  full-size dataset copies on disk:
  - `LOG_DATA_PATH` is read (via `python-dotenv`) by the "bring your own full dataset" scripts —
    `demo/RawLoader_*`, `demo/parser_benchmark/*`, `demo/saner_2024_paper/*`, `demo/unsupervised_models.py`.
    The quick demos (`demo/HDFS_samples.py`, `demo/TB_samples.py`) use bundled sample parquet files instead
    and never touch it. The downloader also doesn't use it — `downloader/download_data.py` reads `root_folder`
    from the YAML config (`downloader/datasets.yml` or one of `tests/datasets_*.yml`) instead.
  - See `.env.sample` for the format if you do need `LOG_DATA_PATH`.
- There are two independent, **unlinked** ways to point tooling at a data directory on disk — nothing in
  the code cross-references them, so keeping them in sync (e.g. both pointing at `~/Datasets`) is on you:
  - `LOG_DATA_PATH` in `.env` — used only by the demo scripts listed above.
  - `root_folder` in a dataset YAML config — used only by `downloader/download_data.py`, optionally
    overridden by its `--location` CLI flag. A test config's `local_copy_folder` (see Common commands
    below) links it back to `downloader/datasets.yml`'s `root_folder`, but that's opt-in per config —
    it does not make `root_folder` itself a shared setting.
- `scikit-learn` needs `gcc`/`g++` to build. The `pip`-installed package does not pull in `tensorflow`, so
  `BertEmbeddings` (`loglead/parsers/bert/`) must have TF installed manually to work.

## Common commands

Run a script with `uv run path/to/script.py` (or `python path/to/script.py` from inside that script's
directory if using a plain pip install — many scripts assume they're run from their own folder and
`os.chdir` to it).

Quick smoke tests (use small parquet samples committed under `demo/samples/`, no download needed):
```
uv run demo/HDFS_samples.py
uv run demo/TB_samples.py
```

Parser benchmark demos:
```
uv run demo/parser_benchmark/ano_detection.py
uv run demo/parser_benchmark/parsing_speed.py
```

Full test suite — downloads real datasets (see Disk space below), then runs loading, enhancing, and
anomaly-detection checks end to end; takes up to ~30 minutes:
```
uv run tests/main.py
```
`tests/main.py` chains together, in order: `downloader/download_data.py --config
tests/datasets_mid_labels.yml` (downloads/prepares data, the default when `--config` is omitted), then
`tests/loaders.py`, `tests/enhancers.py`, `tests/anomaly_detectors.py` via `runpy`. These are plain
scripts, not a pytest suite — there's no test framework, fixtures, or `-k` filtering; run one of the
four stages directly (e.g. `uv run tests/enhancers.py`) to iterate on just that stage once its input
parquet files already exist in `<root_folder>/test_data/`. Each stage prints `MISMATCH!` warnings if a
loaded dataset's row count drifts from the `expected_length` recorded in the config, and raises/prints
on structural problems (missing mandatory columns, null or non-UTF-8 values) rather than asserting —
read the console output to see pass/fail.

There is a fifth stage that `tests/main.py` does **not** chain, because it is cheap enough to run on
its own and useful against configs whose data is too big to load:
```
uv run tests/log_file_detection.py --config tests/datasets_super_comp_labels.yml
```
It checks that `AutoLoader` picks the same loader `create_correct_loader()` picks by name, for every
dataset in a config. Nothing is loaded — only ~1000 lines per file are sampled — so it covers the
datasets `tests/loaders.py` cannot handle on an ordinary machine: Thunderbird/Spirit/Liberty are
30–38 GB unpacked, and AWSCTD expands to 174 M rows and will OOM well under 16 GB. It also reads
those three straight from their `.gz`, since Polars decompresses transparently and they are usually
left packed. Its `BY_NAME` table mirrors `create_correct_loader()`'s if/elif chain and has to stay
in step with it — that chain is the reference answer being checked against.

The MCP server has its own suite, also not chained by `tests/main.py` — it needs the `mcp` extra
(`uv sync --extra mcp`) and two log roots that no config describes:
```
uv run tests/mcp/server.py                  # the default set: data, hadoop, hdfs, split, detect
uv run tests/mcp/server.py --only hadoop    # one stage; 'data' always runs first
uv run tests/mcp/server.py --only bgl       # opt-in: needs the 743 MB loghub BGL download
```
Three stages sit outside that pair of log roots. `split` needs no corpus at all — it builds a
synthetic 1,000-line log, splits it both ways, and checks the slices rejoin into the original file
byte for byte, which is a property of the splitter rather than of any dataset; it runs by default
because a check that only runs when someone has the right download is a check that mostly does not
run. `detect` is synthetic for a second reason on top of that one: it builds a log root of 120
log4j log folders and 3 NDJSON ones, and both built corpora are single-format, so neither can show
what `AutoLoader`'s format sampling does when the sample meets a file that disagrees — the frame
must come out the same at `max_detect_files=50` and at `0`, and it is the mixed case that decides
whether the sampled default is safe. `bgl` is the real single-file case — `~/Datasets/bgl/BGL.log`, 743 MB and 4,747,963 lines —
and is **not** in the default set because it needs that download and writes a second copy of it.
Its numbers are exact for the same reason Hadoop's are: BGL is a plain loghub download, so the line
count is a property of the dataset.

It exercises all 22 tools in `loglead/mcp/server.py` against `~/Datasets/hadoop_renamed` (55 log
folders, 978 files — the multi-file shape, where L3/L4 and `group_by_indices` have something to work
on) and `~/Datasets/hdfs_balanced_5k` (5,000 single-file log folders — where the plot tools'
summary-instead-of-rows, the degenerate L1 plot and result paging bite). Both are *derived* from public
loghub datasets rather than downloadable, so `tests/mcp/make_test_data.py` rebuilds them from
`~/Datasets/hadoop` and `~/Datasets/hdfs` (downloading those through `downloader/download_data.py`
if missing) and stage 1 of the suite runs it — that is why the generator is the first thing in the
directory, and why nothing else runs until it passes. Both are byte-identical wherever they are
built, which is what lets the suite assert exact line counts: `hadoop_renamed` is Hadoop's own
`abnormal_label.txt` baked into the directory names, and `hdfs_balanced_5k` is HDFS_v1 split per
block id with 2,500 blocks per class selected by **hash order, not an RNG** — `random.sample` would
depend on the directory order it was handed (which is how the ad-hoc script that predates this one
produced a sample nobody could reproduce) and on CPython's sampling internals. The chosen sample is
recorded as `EXPECTED_SAMPLE_DIGEST`/`EXPECTED_LINES` and verified on every run, so a copy holding
some other 5,000 blocks is rebuilt rather than quietly failing a count later; change
`SEED`/`BLOCKS_PER_CLASS` and the build prints the two new constants to paste back.

`tests/mcp/benchmark.py` is the sibling that answers "how long does this take", and
`tests/mcp/PERFORMANCE.md` is its committed output:
```
uv run tests/mcp/benchmark.py                    # three log roots x four fractions
uv run tests/mcp/benchmark.py --only bgl         # the third shape; needs ~/Datasets/bgl/BGL.log
uv run tests/mcp/benchmark.py --fractions 0.05   # one fraction
uv run tests/mcp/benchmark.py --tables-only      # rebuild the tables from recorded cells
```
It holds the *arguments* still — one canonical call per tool, `GRID_ROWS` — and moves the **data**:
every tool on each log root at 5, 10, 50 and 100%, reported cold (tables A1–A4) and warm (B1–B4).
The two numbers are the point, and only visible as two: where they differ, the gap is a column the
session computed once and kept, so the cold number is the price of that representation and the warm
one is what the analysis itself costs — the session model's entire justification. A tool that cannot
run on a shape is data too and prints `err` (Hadoop's `anomaly_file_content` at 5%: `n_samples=1
should be >= n_clusters=2`), not a failed run. One rule instead of a hand-maintained list of heavy
cells: a cold call slower than `HEAVY_SECONDS` (20s) gets one warm repeat rather than `--repeat`,
because the third digit of a two-minute call is not what anyone reads.

**What a fraction means differs by log root**, because the two shapes differ in what makes them big.
`hadoop_renamed` and `hdfs_balanced_5k` are many log folders, so a fraction is a fraction of the
*log folders*, hard-linked into a reduced copy — 5,000 folders cost directory entries rather than
bytes, and `stat` still reports the real sizes, so the session fingerprint and `peek_log_root`'s
byte counts are the numbers the full log root would give. The folders are taken as a **stride, not
the first N**: hdfs's names sort into an `Anomaly_` block and a `Normal_` one, so the first 5% is
250 anomalies with nothing to compare against. `bgl_split_10` is ten large folders, so a fraction
there is a fraction of `BGL.log`'s *lines*, taken before the split — the slice count stays 10 and
what shrinks is the text inside each. Reducing bgl by folders would leave one or two slices at 5%,
which is not a smaller version of the same shape. Both reductions are idempotent through a marker
file beside the copy, so a re-run — or the relaunch after a kill — reuses them.

**bgl** is the third log root and the odd one: 10 log folders of ~471,000 lines each, against
Hadoop's ~3,300 and HDFS's ~18. It is not a midpoint of the other two; it is the only one that
stresses the amount of text *inside* a log folder, and the only one where a call **fails** rather
than merely taking a long time. `anomaly_folder_content(target_folder="ALL")` peaks at 14.6 GB and
`3grams` at 11.6 GB: each completes in a fresh process and each is an out-of-memory kill on a 16 GB
machine once anything else has run, which is how both were found — by killing the benchmark. That is
why each (log root, fraction) block runs in a **child process** writing one JSON file per cell: an
OOM kill takes the process with no chance to record anything, so the parent reads the cell named in
`inflight.json`, records it as `OOM` — which is the measurement, on a log root this size — and
relaunches the block to finish the rest. The same files make the grid resumable (a re-run skips
every cell already on disk, under `<datasets>/test_data/mcp_bench_cells`) and `--tables-only` a pure
rewrite. Two consequences for the cells themselves must stay: the anomaly rows are measured at
**one** target rather than `"ALL"`, and the cached open is measured by **close-and-reopen** rather
than by holding a second session beside the first, which would be two 3 GB frames at once. Recorded
by hand rather than by the grid, and worth knowing anyway: `close_log_root` does not give the memory
back — a cold open, close, `gc.collect()` and reopen still left 6.6 GB resident against 2.9 GB for a
single fresh open.

**A rate does not carry across log roots of different shape**, which the grid shows directly: the
same `distance_folder_content` call at 100% is 18.8s on Hadoop's 3,300-line-average folders, 44.5s
on HDFS's 5,000 18-line-average ones and 152.8s on bgl's ten large ones. Use the measured number for
a log root of the shape you have, never an extrapolation from a different one. That is why **the
absolute seconds live here and in `PERFORMANCE.md`, and never in the MCP docstrings**: a client opens
its own log root, of a shape neither benchmark corpus predicts, so "~21 minutes" is at best noise and
at worst a number it trusts over its own measurement. What the `Cost:` line in each tool's docstring
and the COST paragraph in the server `instructions` carry instead is *what a call is proportional
to* (per target, per comparison folder, per file, fixed) plus the ratios that are properties of the
code rather than of a corpus — the ones the measurements settled: `anomaly_*` refit four detectors
**per target** and `target_folder` defaults to `"ALL"`, so that family is the expensive one;
`content_format` swings cost 60–80x on identical data (`Parse-Tip` 0.18s, `Words` 2.78s, `3grams`
11.35s at 10 comparison folders — `Words` ~15x `Parse-Tip`, `3grams` ~4x `Words`); and the UMAP is
tens of times the default scatter and the one big jump in the server (~8s against under 0.3s on
5,000 log folders), while the scatter itself is far cheaper per comparison folder than
`distance_folder_content`, since it counts terms rather than scoring pairs. The client closes the gap
with `elapsed_seconds`, which every result carries: it is told the shape beforehand and measures the
constant itself. Re-run the benchmark when the analysis code changes, and update the docstrings only
where a *ratio* or a scaling shape moved.

The one thing that is *not* reproducible is detector output — `train_KMeans`/`train_IsolationForest`
take no `random_state`, and sklearn's threaded KMeans is not bit-stable regardless — so two identical
runs differ in the third significant figure. Hence no detector score is asserted as a value: the
labelled checks use `rank_auc` (Mann-Whitney, anomalous vs normal log folders) with margin against
the observed spread. A count like "8 of the top 10" was tried and dropped; it moves between 7 and 10
across runs, and on Hadoop, where 44 of 55 log folders are failures, 8 is what chance produces
anyway. The parquet cache is kept between runs under `<datasets>/test_data/mcp_cache` because a cold
read of the HDFS root is two minutes; `--fresh-cache` drops it.

`--config` selects which dataset set runs, and each config is self-contained (its own `root_folder`,
so the sets do not share a `test_data/` folder). The default, `tests/datasets_mid_labels.yml`, covers
bgl/hadoop/hdfs/nezha/adfa/awsctd — the datasets small enough to load and enhance quickly. The three
supercomputer logs (liberty/spirit/thunderbird) are split out into their own config precisely because
they are not quick — up to 38 GB unpacked each — so running them is opt-in:
```
uv run tests/main.py --config tests/datasets_super_comp_labels.yml  # liberty, spirit, thunderbird
```
Plus five more, smaller and faster, covering the newer loaders:
```
uv run tests/main.py --config tests/datasets_json.yml         # JsonLoader: nginx_json, OTRF, ait_ads
uv run tests/main.py --config tests/datasets_access_log.yml   # AccessLogLoader: Kaggle web access log
uv run tests/main.py --config tests/datasets_fmt.yml          # LogfmtLoader: grafana/loki Drain testdata
uv run tests/main.py --config tests/datasets_syslog.yml       # SyslogLoader: loghub Linux, Mac, OpenSSH
uv run tests/main.py --config tests/datasets_csv_tsv.yml      # DelimitedLoader: loghub CSVs, Zeek, IoT-23, IIS
uv run tests/main.py --config tests/datasets_auto.yml         # AutoLoader: the above, detected + one mixed folder
uv run tests/main.py --config tests/datasets_lo2.yml          # LO2Loader: LO2v2 Light-OAuth2 microservice logs
```
`datasets_lo2.yml` is rooted at `~/Datasets` rather than a root of its own, because the unpacked
logs are tens of GB and `local_copy_folder` would duplicate them. Its entry's keys are `LO2Loader`
constructor arguments, and `single_error_type` is the load-bearing one: left unset the loader picks
a *different random error test case per run on every call*, so `expected_length` would never
reproduce. It also downloads only `light-oauth2-logs.zip` (2.9 GB) — the reduced log set the LO2v2
paper's own analysis used — not the 65.6 GB full dataset. Use v2 and not v1: v1's fixed test order leaked startup logs into the
"correct" class (F1 0.976 on Token, vs 0.623 once v2 randomized the order).
`datasets_csv_tsv.yml` has five entries for four corpora, one per way a delimited file can name its
columns: a header row (`loghub_csv`, 16 systems in one folder — and `loghub_bgl`, one of those files
on its own, because it is labelled and a 16-file load would be 87% unlabelled), Zeek's `#fields`
(`zeek`, a 35-log-type output directory; `iot23`, one labelled conn.log) and W3C's `#Fields:`
(`iis`). Two are labelled, which is what puts them through the supervised half of
`anomaly_detectors.py`; `loghub_bgl`'s predictors deliberately mirror the raw-BGL entry in
`datasets_mid_labels.yml`, so the same events read two ways can be compared (F1 0.815 vs 0.844).
`datasets_auto.yml` is the odd one: detection is not a format, so there is nothing of its own to
download. It re-loads corpora the other configs already cover and copies their `expected_length`
values unchanged, so a count that drifts there but not in the original config means detection chose
the wrong loader. Its `mixed` entry downloads three unrelated corpora into **one** folder — the case
§5 item 6 calls normal rather than exceptional — and its `expected_length` is the sum of the three.
`datasets_access_log.yml` needs a manual download — Kaggle only serves that dataset to a logged-in
account, so the entry uses `local_archive:` (see below) rather than a URL. Its log is 3.5 GB /
10,365,152 lines and needs ~11 GB to hold as a frame, and `tests/loaders.py` **reads all of it**:
there is no automatic capping and no memory check for it, so whether this config runs is a property
of the machine, which is part of why it is opt-in rather than in the default set. Don't add one
back — `AccessLogLoader(n_rows=...)` exists for a caller who wants to bound the read, and pointing
the config at a smaller log is the other way; a test harness that quietly reads a different number
of rows per machine was tried and removed. `reduction_fraction` (a fraction of what was read, as
everywhere else) is what pins the cost of the enhancer and detector stages.

**Config split**: `downloader/datasets.yml` is the single, download-only source of truth for every
public dataset LogLead knows about — one `root_folder` (`~/Datasets`), and each entry carries only
what `download_data.py` reads (`name`, `url`/`urls` or `local_archive`/`source_url`, `download`).
Everything a test needs to know beyond that (`log_file`, `labels_file`, `format`, `loader`,
`predictor_cols`, `expected_length`, `reduction_fraction`, ...) lives only in the `tests/datasets_*.yml`
configs, so changing a test expectation never touches the download-only file. Since most
`tests/datasets_*.yml` configs use their own `root_folder` (so their `test_data/` outputs don't
collide with each other), they set `local_copy_folder: '~/Datasets'` to avoid re-downloading data that
`downloader/datasets.yml` already fetched: `download_data.py` copies `<local_copy_folder>/<name>` to
`<root_folder>/<name>` instead of hitting the network, falling back to a normal download if the local
copy isn't there. `tests/datasets_mid_labels.yml` and `tests/datasets_super_comp_labels.yml` don't need
it — they already point `root_folder` straight at `~/Datasets`.

Downloading datasets directly (independent of running tests):
```
uv run downloader/download_data.py                                    # everything in downloader/datasets.yml
uv run downloader/download_data.py --config tests/datasets_json.yml   # one test-specific set instead
```
Edit the `datasets:` list in the relevant YAML and set `download: false` per-entry to skip datasets you
don't need. Disk space: the full set in `downloader/datasets.yml` is ~7 GB to download and ~104 GB
unzipped (Liberty/Spirit/Thunderbird dominate at 30-38 GB each) — make sure ~110 GB is free before running
the unrestricted downloader.

A dataset entry that carries `local_archive: '~/path/to/archive.zip'` instead of `url:`/`urls:` is one
the downloader cannot fetch — it sits behind a login, Kaggle being the usual case. The archive is
unpacked from wherever the user put it and, unlike a downloaded one, is never deleted afterwards. Add
`source_url:` so the "not found" message can say where to get it.

There is no linter, formatter, or CI workflow configured in this repo — don't invent one unless asked.

## Architecture

LogLead is a three-stage pipeline: **Loader → Enhancer → AnomalyDetector**, connected by Polars
DataFrames with a shared column-naming convention. Understanding the convention is usually more useful
than reading any single file:

- `m_*` — mandatory/raw columns produced by a Loader directly from the source log (e.g. `m_message`,
  `m_timestamp`).
- `e_*` — event-level columns added by `EventLogEnhancer` (e.g. `e_words`, `e_trigrams`,
  `e_message_normalized`, `e_event_drain_id`, `e_chars_len`).
- `seq_*` / unprefixed sequence columns — sequence-level columns added by `SequenceEnhancer` (e.g.
  `seq_len`, `duration`, aggregated `e_event_drain_id` lists).
- `normal` / `anomaly` — boolean label columns; `BaseLoader.add_ano_col()` derives whichever one is
  missing from the other, so downstream code can rely on both existing.

Datasets come in two shapes, and which one you're dealing with determines how much of the pipeline
applies:

- **Event-based only** — every log line (event) is independently labeled normal/anomalous; there's no
  grouping of events into a larger unit. Thunderbird/Spirit/Liberty (`ThuSpiLibLoader`) and BGL
  (`BGLLoader`) are like this — the loader only ever populates `self.df`, never `self.df_seq`.
  `demo/TB_samples.py` is the canonical event-based demo; it explicitly skips sequence-level enhancement
  and anomaly detection ("TB is not labeled on sequence level") and predicts directly on event-level
  columns.
- **Sequence-based** — events are grouped into sequences (an ordered set of events that belong together,
  e.g. all log lines for one HDFS block ID), and anomaly labels apply to the whole sequence rather than
  individual lines. HDFS (`HDFSLoader`) and Hadoop are like this — the loader populates both `self.df`
  (with a `seq_id` column) and `self.df_seq` (one row per sequence). `demo/HDFS_samples.py` is the
  canonical sequence-based demo: it runs `SequenceEnhancer` to aggregate event-level columns up to
  `df_seq` before handing that to `AnomalyDetector`.

Don't assume every loader populates `df_seq` — check the specific loader (or just try `.df_seq is None`)
before writing code that aggregates to sequence level.

### Loaders (`loglead/loaders/`)

`BaseLoader` (`base.py`) defines the contract every loader implements: `load()` reads the raw log into
`self.df` (event-level) and, for sequence-based datasets, `self.df_seq` (sequence-level, one row per
sequence with anomaly labels attached there); `preprocess()` does dataset-specific cleanup.
`execute()` drives `load → preprocess → check_for_nulls_and_non_utf8 → check_mandatory_columns →
add_ano_col` and returns `self.df`. Subclasses only need to implement `load()`/`preprocess()` — this is
what "isolates the unique aspects of logs from different systems" so enhancer/anomaly-detection code
never needs to know which dataset it's operating on.

There is a per-directory `loglead/loaders/README.md` documenting every loader and the dataset each
one reads — keep it in sync when adding or changing a loader.

Loaders come in two shapes. Most are **dataset-specific**, one Python class per dataset:
`HDFSLoader`, `HadoopLoader`, `BGLLoader`, `ThuSpiLibLoader` (Thunderbird / Spirit / Liberty
supercomputer logs), `NezhaLoader` (microservice traces from TrainTicket/WebShop systems),
`ADFALoader`, `AWSCTDLoader` (intrusion detection), `ProLoader`, `LO2Loader`. Plus `RawLoader` — any
plain log file, one event per line, no labels; the starting point for new/custom data, and what
`loglead/delta/` and the MCP tools fall back to (`format="raw"`) when a log root should be read as
plain text.

**Multi-line events** are `loaders/line_policy.py`, one implementation behind three keywords
(`RawLoader.missing_timestamp_action`, `SyslogLoader.multiline`, and `HadoopLoader`, which does not
expose one). It separates *which lines start an event* (`event_start('parsed'|'pattern'|'indent')`,
Polars expressions that combine with `|`) from *what happens to the rest* (six policies: `drop`,
`keep`, `fill-lastseen`, `merge-message`, `merge-add-column`, `raise`; `merge` is RawLoader's old
name for `merge-add-column`). Those two used to be welded together per loader, which is why `merge`
meant "into a trace column" in one loader and "into the message" in another. Everything groups
**per file** — a running count over a whole multi-file frame attaches a file's leading continuation
to the *previous* file's last event, and forward-fills a timestamp across the same boundary — so a
new loader should call this rather than write its own. See `loaders/README.md` §Multi-line events.

`AutoLoader` (`loaders/auto.py`) sits above all of them: it samples a file, decides the format, and
**builds one of the other loaders** — it never parses anything itself, which is what keeps the
decision (`detect_format()`, importable on its own) separable from the reading. Two stages, most
specific first: a **dataset probe** that recognizes a public dataset from the label file and
directory layout *beside* the log (so `HDFSLoader` gets its `anomaly_label.csv` and `df_seq`
survives — a dataset whose labels are missing is deliberately not recognized and falls through; where
the label is not a file at all the layout carries it, as with LO2's `correct/` test-case directory),
then a
**per-file format probe** ordered self-declared columns (Zeek `#separator`, W3C `#Fields:`) → JSON →
access log → syslog → logfmt → delimited-with-a-header-row → generic timestamped text →
plain text, each scored as a match rate over a sample. The two delimited tests sit at opposite ends
deliberately: a file that names its own columns is the strongest evidence in the chain, "the first
line looks like a header" the weakest. Three things here are load-bearing and were
measured rather than reasoned: the logfmt test counts `>=2` `key=value` pairs per line and **must
not** be anchored at `^` (real Grafana output puts free text before the pairs — anchoring passes a
1,000-line sample at 0.996 and collapses to 0.199 at 5,000); the sample is the file's head
**plus a chunk from its middle**, because that same file's head is not representative of it; and the
header-row test asks for **at least** as many delimiters as the header rather than exactly as many,
plus header cells that look like names, since RFC-4180 quoting puts commas inside fields (an
exact-count test scores 1.000 on loghub's Apache CSV and 0.000 on its Hadoop one).
The **generic timestamped-text pass alone** scores over lines that could *start* an event rather
than over every line (`line_policy.starts_event()`), and reads with `missing_timestamp_action=
"merge-message"` so the continuation lines are folded back into the event that printed them. Both
halves are the same fact: a stack trace is one event however many lines it prints, so a per-line
score makes a file *less* recognizable the more of it is one exception — on LogDelta's Hadoop demo
root that lost eleven files their format at 7-38% when the same files score 75-96% per event, and
they were the eleven with the most crashes in them. The named formats keep the plain per-line rate,
since an indented line in pretty-printed JSON belongs to a record whose boundary that format's own
parser already knows.
`AutoLoader` normalizes `m_timestamp` to naive microseconds afterwards, since Polars takes the time
unit from the format string and a `%3f` pattern otherwise yields a frame that silently refuses to
`pl.concat` with every other loader's output. When every file in a tree agrees it delegates the
whole tree to one loader; only a genuinely mixed folder pays for one loader per file, stacked with
`diagonal_relaxed`.
**The per-file probe runs on a sample of the files, not on all of them** (`max_detect_files`, 50 by
default; 0 means every file). It is a read per file, ~20ms, which is the entire cost of opening a
large log root — 13.4s of Hadoop's 978 files, 102s of hdfs_balanced_5k's 5,000 — and a log root
split one file per unit is routinely bigger than either; sampled, those are 1.9s and 2.7s for the
same frame. Two things make it safe enough to be the default. The sample is spread across the
distinct **file-name shapes** (`name_shape()`: the name with its digits collapsed, so 978
`container_…_01_000001.log` files are one shape), because a file in another format is nearly always
named differently, and 50 probes of one shape say nothing about the `stderr.json` beside them. And a
sample that disagrees with itself is *not* extrapolated from — a mixed tree needs a decision per
file anyway, so the rest are probed then; the sample only ever short-circuits the unanimous case.
What remains uncovered is a file that shares its siblings' name shape and not their format, which is
why the peek result, the `open_log_root` result and the docstrings all name `max_detect_files=0`.
`detections()` keeps one row per file but fills the evidence columns only for the probed ones
(`probed` says which) — a match rate for a file nobody read would be evidence invented after the
fact.

The newer ones are **spec-driven**: one class per *format family*, configured by a YAML spec rather
than subclassed per dataset — a format then costs a `.yml` file rather than a Python class, which is
what keeps five families covering what would otherwise be dozens of loaders. Their keyword arguments
are exactly the spec keys, so a spec file
is a serialized constructor call and the two forms cannot drift; `format=` takes either a shipped
spec name or a path to your own file.

- `JsonLoader` + `loaders/json_formats/*.yml` — JSON/NDJSON logs. The mapping it supplies is which
  key is the message, the timestamp, the sequence id.
- `AccessLogLoader` + `loaders/access_log_formats/*.yml` — Apache/nginx web access logs (Common,
  Combined, and variants). Positional text, so a format is one regex; specs write it as an nginx
  `log_format` string (`'$remote_addr - - [$time_local] "$request" $status ...'`) which compiles to
  that regex. Splits `$request` into `method`/`path`/`protocol` and types `status`/byte counts as
  numbers, since those — not the message text — are what an access log gives `AnomalyDetector`.
- `DelimitedLoader` + `loaders/delimited_formats/*.yml` — CSV/TSV with a header, plus the
  self-describing variants. Its extra question is `header=`: where the column names come from, as
  `row` (the first line, delimiter sniffed from it), `zeek` (`#fields`, which also declares the
  separator, the `#types` to cast to and the null markers), `w3c` (a `#Fields:` directive, repeated
  at every rotation) or `none` + `columns=`. Decided **per file**, so a folder mixing them reads in
  one call, and files are stacked `diagonal_relaxed` — a Zeek output directory is 35 log types with
  35 different headers and lands as 342 columns. Two things to know before editing it: with no
  message column (Zeek and W3C have none) a row is rendered as `name=value` text, since a delimited
  row is only a log line once the header is back in front of the values; and this is the one family
  that routinely arrives **labelled**, so `label_field`/`normal_values` build `normal` directly and
  the label column is kept out of `m_message` — a message stating the answer makes every detector
  look perfect.

`LogfmtLoader` (`loaders/logfmt.py`) is a fourth family loader but has **no spec directory**, on
purpose: logfmt lines carry their own key names and the names are conventional, so the mapping the
other three need as configuration is just `ts|timestamp|time|t` → `m_timestamp`, `msg|message` →
`m_message`, `level|lvl|severity` → `level`. The kwargs for
overriding those exist and read the same as the others. Each key becomes its own column, so a
tree of files lands wide-and-sparse the way heterogeneous JSON does; candidate keys are *coalesced*
rather than first-wins, because one file routinely mixes `t=` and `ts=` lines from two components.

`SyslogLoader` (`loaders/syslog.py`) also has no spec directory, for the opposite reason: syslog has
exactly two layouts and both are defined by an RFC, so `rfc3164` and `rfc5424` are built-in regexes
(`pattern=` is the escape hatch). Which one applies is decided **per file** from its first lines —
lnav's rule, and the shape §5 item 6 of the support doc asks for — so a directory holding both reads
in one call. Two consequences worth knowing before editing it: RFC 3164 carries **no year**, so
`m_timestamp` is built by prepending `year=` (the current year by default), and a load mixing both
RFCs runs two vectorized parses and coalesces them, since no single strptime covers both. A line that
does not match is normally the second line of a multi-line message rather than garbage, so the knob
for it is `multiline` (`merge-message` default / `merge-add-column` / `keep` / `drop` / `raise` —
`keep` and `drop` named after `RawLoader.missing_timestamp_action`, the merges named for where they
put the text since there are two of them), and `min_match_rate` — not the first bad line — is what
catches a wrong format. Both merges group **per file**, so a file opening with continuation lines
cannot attach them to the previous file's last event. The two differ only in *where* the
continuation text lands, and not where you would guess: `normalize()` keeps just the first line of
`m_message`, so every `parse_*` sees the same thing either way; what changes is `words()`,
`trigrams()`, `alphanumerics()` and `length()`, which read `m_message` whole. `merge-message` feeds
the trace into `e_words`/`e_chars_len`; `merge-add-column` keeps it out by parking it in `trace`.

When adding a format that already fits one of the spec-driven loaders, add a `.yml` spec, not a class.

### Enhancers (`loglead/enhancers/`)

`EventLogEnhancer` (`eventlog.py`) operates on the event-level `df` and is the home for log parsing and
tokenization: `normalize()` (regex masking of IDs/IPs/hex/numbers before parsing), `words()`,
`trigrams()`/`alphanumerics()`, `length()`, and one `parse_*` method per log-parsing algorithm
(`parse_drain`, `parse_spell`, `parse_brain`, `parse_ael`, `parse_iplom`, `parse_pliplom`, `parse_lenma`,
`parse_tip`, `create_neural_emb` for BERT). Each `parse_*` method wraps a parser implementation from
`loglead/parsers/` and writes an `e_event_<parser>_id` column. Methods check prerequisite columns via
`_handle_prerequisites()` and no-op if their output column already exists, so calls can be chained/repeated
cheaply (see any `demo/*_samples.py` for the typical chain).

`SequenceEnhancer` (`sequence.py`) aggregates event-level columns up to `df_seq` (one row per sequence):
`seq_len`, `start_time`/`end_time`/`duration`, `events()` (collect an event-level column into a per-sequence
list), `tokens()`, `next_event_prediction()` (delegates to `loglead/next_event_prediction.py`'s n-gram
model). It needs both `df` and `df_seq` at construction time and joins on `seq_id`.

### Parsers (`loglead/parsers/`)

Each subdirectory is a self-contained implementation of one log-parsing/template-mining algorithm
(`drain3`, `lenma`, `pyspell`, `iplom`, `pl_iplom`, `AEL`, `Brain`, and optionally `bert` — imported inside
a `try/except` in `parsers/__init__.py` since its TF dependency is often missing). `EventLogEnhancer`
is the only intended caller; treat these as internal implementation detail unless working on parsing
accuracy/speed directly.

### Anomaly detection (`loglead/anomaly_detection.py`)

`AnomalyDetector` is a thin, uniform wrapper around sklearn/xgboost models plus two custom ones
(`OOV_detector`, `RarityModel`), so any of them can be driven through the same API regardless of whether
they're supervised or not:
1. `test_train_split(df_seq, test_frac=...)` or `prepare_train_test_data()` — vectorizes whichever of
   `item_list_col` (token/event-id list column, via `CountVectorizer`), `numeric_cols`, or `emb_list_col`
   is set on the instance into train/test matrices. Changing which predictor columns are set requires
   re-calling `prepare_train_test_data()`.
2. `train_LR/train_DT/train_LSVM/train_RF/train_XGB` (supervised), `train_IsolationForest/train_LOF/
   train_OneClassSVM/train_KMeans` (unsupervised), `train_RarityModel/train_OOVDetector` (custom) — or
   `evaluate_all_ads()` to run every registered model in one call.
3. `predict()` scores the held-out test set and prints/stores accuracy, F1, and (if `auc_roc=True` at
   construction) AUC-ROC.

`_ModelResultsStorage` (used when `AnomalyDetector(store_scores=True)`) accumulates scores across many
`evaluate_all_ads()` runs (e.g. looping over representations/datasets) and exposes
`calculate_average_scores()` / `print_confusion_matrices()` for summarizing a benchmark sweep — this is
the mechanism behind LogLead's "nearly 1,000 combinations" benchmarking claim.

`LogDistance` (same file) is a separate utility for comparing two DataFrames' text columns directly
(cosine/jaccard/compression similarity, `diff_lines()`) — not part of the train/predict pipeline.

### Typical end-to-end flow

Loader.execute() → df (+ df_seq) → EventLogEnhancer chained calls (mutate df, add e_* columns) →
SequenceEnhancer chained calls (aggregate into df_seq) → AnomalyDetector(...).test_train_split(df_seq) →
train_*() → predict(). `demo/HDFS_samples.py` and `demo/TB_samples.py` are the canonical worked examples
of this chain and deliberately share most of their code to demonstrate loader-independence of the rest of
the pipeline.

### Log folder comparison (`loglead/delta/`)

A second, **unsupervised** pipeline that sits on top of the primitives above and answers a different
question: given many log folders from the same system, which one looks wrong? Everything here is
*comparative* — a target is always judged against the other log folders, never in isolation — which is
what "delta" refers to.

**`loglead.delta` is not LogDelta and does not depend on it.** `logdelta` is never imported and is not
in `pyproject.toml`/`uv.lock`. This code was *ported from* the sibling project LogDelta (`~/LogDelta`,
which drives the same analyses from a YAML config) so that LogLead could expose it over MCP; the
dependency runs the other way — LogDelta depends on LogLead, so it could not have gone the other way.

**Layering.** `delta/` has zero `mcp` imports and returns plain DataFrames, so it is usable as a
library on its own; `mcp/` imports *from* it, never the reverse. Keep it that way — `mcp` is an
optional extra needing Python ≥3.10 while LogLead itself supports 3.9, so analysis code must not move
under `loglead/mcp/`.

**Vocabulary.** The object under analysis is a **log folder**: any set of logs that belong together —
one test run, one day, one deployment, "last release". LogDelta calls this a *run*, and its own docs
gloss that word as "folder" every time it appears; LogLead uses "log folder" throughout because the
thing is frequently not a run. The three levels are **log folder → log file → log line**. In prose say
"log folder"; in identifiers (tools, params, columns) it is plain `folder`. Note "run" survives here
only as a *verb* (`run_config`, `uv run`) — and `loglead/loaders/lo2.py` has an unrelated `run` column
of its own, which is a different pipeline entirely.

The data shape here is a **log root**: a directory holding *log folders*, loaded into a single
event-level `df` with `folder` and `file_name` columns. Each subdirectory of the log root is one
log folder and can hold several files; a log file sitting directly in the log root, with no
subdirectory, is a log folder of its own -- a log root can hold both kinds at once. There are
no labels and no `df_seq`; comparison is always target-vs-baseline, where the baseline is the other
log folders.

**Which loader reads it** is `read_log_root(..., format=...)`, a *name* resolved through
`LOG_ROOT_FORMATS` — `"auto"` (the default, `AutoLoader` per file), `"raw"`, a family (`"json"`,
`"syslog"`, `"logfmt"`, `"access_log"`, `"delimited"`) or `"family/spec"` (`"json/nginx_json"`,
`"delimited/zeek"`, `"syslog/rfc5424"`; `available_formats()` lists them all). Name-keyed, not
class-keyed, because the value arrives from a model driving MCP — the same reason as
`masking.get_pattern()`. The names *are* `AutoLoader.detections()`'s own format strings, so what
detection reports can be handed straight back to pin it. Three things to know before touching this:
`AutoLoader` is built with `dataset_probe=False` here (the probe hands a whole directory to a dataset
loader, whose frame has no `file_name` — and LogDelta's Hadoop demo root really does keep
`abnormal_label.txt` beside its log folders, so this is not hypothetical); `orig_file_name` is
rebuilt from the strip prefix rather than taken from the loader, since only `RawLoader` produces it;
and `REQUIRED_COLUMNS` (`m_message`, `file_name`) is checked right after loading, because a format
that reads but maps nothing to the message would otherwise surface as empty results several analyses
later. `read_folders()` is the older two-value wrapper (`(df, n_folders)`) kept for existing callers;
`read_log_root()` returns `(df, info)` where `info` carries the detected-format counts.

Three question types × four granularities, one function per cell:

| | Distance (pair) | Anomaly (one vs many) | Visualize (set) |
|---|---|---|---|
| **L1** folder / file names | `distance_folder_filename` | `anomaly_folder(file=True)` | `plot_folder(file=True)` |
| **L2** folder / log text | `distance_folder_content` | `anomaly_folder()` | `plot_folder()` |
| **L3** file | `distance_file_content` | `anomaly_file_content` | `plot_file_content` |
| **L4** line | `distance_line_content` | `anomaly_line_content` | — |

**A single log file is not a log root, and `split.py` is what turns it into one.** Everything here
judges a log folder against the others, so one file — `BGL.log`, 743 MB of 4,747,963 lines — has
nothing to be compared with. `split_log_file` cuts it into slices written **flat**, one file per
slice (`BGL_slice_000.log`, …), because a log file sitting directly in a log root is already a log
folder of its own: the output directory *is* a log root and `read_log_root` needs no help with it.
That is `hdfs_balanced_5k`'s shape, and it is a deliberate one rather than a compromise. **A flat log
root is right whenever the log file *is* the unit of comparison** — an HDFS block, a slice of one
long log — and wrapping each file in a directory of its own would add a level that carries no
information. LogDelta has no such shape; it always compares directories of files, and flat is
LogLead's own extension for data that is naturally one-file-per-unit.

What follows from it is that **L2 is the level**, not that L3/L4 are broken. When a log folder holds
one file, "which file inside this unit is odd" is not a question, and all four L3/L4 tools — which
match a file with its namesake in the other log folders — correctly find nothing. L1 likewise has a
single-valued axis, which `plot_folder_filename` already warns about. `split_log_file`'s own result
says which tools to use, since the alternative is a client running four that return nothing.

Two things about `split.py` worth knowing before editing it. It is the second module in this
package that writes files (`export.py` is the other), which cannot be helped for something whose
job is producing them — it keeps the other two halves of the invariant, no module state and no
`os.chdir`. And **neither mode reads the file into memory**: both stream it through an 8 MB buffer,
so the cost is bytes moved and a 70 GB log costs no more memory than a 700 MB one. `by="lines"`
(the default) counts the lines in one pass and writes in a second, giving every slice the same
number of *events*, which is what makes slices comparable; `by="bytes"` is the single pass
`split -n l/K` makes, and on BGL that leaves a 306k–390k line spread across ten slices. Measured on
the full BGL: 2.6s to split by lines, against 0.74s for GNU `split` doing the byte version — which
is why nothing here shells out to `split`. It would buy nothing and cost the portability.

**`peek_log_root()` is the cheap call that comes before the expensive one.** `read_log_root` reads
and parses everything; peek stats the files and reads a few hundred lines from a handful of them,
reporting the counts, sizes, `detect_format()`'s answer per probed file, real sample lines, and
`notes` saying what to do next — including "this is one file, split it first". It is also the
survey: pointed at a directory of datasets it lists each child. **What the files are called is part
of the answer** (`file_names`, `n_distinct_file_names`): the files are grouped by `name_shape()`,
the same grouping `AutoLoader`'s format sampling spreads itself across, and the count of shapes is
what says whether one detected format can speak for the whole log root — one shape (Hadoop's 978
container logs) and it can; 274 of them (`~/Datasets` read as one root) and the 50-file sample
cannot reach them all, which is what the note then says. The files probed follow the same grouping:
the largest file of each shape first, then the largest files left over, so a log root of 900
container logs and one `stderr.json` probes the odd one rather than a fifth container log. Two other
details are load-bearing. It
walks **once**, attributing every file to its top-level log folder as it goes; an early version
walked again per child and took 17s on `~/Datasets`, against 0.47s now. And that walk is **bounded**
(`_PEEK_FILE_BUDGET`, 20,000 files), because `~/Datasets` holds 765,416 of them and an accurate
count costs one `stat` each — past the budget it stops, sets `truncated`, and marks the children it
never reached `status="not_counted"` rather than leaving them showing zero files, which is
indistinguishable from empty. `estimated_lines` is sampled from 8 seek points of 125 lines, not
counted: measured +1.3% against BGL's true count, where reading the head alone is +5.8%, since a
log's opening lines are not representative of it — the same fact `AutoLoader`'s `_MID_CHUNK_BYTES`
exists for.

Supporting modules: `log_root.py` (loading, peeking, and resolving the `"ALL"`/list/int/`"Prefix*"`
selectors for log folders and files), `split.py` (cutting one file into a log root), `masking.py` (named regex sets — **only ever resolve these by name via `get_pattern()`,
because `EventLogEnhancer.normalize()` `eval()`s what it is handed**), `scoring.py` (`zscore_sum` and
`rank_sum` over the four measures; prefer `rank_sum`, the raw detector scales differ by orders of
magnitude), `export.py` (the only thing that writes files).

**Plot colours** (`visualize.py`) are a colour-blind safe set of five — `GROUP_COLORS`, picked from
the Okabe & Ito / Paul Tol / IBM palettes — **cycled first**, with `GROUP_SYMBOLS` changing only once
they run out, so groups are told apart by shape as well as colour and 5 x 8 of them stay distinct.
`GROUP_COLORS`/`GROUP_SYMBOLS` are copied verbatim from VisualLogAnalyzer's
`dash_app/utils/plots.py` (there is no shared package to import from) — keep the two in step, or the
same log folders come out different colours in the two projects. LogLead's own addition is
`TARGET_SYMBOL`: every plot here has a target log folder drawn as a cross, so `cross` and the
lookalike `x` are held out of the group cycle in `COMPARISON_SYMBOLS`.

**The two folder/file plots are selectable, not a pair, and the UMAP is the opt-in half** (`plots=`,
an allowlist against `visualize.PLOTS`, defaulting to `DEFAULT_PLOTS` = `("scatter",)`). The layout is
essentially the entire cost of `plot_folder` / `plot_file_content` — measured on hdfs_balanced_5k
(91,638 lines, 5,000 log folders), 41s of a 42s call, against ~0.7s for the loading, aggregating,
vectorizing and figure building put together — and the "scatter" figure never uses its output: unique
terms is `(dtm > 0).sum(axis=1)` off the sparse document-term matrix and lines is a Polars
`group_by`, 0.01s for the two. So `_document_term_matrix` and `_umap_2d` are separate (both figures
need the first, only one needs the second), and the default call is 0.6s rather than 44s. The
default is the cheap one **because the alternative is a 40s default avoidable only by reading a
parameter's docs**, and because a caller who cannot see the picture — which over MCP is every caller
— can read `unique_terms`/`lines` and cannot read UMAP coordinates. Two consequences to keep: a
figure not asked for comes back as `None` rather than being dropped from the tuple, and `points_df`
then **omits** `umap_x`/`umap_y` rather than nulling them — a null coordinate reads as a layout that
failed, a missing column as one that never ran. `_umap_2d` densifies on the way in on purpose: UMAP
accepts the sparse matrix but takes its sparse nearest-neighbour path and is *slower* on it (34s vs
13s on the same 5,000 folders). Two costs no flag can remove, so don't go looking: `random_state`
makes umap-learn single-threaded (13s vs 5s warm on 12 cores), which is the price of `random_seed`
being reproducible, and the first layout in a process pays ~29s of numba JIT on top of `import
umap`'s own 14s at `delta/__init__` time.

Flipping that default put one thing at risk that `_STEP_DEFAULTS` (`mcp/server.py`) exists to hold:
LogDelta's plot steps always draw **both** figures and its config format has no key to say so, so
reproducing a config means `run_config` requesting both explicitly. It is a fourth LogDelta-vocabulary
lookup table alongside `_STEP_TOOLS`/`_STEP_ARGS`/`_PREPROCESSING` — keyed by LogDelta's step names,
and merged *under* whatever the config states.

**The three plot tools' MCP descriptions do not use the L1-L4 vocabulary** the other nine analysis
tools do, and say what both axes are. A client sees only the docstring: the level numbers are a
reference to the table above that never reaches it, and are redundant with the tool name anyway,
while "unique terms against lines" is unguessable and is the entire content of the default result.
The same reasoning applies to the other nine tools, which have not been swept yet.

`plot_line_scores` colours by *detector family* (kmeans/IF/RM/OOVD) and splits within a family by
**how the trace is drawn, not by shape**: the raw per-line score is a scatter — every point there is
a log line to hover — and its 10/100-line moving averages are `mode="lines"`, widest window solid and
boldest (`MOVING_AVERAGE_DASHES`). A rolling mean is a path by construction, and drawing it as one
marker per line smears it into a band; LogDelta's `_ano_plot_line_scores` draws all twelve as
`symbol="x"` markers and relies on density to make the averages look like curves. `display_mode`
therefore governs only the raw scores now. Which columns are averages is read from the
`moving_avg_<window>_` prefix (`_moving_average_window`), not from column order, and a name that
does not parse falls back to a scatter rather than raising. **Hover follows the same split, and for
a reason that is about file size**: plotly serializes each trace's `text` array separately, so the
per-line log message on all twelve traces tripled the message text in the file for nothing — 70 MB
of HTML for a 20,000-line file. Only the four raw scatters carry `text`; the eight averages use
`hoverinfo="x+y+name"`, which plotly builds from the coordinates and the trace name and which costs
no per-line array (30 MB for that same file). A point on a rolling mean is not a log line anyway, so
this is what its hover should have said. The remaining bulk is the four surviving copies, which is
the floor if each detector's scatter is to stay hoverable.

**The invariant that differs from LogDelta**: these functions hold no module state, never `os.chdir`,
and never write files — they return DataFrames. Functions that may add an `e_*` column return
`(results, df)` so the caller can keep the enhanced frame. Keeping it is the whole point; LogDelta
discarded it and re-parsed on every step.

**`anomaly_file_content`: LogLead's baseline differs from LogDelta's, and the difference is not
settled.** Both group the baseline by `file_name`. LogDelta builds it *outside* the per-file loop
(`df_other_runs_files = _aggregate_dataframe(df_other_runs, 'file_name', field)`), so the training
set is one document per distinct file name, each merging every comparison run's copy — 47 documents
on `hadoop_renamed`. LogLead currently filters that by the file being scored and groups by `folder`,
so the training set is one document per comparison log folder holding that file — 53 documents for
`container__01_000001.log` on the same log root. Both are "matched by name" in the sense that the
grouping key is the file name; they differ in whether the *other* file names are in the training
set too. LogLead's variant also skips a file no comparison log folder has, which is what
`tests/mcp/server.py` asserts. **Do not change this without asking** — it is a semantic choice about
what a file is judged against, not a defect to be tidied up in either direction.

### MCP server (`loglead/mcp/`)

Exposes `loglead/delta/` as 22 MCP tools. Optional install: `uv sync --extra mcp`; entry point
`loglead-mcp` (`[project.scripts]`).

- `session.py` — `Session` holds one log root's enhanced frame and grows it in place;
  `Session.ensure_content()` adds only the missing column and keeps it. `SessionStore` mirrors each
  frame to a parquet cache keyed on the on-disk fingerprint (file count, total bytes, max mtime) plus
  the preprocessing options (`format`, `max_detect_files`, mask pattern, `file_name_normalizer`,
  `folder_names`/`keep_original`) and a `_PREPROCESSING_VERSION`, so a restart
  re-attaches in ~0.2s instead of re-reading. Bump that version whenever a change inside LogLead
  makes the same inputs produce a different frame — nothing else in the key would notice. Anything that rewrites the frame **must** be in that
  key — a cache hit skips preprocessing entirely and would otherwise serve a wrongly-shaped frame.
  `format` is the heaviest entry: it decides which loader ran and therefore every column, so the same
  logs read as `raw` and as `json` share nothing but their paths. `max_detect_files` is in the key
  for the same reason at one remove — it decides how many files `"auto"` looked at, and a log root
  whose sample missed a second format is read differently at 0 than at 50.
  This module has no `mcp` dependency and is usable on its own — that is how
  `demo/mcp_demo.py` runs.
- `server.py` — one tool per analysis, named exactly like the LogDelta config keys. The two
  non-analysis tools are `peek_log_root` (what is on disk, without loading it) and `split_log_file`,
  which wraps `delta/split.py`. `split_log_file` is the only tool that chooses a path for the caller:
  with no `out_dir` it writes under `STORE.cache_dir/splits/<stem>-<digest>`, the digest taken over
  the source file's fingerprint and the split parameters, so asking for the same split twice reuses
  the slices instead of writing a second copy of a 743 MB log. That default lives in `server.py` and
  not in `delta/split.py` — `delta` must not import `mcp`, so the library function requires an
  explicit `out_dir`. The manifest is written beside the slices as `split_manifest.json` so a reused
  split reports what a fresh one does, line counts included, without reading them back; its
  `elapsed_seconds` is dropped on reuse, since that number is how long the *original* split took. The local `@tool`
  decorator wraps each in `redirect_stdout(sys.stderr)`: LogLead prints freely and stdout carries
  JSON-RPC under the stdio transport. Imports `MCPServer` (SDK 2.x) with a fallback to `FastMCP`
  (SDK 1.x).
- `formatting.py` — every analysis tool returns the same envelope: full table on disk, plus a preview
  sorted by the column that answers the question (`rank_sum` for anomalies) and truncated to
  `max_rows`. Tool results go into a model's context, so unbounded tables are not an option.

**A truncated preview is not the end of the table** (`Session.stash_result` / `query_result`). Every
analysis keeps the frame it previewed in the session and returns a `result_id`, so the follow-up
question — "every log folder under 5 lines", "the row for this one", "everything past a `rank_sum` of
100" — is a filter over an in-memory Polars frame, not another analysis run. No CSV is involved: the
table is already a frame, and `export.py`'s files stay what they always were, the human's copy. Three
things to keep in mind: `where` clauses are structured `[column, op, value]` triples rather than an
expression string, because the clause arrives from a model and nothing here `eval`s what it is handed
(same rule as `masking.get_pattern()`); results are in-memory only, so a stale `result_id` must say
"re-run the analysis" rather than resurrect anything; and the cache is bounded **twice**
(`MAX_RESULTS`, `MAX_RESULT_ROWS`) because one `plot_file_content` call stashes a tiny table per file
while one `anomaly_line_content` call stashes a scored frame per file with a row per log line.

**The three plot tools return no rows at all, and have no `max_rows`** (`_plot_result`) — the only
analysis tools without one, and the only ones that do not build their envelope through
`formatting.result`. A scatter has no top N, so any first-N of the points is one arbitrary corner of
the picture: on hdfs_balanced_5k, sorting by `unique_terms` and taking 60 gives 60 near-identical log
folders spanning terms 63-93, missing three of the four axis extremes *and* the target itself, which
ranks ~4,000th of 5,000 and is the one point the plot draws as a cross. A row cap cannot fix that,
which is what `query_result` is for. So a plot result is `n_rows` + `summary`
(min/p10/p25/median/p75/p90/max per axis — what a caller picks a threshold from, having never seen
the data) + `target` (the target's own row plus its percentile on each axis) + `result_id` + `plots`,
and the points themselves are a query. Note `umap_x`/`umap_y` are columns of that stashed table like
any other, so reading a layout is `query_result(sort_by="umap_x")` rather than a bigger result. Plot
results carry no `artifact` either — the HTML paths are in `plots`, and the old envelope pointed
`artifact` at a plotly file directly under a note promising the full table.

**Anomaly guidance is part of the result, not just the docstrings** (`_anomaly_notes`). The observed
failure mode is a model driving these tools reading one detector's score as a finding, and narrowing
`detectors` to save time — both of which defeat `rank_sum`. So `_ANOMALY_NOTE` rides on every anomaly
result, and `_SUBSET_NOTE`/`_SINGLE_DETECTOR_NOTE` fire *whenever* fewer than four detectors ran,
naming the missing ones. A docstring is read once at tool-registration time; a note is in front of
the model at the moment it is about to over-read a number, which is why the check is at the call site
rather than only in the `detectors:` parameter docs. Keep the note wording aligned with
`delta/scoring.py`'s module docstring — the numbers in it (`rank_sum` starts at 4, tops out at
`4 * n_rows`) are properties of `add_combined_scores`, not slogans.

**Log folder names** come from the directory names and are what every legend, result row and output
file is labelled with, so opaque ids make the analysis unreadable. `open_log_root(folder_names=...)` and
the `set_folder_names` tool apply a caller-supplied `{directory name: meaningful name}` mapping to the
`folder` column (`log_root.apply_folder_names`), keeping the directory name in `folder_original`;
nothing on disk is renamed. Names are always applied to the original, so renaming replaces rather than
stacks. `keep_original_folder_name` (default true) appends the directory name — needed for
`group_by_indices` and `"Prefix*"` wildcards; pass false and the log folder simply *is* the given name,
in which case uniqueness is enforced since every selector filters on `folder`. The names need not be
ground-truth labels (`FailingRunThu` is as valid as `PageRank_MachineDown`), which is why nothing here
says "label".

The library deliberately takes **only a mapping** — datasets record this metadata in wildly different
ways, so producing it is the caller's job. `demo/mcp_demo_hadoop_folder_names.json` ships one as data
(derived from Hadoop's own `abnormal_label.txt`); the demo takes `--folder-names <file.json>`. Two
ordering constraints: name log folders **after** `normalize_file_names` (`strip_folder_id` derives the
id to strip from the raw directory name), and names reach output file names, so
`export.build_file_name` sanitizes them.

**Reading LogDelta YAML** (`run_config`). **LogLead does not depend on LogDelta** — `logdelta` is never
imported, and is not in `pyproject.toml` or `uv.lock`; the dependency runs the other way (LogDelta
depends on LogLead). `run_config` merely `yaml.safe_load`s a config *file* and calls LogLead's own
tools, the way reading a `.csv` implies nothing about Excel.

What it does create is a **naming coupling**: that file format is LogDelta's, so three lookup tables
hold its vocabulary — `_STEP_TOOLS` keys stay LogDelta's step names (`distance_run_file`, …),
`_STEP_ARGS` translates `target_run`/`comparison_runs` to our parameter names, and `_PREPROCESSING`
maps its preprocessing steps. Keep those three tables in LogDelta's vocabulary and never "fix" them to
match ours. Without `_STEP_ARGS` the kwargs filter in `run_config` would drop those arguments
**silently**, and tools whose target defaults to `"ALL"` would score the wrong thing without erroring.

`demo/mcp_demo.py` exercises all 22 tools against a real log root without an MCP client attached
— the fastest way to check a change here.

### WSL Browser Integration
When running in WSL and producing interactive HTML visualizations (e.g. from `plot_folder_*` or `plot_file_*`), open them in the Windows default browser using:
`open-browser <path-to-html>` (or copy to `/mnt/c/Users/mmantyla/AppData/Local/Temp` and launch via PowerShell `Start-Process $env:TEMP\<file>`).
