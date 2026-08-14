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
`datasets_auto.yml` is the odd one: detection is not a format, so there is nothing of its own to
download. It re-loads corpora the other configs already cover and copies their `expected_length`
values unchanged, so a count that drifts there but not in the original config means detection chose
the wrong loader. Its `mixed` entry downloads three unrelated corpora into **one** folder — the case
§5 item 6 calls normal rather than exceptional — and its `expected_length` is the sum of the three.
`datasets_access_log.yml` needs a manual download — Kaggle only serves that dataset to a logged-in
account, so the entry uses `local_archive:` (see below) rather than a URL. Its log is larger than
most machines' RAM, so how much of it gets read is **decided at run time**: the entry states the
dataset's memory cost (`memory_gb_per_million_rows`, `memory_gb_overhead`) and `tests/loaders.py`
turns that plus `psutil`'s free memory into an `n_rows` cap, skipping the dataset entirely below
~7 GB free. Two consequences to keep in mind when editing any dataset entry: `expected_length` is
the *full* file length and a capped read is checked against its cap instead, and `target_rows`
(rather than `reduction_fraction`) pins what reaches the enhancer/detector stages so their cost
does not vary by machine.

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
plain log file, one event per line, no labels; the starting point for new/custom data, and the loader
behind `loglead/delta/` and every MCP tool.

`AutoLoader` (`loaders/auto.py`) sits above all of them: it samples a file, decides the format, and
**builds one of the other loaders** — it never parses anything itself, which is what keeps the
decision (`detect_format()`, importable on its own) separable from the reading. Two stages, most
specific first: a **dataset probe** that recognizes a public dataset from the label file and
directory layout *beside* the log (so `HDFSLoader` gets its `anomaly_label.csv` and `df_seq`
survives — a dataset whose labels are missing is deliberately not claimed and falls through; where
the label is not a file at all the layout carries it, as with LO2's `correct/` test-case directory),
then a
**per-file format probe** ordered JSON → access log → syslog → logfmt → generic timestamped text →
plain text, each scored as a match rate over a sample. Two things here are load-bearing and were
measured rather than reasoned: the logfmt test counts `>=2` `key=value` pairs per line and **must
not** be anchored at `^` (real Grafana output puts free text before the pairs — anchoring passes a
1,000-line sample at 0.996 and collapses to 0.199 at 5,000), and the sample is the file's head
**plus a chunk from its middle**, because that same file's head is not representative of it.
`AutoLoader` normalizes `m_timestamp` to naive microseconds afterwards, since Polars takes the time
unit from the format string and a `%3f` pattern otherwise yields a frame that silently refuses to
`pl.concat` with every other loader's output. When every file in a tree agrees it delegates the
whole tree to one loader; only a genuinely mixed folder pays for one loader per file, stacked with
`diagonal_relaxed`.

The newer ones are **spec-driven**: one class per *format family*, configured by a YAML spec rather
than subclassed per dataset (the reasoning is in `docs/log-format-support.md` §7 and
`docs/log-format-json-loader.md`). Their keyword arguments are exactly the spec keys, so a spec file
is a serialized constructor call and the two forms cannot drift; `format=` takes either a shipped
spec name or a path to your own file.

- `JsonLoader` + `loaders/json_formats/*.yml` — JSON/NDJSON logs. The mapping it supplies is which
  key is the message, the timestamp, the sequence id.
- `AccessLogLoader` + `loaders/access_log_formats/*.yml` — Apache/nginx web access logs (Common,
  Combined, and variants). Positional text, so a format is one regex; specs write it as an nginx
  `log_format` string (`'$remote_addr - - [$time_local] "$request" $status ...'`) which compiles to
  that regex. Splits `$request` into `method`/`path`/`protocol` and types `status`/byte counts as
  numbers, since those — not the message text — are what an access log gives `AnomalyDetector`.

`LogfmtLoader` (`loaders/logfmt.py`) is a third family loader but has **no spec directory**, on
purpose: logfmt lines carry their own key names and the names are conventional, so the mapping the
other two need as configuration is just `ts|timestamp|time|t` → `m_timestamp`, `msg|message` →
`m_message`, `level|lvl|severity` → `level` (docs/log-format-support.md §5 item 2). The kwargs for
overriding those exist and read the same as the other two. Each key becomes its own column, so a
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

The data shape here is a **log root**: a directory whose immediate subdirectories are *log folders*,
loaded via `RawLoader` into a single event-level `df` with `folder` and `file_name` columns. There are
no labels and no `df_seq`; comparison is always target-vs-baseline, where the baseline is the other
log folders.

Three question types × four granularities, one function per cell:

| | Distance (pair) | Anomaly (one vs many) | Visualize (set) |
|---|---|---|---|
| **L1** folder / file names | `distance_folder_filename` | `anomaly_folder(file=True)` | `plot_folder(file=True)` |
| **L2** folder / log text | `distance_folder_content` | `anomaly_folder()` | `plot_folder()` |
| **L3** file | `distance_file_content` | `anomaly_file_content` | `plot_file_content` |
| **L4** line | `distance_line_content` | `anomaly_line_content` | — |

Supporting modules: `log_root.py` (loading, and resolving the `"ALL"`/list/int/`"Prefix*"` selectors for
log folders and files), `masking.py` (named regex sets — **only ever resolve these by name via `get_pattern()`,
because `EventLogEnhancer.normalize()` `eval()`s what it is handed**), `scoring.py` (`zscore_sum` and
`rank_sum` over the four measures; prefer `rank_sum`, the raw detector scales differ by orders of
magnitude), `export.py` (the only thing that writes files).

**The invariant that differs from LogDelta**: these functions hold no module state, never `os.chdir`,
and never write files — they return DataFrames. Functions that may add an `e_*` column return
`(results, df)` so the caller can keep the enhanced frame. Keeping it is the whole point; LogDelta
discarded it and re-parsed on every step.

### MCP server (`loglead/mcp/`)

Exposes `loglead/delta/` as 19 MCP tools. Optional install: `uv sync --extra mcp`; entry point
`loglead-mcp` (`[project.scripts]`).

- `session.py` — `Session` holds one log root's enhanced frame and grows it in place;
  `Session.ensure_content()` adds only the missing column and keeps it. `SessionStore` mirrors each
  frame to a parquet cache keyed on the on-disk fingerprint (file count, total bytes, max mtime) plus
  the preprocessing options (mask pattern, `file_name_normalizer`, `folder_names`/`keep_original`), so a restart
  re-attaches in ~0.2s instead of re-reading. Anything that rewrites the frame **must** be in that
  key — a cache hit skips preprocessing entirely and would otherwise serve a wrongly-shaped frame.
  This module has no `mcp` dependency and is usable on its own — that is how
  `demo/mcp_demo.py` runs.
- `server.py` — one tool per analysis, named exactly like the LogDelta config keys. The local `@tool`
  decorator wraps each in `redirect_stdout(sys.stderr)`: LogLead prints freely and stdout carries
  JSON-RPC under the stdio transport. Imports `MCPServer` (SDK 2.x) with a fallback to `FastMCP`
  (SDK 1.x).
- `formatting.py` — every analysis tool returns the same envelope: full table on disk, plus a preview
  sorted by the column that answers the question (`rank_sum` for anomalies) and truncated to
  `max_rows`. Tool results go into a model's context, so unbounded tables are not an option.

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

`demo/mcp_demo.py` exercises all 19 tools against a real log root without an MCP client attached
— the fastest way to check a change here.
