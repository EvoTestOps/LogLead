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
    and never touch it. The downloader also doesn't use it — `downloader/download_data.py` reads `root_folder` from the YAML config (`datasets.yml`/`tests/datasets.yml`) instead.
  - See `.env.sample` for the format if you do need `LOG_DATA_PATH`.
- There are two independent, **unlinked** ways to point tooling at a data directory on disk — nothing in
  the code cross-references them, so keeping them in sync (e.g. both pointing at `~/Datasets`) is on you:
  - `LOG_DATA_PATH` in `.env` — used only by the demo scripts listed above.
  - `root_folder` in `datasets.yml`/`tests/datasets.yml` — used only by `downloader/download_data.py`
    (`download_data.py:247`), optionally overridden by its `--location` CLI flag.
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
`tests/main.py` chains together, in order: `downloader/download_data.py --config tests/datasets.yml`
(downloads/prepares data), then `tests/loaders.py`, `tests/enhancers.py`, `tests/anomaly_detectors.py` via
`runpy`. These are plain scripts, not a pytest suite — there's no test framework, fixtures, or `-k`
filtering; run one of the four stages directly (e.g. `uv run tests/enhancers.py`) to iterate on just that
stage once its input parquet files already exist in `<root_folder>/test_data/`. Each stage prints
`MISMATCH!` warnings if a loaded dataset's row count drifts from the `expected_length` recorded in
`tests/datasets.yml`, and raises/prints on structural problems (missing mandatory columns, null or
non-UTF-8 values) rather than asserting — read the console output to see pass/fail.

Downloading datasets directly (independent of running tests):
```
uv run downloader/download_data.py                          # everything in downloader/datasets.yml
uv run downloader/download_data.py --config tests/datasets.yml  # smaller set used by the test suite
```
Edit the `datasets:` list in the relevant YAML and set `download: false` per-entry to skip datasets you
don't need. Disk space: the full set in `downloader/datasets.yml` is ~7 GB to download and ~104 GB
unzipped (Liberty/Spirit/Thunderbird dominate at 30-38 GB each) — make sure ~110 GB is free before running
the unrestricted downloader.

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

Concrete loaders: `RawLoader` (any plain log file, no labels — the starting point for new/custom data),
and dataset-specific loaders `HDFSLoader`, `HadoopLoader`, `BGLLoader`, `ThuSpiLibLoader` (Thunderbird /
Spirit / Liberty supercomputer logs), `NezhaLoader` (microservice traces from TrainTicket/WebShop
systems), `ADFALoader`, `AWSCTDLoader` (intrusion detection), `ProLoader`, `GELFLoader`, `LO2Loader`.

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
