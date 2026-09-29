# CLAUDE.md

Guidance for coding agents working in this repo.

LogLead (Log Loader, Enhancer, Anomaly Detector): benchmarks log anomaly detection algorithms/
representations via ~10 dataset loaders, log representation enhancers, and anomaly detection
classifiers, independently swappable. Also used as a backend library by sibling projects LogDelta
and VisualLogAnalyzer — changes to public APIs in `loglead/` can affect them. Data is Polars
DataFrames throughout (not Pandas).

## Build & Test

- Python 3.9–3.12 via `uv`. `uv run <script>` syncs env and installs `loglead` editable — no separate install step.
- `.env` (`LOG_DATA_PATH`) only needed for "bring your own full dataset" scripts; quick demos use bundled sample parquet files instead.
- `LOG_DATA_PATH` (.env) and `root_folder` (dataset YAML config) are independent, unlinked settings — keeping them in sync is on you.
- All tests go through `uv run tests/run.py <suite...>`; `--list` shows suites, steps and what each needs. `smoke` (samples only, minutes) → `mid` (~30 min + downloads) / `formats` / `equivalence` / `memory` / `mcp` / `consumers`; `full` = all of those; `super` (Thunderbird/Spirit/Liberty under a memory cap, hours), `mcp-perf`, `llm` are separate. Each step is its own process; logs + `summary.json` land in `tests/result/runs/<timestamp>/`; non-zero exit on any failure. `--polars` reruns against the oldest supported Polars (1.38.1, which VisualLogAnalyzer pins).
- The steps are plain scripts, not pytest, and can be run directly: dataset stages `tests/loaders.py` / `enhancers.py` / `anomaly_detectors.py --config tests/datasets_*.yml` (read their input parquet from `<root_folder>/test_data/`), `tests/log_file_detection.py`, `tests/streaming.py`, `tests/memory.py`, `tests/consumers.py`, `tests/mcp/server.py` (needs `--extra mcp`). `tests/main.py` is the old mid-suite chain.
- `tests/loaders.py` checks every loaded frame against a fingerprint in `tests/baselines/<config>.json` (row count, schema, nulls, order-sensitive content hash). Record baselines from the reference code, not from a change under test: `tests/run.py <suite> --capture-baselines`. `tests/streaming.py --capture` does the same for its eager outputs (add-only; `--recapture` overwrites).
- Streaming (`loader.sink`, `EventLogEnhancer.from_parquet`, `predict_batches`, `loglead.streaming`) must produce exactly the eager result — `tests/streaming.py` asserts frame equality — and keep heap flat as input grows — `tests/memory.py` asserts that. Read large parquet through `loglead.streaming.iter_batches` (one `scan_parquet().slice()` per batch): Polars' streaming engine (`sink_parquet`, `collect_batches`, `sink_batches`) and pyarrow's `iter_batches` all grow with the file when reading parquet. Python UDFs in a multi-chunk frame run out of row order — rechunk before order-dependent parsers (Drain).
- Downloader: `uv run downloader/download_data.py [--config <cfg>]`. Full set ~7GB download / ~104GB unzipped — check disk space first.
- No linter, formatter, or CI configured — don't invent one unless asked.

## Core Architecture

Pipeline: **Loader → Enhancer → AnomalyDetector**, connected via Polars DataFrames with a shared column-naming convention:
- `m_*` — raw columns from a Loader (e.g. `m_message`, `m_timestamp`).
- `e_*` — event-level columns from `EventLogEnhancer` (e.g. `e_words`, `e_event_drain_id`).
- `seq_*` / unprefixed — sequence-level columns from `SequenceEnhancer`.
- `normal`/`anomaly` — boolean labels; `BaseLoader.add_ano_col()` derives whichever is missing.

Datasets are **event-based** (every line independently labeled, no `df_seq`, e.g. BGL/Thunderbird) or **sequence-based** (`df` + `df_seq`, e.g. HDFS/Hadoop) — check `.df_seq is None` before assuming sequence aggregation applies.

`loglead/delta/` is a second, unsupervised pipeline: given many **log folders** (log root → log folder → log file → log line), compare a target against the others (distance/anomaly/visualize × folder-name/content/file/line granularity). Zero `mcp` imports, no module state, returns plain DataFrames — not the same package as sibling project LogDelta (no dependency on it).

`loglead/mcp/` exposes `delta/` as MCP tools (optional `mcp` extra, Python ≥3.10). Heavy imports (sklearn/xgboost/umap) must stay lazy (function-local, or via an `__init__.py` `_LAZY` table) — never at `loglead/__init__.py` or `delta/log_root.py`/`visualize.py` top level.

## Key Rules

- Only resolve regex/mask patterns by name via `masking.get_pattern()` — `normalize()` `eval()`s what it's handed; never pass raw strings through.
- A new log format that fits an existing spec-driven loader (JSON/access-log/delimited) gets a `.yml` spec, not a Python class.
- `anomaly_file_content`'s baseline-grouping semantics differ intentionally from LogDelta's — don't "fix" this without asking; it's a deliberate design choice, not a bug.
- Keep `loglead/loaders/README.md` in sync when adding/changing a loader.
- Logging in `loglead/`: stdlib `logging.getLogger(__name__)`, never configure logging (handlers/`basicConfig`/levels) or write to stdout — hosts and the stdio MCP transport depend on it. 
- Detector scores are not bit-reproducible (no `random_state`, threaded sklearn) — never assert exact score values in tests; use rank-based checks with margin.
- Do not write comments to code when you make trivial changes
