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
- Smoke tests: `uv run demo/HDFS_samples.py`, `uv run demo/TB_samples.py`.
- Full suite (downloads real datasets, ~30 min): `uv run tests/main.py`, chaining `downloader/download_data.py` then `tests/loaders.py` / `enhancers.py` / `anomaly_detectors.py`. Plain scripts, not pytest — read console output for `MISMATCH!`/errors. Run a stage directly to iterate once its input parquet exists in `<root_folder>/test_data/`. Other dataset families via `--config tests/datasets_*.yml`.
- `uv run tests/log_file_detection.py --config <cfg>` — checks `AutoLoader` format detection without loading full data; covers datasets too large to load.
- MCP server tests (needs `uv sync --extra mcp`): `uv run tests/mcp/server.py`; perf grid: `uv run tests/mcp/benchmark.py` (see `tests/mcp/PERFORMANCE.md`).
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
- Detector scores are not bit-reproducible (no `random_state`, threaded sklearn) — never assert exact score values in tests; use rank-based checks with margin.
