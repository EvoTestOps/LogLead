# Logging status per file

Tracks the migration described in [logging.md](logging.md), one row per Python file in `loglead/`.
Files outside `loglead/` (demos, tests, downloader) get no logging and are not listed.

Last full review: 2026-09-14. Print counts come from
`grep -rnE '(^|[^a-zA-Z_.])print\(' loglead` and include commented-out prints.

## Legend

**Status**
- `done`: follows logging.md; nothing left to do.
- `fix`: already has logging, but it breaks a rule in logging.md.
- `todo`: will get logging.
- `never`: no logging planned; the reason is given. Re-check if the file grows new I/O, loops over
  files, or caught exceptions.

**Priority** (for `fix` and `todo`)
- `P1`: fixes a bug, or runs on most calls in the MCP server and the host programs (loading and
  reading log roots).
- `P2`: called directly by LogDelta or VisualLogAnalyzer, or reached by the MCP server less often.
- `P3`: research and benchmark code: dataset-specific loaders, parser implementations, explainer.

**Keep print**: prints that stay because the caller asked for them (logging.md §4).

When you change a file, update its row: set the status (with the date when `done`) and trim the
plan to what is left.

## Overview

| Status | Files |
|---|---|
| done | 32 |
| fix | 0 |
| todo | 0 |
| never | 25 |

All P1/P2/P3 files converted 2026-09-14. Remaining `never` rows are unchanged from the design pass.

## `loglead/`

| File | Status | Pri | Plan / notes | Keep print |
|---|---|---|---|---|
| `__init__.py` | never | | Lazy re-exports only. | |
| `log.py` | done (2026-09-14) | P1 | `enable_console_logging(level)` added (logging.md §5). | |
| `OOV_detector.py` | never | | Small fit/predict model; `AnomalyDetector` reports on it. | |
| `RarityModel.py` | never | | Same as `OOV_detector.py`. | |
| `anomaly_detection.py` | done (2026-09-14) | P2 | `evaluate_all_ads` "Running X" → INFO. `bayesian_optimization` "F1 optimization time taken" → DEBUG. "Model type not supported for feature importance extraction" → `warnings.warn`. No-labels `warnings.warn` text unchanged (`delta/anomaly.py` still filters on it). | `print_scores`, `print_confusion_matrices`, `_print_evaluation_scores`, "Total time" under `print_scores`, `LogDistance.measure_all_distances(print_values=True)` |
| `column_analyzer.py` | never | | All prints are in `print_predictor_report`. | `print_predictor_report` |
| `explainer.py` | done (2026-09-14) | P3 | `calc_shapvalues`: print before the bare `raise ResourceWarning` moved into `raise ResourceWarning("...")`, no log record. | `print_log_content_from_nn_mapping`, `print_false_positive_content`, `print_false_negative_content`, feature list in `plot` |
| `next_event_prediction.py` | never | | Pure model code. | |

## `loglead/delta/`

| File | Status | Pri | Plan / notes | Keep print |
|---|---|---|---|---|
| `__init__.py` | never | | Lazy re-exports; the print is a docstring example. | docstring example |
| `anomaly.py` | done (2026-09-14) | P2 | `anomaly_file_content`: DEBUG per skipped target file (no comparison folder with that file, or empty target aggregate), plus one INFO per call with the total skipped. `warnings.catch_warnings` filter kept. | |
| `distance.py` | done (2026-09-14) | P3 | `distance_line_content`'s per-file skip (no lines on one side) → DEBUG. `comparable_files` has no silent skip. | |
| `export.py` | never | | Writes one file and returns its path; the caller reports it. | |
| `log_root.py` | done (2026-09-14) | P1 | `prepare_files` missing-files warning → WARNING. `os.stat` failures in `count_log_root_files`/`peek_log_root` → DEBUG. `_probe_file`'s catch-all → DEBUG. `read_log_root`: one INFO with folders/files/format/rows/dropped. | |
| `masking.py` | never | | Looks up patterns by name; raises on an unknown name. | |
| `scoring.py` | never | | Pure column arithmetic. | |
| `split.py` | done (2026-09-14) | P2 | `split_log_file`: one INFO per split (input, parts, output dir); DEBUG per part written. | |
| `visualize.py` | done (2026-09-14) | P3 | DEBUG timings added to `_document_term_matrix` and `_umap_2d`. | |
| `vocabulary.py` | never | | Skipped files are already returned as `skipped_files`. | |

## `loglead/enhancers/`

| File | Status | Pri | Plan / notes | Keep print |
|---|---|---|---|---|
| `__init__.py` | never | | Re-exports. | |
| `eventlog.py` | done (2026-09-14) | P2 | Every `parse_*` wrapped by a `_log_parse` decorator: one INFO (parser name, rows, seconds) plus a guarded DEBUG with the distinct-template count of whichever `*_id` column the call added. Commented IPLoM null-count print deleted. | |
| `sequence.py` | never | | Column derivations on DataFrames. | |

## `loglead/loaders/`

| File | Status | Pri | Plan / notes | Keep print |
|---|---|---|---|---|
| `__init__.py` | never | | Re-exports. | |
| `base.py` | done (2026-09-14) | P1 | `check_for_nulls_and_non_utf8` → one WARNING per column. Added shared `_log_nulls_and_non_utf8` helper used by `auto.py`/`json.py`/`syslog.py`/`logfmt.py`/`access_log.py`/`delimited.py` instead of six duplicated reports. | |
| `auto.py` | done (2026-09-14) | P1 | WARNING: HDFS without `anomaly_label.csv`, assumed `split_component=True`, undecodable characters, `df_seq` dropped on merge, non-UTF-8 column (via shared helper). INFO: no format matched, "is the X dataset", "probed N of M", "detected N formats". | |
| `access_log.py` | done (2026-09-14) | P1 | "N of M lines did not match": WARNING if dropped, INFO if kept unparsed. "Could not read"/"could not parse" → WARNING. Null/UTF-8 report via shared helper. | |
| `adfa.py` | never | | Thin dataset loader; `base.py` covers the shared checks. | |
| `awsctd.py` | done (2026-09-14) | P3 | "No valid data files processed." / "DataFrame is empty" → WARNING. | |
| `bgl.py` | never | | Thin dataset loader. | |
| `delimited.py` | done (2026-09-14) | P1 | `#fields`/`#types` mismatch and "could not parse" → WARNING. Null/UTF-8 report via shared helper. | |
| `hadoop.py` | done (2026-09-14) | P3 | Empty CSV caught as `NoDataError` → DEBUG. | |
| `hdfs.py` | never | | Thin dataset loader. | |
| `json.py` | done (2026-09-14) | P1 | Dropped non-object lines and "could not parse" → WARNING. Null/UTF-8 report via shared helper. | |
| `line_policy.py` | never | | Pure line-grouping functions. | |
| `lo2.py` | done (2026-09-14) | P2 | `__init__`: "Service type set"/single_error_type → INFO; invalid service type → WARNING. `load`: random error type → INFO; "no errors"/"not found"/"not enough unique" → WARNING; per-file processing errors combined into one WARNING per call (logging.md §3.7). `load_metrics`: "Processing metrics" → DEBUG; processing errors combined into one WARNING. | |
| `logfmt.py` | done (2026-09-14) | P1 | "could not parse timestamp" → WARNING. Null/UTF-8 report via shared helper. | |
| `nezha.py` | done (2026-09-14) | P3 | `logger.info` calls logging raw log messages → guarded DEBUG, truncated to 200 chars. "Error concatenating group" → ERROR; "Columns in group" → guarded DEBUG. JSON decoding/processing errors → one WARNING per file, traceback at DEBUG. Commented prints/logger calls and unused `traceback` import removed. | |
| `pro.py` | never | | Thin dataset loader. | |
| `raw.py` | done (2026-09-14) | P2 | `load`: one INFO with files and rows read; DEBUG per file. | |
| `supercomputers.py` | never | | Thin dataset loader. | |
| `syslog.py` | done (2026-09-14) | P1 | "N of M lines did not match": WARNING when dropped, INFO otherwise. "could not parse timestamps" → WARNING. Null/UTF-8 report via shared helper. | |

## `loglead/mcp/`

| File | Status | Pri | Plan / notes | Keep print |
|---|---|---|---|---|
| `__init__.py` | never | | Lazy re-exports. | |
| `__main__.py` | never | | Calls `main()`. | |
| `crash.py` | done (2026-09-14) | P3 | All 8 best-effort `except` blocks now log DEBUG with the swallowed error. ~0.2ms per-call cost unaffected (disabled DEBUG is a level check). | |
| `formatting.py` | never | | Pure result formatting. | |
| `mask_registry.py` | done (2026-09-14) | P3 | `register`/`delete` → INFO. `_check_pattern_compiles` already raises; no log. | |
| `server.py` | done (2026-09-14) | P1 | Crash sweep print → `logger.warning`. Added `--log-level` / `LOGLEAD_MCP_LOG_LEVEL` (default INFO), applied to the `loglead` logger in `main()`. Added the WARNING-to-`notes` bridge in `tool()` via a per-call `_NoteCollector` handler on the `loglead` logger, merged/capped at 10 (logging.md §6). Kept `redirect_stdout(sys.stderr)`. | |
| `session.py` | done (2026-09-14) | P2 | `SessionStore.open`: parquet cache hit/miss → DEBUG. `close`, `clear_cache` → INFO. | |

## `loglead/parsers/`

| File | Status | Pri | Plan / notes | Keep print |
|---|---|---|---|---|
| `__init__.py` | done (2026-09-14) | P1 | Module logger at DEBUG for the optional tensorflow import; no longer calls `logging.warning` on the root logger. | |
| `AEL/AEL.py` | never | | Commented prints left for the next time this file is touched. | |
| `Brain/Brain.py` | never | | Same as `AEL.py`. | |
| `bert/bertembedding.py` | done (2026-09-14) | P3 | Device lists → guarded DEBUG; "Using basebert/albert" → INFO; "Time taken" → DEBUG. | |
| `drain3/drain.py` | never | | Thin wrapper around the `drain3` package. | |
| `iplom/IPLoM.py` | done (2026-09-14) | P3 | `logger.warn` → `logger.warning`. Per-event "discarding" message → DEBUG, plus one WARNING per `Step1()` call with the count. Prints in `except` blocks and the stray print → DEBUG. Commented prints deleted. | `PrintPartitions`, `PrintEventStats` |
| `iplom_llm/iplom_llm.py` | done | | Nothing to log; the wrapped `iplom_llm_parser` package logs under its own name. | |
| `lenma/lenma.py` | done (2026-09-14) | P3 | Stray print in `get_similarity_score` case 4 deleted. | `print_wordlens` |
| `pl_iplom/pl_iplom.py` | done (2026-09-14) | P3 | All `logger.debug` f-string calls converted to passed arguments (logging.md §3.6). | `print_cluster_info` and helpers |
| `pyspell/spell.py` | done (2026-09-14) | P3 | `re_param`'s per-call separator/sequence prints deleted. | `dump` |
