# Changelog

Notable changes to LogLead. Versions follow [semantic versioning](https://semver.org/): a major
bump means something that worked before needs changing.

## Unreleased

### Added

- **Per-dataset masks** in `loglead.delta.masking`, one for each dataset family LogLead loads
  (`hdfs`, `bgl`, `hadoop`, `thunderbird`, `spirit`, `liberty`, `openstack`, `lo2`, `nezha`, `zeek`,
  `access_log`, `iis`, `syslog`, `logfmt`, `loghub`, `gha`, `pro_android`, `comp_ws`, `ait_ads`,
  `security_datasets`), and **`merged`**, their union. All share one ordering, most specific
  first, so an IPv4 address is `<IP>` rather than myllari's `<VERSION>`.

### Changed

- `merged` replaces `myllari_extended` as the default mask of `loglead.delta` and the MCP server.
  Pass `mask_pattern="myllari_extended"` to keep the old templates. `EventLogEnhancer.mask()`'s own
  default is unchanged.
- `list_mask_patterns` lists the per-dataset masks under `dataset_masks` as positions in `merged`,
  where their regexes are.
- `comparison_*` is renamed `baseline_*` throughout `loglead.delta` and the MCP tools:
  `comparison_folders` is now `baseline_folders`, and `distance_line_content` returns `baseline_n`
  and `baseline_pct`. The old names are not accepted. LogDelta configs keep `comparison_runs`.
- The distance tools other than `distance_line_content` and every anomaly tool now return a clean
  range by default: how far apart the baseline folders score from each other, with the target
  placed against it. Pass `threshold=False` for the old result.

## 2.1.0 - 2026-09-22

The headline addition is order-aware anomaly detection: two detectors that score a sequence by the
order of its events rather than by which events it contains.

### Added

- **`NextEventPredictionNgramDetector`** (next event prediction, an n-gram model of event order)
  and **`LookaheadPairsDetector`** (lookahead pairs: which event may follow which within a
  window), trained with `AnomalyDetector.train_next_event_prediction()` and
  `train_lookahead_pairs()`. They need an ordered list of parsed events per row;
  `evaluate_all_ads()` skips them otherwise.
- **`LookaheadPairs`**, alongside `NextEventPredictionNgram` in the new `loglead.sequence_modelling`
  module. It runs in O(n) per sequence.
- **`loglead.delta.sequence_line_event_prediction`** scores every line of a target file by how
  expected it is after the lines before it, learning event order from the same file in the
  comparison folders. The MCP server exposes it as a tool of the same name.

### Changed

- `RarityModel` is renamed to `rarity_detector` and `AnomalyDetector.train_RarityModel()` to
  `train_RarityDetector()`. The old names still work but raise a `DeprecationWarning`; they will be
  removed in 3.0.
- `loglead/next_event_prediction.py` is merged into `loglead/sequence_modelling.py`, and
  `loglead/log.py` is renamed to `loglead/logging.py`. Neither old module name is kept; update any
  direct import.

### Fixed

- With MCP SDK 2.x, a failing MCP tool reached the client as a bare "Error executing tool
  <name>", so an AI agent could not see why its call was rejected. The server now passes the
  exception type and message through on both SDK 1.x and 2.x. Direct Python callers still get
  the original exception.

## 2.0.0 - 2026-09-18

The headline additions are format detection (`AutoLoader`), five spec-driven loaders for everyday
log formats, the `loglead.delta` comparison pipeline, and an MCP server that exposes it to an AI
assistant.

### Breaking changes

| Removed / changed | Replacement |
|---|---|
| Python 3.9 support | Python 3.10-3.13. The MCP SDK needs 3.10+, so 3.9 could only ever install a crippled LogLead. |
| `GELFLoader` | `JsonLoader(format="gelf")` |
| `BaseLoader.lines_not_starting_with_pattern()` | The line-policy options in `loglead/loaders/line_policy.py` |
| `EventLogEnhancer.old_trigrams()` | `EventLogEnhancer.trigrams()` |
| `loglead.explainer` (SHAP-based) and its two demos | Removed with no replacement; `shap` and `nbformat` are no longer dependencies. |
| `tests/datasets.yml` | `tests/datasets_mid_labels.yml` (one of several `tests/datasets_*.yml` configs) |

`EventLogEnhancer.normalize()` is **renamed to `mask()`**. The old name still works and forwards to
`mask()`, but raises a `DeprecationWarning`; it will be removed in 3.0. Rename the call:

```python
# before
df = EventLogEnhancer(df).normalize(regexs=pattern_list)
# after
df = EventLogEnhancer(df).mask(regexs=pattern_list)
```

Dependency floors were raised to the oldest releases actually verified against Python 3.10
(polars >=1.38.1, scikit-learn >=1.5, xgboost >=2.1, scipy >=1.11 among others), and majors that
have broken the build are now capped. `jinja2` was dropped; it was unused.

### Added

- **`AutoLoader`** detects a log's format and builds the loader that reads it, for a single file or
  a whole tree (`filename_pattern`, detected per file). See `demo/AutoLoader_samples.py`.
- **Spec-driven loaders** — `JsonLoader`, `SyslogLoader`, `LogfmtLoader`, `AccessLogLoader` and
  `DelimitedLoader` read a format family from a YAML spec rather than a bespoke Python class.
  Shipped specs cover GELF, nginx (JSON and plus-status), Windows events (OTRF), Zeek, IIS,
  OpenStack, loghub and the common/combined access-log formats. A format with no shipped spec can
  be read by passing the spec keys as keyword arguments, or by pointing at a spec file of your own.
- **`loglead.delta`** — a pipeline that compares many log folders against each other:
  distance, anomaly and visualization questions at folder-name, folder-content, file and line
  granularity. Returns Polars DataFrames and plotly figures, holds no module-level state, writes no
  files.
- **MCP server** — `pip install "loglead[mcp]"`, then run `uv tool install "loglead[mcp]"` or   `loglead-mcp`. Loads, masks and parses a
  log root once per session and reuses it for every later question. An MCPB bundle for Claude
  Desktop is in `mcpb/`.
- **`LO2Loader`** and **OpenStack** dataset support, plus a reorganized downloader
  (`downloader/download_data.py --config <cfg>`) and per-family test configs.
- `EventLogEnhancer.minhash()`, `SequenceEnhancer.category_counts()`, `loglead.column_analyzer`
  (categorical-column profiling and predictor selection), and a `reparse=True` option on the
  event-level enhancers to recompute a column instead of short-circuiting on it.
- Structured logging throughout `loglead/` via `logging.getLogger(__name__)`; the library never
  configures logging or writes to stdout, so it stays safe to embed and to serve over stdio.



## Earlier versions

See the [release history on GitHub](https://github.com/EvoTestOps/LogLead/releases).
