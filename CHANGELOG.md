# Changelog

Notable changes to LogLead. Versions follow [semantic versioning](https://semver.org/): a major
bump means something that worked before needs changing.

## 2.0.0 - 2026-09-17

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
