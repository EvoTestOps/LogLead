# Logging in LogLead

Status: implemented. Per-file progress is tracked in
[logging_status.md](logging_status.md).

## Summary

- **Library:** the standard library `logging` module. No new dependency.
- **One logger per module:** `logger = logging.getLogger(__name__)`, so every LogLead logger sits
  under the `loglead` name and a host can control all of them at once.
- **LogLead never configures logging.** No handlers, no `basicConfig`, no levels inside `loglead/`.
  The program that runs LogLead (a script, LogDelta, VisualLogAnalyzer, the MCP server) decides
  what is shown and where. The only exceptions are the MCP server's `main()` and an opt-in helper
  for scripts (§5).
- **Nothing is written to stdout.** When the MCP server runs over stdio, stdout carries the
  protocol; one stray line breaks the client connection.
- **`print` stays only where the caller asked for printed output** (`print_scores()`,
  `print_values=True`, ...). Everything LogLead reports on its own initiative becomes a log record.
- **The MCP server copies warnings into tool results.** WARNING records from a tool call are added
  to the result's `notes`, so the MCP client sees data problems that today only reach stderr.

## 1. Why the standard library `logging`

LogLead runs inside other programs: LogDelta, VisualLogAnalyzer, the MCP server, notebooks. Those
programs, and the libraries next to LogLead (the MCP SDK, drain3, sklearn), already use standard
`logging`. Records from LogLead then go to whatever handlers the host has set up, with no bridge
and no second configuration. It also works on every supported Python (3.9–3.12).

Alternatives considered:
- **loguru** adds its own stderr output when it is imported and keeps a separate global logger.
  A host would have to configure it separately, and LogLead would add output the host never
  asked for.
- **structlog** is aimed at applications that want structured output. A host that wants JSON logs
  can get them from standard records with its own formatter.

## 2. Where LogLead runs and what happens to its log records

Checked 2026-09-13 against the sibling repositories and the installed MCP SDK (2.0.0).

| Environment | Who configures logging | A `loglead` WARNING | A `loglead` INFO |
|---|---|---|---|
| Scripts, demos, tests, notebooks | nobody | Python's fallback handler prints the bare message to stderr | not shown |
| LogDelta (imports `loglead` directly) | nobody | same as above | not shown |
| VisualLogAnalyzer (imports `loglead` directly) | its `logging.error()` calls, which run `basicConfig()` on first use | stderr, as `WARNING:loglead.loaders.auto:...` | not shown (root level is WARNING) |
| MCP server, `loglead-mcp` | the MCP SDK: `basicConfig(level="INFO")` with a stderr handler, when the server object is created | stderr, and copied into the tool result's `notes` (§6) | stderr |

No code in LogDelta, VisualLogAnalyzer, or LogLead's tests and demos reads LogLead's printed output,
so moving prints to logging breaks no caller. What users will notice:
- Status messages (future INFO records) no longer appear by default.
- Warnings move from stdout to stderr.

Mention both in the release notes.

**Existing bug, fixed first.** `loglead/parsers/__init__.py` calls `logging.warning()` on the root
logger when `BertEmbeddings` cannot be imported, which is the normal case without tensorflow. A root
logger call adds a stderr handler to the root logger if none exists. Verified: after
`import loglead.parsers`, a host's own `logging.basicConfig(...)` does nothing. Every program that
sets up logging after the first parser import loses its configuration.

## 3. Rules for code in `loglead/`

### 3.1 Getting a logger

```python
import logging

logger = logging.getLogger(__name__)
```

- One logger per module, defined right after the imports.
- `import logging` at the top of a module is allowed everywhere, including `delta/log_root.py` and
  `delta/visualize.py`. It is a cheap standard-library import; the lazy-import rule is about
  sklearn, xgboost and umap.
- The module-level logger does not break the "no module state" rule in `delta/`: it holds no data
  from any call.
- Never call `logging.debug/info/warning/error(...)` on the `logging` module itself. Those calls go
  to the root logger and can add a handler to it (§2).
- Use `logger.warning`, never `logger.warn` (a deprecated alias).

### 3.2 Library code never configures logging

Inside `loglead/`, do not:
- add handlers, including `NullHandler`
- call `basicConfig`, `dictConfig` or `fileConfig`
- call `setLevel`, set `propagate`, or call `logging.disable`

Exceptions: the MCP server's `main()` in `loglead/mcp/server.py`, which is a program entry point,
and the opt-in helper in §5.

Why no `NullHandler`: Python's docs suggest libraries attach one to their top-level logger. But
with one attached, a program without logging setup shows nothing at all. Warnings about unparsed
timestamps or dropped lines would silently vanish in LogDelta and the demos, where the `WARNING!`
prints are visible today. Without it, Python's fallback handler still prints WARNING and above to
stderr.

### 3.3 Never write to stdout

- No `print` for diagnostics, and no `StreamHandler(sys.stdout)`.
- The `tool()` wrapper in `mcp/server.py` redirects stdout to stderr during tool calls. Keep it as
  a safety net for prints from third-party libraries, not as permission to print. A handler that
  captured `sys.stdout` before the redirect still writes to the real stdout.

### 3.4 Choosing a level

| Level | Use for | Examples in LogLead |
|---|---|---|
| DEBUG | Detail for debugging LogLead itself: step timings, per-file or per-partition detail, cache hits, expected failures that were caught | PL-IPLoM step timings; a parquet cache hit; a file skipped because it could not be stat'ed |
| INFO | A few milestones per public call that someone running a script wants to see: what was read, what was detected, what was chosen | `AutoLoader: detected 2 format(s) across 14 file(s)`; `Running train_LR`; `JsonLoader: 3 of 12 columns contain nulls` |
| WARNING | The call succeeded, but the result may be wrong or incomplete and a person should look | unparsed timestamps; dropped lines; non-UTF-8 values; requested files that are absent and skipped; `df_seq` dropped when formats are merged |
| ERROR | Something failed, LogLead carried on, and a significant part of the output is missing | a whole group of files could not be combined and is left out |
| CRITICAL | Not used | |

When converting existing prints:
- `WARNING!` or `Warning:` prints become WARNING.
- `Error ...` prints inside `except` blocks become WARNING, or ERROR if a large part of the output
  is lost (§3.8).
- Other status prints become INFO.
- Stray debug prints inside algorithm code become DEBUG, or are deleted.
- Commented-out prints are deleted.

### 3.5 Logging, `warnings`, exceptions or `print`?

| Situation | Use |
|---|---|
| The call cannot produce a result | Raise an exception. Don't also log it. |
| The caller should change how they call the API (a deprecated argument, an option that has no effect, no labels for a supervised method) | `warnings.warn(..., stacklevel=2)`. Callers can filter it, and it is shown once per call site. |
| Something is off with the data or the environment, but the call itself is fine | `logger.warning(...)` |
| The caller asked for printed output | `print` (§4) |
| Progress and status | `logger.info(...)` / `logger.debug(...)` |

The existing `warnings.warn("WARNING! data has no labels...")` in `anomaly_detection.py` stays a
warning. `delta/anomaly.py` filters it by its text, so change both together or neither.

### 3.6 Writing messages

- **Pass arguments, don't pre-format.** Write
  `logger.info("AutoLoader: detected %d format(s) across %d file(s)", n, total)`, not an f-string.
  The string is then only built when the record is actually shown.
- **Guard expensive arguments.** A Polars filter, a `.to_list()` or a count runs even when the level
  is off, so check first:
  ```python
  if logger.isEnabledFor(logging.DEBUG):
      logger.debug("nezha: columns in %s: %s", group, collected_df.columns)
  ```
- **Keep the component prefix used today** (`AutoLoader:`, `JsonLoader:`, `lo2:`). Without host
  configuration Python prints only the message, with no level or logger name, so the prefix is the
  only clue where it came from. Drop the `WARNING!` and `Warning:` prefixes; the level says that.
- **Make each message self-contained.** Name the file, column or folder and give counts
  (`5 of 1200 values`), so the line makes sense in a server log with nothing around it.
- **One event, one call.** A hint that belongs to a warning goes into the same message, even if it
  spans lines. For example, `base.py`'s "You have 4 options" text becomes part of that column's
  warning.
- **No log line content at INFO or above.** The logs LogLead analyses may contain credentials or
  personal data, and server logs are kept and shared. At DEBUG, truncate samples to about 200
  characters.

### 3.7 Volume

- INFO: a handful of records per public call, however large the input.
- Per file, per row or per partition: DEBUG only.
- Repeated warnings in a loop are combined into one record with the count and the first few names:
  `lo2: 3 of 250 log files could not be read and were skipped: a.log, b.log, c.log`.
- Volume also matters for reliability. An MCP client that starts the server over stdio may never
  read the server's stderr. Once that pipe's buffer fills (typically 64 KB on Linux), the server
  blocks on its next write and the tool call hangs.

### 3.8 Exceptions

- Raise or log, not both. Hosts log the exceptions they catch, so doing both reports the problem
  twice.
- When a caught exception makes the result incomplete, log WARNING with the error text, and put
  the traceback at DEBUG only:
  ```python
  except Exception as error:
      logger.warning("lo2: could not read %s, skipped: %s", path, error)
      logger.debug("lo2: traceback for %s", path, exc_info=True)
  ```
- When failure is an expected answer (probing a file's format, the best-effort crash log, reading
  JSON keys from a file that isn't JSON), log at DEBUG or not at all. Don't make expected outcomes
  look like problems.

## 4. Output that stays `print`

These functions exist to print. Turning their output into log records would hide output the caller
explicitly asked for.

| File | Function or flag |
|---|---|
| `anomaly_detection.py` | `AnomalyDetector.print_scores`, `print_confusion_matrices`, `_print_evaluation_scores` (runs when `print_scores`, `auc_roc`, `f1optimize` or `f_importance` is set), the `Total time` line under `print_scores`, and `LogDistance.measure_all_distances(print_values=True)` |
| `column_analyzer.py` | `print_predictor_report` |
| `explainer.py` | `print_log_content_from_nn_mapping`, `print_false_positive_content`, `print_false_negative_content`, the feature list printed by `plot` |
| `parsers/iplom/IPLoM.py` | `PrintPartitions`, `PrintEventStats` |
| `parsers/pl_iplom/pl_iplom.py` | `print_cluster_info` and its helpers |
| `parsers/pyspell/spell.py` | `dump` |
| `parsers/lenma/lenma.py` | `print_wordlens` |
| `delta/__init__.py` | the `print` in the module docstring example (not executed) |

For new code: a function that prints must say so in its name (`print_*`) or through an explicit
flag (`print_values=True`), and should also return the data it prints.

## 5. Guidance for programs that use LogLead

Standard `logging` configuration is enough:

```python
import logging

logging.basicConfig(level=logging.INFO)                        # see LogLead's INFO records too
logging.getLogger("loglead").setLevel(logging.WARNING)         # or: only LogLead's warnings
logging.getLogger("loglead.parsers").setLevel(logging.ERROR)   # quiet one area
```

For scripts, demos and notebooks that have no logging setup, `loglead/log.py` will provide an
opt-in helper:

```python
from loglead.log import enable_console_logging

enable_console_logging("INFO")
```

What it does:
- Attaches one stderr handler to the `loglead` logger. A repeated call changes the level instead of
  adding a second handler.
- Sets the `loglead` logger's level.
- Uses the format `%(levelname)s %(name)s: %(message)s`.

Programs that configure logging themselves should not call it: LogLead's records would then be
printed twice, once by this handler and once by the program's root handler.

## 6. MCP server

- **Level.** The SDK sets the root logger to INFO with a stderr handler. Add a `--log-level` option
  and a `LOGLEAD_MCP_LOG_LEVEL` environment variable that set the `loglead` logger's level
  (default `INFO`). They sit next to the existing `LOGLEAD_MCP_CACHE` and `LOGLEAD_MCP_CRASH_LOG`.
- **Startup message.** The `print(..., file=sys.stderr)` in `main()` about a previous crash becomes
  `logger.warning`.
- **Warnings reach the client through `notes`.**
  - For the duration of each call, the `tool()` wrapper attaches a collecting handler to the
    `loglead` logger. The handler itself is set to WARNING; the logger's level is not changed.
  - After the call, the handler is removed in `finally`, and the collected messages are appended to
    `result["notes"]` after the tool's own notes. This only happens when the result is a dict, the
    same condition as `elapsed_seconds`.
  - Identical messages are merged. At most 10 are kept, followed by a line saying how many more are
    on the server's stderr.
  - Setting `--log-level` above WARNING also removes these notes.
- **Why `notes`, not MCP log notifications.** The model reads tool results. MCP log notifications
  usually end up in a client-side log view, if they are shown at all. `notes` is already where the
  server tells the model what is wrong or what to do next (see the `tool()` docstrings in
  `server.py`).
- **Concurrency.** The server answers one call at a time, so one collecting handler is enough. If
  calls ever run concurrently, key the collection on the call. `crash.py`'s breadcrumb has the same
  caveat.
- **`crash.py`.** Failures it swallows stay swallowed; add DEBUG records so a broken cache directory
  can be diagnosed. This must not add measurable cost to the ~0.2 ms it spends per call. A disabled
  `logger.debug` with arguments costs only a level check.

## 7. Converting a file

1. Find the file's row in [logging_status.md](logging_status.md).
2. Add the module logger (§3.1). Convert each print using the mapping in §3.4, and keep the prints
   listed in §4.
3. Apply §3.6 (arguments, prefixes, no log line content at INFO or above) and §3.7 (combine
   repeated warnings in loops).
4. Verify:
   - Run the smoke test that covers the file:
     - loaders, enhancers, detectors: `uv run demo/HDFS_samples.py`, `uv run demo/TB_samples.py`
     - `auto.py` and the format-spec loaders: `uv run tests/log_file_detection.py --config <cfg>`
     - anything the MCP server reaches (`delta/`, `mcp/`, loaders used through `AutoLoader`):
       `uv run tests/mcp/server.py`
   - In a scratch script, call `enable_console_logging("DEBUG")`, exercise the file, and check that
     the records appear with the intended levels and logger names.
   - Run the checks in §8.
5. Update the row: status `done` with the date, and list any prints that remain.

## 8. Checks

```bash
# Root-logger calls or logging configuration inside the library.
# Expected hits: server.py main() and loglead/log.py only.
grep -rnE 'logging\.(debug|info|warning|warn|error|exception|critical|basicConfig|dictConfig|fileConfig|disable)\(|addHandler|NullHandler|setLevel' loglead

# Deprecated alias.
grep -rnE 'logger\.warn\(' loglead

# Remaining prints. Expected hits: only the functions in section 4.
grep -rnE '(^|[^a-zA-Z_.])print\(' loglead

# Anything writing to stdout directly.
grep -rn 'sys.stdout' loglead
```
