# OOM/crash handling: what's implemented, what's deliberately not, and why

Companion to [`crash.py`](crash.py) and the OOM discussion in `CLAUDE.md`. That
code implements **option A** (report and recover) from the option list below.
This file is the rest of that list, kept so the decision not to build B/C/D
yet is a decision on record rather than something forgotten.

**Status as of writing: only the stdio transport has ever been used against a
real MCP client (Claude Code).** stdio clients spawn the server process
themselves, so they generally *can* restart it — and in practice, do. That is
the load-bearing fact behind "ship A, wait": A's entire value proposition (a
breadcrumb the next process can read) is worthless unless something restarts
the process to read it, and so far something reliably has. Revisit the
priority of everything below once one of these becomes true:

- LogLead MCP is run over `--transport http`/`sse` for real, where nothing
  autoswaps a dead process — that shifts weight toward **B2/B1** (refuse
  before the kill, since there is no restart to recover into) or **C2** (a
  side-car that survives the crash instead of merely reporting it).
- A client turns out *not* to restart reliably (some do, silently retry
  forever with no backoff; some surface a raw connection-closed error and stop
  asking). If crash reports routinely go unread because nothing reconnects,
  A's reporting is real but its audience is not — worth asking users directly
  rather than guessing from the code.
- A client is observed reading `server_crash`/`notes` and *acting on the
  advice* (narrowing the next call) vs. just retrying the same call and dying
  again. That is the actual test of whether A solved the problem or just
  documented it.

## What A already covers

Implemented: [`crash.py`](crash.py), wired into `server.py`'s `tool()`
wrapper and `open_log_root`. Breadcrumb-before-call, swept-on-restart,
reported-once-in-notes, ledger-keyed-on-log-root, recovery call handed back
via `Session.open_args()`. Cost measured at ~0.2ms/call. See `CLAUDE.md`'s
`crash.py` paragraph for the full description and the test stage
(`tests/mcp/server.py --only crash`) that exercises it with a real `SIGKILL`.

What A does **not** do: prevent the kill, survive it (the process still dies,
sessions still vanish), or know *why* it died (it infers "likely OOM" only
from recorded RSS share, per `OOM_LIKELY_SHARE`).

---

## B2 — refuse before the call, based on a memory estimate

The option the user asked to weigh most carefully, because the deployment
range is wide open: a personal 8GB laptop up to a 256GB VM, so a hardcoded
threshold is wrong somewhere on that range by construction. Considerations,
backed by the recorded cells in `~/Datasets/test_data/mcp_bench_cells` (same
data behind `tests/mcp/PERFORMANCE.md`/`PERF_MEMORY.md`):

### The estimate itself is unreliable across shapes

A per-line model is off by >5x across the three benchmark corpora.
`anomaly_folder_content` peak per log line at 100%:

| log root | shape | peak / line |
|---|---|---|
| hadoop_renamed | 55 folders, 180,897 lines | 9.8 KB |
| hdfs_balanced_5k | 5,000 folders, 90,862 lines | 13.7 KB |
| bgl_split_10 | 10 folders, 4,747,963 lines | 2.6 KB |

Per *folder* it is worse — 32MB / 0.25MB / 1,235MB, a 5,000x spread — because
folder count and line count trade off against each other depending on the
dataset's shape (many small folders vs. few huge ones). Calibrating on one
corpus over- or under-refuses on the other two. Lines-per-folder is the
better predictor and it still misses by 5x.

### The same call varies 2x+ on parameters alone

On bgl at 100%, `anomaly_folder_content` peaks at 7.38GB with KMeans only and
**OOMs** with OOVDetector added. `anomaly_folder_filename`: 3.40 / 10.05 /
13.69 / 15.26 GB for KMeans / IsolationForest / RarityModel / OOVDetector
respectively — consistent ordering across all three log roots, so it is a
property of the code (worth modeling), but any single-number estimate for a
tool is really "max over whichever detectors were asked for", and the
expensive end is dominated by the last two.

Side effect worth resolving separately: this makes narrowing `detectors` a
genuine *memory* lever, while `_SUBSET_NOTE` in `server.py` currently
discourages narrowing on *statistical* grounds (weaker rank_sum). Those two
pieces of advice now pull in opposite directions and a future B2 would need
to reconcile them — e.g. only advise dropping OOVDetector/RarityModel
specifically, not "add all four back" unconditionally.

### A call's cost isn't even cleanly measurable, let alone predictable

glibc does not return freed memory to the OS (already recorded in `CLAUDE.md`:
close + `gc.collect()` + reopen still leaves 6.6GB resident against 2.9GB
fresh). This shows up directly in the cells: `anomaly_folder_content
(OOVDetector)` on hadoop reads only a 0.05GB delta above a 2.43GB floor,
because the *previous* call's peak in that same process never came back. So
the same call costs a different amount depending on what else ran earlier in
the session — any predictor is really predicting a **process** peak that
depends on session history, not a call cost that can be looked up per tool.

### The cross-machine part is the easy half, with caveats

Both inputs needed are cheap and local: `df.estimated_size()` (5-140µs,
measured) and `psutil.virtual_memory().available` (46µs, measured). A guard
shaped as `predicted ≈ current_rss + k × lines_in_scope × format_multiplier`
compared against *measured* available memory scales itself from 8GB to 256GB
with no configuration. Three things that would silently break it:

- **Containers report the host, not the limit.** `psutil.virtual_memory()`
  reads host memory; a 4GB container on a 256GB host reads 256GB available.
  Would need to read cgroup v2 `memory.max`/`memory.current` when present —
  this is the single most likely way a "cross-machine" guard is wrong on
  exactly the big-VM case the user is worried about protecting.
- **Swap** turns an OOM into thrashing rather than a kill; a RAM-only guard
  refuses calls that would have finished (slowly), which is a worse outcome
  than letting them run on a box with generous swap.
- **WSL2** (the user's own environment) reads the VM's memory ceiling
  correctly, which is the good case, but is one more thing that needed
  checking rather than assuming.

### A call-site warning is worthless if the call proceeds

This is the design point most worth keeping in mind if B2 gets built: if the
guard lets the call through, any "you're close to the limit" warning rides on
a result that, if the estimate was wrong, never arrives — there is no partial
message. So a call-site guard can only ever be binary (refuse / allow), never
advisory. The advisory version has to live somewhere the result is guaranteed
to come back — which is why **B2-lite** (below) was proposed as an
`open_log_root`-time note instead of a per-call gate.

### Override design, if a refusal is ever built

Avoid a per-call `force=True` parameter on six-plus tool schemas — a model
that can see the flag will reach for it to get past a refusal, which is
exactly the failure mode `_SUBSET_NOTE`/`_ANOMALY_NOTE` already exist to
prevent for a different reason (narrowing detectors to save time). Better
shapes: a server-level `--memory-guard off` flag (operator decision, not
model-reachable), or a two-step override (`set_memory_guard(session_id,
"off")` as its own deliberate call, then retry) so bypassing costs a visible
second action rather than one inline flag flip.

### How A changes B2's cost/benefit

Before A existed, an OOM was silent data loss. Now it is reported, attributed
to a call, and one `open_log_root` call from recovery — so a **false
refusal** (B2 blocks a call that would actually have fit) is comparatively
more costly than it used to be relative to a **false negative** (B2 lets
through a call that OOMs, now safely reported by A). That argues for tuning
any eventual guard conservatively — refuse only the clearly-doomed
(`predicted(2.6KB/line-floor) > available`), not the merely-large.

### Recommended validation path, if/when this gets built

Add a predicted-vs-actual column to `tests/mcp/benchmark.py`'s grid so
estimator accuracy is a number per cell, not an argument. Ship the advisory
(B2-lite) first since it can't be wrong in a way that blocks anything; only
build the hard refusal once the predictor is shown to land within ~2x across
all three benchmark shapes.

### B2-lite — the fallback that's actually easy to justify

Not a guard at all: at `open_log_root` time, once the frame size and folder
count are already known for free, emit a note like *"this log root is 4.7M
lines over 10 folders; on this machine (16GB, 9GB free) `anomaly_*` with
target_folder="ALL" is likely to exceed available memory — call it per
target"*. No threshold sits on the hot path, nothing is ever blocked, it's
machine-aware via the same `psutil` read A already uses, and it composes
directly with the crash ledger note that already prints at that call site
(`crash_log().history(...)` in `open_log_root`). This is the natural
next increment if/when B2 becomes worth doing at all — cheap, low-risk,
and independent of the accuracy problems above since it only ever advises.

---

## Other options considered and not built

From the original five-group survey (A/B/C/D). A is done; B2 is covered
above. The rest, for completeness:

### B1 — preflight budget check (general case of B2)

Same idea as B2 but for every heavy tool, not just the anomaly family:
resolve the targets, read available memory, refuse if the estimate exceeds
it, with an `allow_large=True` escape hatch. Strictly harder than B2 (needs a
cost model per tool family, not just anomaly detectors) and inherits every
reliability problem above. Not started; B2's validation path would need to
generalize before this is worth attempting.

### B3 — change the risky default

`target_folder` defaults to `"ALL"` on every `anomaly_*` tool, and that
family refits four detectors *per target* — so the default itself is the
single most common route to an OOM (see the OOVDetector numbers above: fine
per-target, OOMs at target_folder="ALL" on bgl). Flipping the default to
require an explicit target (or a smaller implicit one, like the single
largest folder) would cut off the worst case by construction rather than by
estimation.

**Why not done:** this is an API semantics change, not a crash-handling
change — it touches `run_config`'s LogDelta-config parity (LogDelta configs
that rely on the "ALL" default would silently start scoring less), the
benchmark's own canonical calls (`GRID_ROWS` in `tests/mcp/benchmark.py`
call `target_folder="ALL"` deliberately, to measure the worst case), and
every existing demo/doc that says "target_folder defaults to ALL". Worth
doing on its own merits (it is arguably the correct default regardless of
OOM) but it's a separate decision from how to handle the OOM that current
behavior can cause, so it was kept out of this batch.

### B4 — process memory limit + RSS watchdog

`--max-memory-gb` at server startup, enforced via `resource.setrlimit
(RLIMIT_AS, ...)`, backed by a watchdog thread (the same 20ms-interval
`_RSSMonitor` pattern `benchmark.py` already uses) that could kill the
process itself before the OS OOM-killer does, or at least log a richer
breadcrumb on the way down.

**Why not done:** `setrlimit(RLIMIT_AS)` makes Python's *own* allocations
raise `MemoryError`, which is catchable — but polars and sklearn do the bulk
of their allocation in Rust/C, where an allocation failure typically
**aborts** the process rather than raising into Python. So this does not
reliably turn an uncatchable kill into a catchable one; it mostly changes
*which* uncatchable death happens (`SIGABRT`/segfault vs. `SIGKILL`), and
plumbing per-process limits through both transports (stdio spawn args vs.
whatever spawns the HTTP server) is nontrivial for that payoff. Its one clear
win — a watchdog thread sampling RSS at 20ms can *detect* the approach to a
limit reliably even if it can't *stop* the allocation — is already achieved
more cheaply by A's start-of-call snapshot; a full watchdog would only add
value by making the crash breadcrumb's memory field a true peak instead of a
floor, which was judged not worth the complexity yet.

### C1 — run heavy tools in a child process

Parent (holding the session) forks or subprocesses for the named heavy
tools; the child's OOM-kill becomes a normal, observable exit code the
*parent* can turn into a proper tool-call error, in the same response, rather
than the connection just dying. This is the only option that gives literally
what the user first asked for ("recover and report back... in the same
call").

**Why not done:** breaks the session/cache model that's the entire point of
this server (see `session.py`'s docstring — "loading is cheap, parsing is
not, a session grows the frame in place"). Any `ensure_content` column
computed in the child during the doomed call is lost unless the child flushes
it to the parquet cache before dying, which it cannot reliably do (same
uncatchable-kill problem, one level down). Getting the result back from a
successful child call means pickling a polars frame across a process
boundary, which is its own cost. Worth doing narrowly (just
`anomaly_*`/`plot_*(umap)`, the tools the benchmark shows are actually heavy)
if C-class work is ever prioritized, but it's a bigger change than A/B and
was judged premature before knowing whether clients even restart reliably —
see the "Status" section at the top.

### C2 — session held in a side-car worker process

Server process (talking to the MCP client) stays tiny; a separate worker
process holds the actual session/frame. Worker OOMs → server survives,
reports immediately (no restart needed), and can re-attach from the parquet
cache in ~0.2s without the *client's* connection ever dropping. Also buys
cancellation and per-call timeouts for free.

**Why not done:** this is a session-layer rewrite (two processes instead of
one, an IPC boundary for every tool call, a supervision story for the worker
itself), justified only if OOM turns out to be routine rather than an
edge case hit mainly at bgl-scale (~4.7M lines, 10 folders) and
`target_folder="ALL"`. Given real usage so far is stdio-only against one
client, C2 solves a problem (surviving without a client restart) that hasn't
been shown to exist yet — the client has always been able to restart.
Revisit under the same conditions listed in "Status" above, particularly if
HTTP transport sees real use (where there's no client-initiated restart to
begin with, and C2's "survives without any restart" property becomes the
only way to avoid downtime for other sessions on that server).

### C3 — partial results as targets complete

`anomaly.anomaly_folder` (in `loglead/delta/anomaly.py`) already loops
per-target and concatenates the frames at the end. Flushing/writing each
target's result as it completes means an OOM at target 7 of 10 leaves 6
usable rows instead of zero, and A's breadcrumb could report exactly which
target was in flight when it died (it already can — `target_folder` is
recorded — but the *caller* currently gets nothing from targets 1-6 either).

**Why not done:** `loglead/delta/` must not write files or hold state (see
CLAUDE.md's invariant: "these functions hold no module state, never
os.chdir, and never write files — they return DataFrames"), so incremental
flushing would have to move the per-target loop, or a callback into it, up
into `mcp/`, which changes where the loop that library callers (LogDelta-config
`run_config`, the demo, direct Python use) rely on actually lives. Not
attempted; would pair naturally with A once the target-level breadcrumb
exists — A already records *which* target was running, so this is the most
directly connected of the C-options if picked up later.

### D1 — reduce the peak itself

For the genuinely catastrophic cells — `anomaly_folder_content (OOVDetector)`
on bgl at 100% (OOM), `anomaly_folder_filename (OOVDetector)` at 15.26GB — no
preflight check and no recovery scheme helps on an 8-16GB machine, because
the call cannot be made to fit by asking more politely. The only real fix is
a smaller peak: a hashing vectorizer instead of `CountVectorizer` (bounded
memory regardless of vocabulary size), chunked aggregation instead of
materializing the full comparison-folder document-term matrix at once, or
narrower dtypes in the detector pipeline.

**Why not done:** this is a numerical/algorithmic project on
`loglead/delta/anomaly.py` and the vectorizer path in `log_root.py`, separate
in kind from "how does the server behave when it runs out of memory" — it's
"how do we need less memory in the first place". Flagged here because for
the worst-measured cells (bgl + OOVDetector + target_folder="ALL") it is the
*only* option in this whole list that actually helps; A only makes the
failure legible, B/C only manage around it.

---

## Summary table

| # | Option | Prevents OOM? | Survives it? | Attributes it? | Status |
|---|---|---|---|---|---|
| A | Breadcrumb + ledger + advice | no | no (reports after) | yes | **done** |
| B1 | Preflight budget check (general) | partially | n/a | n/a | not started |
| B2 | Preflight check (anomaly family) | partially | n/a | n/a | analyzed, not built |
| B2-lite | Advisory note at open time | no (advisory only) | n/a | n/a | proposed, easy, not built |
| B3 | Change target_folder default | partially | n/a | n/a | not started (API change) |
| B4 | RLIMIT_AS + watchdog | unreliable (native allocs abort) | no | marginally better peak | not started |
| C1 | Child process per heavy tool | no | yes, same call | yes | not started (session-model cost) |
| C2 | Side-car worker process | no | yes, no restart needed | yes | not started (bigger rewrite) |
| C3 | Partial results per target | no | partially (partial data) | yes (pairs with A) | not started |
| D1 | Reduce the actual peak | yes, for the worst cells | n/a | n/a | not started (separate project) |

Prompted by: `tests/mcp/PERFORMANCE.md`/`PERF_MEMORY.md` benchmark data
(recorded OOM: `anomaly_folder_content (OOVDetector)` on bgl_split_10 at
100%), and the original six-way option survey done before A was scoped.
