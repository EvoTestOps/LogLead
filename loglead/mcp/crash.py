"""Reporting a crash that cannot be caught.

An out-of-memory kill is ``SIGKILL``: no exception, no traceback, no last line
on stderr, and no answer to the tool call that caused it. MCP has no crash
channel either -- the report would have to travel over the transport that just
died with the process -- and whether the client restarts the server is client
policy rather than protocol. The only channel that survives the kill is the
filesystem, so this module writes down what is about to run *before* running it
and reads what was left behind on the next start. It is the same trick
``tests/mcp/benchmark.py`` uses to keep measuring after the OOM killer takes a
block, moved into the server, where the reader is a model rather than a parent
process.

Three parts:

* a **breadcrumb** per process (``inflight-<pid>.json``), written before every
  call and removed after it, so a file left behind names the call that was
  running, the session it was running against, and how much memory was already
  resident when it started;
* a **ledger** keyed on the log root, so the report outlives the restart that
  found it -- a client opening the same log root next week is still told what
  died there and what to call instead. A crash is only useful if it is
  remembered where the next caller will trip over it;
* **advice**, derived from the recorded arguments, saying which argument made
  the call big. "The server died" is not actionable; "target_folder='ALL' was
  10 log folders and the anomaly family refits four detectors per target" is.

What this cannot do is say *why* the process died. From inside there is nothing
to observe: the successor process finds a breadcrumb and an absence, which is
consistent with an OOM kill, a ``kill -9``, a segfault, or a laptop lid. So the
report states what is known -- the call, and the memory situation when it
started -- and names an OOM kill as the likely cause only when the numbers say
so. Guessing louder than the evidence would teach a client to shrink calls that
were never the problem.

The whole thing runs in front of *every* tool, because which call is the
expensive one is exactly what is not known in advance -- so it is kept to one
encode and one write, measured at **~0.2ms per call**: 7% of the cheapest tool
in ``tests/mcp/PERFORMANCE.md`` (``query_result``, 3ms) and invisible on the
rest. ``LOGLEAD_MCP_CRASH_LOG=0`` turns it off for anyone who disagrees.

Two consequences of being crash-survivable are worth keeping. The breadcrumb is
per *pid*, so several servers may share one cache directory and only reap each
other's remains once the owning process is really gone (checked by pid *and*
process start time, since pids are reused). And every operation here is
best-effort: a crash log that raises would turn a working server into a broken
one over bookkeeping, so failures to read or write it are swallowed.

One breadcrumb per process, not per call, because the server answers one tool
call at a time. Were that to change, two calls in flight would overwrite each
other's breadcrumb and the report would name the wrong one -- key the file on
the call as well as the pid before making tool calls concurrent.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import psutil

#: Crash records kept per cache directory, oldest dropped first. The ledger is
#: read on every ``open_log_root``, so it stays small enough to read in full.
MAX_LEDGER = 20

#: Bounds on a recorded argument: a ``folder_names`` mapping can hold 5,000
#: entries and a breadcrumb has to stay a small write on the hot path.
MAX_STRING = 200
MAX_ITEMS = 20

#: Resident memory (as a share of the machine's total) above which a death
#: mid-call is reported as *likely* an OOM kill rather than merely a death.
#: Not a threshold anything is refused on -- only how confidently it is worded.
OOM_LIKELY_SHARE = 0.5

_SELF = psutil.Process()
#: This process's start time, read once. Paired with the pid in every breadcrumb
#: so a reused pid cannot make a dead server look like a running one.
_SELF_STARTED = _SELF.create_time()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _enabled_by_env():
    return os.environ.get("LOGLEAD_MCP_CRASH_LOG", "1").lower() not in ("0", "false", "no")


def memory_snapshot():
    """What the machine looked like at this moment, in GB.

    Recorded at the *start* of a call, which makes it a floor rather than a
    peak: nothing samples during the call, so a breadcrumb says what was already
    resident and how much room was left, not how much the call went on to ask
    for. That floor is still what distinguishes "died holding 8.5 GB of 15.6"
    from "died holding 0.3".
    """
    try:
        virtual = psutil.virtual_memory()
        return {
            "rss_gb": round(_SELF.memory_info().rss / 1e9, 2),
            "available_gb": round(virtual.available / 1e9, 2),
            "total_gb": round(virtual.total / 1e9, 2),
        }
    except psutil.Error:  # a memory reading is never worth failing a call over
        return {}


def _compact(value):
    """A JSON-safe, bounded copy of a recorded argument."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= MAX_STRING else value[:MAX_STRING] + "..."
    if isinstance(value, dict):
        kept = {str(key): _compact(item) for key, item in list(value.items())[:MAX_ITEMS]}
        if len(value) > MAX_ITEMS:
            kept["..."] = f"{len(value) - MAX_ITEMS} more"
        return kept
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        kept = [_compact(item) for item in items[:MAX_ITEMS]]
        if len(items) > MAX_ITEMS:
            kept.append(f"... {len(items) - MAX_ITEMS} more")
        return kept
    return _compact(repr(value))


def _alive(pid, started_at):
    """Whether that process is still the one that wrote the breadcrumb.

    Both halves matter: a pid alone is reused, so a breadcrumb from a machine
    that has since rebooted would otherwise look like a running server forever.
    """
    if not pid:
        return False
    try:
        process = psutil.Process(int(pid))
        if started_at is None:
            return True
        return abs(process.create_time() - float(started_at)) < 1.0
    except (psutil.Error, TypeError, ValueError):
        return False


def render_call(name, args, keys=None):
    """Render a call the way a caller would write it, e.g. ``tool(a=1, b="x")``.

    Used for both halves of a report: the call that died, and the one that
    brings its session back. A note is read at the moment it is needed, so it
    hands over something runnable rather than describing it.
    """
    keys = list(args) if keys is None else [key for key in keys if key in args]
    rendered = ", ".join(f"{key}={json.dumps(args[key])}" for key in keys)
    return f"{name}({rendered})"


# --------------------------------------------------------------------------- #
# What to do differently next time

def _folder_count(record):
    return (record.get("shape") or {}).get("n_folders")


def smaller_call_advice(record):
    """Which recorded argument made the call big, and what to call instead.

    The rules are the scaling shapes ``tests/mcp/PERFORMANCE.md`` measured, not
    guesses: the ``anomaly_*`` family refits four detectors **per target** and
    ``target_folder`` defaults to ``"ALL"``; ``content_format`` swings cost
    60-80x on identical data; the UMAP layout is tens of times the default
    scatter. Nothing here knows how much memory the call would need -- these are
    the knobs whose *shape* is known, phrased so a client can act on one.
    """
    tool = record.get("tool") or ""
    args = record.get("args") or {}
    folders = _folder_count(record)
    tips = []

    if tool.startswith("anomaly_") and args.get("target_folder") == "ALL":
        scope = f"{folders} log folders" if folders else "every log folder"
        tips.append(
            f'target_folder="ALL" scored {scope} in one call, and the anomaly_* tools '
            f"refit four detectors per target -- name one target log folder, or a "
            f'"Prefix*" subset, and repeat it.'
        )
    if args.get("comparison_folders") == "ALL" and folders and folders > 50:
        tips.append(
            f'comparison_folders="ALL" was {folders} log folders; a named subset or a '
            f'"Prefix*" wildcard compares against fewer.'
        )
    content_format = args.get("content_format")
    if content_format == "3grams":
        tips.append(
            'content_format="3grams" is the most expensive representation there is '
            '(~4x "Words", ~60x "Parse-Tip" on the same data) -- try "Parse-Tip" or '
            '"Words" first.'
        )
    elif content_format == "Words":
        tips.append(
            'content_format="Words" is ~15x "Parse-Tip" on the same data; a parsed '
            "representation answers most questions for a fraction of it."
        )
    if "umap" in (args.get("plots") or ()):
        tips.append(
            'plots=["umap"] builds a layout over every log folder at once; the default '
            '"scatter" answers the same screening question far more cheaply.'
        )
    if args.get("target_files") == "ALL":
        tips.append(
            'target_files="ALL" scored every file of the target log folder; name the '
            "files you care about."
        )
    if not tips:
        tips.append(
            "Narrow whatever was widest -- one target instead of \"ALL\", fewer "
            "comparison log folders, a cheaper content_format -- or split the log root "
            "into smaller pieces."
        )
    return tips


def _died_line(record):
    """The one sentence of fact: what was running, and how it stood for memory."""
    memory = record.get("memory") or {}
    call = render_call(record.get("tool", "a tool"), record.get("args") or {},
                       record.get("explicit"))
    when = record.get("started_at", "an earlier run")
    line = f"The server process died while running {call}, started {when}."
    if memory.get("rss_gb") is not None and memory.get("total_gb"):
        share = memory["rss_gb"] / memory["total_gb"]
        cause = ("an out-of-memory kill is the likely cause"
                 if share >= OOM_LIKELY_SHARE
                 else "the cause is not recorded, and need not have been memory")
        line += (f" It already held {memory['rss_gb']}GB of the machine's "
                 f"{memory['total_gb']}GB when the call began "
                 f"({memory.get('available_gb', '?')}GB free), so {cause}.")
    return line


def crash_notes(records, recovery=None):
    """The notes a client should see once, on the first result after a restart.

    Delivered as notes rather than an error because there is nothing to attach
    an error to: the call that died was never answered, and the connection that
    would have carried the failure closed with the process. This is the first
    moment the server can say anything at all.

    :param recovery: called with one record to render the ``open_log_root`` that
        brings its session back, or ``None`` to leave that out. A callable
        because only the server knows that tool's defaults.
    """
    notes = []
    for record in records[-3:]:  # a restart loop should not bury the result
        notes.append(_died_line(record) + " Nothing was returned for that call, and "
                     "a kill cannot be reported at the time, so this is the first "
                     "chance to say so.")
        for tip in smaller_call_advice(record):
            notes.append("Before repeating it: " + tip)
        call = recovery(record) if recovery is not None else None
        if call:
            notes.append(
                f"Its session is gone with the process, but the frame behind it is "
                f"cached: {call} re-attaches in seconds under the same session_id."
            )
    return notes


def history_note(records):
    """What this log root did to a previous server process. See :meth:`CrashLog.history`."""
    latest = records[-1]
    memory = latest.get("memory") or {}
    held = (f", holding {memory['rss_gb']}GB of {memory['total_gb']}GB"
            if memory.get("rss_gb") is not None and memory.get("total_gb") else "")
    call = render_call(latest.get("tool", "a tool"), latest.get("args") or {},
                       latest.get("explicit"))
    more = f" ({len(records)} deaths recorded on this log root.)" if len(records) > 1 else ""
    return (f"This log root has killed a server process before: {call} on "
            f"{latest.get('started_at', 'an earlier run')}{held}.{more} "
            + " ".join(smaller_call_advice(latest)))


def summarize(record):
    """The compact, structured form of a crash, for a result field."""
    return {
        "tool": record.get("tool"),
        "args": record.get("args"),
        "started_at": record.get("started_at"),
        "session_id": record.get("session_id"),
        "memory": record.get("memory"),
        "shape": record.get("shape"),
    }


# --------------------------------------------------------------------------- #

class CrashLog:
    """Breadcrumbs and the ledger they end up in, under ``<cache_dir>/crash``.

    Lives beside the parquet cache on purpose: the cache is what makes the
    recovery advice true (a re-open re-attaches in seconds), and a client
    pointing at a different cache directory is a different server as far as
    "what died here before" is concerned.
    """

    def __init__(self, cache_dir, enabled=None):
        self.cache_dir = Path(cache_dir)
        self.dir = self.cache_dir / "crash"
        self.ledger_path = self.dir / "ledger.json"
        self.inflight_path = self.dir / f"inflight-{os.getpid()}.json"
        self.enabled = _enabled_by_env() if enabled is None else bool(enabled)
        #: Crashes found by :meth:`sweep` and not yet handed to a client.
        self.pending = []
        self._swept = False
        #: session_id -> size, computed once each. See _shape.
        self._shapes = {}
        self._made_dir = False

    # -- the hot path ------------------------------------------------------ #

    def start_call(self, tool, args, explicit=None, session=None, root=None):
        """Write down what is about to run. Best-effort and deliberately small."""
        if not self.enabled:
            return
        if not self._swept:
            self.sweep()
        record = {
            "pid": os.getpid(),
            "pid_started": _SELF_STARTED,
            "started_at": _now(),
            "tool": tool,
            "args": {name: _compact(value) for name, value in (args or {}).items()},
            "explicit": list(explicit or []),
            "memory": memory_snapshot(),
        }
        if session is not None:
            open_args = session.open_args()
            record["session_id"] = session.session_id
            record["open_args"] = _compact(open_args)
            # A recovery call has to be runnable, and _compact cuts a long
            # mapping short -- replaying a truncated folder_names would rename
            # 20 of 5,000 log folders and say nothing. Name what was cut instead.
            record["truncated_args"] = [key for key, value in open_args.items()
                                        if isinstance(value, (dict, list, tuple))
                                        and len(value) > MAX_ITEMS]
            record["root"] = str(session.root)
            record["shape"] = self._shape(session)
        elif root:
            record["root"] = os.path.abspath(os.path.expanduser(str(root)))
        # Encoded before the file is opened, and written in one call: this runs
        # in front of every tool, including the ones that cost 3ms in total.
        blob = json.dumps(record)
        try:
            if not self._made_dir:
                self.dir.mkdir(parents=True, exist_ok=True)
                self._made_dir = True
            with open(self.inflight_path, "w") as handle:
                handle.write(blob)
        except OSError:
            self.enabled = False  # an unwritable cache dir is not worth retrying per call

    def finish_call(self):
        """The call returned (or raised, which the client was told about): forget it."""
        if not self.enabled:
            return
        try:
            self.inflight_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _shape(self, session):
        """How big the session is, computed once per session and kept.

        ``n_folders`` is an ``n_unique`` over the whole ``folder`` column, which
        is 0.27s on a 4.7M-row frame -- more than several of the tools this runs
        in front of cost in total, and it would be paid again on every call. It
        cannot change for a session (the frame gains columns, never rows or log
        folders; renaming keeps the count), so the first call for a session pays
        for it and the rest read it here.
        """
        shape = self._shapes.get(session.session_id)
        if shape is None:
            shape = {"n_rows": session.df.height,
                     "n_folders": session.df.select("folder").n_unique()}
            self._shapes[session.session_id] = shape
        return shape

    # -- what the last process left behind --------------------------------- #

    def sweep(self):
        """Fold every breadcrumb whose process is gone into the ledger.

        Idempotent: a swept breadcrumb is deleted, so calling this at startup
        *and* lazily on the first call reports each crash exactly once.
        """
        self._swept = True
        if not self.enabled or not self.dir.is_dir():
            return []
        found = []
        for path in sorted(self.dir.glob("inflight-*.json")):
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                record = None
            if record is None:  # a torn write: nothing to report, nothing to keep
                self._unlink(path)
                continue
            if record.get("pid") == os.getpid() or _alive(record.get("pid"),
                                                          record.get("pid_started")):
                continue
            record["detected_at"] = _now()
            found.append(record)
            self._unlink(path)
        if found:
            self._append(found)
            self.pending.extend(found)
        return found

    def take_pending(self):
        """The crashes not yet reported to a client, cleared as they are handed over."""
        records, self.pending = self.pending, []
        return records

    def history(self, root):
        """Every recorded crash on this log root, oldest first."""
        root = str(root)
        return [record for record in self.read_ledger() if record.get("root") == root]

    # -- the ledger -------------------------------------------------------- #

    def read_ledger(self):
        try:
            return json.loads(self.ledger_path.read_text()).get("crashes", [])
        except (OSError, ValueError, AttributeError):
            return []

    def _append(self, records):
        """Add to the ledger, keeping the newest :data:`MAX_LEDGER`.

        Read-modify-write with no lock: two servers crashing at the same moment
        can lose one record, which is a better trade than a lock file that
        outlives a killed process.
        """
        crashes = (self.read_ledger() + list(records))[-MAX_LEDGER:]
        temporary = self.ledger_path.with_suffix(".json.tmp")
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(temporary, "w") as handle:
                json.dump({"crashes": crashes}, handle, indent=1)
            os.replace(temporary, self.ledger_path)
        except OSError:
            pass

    @staticmethod
    def _unlink(path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
