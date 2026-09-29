"""Run a command and record the peak memory of its whole process tree.

Two numbers are sampled from /proc/<pid>/status of the process and all its descendants:
RssAnon (heap and other anonymous memory, the part that gets a process OOM-killed) and the
total RSS, which also counts memory-mapped file pages. Polars memory-maps input files, so total
RSS grows with the file even when the heap stays flat; the heap figure is the one to budget.

cap runs the command inside a systemd user scope with MemoryMax and no swap, so exceeding
it gets the command killed instead of the machine swapping. Linux only.
"""

import os
import shutil
import subprocess
import threading
import time

OOM_EXIT_CODES = {-9, 137}


def _status_kb(pid, key):
    try:
        with open(f"/proc/{pid}/status") as handle:
            for line in handle:
                if line.startswith(key + ":"):
                    return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return 0


def _descendants(pid):
    """The process and all its children, so memory used by subprocesses is counted too."""
    found, stack = [], [pid]
    while stack:
        current = stack.pop()
        found.append(current)
        try:
            for task in os.listdir(f"/proc/{current}/task"):
                with open(f"/proc/{current}/task/{task}/children") as handle:
                    stack.extend(int(child) for child in handle.read().split())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


class TreeMonitor:
    """Track a process tree's peak memory from a background thread while the command runs."""

    def __init__(self, pid, interval=0.1):
        self.pid = pid
        self.interval = interval
        self.peak_anon_kb = 0
        self.peak_rss_kb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            pids = _descendants(self.pid)
            self.peak_anon_kb = max(self.peak_anon_kb, sum(_status_kb(p, "RssAnon") for p in pids))
            self.peak_rss_kb = max(self.peak_rss_kb, sum(_status_kb(p, "VmRSS") for p in pids))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def cap_available():
    """Check whether memory caps work here; they need systemd user scopes, which not every machine has."""
    if not shutil.which("systemd-run"):
        return False
    probe = subprocess.run(["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=1G", "true"],
                           capture_output=True)
    return probe.returncode == 0


def capped(cmd, cap):
    """Run cmd under a hard memory limit such as "6G", so going over kills it instead of swapping."""
    return ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={cap}",
            "-p", "MemorySwapMax=0", *cmd]


def run_measured(cmd, cwd=None, env=None, timeout=None, cap=None, log_path=None, echo=False):
    """Run a command and report its peak memory, exit code, time and output.

    Output is also written to log_path as it arrives, so a hung or killed run still leaves a log.
    """
    if cap:
        cmd = capped(cmd, cap)
    log = open(log_path, "w") if log_path else None
    started = time.time()
    process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, errors="replace",
                               start_new_session=True)
    lines, timed_out = [], False
    timer = None
    if timeout:
        def _kill():
            nonlocal timed_out
            timed_out = True
            try:
                os.killpg(process.pid, 9)
            except ProcessLookupError:
                pass
        timer = threading.Timer(timeout, _kill)
        timer.start()
    with TreeMonitor(process.pid) as monitor:
        for line in process.stdout:
            lines.append(line)
            if log:
                log.write(line)
                log.flush()
            if echo:
                print(line, end="", flush=True)
        process.wait()
    if timer:
        timer.cancel()
    if log:
        log.close()
    return {
        "returncode": process.returncode,
        "seconds": time.time() - started,
        "peak_anon_mb": monitor.peak_anon_kb // 1024,
        "peak_rss_mb": monitor.peak_rss_kb // 1024,
        "output": "".join(lines),
        "oom": process.returncode in OOM_EXIT_CODES and not timed_out,
        "timed_out": timed_out,
    }
