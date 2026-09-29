"""The one entry point for LogLead's tests: named suites of steps, run in isolation, one summary.

    uv run tests/run.py --list                  # suites, their steps, and what each needs
    uv run tests/run.py smoke                   # bundled samples only, no downloads (minutes)
    uv run tests/run.py mid mcp                 # several suites; a step shared by two runs once
    uv run tests/run.py full                    # everything except super, mcp-perf and llm
    uv run tests/run.py mid --only mid:loaders  # a single step (its prerequisites are not re-run)
    uv run tests/run.py smoke --polars 1.38.1   # against another Polars (the oldest supported one)
    uv run tests/run.py mid --capture-baselines # record loader fingerprints as tests/baselines/*.json
    uv run tests/run.py super --dry-run         # show what would run and what would be skipped
    uv run tests/run.py super --datasets thunderbird   # limit download/load steps to named datasets

Every step is a separate process, so a crash or an OOM kill in one does not stop the others; a
step whose prerequisite step failed is skipped. A step fails on a non-zero exit code, or when its
output contains MISMATCH!, a line starting with FAIL or a Python traceback. Steps whose
requirements are missing (data, .env, an extra, a sibling repo, an API key) are SKIPPED with the
reason instead of failing. Each step's output goes to tests/result/runs/<timestamp>/<step>.log
next to a summary.json; the exit code is non-zero if any step failed.

Peak memory in the summary is the process tree's heap (RssAnon), which is what gets a process
OOM-killed; memory-mapped input files are not counted.
"""

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

from memwatch import cap_available, run_measured

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS)
DEMO = os.path.join(ROOT, "demo")
MIN_POLARS = "1.38.1"
SUPER_CAP = "6G"

_FAIL_PATTERNS = [re.compile(p, re.MULTILINE) for p in
                  (r"MISMATCH!", r"^\s*FAIL\b", r"^Traceback \(most recent call last\)")]


@dataclass
class Step:
    name: str
    cmd: list                       # arguments after "uv run"
    cwd: str = ROOT
    needs: list = field(default_factory=list)
    after: list = field(default_factory=list)
    timeout: int = 4 * 3600
    extras: list = field(default_factory=list)
    cap: str = None                 # hard memory cap, e.g. "6G"
    uv: bool = True                 # False: cmd is a full command line, not run through uv
    fail_patterns: bool = True
    doc: str = ""


STEPS = {}
SUITES = {}


def step(**kwargs):
    s = Step(**kwargs)
    STEPS[s.name] = s
    return s.name


def _dataset_pipeline(prefix, config, doc, cap=None, detection=True):
    """download -> loaders -> enhancers -> anomaly_detectors (+ format detection) for one config."""
    cfg = os.path.join(TESTS, config)
    names = [
        step(name=f"{prefix}:download", cmd=["downloader/download_data.py", "--config", cfg],
             needs=["disk:20"], doc=f"download the datasets of {config} that are missing"),
        step(name=f"{prefix}:loaders", cmd=["loaders.py", "--config", cfg], cwd=TESTS,
             after=[f"{prefix}:download"], cap=cap,
             doc=f"load {doc}, check row counts and baseline fingerprints"),
        step(name=f"{prefix}:enhancers", cmd=["enhancers.py", "--config", cfg, "--only-config"], cwd=TESTS,
             after=[f"{prefix}:loaders"], doc="enhance every *_lo.parquet in <root_folder>/test_data"),
        step(name=f"{prefix}:detectors", cmd=["anomaly_detectors.py", "--config", cfg, "--only-config"],
             cwd=TESTS,
             after=[f"{prefix}:enhancers"], doc="anomaly detectors on the enhanced files"),
    ]
    if detection:
        names.append(step(name=f"{prefix}:detection", cmd=["log_file_detection.py", "--config", cfg],
                          cwd=TESTS, after=[f"{prefix}:download"],
                          doc="AutoLoader picks the loader loaders.py would build"))
    return names


# --- smoke: bundled samples only --------------------------------------------------------------
SUITES["smoke"] = [
    step(name="demo:hdfs-samples", cmd=["demo/HDFS_samples.py"], doc="HDFS 2% sample, end to end"),
    step(name="demo:tb-samples", cmd=["demo/TB_samples.py"], doc="Thunderbird 0.125% sample, end to end"),
    step(name="demo:drain-persistence", cmd=["demo/Drain_Persistence.py"],
         doc="Drain with template persistence on the TB sample"),
    step(name="streaming:quick", cmd=["tests/streaming.py", "--quick"],
         doc="streaming/lazy paths equal the eager ones, on samples and synthetic logs"),
    step(name="memory:quick", cmd=["tests/memory.py", "--quick"],
         doc="streaming heap stays flat as synthetic input grows (small sizes)"),
]

# --- demos --------------------------------------------------------------------------------------
SUITES["demos"] = SUITES["smoke"][:3] + [
    step(name="demo:autoloader", cmd=["demo/AutoLoader_samples.py"], doc="AutoLoader format detection demo"),
    step(name="demo:openstack", cmd=["demo/OpenStack_samples.py"], needs=["env:LOG_DATA_PATH"],
         doc="OpenStack from LOG_DATA_PATH"),
    step(name="demo:rawloader-nolabels", cmd=["demo/RawLoader_NoLabels.py"], needs=["env:LOG_DATA_PATH"],
         doc="RawLoader on unlabeled data"),
    step(name="demo:rawloader-hadoop", cmd=["demo/RawLoader_TimeStamps_Hadoop.py"],
         needs=["env:LOG_DATA_PATH"], doc="RawLoader timestamps, Hadoop"),
    step(name="demo:rawloader-mufano", cmd=["demo/RawLoader_TimeStamps_Mufano.py"],
         needs=["env:LOG_DATA_PATH", "data:mufano"], doc="RawLoader timestamps, Mufano"),
    step(name="demo:unsupervised", cmd=["demo/unsupervised_models.py"], needs=["env:LOG_DATA_PATH"],
         doc="unsupervised detectors"),
    step(name="demo:mcp", cmd=["demo/mcp_demo.py"], extras=["mcp"], needs=["extra:mcp"],
         doc="MCP tools driven in process"),
]

# --- dataset pipelines --------------------------------------------------------------------------
SUITES["mid"] = _dataset_pipeline("mid", "datasets_mid_labels.yml",
                                  "BGL, Hadoop, HDFS, Nezha, ADFA, AWSCTD, OpenStack")
SUITES["super"] = _dataset_pipeline("super", "datasets_super_comp_labels.yml",
                                    "Thunderbird, Spirit, Liberty (streamed, under a hard memory cap)",
                                    cap=SUPER_CAP) + [
    step(name="memory:super", cmd=["tests/memory.py", "--real", "thunderbird", "--cap", SUPER_CAP],
         needs=["cap", "data:thunderbird"], timeout=12 * 3600,
         doc=f"full Thunderbird: stream-load, enhance and score under MemoryMax={SUPER_CAP}"),
]
SUITES["formats"] = []
for _cfg in ("access_log", "auto", "csv_tsv", "fmt", "json", "lo2", "syslog"):
    SUITES["formats"] += _dataset_pipeline(f"formats-{_cfg}", f"datasets_{_cfg}.yml", f"datasets_{_cfg}.yml")

# --- streaming equivalence and memory -----------------------------------------------------------
SUITES["equivalence"] = [
    step(name="streaming:full", cmd=["tests/streaming.py"],
         doc="streaming/lazy == eager on samples, synthetic logs and slices of the real datasets"),
]
SUITES["memory"] = [
    step(name="memory:scaling", cmd=["tests/memory.py"],
         doc="heap of each streaming stage stays flat from N to 3N synthetic lines"),
]

# --- MCP ----------------------------------------------------------------------------------------
SUITES["mcp"] = [
    step(name="mcp:server", cmd=["tests/mcp/server.py"], extras=["mcp"], needs=["extra:mcp"],
         fail_patterns=False,  # logs tracebacks on purpose when testing error reporting; exits non-zero on failure
         doc="every MCP tool against the hadoop/hdfs log roots"),
]
SUITES["mcp-perf"] = [
    step(name="mcp:benchmark", cmd=["tests/mcp/benchmark.py"], extras=["mcp"], needs=["extra:mcp"],
         timeout=48 * 3600, fail_patterns=False, doc="time/memory grid, rewrites tests/mcp/PERF*.md"),
]

# --- downstream consumers -----------------------------------------------------------------------
SUITES["consumers"] = [
    step(name="consumers:logdelta", cmd=["tests/consumers.py", "logdelta"], needs=["repo:LogDelta"],
         doc="LogDelta demo configs against this checkout"),
    step(name="consumers:visualloganalyzer", cmd=["tests/consumers.py", "visualloganalyzer"],
         needs=["repo:VisualLogAnalyzer"], doc="VisualLogAnalyzer unit tests against this checkout"),
]

# --- opt-in -------------------------------------------------------------------------------------
SUITES["llm"] = [
    step(name="demo:llm-parser", cmd=["demo/llm_parser.py"], needs=["key:OPENROUTER_API_KEY|API_KEY"],
         doc="LLM parser demo (calls a hosted model, costs money)"),
]

SUITES["full"] = [s for suite in ("smoke", "demos", "equivalence", "memory", "mid", "formats",
                                  "mcp", "consumers") for s in SUITES[suite]]
SUITES["all"] = SUITES["full"] + SUITES["super"]

SUITE_DOCS = {
    "smoke": "bundled samples + quick streaming/memory checks; no downloads",
    "demos": "every demo that runs unattended",
    "mid": "mid-sized labeled datasets, full pipeline (~30 min + downloads)",
    "super": f"Thunderbird/Spirit/Liberty, full pipeline under a {SUPER_CAP} memory cap (hours)",
    "formats": "format-spec loaders: access log, auto, csv/tsv, logfmt, json, lo2, syslog",
    "equivalence": "streaming vs eager equality on real dataset slices",
    "memory": "memory scaling of each streaming stage",
    "mcp": "MCP server tools",
    "mcp-perf": "MCP time/memory benchmark grid (very long)",
    "consumers": "LogDelta and VisualLogAnalyzer against this checkout",
    "llm": "LLM parser demo; opt-in, needs an API key",
    "full": "smoke, demos, equivalence, memory, mid, formats, mcp, consumers",
    "all": "full + super",
}


# --- requirements -------------------------------------------------------------------------------
def _dotenv():
    values = {}
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _root_folder():
    import yaml
    with open(os.path.join(TESTS, "datasets_mid_labels.yml")) as handle:
        return os.path.expanduser(yaml.safe_load(handle)["root_folder"])


_checked = {}


def unmet(requirement):
    """None when the requirement holds, otherwise why it does not."""
    if requirement in _checked:
        return _checked[requirement]
    kind, _, value = requirement.partition(":")
    reason = None
    if kind == "env":
        path = os.environ.get(value) or _dotenv().get(value)
        if not path:
            reason = f"{value} not set (in .env or the environment)"
        elif not os.path.exists(os.path.expanduser(path)):
            reason = f"{value}={path} does not exist"
    elif kind == "data":
        path = os.path.join(_root_folder(), value)
        if not os.path.exists(path):
            reason = f"{path} missing (download it first)"
    elif kind == "disk":
        free = shutil.disk_usage(os.path.expanduser("~")).free / 1024 ** 3
        if free < float(value):
            reason = f"only {free:.0f} GB free, want {value} GB"
    elif kind == "extra":
        if sys.version_info < (3, 10):
            reason = f"the {value} extra needs Python >= 3.10"
        else:
            probe = subprocess.run(["uv", "run", "--extra", value, "python", "-c", f"import {value}"],
                                   cwd=ROOT, capture_output=True)
            if probe.returncode:
                reason = f"cannot install/import the '{value}' extra"
    elif kind == "repo":
        if not os.path.isdir(os.path.join(os.path.dirname(ROOT), value)):
            reason = f"sibling repo ../{value} not found"
    elif kind == "key":
        names = value.split("|")
        if not any(os.environ.get(n) or _dotenv().get(n) for n in names):
            reason = f"none of {names} set"
    elif kind == "cap":
        if not cap_available():
            reason = "systemd-run --user scope with MemoryMax unavailable"
    _checked[requirement] = reason
    return reason


# --- running ------------------------------------------------------------------------------------
def command(s, polars_version):
    if not s.uv:
        return list(s.cmd)
    if not shutil.which("uv"):
        # Without uv, run in this interpreter's environment; extras must already be installed.
        return [sys.executable] + list(s.cmd)
    cmd = ["uv", "run"]
    for extra in s.extras:
        cmd += ["--extra", extra]
    if polars_version:
        cmd += ["--with", f"polars=={polars_version}"]
    return cmd + list(s.cmd)


def failure_reason(result, s):
    if result["timed_out"]:
        return "TIMEOUT"
    if result["oom"]:
        return "OOM"
    if result["returncode"]:
        return f"exit {result['returncode']}"
    if s.fail_patterns:
        for pattern in _FAIL_PATTERNS:
            match = pattern.search(result["output"])
            if match:
                line = result["output"][match.start():].split("\n", 1)[0].strip()
                return f"output: {line[:80]}"
    return None


def resolve(names):
    selected = []
    for name in names:
        if name in SUITES:
            selected += SUITES[name]
        elif name in STEPS:
            selected.append(name)
        else:
            sys.exit(f"Unknown suite or step '{name}'. See --list.")
    return list(dict.fromkeys(selected))


def print_list():
    print("Suites:")
    for name, steps in SUITES.items():
        print(f"  {name:12} {SUITE_DOCS.get(name, '')}  [{len(steps)} steps]")
    print("\nSteps:")
    for name, s in STEPS.items():
        needs = ", ".join(s.needs + [f"after {a}" for a in s.after])
        cap = f" cap={s.cap}" if s.cap else ""
        print(f"  {name:32} {s.doc}{cap}" + (f"  (needs {needs})" if needs else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suites", nargs="*", help="suites or step names; see --list")
    parser.add_argument("--list", action="store_true", help="list suites and steps")
    parser.add_argument("--only", nargs="+", default=None, help="run only these steps of the selection")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without running it")
    parser.add_argument("--polars", nargs="?", const=MIN_POLARS, default=None,
                        help=f"run against this Polars version (bare flag: {MIN_POLARS}, the oldest supported)")
    parser.add_argument("--capture-baselines", action="store_true",
                        help="loader steps record tests/baselines/*.json instead of checking them")
    parser.add_argument("--datasets", nargs="+", metavar="NAME", default=None,
                        help="limit download and loader steps to these datasets of their config")
    parser.add_argument("--no-cap", action="store_true", help="ignore the memory caps of capped steps")
    parser.add_argument("--echo", action="store_true", help="also print step output live")
    parser.add_argument("--out", default=None, help="log folder (default tests/result/runs/<timestamp>)")
    args = parser.parse_args()

    if args.list or not args.suites:
        print_list()
        return 0

    selected = resolve(args.suites)
    if args.only:
        selected = [s for s in selected if s in args.only]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or os.path.join(TESTS, "result", "runs", stamp)
    if not args.dry_run:
        os.makedirs(out, exist_ok=True)
    print(f"LogLead tests: {len(selected)} step(s)  python {platform.python_version()}"
          + (f"  polars=={args.polars}" if args.polars else "") + (f"  logs: {out}" if not args.dry_run else ""))

    results = {}
    for name in selected:
        s = STEPS[name]
        cmd = command(s, args.polars)
        if args.capture_baselines and name.endswith(":loaders"):
            cmd += ["--baseline", "capture"]
        if args.datasets and name.endswith(":loaders"):
            cmd += ["--only", *args.datasets]
        if args.datasets and name.endswith(":download"):
            cmd += ["--datasets", *args.datasets]
        cap = None if args.no_cap else s.cap
        if cap and unmet("cap"):
            cap = None
            print(f"  note: {name} runs without its {s.cap} cap ({unmet('cap')})")
        blocked = [a for a in s.after if a in results and results[a]["status"] not in ("PASS", "PLAN")]
        reasons = [r for r in (unmet(n) for n in s.needs) if r]
        if blocked:
            status, reason = "SKIP", f"{', '.join(blocked)} did not pass"
        elif reasons:
            status, reason = "SKIP", "; ".join(reasons)
        elif args.dry_run:
            status, reason = "PLAN", " ".join(cmd) + (f"  [cap {cap}]" if cap else "")
        else:
            print(f"-> {name:34} ", end="", flush=True)
            result = run_measured(cmd, cwd=s.cwd, timeout=s.timeout, cap=cap,
                                  log_path=os.path.join(out, name.replace(":", "_") + ".log"), echo=args.echo)
            reason = failure_reason(result, s)
            status = "FAIL" if reason else "PASS"
            results[name] = {"status": status, "reason": reason, "seconds": round(result["seconds"], 1),
                             "peak_heap_mb": result["peak_anon_mb"], "peak_rss_mb": result["peak_rss_mb"],
                             "cmd": cmd, "cap": cap}
            print(f"{status}  {result['seconds']:7.1f}s  heap {result['peak_anon_mb']:>6} MB"
                  + (f"  {reason}" if reason else ""))
            continue
        results[name] = {"status": status, "reason": reason}
        print(f"   {name:34} {status}  {reason}")

    if args.dry_run:
        return 0
    with open(os.path.join(out, "summary.json"), "w") as handle:
        json.dump({"started": stamp, "suites": args.suites, "polars": args.polars, "steps": results},
                  handle, indent=1)
    counts = {k: sum(1 for r in results.values() if r["status"] == k) for k in ("PASS", "FAIL", "SKIP")}
    print("\n" + "=" * 100)
    for name, r in results.items():
        timing = f"{r['seconds']:8.1f}s  heap {r['peak_heap_mb']:>6} MB" if "seconds" in r else " " * 23
        print(f"{r['status']:5} {name:34} {timing}  {r['reason'] or ''}")
    print("=" * 100)
    print(f"{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped.  Logs: {out}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
