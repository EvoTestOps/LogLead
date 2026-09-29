"""Run the sibling projects that use LogLead as a library against this checkout.

    uv run tests/consumers.py logdelta
    uv run tests/consumers.py visualloganalyzer

Neither repo is modified: each runs in a throwaway uv environment that installs the sibling and
this LogLead checkout as editable packages, with outputs written to a temporary folder.

* logdelta: three of its demo/full configs (file-name anomaly detection, run distance, file
  content anomaly detection) through its config-runner CLI, on its bundled Hadoop demo data.
  Passes when each exits 0 and writes at least one non-empty table.
* visualloganalyzer: its non-browser pytest files. Its requirements.txt pins LogLead to a git
  branch and pins tipping; both pins are dropped so this checkout's own requirements apply.
"""

import glob
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIBLINGS = os.path.dirname(ROOT)
PYTHON = "3.12"

LOGDELTA_CONFIGS = ["demo_anodetect_1", "demo_dist_1", "demo_anodetect_3"]
VLA_TESTS = ["tests/test_run_analysis_functions.py", "tests/test_log_analysis_pipeline.py"]


def logdelta():
    repo = os.path.join(SIBLINGS, "LogDelta")
    demo = os.path.join(repo, "demo", "full")
    failures = []
    with tempfile.TemporaryDirectory(prefix="loglead_consumer_") as work:
        os.symlink(os.path.join(demo, "Hadoop"), os.path.join(work, "Hadoop"))
        for name in LOGDELTA_CONFIGS:
            with open(os.path.join(demo, f"{name}.yml")) as handle:
                text = handle.read()
            text = "\n".join(
                f"output_folder: Out/{name}" if line.startswith("output_folder:")
                else 'table_output: "csv"' if line.startswith("table_output:") else line
                for line in text.splitlines())
            config = os.path.join(work, f"{name}.yml")
            with open(config, "w") as handle:
                handle.write(text)
            cmd = ["uv", "run", "--no-project", "--python", PYTHON, "--with-editable", repo,
                   "--with-editable", ROOT, "config-runner", "-c", config]
            print(f"LogDelta {name}: {' '.join(cmd)}", flush=True)
            # LogDelta resolves relative paths against $PWD rather than the working directory.
            result = subprocess.run(cmd, cwd=work, capture_output=True, text=True,
                                    env=dict(os.environ, PWD=work))
            tables = [f for f in glob.glob(os.path.join(work, "Out", name, "*")) if os.path.getsize(f)]
            if result.returncode or not tables:
                print(result.stdout[-3000:], result.stderr[-3000:])
                print(f"FAIL LogDelta {name}: exit {result.returncode}, {len(tables)} table(s) written")
                failures.append(name)
            else:
                print(f"ok   LogDelta {name}: {len(tables)} table(s) written")
    return failures


def visualloganalyzer():
    repo = os.path.join(SIBLINGS, "VisualLogAnalyzer")
    with tempfile.TemporaryDirectory(prefix="loglead_consumer_") as work:
        requirements = os.path.join(work, "requirements.txt")
        with open(os.path.join(repo, "requirements.txt")) as source, open(requirements, "w") as target:
            for line in source:
                if line.lower().startswith(("loglead ", "loglead@", "tipping=")):
                    continue
                target.write(line)
        cmd = ["uv", "run", "--no-project", "--python", PYTHON, "--with-requirements", requirements,
               "--with-editable", ROOT, "python", "-m", "pytest", "-q", "-p", "no:cacheprovider", *VLA_TESTS]
        print(f"VisualLogAnalyzer: {' '.join(cmd)}", flush=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, env=env)
        tail = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        if result.returncode:
            print(result.stdout[-5000:], result.stderr[-3000:])
            print(f"FAIL VisualLogAnalyzer: {tail}")
            return ["pytest"]
        print(f"ok   VisualLogAnalyzer: {tail}")
        return []


if __name__ == "__main__":
    targets = sys.argv[1:] or ["logdelta", "visualloganalyzer"]
    failed = []
    for target in targets:
        failed += {"logdelta": logdelta, "visualloganalyzer": visualloganalyzer}[target]()
    sys.exit(1 if failed else 0)
