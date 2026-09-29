"""Memory tests for the streaming path: heap must not grow with the input.

    uv run tests/memory.py --quick                         # 1M vs 3M synthetic lines (smoke suite)
    uv run tests/memory.py                                 # 2M vs 6M synthetic lines
    uv run tests/memory.py --real thunderbird --cap 6G     # the full dataset under a hard memory cap

Each workload runs in its own process (--workload), and the peak heap (RssAnon) of that process
tree is sampled. Memory-mapped input pages are not counted: they are page cache the kernel can drop.

Scaling mode runs every workload on synthetic Thunderbird logs of N and 3N lines and fails a
streaming workload when its heap grows by more than SLOPE times the added input size plus
SLACK_MB. The eager loader is measured the same way and reported for comparison, not asserted:
it grows by roughly four times the input.

Real mode runs the streaming pipeline on the whole dataset, each stage under --cap (a systemd
scope with MemoryMax and no swap), and fails if any stage is killed or errors. --show-eager also
runs the eager loader under the same cap, to show that it is what gets killed.
"""

import argparse
import json
import os
import sys
import time

TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TESTS)

from memwatch import cap_available, run_measured  # noqa: E402

SLOPE = 0.25
SLACK_MB = 64
BATCH = 200_000
REAL_FILES = {"thunderbird": "thunderbird/tbird2.log", "spirit": "spirit/spirit2.log",
              "liberty": "liberty/liberty2.log"}


# --- workloads, each run in a child process -----------------------------------------------------
def workload(name, source, out, batch_size, rows, split_component, chunk_bytes=None):
    import polars as pl

    from loglead import AnomalyDetector
    from loglead.enhancers import EventLogEnhancer
    from loglead.loaders import ThuSpiLibLoader
    from loglead.streaming import iter_batches, sample, write_batches

    started = time.time()
    if name == "load-eager":
        df = ThuSpiLibLoader(source, split_component=split_component).execute()
        result = {"rows": df.height}
    elif name == "load-stream":
        ThuSpiLibLoader(source, split_component=split_component).sink(out, chunk_bytes=chunk_bytes)
        result = {"rows": pl.scan_parquet(out).select(pl.len()).collect().item()}
    elif name == "enhance-stream":
        if rows:
            head = out + ".head.parquet"
            write_batches(_first_rows(iter_batches(source, batch_size), rows), head)
            source = head
        enhancer = EventLogEnhancer.from_parquet(source)
        enhancer.length()
        enhancer.mask()
        enhancer.words()
        enhancer.parse_drain()
        enhancer.sink(out, batch_size=batch_size)
        result = {"rows": pl.scan_parquet(out).select(pl.len()).collect().item()}
    elif name == "score-stream":
        # Lines too short to have a message have null e_words, which no detector accepts; the
        # dataset tests drop them before detection too.
        train = sample(source, n=100_000, seed=0).filter(pl.col("e_words").is_not_null())
        sad = AnomalyDetector(item_list_col="e_words", print_scores=False, auc_roc=True)
        sad.test_train_split(train.select("e_words", "anomaly"), test_frac=0.2)
        sad.train_LR()
        sad.predict_to_parquet(source, out, batch_size=batch_size, where=pl.col("e_words").is_not_null())
        scored = pl.scan_parquet(out)
        result = {"rows": scored.select(pl.len()).collect().item(),
                  "flagged": scored.select(pl.col("pred_ano").sum()).collect().item()}
    else:
        raise ValueError(name)
    result["seconds"] = round(time.time() - started, 1)
    print("RESULT " + json.dumps(result))


def _first_rows(batches, rows):
    for batch in batches:
        if rows <= 0:
            return
        yield batch.head(rows)
        rows -= batch.height


def _run(name, source, out, cap=None, batch_size=BATCH, rows=0, split_component=True, timeout=None,
         chunk_bytes=None):
    cmd = [sys.executable, os.path.abspath(__file__), "--workload", name, "--input", source, "--output", out,
           "--batch-size", str(batch_size), "--rows", str(rows)]
    if chunk_bytes:
        cmd += ["--chunk-bytes", str(chunk_bytes)]
    if not split_component:
        cmd.append("--no-split-component")
    measured = run_measured(cmd, cap=cap, timeout=timeout)
    lines = [l for l in measured["output"].splitlines() if l.startswith("RESULT ")]
    measured["result"] = json.loads(lines[-1][7:]) if lines else {}
    if measured["returncode"] and not measured["oom"]:
        print(measured["output"][-3000:])
    return measured


# --- scaling ------------------------------------------------------------------------------------
def scaling(workdir, lines):
    import synthetic

    failures = []
    inputs = {}
    for n in (lines, 3 * lines):
        path = os.path.join(workdir, f"tb_{n}.log")
        if not os.path.exists(path):
            synthetic.write("thunderbird", n, path, seed=11)
        inputs[n] = path
    added_mb = (os.path.getsize(inputs[3 * lines]) - os.path.getsize(inputs[lines])) / 1024 ** 2
    # Both sizes must span several chunks/batches, or the smaller run just holds less of one.
    chunk_bytes = min(64 << 20, os.path.getsize(inputs[lines]) // 4)
    batch_size = min(BATCH, lines // 4)
    allowed = SLOPE * added_mb + SLACK_MB
    print(f"Synthetic Thunderbird: {lines:,} lines ({os.path.getsize(inputs[lines]) / 1024 ** 2:.0f} MB) vs "
          f"{3 * lines:,} lines, loaded in {chunk_bytes / 1024 ** 2:.0f} MB chunks, processed in batches of "
          f"{batch_size:,} rows; streaming heap may grow by at most {allowed:.0f} MB "
          f"({SLOPE} x {added_mb:.0f} MB added input + {SLACK_MB} MB).\n")
    print(f"{'workload':16} {'heap N':>9} {'heap 3N':>9} {'growth':>9} {'per input MB':>13} "
          f"{'time 3N':>8}  verdict")

    stages = [("load-eager", lambda n: inputs[n], "loaded_eager_{n}.parquet"),
              ("load-stream", lambda n: inputs[n], "loaded_{n}.parquet"),
              ("enhance-stream", lambda n: os.path.join(workdir, f"loaded_{n}.parquet"), "enhanced_{n}.parquet"),
              ("score-stream", lambda n: os.path.join(workdir, f"enhanced_{n}.parquet"), "scored_{n}.parquet")]
    for name, source, out in stages:
        peaks, seconds, broken = {}, {}, False
        for n in (lines, 3 * lines):
            measured = _run(name, source(n), os.path.join(workdir, out.format(n=n)),
                            batch_size=batch_size, chunk_bytes=chunk_bytes)
            if measured["returncode"]:
                broken = True
                print(f"{name:16} FAIL: exit {measured['returncode']}")
                break
            peaks[n], seconds[n] = measured["peak_anon_mb"], measured["seconds"]
        if broken:
            failures.append(name)
            continue
        growth = peaks[3 * lines] - peaks[lines]
        asserted = name != "load-eager"
        ok = growth <= allowed
        verdict = ("ok" if ok else "FAIL grows with input") if asserted else "(eager, reported only)"
        if asserted and not ok:
            failures.append(name)
        print(f"{name:16} {peaks[lines]:>7} MB {peaks[3 * lines]:>7} MB {growth:>+7} MB "
              f"{growth / added_mb:>13.2f} {seconds[3 * lines]:>7.0f}s  {verdict}")
    return failures


# --- real dataset under a cap -------------------------------------------------------------------
def real(dataset, cap, workdir, show_eager, drain_rows):
    import yaml

    with open(os.path.join(TESTS, "datasets_super_comp_labels.yml")) as handle:
        root = os.path.expanduser(yaml.safe_load(handle)["root_folder"])
    source = os.path.join(root, REAL_FILES[dataset])
    if not os.path.exists(source):
        print(f"FAIL {source} not found; download it first (uv run downloader/download_data.py "
              f"--config tests/datasets_super_comp_labels.yml --datasets {dataset})")
        return ["missing data"]
    if not cap_available():
        print("FAIL systemd-run --user scopes with MemoryMax are unavailable, cannot enforce the cap")
        return ["no cap"]
    split = dataset != "liberty"
    size_gb = os.path.getsize(source) / 1024 ** 3
    print(f"{dataset}: {source} ({size_gb:.1f} GB), every stage under MemoryMax={cap}, no swap.\n")
    loaded = os.path.join(workdir, f"{dataset}_loaded.parquet")
    enhanced = os.path.join(workdir, f"{dataset}_enhanced.parquet")
    scored = os.path.join(workdir, f"{dataset}_scored.parquet")
    stages = []
    if show_eager:
        stages.append(("load-eager", source, os.devnull, 0, False))
    stages += [("load-stream", source, loaded, 0, True),
               ("enhance-stream", loaded, enhanced, drain_rows, True),
               ("score-stream", enhanced, scored, 0, True)]
    failures = []
    print(f"{'stage':16} {'status':8} {'time':>8} {'heap':>9} {'rss':>9}  result")
    for name, src, out, rows, asserted in stages:
        measured = _run(name, src, out, cap=cap, rows=rows, split_component=split, timeout=24 * 3600)
        status = "OOM" if measured["oom"] else ("ok" if measured["returncode"] == 0 else f"exit {measured['returncode']}")
        if asserted and status != "ok":
            failures.append(name)
            status = "FAIL " + status
        note = "" if asserted else "  (eager, expected to be killed)"
        print(f"{name:16} {status:8} {measured['seconds']:>7.0f}s {measured['peak_anon_mb']:>6} MB "
              f"{measured['peak_rss_mb']:>6} MB  {measured['result']}{note}", flush=True)
        if asserted and status != "ok":
            break
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="1M vs 3M lines instead of 2M vs 6M")
    parser.add_argument("--lines", type=int, default=None, help="N for the scaling test")
    parser.add_argument("--real", choices=sorted(REAL_FILES), help="run the full dataset under --cap")
    parser.add_argument("--cap", default="6G", help="memory cap for --real (default 6G)")
    parser.add_argument("--show-eager", action="store_true", help="with --real, also run the eager loader")
    parser.add_argument("--drain-rows", type=int, default=0,
                        help="with --real, enhance only the first N rows (0: all)")
    parser.add_argument("--workdir", default=None,
                        help="where inputs/outputs go (default <root_folder>/test_data/memory)")
    parser.add_argument("--workload", help=argparse.SUPPRESS)
    parser.add_argument("--input", help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=BATCH, help=argparse.SUPPRESS)
    parser.add_argument("--rows", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--no-split-component", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--chunk-bytes", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.workload:
        workload(args.workload, args.input, args.output, args.batch_size, args.rows, not args.no_split_component,
                 args.chunk_bytes)
        return 0

    workdir = args.workdir or os.path.join(os.path.expanduser("~/Datasets"), "test_data", "memory")
    os.makedirs(workdir, exist_ok=True)
    if args.real:
        failures = real(args.real, args.cap, workdir, args.show_eager, args.drain_rows)
    else:
        failures = scaling(workdir, args.lines or (1_000_000 if args.quick else 2_000_000))
    print()
    if failures:
        print(f"FAIL memory test: {', '.join(failures)}")
        return 1
    print("Memory test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
