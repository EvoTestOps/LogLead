"""The streaming (lazy, batched) paths must produce exactly what the eager paths produce.

    uv run tests/streaming.py --quick     # samples + small synthetic logs (part of the smoke suite)
    uv run tests/streaming.py             # + larger synthetic logs and 300k-line slices of real datasets
    uv run tests/streaming.py --capture   # record missing eager baselines (tests/baselines/streaming_eager_*.json)

Like the rest of tests/ this is a plain script: every check prints ok or FAIL, a failing
section does not stop the others, and the exit code is non-zero if anything failed.

What is compared:

* Loaders: execute() vs scan().collect() vs sink() read back, frame-equal, including
  df_seq for sequence loaders, and the null/non-UTF-8 warnings each path logs.
* Enhancers: every expression enhancer on a DataFrame vs on a LazyFrame (collected, and sunk).
  Drain and minhash are stateful/row-batched, so the eager and batched runs each happen in a fresh
  process (Drain's template miner is a module-level singleton) and must give identical frames.
* Detectors: predict() vs predict_batches() / predict_to_parquet() with the same fitted
  model: identical binary predictions, and scores equal to float tolerance.
* loglead.streaming.sample: deterministic, close to the requested size, order-preserving.
* Eager outputs vs baselines recorded on the code before the streaming feature, so a change made
  for streaming cannot silently alter the eager behaviour that other projects depend on.
"""

import argparse
import contextlib
import logging
import os
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

import fingerprint as fp
import synthetic

TESTS = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(os.path.dirname(TESTS), "demo", "samples")
DATASETS = os.path.expanduser("~/Datasets")
BASELINE = os.path.join(TESTS, "baselines", "streaming_eager_{mode}.json")
SLICE_LINES = 300_000


class Check:
    def __init__(self):
        self.passed, self.failures = 0, []

    def __call__(self, name, ok, detail=""):
        if ok:
            self.passed += 1
            print(f"  ok   {name}")
        else:
            self.failures.append(name)
            print(f"  FAIL {name}" + (f"  [{detail}]" if detail else ""))
        return ok

    def equal(self, name, left, right, **kwargs):
        try:
            assert_frame_equal(left, right, **kwargs)
            return self(name, True)
        except AssertionError as error:
            return self(name, False, str(error).replace("\n", " ")[:400])

    def section(self, title, fn, *args):
        print(f"\n== {title}")
        try:
            fn(self, *args)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            self(f"{title} raised", False)


@contextlib.contextmanager
def captured_warnings(logger_name="loglead"):
    """Collect WARNING+ messages logged under logger_name while the block runs."""
    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Collect(level=logging.WARNING)
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


# --- inputs -------------------------------------------------------------------------------------
def synthetic_inputs(workdir, lines):
    tb = synthetic.write("thunderbird", lines, os.path.join(workdir, "tb.log"), seed=1)
    bgl = synthetic.write("bgl", lines, os.path.join(workdir, "bgl.log"), seed=2)
    hdfs, labels = synthetic.write("hdfs", lines, os.path.join(workdir, "hdfs", "HDFS.log"), seed=3)
    return tb, bgl, hdfs, labels


def real_slice(workdir, relative, lines=SLICE_LINES):
    source = os.path.join(DATASETS, relative)
    if not os.path.exists(source):
        return None
    target = os.path.join(workdir, "slice_" + relative.replace("/", "_"))
    if not os.path.exists(target):
        with open(source, "rb") as src, open(target, "wb") as dst:
            for i, line in enumerate(src):
                if i >= lines:
                    break
                dst.write(line)
    return target


# The BGL timestamp; the synthetic log's too-short lines have none, so keep and drop differ.
RAW_TIMESTAMPS = {"timestamp_pattern": r"(\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d{6})",
                  "timestamp_format": "%Y-%m-%d-%H.%M.%S%.f"}


def loader_cases(workdir, quick):
    from loglead.loaders import BGLLoader, HDFSLoader, RawLoader, ThuSpiLibLoader
    tb, bgl, hdfs, labels = synthetic_inputs(workdir, 30_000 if quick else 300_000)
    lines = 30_000 if quick else 300_000
    tb_bad = synthetic.write("thunderbird", lines, os.path.join(workdir, "tb_bad.log"), seed=6, awkward=True)
    bgl_bad = synthetic.write("bgl", lines, os.path.join(workdir, "bgl_bad.log"), seed=7, awkward=True)
    cases = [
        ("synthetic thunderbird", ThuSpiLibLoader, tb, {}),
        ("synthetic liberty-style (no component split)", ThuSpiLibLoader, tb, {"split_component": False}),
        ("synthetic bgl", BGLLoader, bgl, {}),
        ("synthetic hdfs", HDFSLoader, hdfs, {"labels_file_name": labels}),
        ("synthetic thunderbird with invalid utf-8 and unbalanced quotes", ThuSpiLibLoader, tb_bad, {}),
        ("synthetic bgl with invalid utf-8 and unbalanced quotes", BGLLoader, bgl_bad, {}),
        ("raw synthetic thunderbird with invalid utf-8 and unbalanced quotes", RawLoader, tb_bad, {}),
        ("raw synthetic bgl, timestamps, keep", RawLoader, bgl_bad, dict(RAW_TIMESTAMPS, missing_timestamp_action="keep")),
        ("raw synthetic bgl, timestamps, drop", RawLoader, bgl_bad, dict(RAW_TIMESTAMPS, missing_timestamp_action="drop")),
        ("raw multi-line, raise on a clean log", RawLoader,
         synthetic.write_multiline(lines // 10, os.path.join(workdir, "clean.log"), continuations=False),
         dict(RAW_TIMESTAMPS, missing_timestamp_action="raise")),
    ]
    multiline = synthetic.write_multiline(lines // 10, os.path.join(workdir, "multiline.log"), seed=8)
    for policy in ("keep", "drop", "fill-lastseen", "merge-message", "merge-add-column"):
        cases.append((f"raw multi-line, {policy}", RawLoader, multiline,
                      dict(RAW_TIMESTAMPS, missing_timestamp_action=policy)))
    if not quick:
        for name, loader, relative, kwargs in (
                ("bgl slice", BGLLoader, "bgl/BGL.log", {}),
                ("thunderbird slice", ThuSpiLibLoader, "thunderbird/tbird2.log", {}),
                ("spirit slice", ThuSpiLibLoader, "spirit/spirit2.log", {}),
                ("liberty slice", ThuSpiLibLoader, "liberty/liberty2.log", {"split_component": False}),
                ("hdfs slice", HDFSLoader, "hdfs/HDFS.log",
                 {"labels_file_name": os.path.join(DATASETS, "hdfs/preprocessed/anomaly_label.csv")})):
            path = real_slice(workdir, relative)
            if path is None:
                print(f"  (skipping {name}: {relative} not downloaded)")
                continue
            cases.append((name, loader, path, kwargs))
    return cases


# --- loaders ------------------------------------------------------------------------------------
def _sorted_seq(frame):
    return None if frame is None else frame.sort("seq_id")


def test_loaders(check, workdir, quick, baselines, capture):
    for name, loader_class, path, kwargs in loader_cases(workdir, quick):
        with captured_warnings() as eager_logs:
            eager_loader = loader_class(path, **kwargs)
            eager = eager_loader.execute()
        key = f"loader {name}"
        _baseline(check, key, eager, baselines, capture)
        if eager_loader.df_seq is not None:
            _baseline(check, key + " seq", _sorted_seq(eager_loader.df_seq), baselines, capture)

        lazy_loader = loader_class(path, **kwargs)
        check(f"{name}: supports streaming", getattr(lazy_loader, "supports_streaming", False))
        lazy = lazy_loader.scan()
        check(f"{name}: scan() is a LazyFrame", isinstance(lazy, pl.LazyFrame), type(lazy).__name__)
        check.equal(f"{name}: scan().collect() == execute()", lazy.collect(engine="streaming"), eager)

        out = os.path.join(workdir, f"sink_{abs(hash(name))}.parquet")
        seq_out = out.replace(".parquet", "_seq.parquet")
        with captured_warnings() as sink_logs:
            sink_loader = loader_class(path, **kwargs)
            sink_loader.sink(out, seq_path=seq_out)
        check.equal(f"{name}: sink() read back == execute()", pl.read_parquet(out), eager)
        small = out.replace(".parquet", "_small.parquet")
        loader_class(path, **kwargs).sink(small, check=False, chunk_bytes=65_536)
        check.equal(f"{name}: sink() in 64 KB chunks == execute()", pl.read_parquet(small), eager)
        if eager_loader.df_seq is not None:
            check.equal(f"{name}: sink() df_seq == execute() df_seq",
                        _sorted_seq(pl.read_parquet(seq_out)), _sorted_seq(eager_loader.df_seq))
            check.equal(f"{name}: loader.df_seq after sink()", _sorted_seq(sink_loader.df_seq),
                        _sorted_seq(eager_loader.df_seq))
        check(f"{name}: sink() logs the same null/non-UTF-8 warnings", sorted(eager_logs) == sorted(sink_logs),
              f"eager {eager_logs[:3]} vs sink {sink_logs[:3]}")
        if "invalid utf-8" in name:
            check(f"{name}: invalid bytes are reported as non-UTF-8",
                  any("non-UTF-8" in message for message in eager_logs), f"{eager_logs[:3]}")
    _raw_fallback(check, workdir)


def _raw_fallback(check, workdir):
    """RawLoader's 'raise' must fail the same way streaming, and many files can't stream."""
    from loglead.loaders import RawLoader
    multiline = os.path.join(workdir, "multiline.log")
    kwargs = dict(RAW_TIMESTAMPS, missing_timestamp_action="raise")
    try:
        RawLoader(multiline, **kwargs).execute()
        expected = None
    except ValueError as error:
        expected = str(error)
    check("raw raise: execute() raises on continuation lines", expected is not None)
    for name, run in (("scan()", lambda: RawLoader(multiline, **kwargs).scan()),
                      ("sink()", lambda: RawLoader(multiline, **kwargs).sink(out)),
                      ("sink() in 64 KB chunks", lambda: RawLoader(multiline, **kwargs).sink(out, chunk_bytes=65_536))):
        out = os.path.join(workdir, "sink_raw_raise.parquet")
        if os.path.exists(out):
            os.remove(out)
        try:
            run()
            got = None
        except ValueError as error:
            got = str(error)
        check(f"raw raise: {name} raises what execute() does", got == expected, f"{got} vs {expected}")
        check(f"raw raise: {name} leaves no file", not os.path.exists(out))

    bgl = os.path.join(workdir, "bgl_bad.log")
    folder = os.path.join(workdir, "raw_folder")
    os.makedirs(folder, exist_ok=True)
    for part in ("a.log", "b.log"):
        with open(bgl, "rb") as src, open(os.path.join(folder, part), "wb") as dst:
            dst.write(src.read(200_000))
    kwargs = {"filename_pattern": "*.log"}
    eager = RawLoader(folder, **kwargs).execute()
    check("raw many files: does not claim to stream", not RawLoader(folder, **kwargs).supports_streaming)
    out = os.path.join(workdir, "sink_raw_many.parquet")
    RawLoader(folder, **kwargs).sink(out)
    check.equal("raw many files: sink() falls back and == execute()", pl.read_parquet(out), eager)


# --- enhancers ----------------------------------------------------------------------------------
EXPRESSION_STEPS = [
    ("length", {}), ("mask", {}), ("words", {}), ("alphanumerics", {}), ("trigrams", {}),
    ("words", {"column": "e_message_normalized", "reparse": True}),
]


def _enhancer_inputs(workdir, quick):
    inputs = {
        "tb sample": pl.read_parquet(os.path.join(SAMPLES, "tb_0125percent.parquet")),
        "hdfs sample": pl.read_parquet(os.path.join(SAMPLES, "hdfs_events_2percent.parquet")),
    }
    if not quick:
        from loglead.loaders import ThuSpiLibLoader
        tb = synthetic.write("thunderbird", 200_000, os.path.join(workdir, "tb_enh.log"), seed=4)
        inputs["synthetic thunderbird"] = ThuSpiLibLoader(tb).execute()
    return inputs


def _apply(enhancer, steps):
    for method, kwargs in steps:
        getattr(enhancer, method)(**kwargs)
    return enhancer


def test_enhancer_expressions(check, workdir, quick, baselines, capture):
    from loglead.enhancers import EventLogEnhancer
    for name, df in _enhancer_inputs(workdir, quick).items():
        eager = _apply(EventLogEnhancer(df), EXPRESSION_STEPS).df
        _baseline(check, f"expressions {name}", eager, baselines, capture)
        lazy_enhancer = _apply(EventLogEnhancer(df.lazy()), EXPRESSION_STEPS)
        check(f"{name}: enhancer keeps a LazyFrame lazy", isinstance(lazy_enhancer.df, pl.LazyFrame),
              type(lazy_enhancer.df).__name__)
        check.equal(f"{name}: lazy expressions == eager", lazy_enhancer.df.collect(), eager)
        out = os.path.join(workdir, "enh_expr.parquet")
        _apply(EventLogEnhancer(df.lazy()), EXPRESSION_STEPS).sink(out)
        check.equal(f"{name}: sink() read back == eager", pl.read_parquet(out), eager)
        source = os.path.join(workdir, "enh_source.parquet")
        df.write_parquet(source)
        enhancer = _apply(EventLogEnhancer.from_parquet(source), EXPRESSION_STEPS)
        enhancer.sink(out, batch_size=7_777)
        check.equal(f"{name}: from_parquet() + sink() == eager", pl.read_parquet(out), eager)
        second = os.path.join(workdir, "enh_second.parquet")
        enhancer.alphanumerics(column="e_message_normalized", reparse=True)
        enhancer.sink(second, batch_size=5_000)
        again = EventLogEnhancer(eager.clone())
        again.alphanumerics(column="e_message_normalized", reparse=True)
        check.equal(f"{name}: enhancing the sunk file again == eager", pl.read_parquet(second), again.df)

    tb = pl.read_parquet(os.path.join(SAMPLES, "tb_0125percent.parquet")).lazy()
    for method in ("parse_tip", "parse_brain", "parse_spell", "create_neural_emb"):
        enhancer = EventLogEnhancer(tb)
        enhancer.mask()
        try:
            getattr(enhancer, method)()
            check(f"{method} on a LazyFrame raises NotImplementedError", False, "did not raise")
        except NotImplementedError:
            check(f"{method} on a LazyFrame raises NotImplementedError", True)


STATEFUL = {
    # name: (input sample, enhancer steps, batch sizes to try[, first rows only])
    "drain tb": ("tb_0125percent.parquet", [("mask", {}), ("parse_drain", {})], [997, 50_000, 1_000_000]),
    "drain tb templates": ("tb_0125percent.parquet",
                           [("mask", {}), ("parse_drain", {"templates": True})], [4_096]),
    "drain hdfs": ("hdfs_events_2percent.parquet", [("mask", {}), ("parse_drain", {})], [9_999]),
    "drain + minhash tb": ("tb_0125percent.parquet",
                           [("mask", {}), ("words", {}), ("minhash", {"token_column": "e_words"}),
                            ("parse_drain", {})], [5_000]),
    # All-rows parsers do not stream; they are here for the multi-chunk check only.
    "spell hdfs": ("hdfs_events_2percent.parquet", [("mask", {}), ("parse_spell", {})], [], 20_000),
    "lenma tb": ("tb_0125percent.parquet", [("mask", {}), ("words", {}), ("parse_lenma", {})], [], 1_000),
}
CHUNKS = 64


def child(mode, case, out, batch_size):
    """Run one stateful case and save the result.

    Runs in a fresh process because Drain's template miner is a module-level singleton, which
    would carry templates over from an earlier run.
    """
    from loglead.enhancers import EventLogEnhancer
    sample, steps, _, *rows = STATEFUL[case]
    source = os.path.join(SAMPLES, sample)
    if rows:
        head = out + ".head.parquet"
        pl.read_parquet(source).head(rows[0]).write_parquet(head)
        source = head
    if mode == "eager":
        _apply(EventLogEnhancer(pl.read_parquet(source)), steps).df.write_parquet(out)
    elif mode == "chunked":
        df = pl.read_parquet(source)
        size = -(-df.height // CHUNKS)
        df = pl.concat([df.slice(offset, size) for offset in range(0, df.height, size)], rechunk=False)
        assert df.n_chunks() > 1, "input did not split into chunks"
        _apply(EventLogEnhancer(df), steps).df.write_parquet(out)
    elif mode == "parquet":
        _apply(EventLogEnhancer.from_parquet(source), steps).sink(out, batch_size=int(batch_size))
    else:
        _apply(EventLogEnhancer(pl.scan_parquet(source)), steps).sink(out, batch_size=int(batch_size))


def _run_child(mode, case, out, batch_size=0):
    result = subprocess.run([sys.executable, __file__, "--child", mode, case, out, str(batch_size)],
                            capture_output=True, text=True)
    if result.returncode:
        print(result.stdout[-2000:], result.stderr[-4000:])
    return result.returncode == 0


def test_enhancer_stateful(check, workdir, quick, baselines, capture):
    cases = ["drain tb", "drain hdfs", "spell hdfs"] if quick else list(STATEFUL)
    for case in cases:
        eager_out = os.path.join(workdir, "stateful_eager.parquet")
        if not check(f"{case}: eager run", _run_child("eager", case, eager_out)):
            continue
        eager = pl.read_parquet(eager_out)
        _baseline(check, f"stateful {case}", eager, baselines, capture)
        out = os.path.join(workdir, "stateful_chunked.parquet")
        label = f"{case}: eager on a {CHUNKS}-chunk frame"
        if check(f"{label} runs", _run_child("chunked", case, out)):
            check.equal(f"{label} == single chunk", pl.read_parquet(out), eager)
        for batch_size in STATEFUL[case][2][: 2 if quick else None]:
            for mode in ("parquet", "lazy"):
                out = os.path.join(workdir, "stateful_stream.parquet")
                label = f"{case}: {mode}, batch_size={batch_size}"
                if check(f"{label} runs", _run_child(mode, case, out, batch_size)):
                    check.equal(f"{label} == eager", pl.read_parquet(out), eager)


# --- detectors ----------------------------------------------------------------------------------
DETECTORS = ["train_LR", "train_DT", "train_IsolationForest", "train_KMeans", "train_RarityDetector",
             "train_OOVDetector", "train_LSVM"]


def _compare_predictions(check, name, whole, batched, exact_scores=True):
    if not check(f"{name}: same rows", whole.height == batched.height, f"{whole.height} vs {batched.height}"):
        return
    check.equal(f"{name}: pred_ano identical", batched.select("pred_ano"), whole.select("pred_ano"))
    if "pred_ano_proba" in whole.columns:
        a, b = whole["pred_ano_proba"].to_numpy(), batched["pred_ano_proba"].to_numpy()
        if exact_scores:
            check(f"{name}: scores equal", np.allclose(a, b, rtol=1e-9, atol=1e-12),
                  f"max diff {np.max(np.abs(a - b)):.3g}")
        else:
            corr = np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]
            check(f"{name}: scores rank-correlate (>0.95)", corr > 0.95, f"spearman {corr:.3f}")


def test_detectors(check, workdir, quick, baselines, capture):
    from loglead import AnomalyDetector
    from loglead.enhancers import EventLogEnhancer
    df = pl.read_parquet(os.path.join(SAMPLES, "tb_0125percent.parquet"))
    df = EventLogEnhancer(df).words()
    df = EventLogEnhancer(df).length()
    for predictors in ({"item_list_col": "e_words"},
                       {"numeric_cols": ["e_words_len", "e_chars_len"], "categorical_cols": ["component"]}):
        label = next(iter(predictors.values()))
        sad = AnomalyDetector(print_scores=False, auc_roc=True, **predictors)
        sad.test_train_split(df, test_frac=0.5)
        for method in DETECTORS if not quick else DETECTORS[:4]:
            if method in ("train_RarityDetector", "train_OOVDetector") and "item_list_col" not in predictors:
                continue
            getattr(sad, method)()
            whole = sad.predict()
            batched = pl.concat(list(sad.predict_batches(sad.test_df, batch_size=1_001)))
            _compare_predictions(check, f"{label} {method} predict_batches", whole, batched,
                                 exact_scores=method != "train_LSVM")
        out = os.path.join(workdir, "scores.parquet")
        check("model's test_df restored after predict_batches",
              getattr(sad.model, "test_df", sad.test_df) is sad.test_df)
        sad.train_IsolationForest()
        whole = sad.predict()
        sad.predict_to_parquet(sad.test_df.lazy(), out, batch_size=2_500)
        _compare_predictions(check, f"{label} predict_to_parquet(LazyFrame)", whole, pl.read_parquet(out))
        test_file = os.path.join(workdir, "test_df.parquet")
        sad.test_df.write_parquet(test_file)
        sad.predict_to_parquet(test_file, out, batch_size=3_000)
        _compare_predictions(check, f"{label} predict_to_parquet(path)", whole, pl.read_parquet(out))
        keep = pl.col("anomaly")
        sad.predict_to_parquet(test_file, out, batch_size=3_000, where=keep)
        _compare_predictions(check, f"{label} predict_to_parquet(where=...)", whole.filter(keep),
                             pl.read_parquet(out))
        check("predict_to_parquet keeps the input columns",
              set(sad.test_df.columns) <= set(pl.read_parquet_schema(out)), "")
    if not quick:
        _sequence_detectors(check)


def _sequence_detectors(check):
    """Sequence detectors score the sequences of their test_df, not the feature matrix."""
    from loglead import AnomalyDetector
    from loglead.enhancers import EventLogEnhancer, SequenceEnhancer
    events = pl.read_parquet(os.path.join(SAMPLES, "hdfs_events_2percent.parquet"))
    seqs = pl.read_parquet(os.path.join(SAMPLES, "hdfs_seqs_2percent.parquet"))
    enhancer = EventLogEnhancer(events)
    enhancer.mask()
    events = enhancer.parse_drain()
    seqs = SequenceEnhancer(df=events, df_seq=seqs).events()
    sad = AnomalyDetector(item_list_col="e_event_drain_id", print_scores=False, auc_roc=True)
    sad.test_train_split(seqs, test_frac=0.5)
    for method in ("train_next_event_prediction", "train_lookahead_pairs", "train_OOVDetector"):
        getattr(sad, method)()
        whole = sad.predict()
        batched = pl.concat(list(sad.predict_batches(sad.test_df, batch_size=333)))
        _compare_predictions(check, f"hdfs sequences {method} predict_batches", whole, batched)


# --- sampling -----------------------------------------------------------------------------------
def test_sample(check, workdir, quick, baselines, capture):
    from loglead.streaming import sample
    lf = pl.LazyFrame({"i": range(200_000), "s": [str(i) for i in range(200_000)]})
    a = sample(lf, fraction=0.1, seed=1)
    b = sample(lf, fraction=0.1, seed=1)
    c = sample(lf, fraction=0.1, seed=2)
    check("sample returns a DataFrame", isinstance(a, pl.DataFrame), type(a).__name__)
    check.equal("sample is deterministic for a seed", a, b)
    check("a different seed gives different rows", not a.equals(c))
    check("fraction=0.1 gives ~10% (±10%)", 18_000 <= a.height <= 22_000, str(a.height))
    check("sample keeps the input order", a["i"].is_sorted())
    check("sample keeps the columns", a.columns == ["i", "s"])
    n = sample(lf, n=5_000, seed=3)
    check("n=5000 gives ~5000 rows (±15%)", 4_250 <= n.height <= 5_750, str(n.height))
    check("n larger than the frame returns everything", sample(lf, n=10**9).height == 200_000)
    path = os.path.join(workdir, "sample_source.parquet")
    lf.collect().write_parquet(path)
    check.equal("sample(path) == sample(LazyFrame) for a seed", sample(path, fraction=0.1, seed=1, batch_size=7_000), a)
    check.equal("sample(path, n=...) == sample(LazyFrame, n=...)", sample(path, n=5_000, seed=3, batch_size=9_000), n)

    whole = fp.fingerprint(lf.collect())
    check("fingerprint(path) read in batches == fingerprint(DataFrame)",
          fp.fingerprint(path, batch_size=33_333) == whole, "")


# --- baselines ----------------------------------------------------------------------------------
def _baseline(check, key, frame, baselines, capture):
    if capture:
        if key in baselines and capture != "recapture":
            print(f"  kept existing baseline for {key}")
            return
        baselines[key] = fp.fingerprint(frame)
        print(f"  captured {key}")
        return
    actual = fp.fingerprint(frame)
    expected = baselines.get(key)
    if expected is None:
        print(f"  (no eager baseline for {key}; run with --capture on the reference code)")
        return
    problems = fp.compare(expected, actual)
    check(f"{key}: eager output matches its baseline", not problems, "; ".join(problems))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="samples and small synthetic logs only")
    parser.add_argument("--capture", action="store_true",
                        help="record eager outputs missing from the baseline file; skip the streaming checks")
    parser.add_argument("--recapture", action="store_true",
                        help="like --capture, but also overwrite existing baselines")
    parser.add_argument("--workdir", default=None, help="keep inputs/outputs here instead of a temp folder")
    parser.add_argument("--child", nargs=4, metavar=("MODE", "CASE", "OUT", "BATCH"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child(*args.child)
        return 0

    print(f"polars {pl.__version__}, python {sys.version.split()[0]}")
    baseline_file = BASELINE.format(mode="quick" if args.quick else "full")
    baselines = fp.load_baselines(baseline_file)
    check = Check()
    with tempfile.TemporaryDirectory(prefix="loglead_streaming_") as tmp:
        workdir = args.workdir or tmp
        os.makedirs(workdir, exist_ok=True)
        if args.capture or args.recapture:
            _capture(check, workdir, args.quick, baselines, "recapture" if args.recapture else "capture")
            fp.save_baselines(baseline_file, baselines)
            print(f"\nEager baselines written to {baseline_file}")
            return 0
        for title, fn in (("loaders: eager vs scan vs sink", test_loaders),
                          ("enhancers: expressions on DataFrame vs LazyFrame", test_enhancer_expressions),
                          ("enhancers: stateful parsers in batches", test_enhancer_stateful),
                          ("detectors: predict vs predict_batches", test_detectors),
                          ("streaming.sample", test_sample)):
            check.section(title, fn, workdir, args.quick, baselines, False)
    print("\n" + "=" * 70)
    if check.failures:
        print(f" {len(check.failures)} FAILED, {check.passed} passed")
        for name in check.failures:
            print(f"   - {name}")
        return 1
    print(f" all {check.passed} checks passed")
    return 0


def _capture(check, workdir, quick, baselines, mode):
    """Record eager outputs only: the parts of each test that run on the reference code."""
    from loglead.enhancers import EventLogEnhancer
    for name, loader_class, path, kwargs in loader_cases(workdir, quick):
        loader = loader_class(path, **kwargs)
        _baseline(check, f"loader {name}", loader.execute(), baselines, mode)
        if loader.df_seq is not None:
            _baseline(check, f"loader {name} seq", _sorted_seq(loader.df_seq), baselines, mode)
    for name, df in _enhancer_inputs(workdir, quick).items():
        _baseline(check, f"expressions {name}", _apply(EventLogEnhancer(df), EXPRESSION_STEPS).df, baselines, mode)
    for case in (["drain tb", "drain hdfs", "spell hdfs"] if quick else STATEFUL):
        out = os.path.join(workdir, "stateful_eager.parquet")
        if _run_child("eager", case, out):
            _baseline(check, f"stateful {case}", pl.read_parquet(out), baselines, mode)


if __name__ == "__main__":
    sys.exit(main())
