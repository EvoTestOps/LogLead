"""Content fingerprints of loader output, recorded as baselines and checked against them.

A fingerprint is the row count, the schema, and per column the null count and an order-sensitive
hash of the values. It is computed as one streaming aggregation, so it also works on a frame too
large to hold in memory (pass a LazyFrame, e.g. pl.scan_parquet).

Hashes are only comparable under the Polars version that produced them. A baseline recorded under
another version still has its row count, schema and null counts checked; the hash comparison is
skipped with a note.
"""

import json
import os

import polars as pl

_INDEX = "__fp_row"


def fingerprint(frame, batch_size=500_000):
    """Summarize a frame so a later run can check that loading still gives the same rows.

    Works on a DataFrame, LazyFrame or parquet path; a path is read in batches, so a file of any
    size can be checked. All three give the same fingerprint for the same rows.
    """
    if isinstance(frame, (str, os.PathLike)):
        from loglead.streaming import iter_batches
        schema = pl.scan_parquet(frame).collect_schema()
        batches = iter_batches(frame, batch_size)
    else:
        schema = frame.lazy().collect_schema()
        batches = [frame.lazy()]
    height, nulls, hashes, offset = 0, dict.fromkeys(schema, 0), {}, 0
    for batch in batches:
        lf = batch.lazy().with_row_index(_INDEX, offset=offset)
        aggs = [pl.len().alias("__height")]
        for name, dtype in schema.items():
            aggs.append(pl.col(name).null_count().alias(f"n:{name}"))
            if dtype == pl.Object:
                continue
            # Hashing the value together with its row number makes the fingerprint order-sensitive.
            aggs.append(pl.struct(pl.col(_INDEX), pl.col(name)).hash(seed=0).sum().alias(f"h:{name}"))
        row = lf.select(aggs).collect(engine="streaming").row(0, named=True)
        height += row["__height"]
        offset += row["__height"]
        for name in schema:
            nulls[name] += row[f"n:{name}"]
            if f"h:{name}" in row:
                # Polars' UInt64 sum wraps; summing the batch sums modulo 2**64 gives the same value.
                hashes[name] = (hashes.get(name, 0) + (row[f"h:{name}"] or 0)) % 2 ** 64
    return {
        "polars": pl.__version__,
        "height": height,
        "schema": {name: str(dtype) for name, dtype in schema.items()},
        "nulls": nulls,
        "hashes": {name: str(value) for name, value in hashes.items()},
    }


def compare(expected, actual):
    """Differences between two fingerprints as a list of strings; empty when they match."""
    problems = []
    if expected["height"] != actual["height"]:
        problems.append(f"height {expected['height']} -> {actual['height']}")
    if expected["schema"] != actual["schema"]:
        exp, act = expected["schema"], actual["schema"]
        missing = [c for c in exp if c not in act]
        extra = [c for c in act if c not in exp]
        changed = [f"{c}: {exp[c]} -> {act[c]}" for c in exp if c in act and exp[c] != act[c]]
        if list(exp) != list(act) and not missing and not extra:
            problems.append(f"column order {list(exp)} -> {list(act)}")
        if missing:
            problems.append(f"missing columns {missing}")
        if extra:
            problems.append(f"extra columns {extra}")
        if changed:
            problems.append(f"dtype changes {changed}")
    for name, count in expected["nulls"].items():
        if name in actual["nulls"] and actual["nulls"][name] != count:
            problems.append(f"nulls in {name}: {count} -> {actual['nulls'][name]}")
    if expected.get("polars") == actual.get("polars"):
        for name, value in expected["hashes"].items():
            if name in actual["hashes"] and actual["hashes"][name] != value:
                problems.append(f"content of {name} differs")
    return problems


def hashes_comparable(expected, actual):
    return expected.get("polars") == actual.get("polars")


def load_baselines(path):
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def save_baselines(path, baselines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(baselines, handle, indent=1, sort_keys=True)
        handle.write("\n")


def baseline_path(config_path):
    """Where a dataset config's baselines are kept: tests/baselines/<config name>.json."""
    stem = os.path.splitext(os.path.basename(config_path))[0]
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines", f"{stem}.json")
