import polars as pl

__all__ = ['profile_columns', 'select_predictors', 'print_predictor_report',
           'LEAKY_COLUMNS', 'LEAKY_SUBSTRINGS']

"""
Picking predictor columns out of a loaded log frame.

Loaders keep the fields the raw log came with - Thunderbird lands `userid`, `location`,
`component`, `pid`; BGL lands `type`, `level`, `node`; a JSON log lands whatever its keys were.
These are predictors in their own right, but they cannot simply be handed to AnomalyDetector: a
log frame also holds the label, several identifiers, and columns with no variation at all, and
those three failure modes look exactly like good columns if you only count nulls.

Measured on the datasets in this repo, which is where the guards below come from:

- `label` (Thunderbird, BGL) *is* the target. `anomaly == (label != "-")` holds for every
  Thunderbird row, so a model given `label` scores perfectly and has learned nothing.
- BGL's `time` has 94,959 distinct values in 94,959 rows - no nulls, maximum variation, and
  useless: an identifier, one one-hot column per row.
- nginx's `remote_user` has exactly one distinct value, so it carries no information.
- BGL's `node` and `noderepeat` both have 36,520 distinct values, i.e. one is a copy of the other.

So "fewest nulls" ranks candidates but cannot choose them. Four guards reject first, and the
ranking then applies to what survives.
"""

# Columns that are the label, are derived from it, or identify a row rather than describe it.
# Matched case-insensitively against the whole name.
LEAKY_COLUMNS = frozenset({
    "pred_ano", "pred_ano_proba",           # AnomalyDetector's own output
    "seq_id", "row_nr", "index",            # identifiers
    "m_message", "e_message_normalized",    # the message itself - that is item_list_col's job
})

# Matched case-insensitively anywhere in the name, because label columns are named freely and an
# exact list keeps missing them. Every one of these was found leaking in this repo's own data:
#   label   Thunderbird/BGL `label` *is* the target (anomaly == label != "-"); Hadoop's sequence
#           frame spells it `Label`, which a case-sensitive exact match walks straight past, and
#           its values map onto the target perfectly ("Normal" -> 0% anomalies, everything else
#           -> 100%).
#   anomal  Nezha's `anomaly_folder`: false means 0% anomalies, so it gives the negative class away.
#   normal  LogLead's own `normal` column, and Nezha's `normal_json`, which leaks the same way.
# The cost of the substring rule is that a genuine predictor with one of these in its name is
# rejected too - `e_message_normalized` is, though it was on the exact list anyway. Rejecting a
# usable column is recoverable; silently training on the answer is not.
LEAKY_SUBSTRINGS = ("label", "anomal", "normal")

_NUMERIC = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64, pl.Boolean)
_CATEGORICAL = (pl.Utf8, pl.Categorical, pl.Enum)


def _role(dtype):
    """How a column could be used as a predictor, or None if it cannot be."""
    if dtype in _NUMERIC:
        return "numeric"
    if dtype in _CATEGORICAL:
        return "categorical"
    # Temporal columns are raw clock readings: near-unique, and meaningful only once turned into
    # something like hour-of-day, which is a separate job. Lists/structs are not scalars at all.
    return None


def profile_columns(df, exclude=None):
    """Per-column statistics for deciding what can serve as a predictor.

    Returns one row per column with dtype, role, null_frac, distinct and unique_frac. The
    expensive statistic - top_share, the share of the most common value - is computed only for
    columns that are still candidates, and is null for the rest; a group_by per column over a
    multi-million row frame is not worth paying for a column already ruled out.
    """
    extra = {c.lower() for c in (exclude or ())}
    rows = df.height

    def reserved(name):
        lowered = name.lower()
        return (lowered in LEAKY_COLUMNS or lowered in extra
                or any(token in lowered for token in LEAKY_SUBSTRINGS))

    null_counts = dict(zip(df.columns, df.null_count().row(0)))
    # Count distinct values only where the answer can change a decision. A list- or struct-typed
    # column is rejected on its dtype regardless, and counting distinct values in one means hashing
    # every nested element: on ait_ads (132k rows, 37 nested columns) doing it for the whole frame
    # allocated 5.4 GB and got the process OOM-killed.
    countable = [c for c in df.columns if _role(df.schema[c]) is not None]
    distincts = dict(zip(countable, df.select([pl.col(c).n_unique() for c in countable]).row(0))) \
        if countable else {}

    records = []
    for name in df.columns:
        distinct = distincts.get(name)
        records.append({
            "column": name,
            "dtype": str(df.schema[name]),
            "role": _role(df.schema[name]),
            "null_frac": null_counts[name] / rows if rows else 0.0,
            "distinct": distinct,
            "unique_frac": (distinct / rows if rows else 0.0) if distinct is not None else None,
            "reserved": reserved(name),
        })
    return pl.DataFrame(records, schema_overrides={"distinct": pl.Int64, "unique_frac": pl.Float64})


def select_predictors(df, n_columns=5, exclude=None, max_null_frac=0.5, max_unique_frac=0.5,
                      max_distinct=10000, max_dominance=0.999):
    """Choose up to `n_columns` predictor columns, and say why the others were dropped.

    Ranking is by fewest nulls, tie-broken by fewest distinct values, which prefers compact
    categories over sprawling ones when - as is usual for these datasets - nothing is null.

    The guards, in the order they are applied:
      reserved     the label, a column derived from it, or an identifier (see LEAKY_COLUMNS)
      unusable     not a scalar dtype: list, struct, or a raw timestamp
      too_null     more than `max_null_frac` missing
      constant     a single distinct value, so no variation to learn from
      identifier   `max_unique_frac` of rows or more are distinct, or more than `max_distinct`
                   values, i.e. a key rather than a category
      dominated    one value covers more than `max_dominance` of rows

    Returns (numeric_cols, categorical_cols, profile), where profile is the frame from
    profile_columns() plus `selected` and `reason` columns. The two lists are separate because
    AnomalyDetector consumes them differently: numeric columns go in as they are, categorical
    ones need encoding first.
    """
    profile = profile_columns(df, exclude=exclude)
    reasons, candidates = [], []

    for row in profile.iter_rows(named=True):
        if row["reserved"]:
            reason = "reserved"
        elif row["role"] is None:
            reason = "unusable"
        elif row["null_frac"] > max_null_frac:
            reason = "too_null"
        elif row["distinct"] <= 1:
            reason = "constant"
        elif row["unique_frac"] >= max_unique_frac or row["distinct"] > max_distinct:
            reason = "identifier"
        else:
            reason = None
            candidates.append(row["column"])
        reasons.append(reason)

    # top_share needs a group_by per column, so only the survivors pay for it.
    shares = {}
    for name in candidates:
        counts = df.get_column(name).value_counts(sort=True)
        shares[name] = counts.row(0)[1] / df.height if df.height else 0.0

    reasons = [
        "dominated" if r is None and shares.get(c, 0.0) > max_dominance else r
        for c, r in zip(profile["column"], reasons)
    ]
    profile = profile.with_columns(
        pl.Series("top_share", [shares.get(c) for c in profile["column"]], dtype=pl.Float64),
        pl.Series("reason", reasons, dtype=pl.Utf8),
    )

    ranked = (profile.filter(pl.col("reason").is_null())
                     .sort(["null_frac", "distinct", "column"])
                     .head(n_columns))
    chosen = set(ranked["column"])
    profile = profile.with_columns(pl.col("column").is_in(list(chosen)).alias("selected"))
    # Kept for the report: a column can pass every guard and still lose to a better-ranked one.
    profile = profile.with_columns(
        pl.when(pl.col("reason").is_not_null()).then(pl.col("reason"))
          .when(pl.col("selected")).then(pl.lit("SELECTED"))
          .otherwise(pl.lit("not_in_top_n")).alias("reason"))

    numeric = [r["column"] for r in ranked.iter_rows(named=True) if r["role"] == "numeric"]
    categorical = [r["column"] for r in ranked.iter_rows(named=True) if r["role"] == "categorical"]
    return numeric, categorical, profile.drop("reserved")


def print_predictor_report(profile, name=""):
    """Print what was chosen and what was rejected. The rejections are the informative half."""
    print(f"Predictor columns{' for ' + name if name else ''}:")
    header = (f"  {'column':<24}{'role':<13}{'dtype':<10}{'null%':>7}{'distinct':>10}"
              f"{'uniq%':>7}{'top%':>7}  decision")
    print(header)
    print("  " + "-" * (len(header) - 2))
    order = {"SELECTED": 0}
    for row in profile.sort([pl.col("reason").replace_strict(order, default=1), "column"]).iter_rows(named=True):
        top = f"{row['top_share']*100:>6.1f}%" if row["top_share"] is not None else f"{'-':>7}"
        # distinct is not measured for columns rejected on dtype - see profile_columns.
        distinct = f"{row['distinct']:>10}" if row["distinct"] is not None else f"{'-':>10}"
        uniq = f"{row['unique_frac']*100:>6.1f}%" if row["unique_frac"] is not None else f"{'-':>7}"
        print(f"  {row['column']:<24}{str(row['role'] or '-'):<13}{row['dtype'][:9]:<10}"
              f"{row['null_frac']*100:>6.1f}%{distinct}{uniq}"
              f"{top}  {row['reason']}")
