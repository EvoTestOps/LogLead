"""Combining several distance/anomaly measures into one comparable score.

The four distance measures and the four anomaly detectors all live on wildly
different scales (KMeans returns a cluster distance, IsolationForest a shifted
decision function, RarityModel and OOVDetector raw counts). Summing them raw is
meaningless, so results get combined two ways:

* ``zscore_sum`` -- per-column z-score, then summed. Sensitive to outliers.
* ``rank_sum``   -- per-column rank, then summed. Scale-free, and what
  LogDelta recommends sorting by: a measure that comes out wildly distorted
  drags a z-score sum with it, but moves a rank sum by at most one rank.

``rank_sum`` is a *within-result* ordering, not an absolute score. Rank 1 is the
least anomalous row on a measure, so over four measures the sum starts at 4 and
rises to ``4 * n_rows``; comparing one across results of different sizes, or over
different numbers of measures, is meaningless.

Ported from LogDelta's ``log_analysis_functions.py``.
"""

import numpy as np
import polars as pl
from scipy.stats import rankdata, zscore

#: Column names produced by the four distance measures.
DISTANCE_COLUMNS = ["cosine", "jaccard", "compression", "containment"]

#: Column names produced by the four anomaly detectors.
ANOMALY_COLUMNS = [
    "kmeans_pred_ano_proba",
    "IF_pred_ano_proba",
    "RM_pred_ano_proba",
    "OOVD_pred_ano_proba",
]


def _matrix(rows, columns):
    return np.array(
        [[np.nan if row.get(col) is None else row[col] for col in columns] for row in rows],
        dtype=float,
    )


def add_combined_scores(data, columns=None):
    """Append ``zscore_sum`` and ``rank_sum`` over ``columns``.

    :param data: a ``pl.DataFrame`` or a list of dicts.
    :param columns: measure columns to combine. Defaults to
        :data:`ANOMALY_COLUMNS` for a DataFrame and :data:`DISTANCE_COLUMNS`
        for a list, matching LogDelta's behaviour.
    :returns: the same type as ``data``, with the two extra fields.

    Columns absent from the data are skipped, so running a subset of the
    detectors still produces a usable combined score. Returns ``data``
    unchanged if fewer than two rows or no measure columns are present --
    a z-score over one row is undefined.
    """
    is_frame = isinstance(data, pl.DataFrame)
    if is_frame:
        columns = columns if columns is not None else ANOMALY_COLUMNS
        rows = data.to_dicts()
    elif isinstance(data, list):
        columns = columns if columns is not None else DISTANCE_COLUMNS
        rows = data
    else:
        raise ValueError(
            f"Unsupported datatype: {type(data)}. Supported: pl.DataFrame, list"
        )

    present = [col for col in columns if rows and col in rows[0]]
    if not rows or not present:
        return data

    matrix = _matrix(rows, present)
    if len(rows) < 2:
        zscore_sum = np.zeros(len(rows))
        rank_sum = np.full(len(rows), float(len(present)))
    else:
        zscore_sum = np.apply_along_axis(
            lambda col: zscore(col, nan_policy="omit"), axis=0, arr=matrix
        ).sum(axis=1)
        rank_sum = np.apply_along_axis(
            lambda col: rankdata(col, nan_policy="omit"), axis=0, arr=matrix
        ).sum(axis=1)

    for idx, row in enumerate(rows):
        row["zscore_sum"] = float(zscore_sum[idx])
        row["rank_sum"] = float(rank_sum[idx])

    return pl.DataFrame(rows) if is_frame else rows


def moving_averages(df, window_size):
    """Rolling mean of every numeric column, as ``moving_avg_<window>_<col>``.

    :returns: a frame of *only* the new columns, ready to ``with_columns`` onto
        the original.
    """
    numeric = [
        col
        for col, dtype in df.schema.items()
        if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    ]
    if not numeric:
        raise ValueError("No numeric columns found in the DataFrame")
    return df.select(
        [
            pl.col(col).rolling_mean(window_size).alias(f"moving_avg_{window_size}_{col}")
            for col in numeric
        ]
    )


def normalize_measure_columns(df, columns):
    """Min-max normalize ``columns`` against a *shared* min and max.

    A detector's raw score and its 10/100-line moving averages belong to one
    measure family and must stay on a common scale so they can be plotted
    together. Nulls are filled with the column median before computing the
    range.
    """
    subset = df.select(columns)
    filled = subset.with_columns(pl.all().fill_null(pl.all().median()))
    measure_min = filled.min().to_numpy().min()
    measure_max = filled.max().to_numpy().max()

    if measure_max == measure_min or not np.isfinite([measure_min, measure_max]).all():
        return subset

    return subset.select(
        [
            ((pl.col(col) - measure_min) / (measure_max - measure_min)).alias(col)
            for col in columns
        ]
    )
