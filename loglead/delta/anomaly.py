"""Unsupervised anomaly scoring of a target against a baseline of other log folders.

The shape is always the same: **comparison log folders are the training set, the
target is the test set**. There are no labels, so only unsupervised detectors
apply and the output is a score per object, not a verdict.

Four levels, mirroring LogDelta's config step names:

* **L1** ``anomaly_folder_filename`` -- score each log folder, by its file names.
* **L2** ``anomaly_folder_content``  -- score each log folder, by its log text.
* **L3** ``anomaly_file_content``    -- score each file of the target log folder.
* **L4** ``anomaly_line_content``    -- score each *line* of a target file.

Scores from the four detectors are on incomparable scales, so every result also
carries ``zscore_sum`` and ``rank_sum``. **Run all four and sort by ``rank_sum``**:
it is the sum of the per-detector ranks, so with four detectors it starts at 4 --
the row every detector ranks least anomalous -- and higher is more anomalous. Any
one detector can be badly distorted, which is also why ``rank_sum`` beats
``zscore_sum``: one distorted measure moves a z-score sum a long way and a rank sum
by at most one rank. Narrowing ``detectors`` is what breaks this, since ``rank_sum``
then combines fewer measures (with one detector it is just that detector's rank).
"""

import warnings

import polars as pl

from .. import AnomalyDetector
from . import log_root, scoring

#: detector name -> (AnomalyDetector method, output column)
DETECTORS = {
    "KMeans": ("train_KMeans", "kmeans_pred_ano_proba"),
    "IsolationForest": ("train_IsolationForest", "IF_pred_ano_proba"),
    "RarityModel": ("train_RarityModel", "RM_pred_ano_proba"),
    "OOVDetector": ("train_OOVDetector", "OOVD_pred_ano_proba"),
}

DEFAULT_DETECTORS = list(DETECTORS)

_NO_LABELS = "WARNING! data has no labels. Only unsupervised methods will work."


def run_anomaly_detection(
    train_df, test_df, field, detectors=None, vectorizer="Count", detector_params=None,
):
    """Score every row of ``test_df`` against a model fitted on ``train_df``.

    :param field: column holding the content, ``Utf8`` or ``List[Utf8]``.
    :param detectors: subset of :data:`DETECTORS`. ``None`` runs all four.
    :param detector_params: per-detector kwargs, e.g.
        ``{"KMeans": {"n_clusters": 3}, "RarityModel": {"threshold": 100}}``.
        Forwarded to the ``train_*`` method; LogDelta hardcoded these.
    :returns: ``test_df`` with one score column per detector appended.
    """
    detectors = DEFAULT_DETECTORS if detectors is None else list(detectors)
    unknown = [d for d in detectors if d not in DETECTORS]
    if unknown:
        raise ValueError(
            f"Unknown detectors {unknown}. Valid options: {DEFAULT_DETECTORS}"
        )
    detector_params = detector_params or {}

    vectorizer_class = log_root.create_vectorizer(vectorizer)

    sad = AnomalyDetector(item_list_col=field, print_scores=False, auc_roc=True)
    sad.train_df = train_df
    sad.test_df = test_df
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", _NO_LABELS, UserWarning)
        sad.prepare_train_test_data(vectorizer_class=vectorizer_class)

        # Start from the full test frame so every level keeps its context
        # columns (m_message, file_name, folder). LogDelta started from whichever
        # detector happened to run first and kept only the score when that was
        # not KMeans.
        result = test_df
        for name in detectors:
            method_name, out_col = DETECTORS[name]
            getattr(sad, method_name)(**detector_params.get(name, {}))
            scores = sad.predict().select(pl.col("pred_ano_proba").alias(out_col))
            result = result.with_columns(scores)

    return result


def _score_objects(df, field, target_folder, comparison_folders, group_by, detectors,
                   vectorizer, detector_params):
    """Aggregate target and baseline to one row per ``group_by`` value, then score."""
    target_df, comparison_folder_names = log_root.prepare_folders(df, target_folder, comparison_folders)
    target_agg = log_root.aggregate_dataframe(target_df, group_by, field)
    baseline_agg = log_root.aggregate_dataframe(
        df.filter(pl.col("folder").is_in(comparison_folder_names)), group_by, field
    )
    scored = run_anomaly_detection(
        baseline_agg, target_agg, field,
        detectors=detectors, vectorizer=vectorizer, detector_params=detector_params,
    )
    return scored, comparison_folder_names


def anomaly_folder(
    df, target_folder, comparison_folders="ALL", file=False, detectors=None, mask=True,
    content_format="Words", vectorizer="Count", detector_params=None,
):
    """L1/L2: score whole log folders.

    :param file: ``True`` describes a log folder by its *file names* (L1),
        ``False`` by its log *text* (L2). L1 forces ``content_format="File"``.
    :param target_folder: exact name, ``"ALL"``, an int N, or a ``"Prefix*"``
        wildcard -- each resolved target gets its own baseline.
    :returns: ``(results_df, df)`` -- one row per scored log folder.
    """
    if file:
        content_format = "File"
    df, field = log_root.prepare_content(df, mask, content_format)
    target_folder_names = log_root.resolve_target_folders(df, target_folder)

    frames = []
    for name in target_folder_names:
        scored, comparison_folder_names = _score_objects(
            df, field, name, comparison_folders, "folder",
            detectors, vectorizer, detector_params,
        )
        # LogDelta forwarded no vectorizer here, so L1/L2 always used Count.
        frames.append(
            scored.with_columns(pl.lit(" ".join(comparison_folder_names)).alias("comparison_folders"))
        )

    results = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()
    results = scoring.add_combined_scores(results, scoring.ANOMALY_COLUMNS)
    return results, df


def anomaly_file_content(
    df, target_folder, comparison_folders="ALL", target_files="ALL", detectors=None, mask=True,
    content_format="Words", vectorizer="Count", detector_params=None,
):
    """L3: score each file of the target log folder against the same files elsewhere.

    :returns: ``(results_df, df)`` -- one row per (target log folder, file).
    """
    df, field = log_root.prepare_content(df, mask, content_format)
    target_folder_names = log_root.resolve_target_folders(df, target_folder)

    frames = []
    for folder_name in target_folder_names:
        target_df, comparison_folder_names = log_root.prepare_folders(df, folder_name, comparison_folders)
        # Resolve against this log folder's own files, not the previous iteration's.
        file_names = log_root.prepare_files(target_df, target_files)
        baseline_agg = log_root.aggregate_dataframe(
            df.filter(pl.col("folder").is_in(comparison_folder_names)), "file_name", field
        )
        if baseline_agg.height == 0:
            continue

        for file_name in file_names:
            target_agg = log_root.aggregate_dataframe(
                target_df.filter(pl.col("file_name") == file_name), "file_name", field
            )
            if target_agg.height == 0:
                continue
            scored = run_anomaly_detection(
                baseline_agg, target_agg, field,
                detectors=detectors, vectorizer=vectorizer, detector_params=detector_params,
            )
            frames.append(scored.with_columns([
                pl.lit(folder_name).alias("target_folder"),
                pl.lit(" ".join(comparison_folder_names)).alias("comparison_folders"),
            ]))

    results = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()
    results = scoring.add_combined_scores(results, scoring.ANOMALY_COLUMNS)
    return results, df


def anomaly_line_content(
    df, target_folder, comparison_folders="ALL", target_files="ALL", detectors=None, mask=True,
    content_format="Words", vectorizer="Count", detector_params=None,
):
    """L4: score every line of a target file against the same file elsewhere.

    This is the drill-down level: each row is one real log line, so the score
    sits next to the message that earned it.

    :returns: ``(per_file, df)`` where ``per_file`` is a list of
        ``(target_folder, file_name, scored_df)``. Each ``scored_df`` carries
        ``line_number``, the original columns, one score column per detector,
        and 10/100-line moving averages of each.
    """
    df, field = log_root.prepare_content(df, mask, content_format)
    target_folder_names = log_root.resolve_target_folders(df, target_folder)

    per_file = []
    for folder_name in target_folder_names:
        target_df, comparison_folder_names = log_root.prepare_folders(df, folder_name, comparison_folders)
        file_names = log_root.prepare_files(target_df, target_files)
        other_folders_df = df.filter(pl.col("folder").is_in(comparison_folder_names))

        for file_name in file_names:
            target_lines = target_df.filter(pl.col("file_name") == file_name)
            baseline_lines = other_folders_df.filter(pl.col("file_name") == file_name)
            if baseline_lines.height == 0 or target_lines.height == 0:
                continue

            scored = run_anomaly_detection(
                baseline_lines, target_lines, field,
                detectors=detectors, vectorizer=vectorizer, detector_params=detector_params,
            )
            score_cols = [
                col for _, col in DETECTORS.values() if col in scored.columns
            ]
            score_only = scored.select(score_cols)
            scored = scored.with_columns(scoring.moving_averages(score_only, 10))
            scored = scored.with_columns(scoring.moving_averages(score_only, 100))
            scored = scored.with_row_index("line_number")
            per_file.append((folder_name, file_name, scored))

    return per_file, df
