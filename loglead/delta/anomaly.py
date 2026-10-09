"""Unsupervised anomaly scoring of a target against a baseline of other log folders.

The shape is always the same: **baseline log folders are the training set, the
target is the test set**. There are no labels, so only unsupervised detectors
apply and the output is a score per object, not a verdict.

Four functions, mirroring LogDelta's config step names:

* ``anomaly_folder_filename`` -- score each log folder, by its file names.
* ``anomaly_folder_content``  -- score each log folder, by its log text.
* ``anomaly_file_content``    -- score each file of the target log folder.
* ``anomaly_line_content``    -- score each *line* of a target file.

Scores from the four detectors are on incomparable scales, so every result also
carries ``zscore_sum`` and ``rank_sum``. **Run all four and sort by ``rank_sum``**:
it is the sum of the per-detector ranks, so with four detectors it starts at 4 --
the row every detector ranks least anomalous -- and higher is more anomalous. Any
one detector can be badly distorted, which is also why ``rank_sum`` beats
``zscore_sum``: one distorted measure moves a z-score sum a long way and a rank sum
by at most one rank. Narrowing ``detectors`` is what breaks this, since ``rank_sum``
then combines fewer measures (with one detector it is just that detector's rank).

``clean_range`` gives the scores a scale without a hand-picked threshold: a
sample of the baseline folders is each scored like a target against the
others, and ``scale_to_clean_range`` marks the rows scored above all of them.
"""

import logging
import warnings

import polars as pl

from .. import AnomalyDetector
from . import log_root, scoring
from .scoring import MAX_CLEAN_FOLDERS, MIN_CLEAN_FOLDERS, clean_sample

logger = logging.getLogger(__name__)

#: detector name -> (AnomalyDetector method, output column)
DETECTORS = {
    "KMeans": ("train_KMeans", "kmeans_pred_ano_proba"),
    "IsolationForest": ("train_IsolationForest", "IF_pred_ano_proba"),
    "RarityDetector": ("train_RarityDetector", "RM_pred_ano_proba"),
    "OOVDetector": ("train_OOVDetector", "OOVD_pred_ano_proba"),
}

#: deprecated detector-name alias -> current name in DETECTORS
_DEPRECATED_DETECTOR_ALIASES = {
    "RarityModel": "RarityDetector",
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
        ``{"KMeans": {"n_clusters": 3}, "RarityDetector": {"threshold": 100}}``.
        Forwarded to the ``train_*`` method; LogDelta hardcoded these.
    :returns: ``test_df`` with one score column per detector appended.
    """
    detectors = DEFAULT_DETECTORS if detectors is None else list(detectors)
    unknown = [d for d in detectors if d not in DETECTORS and d not in _DEPRECATED_DETECTOR_ALIASES]
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
            lookup_name = name
            if name in _DEPRECATED_DETECTOR_ALIASES:
                lookup_name = _DEPRECATED_DETECTOR_ALIASES[name]
                warnings.warn(
                    f"Detector name {name!r} is deprecated, use {lookup_name!r} instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            method_name, out_col = DETECTORS[lookup_name]
            getattr(sad, method_name)(**detector_params.get(name, {}))
            scores = sad.predict().select(pl.col("pred_ano_proba").alias(out_col))
            result = result.with_columns(scores)

    return result


def _group_by_baseline(df, target_folder_names, baseline_folders):
    """Group targets that share the same baseline log folders, in first-seen order.

    A target is never in its own baseline, so targets only need separate fits when
    one sits in another's baseline set. Targets with an identical baseline are
    scored against one fit: every detector scores rows independently, so the
    scores match per-target fits.
    """
    groups = {}
    for name in target_folder_names:
        _, baseline_folder_names = log_root.prepare_folders(df, name, baseline_folders)
        groups.setdefault(tuple(baseline_folder_names), []).append(name)
    return groups


def _group_files_by_baseline(df, target_folder_names, baseline_folders, target_files):
    """File-level :func:`_group_by_baseline`: one fit per (baseline log folders, file name).

    :returns: ``(jobs, groups)`` -- ``jobs`` lists every (target, file) in the
        order the per-target loop visited them; ``groups`` maps
        ``(baseline_folder_names, file_name)`` to the targets sharing that fit.
    """
    jobs, groups = [], {}
    for name in target_folder_names:
        target_df, baseline_folder_names = log_root.prepare_folders(df, name, baseline_folders)
        # Resolve against this log folder's own files, not the previous iteration's.
        for file_name in log_root.prepare_files(target_df, target_files):
            jobs.append((name, file_name))
            groups.setdefault((tuple(baseline_folder_names), file_name), []).append(name)
    return jobs, groups


def anomaly_folder(
    df, target_folder, baseline_folders="ALL", file=False, detectors=None, mask=True,
    content_format="Words", vectorizer="Count", detector_params=None,
):
    """Score whole log folders.

    :param file: ``True`` describes a log folder by its *file names* and
        forces ``content_format="File"``; ``False`` describes it by its log
        *text*.
    :param target_folder: exact name, ``"ALL"``, an int N, or a ``"Prefix*"``
        wildcard. Targets sharing a baseline are scored against one fit; a target
        is never in its own baseline, so overlapping ones get separate fits.
    :returns: ``(results_df, df)`` -- one row per scored log folder.
    """
    if file:
        content_format = "File"
    df, field = log_root.prepare_content(df, mask, content_format)
    target_folder_names = log_root.resolve_target_folders(df, target_folder)
    order = {name: i for i, name in enumerate(target_folder_names)}

    frames = []
    for baseline_folder_names, names in _group_by_baseline(
        df, target_folder_names, baseline_folders
    ).items():
        target_agg = log_root.aggregate_dataframe(
            df.filter(pl.col("folder").is_in(names)), "folder", field
        )
        baseline_agg = log_root.aggregate_dataframe(
            df.filter(pl.col("folder").is_in(baseline_folder_names)), "folder", field
        )
        # LogDelta forwarded no vectorizer here, so anomaly_folder always used Count.
        scored = run_anomaly_detection(
            baseline_agg, target_agg, field,
            detectors=detectors, vectorizer=vectorizer, detector_params=detector_params,
        )
        frames.append(
            scored.with_columns(pl.lit(" ".join(baseline_folder_names)).alias("baseline_folders"))
        )

    results = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()
    if frames:
        results = results.sort(pl.col("folder").replace_strict(order, return_dtype=pl.Int64))
    results = scoring.add_combined_scores(results, scoring.ANOMALY_COLUMNS)
    return results, df


def anomaly_file_content(
    df, target_folder, baseline_folders="ALL", target_files="ALL", detectors=None, mask=True,
    content_format="Words", vectorizer="Count", detector_params=None,
):
    """Score each file of the target log folder against the same file elsewhere.

    Files are matched **by name across log folders**, the same rule
    ``distance_file_content`` and the two line-level functions use: the baseline
    for ``security.log`` is the other log folders' ``security.log``, one
    document each. A target file no baseline log folder has is skipped, since
    there is nothing to judge it against.

    That makes this level meaningless on a log root where every log folder holds
    one uniquely-named file -- a split single file, say -- because no name is
    shared. Use :func:`anomaly_folder` there.

    :returns: ``(results_df, df)`` -- one row per (target log folder, file).
    """
    df, field = log_root.prepare_content(df, mask, content_format)
    target_folder_names = log_root.resolve_target_folders(df, target_folder)
    jobs, groups = _group_files_by_baseline(df, target_folder_names, baseline_folders, target_files)

    scored_by_job = {}
    skipped = 0
    for (baseline_folder_names, file_name), names in groups.items():
        # The baseline is *this file* as the other log folders wrote it: one
        # document per baseline log folder that has a file of this name.
        # Not one document per file name, which is what LogDelta's original
        # computed (hoisted out of this loop, so it never saw file_name) and
        # what its own "Found no files matching files in comparisons runs"
        # message shows it did not mean to: that scores security.log against
        # the other *kinds* of file rather than against the other runs'
        # security.log, which is the comparison this level exists to make.
        baseline_agg = log_root.aggregate_dataframe(
            df.filter(pl.col("folder").is_in(baseline_folder_names)
                      & (pl.col("file_name") == file_name)), "folder", field
        )
        if baseline_agg.height == 0:
            # no baseline log folder has a file of this name
            logger.debug("anomaly_file_content: %s in %s has no baseline log folder with that "
                         "file, skipped.", file_name, names)
            skipped += len(names)
            continue
        target_agg = log_root.aggregate_dataframe(
            df.filter(pl.col("folder").is_in(names) & (pl.col("file_name") == file_name)),
            "folder", field,
        )
        scored = run_anomaly_detection(
            baseline_agg, target_agg, field,
            detectors=detectors, vectorizer=vectorizer, detector_params=detector_params,
        )
        scored = scored.rename({"folder": "target_folder"}).with_columns(
            pl.lit(file_name).alias("file_name"),
            pl.lit(" ".join(baseline_folder_names)).alias("baseline_folders"),
        )
        scored = scored.select(
            "file_name", pl.exclude("file_name", "target_folder", "baseline_folders"),
            "target_folder", "baseline_folders",
        )
        for name in names:
            scored_by_job[(name, file_name)] = scored.filter(pl.col("target_folder") == name)

    if skipped:
        logger.info("anomaly_file_content: skipped %d target file(s) with no comparable data.",
                    skipped)
    frames = [scored_by_job[job] for job in jobs if job in scored_by_job]
    results = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()
    results = scoring.add_combined_scores(results, scoring.ANOMALY_COLUMNS)
    return results, df


def anomaly_line_content(
    df, target_folder, baseline_folders="ALL", target_files="ALL", detectors=None, mask=True,
    content_format="Words", vectorizer="Count", detector_params=None,
):
    """Score every line of a target file against the same file elsewhere.

    This is the drill-down level: each row is one real log line, so the score
    sits next to the message that earned it.

    :returns: ``(per_file, df)`` where ``per_file`` is a list of
        ``(target_folder, file_name, scored_df)``. Each ``scored_df`` carries
        ``line_number``, the original columns, one score column per detector,
        and 10/100-line moving averages of each.
    """
    df, field = log_root.prepare_content(df, mask, content_format)
    target_folder_names = log_root.resolve_target_folders(df, target_folder)
    jobs, groups = _group_files_by_baseline(df, target_folder_names, baseline_folders, target_files)

    scored_by_job = {}
    for (baseline_folder_names, file_name), names in groups.items():
        baseline_lines = df.filter(pl.col("folder").is_in(baseline_folder_names)
                                   & (pl.col("file_name") == file_name))
        if baseline_lines.height == 0:
            continue
        # Concatenated in target order; filtering by folder below keeps each
        # target's lines in their original order.
        target_lines = pl.concat(
            [df.filter((pl.col("folder") == name) & (pl.col("file_name") == file_name))
             for name in names],
            how="vertical_relaxed",
        )
        scored_group = run_anomaly_detection(
            baseline_lines, target_lines, field,
            detectors=detectors, vectorizer=vectorizer, detector_params=detector_params,
        )
        score_cols = [
            col for _, col in DETECTORS.values() if col in scored_group.columns
        ]
        for name in names:
            scored = scored_group.filter(pl.col("folder") == name)
            if scored.height == 0:
                continue
            score_only = scored.select(score_cols)
            scored = scored.with_columns(scoring.moving_averages(score_only, 10))
            scored = scored.with_columns(scoring.moving_averages(score_only, 100))
            scored_by_job[(name, file_name)] = scored.with_row_index("line_number")

    per_file = [(name, file_name, scored_by_job[(name, file_name)])
                for name, file_name in jobs if (name, file_name) in scored_by_job]
    return per_file, df


_RANGE_SCHEMA = {"detector": pl.Utf8, "clean_min": pl.Float64,
                 "clean_mid": pl.Float64, "clean_max": pl.Float64}

#: score column -> detector name, for naming the clean range rows.
_DETECTOR_OF = {column: name for name, (_, column) in DETECTORS.items()}


def _left_out_scores(df, field, name, clean_names, lines, detectors, vectorizer,
                     detector_params):
    """Score clean folder ``name`` with the other clean folders as the baseline,
    as it would be scored if it were the target. Per detector, the folder's score,
    or with ``lines`` its highest line score. None when no other clean folder has
    data here, or the detectors cannot fit that baseline."""
    train = df.filter(pl.col("folder").is_in([other for other in clean_names if other != name]))
    test = df.filter(pl.col("folder") == name)
    if train.height == 0 or test.height == 0:
        return None
    if not lines:
        train = log_root.aggregate_dataframe(train, "folder", field)
        test = log_root.aggregate_dataframe(test, "folder", field)
    try:
        scored = run_anomaly_detection(train, test, field, detectors=detectors,
                                       vectorizer=vectorizer, detector_params=detector_params)
    except ValueError as error:
        # The baseline is one folder smaller than the target's, which can be too
        # few for the detector parameters (KMeans n_clusters=3 on two folders).
        logger.debug("clean range: scoring %s left out failed: %s", name, error)
        return None
    return {column: scored.get_column(column).max()
            for _, column in DETECTORS.values() if column in scored.columns}


def clean_range(df, field, clean_names, file_name=None, lines=False, detectors=None,
                vectorizer="Count", detector_params=None, known=None, get_range=None,
                key=()):
    """How the detectors score clean runs, so a target's score can be read against
    them instead of needing a hand-picked threshold.

    Each folder of clean_sample(clean_names) is scored with the remaining clean
    folders as the baseline -- leave one out, since a baseline that includes the
    folder itself scores it as normal by construction (OOVDetector always gives
    0). A folder's value is its score, or with ``lines`` its highest line score,
    so clean_max is the worst line any sampled clean run had. ``known`` maps a
    folder name to scores the caller already has from exactly that baseline,
    which skips its fit; with target_folder="ALL" every sampled folder is
    known. ``file_name`` restricts both sides to that file. get_range(key,
    build) lets a caller cache ranges across calls; ``key`` names what ``field``
    was built from. None with fewer than MIN_CLEAN_FOLDERS clean folders.
    """
    if len(clean_names) < MIN_CLEAN_FOLDERS:
        return None
    names = clean_sample(clean_names)
    known = known or {}

    def build():
        base = df if file_name is None else df.filter(pl.col("file_name") == file_name)
        values = {}
        for name in names:
            scores = known.get(name)
            if scores is None:
                scores = _left_out_scores(base, field, name, clean_names, lines, detectors,
                                          vectorizer, detector_params)
            for column, value in (scores or {}).items():
                values.setdefault(column, []).append(value)
        rows = [{"detector": _DETECTOR_OF[column], **scoring.range_row(values[column])}
                for _, column in DETECTORS.values() if column in values]
        return pl.DataFrame(rows, schema=_RANGE_SCHEMA)

    if get_range is None:
        return build()
    params = tuple(sorted((name, tuple(sorted(value.items())))
                          for name, value in (detector_params or {}).items()))
    full_key = ("anomaly", *key, file_name, lines, vectorizer, tuple(detectors or ()), params,
                tuple(names), tuple(clean_names))
    return get_range(full_key, build)


def folders_with_file(df, file_name):
    return set(df.filter(pl.col("file_name") == file_name).get_column("folder").unique().to_list())


def clean_folders(baseline_folder_names, target_folder_names, present=None):
    """The baseline folders a clean range is formed from: those that are not
    also targets, or all of them when fewer than MIN_CLEAN_FOLDERS would remain
    -- with target_folder="ALL" every folder is a target, and each one's own
    score is then a leave-one-out score. ``present`` keeps only folders that
    have a given file."""
    baseline = [name for name in baseline_folder_names
                if present is None or name in present]
    targets = set(target_folder_names)
    clean = [name for name in baseline if name not in targets]
    return clean if len(clean) >= MIN_CLEAN_FOLDERS else baseline


def known_scores(df, baseline_folders, clean_names, scored, present=None):
    """Scores the call already has for sampled clean folders that were targets
    against exactly the other clean folders, so clean_range need not fit them
    again. ``scored`` maps a folder name to its scored rows (one row, or a
    file's lines); a folder's value is the highest score per detector.
    ``present``, the folders that have the file being scored, narrows each
    baseline the way the file-level fits do."""
    known = {}
    for name in clean_sample(clean_names):
        frame = scored.get(name)
        if frame is None or frame.height == 0:
            continue
        _, baseline = log_root.prepare_folders(df, name, baseline_folders)
        baseline = set(baseline) if present is None else set(baseline) & present
        if baseline == set(clean_names) - {name}:
            known[name] = {column: frame.get_column(column).max()
                           for _, column in DETECTORS.values() if column in frame.columns}
    return known


def _ranged_columns(results, clean):
    """(detector, score column, row) per clean range row that applies to ``results``."""
    for row in clean:
        column = DETECTORS[row["detector"]][1]
        if column in results.columns and row["clean_max"] is not None:
            yield row["detector"], column, row


def scale_to_clean_range(results, clean, per_detector=True):
    """Add ``above_clean_max`` -- how many detectors score the row above the
    highest score any sampled clean run got -- and, with ``per_detector``,
    ``<detector>_threshold_score`` per detector from scoring.threshold_score."""
    above, scaled = [], []
    for name, column, row in _ranged_columns(results, clean.iter_rows(named=True)):
        above.append((pl.col(column) > row["clean_max"]).fill_null(False).cast(pl.Int64))
        if per_detector:
            value = scoring.threshold_score(pl.col(column), row)
            value = pl.lit(None, dtype=pl.Float64) if value is None else value.cast(pl.Float64)
            scaled.append(value.alias(f"{name}_threshold_score"))
    if not above:
        return results
    return results.with_columns(*scaled, pl.sum_horizontal(above).alias("above_clean_max"))


def above_clean_max_pct(scored, clean):
    """Per detector, the percentage of ``scored``'s lines above clean_max, as
    ``<detector>_above_clean_max_pct``. ``clean`` is clean range rows as dicts."""
    if scored.height == 0:
        return {}
    return {f"{name}_above_clean_max_pct": round(
                100.0 * (scored.get_column(column) > row["clean_max"]).fill_null(False).sum()
                / scored.height, 2)
            for name, column, row in _ranged_columns(scored, clean)}
