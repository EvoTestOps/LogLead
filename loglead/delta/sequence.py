"""Scoring the *order* of log lines, against the same file in other log folders.

The detectors in :mod:`anomaly` see each line, or a whole file, as a bag of
tokens, so a familiar line in an unfamiliar place goes unnoticed. The detectors
here model event order instead: each line is one event, each log file is one
sequence of events, and the baseline is the same-named file in the comparison
log folders, one training sequence per file.

* ``NEP`` -- next event prediction with an n-gram model
  (:class:`~loglead.sequence_modelling.NextEventPredictionNgram`).
  ``NEP_pred_ano_proba`` is ``1 - p``, where ``p`` is how likely the line's
  event is after the ``ngrams - 1`` events before it, relative to the most
  likely continuation. 0 means the line is the predicted one, 1 means the
  baseline never saw this n-gram.
* ``LAP`` -- lookahead pairs (:class:`~loglead.sequence_modelling.LookaheadPairs`).
  ``LAP_pred_ano_proba`` is the share of the line's pairs with the ``window``
  lines before it -- this event, that many lines after that one -- that the
  baseline never had. It tolerates variation the n-gram does not: a line out of
  place mismatches only the pairs it breaks, not every n-gram it takes part in.

Their scores are on the same 0-1 scale but mean different things, so results
also carry ``rank_sum`` and ``zscore_sum`` over both, as in :mod:`anomaly`.

A line needs to be one event, so ``content_format`` must yield one string per
line: ``Parse-<Algorithm>`` (template ids) or ``Sklearn`` (the masked message
itself as the event).
"""

import logging

import polars as pl

from ..sequence_modelling import LookaheadPairs, NextEventPredictionNgram
from . import log_root, scoring

logger = logging.getLogger(__name__)

#: detector name -> output column
DETECTORS = {
    "NEP": "NEP_pred_ano_proba",
    "LAP": "LAP_pred_ano_proba",
}

DEFAULT_DETECTORS = list(DETECTORS)

#: Score columns, for :func:`scoring.add_combined_scores`.
SEQUENCE_COLUMNS = list(DETECTORS.values())

_NULL_EVENT = "<NULL>"


def _events(df, field):
    return df.get_column(field).fill_null(_NULL_EVENT).to_list()


def _nep_scores(model, events):
    """Per-line NEP columns for one file's events, in line order."""
    preds, _, scores_abs, _, scores_nmax = model.predict_and_score(events)
    # The last n-gram predicts the end-of-sequence marker, which is no line of the file.
    n = len(events)
    return pl.DataFrame({
        "NEP_pred_ano_proba": [1.0 - p for p in scores_nmax[:n]],
        "nep_abs": scores_abs[:n],
        "nep_predict": preds[:n],
    }, schema={"NEP_pred_ano_proba": pl.Float64, "nep_abs": pl.Int64, "nep_predict": pl.Utf8})


def _lap_scores(model, events):
    """Per-line LAP columns for one file's events, in line order."""
    mismatches, scores = model.predict_and_score(events)
    # As for NEP, the last element judges the end-of-sequence marker.
    n = len(events)
    return pl.DataFrame({
        "LAP_pred_ano_proba": scores[:n],
        "lap_unseen": mismatches[:n],
    }, schema={"LAP_pred_ano_proba": pl.Float64, "lap_unseen": pl.Int64})


def sequence_line_event_prediction(
    df, target_folder, comparison_folders="ALL", target_files="ALL", detectors=None, mask=True,
    content_format="Parse-Drain", ngrams=5, window=10,
):
    """Score every line of a target file by how expected it is after the lines before it.

    :param content_format: ``Parse-<Algorithm>`` or ``Sklearn``; see the module docstring.
    :param detectors: subset of :data:`DETECTORS`. ``None`` runs both.
    :param ngrams: n-gram length for NEP: the previous ``ngrams - 1`` events predict the next.
    :param window: how many earlier lines each line is paired with for LAP.
    :returns: ``(per_file, df)`` where ``per_file`` is a list of
        ``(target_folder, file_name, scored_df)``. Each ``scored_df`` carries
        ``line_number``, the original columns, one score column per detector,
        10/100-line moving averages of each, and for NEP ``nep_abs`` (how often
        the baseline saw the line's n-gram), ``nep_predict`` (the event it
        expected) and ``nep_expected`` (a baseline line of that event), and for
        LAP ``lap_unseen`` (how many of the line's pairs the baseline never had).
    """
    detectors = DEFAULT_DETECTORS if detectors is None else list(detectors)
    unknown = [d for d in detectors if d not in DETECTORS]
    if unknown:
        raise ValueError(f"Unknown detectors {unknown}. Valid options: {DEFAULT_DETECTORS}")
    if not detectors:
        raise ValueError(f"No detectors given. Valid options: {DEFAULT_DETECTORS}")

    df, field = log_root.prepare_content(df, mask, content_format)
    if df.schema[field] != pl.Utf8:
        raise ValueError(
            f"content_format {content_format!r} gives each line a list of tokens, but sequence "
            f"scoring needs one event per line. Use 'Parse-<Algorithm>' or 'Sklearn'."
        )
    text_field = "e_message_normalized" if mask else "m_message"
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

            # One training sequence per physical file, lines kept in file order.
            train = [_events(part, field) for part in
                     baseline_lines.partition_by("orig_file_name", maintain_order=True)]
            scorers = []
            if "NEP" in detectors:
                nep = NextEventPredictionNgram(ngrams=ngrams)
                nep.create_ngram_model(train)
                scorers.append(lambda events, model=nep: _nep_scores(model, events))
            if "LAP" in detectors:
                lap = LookaheadPairs(window=window)
                lap.create_model(train)
                scorers.append(lambda events, model=lap: _lap_scores(model, events))

            parts = []
            for part in target_lines.partition_by("orig_file_name", maintain_order=True):
                events = _events(part, field)
                parts.append(pl.concat([part, *(score(events) for score in scorers)],
                                       how="horizontal"))
            scored = pl.concat(parts)
            if "NEP" in detectors:
                examples = (baseline_lines.group_by(field, maintain_order=True)
                            .agg(pl.col(text_field).first().alias("nep_expected"))
                            .rename({field: "nep_predict"}))
                scored = scored.join(examples, on="nep_predict", how="left", maintain_order="left")

            score_cols = [DETECTORS[name] for name in detectors]
            score_only = scored.select(score_cols)
            scored = scored.with_columns(scoring.moving_averages(score_only, 10))
            scored = scored.with_columns(scoring.moving_averages(score_only, 100))
            scored = scored.with_row_index("line_number")
            per_file.append((folder_name, file_name, scored))
            logger.debug("sequence_line_event_prediction: %s/%s, %d lines against %d baseline "
                         "file(s)", folder_name, file_name, scored.height, len(train))

    return per_file, df
