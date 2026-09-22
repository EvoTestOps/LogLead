"""Scoring the *order* of log lines, against the same file in other log folders.

The detectors in :mod:`anomaly` see each line, or a whole file, as a bag of
tokens, so a familiar line in an unfamiliar place goes unnoticed. The detectors
here model event order instead: each line is one event, each log file is one
sequence of events, and the baseline is the same-named file in the comparison
log folders, one training sequence per file.

* ``NEP`` -- next event prediction with an n-gram model
  (:class:`~loglead.next_event_prediction.NextEventPredictionNgram`).
  ``NEP_pred_ano_proba`` is ``1 - p``, where ``p`` is how likely the line's
  event is after the ``ngrams - 1`` events before it, relative to the most
  likely continuation. 0 means the line is the predicted one, 1 means the
  baseline never saw this n-gram.

A line needs to be one event, so ``content_format`` must yield one string per
line: ``Parse-<Algorithm>`` (template ids) or ``Sklearn`` (the masked message
itself as the event).
"""

import logging

import polars as pl

from ..next_event_prediction import NextEventPredictionNgram
from . import log_root, scoring

logger = logging.getLogger(__name__)

#: detector name -> output column
DETECTORS = {
    "NEP": "NEP_pred_ano_proba",
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


def sequence_line_event_prediction(
    df, target_folder, comparison_folders="ALL", target_files="ALL", detectors=None, mask=True,
    content_format="Parse-Drain", ngrams=5,
):
    """Score every line of a target file by how expected it is after the lines before it.

    :param content_format: ``Parse-<Algorithm>`` or ``Sklearn``; see the module docstring.
    :param ngrams: n-gram length for NEP: the previous ``ngrams - 1`` events predict the next.
    :returns: ``(per_file, df)`` where ``per_file`` is a list of
        ``(target_folder, file_name, scored_df)``. Each ``scored_df`` carries
        ``line_number``, the original columns, one score column per detector,
        10/100-line moving averages of each, and for NEP ``nep_abs`` (how often
        the baseline saw the line's n-gram), ``nep_predict`` (the event it
        expected) and ``nep_expected`` (a baseline line of that event).
    """
    detectors = DEFAULT_DETECTORS if detectors is None else list(detectors)
    unknown = [d for d in detectors if d not in DETECTORS]
    if unknown:
        raise ValueError(f"Unknown detectors {unknown}. Valid options: {DEFAULT_DETECTORS}")

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
            model = NextEventPredictionNgram(ngrams=ngrams)
            model.create_ngram_model(train)
            examples = (baseline_lines.group_by(field, maintain_order=True)
                        .agg(pl.col(text_field).first().alias("nep_expected"))
                        .rename({field: "nep_predict"}))

            parts = []
            for part in target_lines.partition_by("orig_file_name", maintain_order=True):
                parts.append(pl.concat([part, _nep_scores(model, _events(part, field))],
                                       how="horizontal"))
            scored = pl.concat(parts)
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
