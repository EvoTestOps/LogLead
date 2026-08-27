"""Turning DataFrames into token-bounded tool results.

An MCP tool result goes straight into a model's context, so returning a 4000-row
table is both useless and expensive. Every analysis tool therefore returns the
same envelope: the full table on disk, and a short, *sorted* preview inline --
sorted by the column that actually answers the question, so the interesting rows
are the ones that survive truncation.
"""

import math

import polars as pl

#: Rows returned inline unless a tool overrides it.
DEFAULT_MAX_ROWS = 25

#: Columns never worth spending context on: raw tokenizations and long
#: intermediate text that the caller can fetch with read_log_lines instead.
_NOISY_COLUMNS = (
    "e_words", "e_trigrams", "e_alphanumerics", "e_bert_emb",
    "e_words_len", "e_trigrams_len", "e_alphanumerics_len",
    "orig_file_name", "e_message_normalized",
)


def _clean_value(value):
    """Make a Polars cell JSON-safe and compact."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 6)
    if isinstance(value, (list, tuple)):
        return list(value)[:20]
    return value


def rows_to_records(df, max_rows, drop_columns=()):
    """Take the first ``max_rows`` rows as plain dicts, minus noisy columns."""
    drop = set(_NOISY_COLUMNS) | set(drop_columns)
    keep = [col for col in df.columns if col not in drop]
    subset = df.select(keep).head(max_rows)
    return [
        {key: _clean_value(value) for key, value in row.items()}
        for row in subset.iter_rows(named=True)
    ]


def sort_for_preview(df, sort_by, descending=True):
    """Sort by the first of ``sort_by`` that exists, ignoring the rest.

    Which combined score is available depends on how many detectors ran, so
    callers pass a preference list rather than a single column.
    """
    if isinstance(sort_by, str):
        sort_by = [sort_by]
    for column in sort_by or []:
        if column in df.columns:
            return df.sort(column, descending=descending, nulls_last=True), column
    return df, None


def result(
    session, analysis, level, params, df, artifact=None, max_rows=DEFAULT_MAX_ROWS,
    sort_by=None, descending=True, drop_columns=(), notes=None, extra=None,
):
    """Build the standard tool envelope around a results DataFrame.

    :param sort_by: column name or preference list to sort the preview by.
    :param artifact: path to the full table on disk, if one was written.
    :param extra: analysis-specific fields merged into the envelope.
    """
    df = df if df is not None else pl.DataFrame()
    sorted_df, sorted_by = sort_for_preview(df, sort_by, descending)
    records = rows_to_records(sorted_df, max_rows, drop_columns)

    payload = {
        "session_id": session.session_id if session is not None else None,
        "analysis": analysis,
        "level": level,
        "params": params,
        "n_rows": df.height,
        "sorted_by": sorted_by,
        "rows": records,
        "truncated": df.height > len(records),
        "artifact": artifact,
        "notes": list(notes or []),
    }
    if payload["truncated"]:
        payload["notes"].append(
            f"Showing {len(records)} of {df.height} rows"
            + (f", sorted by {sorted_by} descending." if sorted_by else ".")
            + (" Full table: see 'artifact'." if artifact else "")
        )
    if extra:
        payload.update(extra)
    return payload
