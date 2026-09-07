"""Turning DataFrames into token-bounded tool results.

An MCP tool result goes straight into a model's context, so returning a 4000-row
table is both useless and expensive. Every analysis tool therefore returns the
same envelope: the full table on disk, and a short, *sorted* preview inline --
sorted by the column that actually answers the question, so the interesting rows
are the ones that survive truncation.

Truncation used to end the conversation with the table: the rows that survived
were the ones this module chose. They now carry a ``result_id`` instead, and the
table itself stays in the session, so the caller can ask for the rows it wants
(``query_result``).

The plot tools do not come through here at all, for the same reason: a scatter
has no top N, so they describe their axes with :func:`numeric_summary` and leave
the points to a query. That is also what a caller needs in order to pick a
threshold, having never seen the data.
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


def numeric_summary(df, columns=None):
    """Range and quartiles per numeric column: the scale a row is read against.

    A single row means nothing on its own -- "unique_terms 23" is unremarkable
    or extreme depending on the other 4,999 log folders. This is also what a
    caller picks a ``query_result`` threshold from, so it carries the tails
    (p10/p90) and not just the quartiles.
    """
    summary = {}
    for column in columns if columns is not None else df.columns:
        if column not in df.columns:
            continue
        series = df[column]
        if not series.dtype.is_numeric():
            continue
        series = series.drop_nulls()
        if series.len() == 0:
            continue
        summary[column] = {
            "min": _clean_value(series.min()),
            "p10": _clean_value(series.quantile(0.10)),
            "p25": _clean_value(series.quantile(0.25)),
            "median": _clean_value(series.median()),
            "p75": _clean_value(series.quantile(0.75)),
            "p90": _clean_value(series.quantile(0.90)),
            "max": _clean_value(series.max()),
        }
    return summary


def percentile_of(df, column, value):
    """Where ``value`` falls among ``column``, as a 0-100 percentile.

    Answers "is this number unusual here?", which is the question a target row
    is in the result to answer.
    """
    series = df[column].drop_nulls()
    if series.len() == 0 or value is None:
        return None
    return round(100.0 * (series <= value).sum() / series.len(), 1)


def query_hint(session_id, result_id, df, sorted_by=None):
    """A ready-to-run ``query_result`` call for the rows that were not shown.

    Spelled out with a real column name because a note is read at the moment the
    caller needs it, while a tool docstring was read once at registration time.
    """
    column = sorted_by
    if column is None:
        numeric = [col for col in df.columns if df[col].dtype.is_numeric()]
        column = numeric[0] if numeric else (df.columns[0] if df.columns else "column")
    return (
        f'Query the rest with query_result(session_id="{session_id}", '
        f'result_id="{result_id}", where=[["{column}", ">", <value>]]).'
    )


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
    result_id = session.stash_result(analysis, df) if session is not None else None

    payload = {
        "session_id": session.session_id if session is not None else None,
        "analysis": analysis,
        "level": level,
        "params": params,
        "n_rows": df.height,
        "sorted_by": sorted_by,
        "result_id": result_id,
        "rows": records,
        "truncated": df.height > len(records),
        "artifact": artifact,
        "notes": list(notes or []),
    }
    if payload["truncated"]:
        payload["notes"].append(
            f"Showing {len(records)} of {df.height} rows"
            + (f", sorted by {sorted_by} descending." if sorted_by else ".")
            + (" The whole table was also written to 'artifact'." if artifact else "")
        )
        if result_id is not None:
            payload["notes"].append(query_hint(session.session_id, result_id, df, sorted_by))
    if extra:
        payload.update(extra)
    return payload
