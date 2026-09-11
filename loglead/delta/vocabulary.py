"""New tokens: what a target log folder has that its comparison folders never had.

The baseline vocabulary is the set of distinct tokens in the comparison log
folders; a target token outside it is new. Tokens are the elements of a content
column -- ``e_words`` (the text split on spaces) unless another is named. A
``Utf8`` column such as a parser's event id is one token per line, so the same
functions list new message types.

Counted per line, new tokens are what ``OOVDetector`` scores: its score for a
line is how many of the line's tokens its training vocabulary lacks.

Every function returns Polars DataFrames and holds no state. ``get_vocabulary``
lets a caller keep baseline vocabularies between calls.
"""

import polars as pl

from . import log_root

#: Content formats whose column is not tokens: raw text, and file names.
_NOT_TOKENS = ("Sklearn", "File")


def check_content_format(content_format):
    """Raise unless ``content_format`` yields tokens: Words, 3grams, or Parse-<Algorithm>."""
    log_root.content_column(True, content_format)  # an unknown name raises here
    if content_format in _NOT_TOKENS:
        raise ValueError(
            f"content_format {content_format!r} has no tokens to compare. "
            "Use 'Words', '3grams', or 'Parse-<Algorithm>'."
        )


def _tokens(df, field):
    """``field`` as a list of tokens: a list column as it is, a string as a one-token list."""
    dtype = df.schema[field]
    if dtype == pl.List(pl.Utf8):
        return pl.col(field)
    if dtype == pl.Utf8:
        return pl.concat_list(pl.col(field))
    raise ValueError(
        f"Unsupported datatype {dtype} in field {field}. Supported: Utf8, List[Utf8]"
    )


def vocabulary(df, field="e_words", by_file_name=False):
    """Distinct tokens of ``field`` in ``df``, as a ``token`` column.

    With ``by_file_name``, distinct ``(file_name, token)`` pairs instead: one
    vocabulary per file name.
    """
    keys = ["file_name"] if by_file_name else []
    return (df.select(*keys, _tokens(df, field).alias("token"))
              .explode("token")
              .drop_nulls("token")
              .unique())


def baseline_vocabulary(df, comparison_folders, field="e_words", by_file_name=False,
                        get_vocabulary=None):
    """The :func:`vocabulary` of ``comparison_folders``, a list of log folder names.

    :param get_vocabulary: ``get_vocabulary(key, build)`` returning the
        vocabulary for ``key``, where ``build()`` computes it. ``None`` computes it.
    :raises ValueError: if ``comparison_folders`` is empty.
    """
    if not comparison_folders:
        raise ValueError(
            "No comparison log folders to build a baseline from. The target log folder "
            "is never its own baseline, so name at least one other log folder."
        )

    def build():
        return vocabulary(df.filter(pl.col("folder").is_in(list(comparison_folders))),
                          field, by_file_name)

    if get_vocabulary is None:
        return build()
    return get_vocabulary((field, tuple(sorted(comparison_folders)), by_file_name), build)


def _new_token_rows(df, vocab, field):
    """``(row, token)`` per token of ``df`` that ``vocab`` lacks; ``row`` is its row in ``df``.

    A ``vocab`` with a ``file_name`` column is matched per file name.
    """
    keys = [column for column in vocab.columns if column != "token"]
    return (df.select(*keys, _tokens(df, field).alias("token"))
              .with_row_index("row")
              .explode("token")
              .drop_nulls("token")
              .join(vocab, on=[*keys, "token"], how="anti", maintain_order="left")
              .select("row", "token"))


def annotate(df, vocab, field="e_words", column="new_tokens"):
    """``df`` plus ``column``: each row's tokens that ``vocab`` lacks, in order, repeats kept.

    A row with none gets an empty list, so the list's length is the row's
    new-token count.
    """
    per_row = (_new_token_rows(df, vocab, field)
               .group_by("row", maintain_order=True)
               .agg(pl.col("token").alias(column)))
    return (df.with_row_index("_row")
              .join(per_row.rename({"row": "_row"}), on="_row", how="left",
                    maintain_order="left")
              .with_columns(pl.col(column).fill_null(pl.lit([], dtype=pl.List(pl.Utf8))))
              .drop("_row"))


def _summarize(df, rows, message_column):
    """Collapse :func:`_new_token_rows` output to one row per token. See :func:`token_table`."""
    per_file = [column for column in ("folder", "file_name") if column in df.columns]
    located = df.select(
        "file_name",
        pl.int_range(pl.len()).over(per_file).alias("line_number"),
        pl.col(message_column).alias("sample_line"),
    ).with_row_index("row")
    return (rows.join(located.select("row", "file_name"), on="row", how="left")
                .group_by("token")
                .agg(pl.len().alias("count"),
                     pl.col("row").n_unique().alias("n_lines"),
                     pl.col("file_name").n_unique().alias("n_files"),
                     pl.col("row").min())
                .join(located, on="row", how="left")
                .drop("row")
                .sort(["count", "token"], descending=[True, False]))


def token_table(df, vocab, field="e_words", message_column="m_message"):
    """One row per token of ``df`` that ``vocab`` lacks.

    Columns: ``token``; ``count``, its occurrences; ``n_lines`` and ``n_files``
    it occurs on; and where it first occurs -- ``file_name``, ``line_number``
    (0-based within the file, as ``read_log_lines`` numbers them) and
    ``sample_line`` (that line's ``message_column``). Sorted by ``count``
    descending, then ``token``. ``df`` needs a ``file_name`` column.
    """
    return _summarize(df, _new_token_rows(df, vocab, field), message_column)


def new_token_table(df, target_folder, comparison_folders="ALL", target_files="ALL",
                    field="e_words", match_file_name=False, get_vocabulary=None,
                    message_column="m_message"):
    """Every new token in ``target_folder``, judged against ``comparison_folders``.

    :param target_folder: exact log folder name.
    :param comparison_folders: ``"ALL"``, a list, an int N, or a ``"Prefix*"``
        wildcard, resolved by :func:`log_root.prepare_folders`.
    :param target_files: which of the target's files, resolved by
        :func:`log_root.prepare_files`.
    :param match_file_name: judge each file against the same-named files of
        the comparison folders only, skipping a target file none of them has.
        Otherwise the baseline is every line of the comparison folders.
    :param get_vocabulary: see :func:`baseline_vocabulary`.
    :returns: ``(table, info)`` -- :func:`token_table`'s frame, and a dict of
        ``comparison_folders``, ``target_files``, ``skipped_files``,
        ``n_lines``, ``lines_with_new_tokens``, ``new_token_occurrences`` and
        ``baseline_vocabulary_size``.
    """
    target_df, comparison_names = log_root.prepare_folders(df, target_folder, comparison_folders)
    files = log_root.prepare_files(target_df, target_files)
    vocab = baseline_vocabulary(df, comparison_names, field, match_file_name, get_vocabulary)

    skipped = []
    if match_file_name:
        known = set(vocab.get_column("file_name").unique().to_list())
        skipped = [name for name in files if name not in known]
        files = [name for name in files if name in known]
    target_df = target_df.filter(pl.col("file_name").is_in(files))

    rows = _new_token_rows(target_df, vocab, field)
    return _summarize(target_df, rows, message_column), {
        "comparison_folders": comparison_names,
        "target_files": files,
        "skipped_files": skipped,
        "n_lines": target_df.height,
        "lines_with_new_tokens": rows.get_column("row").n_unique(),
        "new_token_occurrences": rows.height,
        "baseline_vocabulary_size": vocab.height,
    }
