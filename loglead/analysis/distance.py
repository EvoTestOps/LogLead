"""Pairwise distance between runs, files, and lines.

Four levels, mirroring LogDelta's config step names:

* **L1** ``distance_run_file``    -- run vs run over *file names* only. Never opens a file.
* **L2** ``distance_run_content`` -- run vs run over log *text*.
* **L3** ``distance_file_content``-- file vs same-named file, across runs.
* **L4** ``distance_line_content``-- line-by-line diff of one file across runs.

Every function returns a ``pl.DataFrame`` and writes nothing. All measures are
**distances**, so larger means more different, and 0 means identical.
"""

import polars as pl

from .. import LogDistance
from . import corpus, scoring


def distance_run_file(df, target_run, comparison_runs="ALL"):
    """L1: compare runs by which file names they contain.

    :returns: one row per comparison run with set overlaps, ``jaccard distance``
        and ``overlap distance``.
    """
    target_df, comparison_run_names = corpus.prepare_runs(df, target_run, comparison_runs)
    target_files = target_df.select("file_name").unique()

    results = []
    for other_run in comparison_run_names:
        other_files = df.filter(pl.col("run") == other_run).select("file_name").unique()
        other_series = other_files.get_column("file_name")
        target_series = target_files.get_column("file_name")

        only_in_target = target_files.filter(~pl.col("file_name").is_in(other_series)).height
        only_in_comparison = other_files.filter(~pl.col("file_name").is_in(target_series)).height
        intersection = target_files.filter(pl.col("file_name").is_in(other_series)).height
        union = pl.concat([target_files, other_files]).unique().height

        smaller = min(target_files.height, other_files.height)
        results.append({
            "target_run": target_run,
            "comparison_run": other_run,
            "files only in target": only_in_target,
            "files only in comparison": only_in_comparison,
            "union": union,
            "intersection": intersection,
            "jaccard distance": 1 - (intersection / union) if union else None,
            # LogDelta used min(run1, run1) here -- always 1.0 unless run1 was
            # the smaller side. Fixed to compare against the true smaller set.
            "overlap distance": 1 - (intersection / smaller) if smaller else None,
        })

    return pl.DataFrame(results)


def distance_run_content(
    df, target_run, comparison_runs="ALL", mask=True,
    content_format="Words", vectorizer="Count",
):
    """L2: compare runs by their whole log text.

    :returns: ``(results_df, df)`` -- one row per comparison run with all four
        distances plus ``zscore_sum``/``rank_sum``, and the (possibly enhanced)
        input frame so the caller can retain any newly computed column.
    """
    df, field = corpus.prepare_content(df, mask, content_format)
    vectorizer_class = corpus.create_vectorizer(vectorizer)
    target_df, comparison_run_names = corpus.prepare_runs(df, target_run, comparison_runs)

    results = []
    for other_run in comparison_run_names:
        other_df = df.filter(pl.col("run") == other_run)
        distance = LogDistance(target_df, other_df, vectorizer=vectorizer_class, field=field)
        results.append({
            "target_run": target_run,
            "comparison_run": other_run,
            "target_lines": distance.size1,
            "comparison_lines": distance.size2,
            "cosine": distance.cosine(),
            "jaccard": distance.jaccard(),
            "compression": distance.compression(),
            "containment": distance.containment(),
        })

    results = scoring.add_combined_scores(results, scoring.DISTANCE_COLUMNS)
    return pl.DataFrame(results), df


def distance_file_content(
    df, target_run, comparison_runs="ALL", target_files="ALL", mask=True,
    content_format="Words", vectorizer="Count",
):
    """L3: compare each file against the same-named file in other runs.

    Only files present in *both* runs are compared. If ``target_files`` is
    given, the comparison is further restricted to that set.

    :returns: ``(results_df, df)`` -- one row per (file, comparison run).
    """
    df, field = corpus.prepare_content(df, mask, content_format)
    vectorizer_class = corpus.create_vectorizer(vectorizer)
    target_df, comparison_run_names = corpus.prepare_runs(df, target_run, comparison_runs)

    wanted = None
    if target_files != "ALL":
        wanted = set(corpus.prepare_files(target_df, target_files))

    results = []
    for other_run in comparison_run_names:
        other_df = df.filter(pl.col("run") == other_run)
        other_names = other_df.get_column("file_name").unique()
        matching = (
            target_df.select("file_name")
            .unique()
            .filter(pl.col("file_name").is_in(other_names))
            .get_column("file_name")
            .to_list()
        )
        if wanted is not None:
            matching = [name for name in matching if name in wanted]
        if not matching:
            continue

        for file_name in sorted(matching):
            target_file_df = target_df.filter(pl.col("file_name") == file_name)
            other_file_df = other_df.filter(pl.col("file_name") == file_name)
            # LogDelta dropped `vectorizer` here, silently always using Count.
            distance = LogDistance(
                target_file_df, other_file_df, vectorizer=vectorizer_class, field=field
            )
            results.append({
                "file_name": file_name,
                "target_run": target_run,
                "comparison_run": other_run,
                "target_lines": distance.size1,
                "comparison_lines": distance.size2,
                "cosine": distance.cosine(),
                "jaccard": distance.jaccard(),
                "compression": distance.compression(),
                "containment": distance.containment(),
            })

    # LogDelta recomputed this inside the comparison loop, over a growing list.
    results = scoring.add_combined_scores(results, scoring.DISTANCE_COLUMNS)
    return pl.DataFrame(results), df


def distance_line_content(
    df, target_run, comparison_runs="ALL", target_files="ALL", mask=True,
):
    """L4: line-by-line diff of a file between the target run and other runs.

    :returns: a list of ``(file_name, comparison_run, diff_df)``. Each
        ``diff_df`` has ``line_number``, ``difference`` (``' '`` unchanged,
        ``'-'`` only in target, ``'+'`` only in comparison, ``'?'`` hint) and
        ``content``.
    """
    field = "e_message_normalized" if mask else "m_message"
    if mask and field not in df.columns:
        raise ValueError(
            "mask=True requires the corpus to have been normalized. "
            "Open the corpus with mask=True."
        )

    target_df, comparison_run_names = corpus.prepare_runs(df, target_run, comparison_runs)
    file_names = corpus.prepare_files(target_df, target_files)
    other_runs_df = df.filter(pl.col("run").is_in(comparison_run_names))

    diffs = []
    for other_run in comparison_run_names:
        other_run_df = other_runs_df.filter(pl.col("run") == other_run)
        for file_name in file_names:
            target_file_df = target_df.filter(pl.col("file_name") == file_name)
            other_file_df = other_run_df.filter(pl.col("file_name") == file_name)
            if other_file_df.height == 0:
                continue
            distance = LogDistance(target_file_df, other_file_df, field=field)
            diffs.append((file_name, other_run, distance.diff_lines()))

    return diffs


def summarize_diff(diff_df):
    """Counts per ``difference`` marker, for a compact tool result."""
    counts = (
        diff_df.group_by("difference")
        .agg(pl.len().alias("n"))
        .to_dict(as_series=False)
    )
    by_marker = dict(zip(counts["difference"], counts["n"]))
    return {
        "unchanged": by_marker.get(" ", 0),
        "only_in_target": by_marker.get("-", 0),
        "only_in_comparison": by_marker.get("+", 0),
        "hints": by_marker.get("?", 0),
        "total": diff_df.height,
    }
