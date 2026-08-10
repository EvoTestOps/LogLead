"""Pairwise distance between log folders, files, and lines.

Four levels, mirroring LogDelta's config step names:

* **L1** ``distance_folder_filename`` -- log folder vs log folder over *file
  names* only. Never opens a file.
* **L2** ``distance_folder_content``  -- log folder vs log folder over log *text*.
* **L3** ``distance_file_content``    -- file vs same-named file, across log folders.
* **L4** ``distance_line_content``    -- line-by-line diff of one file across log
  folders.

Every function returns a ``pl.DataFrame`` and writes nothing. All measures are
**distances**, so larger means more different, and 0 means identical.
"""

import polars as pl

from .. import LogDistance
from . import log_root, scoring


def distance_folder_filename(df, target_folder, comparison_folders="ALL"):
    """L1: compare log folders by which file names they contain.

    :returns: one row per comparison log folder with set overlaps, ``jaccard distance``
        and ``overlap distance``.
    """
    target_df, comparison_folder_names = log_root.prepare_folders(df, target_folder, comparison_folders)
    target_files = target_df.select("file_name").unique()

    results = []
    for other_folder in comparison_folder_names:
        other_files = df.filter(pl.col("folder") == other_folder).select("file_name").unique()
        other_series = other_files.get_column("file_name")
        target_series = target_files.get_column("file_name")

        only_in_target = target_files.filter(~pl.col("file_name").is_in(other_series)).height
        only_in_comparison = other_files.filter(~pl.col("file_name").is_in(target_series)).height
        intersection = target_files.filter(pl.col("file_name").is_in(other_series)).height
        union = pl.concat([target_files, other_files]).unique().height

        smaller = min(target_files.height, other_files.height)
        results.append({
            "target_folder": target_folder,
            "comparison_folder": other_folder,
            "files only in target": only_in_target,
            "files only in comparison": only_in_comparison,
            "union": union,
            "intersection": intersection,
            "jaccard distance": 1 - (intersection / union) if union else None,
            # LogDelta used min(folder1, folder1) here -- always 1.0 unless folder1 was
            # the smaller side. Fixed to compare against the true smaller set.
            "overlap distance": 1 - (intersection / smaller) if smaller else None,
        })

    return pl.DataFrame(results)


def distance_folder_content(
    df, target_folder, comparison_folders="ALL", mask=True,
    content_format="Words", vectorizer="Count",
):
    """L2: compare log folders by their whole log text.

    :returns: ``(results_df, df)`` -- one row per comparison log folder with all four
        distances plus ``zscore_sum``/``rank_sum``, and the (possibly enhanced)
        input frame so the caller can retain any newly computed column.
    """
    df, field = log_root.prepare_content(df, mask, content_format)
    vectorizer_class = log_root.create_vectorizer(vectorizer)
    target_df, comparison_folder_names = log_root.prepare_folders(df, target_folder, comparison_folders)

    results = []
    for other_folder in comparison_folder_names:
        other_df = df.filter(pl.col("folder") == other_folder)
        distance = LogDistance(target_df, other_df, vectorizer=vectorizer_class, field=field)
        results.append({
            "target_folder": target_folder,
            "comparison_folder": other_folder,
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
    df, target_folder, comparison_folders="ALL", target_files="ALL", mask=True,
    content_format="Words", vectorizer="Count",
):
    """L3: compare each file against the same-named file in other log folders.

    Only files present in *both* log folders are compared. If ``target_files`` is
    given, the comparison is further restricted to that set.

    :returns: ``(results_df, df)`` -- one row per (file, comparison log folder).
    """
    df, field = log_root.prepare_content(df, mask, content_format)
    vectorizer_class = log_root.create_vectorizer(vectorizer)
    target_df, comparison_folder_names = log_root.prepare_folders(df, target_folder, comparison_folders)

    wanted = None
    if target_files != "ALL":
        wanted = set(log_root.prepare_files(target_df, target_files))

    results = []
    for other_folder in comparison_folder_names:
        other_df = df.filter(pl.col("folder") == other_folder)
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
                "target_folder": target_folder,
                "comparison_folder": other_folder,
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
    df, target_folder, comparison_folders="ALL", target_files="ALL", mask=True,
):
    """L4: line-by-line diff of a file between the target log folder and others.

    :returns: a list of ``(file_name, comparison_folder, diff_df)``. Each
        ``diff_df`` has ``line_number``, ``difference`` (``' '`` unchanged,
        ``'-'`` only in target, ``'+'`` only in comparison, ``'?'`` hint) and
        ``content``.
    """
    field = "e_message_normalized" if mask else "m_message"
    if mask and field not in df.columns:
        raise ValueError(
            "mask=True requires the log root to have been normalized. "
            "Open the log root with mask=True."
        )

    target_df, comparison_folder_names = log_root.prepare_folders(df, target_folder, comparison_folders)
    file_names = log_root.prepare_files(target_df, target_files)
    other_folders_df = df.filter(pl.col("folder").is_in(comparison_folder_names))

    diffs = []
    for other_folder in comparison_folder_names:
        other_folder_df = other_folders_df.filter(pl.col("folder") == other_folder)
        for file_name in file_names:
            target_file_df = target_df.filter(pl.col("file_name") == file_name)
            other_file_df = other_folder_df.filter(pl.col("file_name") == file_name)
            if other_file_df.height == 0:
                continue
            distance = LogDistance(target_file_df, other_file_df, field=field)
            diffs.append((file_name, other_folder, distance.diff_lines()))

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
