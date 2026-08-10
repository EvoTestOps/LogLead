"""Scatter-plot a *set* of log folders so the odd one out becomes visible.

Two views are produced per call:

* **UMAP** -- each log folder's document-term vector reduced to 2D. Answers "which
  log folder sits apart from the cluster?".
* **Simple** -- unique terms (or file count at L1) against line count, log-y.
  Cruder but directly interpretable, and often enough on its own: LogDelta's
  own walkthrough separated normal from anomalous Hadoop log folders at ~86% accuracy
  with a single threshold on this plot.

Unlike LogDelta these functions also return the underlying coordinates as a
DataFrame, so a caller that cannot look at the picture can still reason about
the positions.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import umap

from . import log_root, scoring


def _aggregate_folder_documents(df, field, content_format, grouped):
    """One document per log folder, plus the frame carrying its group labels."""
    cols = ["folder", field] + (["group"] if grouped else [])
    agg = [pl.col(field)] + ([pl.col("group").first()] if grouped else [])

    if content_format == "Sklearn" or content_format.startswith("Parse-"):
        folder_groups = df.select(cols).group_by("folder").agg(agg)
    elif content_format == "File":
        unique_agg = [pl.col(field).unique()] + ([pl.col("group").first()] if grouped else [])
        folder_groups = df.select(cols).group_by("folder").agg(unique_agg)
    else:
        folder_groups = df.select(cols).explode(field).group_by("folder").agg(agg)

    folder_groups = folder_groups.sort("folder")
    column_data = folder_groups.select(pl.col(field))
    if content_format == "Sklearn":
        # Sklearn mode analyses raw text, so rejoin the per-folder list of lines
        column_data = column_data.with_columns(pl.col(field).list.join(" "))
    documents = column_data.to_series().to_list()
    return folder_groups, documents


def _dtm_and_umap(documents, content_format, vectorizer_type, random_seed=None):
    """Vectorize documents and reduce to 2D.

    :returns: ``(embeddings_2d, unique_terms_per_document)``.
    """
    # Pre-tokenized input (word/trigram/template-id lists) must bypass
    # sklearn's own tokenizer; raw text ("Sklearn") must not.
    params = (
        {}
        if content_format == "Sklearn"
        else {"tokenizer": lambda x: x, "preprocessor": None,
              "token_pattern": None, "lowercase": False}
    )
    vect = log_root.create_vectorizer(vectorizer_type)(**params)
    dtm = vect.fit_transform(documents)

    reducer = umap.UMAP(random_state=random_seed) if isinstance(random_seed, int) else umap.UMAP()
    embeddings_2d = reducer.fit_transform(dtm.toarray())
    unique_terms = (dtm > 0).sum(axis=1)
    return embeddings_2d, unique_terms


def _points_frame(embeddings_2d, unique_terms, line_counts, folder_groups, grouped):
    folders = folder_groups["folder"].to_list()
    groups = (
        folder_groups.get_column("group").to_list() if grouped else ["all"] * len(folders)
    )
    return pl.DataFrame({
        "folder": folders,
        "group": groups,
        "umap_x": np.asarray(embeddings_2d)[:, 0],
        "umap_y": np.asarray(embeddings_2d)[:, 1],
        "unique_terms": np.asarray(unique_terms).ravel(),
        "lines": np.asarray(line_counts).ravel(),
    })


def _figures(points, target_folder, file, title_subject):
    """Build the UMAP and the simple scatter from the points frame."""
    folders = points["folder"].to_list()
    groups = points["group"].to_list()
    unique_groups = sorted(set(groups))
    if unique_groups == ["all"]:
        color_map = {"all": "blue"}
    else:
        color_map = {
            g: f"rgba({(i * 50) % 255}, {(i * 100) % 255}, {(i * 150) % 255}, 1)"
            for i, g in enumerate(unique_groups)
        }
    # Target gets a cross so it is findable among the circles. The target's
    # symbol label carries its full folder name, since "group" is a shared color
    # category (e.g. all log folders of one application) and cannot tell the
    # target apart from the comparison folders on its own without hovering.
    target_label = f"target: {target_folder}"
    symbols = ["comparison" if folder != target_folder else target_label for folder in folders]
    symbol_map = {"comparison": "circle", target_label: "cross"}
    title = f"{title_subject}<br>Target log folder (cross):<br>{target_folder}"
    # Plotly labels hover rows and the legend with the raw column name, so spell
    # the concept out rather than showing a bare "folder".
    labels = {"folder": "Log folder", "group": "Group", "symbol": "Role"}

    fig_umap = px.scatter(
        {"UMAP1": points["umap_x"].to_list(), "UMAP2": points["umap_y"].to_list(),
         "folder": folders, "group": groups},
        x="UMAP1", y="UMAP2", color="group", symbol=symbols,
        color_discrete_map=color_map, symbol_map=symbol_map, labels=labels,
        hover_data={"folder": True, "UMAP1": False, "UMAP2": False},
        title=title, opacity=0.7,
    )

    x_title = "Files" if file is True else "Unique terms"
    x_values = np.asarray(points["unique_terms"].to_list(), dtype=float)
    y_values = np.asarray(points["lines"].to_list(), dtype=float)

    # Log folders frequently share an exact line/term count, so identical points would
    # hide each other. Jitter proportionally, on the log scale for the log axis.
    jitter = 0.0033
    x_range = x_values.max() - x_values.min() if len(x_values) else 0.0
    x_jittered = x_values + np.random.normal(0, jitter * x_range, size=x_values.shape)
    log_y = np.log10(y_values + 1e-10)
    log_range = log_y.max() - log_y.min() if len(log_y) else 0.0
    y_jittered = 10 ** (log_y + np.random.normal(0, jitter * log_range, size=log_y.shape))

    fig_simple = px.scatter(
        {x_title: x_jittered, "Lines": y_jittered, "folder": folders, "group": groups},
        x=x_title, y="Lines", color="group", symbol=symbols,
        color_discrete_map=color_map, symbol_map=symbol_map, labels=labels,
        hover_data={"folder": True, x_title: True, "Lines": True, "group": False},
        title=title, opacity=0.7, log_y=True,
    )
    return fig_umap, fig_simple


def plot_folder(
    df, target_folder, comparison_folders="ALL", file=True, random_seed=None,
    group_by_indices=None, mask=True, content_format="Words", vectorizer="Count",
):
    """L1/L2: plot every log folder as one point.

    :param file: ``True`` describes a log folder by its file names (L1, forces
        ``content_format="File"``), ``False`` by its log text (L2).
    :param group_by_indices: underscore-separated parts of the folder name to
        colour by, e.g. ``[0, 1]``.
    :param random_seed: int makes UMAP reproducible. LogDelta accepted this
        parameter but discarded it here, so its folder-level plots moved
        between log folders.
    :returns: ``(points_df, fig_umap, fig_simple, df)``. ``points_df`` has one
        row per log folder: ``folder, group, umap_x, umap_y, unique_terms, lines``.
    """
    if file:
        content_format = "File"
    grouped = bool(group_by_indices)
    if grouped:
        df = log_root.group_folders_by_indices(df, group_by_indices)

    _, comparison_folder_names = log_root.prepare_folders(df, target_folder, comparison_folders)
    included = df.filter(pl.col("folder").is_in([target_folder] + comparison_folder_names))
    included, field = log_root.prepare_content(included, mask, content_format)

    folder_groups, documents = _aggregate_folder_documents(included, field, content_format, grouped)
    embeddings_2d, unique_terms = _dtm_and_umap(
        documents, content_format, vectorizer, random_seed
    )
    line_counts = (
        included.group_by("folder").agg(pl.len().alias("lines")).sort("folder")
        .get_column("lines").to_numpy()
    )

    points = _points_frame(embeddings_2d, unique_terms, line_counts, folder_groups, grouped)
    subject = (
        "File Name Comparison Between Log Folders" if file
        else "Log Text Comparison Between Log Folders"
    )
    fig_umap, fig_simple = _figures(points, target_folder, file, subject)
    return points, fig_umap, fig_simple, df


def plot_file_content(
    df, target_folder, comparison_folders="ALL", target_files="ALL", random_seed=None,
    group_by_indices=None, mask=True, content_format="Words", vectorizer="Count",
):
    """L3: for each target file, plot each log folder's copy of that file as one point.

    :returns: ``(per_file, df)`` where ``per_file`` is a list of
        ``(file_name, points_df, fig_umap, fig_simple)``.
    """
    grouped = bool(group_by_indices)
    if grouped:
        df = log_root.group_folders_by_indices(df, group_by_indices)

    target_df, comparison_folder_names = log_root.prepare_folders(df, target_folder, comparison_folders)
    file_names = log_root.prepare_files(target_df, target_files)
    included = df.filter(pl.col("folder").is_in([target_folder] + comparison_folder_names))
    included, field = log_root.prepare_content(included, mask, content_format)

    per_file = []
    for file_name in file_names:
        file_df = included.filter(pl.col("file_name") == file_name)
        if file_df.select("folder").n_unique() < 2:
            # UMAP needs more than one point to say anything
            continue
        folder_groups, documents = _aggregate_folder_documents(
            file_df, field, content_format, grouped
        )
        embeddings_2d, unique_terms = _dtm_and_umap(
            documents, content_format, vectorizer, random_seed
        )
        line_counts = (
            file_df.group_by("folder").agg(pl.len().alias("lines")).sort("folder")
            .get_column("lines").to_numpy()
        )
        points = _points_frame(embeddings_2d, unique_terms, line_counts, folder_groups, grouped)
        fig_umap, fig_simple = _figures(
            points, target_folder, file_name,
            f"Textual Content Comparison Between Files: {file_name}",
        )
        per_file.append((file_name, points, fig_umap, fig_simple))

    return per_file, df


def plot_line_scores(df, title, display_mode="markers"):
    """Chronological plot of L4 per-line anomaly scores.

    Each detector's raw score and its two moving averages are min-max
    normalized *as a family* so they share one 0-1 axis; across families the
    shapes stay comparable even though the raw scales are not.
    """
    measure_groups = {
        prefix: [col for col in df.columns if prefix in col]
        for prefix in ("kmeans", "IF", "RM", "OOVD")
    }
    line_numbers = df["line_number"].to_list()
    messages = df["m_message"].to_list()

    normalized = df
    for columns in measure_groups.values():
        if columns:
            normalized = normalized.with_columns(
                scoring.normalize_measure_columns(df, columns)
            )

    fig = go.Figure()
    for columns in measure_groups.values():
        for col in columns:
            if col not in normalized.columns:
                continue
            fig.add_trace(go.Scatter(
                x=line_numbers,
                y=normalized[col].to_list(),
                mode=display_mode,
                name=col,
                text=[f"Log: {msg[:100]}<br>{msg[100:205]}" for msg in messages],
                hoverinfo="text",
                connectgaps=False,
                marker=dict(symbol="x", size=4),
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Line Number",
        yaxis_title="Normalized Anomaly Score (0-1)",
        template="plotly_white",
    )
    return fig
