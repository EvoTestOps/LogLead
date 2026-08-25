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
import plotly.graph_objects as go
import polars as pl
import umap

from . import log_root, scoring

# Colors and shapes are combined, so the number of groups that stay apart is the
# count of colors times the count of shapes. The colors are picked from the
# published colour-blind safe palettes (Okabe & Ito, Paul Tol, IBM) as the five
# that stay furthest apart when simulated for protanopia, deuteranopia and
# tritanopia, while keeping enough contrast against both the light and the dark
# theme. The shape carries the difference for anyone who sees no color at all.
# Kept identical to VisualLogAnalyzer's ``dash_app/utils/plots.py`` so the same
# log folders come out the same colour in both projects.
GROUP_COLORS = [
    "#33BBEE",  # blue
    "#E69F00",  # orange
    "#117733",  # green
    "#AA3377",  # magenta
    "#785EF0",  # violet
]

GROUP_SYMBOLS = [
    "circle",
    "square",
    "diamond",
    "triangle-up",
    "triangle-down",
    "cross",
    "x",
    "star",
    "pentagon",
    "hexagram",
]

# Unlike VisualLogAnalyzer every plot here has a target log folder, drawn as a
# cross so it is findable among the others. The two shapes that would be read as
# that cross are therefore kept out of the group cycle, which still leaves
# 5 x 8 = 40 groups that stay apart.
TARGET_SYMBOL = "cross"
COMPARISON_SYMBOLS = [s for s in GROUP_SYMBOLS if s not in (TARGET_SYMBOL, "x")]

# A neutral outline keeps the palest markers visible in both themes.
MARKER_OUTLINE = {"width": 1, "color": "rgba(128, 128, 128, 0.8)"}

# A rolling mean changes slowly by construction, so a moving average *is* a path
# and is drawn as one; one marker per log line only smears it into a band. Dash
# and width carry the window, widest window solid and boldest, so a family's two
# averages stay apart for a reader who cannot tell its colour from another's.
MOVING_AVERAGE_PREFIX = "moving_avg_"
MOVING_AVERAGE_DASHES = ["solid", "dash", "dot", "longdash", "dashdot"]


def _group_marker(index, size=8, symbol_index=None):
    """Colour and shape for one group.

    The colours are cycled through first and the shape changes once they run
    out, so every combination is used before any of them comes back.
    ``symbol_index`` overrides that, for callers whose shape means something of
    its own rather than "the colours ran out".
    """
    if symbol_index is None:
        symbol_index = index // len(GROUP_COLORS)
    return {
        "color": GROUP_COLORS[index % len(GROUP_COLORS)],
        "symbol": COMPARISON_SYMBOLS[symbol_index % len(COMPARISON_SYMBOLS)],
        "size": size,
        "line": dict(MARKER_OUTLINE),
    }


def _moving_average_window(col):
    """Window of a ``moving_avg_<window>_<col>`` column, or ``None`` if it is raw."""
    if not col.startswith(MOVING_AVERAGE_PREFIX):
        return None
    window = col[len(MOVING_AVERAGE_PREFIX):].split("_", 1)[0]
    return int(window) if window.isdigit() else None


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


def _add_group_traces(fig, points, target_folder, x_values, y_values, hovertemplate):
    """One marker trace per group, each with its own colour and shape.

    The target log folder is drawn as a trace of its own, in its group's colour
    but with the reserved shape at a larger size, since "group" is a shared
    colour category (e.g. all log folders of one application) and cannot tell
    the target apart from the comparison folders on its own without hovering.
    """
    folders = points["folder"].to_list()
    groups = points["group"].to_list()
    unique_groups = sorted(set(groups))
    # "all" is what group_folders_by_indices writes when nothing was grouped, so
    # using it as a legend entry would label every ungrouped plot with a word
    # that says nothing.
    ungrouped = unique_groups == ["all"]

    def add(name, marker, rows, opacity):
        if not rows:
            return
        fig.add_trace(go.Scatter(
            x=[x_values[i] for i in rows],
            y=[y_values[i] for i in rows],
            mode="markers",
            text=[folders[i] for i in rows],
            hovertemplate=hovertemplate,
            name=name,
            marker=marker,
            opacity=opacity,
        ))

    for index, group in enumerate(unique_groups):
        add(
            "Log folders" if ungrouped else group,
            _group_marker(index),
            [i for i, (folder, g) in enumerate(zip(folders, groups))
             if g == group and folder != target_folder],
            0.7,
        )

    target_rows = [i for i, folder in enumerate(folders) if folder == target_folder]
    if target_rows:
        marker = _group_marker(unique_groups.index(groups[target_rows[0]]), size=14)
        marker["symbol"] = TARGET_SYMBOL
        marker["line"] = {**MARKER_OUTLINE, "width": 2}
        add(f"target: {target_folder}", marker, target_rows, 1.0)


def _figures(points, target_folder, file, title_subject):
    """Build the UMAP and the simple scatter from the points frame."""
    title = f"{title_subject}<br>Target log folder (cross):<br>{target_folder}"
    # The trace name carries the group, so hover spells the log folder out
    # rather than showing a bare column name.
    legend_note = "<extra>%{fullData.name}</extra>"

    fig_umap = go.Figure()
    _add_group_traces(
        fig_umap, points, target_folder,
        points["umap_x"].to_list(), points["umap_y"].to_list(),
        hovertemplate="Log folder: %{text}" + legend_note,
    )
    fig_umap.update_layout(
        title=title, xaxis_title="UMAP1", yaxis_title="UMAP2", legend_title_text="Group",
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

    fig_simple = go.Figure()
    _add_group_traces(
        fig_simple, points, target_folder, x_jittered.tolist(), y_jittered.tolist(),
        hovertemplate=(
            "Log folder: %{text}<br>" + x_title + ": %{x:,.0f}<br>Lines: %{y:,.0f}"
            + legend_note
        ),
    )
    fig_simple.update_layout(
        title=title, xaxis_title=x_title, yaxis_title="Lines", yaxis_type="log",
        legend_title_text="Group",
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

    One detector family is one colour, so its raw score and its averages read as
    belonging together. Within a family the raw score is a scatter -- every point
    there is a log line someone may want to hover -- and the averages are lines.

    :param display_mode: how the *raw* per-line scores are drawn (``"markers"``,
        ``"lines"``, ``"lines+markers"``). The moving averages ignore it and are
        always lines.
    """
    measure_groups = {
        prefix: [col for col in df.columns if prefix in col]
        for prefix in ("kmeans", "IF", "RM", "OOVD")
    }
    line_numbers = df["line_number"].to_list()
    messages = df["m_message"].to_list()
    hover_text = [f"Log: {msg[:100]}<br>{msg[100:205]}" for msg in messages]

    normalized = df
    for columns in measure_groups.values():
        if columns:
            normalized = normalized.with_columns(
                scoring.normalize_measure_columns(df, columns)
            )

    scatters, lines = [], []
    for family_index, columns in enumerate(measure_groups.values()):
        columns = [col for col in columns if col in normalized.columns]
        if not columns:
            continue
        color = GROUP_COLORS[family_index % len(GROUP_COLORS)]
        # Widest window first, so the smoothest trend takes the solid, boldest line.
        windows = sorted(
            {w for w in map(_moving_average_window, columns) if w is not None},
            reverse=True,
        )
        for member_index, col in enumerate(columns):
            shared = dict(
                x=line_numbers,
                y=normalized[col].to_list(),
                name=col,
                text=hover_text,
                hoverinfo="text",
                connectgaps=False,
                # Draw order puts every line over every scatter; legendrank keeps
                # the legend grouped by detector family regardless.
                legendrank=1000 + family_index * 10 + member_index,
            )
            window = _moving_average_window(col)
            if window is None:
                # The family's own shape, so the four scatters stay apart for a
                # reader who cannot tell their colours apart.
                marker = _group_marker(family_index, size=4, symbol_index=family_index)
                # One marker per log line, so an outline would fill the gaps
                # between them into a smear.
                marker["line"] = {"width": 0}
                scatters.append(go.Scatter(
                    mode=display_mode, marker=marker, line=dict(color=color),
                    # Subordinate to the trend lines drawn over it.
                    opacity=0.55, **shared,
                ))
            else:
                rank = windows.index(window)
                lines.append(go.Scatter(
                    mode="lines",
                    line=dict(
                        color=color,
                        dash=MOVING_AVERAGE_DASHES[rank % len(MOVING_AVERAGE_DASHES)],
                        width=round(max(1.2, 2.6 - 0.7 * rank), 2),
                    ),
                    **shared,
                ))

    fig = go.Figure()
    for trace in scatters + lines:
        fig.add_trace(trace)

    fig.update_layout(
        title=title,
        xaxis_title="Line Number",
        yaxis_title="Normalized Anomaly Score (0-1)",
        template="plotly_white",
    )
    return fig
