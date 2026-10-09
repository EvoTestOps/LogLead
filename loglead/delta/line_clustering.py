"""Line clustering: which kinds of line a file has that the baseline does not.

A single line has no meaningful distance, so ``log_line_clustering`` groups a
file's lines into buckets and compares how often each bucket occurs in the
target and in the same file of the baseline log folders. ``target_only``
buckets are extra lines, ``baseline_only`` buckets missing ones, and a bucket on
both sides at very different rates is a frequency shift.

Its ``measures`` are bucket granularities (``Exact``, ``Prefix``, ``Minhash``)
rather than vector distances: ``content_format`` picks the representation and a
measure decides how coarsely that representation is grouped. Each one yields
its own bucket histogram, so there is no ``rank_sum`` combining them. There is
no clean range either: the buckets are already relative to the pooled baseline
folders.
"""

import logging

import numpy as np
import polars as pl

from ..enhancers import EventLogEnhancer
from . import log_root

logger = logging.getLogger(__name__)

#: How coarsely a line's content representation is bucketed. ``content_format``
#: picks the representation; a measure decides how coarsely it is grouped. Unlike
#: :data:`distance.DISTANCE_MEASURES` these are not vector distances, so each one yields
#: its own bucket histogram rather than a column combined into ``rank_sum``.
BUCKET_MEASURES = ("Exact", "Prefix", "Minhash")

#: Bucket measures to run when the caller names none, coarse first. Both are
#: near-free, so the pair runs on every call.
#:
#: ``Minhash`` is left out of the default: it is a second pass over what these
#: flagged. Measured on 94k BGL lines it costs about 2x the pair above over
#: ``content_format="Words"`` and about 10x over ``"3grams"``.
DEFAULT_BUCKET_MEASURES = ["Prefix", "Exact"]


def _resolve_bucket_measures(measures):
    measures = list(DEFAULT_BUCKET_MEASURES) if measures is None else list(measures)
    if not measures:
        raise ValueError("At least one measure is required.")
    unknown = [m for m in measures if m not in BUCKET_MEASURES]
    if unknown:
        raise ValueError(
            f"Unknown measures {unknown}. Valid options: {list(BUCKET_MEASURES)}"
        )
    return measures


def _require_tokens(schema, field, measure, content_format):
    """``Prefix`` and ``Minhash`` read a line's tokens, so they need a list column."""
    if schema[field] != pl.List(pl.Utf8):
        raise ValueError(
            f"Measure {measure!r} needs a token column, but content_format "
            f"{content_format!r} gives {field!r}, which is {schema[field]}. Use "
            f"content_format 'Words' or '3grams', or measure 'Exact', which "
            f"buckets any representation."
        )


def _minhash_column(field):
    """Where :meth:`EventLogEnhancer.minhash` writes a signature over ``field``."""
    return f"e_minhash_{field.removeprefix('e_')}"


def _bucket_label(schema, field, measure, prefix_tokens):
    """One string per row to group on, for one measure."""
    if measure == "Exact":
        # A token list is joined back into a line; a scalar -- a parser's event
        # id, the masked text -- is cast as it stands.
        if schema[field] == pl.List(pl.Utf8):
            return pl.col(field).list.join(" ")
        return pl.col(field).cast(pl.Utf8, strict=False)
    if measure == "Prefix":
        return pl.col(field).list.head(prefix_tokens).list.join(" ")
    return pl.col(_minhash_column(field))


def _divergences(bucket_df, target_lines, baseline_lines):
    """Distribution distances between the target and baseline histograms."""
    p = bucket_df.get_column("target_n").to_numpy() / target_lines
    q = bucket_df.get_column("baseline_n").to_numpy() / baseline_lines
    m = (p + q) / 2
    # 0 log 0 is 0 here, so each side contributes only over its own support.
    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where(p > 0, p * np.log2(np.divide(p, m, where=m > 0)), 0.0)
        right = np.where(q > 0, q * np.log2(np.divide(q, m, where=m > 0)), 0.0)
    only = bucket_df.get_column("target_only").to_numpy()
    missing = bucket_df.get_column("baseline_only").to_numpy()
    return {
        "n_buckets": bucket_df.height,
        "n_target_only": int(only.sum()),
        "target_only_mass": float(p[only].sum() * 100),
        "n_baseline_only": int(missing.sum()),
        "baseline_only_mass": float(q[missing].sum() * 100),
        "js_divergence": float(0.5 * left.sum() + 0.5 * right.sum()),
        "total_variation": float(0.5 * np.abs(p - q).sum()),
        "target_lines": target_lines,
        "baseline_lines": baseline_lines,
    }


def comparable_files(df, target_folder, baseline_folders="ALL", target_files="ALL"):
    """File names the target log folder shares with at least one baseline one.

    Files are matched by name across log folders, so a log root whose folders
    share no file name has nothing to compare and this is empty. Cheap enough
    to call before deciding whether to build a content representation at all.
    """
    target_df, baseline_folder_names = log_root.prepare_folders(
        df, target_folder, baseline_folders
    )
    file_names = log_root.prepare_files(target_df, target_files)
    shared = set(
        df.filter(pl.col("folder").is_in(baseline_folder_names))
        .get_column("file_name").unique().to_list()
    )
    return [name for name in file_names if name in shared]


def require_bucket_mask(mask):
    """Reject an unmasked bucket analysis, before any content is materialized.

    Callers that cache a content column have to run this *before* preparing one,
    so a rejected call cannot leave the column built from the wrong source.
    """
    if not mask:
        raise ValueError(
            "log_line_clustering requires mask=True. On raw lines almost every "
            "line is distinct, so nearly all of them fall into target-only "
            "buckets and the histogram carries no signal."
        )


def _bucket_histogram(target_df, baseline_df, label, measure):
    """One row per bucket, with both sides' share of it."""
    target = (
        target_df.select(label.alias("bucket"), "m_message")
        .group_by("bucket")
        .agg(pl.len().alias("target_n"),
             pl.col("m_message").first().alias("representative_line"))
    )
    baseline = (
        baseline_df.select(label.alias("bucket"), "m_message")
        .group_by("bucket")
        .agg(pl.len().alias("baseline_n"),
             pl.col("m_message").first().alias("_baseline_line"))
    )
    n_target, n_baseline = target_df.height, baseline_df.height

    buckets = (
        target.join(baseline, on="bucket", how="full", coalesce=True)
        .with_columns(pl.col("target_n").fill_null(0),
                      pl.col("baseline_n").fill_null(0))
        .with_columns(
            pl.lit(measure).alias("measure"),
            # A bucket only the baseline side has still needs a readable line.
            pl.coalesce("representative_line", "_baseline_line")
              .alias("representative_line"),
            (pl.col("target_n") / n_target * 100).alias("target_pct"),
            (pl.col("baseline_n") / n_baseline * 100).alias("baseline_pct"),
        )
        .with_columns(
            (pl.col("target_pct") - pl.col("baseline_pct")).alias("delta_pct"),
            (pl.col("baseline_n") == 0).alias("target_only"),
            (pl.col("target_n") == 0).alias("baseline_only"),
        )
        .select("measure", "bucket", "representative_line",
                "target_n", "target_pct", "baseline_n", "baseline_pct",
                "delta_pct", "target_only", "baseline_only")
        # Target-only buckets first, then baseline-only ones, each by how many
        # lines they hold: the planted-anomaly bucket outranks the singleton
        # noise floor, and a missing step outranks the shared buckets.
        .sort(["target_only", "baseline_only", "target_n", "baseline_n", "delta_pct"],
              descending=True)
    )
    return buckets, _divergences(buckets, n_target, n_baseline)


def log_line_clustering(
    df, target_folder, baseline_folders="ALL", target_files="ALL", mask=True,
    content_format="Words", measures=None, prefix_tokens=3, minhash_rows=4,
):
    """Compare a file's distribution of line types against the same file elsewhere.

    Every line is bucketed by a cheap hash of its content, then the target's
    bucket histogram is compared with the baseline log folders'. Buckets
    holding target lines and no baseline lines are point anomalies; buckets
    present on both sides at very different rates are distribution shifts,
    which a nearest-neighbour distance could not see at all.

    The baseline log folders are pooled into one baseline, so ``target_only``
    means "absent from every baseline log folder", not from one of them.
    ``baseline_only`` is the reverse: lines the target is missing, such as a
    step that did not run.

    :param content_format: the representation to bucket, as elsewhere. ``Prefix``
        and ``Minhash`` read a line's tokens, so they need ``"Words"`` or
        ``"3grams"``; ``Exact`` buckets any of them, a parser's event id included.
    :param measures: subset of :data:`BUCKET_MEASURES`, **coarse first**. A coarse
        measure such as ``Prefix`` absorbs benign variation and so carries a lower
        false-positive floor, but is blind to anomalies that differ only late in
        the line; ``Exact`` is never blind but is noisier. Running both is cheap,
        and the coarsest measure that flags a bucket is the strength of the
        evidence. ``None`` runs :data:`DEFAULT_BUCKET_MEASURES`.

        ``Minhash`` buckets by similarity rather than by position, so unlike
        ``Prefix`` it is not blind to a late-line anomaly and unlike ``Exact`` it
        tolerates masking that missed a parameter. It absorbs probabilistically --
        two lines share a bucket with probability ``J ** minhash_rows`` -- so a
        bucket it does not flag is weaker evidence than one a deterministic
        measure does not flag.
    :param prefix_tokens: how many leading tokens ``Prefix`` groups on.
    :param minhash_rows: min-hashes per ``Minhash`` signature. More rows means
        fewer collisions, so finer buckets.
    :returns: ``(per_file, summary_df, df)``. ``per_file`` is a list of
        ``(target_folder, file_name, bucket_df)``, one entry per file present in
        both the target and at least one baseline log folder; ``summary_df``
        has one row per (file, measure) with ``target_only_mass``,
        ``baseline_only_mass``,
        ``js_divergence`` and ``total_variation``; ``df`` is the (possibly
        enhanced) input frame, to be kept so a session avoids re-parsing.
    """
    require_bucket_mask(mask)
    measures = _resolve_bucket_measures(measures)
    if prefix_tokens < 1:
        raise ValueError(f"prefix_tokens must be >= 1, got {prefix_tokens}")
    if minhash_rows < 1:
        raise ValueError(f"minhash_rows must be >= 1, got {minhash_rows}")

    # Before materializing anything: prepare_content runs over the whole log
    # root, while the analysis only ever reads files the target shares with a
    # baseline log folder. On a log root where no name is shared -- ten slices
    # of one split file, say -- that work would buy an empty result.
    comparable = comparable_files(df, target_folder, baseline_folders, target_files)
    if not comparable:
        return [], pl.DataFrame(), df

    df, field = log_root.prepare_content(df, mask, content_format)
    # Every measure groups on tokens, Exact included -- what sets Exact apart is
    # that it never reaches *inside* the list. Exact joins whatever it is handed. 
    # Prefix slices the list and Minhash explodes it, so those 
    # need List(Utf8) and have to be checked here.
    for measure in measures:
        if measure != "Exact":
            _require_tokens(df.schema, field, measure, content_format)

    # Only content_format's column is cached; a measure is computed per call, the
    # same contract the vectorized measures keep. Minhash is the one that cannot
    # be a plain expression -- it explodes and regroups -- so it rides on a
    # working copy that is never handed back for a session to cache.
    work = df
    if "Minhash" in measures:
        work = EventLogEnhancer(df).minhash(field, rows=minhash_rows)
    labels = {measure: _bucket_label(work.schema, field, measure, prefix_tokens)
              for measure in measures}

    # After prepare_content, so these views carry the content column.
    target_df, baseline_folder_names = log_root.prepare_folders(
        work, target_folder, baseline_folders
    )
    baseline_df = work.filter(pl.col("folder").is_in(baseline_folder_names))

    per_file, summaries = [], []
    for file_name in comparable:
        target_lines = target_df.filter(pl.col("file_name") == file_name)
        baseline_lines = baseline_df.filter(pl.col("file_name") == file_name)
        # No baseline log folder has a file of this name, so there is nothing
        # to judge it against -- the same rule distance_file_content applies.
        if target_lines.height == 0 or baseline_lines.height == 0:
            logger.debug("log_line_clustering: %s/%s has no lines on one side, skipped.",
                         target_folder, file_name)
            continue

        frames = []
        for measure in measures:
            buckets, summary = _bucket_histogram(
                target_lines, baseline_lines, labels[measure], measure
            )
            frames.append(buckets)
            summaries.append({
                "target_folder": target_folder,
                "file_name": file_name,
                "measure": measure,
                "baseline_folders": " ".join(baseline_folder_names),
                **summary,
            })
        per_file.append(
            (target_folder, file_name, pl.concat(frames, how="vertical_relaxed"))
        )

    return per_file, pl.DataFrame(summaries), df


def lines_in_bucket(df, folder, file_name, bucket, measure="Prefix", mask=True,
                    content_format="Words", prefix_tokens=3, minhash_rows=4):
    """The lines behind one of :func:`log_line_clustering`'s buckets.

    A bucket row says how many lines it holds and shows one of them; this
    returns all of them. The label is recomputed rather than stored, so the
    content and measure parameters have to match the run that produced the
    bucket -- a ``Prefix`` bucket found at ``prefix_tokens=3`` does not exist
    at ``prefix_tokens=4``, and no line matches.

    :param bucket: a ``bucket`` value from a bucket table.
    :returns: ``(lines, df)``. ``lines`` carries a ``line_number`` counted
        within the file, numbered as the line-reading tools number it; ``df``
        is the (possibly enhanced) input frame, to be kept so a session avoids
        re-parsing.
    """
    if measure not in BUCKET_MEASURES:
        raise ValueError(
            f"Unknown measure {measure!r}. Valid options: {list(BUCKET_MEASURES)}"
        )
    df, field = log_root.prepare_content(df, mask, content_format)
    if measure != "Exact":
        _require_tokens(df.schema, field, measure, content_format)

    # Minhash rides on a working copy for the same reason log_line_clustering
    # keeps it off the session frame: it explodes and regroups.
    work = df
    if measure == "Minhash":
        work = EventLogEnhancer(df).minhash(field, rows=minhash_rows)
    label = _bucket_label(work.schema, field, measure, prefix_tokens)

    lines = (
        work.filter((pl.col("folder") == folder) & (pl.col("file_name") == file_name))
        .with_row_index("line_number")
        .filter(label == pl.lit(bucket))
    )
    return lines, df


def summarize_line_buckets(bucket_df):
    """Per-measure counts for one file, for a compact tool result."""
    return (
        bucket_df.group_by("measure", maintain_order=True)
        .agg(
            pl.len().alias("buckets"),
            pl.col("target_only").sum().alias("target_only_buckets"),
            pl.col("target_n").filter(pl.col("target_only")).sum()
              .alias("target_only_lines"),
            pl.col("target_n").sum().alias("target_lines"),
            pl.col("baseline_only").sum().alias("baseline_only_buckets"),
            pl.col("baseline_n").filter(pl.col("baseline_only")).sum()
              .alias("baseline_only_lines"),
            pl.col("baseline_n").sum().alias("baseline_lines"),
        )
        .with_columns(
            (pl.col("target_only_lines") / pl.col("target_lines") * 100)
            .alias("target_only_pct"),
            (pl.col("baseline_only_lines") / pl.col("baseline_lines") * 100)
            .alias("baseline_only_pct"),
        )
    )
