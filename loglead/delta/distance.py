"""Pairwise distance between log folders and files, and line clustering.

Three distance functions, mirroring LogDelta's config step names, and one
line-level clustering function:

* ``distance_folder_filename`` -- log folder vs log folder over *file
  names* only. Never opens a file.
* ``distance_folder_content``  -- log folder vs log folder over log *text*.
* ``distance_file_content``    -- file vs same-named file, across log folders.
* ``log_line_clustering``      -- bucket-histogram comparison of one file's
  lines against the same file in the baseline log folders.

Every function returns a ``pl.DataFrame`` and writes nothing. The measures of
the three distance functions are **distances**, so larger means more different,
and 0 means identical. A single line has no meaningful distance, so
``log_line_clustering`` instead groups lines into clusters and profiles how often
each cluster occurs on the target and the baseline side.

``distance_folder_content``/``distance_file_content`` compute cosine, jaccard
and containment per comparison by default (matrix ops on the vectors already
built for the comparison). ``compression`` (a bz2 pass over the full text) is
opt-in via ``measures``: it gets slow on large log folders and its ranking has
been unreliable, and agents rarely change defaults. ``measures`` also narrows
to a subset, run in isolation; narrowing weakens ``rank_sum``/``zscore_sum``
the same way narrowing ``detectors`` does for the anomaly tools.

``log_line_clustering`` takes ``measures`` too, but its measures are bucket
granularities (``Exact``, ``Prefix``, ``Minhash``) rather than vector distances:
``content_format`` picks the representation and a measure decides how coarsely
that representation is grouped. Each one yields its own bucket histogram, so
there is no ``rank_sum`` combining them.

``clean_range``/``filename_clean_range``/``file_content_clean_range`` give a
distance its scale without a hand-picked threshold: how much a sample of the
baseline folders differ from each other, which ``scale_to_clean_range`` then
places the target against. ``log_line_clustering`` has none; its
``target_only`` buckets are already relative to the pooled baseline folders.
"""

import logging

import numpy as np
import polars as pl

from .. import LogDistance
from ..enhancers import EventLogEnhancer
from . import log_root, scoring
from .scoring import MAX_CLEAN_FOLDERS, MIN_CLEAN_FOLDERS, clean_sample

logger = logging.getLogger(__name__)

#: distance measure name -> ``LogDistance`` method name.
DISTANCE_MEASURES = {"cosine": "cosine", "jaccard": "jaccard",
                     "compression": "compression", "containment": "containment"}

DEFAULT_MEASURES = ["cosine", "jaccard", "containment"]


def _resolve_measures(measures):
    measures = DEFAULT_MEASURES if measures is None else list(measures)
    unknown = [m for m in measures if m not in DISTANCE_MEASURES]
    if unknown:
        raise ValueError(f"Unknown measures {unknown}. Valid options: {list(DISTANCE_MEASURES)}")
    return measures


def distance_folder_filename(df, target_folder, baseline_folders="ALL"):
    """Compare log folders by which file names they contain.

    :returns: one row per baseline log folder with set overlaps, ``jaccard distance``
        and ``overlap distance``.
    """
    target_df, baseline_folder_names = log_root.prepare_folders(df, target_folder, baseline_folders)
    target_files = target_df.select("file_name").unique()

    results = []
    for other_folder in baseline_folder_names:
        other_files = df.filter(pl.col("folder") == other_folder).select("file_name").unique()
        other_series = other_files.get_column("file_name")
        target_series = target_files.get_column("file_name")

        # .implode() because polars 1.x deprecated passing a bare Series here:
        # a same-dtype collection is ambiguous between "is in this set" and an
        # element-wise comparison, and imploding says which one is meant.
        only_in_target = target_files.filter(~pl.col("file_name").is_in(other_series.implode())).height
        only_in_baseline = other_files.filter(~pl.col("file_name").is_in(target_series.implode())).height
        intersection = target_files.filter(pl.col("file_name").is_in(other_series.implode())).height
        union = pl.concat([target_files, other_files]).unique().height

        smaller = min(target_files.height, other_files.height)
        results.append({
            "target_folder": target_folder,
            "baseline_folder": other_folder,
            "files only in target": only_in_target,
            "files only in baseline": only_in_baseline,
            "union": union,
            "intersection": intersection,
            "jaccard distance": 1 - (intersection / union) if union else None,
            # LogDelta used min(folder1, folder1) here -- always 1.0 unless folder1 was
            # the smaller side. Fixed to compare against the true smaller set.
            "overlap distance": 1 - (intersection / smaller) if smaller else None,
        })

    return pl.DataFrame(results)


def distance_folder_content(
    df, target_folder, baseline_folders="ALL", mask=True,
    content_format="Words", vectorizer="Count", measures=None,
):
    """Compare log folders by their whole log text.

    :param measures: subset of :data:`DISTANCE_MEASURES` to compute. ``None``
        computes :data:`DEFAULT_MEASURES` (all but ``compression``); a measure
        left out is skipped entirely, not just hidden -- narrowing this is how one measure's own cost is isolated.
    :returns: ``(results_df, df)`` -- one row per baseline log folder with the
        requested distances plus ``zscore_sum``/``rank_sum`` over just those, and
        the (possibly enhanced) input frame so the caller can retain any newly
        computed column.
    """
    measures = _resolve_measures(measures)
    df, field = log_root.prepare_content(df, mask, content_format)
    vectorizer_class = log_root.create_vectorizer(vectorizer)
    target_df, baseline_folder_names = log_root.prepare_folders(df, target_folder, baseline_folders)

    results = []
    for other_folder in baseline_folder_names:
        other_df = df.filter(pl.col("folder") == other_folder)
        distance = LogDistance(target_df, other_df, vectorizer=vectorizer_class, field=field)
        row = {
            "target_folder": target_folder,
            "baseline_folder": other_folder,
            "target_lines": distance.size1,
            "baseline_lines": distance.size2,
        }
        for name in measures:
            row[name] = getattr(distance, DISTANCE_MEASURES[name])()
        results.append(row)

    results = scoring.add_combined_scores(results, scoring.DISTANCE_COLUMNS)
    return pl.DataFrame(results), df




def _folder_text(frame, field):
    # The same joining LogDistance does, so both paths vectorize identical text.
    if frame.schema[field] == pl.List(pl.Utf8):
        return frame.select(pl.col(field).list.join(" ").str.concat(" ")).item()
    return frame.select(pl.col(field).str.concat(" ")).item()


def _count_pair_distances(folders, field, measures):
    """Every pairwise cosine/jaccard/containment from one shared word count.

    Gives the values LogDistance gives pair by pair: a word absent from both
    folders of a pair adds zero to every count, dot product and set size, so a
    shared vocabulary changes nothing. Not true for Tfidf, whose weights depend
    on which folders were fitted. Words are counted one folder at a time with
    CountVectorizer's own tokenizer, since fitting all sampled folders at once
    holds every token of every folder in memory -- gigabytes on bgl-sized
    folders. None where both folders have no words, as LogDistance returns for
    an empty vocabulary.
    """
    from collections import Counter

    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import CountVectorizer

    size = len(folders)
    analyze = CountVectorizer().build_analyzer()
    folder_counts = [Counter(analyze(_folder_text(folder, field))) for folder in folders]
    vocabulary = {}
    rows, columns, data = [], [], []
    for row, folder_count in enumerate(folder_counts):
        for word, count in folder_count.items():
            rows.append(row)
            columns.append(vocabulary.setdefault(word, len(vocabulary)))
            data.append(count)
    if not vocabulary:
        return {measure: [[None] * size for _ in range(size)] for measure in measures}
    counts = csr_matrix((np.array(data, dtype=float), (rows, columns)),
                        shape=(size, len(vocabulary)))
    binary = (counts > 0).astype(float)
    words = np.asarray(binary.sum(axis=1)).ravel()
    shared = (binary @ binary.T).toarray()
    norms = np.sqrt(np.asarray(counts.multiply(counts).sum(axis=1)).ravel())
    dots = (counts @ counts.T).toarray()

    def pair(measure, i, j):
        if words[i] == 0 and words[j] == 0:
            return None
        if measure == "cosine":
            if norms[i] == 0 or norms[j] == 0:
                return 1.0
            return float(1 - dots[i, j] / (norms[i] * norms[j]))
        if measure == "jaccard":
            return float(1 - shared[i, j] / (words[i] + words[j] - shared[i, j]))
        smaller = min(words[i], words[j])
        return float(1 - shared[i, j] / smaller) if smaller > 0 else 1.0

    return {measure: [[pair(measure, i, j) for j in range(size)] for i in range(size)]
            for measure in measures}


def _content_pair_distances(folders, field, measures, vectorizer):
    """Every pairwise distance between ``folders``: with Count, cosine, jaccard
    and containment come from one shared word count instead of one vectorizer
    fit per pair; compression, and every measure under Tfidf, go pair by pair.
    Unordered pairs suffice: cosine, jaccard and containment are symmetric,
    compression nearly so."""
    vectorized = [m for m in measures if vectorizer == "Count" and m != "compression"]
    paired = [m for m in measures if m not in vectorized]
    values = {}
    if vectorized:
        values.update(_count_pair_distances(folders, field, vectorized))
    if paired:
        vectorizer_class = log_root.create_vectorizer(vectorizer)
        for measure in paired:
            values[measure] = [[None] * len(folders) for _ in folders]
        for i in range(len(folders)):
            for j in range(i + 1, len(folders)):
                distance = LogDistance(folders[i], folders[j],
                                       vectorizer=vectorizer_class, field=field)
                for measure in paired:
                    value = getattr(distance, DISTANCE_MEASURES[measure])()
                    values[measure][i][j] = values[measure][j][i] = value
    return values


_RANGE_SCHEMA = {"measure": pl.Utf8, "clean_min": pl.Float64,
                 "clean_mid": pl.Float64, "clean_max": pl.Float64}


def _range_frame(values, measures):
    """One clean range row per measure from a pairwise distance matrix: a
    folder's value is its median distance to the other sampled folders, the
    same statistic scale_to_clean_range takes for the target."""
    rows = []
    for measure in measures:
        size = len(values[measure])
        medians = [scoring.median([values[measure][i][j] for j in range(size) if j != i])
                   for i in range(size)]
        rows.append({"measure": measure, **scoring.range_row(medians)})
    return pl.DataFrame(rows, schema=_RANGE_SCHEMA)


def clean_range(df, baseline_folder_names, mask=True, content_format="Words",
                vectorizer="Count", measures=None, get_range=None):
    """How much clean runs differ from each other, so a target's distance can be
    read against normal variation instead of needing a hand-picked threshold.

    Formed from clean_sample of the baseline folders. get_range(key, build)
    lets a caller cache ranges across calls. None with fewer than
    MIN_CLEAN_FOLDERS baseline folders.
    """
    if len(baseline_folder_names) < MIN_CLEAN_FOLDERS:
        return None
    measures = _resolve_measures(measures)
    names = clean_sample(baseline_folder_names)

    def build():
        prepared, field = log_root.prepare_content(df, mask, content_format)
        prepared = prepared.filter(pl.col("folder").is_in(names))
        folders = [prepared.filter(pl.col("folder") == name) for name in names]
        return _range_frame(_content_pair_distances(folders, field, measures, vectorizer),
                            measures)

    if get_range is None:
        return build()
    key = ("folder_content", mask, content_format, vectorizer, tuple(measures), tuple(names))
    return get_range(key, build)


#: distance_folder_filename's measures, named as its result columns.
FILENAME_MEASURES = ["jaccard distance", "overlap distance"]


def filename_clean_range(df, baseline_folder_names, get_range=None):
    """clean_range for distance_folder_filename: how much the sampled baseline
    folders' sets of file names differ from each other."""
    if len(baseline_folder_names) < MIN_CLEAN_FOLDERS:
        return None
    names = clean_sample(baseline_folder_names)

    def build():
        files = {name: set(part.get_column("file_name").to_list()) for (name,), part in
                 df.filter(pl.col("folder").is_in(names)).select("folder", "file_name")
                 .unique().partition_by("folder", as_dict=True).items()}
        sets = [files.get(name, set()) for name in names]
        values = {measure: [[None] * len(names) for _ in names] for measure in FILENAME_MEASURES}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                shared = len(sets[i] & sets[j])
                union = len(sets[i] | sets[j])
                smaller = min(len(sets[i]), len(sets[j]))
                pair = {"jaccard distance": 1 - shared / union if union else None,
                        "overlap distance": 1 - shared / smaller if smaller else None}
                for measure, value in pair.items():
                    values[measure][i][j] = values[measure][j][i] = value
        return _range_frame(values, FILENAME_MEASURES)

    if get_range is None:
        return build()
    return get_range(("folder_filename", tuple(names)), build)


def file_content_clean_range(df, results, mask=True, content_format="Words",
                             vectorizer="Count", measures=None, get_range=None):
    """clean_range for distance_file_content, one per file name: formed from the
    baseline folders that have that file, and scaled against the target's
    same file. A file compared against fewer than MIN_CLEAN_FOLDERS baseline
    folders gets no rows. Returns the scaled table, or None when no file had
    enough baseline folders."""
    measures = _resolve_measures(measures)
    frames = []
    file_names = results.get_column("file_name").unique().to_list() if results.height else []
    for file_name in sorted(file_names):
        file_results = results.filter(pl.col("file_name") == file_name)
        baseline_folder_names = file_results.get_column("baseline_folder").to_list()
        if len(baseline_folder_names) < MIN_CLEAN_FOLDERS:
            continue
        names = clean_sample(baseline_folder_names)

        def build(file_name=file_name, names=names):
            prepared, field = log_root.prepare_content(df, mask, content_format)
            prepared = prepared.filter(pl.col("folder").is_in(names)
                                       & (pl.col("file_name") == file_name))
            folders = [prepared.filter(pl.col("folder") == name) for name in names]
            return _range_frame(_content_pair_distances(folders, field, measures, vectorizer),
                                measures)

        key = ("file_content", file_name, mask, content_format, vectorizer, tuple(measures),
               tuple(names))
        clean = build() if get_range is None else get_range(key, build)
        frames.append(scale_to_clean_range(file_results, clean)
                      .select(pl.lit(file_name).alias("file_name"), pl.all()))
    return pl.concat(frames) if frames else None


def scale_to_clean_range(results, clean):
    """Place the target against the clean range with scoring.threshold_score, taking
    the target as the median of results' distances to the baseline folders."""
    scaled = [scoring.threshold_score(scoring.median(results[row["measure"]].to_list())
                                   if row["measure"] in results.columns else None, row)
              for row in clean.iter_rows(named=True)]
    return clean.with_columns(pl.Series("threshold_score", scaled, dtype=pl.Float64))


def distance_file_content(
    df, target_folder, baseline_folders="ALL", target_files="ALL", mask=True,
    content_format="Words", vectorizer="Count", measures=None,
):
    """Compare each file against the same-named file in other log folders.

    Only files present in *both* log folders are compared. If ``target_files`` is
    given, the comparison is further restricted to that set.

    :param measures: subset of :data:`DISTANCE_MEASURES` to compute. ``None``
        computes :data:`DEFAULT_MEASURES` (all but ``compression``); a measure
        left out is skipped entirely, not just hidden -- narrowing this is how one measure's own cost is isolated.
    :returns: ``(results_df, df)`` -- one row per (file, baseline log folder),
        with the requested distances plus ``zscore_sum``/``rank_sum`` over
        just those.
    """
    measures = _resolve_measures(measures)
    df, field = log_root.prepare_content(df, mask, content_format)
    vectorizer_class = log_root.create_vectorizer(vectorizer)
    target_df, baseline_folder_names = log_root.prepare_folders(df, target_folder, baseline_folders)

    wanted = None
    if target_files != "ALL":
        wanted = set(log_root.prepare_files(target_df, target_files))

    target_names = set(target_df.get_column("file_name").unique().to_list())
    if wanted is not None:
        target_names &= wanted

    pairs, target_files_df = {}, {}
    if target_names:
        wanted_names = list(target_names)
        pairs = (df.filter(pl.col("folder").is_in(baseline_folder_names)
                           & pl.col("file_name").is_in(wanted_names))
                 .partition_by(["folder", "file_name"], as_dict=True))
        target_files_df = {key[0]: part for key, part in
                           target_df.filter(pl.col("file_name").is_in(wanted_names))
                           .partition_by("file_name", as_dict=True).items()}
    names_per_folder = {}
    for folder_name, file_name in pairs:
        names_per_folder.setdefault(folder_name, []).append(file_name)

    results = []
    # Comparison-folder order, then file name sorted within it -- the order the
    # per-folder loop produced, so the table reads the same as before.
    for other_folder in baseline_folder_names:
        for file_name in sorted(names_per_folder.get(other_folder, ())):
            # LogDelta dropped `vectorizer` here, silently always using Count.
            distance = LogDistance(
                target_files_df[file_name], pairs[(other_folder, file_name)],
                vectorizer=vectorizer_class, field=field
            )
            row = {
                "file_name": file_name,
                "target_folder": target_folder,
                "baseline_folder": other_folder,
                "target_lines": distance.size1,
                "baseline_lines": distance.size2,
            }
            for name in measures:
                row[name] = getattr(distance, DISTANCE_MEASURES[name])()
            results.append(row)

    # LogDelta recomputed this inside the comparison loop, over a growing list.
    results = scoring.add_combined_scores(results, scoring.DISTANCE_COLUMNS)
    return pl.DataFrame(results), df


#: How coarsely a line's content representation is bucketed. ``content_format``
#: picks the representation; a measure decides how coarsely it is grouped. Unlike
#: :data:`DISTANCE_MEASURES` these are not vector distances, so each one yields
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
    return {
        "n_buckets": bucket_df.height,
        "n_target_only": int(only.sum()),
        "target_only_mass": float(p[only].sum() * 100),
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
        )
        .select("measure", "bucket", "representative_line",
                "target_n", "target_pct", "baseline_n", "baseline_pct",
                "delta_pct", "target_only")
        # Target-only buckets first, then by how much of the target they hold:
        # the planted-anomaly bucket outranks the singleton noise floor.
        .sort(["target_only", "target_n", "delta_pct"], descending=[True, True, True])
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
        )
        .with_columns(
            (pl.col("target_only_lines") / pl.col("target_lines") * 100)
            .alias("target_only_pct")
        )
    )
