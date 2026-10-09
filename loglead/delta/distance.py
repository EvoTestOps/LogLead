"""Pairwise distance between log folders and files.

Three distance functions, mirroring LogDelta's config step names:

* ``distance_folder_filename`` -- log folder vs log folder over *file
  names* only. Never opens a file.
* ``distance_folder_content``  -- log folder vs log folder over log *text*.
* ``distance_file_content``    -- file vs same-named file, across log folders.

Every function returns a ``pl.DataFrame`` and writes nothing. The measures are
**distances**, so larger means more different, and 0 means identical. A single
line has no meaningful distance; :mod:`line_clustering` covers the line level.

``distance_folder_content``/``distance_file_content`` compute cosine, jaccard
and containment per comparison by default (matrix ops on the vectors already
built for the comparison). ``compression`` (a bz2 pass over the full text) is
opt-in via ``measures``: it gets slow on large log folders and its ranking has
been unreliable, and agents rarely change defaults. ``measures`` also narrows
to a subset, run in isolation; narrowing weakens ``rank_sum``/``zscore_sum``
the same way narrowing ``detectors`` does for the anomaly tools.

``clean_range``/``filename_clean_range``/``file_content_clean_range`` give a
distance its scale without a hand-picked threshold: how much a sample of the
baseline folders differ from each other, which ``scale_to_clean_range`` then
places the target against.
"""

import numpy as np
import polars as pl

from .. import LogDistance
from . import log_root, scoring
from .scoring import MAX_CLEAN_FOLDERS, MIN_CLEAN_FOLDERS, clean_sample

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
                vectorizer="Count", measures=None, get_range=None, file_name=None):
    """How much clean runs differ from each other, so a target's distance can be
    read against normal variation instead of needing a hand-picked threshold.

    Formed from clean_sample of the baseline folders, restricted to
    ``file_name`` when given. get_range(key, build) lets a caller cache ranges
    across calls. None with fewer than MIN_CLEAN_FOLDERS baseline folders.
    """
    if len(baseline_folder_names) < MIN_CLEAN_FOLDERS:
        return None
    measures = _resolve_measures(measures)
    names = clean_sample(baseline_folder_names)

    def build():
        prepared, field = log_root.prepare_content(df, mask, content_format)
        keep = pl.col("folder").is_in(names)
        if file_name is not None:
            keep = keep & (pl.col("file_name") == file_name)
        prepared = prepared.filter(keep)
        folders = [prepared.filter(pl.col("folder") == name) for name in names]
        return _range_frame(_content_pair_distances(folders, field, measures, vectorizer),
                            measures)

    if get_range is None:
        return build()
    key = ("content", file_name, mask, content_format, vectorizer, tuple(measures), tuple(names))
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
    frames = []
    file_names = results.get_column("file_name").unique().to_list() if results.height else []
    for file_name in sorted(file_names):
        file_results = results.filter(pl.col("file_name") == file_name)
        clean = clean_range(df, file_results.get_column("baseline_folder").to_list(), mask,
                            content_format, vectorizer, measures, get_range, file_name)
        if clean is not None:
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
