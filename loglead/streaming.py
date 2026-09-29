"""Tools for logs too big to fit in memory.

The usual flow: loader.sink(path) saves the parsed log to parquet;
EventLogEnhancer.from_parquet(path) enhances it batch by batch and enhancer.sink() saves the
result; AnomalyDetector trains on sample(path, ...) and scores the whole file with
predict_to_parquet(path, ...).

Pass a parquet path wherever you can: it is read one batch at a time, so memory stays flat.
A LazyFrame works too, but then Polars' streaming engine reads the parquet, and its memory
grows with the file.
"""

import logging
import os

import polars as pl

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_BATCH_SIZE", "iter_batches", "count_rows", "sample", "write_batches"]

DEFAULT_BATCH_SIZE = 200_000
_ROW = "__loglead_sample_row"


def _is_path(source):
    return isinstance(source, (str, os.PathLike))


def iter_batches(source, batch_size=DEFAULT_BATCH_SIZE, columns=None):
    """Go through a log a batch of rows at a time, so a big file never has to fit in memory.

    Each batch is one chunk because order-dependent parsers such as Drain give different
    results when map_elements runs over several chunks.
    """
    if _is_path(source):
        # One slice query per batch: its memory stays flat over a whole file, where Polars'
        # streaming scan of parquet and pyarrow's ParquetFile.iter_batches both grow with it.
        lf = pl.scan_parquet(source)
        if columns:
            lf = lf.select(columns)
        for offset in range(0, count_rows(source), batch_size):
            yield lf.slice(offset, batch_size).collect().rechunk()
        return
    if isinstance(source, pl.DataFrame):
        frame = source.select(columns) if columns else source
        for offset in range(0, frame.height, batch_size):
            yield frame.slice(offset, batch_size).rechunk()
        return
    lf = source.select(columns) if columns else source
    for batch in lf.collect_batches(chunk_size=batch_size, engine="streaming"):
        yield batch.rechunk()


def count_rows(source):
    """Count rows without loading them; for a parquet file only its metadata is read."""
    if _is_path(source):
        import pyarrow.parquet as pq
        return pq.ParquetFile(source).metadata.num_rows
    if isinstance(source, pl.DataFrame):
        return source.height
    return source.select(pl.len()).collect(engine="streaming").item()


def sample(source, n=None, fraction=None, seed=0, batch_size=DEFAULT_BATCH_SIZE):
    """Pick a random subset of rows to train on, when the whole log doesn't fit in memory.

    Pass n (about how many rows) or fraction. Rows keep their order. A row is kept when the
    hash of its row number is below a threshold, so the same seed picks the same rows every run,
    whether source is a path, DataFrame or LazyFrame. A path is read batch by batch.
    """
    if (n is None) == (fraction is None):
        raise ValueError("pass exactly one of n or fraction")
    if n is not None:
        total = count_rows(source)
        fraction = 1.0 if n >= total else n / total
    if fraction >= 1:
        return pl.read_parquet(source) if _is_path(source) else source.lazy().collect(engine="streaming")
    threshold = pl.lit(int(max(fraction, 0.0) * 2 ** 64), dtype=pl.UInt64)
    if not _is_path(source):
        return (source.lazy().with_row_index(_ROW)
                .filter(pl.col(_ROW).hash(seed) < threshold)
                .drop(_ROW)
                .collect(engine="streaming"))
    def kept():
        offset = 0
        for batch in iter_batches(source, batch_size):
            yield (batch.with_row_index(_ROW, offset=offset)
                   .filter(pl.col(_ROW).hash(seed) < threshold)
                   .drop(_ROW))
            offset += batch.height

    # Kept rows go through a temporary file: rows filtered out of a batch keep that whole batch's
    # buffers alive, so collecting them in memory would hold every batch read.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="loglead_sample_") as folder:
        path = os.path.join(folder, "sample.parquet")
        write_batches(kept(), path, empty=pl.scan_parquet(source).head(0).collect())
        return pl.read_parquet(path)


def write_batches(batches, path, empty=None):
    """Save batches of rows as one parquet file, without holding them all in memory.

    With no batches, empty is written instead, so the file still has the right columns.
    Returns the number of rows written.
    """
    import pyarrow.parquet as pq

    writer, rows = None, 0
    try:
        for batch in batches:
            table = batch.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            rows += batch.height
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        (empty if empty is not None else pl.DataFrame()).write_parquet(path)
    return rows
