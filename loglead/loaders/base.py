import copy
import io
import json
import logging
import os

import polars as pl

from loglead.streaming import iter_batches, write_batches

logger = logging.getLogger(__name__)

__all__ = ['BaseLoader']


def _names(frame):
    """Column names that also work on a LazyFrame, where .columns gives a PerformanceWarning."""
    if isinstance(frame, pl.LazyFrame):
        return frame.collect_schema().names()
    return frame.columns


# Base class
class BaseLoader:
    # This csv separator should never be found.
    # We try to disable polars from doing csv splitting.
    # Instead we do it manually to get it correctly done.
    _csv_separator = "\a" 
    _mandatory_columns = ["m_message", "m_timestamp"]
    # Only set this when the log is one line-oriented file and preprocess() uses expressions
    # only. Then preprocessing any run of whole lines gives the same rows as the full file,
    # which is what lets sink() work one chunk at a time.
    supports_streaming = False
    # Size of the pieces sink() reads the log in; peak memory grows with it.
    sink_chunk_bytes = 64 << 20 #64 MB = 2^20 * 64
    # A block's lines can be spread over several chunks, so the per-chunk copies in sink()
    # skip building sequences; sink() builds them once from the written file.
    _in_chunk = False
    
    def __init__(self, filename, df=None, df_seq=None):
        self.filename = filename
        self.df = df  # Event level dataframe
        self.df_seq = df_seq  # Sequence level dataframe

    def preprocess(self):
        raise NotImplementedError

    def execute(self):
        if self.df is None:
            self.load()
        self.preprocess()
        self.check_for_nulls_and_non_utf8()
        self.check_mandatory_columns()
        self.add_ano_col()
        return self.df

    def csv_options(self):
        """How to parse the log file. load(), scan() and sink() all use it, so they read the same rows."""
        raise NotImplementedError

    def load(self):
        self.df = pl.read_csv(self.filename, **self.csv_options())

    def scan_source(self):
        """Open the raw log lazily for scan(); nothing is read until the query runs."""
        return pl.scan_csv(self.filename, **self.csv_options())

    def _line_chunks(self, chunk_bytes):
        """Read the log file in pieces that end at a line end, so no line is split between pieces."""
        with open(self.filename, "rb") as handle:
            rest = b""
            while True:
                block = handle.read(chunk_bytes)
                if not block:
                    if rest:
                        yield rest
                    return
                block = rest + block
                cut = block.rfind(b"\n")
                if cut < 0:
                    rest = block
                    continue
                yield block[:cut + 1]
                rest = block[cut + 1:]

    def _preprocessed_chunks(self, chunk_bytes):
        """Parse and preprocess the log one piece at a time, as execute() would.

        Skips the null/non-UTF-8 report: sink() makes one for the whole file afterwards, so the
        counts aren't split per piece.
        """
        options = self.csv_options()
        carry = None
        for chunk in self._line_chunks(chunk_bytes):
            lines = pl.read_csv(io.BytesIO(chunk), **options)
            if carry is not None:
                lines = pl.concat([carry, lines])
            cut = self._event_cut(lines)
            carry = lines.slice(cut)
            if cut:
                yield self._preprocess_part(lines.slice(0, cut))
        if carry is not None and carry.height:
            yield self._preprocess_part(carry)

    def _event_cut(self, lines):
        """How many of these raw lines can be preprocessed now. The rest wait for the next chunk.

        One line is one event here. Loaders that join lines into events stop before the last event,
        which may go on in the next chunk.
        """
        return lines.height

    def _preprocess_part(self, lines):
        part = copy.copy(self)
        part._in_chunk = True
        part.df = lines
        part.df_seq = None
        part.preprocess()
        part.check_mandatory_columns()
        part.add_ano_col()
        return part.df

    def scan(self):
        """Prepare the log for loading without reading it, so Polars can filter or pick columns
        before anything is in memory.

        The null/non-UTF-8 report needs the data, so it isn't made here; sink() makes it.
        Loaders that can't stream load everything with execute() first.
        """
        if not self.supports_streaming:
            logger.warning("%s cannot stream; scan() loads the whole log into memory first.",
                           type(self).__name__)
            return self.execute().lazy()
        if self.df is None:
            self.df = self.scan_source()
        self.preprocess()
        self.check_mandatory_columns()
        self.add_ano_col()
        return self.df

    def sink(self, path, seq_path=None, check=True, chunk_bytes=None):
        """Save large log files by streaming them to disk as Parquet files,
        For logs too big to hold in memory.
        The rows are the same as execute, which loads to memory.
        The log is read a chunk at a time,
        so peak memory is one chunk however big the log is. Sequences (HDFS
        blocks) are built from that file at the end, because a block's lines can be spread over
        chunks; the sequence frame is small enough to keep in memory. Loaders that can't stream
        fall back to execute() and need the whole log in memory.

        The null/non-UTF-8 report costs a second pass over the written file; check=False skips it.
        """
        if self.supports_streaming:
            # The empty frame goes through the same per-chunk preprocessing, so it has the same
            # columns and doesn't run scan(), which reads the whole log for RawLoader's 'raise'.
            empty = self._preprocess_part(self.scan_source().head(0).collect())
            try:
                write_batches(self._preprocessed_chunks(chunk_bytes or self.sink_chunk_bytes), path,
                              empty=empty)
            except BaseException:
                # A half-written file would look like a finished one.
                if os.path.exists(path):
                    os.remove(path)
                raise
        else:
            self.scan().collect().write_parquet(path)
        self.df = pl.scan_parquet(path)
        if check and self.supports_streaming:  # execute() already reported otherwise
            self.check_for_nulls_and_non_utf8(source=path)
        df_seq = self.sequences()
        if df_seq is not None:
            self.df_seq = df_seq.lazy().collect(engine="streaming")
            self._add_seq_ano_col()
            if seq_path:
                self.df_seq.write_parquet(seq_path)
        elif isinstance(self.df_seq, pl.LazyFrame):
            self.df_seq = self.df_seq.collect(engine="streaming")
            if seq_path:
                self.df_seq.write_parquet(seq_path)
        return path

    def sequences(self):
        """Build the sequence-level frame (e.g. HDFS blocks) from self.df; most loaders have none.

        Kept apart from preprocess() so sink() can build it once from the whole written file.
        """
        return None

    def add_ano_col(self):
        # Check if the 'normal' column exists
        if self.df is not None and "normal" in _names(self.df):
            # Create the 'anomaly' column by inverting the boolean values of the 'normal' column
            self.df = self.df.with_columns(pl.col("normal").not_().alias("anomaly"))
        self._add_seq_ano_col()

        # Check if the 'anomaly' column exists but no normal column
        if self.df is not None and "anomaly" in _names(self.df) and not "normal" in _names(self.df):
            # Create the 'normal' column by inverting the boolean values of the 'anomaly' column
            self.df = self.df.with_columns(pl.col("anomaly").not_().alias("normal"))
        # self._mandatory_columns = ["m_message"]

    def _add_seq_ano_col(self):
        if self.df_seq is not None and "normal" in _names(self.df_seq):
            # Create the 'anomaly' column by inverting the boolean values of the 'normal' column
            self.df_seq = self.df_seq.with_columns(pl.col("normal").not_().alias("anomaly"))

    def check_for_nulls_and_non_utf8(self, source=None):
        """Warn about columns with nulls or broken characters, so bad loader output is noticed
        before it skews results.

        With source (a parquet path) the counts are made batch by batch over that file, so a
        file too big for memory can be checked too.
        """
        issue_counts = {}  # Dictionary to store counts of both nulls and non-UTF-8 issues for each column

        # All columns in one aggregation, so each batch is scanned once.
        schema = self.df.lazy().collect_schema()
        aggs = [pl.len().alias("__rows")]
        for i, (col, dtype) in enumerate(schema.items()):
            aggs.append(pl.col(col).null_count().alias(f"__nulls{i}"))
            if dtype == pl.Utf8:  # Check non-UTF-8 only for string columns
                aggs.append(pl.col(col).str.contains("�").sum().alias(f"__utf8{i}"))
        counts = {}
        for frame in (iter_batches(source) if source is not None else [self.df]):
            row = frame.lazy().select(aggs).collect(engine="streaming").row(0, named=True)
            for key, value in row.items():
                counts[key] = counts.get(key, 0) + (value or 0)
        rows = counts.get("__rows", 0)
        for i, col in enumerate(schema):
            null_count = counts[f"__nulls{i}"]
            if null_count > 0:
                issue_counts[col] = {"nulls": null_count}
            non_utf8_count = counts.get(f"__utf8{i}") or 0
            if non_utf8_count > 0:
                if col in issue_counts:
                    issue_counts[col]["non_utf8"] = non_utf8_count
                else:
                    issue_counts[col] = {"non_utf8": non_utf8_count}

        # Log the results
        for col, issues in issue_counts.items():
            issue_types = []
            if "nulls" in issues:
                issue_types.append(f"{issues['nulls']} null")
            if "non_utf8" in issues:
                issue_types.append(f"{issues['non_utf8']} non-UTF-8 encoded")
            issue_description = " and ".join(issue_types)

            investigate = []
            if "nulls" in issues:
                investigate.append(f"<DF_NAME>.filter(pl.col('{col}').is_null())")
            if "non_utf8" in issues:
                investigate.append(f"<DF_NAME>.filter(pl.col('{col}').str.contains('�'))")

            logger.warning(
                "Column '%s' has %s values out of %d. You have 4 options: 1) do nothing and hope "
                "for the best, 2) drop the column, 3) filter out rows with %s values, "
                "4) investigate and fix your Loader or Data. To investigate: %s",
                col, issue_description, rows, issue_description, " ; ".join(investigate))

    def _log_nulls_and_non_utf8(self, prefix, sparse_reason, non_utf8_suffix=None):
        """Shared sparse-column/non-UTF-8 report for loaders where sparse columns are the expected
        shape rather than a defect (docs/logging.md - duplicated across the format-spec loaders)."""
        sparse = [(c, n) for c, n in zip(self.df.columns, self.df.null_count().row(0)) if n]
        if sparse:
            worst = sorted(sparse, key=lambda item: -item[1])[:3]
            listed = ", ".join(f"{c} ({n})" for c, n in worst)
            logger.info("%s: %d of %d columns contain nulls out of %d rows - %s Most null: %s.",
                        prefix, len(sparse), self.df.width, len(self.df), sparse_reason, listed)

        for column, dtype in self.df.schema.items():
            if dtype == pl.Utf8:
                bad = self.df.filter(pl.col(column).str.contains("�")).height
                if bad:
                    suffix = non_utf8_suffix or (
                        f"To investigate: <DF_NAME>.filter(pl.col('{column}').str.contains('�'))")
                    logger.warning("%s: column '%s' has %d non-UTF-8 encoded value(s) out of %d. %s",
                                    prefix, column, bad, len(self.df), suffix)

    def check_mandatory_columns(self):
        missing_columns = [col for col in self._mandatory_columns if col not in _names(self.df)]
        if missing_columns:
            raise ValueError(f"Missing mandatory columns: {', '.join(missing_columns)}")
                  
        if 'm_time_stamp' in self._mandatory_columns and not isinstance(self.df.column("m_time_stamp").dtype, pl.datatypes.Datetime):
            raise TypeError("Column 'm_time_stamp' is not of type Polars.Datetime")

    def _split_and_unnest(self, field_names):
        # An expression rather than Series operations, so the same code runs lazily in scan().
        self.df = self.df.select(
            pl.col("column_1").str.splitn(" ", n=len(field_names)).struct.rename_fields(field_names).alias("fields")
        ).unnest("fields")
      
    def reduce_dataframes(self, frac=0.5, random_state=42):
        # If df_sequences is present, reduce its size
        if hasattr(self, 'df_seq') and self.df_seq is not None:
            # Sample df_seq
            df_seq_temp = self.df_seq.sample(fraction=frac, seed=random_state)

            # Check if df_seq still has at least one row
            if len(df_seq_temp) == 0:
                # If df_seq is empty after sampling, randomly select one row from the original df_seq
                self.df_seq = self.df_seq.sample(n=1)
            else:
                self.df_seq = df_seq_temp
            # Update df to include only the rows that have seq_id values present in the filtered df_seq
            # .implode(): polars 1.x deprecated passing a bare Series to is_in.
            self.df = self.df.filter(pl.col("seq_id").is_in(self.df_seq["seq_id"].implode()))

            # self.df_seq = self.df_seq.sample(fraction=frac)
            # Update df to include only the rows that have seq_id values present in the filtered df_sequences
            # self.df = self.df.filter(pl.col("seq_id").is_in(self.df_seq["seq_id"]))
        else:
            # If df_sequences is not present, just reduce df
            self.df = self.df.sample(fraction=frac, seed=random_state)

        return self.df
    
    @staticmethod
    def parse_json(json_line):
        json_data = json.loads(json_line)
        return pl.DataFrame([json_data])
