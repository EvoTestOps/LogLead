import polars as pl

from .base import BaseLoader

__all__ = ['HDFSLoader']


class HDFSLoader(BaseLoader):
    
    def __init__(self, filename, df=None, df_seq=None, labels_file_name=None):
        self.labels_file_name = labels_file_name
        super().__init__(filename, df, df_seq)
          
    supports_streaming = True

    def csv_options(self):
        return dict(has_header=False, infer_schema_length=0, separator=self._csv_separator)

    def preprocess(self):
        # self._split_columns()
        self._split_and_unnest(["date", "time", "id", "level", "component", "m_message"])
        self._extract_seq_id()
        self._parse_datetimes()
        # Aggregate labels to sequence dataframe info that is at BlockID level
        self.df_seq = None if self._in_chunk else self.sequences()

    def sequences(self):
        """One row per HDFS block with its label from the labels file; HDFS is labeled per block."""
        df_seq = self.df.select(pl.col("seq_id")).unique()
        df_temp = pl.read_csv(self.labels_file_name, has_header=True)
        if isinstance(df_seq, pl.LazyFrame):
            df_temp = df_temp.lazy()
        df_seq = df_seq.join(df_temp, left_on='seq_id', right_on="BlockId")
        df_seq = df_seq.with_columns(
            pl.col("Label").str.starts_with("Normal").alias("normal"),
        )
        return df_seq.drop("Label")

    def _extract_seq_id(self):
        # seq_id = self.df.select(pl.col("m_message").str.extract(r"blk_(-?\d+)", group_index=1).alias("seq_id"))
        self.df = self.df.with_columns(
            pl.col("m_message").str.extract(r"(blk_[-?\d]+)", group_index=1).alias("seq_id"))

    def _parse_datetimes(self):
        self.df = self.df.with_columns(
            pl.concat_str([pl.col("date"), pl.col("time")]).str.strptime(pl.Datetime, "%y%m%d%H%M%S")
            .alias("m_timestamp"))
