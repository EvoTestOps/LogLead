from loglead.loaders.base import BaseLoader
import polars as pl

class HDFSLoader(BaseLoader):
    
    def __init__(self, filename, df=None, df_seq=None, labels_file_name=None):
        self.labels_file_name = labels_file_name
        super().__init__(filename, df, df_seq)
          
    def load(self):
        self.df = pl.read_csv(self.filename, has_header=False, infer_schema_length=0, separator=self._csv_separator)

    def preprocess(self):
        #self._split_columns()
        self._split_and_unnest(["date", "time", "id", "level", "component", "m_message"])
        self._extract_seq_id()
        self._parse_datetimes()
        #Aggregate labels to sequence dataframe info that is at BlockID level
        self.df_seq = self.df.select(pl.col("seq_id")).unique()  
        df_temp = pl.read_csv(self.labels_file_name, has_header=True)
        self.df_seq = self.df_seq.join(df_temp, left_on='seq_id', right_on="BlockId")
        self.df_seq = self.df_seq.with_columns(
            pl.col("Label").str.starts_with("Normal").alias("normal"),
        )
        self.df_seq = self.df_seq.drop("Label")

    def _extract_seq_id(self):
        #seq_id = self.df.select(pl.col("m_message").str.extract(r"blk_(-?\d+)", group_index=1).alias("seq_id"))
        seq_id = self.df.select(pl.col("m_message").str.extract(r"(blk_[-?\d]+)", group_index=1).alias("seq_id"))
        self.df = self.df.with_columns(seq_id)

    def _parse_datetimes(self):
        parsed_times = self.df.select(pl.concat_str([pl.col("date"), pl.col("time")]).alias("m_timestamp"))
        parsed_times = parsed_times.to_series().str.strptime(pl.Datetime, "%y%m%d%H%M%S")
        self.df = self.df.with_columns(parsed_times)