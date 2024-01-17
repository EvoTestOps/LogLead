from loglead.loaders.base import BaseLoader
import polars as pl
import glob
import os

# Processor for the Pro log file - Not open dataset
class ProLoader(BaseLoader):
    def load(self):
        queries = []
        for file in glob.glob(self.filename):
            try:         
                q = pl.scan_csv(file, has_header=False, infer_schema_length=0, separator=self._csv_separator)
                q = q.with_columns((pl.lit(os.path.basename(file))).alias('seq_id'))
                queries.append(q)
            except pl.exceptions.NoDataError: # some CSV files can be empty.
                continue
        dataframes = pl.collect_all(queries)
        self.df = pl.concat(dataframes)

    def preprocess(self):
        self._remove_extra_spaces()
        #Store seq_id from files
        df_seq_id = self.df.select(pl.col("seq_id"))
        self._split_and_unnest(["count", "date", "time", "system", "nr1", "nr2", "log_level", "m_message"])
        #Concat seq_id to main dataframe
        self.df = pl.concat([self.df, df_seq_id], how="horizontal")  
        self._parse_datetimes()
        #Dataframe for aggrating to sequence level
        self.df_seq = self.df.select(pl.col("seq_id")).unique()   
        #Filename holds the anomaly info. 
        self.df_seq = self.df_seq.with_columns(
            pl.col("seq_id").str.starts_with("success").alias("normal"),
        )

    #Needed because multiple spaces and no regex support
    #https://github.com/pola-rs/polars/issues/4819
    def _remove_extra_spaces(self):
        #df2 = self.df.with_columns(pl.col("column_1").str.extract_all(r"\S+").list.join(" ")) #Below is faster
        df2 = self.df.with_columns(pl.col("column_1").str.replace_all(r"\s+", " "))         
        self.df = df2

    def _parse_datetimes(self):
        parsed_times = self.df.select(pl.concat_str([pl.col("date"), pl.col("time")]).alias("m_timestamp"))
        parsed_times = parsed_times.to_series().str.strptime(pl.Datetime, "%d.%m.%Y%H:%M:%S%.3f")
        self.df = self.df.with_columns(parsed_times)
