import polars as pl

from .base import BaseLoader

__all__ = ['GELFLoader']


# Process log files created with the GELF logging driver.
#
class GELFLoader(BaseLoader):
    def load(self):
        with open(self.filename, 'r') as file:
            lines = file.readlines()

        # Assuming each line in the file is a separate JSON object
        json_frames = [self.parse_json(line) for line in lines]
        self.df = pl.concat(json_frames)

    def preprocess(self):
        # Rename some columns to match the expected column names and parse datetime
        self.df = self.df.with_columns(
            pl.col("message").alias("m_message")
            ).drop("message")

        parsed_timestamps = self.df.select(
            pl.col("@timestamp").str.strptime(pl.Datetime, strict=False).alias("m_timestamp")
        ).drop("@timestamp")
        self.df = self.df.with_columns(parsed_timestamps)
