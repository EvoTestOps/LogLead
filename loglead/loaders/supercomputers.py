import polars as pl

from .base import BaseLoader

__all__ = ['ThuSpiLibLoader']


# Processor for the Thunderbird, Spirit and Liberty log files
class ThuSpiLibLoader(BaseLoader):
    supports_streaming = True

    def __init__(self, filename, df=None, df_seq=None, split_component=True):
        self.split_component = split_component
        super().__init__(filename, df, df_seq)

    def csv_options(self):
        # Invalid UTF-8 (Thunderbird has some) becomes U+FFFD and is reported by the non-UTF-8
        # check; no quote handling, since an unbalanced quote would otherwise swallow line ends.
        return dict(has_header=False, infer_schema_length=0, separator=self._csv_separator,
                    ignore_errors=True, encoding="utf8-lossy", quote_char=None)
    
    def preprocess(self):
        if self.split_component:
            self._split_and_unnest(["label", "timestamp", "date", "userid", "month", 
                                    "day", "time", "location", "component_pid", "m_message"])
            self._split_component_and_pid()
        else:
            self._split_and_unnest(["label", "timestamp", "date", "userid", "month", 
                                    "day", "time", "location", "m_message"])
        # parse datatime
        self.df = self.df.with_columns(m_timestamp=pl.from_epoch(pl.col("timestamp").cast(pl.Int64)))
        # Label contains multiple anomaly cases. Convert to binary
        self.df = self.df.with_columns(normal=pl.col("label").str.starts_with("-"))

    # Reason for extra processing. We want so separte pid from component and in the log file they are embedded
    # Data description
    # https://github.com/logpai/loghub/blob/master/Thunderbird/Thunderbird_2k.log_structured.csv
    def _split_component_and_pid(self):
        split = pl.col("component_pid").str.splitn("[", n=2)
        self.df = self.df.with_columns(
            split.struct.field("field_0").str.strip_chars_end(":").alias("component"),
            split.struct.field("field_1").str.strip_chars_end("]:").alias("pid"),
        )
        self.df = self.df.drop("component_pid")
        self.df = self.df.select(["label", "timestamp", "date", "userid", "month", 
                                  "day", "time", "location", "component", "pid", "m_message"])
