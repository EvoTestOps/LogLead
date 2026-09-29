import polars as pl

from .base import BaseLoader

__all__ = ['BGLLoader']


# Processor for the BGL log file
# At the moment there are 34470 null messages that are not handled by the loader
class BGLLoader(BaseLoader):
    supports_streaming = True

    def csv_options(self):
        return dict(has_header=False, infer_schema_length=0, separator=self._csv_separator,
                    ignore_errors=True, encoding="utf8-lossy", quote_char=None)

    def preprocess(self):
        self._split_and_unnest(["label", "timestamp", "date", "node", "time",
                                "noderepeat", "type", "component", "level", "m_message"])
        self.df = self.df.with_columns(normal=pl.col("label").str.starts_with("-"))  # Same format as Tb
        # Parse datatime
        self.df = self.df.with_columns(m_timestamp=pl.from_epoch(pl.col("timestamp").cast(pl.Int64)))
