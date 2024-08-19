import polars as pl

from .base import BaseLoader

__all__ = ['RawLoader']


# Processor for an arbitrary log file. One log event per line. 
class RawLoader(BaseLoader):
    def load(self):
        print(f"WARNING! You are using RawLoader. This results in dataframe with single column only titled: m_message. "
              f"Consider implementing dataset specific loader")
        self.df = pl.read_csv(self.filename, has_header=False, infer_schema_length=0, 
                              separator=self._csv_separator, ignore_errors=True)
        self.df = self.df.rename({"column_1": "m_message"})

    #No preprocessing for raw
    def preprocess(self):
        pass

    #No mandatory columns either. 
    def check_mandatory_columns(self):
        pass