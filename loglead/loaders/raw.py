import polars as pl
import glob
import os

from .base import BaseLoader

__all__ = ['RawLoader']


# Processor for an arbitrary log file. One log event per line. 
class RawLoader(BaseLoader):
    def __init__(self, filename, multi_file=False):
        self.multi_file = multi_file
        super().__init__(filename)
                 
    def load(self):
        print(f"WARNING! You are using RawLoader. This results in dataframe with single column only titled: m_message. "
              f"Consider implementing dataset specific loader")
        if self.multi_file:
            queries = []
            for file in glob.glob(self.filename):
                try:         
                    q = pl.scan_csv(file, has_header=False, infer_schema_length=0, separator=self._csv_separator)
                    q = q.with_columns((pl.lit(os.path.basename(file))).alias('file_name'))
                    queries.append(q)
                except pl.exceptions.NoDataError:  # some CSV files can be empty.
                    continue
            dataframes = pl.collect_all(queries)
            self.df = pl.concat(dataframes)
        else:
            self.df = pl.read_csv(self.filename, has_header=False, infer_schema_length=0, 
                                separator=self._csv_separator, ignore_errors=True)
            
        self.df = self.df.rename({"column_1": "m_message"})

    #No preprocessing for raw
    def preprocess(self):
        pass

    #No mandatory columns either. 
    def check_mandatory_columns(self):
        pass