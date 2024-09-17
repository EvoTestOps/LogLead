import polars as pl
import glob
import os
import warnings
from .base import BaseLoader

__all__ = ['RawLoader']


# Processor for an arbitrary log file. One log event per line.
# You are using RawLoader. This results in a DataFrame with a single column only titled: 'm_message'. 
# Consider implementing a dataset-specific loader for better functionality. 
class RawLoader(BaseLoader):
    def __init__(self, filename, filename_pattern=None, min_file_size=0, strip_full_data_path=None):
        self.min_file_size = min_file_size
        self.filename_pattern = filename_pattern
        self.strip_full_data_prefix = strip_full_data_path
        super().__init__(filename)
           
    def load(self):
        force_schema = {'column_1': pl.String} # We should not need this infer_schem = False should be enough. However, it is not.
        if  self.filename_pattern: #self.nested
            queries = []
            for subdir, _, _ in os.walk(self.filename):
                #seq_id = os.path.basename(subdir)
                file_pattern = os.path.join(subdir, self.filename_pattern)
                # Iterate over all files in the subdirectory that match the given pattern
                for file in glob.glob(file_pattern):
                    if os.path.getsize(file) > self.min_file_size:
                        #These should be the default settings also in other loaders
                        q = pl.scan_csv(file, has_header=False, schema = force_schema,  infer_schema=False, quote_char=None,
                                        separator=self._csv_separator, 
                                        encoding="utf8-lossy", include_file_paths="file_name", truncate_ragged_lines=True)
                        if self.strip_full_data_prefix:
                            q = q.with_columns(pl.col("file_name").str.strip_prefix(self.strip_full_data_prefix))
                        queries.append(q)
            dataframes = pl.collect_all(queries)
            self.df = pl.concat(dataframes)

        else:
            self.df = pl.read_csv(self.filename, has_header=False, schema = force_schema, infer_schema=False, quote_char=None,
                                separator=self._csv_separator, encoding="utf8-lossy",  truncate_ragged_lines=True)
            
        self.df = self.df.rename({"column_1": "m_message"})

    #No preprocessing for raw
    def preprocess(self):
        pass

    #No mandatory columns either. 
    def check_mandatory_columns(self):
        pass