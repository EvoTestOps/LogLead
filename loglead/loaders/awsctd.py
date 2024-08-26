from glob import glob
import os

import polars as pl

from .base import BaseLoader

__all__ = ['AWSCTDLoader']


class AWSCTDLoader(BaseLoader):
    """
    Note, that this dataset already consists of event IDs, so further enhancing is not required.
    Here the "filename" needs to point to the directory called "CSV" that can be extracted from the 7z-file in:
    https://github.com/DjPasco/AWSCTD
    """

    def __init__(self, filename, df=None, df_seq=None):
        super().__init__(filename, df, df_seq)
        self._mandatory_columns = ["m_message"]
    
    def load(self):
        queries = []
        # Walk through the directory and find all CSV files
        for subdir, _, _ in os.walk(self.filename):
            for file in glob(os.path.join(subdir, '*.csv')):
                seq_id_base = os.path.basename(subdir) + '/' + os.path.basename(file).replace('.csv', '')
                
                q = pl.scan_csv(file, has_header=False, infer_schema_length=0, separator='\n', new_columns=['m_message'])

                q = q.with_columns(
                    pl.lit(seq_id_base).alias('seq_id')  # Use the directory and file name as seq_id
                )
                queries.append(q)

        # Collect and concatenate all queries if any
        if queries:
            self.df_seq = pl.concat(pl.collect_all(queries))
            self.df = self.df_seq # Saving this here just in case it's mandatory somewhere, but the actual df is created in preprocessing
        else:
            print("No valid data files processed.")

    def preprocess(self):
        if self.df_seq is not None:
            # Split 'm_message' into an array of items
            self.df_seq = self.df_seq.with_columns(
                pl.col('m_message').str.split(",")
            )

            self.df_seq = self.df_seq.with_columns(
                pl.col('m_message').map_elements(lambda x: x[-1] if len(x) > 0 else None, return_dtype=pl.String).alias('label')
            )
            self.df_seq = self.df_seq.with_columns(
                pl.col('m_message').map_elements(lambda x: x[:-1] if len(x) > 1 else None, return_dtype=pl.List(pl.String))
            )
            self.df_seq = self.df_seq.with_columns(
                pl.col('label').map_elements(lambda label: "Normal" if label == "Clean" else label, return_dtype=pl.String).alias('label')
            )

            # Explode the 'split_message' while retaining 'seq_id' and 'label' for each exploded item
            self.df = self.df_seq.explode('m_message')

            # Create a 'normal' column that is True where label is 'Normal', otherwise False
            self.df_seq = self.df_seq.with_columns(
                (pl.col('label') == "Normal").alias('normal'),
            )
            self.df_seq = self.df_seq.with_columns(
                (~pl.any('normal')).alias('anomaly')
            )

        else:
            print("DataFrame is empty, no data to process.")


