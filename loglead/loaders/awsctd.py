from glob import glob
import logging
import os

import polars as pl

from .base import BaseLoader

logger = logging.getLogger(__name__)

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
            logger.warning("AWSCTDLoader: no valid data files processed.")

    def preprocess(self):
        if self.df_seq is not None:
            # Split 'm_message' into an array of items
            self.df_seq = self.df_seq.with_columns(
                pl.col('m_message').str.split(",")
            )

            # Vectorized label/message extraction. The trailing element of each line is the
            # label; everything before it is the syscall-id sequence. Avoiding map_elements here
            # matters because this dataset explodes to ~175M event rows (self.df below) - a
            # Python-level UDF over the pre-explode column was materializing multiple full extra
            # copies of the data and OOM-killing the process on machines with ~15GB RAM.
            msg_len = pl.col('m_message').list.len()
            self.df_seq = self.df_seq.with_columns(
                pl.when(msg_len > 0).then(pl.col('m_message').list.last()).otherwise(None).alias('label')
            )
            self.df_seq = self.df_seq.with_columns(
                pl.when(msg_len > 1)
                .then(pl.col('m_message').list.slice(0, msg_len - 1))
                .otherwise(None)
                .alias('m_message')
            )
            self.df_seq = self.df_seq.with_columns(
                pl.when(pl.col('label') == "Clean").then(pl.lit("Normal")).otherwise(pl.col('label')).alias('label')
            )

            # seq_id/label get broadcast to every exploded row below (~175M of them from ~590k
            # lines); dictionary-encoding them first keeps that from ballooning memory the way
            # repeated plain strings would.
            self.df_seq = self.df_seq.with_columns(
                pl.col('seq_id').cast(pl.Categorical),
                pl.col('label').cast(pl.Categorical),
            )

            # Explode through a dictionary-encoded copy of the list column: the ~175M-row
            # intermediate this produces never needs a full Utf8 buffer, only small integer
            # codes into the (~few hundred value) dictionary. Cast back to Utf8 afterwards since
            # AnomalyDetector._prepare_data only accepts Utf8 / List[Utf8] for this column, and
            # leave self.df_seq's own m_message as List(Utf8) for the same reason.
            self.df = (
                self.df_seq
                .with_columns(pl.col('m_message').cast(pl.List(pl.Categorical)))
                .explode('m_message')
                .with_columns(pl.col('m_message').cast(pl.Utf8))
            )

            # Create a 'normal' column that is True where label is 'Normal', otherwise False
            self.df_seq = self.df_seq.with_columns(
                (pl.col('label') == "Normal").alias('normal'),
            )
            self.df_seq = self.df_seq.with_columns(
                (~pl.col('normal')).alias('anomaly')
            )

        else:
            logger.warning("AWSCTDLoader: DataFrame is empty, no data to process.")


