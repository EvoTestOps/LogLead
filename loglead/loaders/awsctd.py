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
        m_messages = []
        seq_ids = []
        labels = []

        logfiles = [y for x in os.walk(self.filename) for y in glob(os.path.join(x[0], '*.csv'))]
        for logfile in logfiles:
            with open(logfile) as logsource:
                logfile_parts = logfile.strip('\n').split('/')
                line_nr = 1
                for line in logsource:
                    line_parts = line.strip('\n').split(',')
                    label = line_parts[-1]
                    if label == "Clean":
                        label = "Normal"
                    # Use filename + incrementing id per sequence
                    seq_id = logfile_parts[-2] + '/' + logfile_parts[-1].replace('.csv', '') + '_' + str(line_nr)

                    for event_id in line_parts[:-1]:
                        # Append data to lists
                        m_messages.append(event_id)
                        seq_ids.append(seq_id)
                        labels.append(label)

                    line_nr += 1

        self.df = pl.DataFrame({
            'm_message': m_messages,
            'seq_id': seq_ids,
            'label': labels
        })

    def preprocess(self):
        # Here the sequence based dataframe is created
        self.df_seq = self.df.with_columns(
            (pl.col('label') != "Normal").alias('is_anomaly')
        ).groupby('seq_id').agg([
            pl.col('m_message'),  # Aggregates all 'event_type' into a list
            pl.any('is_anomaly').alias('anomaly'),  # Checks if there's any label not equal to 'Normal'
            (~pl.any('is_anomaly')).alias('normal')  # Adds the opposite of 'anomaly' as 'normal'
        ])
