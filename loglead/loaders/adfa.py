from glob import glob
import os

import polars as pl

from .base import BaseLoader

__all__ = ['ADFALoader']


class ADFALoader(BaseLoader):
    """
    Note, that this dataset already consists of event IDs, so further enhancing is not required.
    Here the "filename" needs to point to the directory called "ADFA-LD" that can be extracted from the zip in:
    https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset

    """
    
    def __init__(self, filename, df=None, df_seq=None):
        super().__init__(filename, df, df_seq)
        self._mandatory_columns = ["m_message"]

    def load(self):
        m_messages = []
        seq_ids = []
        labels = []
        self.filename = self.filename.replace('\\', '/') # Windows issues


        logfiles = [y for x in os.walk(self.filename) for y in glob(os.path.join(x[0], '*.txt'))]
        for logfile in logfiles:
            if 'ADFA-LD+Syscall+List.txt' in logfile:
                continue  # Skip label file

            with open(logfile) as logsource:
                logfile_parts = logfile.strip('\n').split('/')
                
                # Determine label based on directory naming
                if 'Attack_Data_Master' in logfile:
                    label = '_'.join(logfile_parts[3].split('_')[:-1])
                else:
                    label = 'Normal'

                # Use file name as sequence identifier
                seq_id = logfile_parts[-1].replace('.txt', '')
                line_nr = 1
                for line in logsource:
                    for event_id in line.strip('\n ').split(' '):
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
        ).group_by('seq_id').agg([
            pl.col('m_message'),  # Aggregates all 'm_message' into a list
            pl.any('is_anomaly').alias('anomaly'),  # Checks if there's any label not equal to 'Normal'
            (~pl.any('is_anomaly')).alias('normal')  # Adds the opposite of 'anomaly' as 'normal'
        ])
