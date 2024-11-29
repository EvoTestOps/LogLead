import polars as pl
import glob
import os
import warnings
from .base import BaseLoader

__all__ = ['RawLoader']
# Processor for an arbitrary log file. One log event per line.
# You are using RawLoader. This results in a DataFrame with a single column only titled: 'm_message'. 
# Consider implementing a dataset-specific loader for better functionality. 

"""
RawLoader Class

The RawLoader class is a specialized data loader an arbitrary log file. One log event per line.
You might want consider implementing a logfile-specific loader for better functionality, but this class will get you quickly started.   

- filename (str): Path to the main file or directory to be loaded.
- filename_pattern (str, optional): Pattern for matching files in nested directories, allowing for flexible file selection within a directory structure.
- min_file_size (int, optional): Minimum file size (in bytes) required for files to be loaded, filtering out small or empty files.
- strip_full_data_path (str, optional): Prefix to be removed from file paths shown in the dataframe.
- timestamp_pattern (str, optional): Regex pattern for identifying timestamps within each log message, required if timestamp processing is desired.
- missing_timestamp_action (str, optional): Action to take when timestamps are missing ('drop', 'merge', 'keep', 'fill-lastseen').
- timestamp_format (str, optional): Format for parsing timestamps, required if timestamp_pattern is specified.

Key Methods:
1. load(): Loads data files based on the specified filename and filename_pattern, applying filtering by file size. If nested directory patterns are specified, only files that match the pattern and minimum size are loaded.
2. preprocess(): Processes log messages for timestamp extraction based on timestamp_pattern and timestamp_format. This method ensures both parameters are specified together if provided.
3. _parse_timestamp(): A private method for parsing and formatting timestamps within the log data. Supports several strategies for handling missing timestamps, including filling with the last seen timestamp, merging multi-line entries, and dropping lines without timestamps.
4. check_mandatory_columns(): Placeholder method for checking mandatory columns, which is intentionally left empty as there are no required columns in this loader.

Timestamp Handling Strategies:
- 'drop': Removes rows without timestamps.
- 'keep': Retains rows without timestamps.
- 'fill-lastseen': Replaces missing timestamps with the most recent valid timestamp.
- 'merge': Groups and merges multi-line log entries without timestamps, creating a single entry with the earliest timestamp for the group.
"""


class RawLoader(BaseLoader):
    def __init__(self, filename, filename_pattern=None, min_file_size=0, strip_full_data_path=None, 
        timestamp_pattern=None, timestamp_format=None, missing_timestamp_action='keep',date_from_files=False):

        self.min_file_size = min_file_size
        self.filename_pattern = filename_pattern
        self.strip_full_data_prefix = strip_full_data_path
        self.timestamp_pattern = timestamp_pattern  # Optional parameter for timestamp extraction
        self.missing_timestamp_action = missing_timestamp_action  # Options: 'drop', 'merge', 'keep' 'fill-lastseen'
        self.timestamp_format = timestamp_format # Optional parameter for timestamp format 
        self.timestamp_date_from_files = date_from_files
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
                            q = q.with_columns(
                                pl.col("file_name").alias("orig_file_name"),
                                pl.col("file_name").str.strip_prefix(self.strip_full_data_prefix).alias("file_name"))
                        queries.append(q)
             # Check if any valid queries were collected
            if not queries:
                raise ValueError(f"No valid files found matching pattern {self.filename_pattern} "
                 f"in directory {os.path.abspath(self.filename)}. "
                 f"Ensure the pattern is correct and the files are large enough to process.")


            dataframes = pl.collect_all(queries)
            self.df = pl.concat(dataframes)

        else:
            self.df = pl.read_csv(self.filename, has_header=False, schema = force_schema, infer_schema=False, quote_char=None,
                                separator=self._csv_separator, encoding="utf8-lossy",  truncate_ragged_lines=True)
            
        self.df = self.df.rename({"column_1": "m_message"})


    #Time stamp preprocessing support if pattern given. 
    def preprocess(self):

        if self.timestamp_pattern and self.timestamp_format:
            self._parse_timestamp()
        # Check if only one of the two is specified
        elif (self.timestamp_pattern is None) != (self.timestamp_format is None):
            # Inform the user that both must be specified if one is provided
            missing_info = "timestamp_format" if self.timestamp_pattern else "timestamp_pattern"
            raise ValueError(f"Both timestamp_pattern and timestamp_format must be specified together. Missing: {missing_info}.")
        
        if self.timestamp_date_from_files:
            self._collect_date_from_files()

    def _collect_date_from_files(self):
            from datetime import datetime, timezone
            unique_files = self.df.select("orig_file_name").unique().to_series().to_list()
             # Convert timestamps to strings in a consistent format first
            file_dates = {
                file: datetime.fromtimestamp(os.path.getmtime(file), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") 
                for file in unique_files
            }
            
            # Add modification date as a new column and convert to timestamp
            self.df = self.df.with_columns([
                pl.col("orig_file_name")
                .replace(file_dates)
                .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
                .alias("file_date"),
                
                # Convert m_timestamp to time type for comparison
                pl.col("m_timestamp").cast(pl.Time).alias("m_timestamp_time")
            ])

            self.df = self.df.with_columns([
                pl.when(
                    # When log time is before file time and the difference crosses midnight
                    (pl.col("file_date").dt.time() < pl.col("m_timestamp_time")) &
                    (pl.col("m_timestamp_time") > pl.time(hour=23))
                )
                .then(
                    # Use previous day from file_date
                    pl.col("file_date").dt.offset_by("-1d")
                )
                .otherwise(pl.col("file_date"))
                .alias("file_date")
            ])

            # Combine file_date with m_timestamp
            self.df = self.df.with_columns([
                pl.col("file_date").dt.combine(pl.col("m_timestamp_time")).alias("m_timestamp")
            ])
            self.df = self.df.drop("orig_file_name")

    def _parse_timestamp(self):
        # Extract the timestamp to own column
        self.df = self.df.with_columns([
            pl.col("m_message").str.extract(self.timestamp_pattern, group_index=1).alias("timestamp_str"),
            pl.col("m_message").str.replace(self.timestamp_pattern, '').alias("m_message")
            #pl.col("m_message").str.replace_first(self.timestamp_pattern, '').str.strip().alias("m_message")
        ])
        
        #parse the string timestamp to actual timestamp
        self.df = self.df.with_columns(
            pl.col("timestamp_str").str.strptime(pl.Datetime, self.timestamp_format, strict=True).alias("m_timestamp")
        ).drop("timestamp_str")
        # Reorder columns to have 'm_timestamp' as the first column
        self.df = self.df.select(["m_timestamp"] + [col for col in self.df.columns if col != "m_timestamp"])

        if self.missing_timestamp_action == 'drop':
            # Drop rows without a timestamp
            self.df = self.df.filter(pl.col("m_timestamp").is_not_null())
        elif self.missing_timestamp_action == 'keep':
            # Keep rows without a timestamp, timestamp will be None
            pass
        
        elif self.missing_timestamp_action == 'fill-lastseen':
            # Fill missing timestamps with the last seen valid timestamp
            self.df = self.df.with_columns(
                pl.col("m_timestamp").fill_null(strategy='backward')
            )

        elif self.missing_timestamp_action == 'merge':
            # Create a flag indicating if the row starts with a timestamp
            self.df = self.df.with_columns(
                pl.col("m_timestamp").is_not_null().cast(pl.Boolean).fill_null(False).alias("flag")
            )

            # Create groups by taking a cumulative sum over the flag
            self.df = self.df.with_columns(pl.col("flag").cum_sum().alias("group"))
            # Define the columns for aggregation, including file_name if it exists
            aggregation_columns = [
                pl.col("m_timestamp").first().alias("m_timestamp"),
                pl.col("m_message").first().alias("m_message"),
                pl.col("m_message").filter(pl.col("m_timestamp").is_null()).str.concat("\n").alias("trace")
            ]

            # Include 'filename' column if it exists in the DataFrame
            if "file_name" in self.df.columns:
                aggregation_columns.append(pl.col("file_name").first().alias("file_name"))

            # Perform the aggregation
            merged_df = self.df.group_by("group", maintain_order=True).agg(aggregation_columns)

            # Drop the 'group' column
            self.df = merged_df.drop("group")

    #No mandatory columns either. 
    def check_mandatory_columns(self):
        pass