from loglead.loaders.base import BaseLoader
import polars as pl
import os
import glob

class HadoopLoader(BaseLoader):
    def __init__(self, filename, df=None, df_seq=None, filename_pattern=None, labels_file_name=None):
        self.labels_file_name = labels_file_name
        self.filename_pattern = filename_pattern
        self.event_pattern = r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}'#Each log line should start with this
        super().__init__(filename, df, df_seq)

    # see details https://github.com/logpai/loghub/blob/master/Hadoop/Hadoop_2k.log_structured.csv
    def _extract_process(self):
        # Extract data from within []
        self.df = self.df.with_columns(
            pl.col("column_1").str.extract(r"\[(.*?)\]").alias("process")
        )
        # Remove content inside [] and the surrounding spaces, and replace consecutive spaces with a single space
        self.df = self.df.with_columns(
            pl.col("column_1")
            .str.replace(r"\s*\[.*?\]\s*", " ") #first [] contain process id extracted above
            .str.replace_all(r"\s+", " ") #remove all extra white spaces that might be left. 
            .alias("column_1")
        )
        return self.df

    def load(self):
        queries = []
        # Iterate over all subdirectories
        for subdir, _, _ in os.walk(self.filename):
            seq_id = os.path.basename(subdir)
            file_pattern = os.path.join(subdir, self.filename_pattern)
            # Iterate over all files in the subdirectory that match the given pattern
            for file in glob.glob(file_pattern):
                try:
                    q = pl.scan_csv(file, has_header=False, infer_schema_length=0, separator=self._csv_separator, row_count_name="row_nr_per_file")
                    q = q.with_columns(
                        pl.lit(seq_id).alias('seq_id'), #Folder is seq_id
                        pl.lit(os.path.basename(file)).alias('seq_id_sub') #File is seq_id_sub
                    )
                    queries.append(q)
                except pl.exceptions.NoDataError: # some CSV files can be empty.
                    continue
        dataframes = pl.collect_all(queries)
        self.df = pl.concat(dataframes)
    
    #Occasionally multiline entries exists, e.g. log message followed by stack trace here we merge them to one log event. 
    def _merge_multiline_entries(self):
         # Sort the dataframe by "seq_id_sub" and "row_nr_per_file" to ensure correct lines are merged together
        self.df = self.df.sort(["seq_id_sub", "row_nr_per_file"])
        
        # Create a flag column that determines if the row starts with the pattern.
        #self.df = self.df.with_columns(pl.col("column_1").str.contains(self.event_pattern).cast(pl.Boolean).alias("flag"))
        #The above should work but for some reason there are some nulls. Force the nulls to False
        self.df = self.df.with_columns(
                pl.col("column_1").str.contains(self.event_pattern).cast(pl.Boolean).fill_null(False).alias("flag")
        )


        # Generate groups by taking a cumulative sum over the flag. This will group multi-line entries together.
        self.df = self.df.with_columns(pl.col("flag").cumsum().alias("group"))
        # Calculate number of lines in each group

        # Merge the entries in each group
        df_grouped = self.df.groupby("group").agg(
            pl.col("column_1").str.concat("\n").alias("column_1"),
            pl.col("seq_id").first().alias("seq_id"),
            pl.col("seq_id_sub").first().alias("seq_id_sub"),
            pl.col("row_nr_per_file").first().alias("row_nr_per_file")
        )
        self.df = df_grouped
        #Debug
        #df_grouped = df_grouped.with_columns(pl.col("column_1").str.n_chars().alias("entry_length"))
        # Find the entry with the longest length
        #max_entry = df_grouped.filter(pl.col("entry_length") == df_grouped.select(pl.col("entry_length").max()))
        #longest_entry_length = max_entry["entry_length"][0]
        #longest_entry_content = max_entry["column_1"][0]
        #longest_entry_seq_id_sub = max_entry["seq_id_sub"][0]

        #print(f"Longest merged entry in column_1 has {longest_entry_length} characters.")
        #print(f"Seq_id_sub of the longest entry: {longest_entry_seq_id_sub}")
        #print("Content of the longest entry:")
        #print(longest_entry_content)
        return self.df

    def preprocess(self):    
        self._merge_multiline_entries()
        self._extract_process()
        #Store columns 
        df_store_cols = self.df.select(pl.col("seq_id","seq_id_sub", "process","row_nr_per_file", "column_1"))
        #This method splits the string and overwrite self.df
        self._split_and_unnest(["date", "time","level", "component","m_message"])
        #Merge the stored columns back to self.df
        self.df = pl.concat([self.df, df_store_cols], how="horizontal")
        self._parse_datetimes()
        self.df = self.df.with_columns(pl.col("m_message").fill_null("<EMPTY LOG MESSAGE>"))
        #Aggregate labels to sequence dataframe info that is at Application ID level
        self.df_seq = self.df.select(pl.col("seq_id")).unique() #sequence level dataframe
        label_df = self._parse_labels() #parse labels
        self.df_seq = self.df_seq.join(label_df, left_on='seq_id', right_on="app_id") #merge
        self.df_seq = self.df_seq.with_columns(
            pl.col("Label").str.starts_with("Normal").alias("normal"),
        )
        
    def _parse_labels(self):
        parsed_labels = []
        with open(self.labels_file_name, "r") as file:
            lines = file.readlines()
        app_name = None
        anomaly = None
        for line in lines:
            line = line.strip()
            if line.startswith("###"):
                app_name = line.split("###")[1].strip()
            elif line.endswith(":"):
                anomaly = line[:-1]
            elif line.startswith("+"):
                app_id = line.split("+")[1].strip()
                parsed_labels.append((app_name, anomaly, app_id))
                
        label_df = pl.DataFrame({
            "app_id": [x[2] for x in parsed_labels],
            "app_name": [x[0] for x in parsed_labels],
            "Label": [x[1] for x in parsed_labels]
        })
        return label_df
    
        
        
    def _parse_datetimes(self):
        parsed_times = self.df.select(pl.concat_str([pl.col("date"), pl.col("time")]).alias("m_timestamp"))

        #parsed_times = parsed_times.to_series().str.strptime(pl.Datetime, "%Y-%m-%d%H:%M:%S,%3f")
        #parsed_times = parsed_times.with_columns(
        #   pl.col("m_timestamp").str.replace(r",", r".")
        #)
        parsed_times = parsed_times.with_columns(
            pl.col("m_timestamp").str.strptime(pl.Datetime, "%Y-%m-%d%H:%M:%S,%3f", strict=False)
        )
        self.df = self.df.with_columns(parsed_times)
  