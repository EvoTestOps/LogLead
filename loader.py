

import glob
import polars as pl
import os
from collections import Counter
import json


# Base class
class BaseLoader:
    #This should be csv separator nowhere. 
    #We try to prevent polars from trying to do csv spltting
    #Instead we do it manually. 
    _csv_separator = "\a" 
    _mandatory_columns = ["m_message", "m_timestamp"]
    
    def __init__(self, filename, df=None, df_sequences=None):
        self.filename = filename
        self.df = df #Event level dataframe
        self.df_sequences = df_sequences #sequence level dataframe

    def load(self):
        raise NotImplementedError

    def preprocess(self):
        raise NotImplementedError

    def execute(self):
        if self.df is None:
            self.load()
        self.preprocess()
        self.check_for_nulls() 
        self.check_mandatory_columns()
        return self.df
    
    def check_for_nulls(self):
        null_counts = {}  # Dictionary to store count of nulls for each column
        for col in self.df.columns:
            null_count = self.df.filter(self.df[col].is_null()).shape[0]
            if null_count > 0:
                null_counts[col] = null_count
        # Print the results
        if null_counts:
            for col, count in null_counts.items():
                print(f"WARNING! Column '{col}' has {count} null values. This can cause problems during analysis. ")
                print(f"To investigate: <DF_NAME>.filter(<DF_NAME>['{col}'].is_null())")
                
    def check_mandatory_columns(self):
        missing_columns = [col for col in self._mandatory_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing mandatory columns: {', '.join(missing_columns)}")
                  
        if 'm_time_stamp' in self._mandatory_columns and not isinstance(self.df.column("m_time_stamp").dtype, pl.datatypes.Datetime):
            raise TypeError("Column 'm_time_stamp' is not of type Polars.Datetime")

    def _split_and_unnest(self, field_names):
        #split_cols = self.df["column_1"].str.splitn(" ", n=len(field_names))
        split_cols = self.df.select(pl.col("column_1")).to_series().str.splitn(" ", n=len(field_names))
        split_cols = split_cols.struct.rename_fields(field_names)
        split_cols = split_cols.alias("fields")
        split_cols = split_cols.to_frame()
        self.df = split_cols.unnest("fields")
      
    def lines_not_starting_with_pattern(self, pattern=None):
        if self.df is None:
            self.load()
        if pattern is None:
            pattern = self.event_pattern

        # Filter lines that do not start with the pattern.
        non_matching_lines_df = self.df.filter(~pl.col("column_1").str.contains(pattern))
        # Filter lines that do start with the pattern.
        matching_lines_df = self.df.filter(pl.col("column_1").str.contains(pattern))
        
        # Get the number of lines that do not start with the pattern.
        non_matching_lines_count = non_matching_lines_df.shape[0]
        # Get the number of lines that do start with the pattern.
        matching_lines_count = matching_lines_df.shape[0]
            
        return non_matching_lines_df, non_matching_lines_count, matching_lines_df, matching_lines_count 

    def reduce_dataframes(self, frac=0.5):
        """
        Reduce the size of and update df accordingly.
        
        Parameters:
        - fraction: The fraction of rows to retain. Default is 0.5 (50%).
        """
        
        # If df_sequences is present, reduce its size
        if hasattr(self, 'df_sequences') and self.df_sequences is not None:
            self.df_sequences = self.df_sequences.sample(fraction=frac)
            # Update df to include only the rows that have seq_id values present in the filtered df_sequences
            self.df = self.df.filter(pl.col("seq_id").is_in(self.df_sequences["seq_id"]))
        else:
            # If df_sequences is not present, just reduce df
            self.df = self.df.sample(fraction=frac)

        return self.df
    
    def parse_json(self, json_line):
        json_data = json.loads(json_line)
        return pl.DataFrame([json_data])



            
# Processor for the Pro log file
class ProLoader(BaseLoader):
    def load(self):
        queries = []
        for file in glob.glob(self.filename):
            try:         
                q = pl.scan_csv(file, has_header=False, infer_schema_length=0, separator=self._csv_separator)
                q = q.with_columns((pl.lit(os.path.basename(file))).alias('seq_id'))
                queries.append(q)
            except pl.exceptions.NoDataError: # some CSV files can be empty.
                continue
        dataframes = pl.collect_all(queries)
        self.df = pl.concat(dataframes)

    def preprocess(self):
        self._remove_extra_spaces()
        #Store seq_id from files
        df_seq_id = self.df.select(pl.col("seq_id"))
        self._split_and_unnest(["count", "date", "time", "system", "nr1", "nr2", "log_level", "m_message"])
        #Concat seq_id to main dataframe
        self.df = pl.concat([self.df, df_seq_id], how="horizontal")  
        self._parse_datetimes()
        #Dataframe for aggrating to sequence level
        self.df_sequences = self.df.select(pl.col("seq_id")).unique()   
        #Filename holds the anomaly info. 
        self.df_sequences = self.df_sequences.with_columns(
            pl.col("seq_id").str.starts_with("success").alias("normal"),
        )

    #Needed because multiple spaces and no regex support
    #https://github.com/pola-rs/polars/issues/4819
    def _remove_extra_spaces(self):
        #df2 = self.df.with_columns(pl.col("column_1").str.extract_all(r"\S+").list.join(" ")) #Below is faster
        df2 = self.df.with_columns(pl.col("column_1").str.replace_all(r"\s+", " "))         
        self.df = df2

    def _parse_datetimes(self):
        parsed_times = self.df.select(pl.concat_str([pl.col("date"), pl.col("time")]).alias("m_timestamp"))
        parsed_times = parsed_times.to_series().str.strptime(pl.Datetime, "%d.%m.%Y%H:%M:%S%.3f")
        self.df = self.df.with_columns(parsed_times)


class HadoopLoader(BaseLoader):
    def __init__(self, filename, df=None, df_sequences=None, filename_pattern=None, labels_file_name=None):
        self.labels_file_name = labels_file_name
        self.filename_pattern = filename_pattern
        self.event_pattern = r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}'#Each log line should start with this
        super().__init__(filename, df, df_sequences)

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
                    q = pl.scan_csv(file, has_header=False, infer_schema_length=0, separator=self._csv_separator, row_count_name="row_nr")
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
         # Sort the dataframe by "seq_id_sub" and "row_nr" to ensure correct lines are merged together
        self.df = self.df.sort(["seq_id_sub", "row_nr"])
        
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
            pl.col("row_nr").first().alias("row_nr")
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
        df_store_cols = self.df.select(pl.col("seq_id","seq_id_sub", "process","row_nr", "column_1"))
        #This method splits the string and overwrite self.df
        self._split_and_unnest(["date", "time","level", "component","m_message"])
        #Merge the stored columns back to self.df
        self.df = pl.concat([self.df, df_store_cols], how="horizontal")
        self._parse_datetimes()
        self.df = self.df.with_columns(pl.col("m_message").fill_null("<EMPTY LOG MESSAGE>"))
        #Aggregate labels to sequence dataframe info that is at Application ID level
        self.df_sequences = self.df.select(pl.col("seq_id")).unique() #sequence level dataframe
        label_df = self._parse_labels() #parse labels
        self.df_sequences = self.df_sequences.join(label_df, left_on='seq_id', right_on="app_id") #merge
        self.df_sequences = self.df_sequences.with_columns(
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
  


# Processor for the HDFS log file
class HDFSLoader(BaseLoader):
    
    def __init__(self, filename, df=None, df_sequences=None, labels_file_name=None):
        self.labels_file_name = labels_file_name
        super().__init__(filename, df, df_sequences)
          
    def load(self):
        self.df = pl.read_csv(self.filename, has_header=False, infer_schema_length=0, separator=self._csv_separator)

    def preprocess(self):
        #self._split_columns()
        self._split_and_unnest(["date", "time", "id", "level", "component", "m_message"])
        self._extract_seq_id()
        self._parse_datetimes()
        #Aggregate labels to sequence dataframe info that is at BlockID level
        self.df_sequences = self.df.select(pl.col("seq_id")).unique()  
        df_temp = pl.read_csv(self.labels_file_name, has_header=True)
        self.df_sequences = self.df_sequences.join(df_temp, left_on='seq_id', right_on="BlockId")
        self.df_sequences = self.df_sequences.with_columns(
            pl.col("Label").str.starts_with("Normal").alias("normal"),
        )
        self.df_sequences = self.df_sequences.drop("Label")

    def _extract_seq_id(self):
        #seq_id = self.df.select(pl.col("m_message").str.extract(r"blk_(-?\d+)", group_index=1).alias("seq_id"))
        seq_id = self.df.select(pl.col("m_message").str.extract(r"(blk_[-?\d]+)", group_index=1).alias("seq_id"))
        self.df = self.df.with_columns(seq_id)

    def _parse_datetimes(self):
        parsed_times = self.df.select(pl.concat_str([pl.col("date"), pl.col("time")]).alias("m_timestamp"))
        parsed_times = parsed_times.to_series().str.strptime(pl.Datetime, "%y%m%d%H%M%S")
        self.df = self.df.with_columns(parsed_times)

# Processor for the Thunderbird log file
class ThunderbirdLoader(BaseLoader):
    def load(self):
        self.df = pl.read_csv(self.filename, has_header=False, infer_schema_length=0, 
                              separator=self._csv_separator, ignore_errors=True) #There is one UTF error in the file
    
    def preprocess(self):
        self._split_and_unnest(["label", "timestamp", "date", "userid", "month", 
                                "day", "time", "location", "component_pid", "m_message"])
        self._split_component_and_pid()
        #parse datatime
        self.df = self.df.with_columns(m_timestamp = pl.from_epoch(pl.col("timestamp")))
        #Label contains multiple anomaly cases. Convert to binary
        self.df = self.df.with_columns(normal = pl.col("label").str.starts_with("-"))

    #Reason for extra processing. We want so separte pid from component and in the log file they are embedded
    #Data description
    #https://github.com/logpai/loghub/blob/master/Thunderbird/Thunderbird_2k.log_structured.csv     
    def _split_component_and_pid(self):
        component_and_pid = self.df.select(pl.col("component_pid")).to_series().str.splitn("[", n=2)
        component_and_pid = component_and_pid.struct.rename_fields(["component", "pid"])
        component_and_pid = component_and_pid.alias("fields")
        component_and_pid = component_and_pid.to_frame()
        component_and_pid = component_and_pid.unnest("fields")
        component_and_pid = component_and_pid.with_columns(pl.col("component").str.rstrip(":"))
        component_and_pid = component_and_pid.with_columns(pl.col("pid").str.rstrip("]:"))
        self.df= pl.concat([self.df, component_and_pid], how="horizontal")  
        self.df = self.df.drop("component_pid")
        self.df = self.df.select(["label", "timestamp", "date", "userid", "month", 
                                "day", "time", "location", "component","pid", "m_message"])


# Processor for the BGL log file
# At the moment there are 34470 null messages that are not handled by the loader
class BGLLoader(BaseLoader):
    def load(self):
        self.df = pl.read_csv(self.filename, has_header=False, infer_schema_length=0, 
                              separator=self._csv_separator, ignore_errors=True)
    def preprocess(self):
        self._split_and_unnest(["label", "timestamp", "date", "node", "time", 
                            "noderepeat", "type", "component", "level", "m_message"])
        #parse datatime
        self.df = self.df.with_columns(m_timestamp = pl.from_epoch(pl.col("timestamp")))
      
#Process log files created with the GELF logging driver.
# 
class GELFLoader(BaseLoader):
    def load(self):
        with open(self.filename, 'r') as file:
            lines = file.readlines()

        # Assuming each line in the file is a separate JSON object
        json_frames = [self.parse_json(line) for line in lines]
        self.df = pl.concat(json_frames)

    def preprocess(self):
        # Rename some columns to match the expected column names and parse datetime
        self.df = self.df.with_columns(
            pl.col("message").alias("m_message")
            ).drop("message")
        
        parsed_timestamps = self.df.select(
            pl.col("@timestamp").str.strptime(pl.Datetime, strict=False).alias("m_timestamp")
        ).drop("@timestamp")
        self.df = self.df.with_columns(parsed_timestamps)