import json

import polars as pl

__all__ = ['BaseLoader']


# Base class
class BaseLoader:
    # This csv separator should never be found.
    # We try to disable polars from doing csv splitting.
    # Instead we do it manually to get it correctly done.
    _csv_separator = "\a" 
    _mandatory_columns = ["m_message", "m_timestamp"]
    
    def __init__(self, filename, df=None, df_seq=None):
        self.filename = filename
        self.df = df  # Event level dataframe
        self.df_seq = df_seq  # Sequence level dataframe

    def load(self):
        raise NotImplementedError
        
    def preprocess(self):
        raise NotImplementedError

    def execute(self):
        if self.df is None:
            self.load()
        self.preprocess()
        self.check_for_nulls_and_non_utf8()
        self.check_mandatory_columns()
        self.add_ano_col()
        return self.df
    
    def add_ano_col(self):
        # Check if the 'normal' column exists
        if self.df is not None and "normal" in self.df.columns:
            # Create the 'anomaly' column by inverting the boolean values of the 'normal' column
            self.df = self.df.with_columns(pl.col("normal").not_().alias("anomaly"))
        if self.df_seq is not None and "normal" in self.df_seq:
            # Create the 'anomaly' column by inverting the boolean values of the 'normal' column
            self.df_seq = self.df_seq.with_columns(pl.col("normal").not_().alias("anomaly"))

        # Check if the 'anomaly' column exists but no normal column
        if self.df is not None and "anomaly" in self.df.columns and not "normal" in self.df.columns:
            # Create the 'normal' column by inverting the boolean values of the 'anomaly' column
            self.df = self.df.with_columns(pl.col("anomaly").not_().alias("normal"))
        # self._mandatory_columns = ["m_message"]

    def check_for_nulls_and_non_utf8(self):
        issue_counts = {}  # Dictionary to store counts of both nulls and non-UTF-8 issues for each column

        # Check for null values and non-UTF-8 values
        for col in self.df.columns:
            null_count = self.df.filter(self.df[col].is_null()).shape[0]
            if null_count > 0:
                issue_counts[col] = {"nulls": null_count}

            if self.df[col].dtype == pl.Utf8:  # Check non-UTF-8 only for string columns
                non_utf8_count = self.df.filter(pl.col(col).str.contains("�")).shape[0]
                if non_utf8_count > 0:
                    if col in issue_counts:
                        issue_counts[col]["non_utf8"] = non_utf8_count
                    else:
                        issue_counts[col] = {"non_utf8": non_utf8_count}

        # Print the results
        if issue_counts:
            for col, issues in issue_counts.items():
                issue_types = []
                if "nulls" in issues:
                    issue_types.append(f"{issues['nulls']} null")
                if "non_utf8" in issues:
                    issue_types.append(f"{issues['non_utf8']} non-UTF-8 encoded")

                issue_description = " and ".join(issue_types)
                print(f"WARNING! Column '{col}' has {issue_description} values out of {len(self.df)}.")
                
                # The merged options block
                print(f"You have 4 options:"
                    f" 1) Do nothing and hope for the best"
                    f", 2) Drop the column"
                    f", 3) Filter out rows with {issue_description} values"
                    f", 4) Investigate and fix your Loader or Data")

                # Instructions to investigate specific issues
                if "nulls" in issues:
                    print(f"To investigate null values: <DF_NAME>.filter(pl.col('{col}').is_null())")
                if "non_utf8" in issues:
                    print(f"To investigate non-UTF-8 values: <DF_NAME>.filter(pl.col('{col}').str.contains('�'))")

    def check_mandatory_columns(self):
        missing_columns = [col for col in self._mandatory_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing mandatory columns: {', '.join(missing_columns)}")
                  
        if 'm_time_stamp' in self._mandatory_columns and not isinstance(self.df.column("m_time_stamp").dtype, pl.datatypes.Datetime):
            raise TypeError("Column 'm_time_stamp' is not of type Polars.Datetime")

    def _split_and_unnest(self, field_names):
        # split_cols = self.df["column_1"].str.splitn(" ", n=len(field_names))
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

    def reduce_dataframes(self, frac=0.5, random_state=42):
        # If df_sequences is present, reduce its size
        if hasattr(self, 'df_seq') and self.df_seq is not None:
            # Sample df_seq
            df_seq_temp = self.df_seq.sample(fraction=frac, seed=random_state)

            # Check if df_seq still has at least one row
            if len(df_seq_temp) == 0:
                # If df_seq is empty after sampling, randomly select one row from the original df_seq
                self.df_seq = self.df_seq.sample(n=1)
            else:
                self.df_seq = df_seq_temp
            # Update df to include only the rows that have seq_id values present in the filtered df_seq
            self.df = self.df.filter(pl.col("seq_id").is_in(self.df_seq["seq_id"]))

            # self.df_seq = self.df_seq.sample(fraction=frac)
            # Update df to include only the rows that have seq_id values present in the filtered df_sequences
            # self.df = self.df.filter(pl.col("seq_id").is_in(self.df_seq["seq_id"]))
        else:
            # If df_sequences is not present, just reduce df
            self.df = self.df.sample(fraction=frac, seed=random_state)

        return self.df
    
    @staticmethod
    def parse_json(json_line):
        json_data = json.loads(json_line)
        return pl.DataFrame([json_data])
