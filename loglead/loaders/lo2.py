import os
import polars as pl
from datetime import datetime
from .base import BaseLoader
import random
import json

__all__ = ['LO2Loader']

# note that metric loader is implemented but it hasn't been used much so there are likely issues

class LO2Loader(BaseLoader):
    def __init__(self, filename, df=None, df_seq=None, n_runs=53, errors_per_run=1, dup_errors=True, single_error_type=None, single_service=""):
        """
        :param filename: Path to the data directory (this is the root where the runs are).
        :param n_runs: Number of runs to process.
        :param errors_per_run: Number of errors to include per run.
        :param dup_errors: Whether duplicate errors are allowed across runs.
        :param single_error_type: A specific error type to use exclusively across all runs, or "random" to select one randomly.
        :param single_service: A specific service instead of all to use in the analysis. Options: client, code, key, refresh-token, service, token, user
        """
        self.filename = filename
        self.n_runs = n_runs
        self.errors_per_run = errors_per_run
        self.dup_errors = dup_errors
        self.single_error_type = single_error_type
        if single_service in ["", "client", "code", "key", "refresh-token", "service", "token", "user"]:
            self.service_type = "oauth2-oauth2-"+single_service
            print("Service type set:", single_service)
        else:
            print("Invalid service type given!")

        # Adjust settings if single_error_type is set
        if self.single_error_type:
            self.dup_errors = True
            self.errors_per_run = 1
            print(f"single_error_type is set to '{self.single_error_type}'. "
                  f"Setting dup_errors to True and errors_per_run to 1.")

        self.selected_random_error = None  # Store the randomly chosen error if single_error_type == "random"
        self.metrics_df = None
        self.used_errors = set()  # Track used error cases when dup_errors is False

        super().__init__(filename, df, df_seq)

    def load(self):
        data = []
        n = 0
        total_errors_needed = self.n_runs * self.errors_per_run

        for run in os.listdir(self.filename):
            run_path = os.path.join(self.filename, run)
            if os.path.isdir(run_path):
                n += 1
                if n > self.n_runs:
                    break

                test_cases = os.listdir(run_path)
                available_errors = [tc for tc in test_cases if tc != "correct"]

                if self.single_error_type == "random":
                    # Select a random error type in the first run
                    if self.selected_random_error is None:
                        if available_errors:
                            self.selected_random_error = random.choice(available_errors)
                            print(f"Randomly selected error type: {self.selected_random_error}")
                        else:
                            print(f"No errors available in the first run to select randomly. Skipping.")
                            continue
                    error_cases = [self.selected_random_error]
                elif self.single_error_type:
                    # Use only the specified single error type
                    if self.single_error_type in available_errors:
                        error_cases = [self.single_error_type]
                    else:
                        print(f"Warning: Error type '{self.single_error_type}' not found in run {run}. Skipping.")
                        continue
                elif not self.dup_errors:
                    # Filter out previously used errors
                    available_errors = [tc for tc in available_errors if tc not in self.used_errors]

                    # Check if we have enough unique errors remaining
                    if len(available_errors) < self.errors_per_run:
                        print(f"Warning: Not enough unique errors available for run {run}. "
                              f"Needed {self.errors_per_run}, but only {len(available_errors)} remaining.")
                        error_cases = available_errors  # Use all remaining unique errors
                    else:
                        error_cases = random.sample(available_errors, self.errors_per_run)

                    # Update used errors set
                    self.used_errors.update(error_cases)
                else:
                    # Original behavior with duplicate errors allowed
                    error_cases = random.sample(available_errors, 
                                                min(self.errors_per_run, len(available_errors)))

                selected_cases = ["correct"] + error_cases

                for test_case in selected_cases:
                    test_case_path = os.path.join(run_path, test_case)
                    if os.path.isdir(test_case_path):
                        for log_file in os.listdir(test_case_path):
                            log_file_path = os.path.join(test_case_path, log_file)
                            if os.path.isfile(log_file_path) and self.service_type in log_file:
                                #print(f"Processing: {log_file_path}")
                                try:
                                    log_df = self._process_log_file(log_file_path, run, test_case, log_file)
                                    if log_df is not None:
                                        data.append(log_df)
                                except Exception as e:
                                    print(f"Error processing {log_file_path}: {e}")

        # Check if we got enough errors when dup_errors is False
        if not self.dup_errors and len(self.used_errors) < total_errors_needed:
            print(f"Warning: Could not find enough unique errors. "
                  f"Needed {total_errors_needed}, but only found {len(self.used_errors)}")

        if data:
            self.df = pl.concat(data, how="vertical").drop_nulls()
            self.df = self.df.with_columns(normal=pl.col("test_case") == "correct")
            self._parse_timestamps()

    def load_metrics(self):
        temporal_metrics = [
            'metric_node_load1.json'
        ]
        metrics_data = []
        for run in os.listdir(self.filename):
            run_path = os.path.join(self.filename, run)
            if os.path.isdir(run_path):
                test_cases = os.listdir(run_path)
                for test_case in test_cases:
                    metrics_path = os.path.join(run_path, test_case, "metrics")
                    if os.path.isdir(metrics_path):
                        for metrics_file in os.listdir(metrics_path):
                            metrics_file_path = os.path.join(metrics_path, metrics_file)
                            if metrics_file in temporal_metrics:
                                print(f"Processing metrics: {metrics_file_path}")
                                try:
                                    metrics_df = self._process_metrics_file(metrics_file_path, run, test_case)
                                    if metrics_df is not None:
                                        metrics_data.append(metrics_df)
                                except Exception as e:
                                    print(f"Error processing metrics {metrics_file_path}: {e}")
        if metrics_data:
            self.metrics_df = pl.concat(metrics_data, how="vertical")

    def _process_log_file(self, log_file_path, run, test_case, log_file):
        with open(log_file_path, 'r', encoding="utf-8") as f:
            log_lines = f.readlines()
        
        log_lines_cleaned = [str(line).strip() for line in log_lines if line.strip()]
        
        if log_lines_cleaned:
            service = os.path.splitext(log_file)[0]  # Get service name from log file name without extension
            log_df = pl.DataFrame({
                "m_message": log_lines_cleaned
            })
            
            log_df = log_df.with_columns(
                pl.lit(run).alias("run"),
                pl.lit(test_case).alias("test_case"),
                pl.lit(service).alias("service"),
                pl.concat_str([pl.lit(run), pl.lit(test_case), pl.lit(service)], separator="__").alias("seq_id")
            )
            
            return log_df
        
        return None
    
    def _process_metrics_file(self, metrics_file_path, run, test_case):
        with open(metrics_file_path, 'r') as f:
            metrics_json = json.load(f)
        
        metrics_list = []
        for metric in metrics_json:
            if metric['metric'].get('job') == 'node':
                metric_name = metric['metric']['__name__']
                for timestamp, value in metric['values']:
                    metrics_list.append({
                        'run': run,
                        'test_case': test_case,
                        'metric_name': metric_name,
                        'timestamp': timestamp,
                        'value': float(value)
                    })
        if metrics_list:
            df = pl.DataFrame(metrics_list)
            df = df.with_columns(
                pl.from_epoch(pl.col('timestamp'), time_unit='s').alias('timestamp')
            )
            return df
        return None

    def preprocess(self):
        if self.df is not None:
            self._create_df_seq()
            self._add_anomaly_column()

    def _create_df_seq(self):
        self.df_seq = self.df.group_by("seq_id").agg([
            pl.col("test_case").first().alias("test_case"),
            pl.col("service").first().alias("service"),
            pl.col("m_message").str.concat("\n"),
            pl.col("normal").any().alias("normal"),
            pl.col("m_timestamp").min().alias("start_time"),
            pl.col("m_timestamp").max().alias("end_time")
        ])
        self.df_seq = self.df_seq.drop(["test_case", "service"])

    def _add_anomaly_column(self):
        self.df = self.df.with_columns(
            (~self.df['normal']).alias('anomaly')
        )
        self.df_seq = self.df_seq.with_columns(
            (~self.df_seq['normal']).alias('anomaly')
        )

    # The logs generally don't have date, so the it will 1900-01-01
    def _parse_timestamps(self):
        def parse_timestamp(s: str) -> datetime | None:
            formats = [
                "%Y-%m-%d %H:%M:%S",  # YYYY-MM-DD HH:MM:SS
                "%H:%M:%S.%f",        # HH:MM:SS.mmm
                "%Y-%m-%dT%H:%M:%S",  # ISO format
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    continue
            return None

        self.df = self.df.with_columns([
            pl.col("m_message").str.extract(r'^(\d{2}:\d{2}:\d{2}.\d{3})').alias("m_timestamp")
        ])

        # Remove rows where timestamp is null
        self.df = self.df.filter(pl.col("m_timestamp").is_not_null())


        # Parse the timestamp into a datetime object, specifying the return type
        self.df = self.df.with_columns([
            pl.col("m_timestamp").map_elements(parse_timestamp, return_dtype=pl.Datetime)
        ])

