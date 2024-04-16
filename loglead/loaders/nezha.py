import glob
import os
from loglead.loaders.base import BaseLoader
import polars as pl
import re
import json

import traceback
import logging
logger = logging.getLogger(__name__)

class NezhaLoader(BaseLoader):
    def __init__(self, filename, system, df=None, df_seq=None):
        """
        Parameters:
        system_name (str): Either TrainTicket or WebShop

        Raises:
        ValueError: If the system_name is not a valid option.
        """
         # Define the valid options
        valid_options = ['TrainTicket', 'WebShop']
        # Check if the system_name is in the list of valid options
        if system in valid_options:
            self.system = system
            super().__init__(filename, df, df_seq)
        else:
            # If it's not, raise an error
            raise ValueError(f"Invalid system name: {system}. Valid options are: {', '.join(valid_options)}")


    def load(self):
        log_queries = []
        metric_queries = {
            'src_dst': [],
            'front_srv': [],
            'dep': [],
            'default': []
        }
        trace_queries = []
        label_queries = []
        all_label_df = pl.DataFrame()
        date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')  # Regular expression for 'dddd-dd-dd'

        # Iterate over all subdirectories in the base directory
        for subdir, dirs, files in os.walk(self.filename):
            # Check if the current subdir is within rca_data or construct_data
            if "rca_data" in subdir:
                ano_folder = True
            elif "construct_data" in subdir:
                ano_folder = False
            else:
                continue  # Skip if it's neither of those folders
            match = date_pattern.match(os.path.basename(subdir))
            if match:
                date_str = match.group()  # Extract the matched date string
                #We want WebShop data but are in the train ticket folder -> skip this data
                if ((date_str == "2023-01-29" or date_str =="2023-01-30") and (self.system == "WebShop")):
                    continue
                #We want TrainTicket data but are not in the train ticket folder skip
                elif ((date_str == "2022-08-23" or date_str =="2022-08-22") and (self.system == "TrainTicket")):
                    continue
               
                #Log
                log_path = os.path.join(subdir, "log")
                if os.path.exists(log_path):
                    self.load_log(log_path, date_str, log_queries, ano_folder)
                #Metrics
                metric_path = os.path.join(subdir, "metric")
                if os.path.exists(metric_path):
                    self.load_metric(metric_path, date_str, metric_queries, ano_folder)
                #Traces
                trace_path = os.path.join(subdir, "trace")
                if os.path.exists(trace_path):
                    self.load_trace(trace_path, date_str, trace_queries, ano_folder)
                #Labels
                label_file = os.path.join(subdir, f"{date_str}-fault_list.json")
                if os.path.exists(label_file):
                    #self.process_label(label_path, date_str, label_queries)
                    label_df = self.load_label(label_file, date_str)
                    if label_df is not None:
                        all_label_df = pl.concat([all_label_df, label_df])
            #RCAs per service
            #Hipster
            rca_hip_file = os.path.join(subdir, f"root_cause_hipster.json")
            if os.path.exists(rca_hip_file):
                self.df_rca_hip = self.load_rca(rca_hip_file)
            #TrainTicket
            rca_ts_file = os.path.join(subdir, f"root_cause_ts.json")
            if os.path.exists(rca_ts_file):
                self.df_rca_ts = self.load_rca(rca_ts_file)


        self.df_label = all_label_df
        #Collect files that were read with lazy_frame
        #Collect logs
        dataframes = pl.collect_all(log_queries)
        self.df = pl.concat(dataframes)
        self.df = self.df.rename({"Log":"raw_m_message"})
        #Collect traces
        dataframes = pl.collect_all(trace_queries)
        self.df_trace = pl.concat(dataframes)
        # Collect metrics
        for group, queries in metric_queries.items():
            if queries:
                try:
                    dataframes = pl.collect_all(queries)
                    if dataframes:
                        # Standardize column order based on the first DataFrame
                        reference_columns = dataframes[0].columns
                        standardized_dfs = [df.select(reference_columns) for df in dataframes]
                        df = pl.concat(standardized_dfs)
                        #Metrics are set here
                        setattr(self, f'df_metric_{group}', df)
                except pl.exceptions.ShapeError as e:
                    print(f"Error concatenating group '{group}': {e}")
                    # Debugging: print out column names for each DataFrame in the group
                    for q in queries:
                        collected_df = q.collect()
                        print(f"Columns in {group}: {collected_df.columns}")
                    raise


    def load_rca (self, file_path):
        # Read JSON data from a file
        with open(file_path, 'r') as file:
            data = json.load(file)
        # Reshape the data
        reshaped_data = []
        for service, metrics in data.items():
            row = {'service': service}
            row.update(metrics)
            reshaped_data.append(row)
        # Create a DataFrame
        return pl.DataFrame(reshaped_data)

    def load_label(self, file_path, date_str):
        label_data = pl.DataFrame()
        try:
            with open(file_path, 'r') as file:
                data = json.load(file) #Polars JSON reader cannot handel the non-standard json so using json.load
            # Iterate over each key in the JSON and extract the records
            for key in data:
                records = data[key]
                if records:  # Check if the list is not empty
                    df = pl.DataFrame(records)
                    df = df.with_columns(
                        pl.lit(date_str).alias('date_folder'),
                        pl.lit(os.path.basename(file_path)).alias('label_file_name')
                    )
                    label_data = label_data.vstack(df)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error in file: {file_path}")
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error processing file: {file_path}")
            print(f"Error: {e}")
            traceback.print_exc()
        return label_data
    
    def load_metric(self, folder_path, date_str, queries, ano_folder):
        if (ano_folder): #Metrics are copied to both ano_folder and normal_data folder
            return
        file_pattern = os.path.join(folder_path, "*.csv")
        for file in glob.glob(file_pattern):
            try:
                file_name = os.path.basename(file)
                q = pl.scan_csv(file, has_header=True, infer_schema_length=0, separator=",")
                q = q.with_columns(
                    pl.lit(date_str).alias('date_folder'),  # Folder is date info
                    pl.lit(os.path.basename(file)).alias('metric_file_name')  # File name storage
                )
                if "source_50.csv" in file_name or "destination_50.csv" in file_name:
                    queries['src_dst'].append(q)   
                elif "front_service.csv" in file_name:
                    queries['front_srv'].append(q)
                elif "dependency.csv" in file_name:
                    queries['dep'].append(q)
                else:
                    queries['default'].append(q)
            except pl.exceptions.NoDataError:
                continue

    def load_trace(self, folder_path, date_str, queries, ano_folder):
        file_pattern = os.path.join(folder_path, "*.csv")
        for file in glob.glob(file_pattern):
            try:
                q = pl.scan_csv(file, has_header=True, separator=",", row_count_name="row_nr_per_file")
                q = q.with_columns(
                    pl.lit(date_str).alias('date_folder'),  # Folder is date info
                    pl.lit(os.path.basename(file)).alias('file_name'),  # File name storage
                    pl.lit(ano_folder).alias('anomaly_folder')
                )
                queries.append(q)
            except pl.exceptions.NoDataError:  # some CSV files can be empty.
                continue
    
    def load_log(self, folder_path, date_str, queries, ano_folder):
        file_pattern = os.path.join(folder_path,  "*.csv")
        for file in glob.glob(file_pattern):
            try:
                q = pl.scan_csv(file, has_header=True, infer_schema_length=0, separator=",", 
                                row_count_name="row_nr_per_file", truncate_ragged_lines=True)
                
                q = q.with_columns(
                    pl.lit(date_str).alias('date_folder'),  # Folder is date info
                    pl.lit(os.path.basename(file)).alias('file_name'),  # File name storage
                    pl.lit(ano_folder).alias('anomaly_folder'),
                    #pl.lit(system_name).alias('system_name')
                )
                queries.append(q)
            except pl.exceptions.NoDataError:  # some CSV files can be empty.
                continue

    # def _execute(self):
    #     if self.df is None:
    #         self.load()
    #     self.preprocess()
    #     return self.df

    def preprocess(self):
        #This removes 221 rows that apparently leak anomaly injection info to logs
        self.df = self.df.with_columns(pl.col("raw_m_message").str.replace("Inject cpu successfully", "").str.replace("Inject cpu successfully", ""))
        self._extract_log_message()
        self.parse_timestamps()
        self.process_metrics()
        self.add_labels_to_metrics()
        self.df = self.df.with_row_count()
        self.df_trace = self.df_trace.with_row_count()
        self.df = self.add_labels_to_df(self.df)
        self.df_trace = self.add_labels_to_df(self.df_trace)
        
        self.df = self.df.with_columns(
            pl.concat_str([pl.col("date_folder"),pl.lit(" "), pl.col("file_name"), pl.lit(" "), pl.col("PodName")]).alias("seq_id")
        )
        self.df_trace = self.df_trace.with_columns(
            pl.concat_str([pl.col("date_folder"),pl.lit(" "), pl.col("file_name"), pl.lit(" "), pl.col("PodName")]).alias("seq_id")
        )

        self.df_seq = self.add_labels_to_df_seq(self.df)
        self.df_trace_seq = self.add_labels_to_df_seq(self.df_trace)


    def process_metrics(self):            
        self.df_label = self.df_label.with_columns(
            (pl.col("m_timestamp") + pl.duration(minutes=1)).alias("m_timestamp+1"),
            (pl.col("m_timestamp") + pl.duration(minutes=3)).alias("m_timestamp+3"),
            (pl.col("m_timestamp") + pl.duration(minutes=4)).alias("m_timestamp+4"),
        )
        self.df_metric_default = self.df_metric_default.with_row_count()
        column_names = [
            "CpuUsage(m)", "CpuUsageRate(%)", "MemoryUsage(Mi)", "MemoryUsageRate(%)",
            "SyscallRead","SyscallWrite","NetworkReceiveBytes", "NetworkTransmitBytes",
            "PodClientLatencyP90(s)", "PodServerLatencyP90(s)", "PodClientLatencyP95(s)",
            "PodServerLatencyP95(s)", "PodClientLatencyP99(s)", "PodServerLatencyP99(s)",
            "PodWorkload(Ops)", "PodSuccessRate(%)", "NodeCpuUsageRate(%)",
            "NodeMemoryUsageRate(%)", "NodeNetworkReceiveBytes"
        ]
        for col in column_names:
            self.df_metric_default = self.df_metric_default.cast({col:pl.Float64})



    def _extract_log_message(self):
        self.df = self.df.with_row_count("row_key")#Used for matching later
        #Splitting criteria. We have valid json and invalid flag
        if self.system == "WebShop":
        
            self.df = self.df.with_columns(normal_json = (pl.col("raw_m_message").str.contains("message")) &
                            (pl.col("raw_m_message").str.contains("severity")))# &
                            #(pl.col("raw_m_message").str.contains("timestamp")))
                            

            df_normal_json = self.df.filter(pl.col("normal_json")).select("raw_m_message", "row_key",  "SpanID" )
            df_abnormal_json = self.df.filter(~pl.col("normal_json")).select("raw_m_message", "row_key", "SpanID" )
                    
            logger.info (f"WS df_normal_json: {df_normal_json[0]['raw_m_message'] [0]}")
            logger.info (f"WS df_abnormal_json: {df_abnormal_json[0]['raw_m_message'][0]}")
        
            df_normal_json = df_normal_json.with_columns(pl.col("raw_m_message").str.json_decode())
            df_normal_json = df_normal_json.with_columns(pl.col("raw_m_message").struct.field("log"))
            #Debug bad JSON
            # for index, row in enumerate(df_normal_json.to_dicts()):
            #     try:
            #         if index % 100 == 0:
            #             print(".", end="")
            #         row["log"] = df_normal_json[index].with_columns(pl.col("log").str.json_decode())
            #     except Exception as e:
            #         logger.error(f"Row {index}:{df_normal_json[index]['log'][0]} {str(e)}")


            df_normal_json = df_normal_json.with_columns(pl.col("log").str.json_decode())
            #extract message and severity
            df_normal_json = df_normal_json.with_columns(pl.col("log").struct.field("message"))
            df_normal_json = df_normal_json.with_columns(pl.col("log").struct.field("severity"))
            df_normal_json = df_normal_json.drop(["raw_m_message", "log"])

            df_abnormal_json = df_abnormal_json.with_columns(severity = None)
            df_abnormal_json = df_abnormal_json.rename({"raw_m_message":"message"})
            #if self.system == "WebShop":
            df_abnormal_json = df_abnormal_json.select(df_normal_json.columns)
            #The dataframes have now equal fields and no overlap -> vertical stack
            df_t3 = df_normal_json.vstack(df_abnormal_json)

        #Prepare abnormal for merge
        else:
            #self.df = self.df.with_columns(normal_json = (pl.col("raw_m_message").
            #                                              str.contains("log")) &
            #                (pl.col("raw_m_message").str.contains("time")) &
            #                (pl.col("raw_m_message").str.contains("TraceID")))

            df_normal_json = self.df#.filter(pl.col("normal_json")).select("raw_m_message", "row_key",  "SpanID" )
            #df_abnormal_json = self.df.filter(~pl.col("normal_json")).select("raw_m_message", "row_key", "SpanID" )
           
            #df_abnormal_json =  self.df.select("raw_m_message", "row_key",  "SpanID" )
            logger.info (f"TT df_normal_json: {df_normal_json[0]['raw_m_message'][0]}")

            #logger.info (f"TT df_abnormal_json: {df_abnormal_json[0]['raw_m_message']}")

            #We split on the closing { as there should not be anything after that
            df_normal_json = df_normal_json.with_columns(
                [
                    pl.col("raw_m_message")
                    .str.splitn("}ulate distance]", 2)
                    .struct.rename_fields(["raw_m_message_fix", "json_error_part"])
                    .alias("fields"),
                ]
            ).unnest("fields")
            df_normal_json = df_normal_json.with_columns(
                pl.when(pl.col("json_error_part").is_not_null()) # replace "other_field" with your actual field
                .then(pl.col("raw_m_message_fix") + pl.lit('}'))
                .otherwise(pl.col("raw_m_message_fix"))
            )
            #Debug bad JSON
            # for index, row in enumerate(df_normal_json.to_dicts()):
            #     try:
            #         if index % 100 == 0:
            #             print(".", end="")
            #         row["raw_m_message_fix"] = df_normal_json[index].with_columns(pl.col("raw_m_message_fix").str.json_decode())
            #     except Exception as e:
            #         logger.error(f"Row {index}:{df_normal_json[index]['raw_m_message_fix'][0]} {str(e)}")
            # #Fix broken JSON
            df_normal_json = df_normal_json.drop("json_error_part") 
            df_normal_json = df_normal_json.with_columns(pl.col("raw_m_message_fix").str.json_decode())
            df_normal_json = df_normal_json.with_columns(pl.col("raw_m_message_fix").struct.field("log"))
            #df_abnormal_json = df_abnormal_json.with_columns(pl.col("log")
            #TODO toimisi uudemmalla 0.20
            #df_normal_json = df_normal_json.with_columns(
            #    pl.col("log").str.split(by=" ").arr.get(1).alias("severity")
            #)


            df_normal_json = df_normal_json.with_columns(
                [
                    pl.col("log")
                    .str.splitn("", 3)
                    .struct.rename_fields(["_time", "severity", "_rest"])
                    .alias("fields"),
                ]
            ).unnest("fields")

            df_normal_json = df_normal_json.rename({"raw_m_message":"message"})
            df_normal_json = df_normal_json.drop(["_time", "_rest", "raw_m_message_fix"])
                                

            #df_abnormal_json = df_abnormal_json.with_columns(severity = pl.lit(""))
            #df_abnormal_json = df_abnormal_json.rename({"raw_m_message":"message"})
            df_t3 = df_normal_json#.vstack(df_abnormal_json)
            #df_t3 = df_abnormal_json

        #Each log message contains span and trace ids remove them here as they are already separate columns
        #"message\":\"TraceID: 04c707faa29852d058b7ad236b6ef47a SpanID: 7f8791f4ed419539 Get currency data successful\",
        #Remove extra beginning
        df_t3 = df_t3.with_columns(pl.col("message").str.split_exact(df_t3["SpanID"],1)
                                .alias("fields")
                                .struct.rename_fields(["redu1", "message_part"])
                                ).unnest("fields")
        #Remove extra end that is in ones coming from df_abnormal_json
        df_t3 = df_t3.with_columns(pl.col("message_part").str.split_exact('\",', 1)
                                .alias("fields")
                                .struct.rename_fields(["m_message", "redu2"])
                                ).unnest("fields")
        #Lose any extra preceeding and trailing characters.
        df_t3 = df_t3.with_columns(
            pl.col("m_message")
            .str.strip_chars_start()
            .str.strip_chars_end('\n\\\\\\n'))
        #Drop unnecessary columns and merge to main df
        #df_t3 = df_t3.drop(["message", "redu1", "redu2", "message_part", "SpanID", "normal_json"])
        df_t3 = df_t3.select(["row_key", "severity", "m_message"])
        self.df =self.df.join(df_t3, "row_key", "left")
        self.df = self.df.drop(["normal_json"])

    #Epoch is corrupted using human readable format
    # https://github.com/IntelligentDDS/Nezha/issues/8
    def _parse_timestamps_epoch(self):
        #Logs
        #There are some timestamp starting with -6 when should 16
        ## https://github.com/IntelligentDDS/Nezha/issues/8
        self.df = self.df.with_columns(
                m_timestamp=pl.when(pl.col('TimeUnixNano').str.starts_with("-6"))
                        .then(pl.col('TimeUnixNano').str.replace(r".$",""))#The minus lines are one element too long
                        .otherwise(pl.col('TimeUnixNano')))
        self.df = self.df.with_columns(m_timestamp = pl.col("m_timestamp").str.replace(r"^-6","16"))
        self.df = self.df.with_columns(m_timestamp = pl.col("m_timestamp").str.to_integer())
        self.df = self.df.with_columns(m_timestamp = pl.from_epoch(pl.col("m_timestamp"), time_unit="ns"))
        #Traces
        self.df_trace = self.df_trace.with_columns(m_timestamp = pl.from_epoch(pl.col("StartTimeUnixNano"), time_unit="ns"))
        self.df_trace = self.df_trace.with_columns(m_timestamp_end = pl.from_epoch(pl.col("EndTimeUnixNano"), time_unit="ns"))
        #self.df_metric_default = self.df_metric_default.with_columns(m_timestamp = pl.from_epoch(pl.col("TimeStamp")))
        #For some reason some metric datasets have incorrect unix time e.g. 1861140279 when it should be 1661140279.
        # https://github.com/IntelligentDDS/Nezha/issues/8
        self.df_metric_default = self.df_metric_default.with_columns(m_timestamp = pl.col("TimeStamp").str.replace(r"^18","16"))
        self.df_metric_default = self.df_metric_default.with_columns(m_timestamp = pl.from_epoch(pl.col("m_timestamp")))
        #Labels
        self.df_label = self.df_label.with_columns(m_timestamp = pl.from_epoch(pl.col("inject_timestamp"), time_unit="s"))

    def parse_timestamps(self):
        #Epoch is corrupted using human readable format
        # https://github.com/IntelligentDDS/Nezha/issues/8
        #Logs
        self.df = self.df.with_columns(
            pl.coalesce(
                # Handeling Not consistent format
                # Most are formated  2023-01-29T09:33:09.036923751Z
                pl.col('Timestamp').str.strptime(pl.Datetime, "%FT%H:%M:%S%.9fZ",strict=False),
                #While others are 2023-01-29T09:33:14.716
                pl.col('Timestamp').str.strptime(pl.Datetime, "%FT%H:%M:%S%.3f",strict=False),
            ).alias("m_timestamp")
        )
        #Traces. Only Epoch time available
        self.df_trace = self.df_trace.with_columns(m_timestamp = pl.from_epoch(pl.col("StartTimeUnixNano"), time_unit="ns"))
        self.df_trace = self.df_trace.with_columns(m_timestamp_end = pl.from_epoch(pl.col("EndTimeUnixNano"), time_unit="ns"))
        #Metric
        self.df_metric_default = self.df_metric_default.with_columns(
            m_timestamp =  pl.col('Time').str.split(" +0000").list[0]
            )   
        self.df_metric_default = self.df_metric_default.with_columns(
            m_timestamp =  pl.col('m_timestamp').str.strptime(pl.Datetime, "%F %H:%M:%S%.9f") 
            )                                                                 
        #Labels
        self.df_label = self.df_label.with_columns(m_timestamp =  pl.col('inject_time').str.strptime(pl.Datetime, "%F %H:%M:%S"))

    def add_labels_to_metrics(self):
        # Join labels with metrics
        df_metrics_labels = self.df_label.lazy().join(
            self.df_metric_default.lazy(),
            left_on="inject_pod",
            right_on="PodName",
            how="inner",
            suffix="_metric"
        ).collect(streaming=True)

        # Calculate full anomaly flag
        df_metrics_labels = df_metrics_labels.with_columns(
            pl.when((pl.col("m_timestamp_metric") >= pl.col("m_timestamp+1")) &
                    (pl.col("m_timestamp_metric") <= pl.col("m_timestamp+3")))
            .then(True)
            .otherwise(False)
            .alias("is_full_anomaly")
        )

        # Calculate anomaly ratio for early metrics
        df_metrics_labels_early = df_metrics_labels.filter(
            (pl.col("m_timestamp_metric") > pl.col("m_timestamp")) &
            (pl.col("m_timestamp_metric") < pl.col("m_timestamp+1"))
        ).with_columns(
            ((pl.col("m_timestamp_metric") - pl.col("m_timestamp")) / pl.duration(minutes=1)).alias("ano_ratio")
        )

        # Calculate anomaly ratio for late metrics
        df_metrics_labels_late = df_metrics_labels.filter(
            (pl.col("m_timestamp_metric") > pl.col("m_timestamp+3")) &
            (pl.col("m_timestamp_metric") < pl.col("m_timestamp+4"))
        ).with_columns(
            ((pl.col("m_timestamp+4") - pl.col("m_timestamp_metric")) / pl.duration(minutes=1)).alias("ano_ratio")
        )

        # Combine early and late metrics
        df_metrics_labels_combined = df_metrics_labels_early.vstack(df_metrics_labels_late)

        # Handle full anomalies
        df_metrics_labels_full = df_metrics_labels.filter(pl.col("is_full_anomaly")).with_columns(pl.lit(1.0).alias("ano_ratio"))

        # Stack all together
        df_metrics_labels_final = df_metrics_labels_full.vstack(df_metrics_labels_combined)

        # Select relevant columns
        df_anomalies = df_metrics_labels_final.select(["row_nr", "is_full_anomaly", "ano_ratio", "inject_type"])

        # Join and update the default metrics DataFrame
        self.df_metric_default = self.df_metric_default.join(
            df_anomalies, 
            on="row_nr", 
            how="left"
        ).with_columns([
            pl.col("is_full_anomaly").fill_null(False),
            pl.col("ano_ratio").fill_null(0),
            ((pl.col("inject_type") == "cpu_consumed") |
            (pl.col("inject_type") == "network_delay") |
            (pl.col("inject_type") == "cpu_contention")).alias("metric_anomaly")
        ])

    def add_labels_to_df(self, df_to_modify):
        df_logs_labels_anos = self.df_label.lazy().join(
                df_to_modify.lazy(),
                left_on="inject_pod",
                right_on="PodName",
                how="inner",
                suffix="_log"
            ).filter(
            (pl.col("m_timestamp_log") > pl.col("m_timestamp")) &
            (pl.col("m_timestamp_log") <= pl.col("m_timestamp+3"))
            ).collect(streaming=True)

        df_logs_labels_anos = df_logs_labels_anos.with_columns(pl.lit(True).alias("anomaly"))
        df_logs_labels_anos = df_logs_labels_anos.select(["row_nr", "anomaly", "inject_type"])

        df_to_modify = df_to_modify.join(
            df_logs_labels_anos, 
            on="row_nr", 
            how="left"
        )
        df_to_modify = df_to_modify.with_columns(pl.col("anomaly").fill_null(False)) 
        df_to_modify = df_to_modify.with_columns(
            (
                (pl.col("inject_type") == "cpu_consumed") |
                (pl.col("inject_type") == "network_delay") |
                (pl.col("inject_type") == "cpu_contention")
            ).alias("metric_anomaly")
        )
        return df_to_modify

    def add_labels_to_df_seq(self, df_for_agg):   
        #Also system type 
        df_seq = df_for_agg.select(pl.col("seq_id")).unique()  
        df_temp = df_for_agg.group_by('seq_id').agg(
        (pl.col("anomaly").sum()/ pl.count()).alias("ano_count")) 
        # Join this result with df_sequences on seq_id
        df_seq = df_seq.join(df_temp, on='seq_id')
        return df_seq

    def merge_logs_and_traces(self):
        self.df_trace = self.df_trace.with_columns(pl.col("seq_id").str.replace_all("_trace.csv", ""))
        self.df = self.df.with_columns(pl.col("seq_id").str.replace_all("_log.csv", ""))
        self.df_merge = self.df .join(self.df_trace, on="seq_id", how="inner")
        return self.df_merge

# full_data = "/home/mmantyla/Datasets"
# loader = NezhaLoader(filename= f"{full_data}/nezha/",) 
# df = loader.execute()
# loader.df.describe()
# loader.df_trace.describe()
# loader.df_metric_default.describe()

