# This file performs same actions in both LogLEAD and in LogParsers code.
# LogParsers loading routine has been slightly modified. Modifications are explained in the separate file
 
import time
import datetime

import polars as pl
import pandas as pd
import regex as re
from dotenv import dotenv_values

from loglead.loaders import *

# Adjust full data source
envs = dotenv_values()
full_data = envs.get("LOG_DATA_PATH")

logparser_format = {
    "hdfs": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
    "bgl": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
    # "tb" : "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>"
}

loglead_format = {
    "hdfs": ["date", "time", "id", "level", "component", "m_message"],
    "bgl": ["label", "timestamp", "date", "node", "time", "noderepeat", "type", "component", "level", "m_message"],
    # "tb" : ["label", "timestamp", "date", "userid", "month", "day", "time", "location", "component_pid", "m_message"],
    "hadoop": ["date", "time", "level", "component", "m_message"]
}


def log_to_dataframe(log_file, log_format):
    """Function to transform log file to dataframe
    Routine loads data from logfile and creates a dataframe.
    Routine taken from LogPai/Logpaser/Drain project
    https://github.com/logpai/logparser/blob/main/logparser/Drain/Drain.py#L327C1-L344C21
    Following modification were made.
    1) Converted to normal function instead of class member function
    2) Added a counter that reports progress every 1,000,000 rows
    3) Added errors=ignore as there is a non-utf character in TB.
    4) Moved generate_logformat_regex inside this function
    # =========================================================================
    # Copyright (C) 2016-2023 LOGPAI (https://github.com/logpai).
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # =========================================================================
    """
    def generate_logformat_regex(logformat):
        """Function to generate regular expression to split log messages
        Routine from https://github.com/logpai/logparser/blob/main/logparser/Drain/Drain.py#L346
        Changes
        1) Changed to normal function from class member function
        """
        headers = []
        splitters = re.split(r"(<[^<>]+>)", logformat)
        regex = ""
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(" +", "\\\s+", splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip("<").strip(">")
                regex += "(?P<%s>.*?)" % header
                headers.append(header)
        regex = re.compile("^" + regex + "$")
        return headers, regex

    headers, regex = generate_logformat_regex(log_format)
    log_messages = []
    linecount = 0
    #    with open(log_file, "r") as fin:
    with open(log_file, "r", errors="replace") as fin:
        for line in fin.readlines():
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                linecount += 1
                if linecount % 1000000 == 0:
                    print(f"{datetime.datetime.now().strftime('%H:%M:%S')}", flush=True)
            except Exception as e:
                continue
                #print("[Warning] Skip line: " + line)
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, "LineId", None)
    logdf["LineId"] = [i + 1 for i in range(linecount)]
    print("Total lines: ", len(logdf))
    return logdf


def create_correct_loader(dataset):
    if dataset == "hdfs":
        loader = HDFSLoader(filename=f"{full_data}/hdfs/HDFS.log",
                            labels_file_name=f"{full_data}/hdfs/anomaly_label.csv")
    elif dataset == "tb":
        # Might take 2-3 minutes in HPC cloud. In desktop out of memory
        loader = ThuSpiLibLoader(filename=f"{full_data}/thunderbird/Thunderbird.log")
    elif dataset == "bgl":
        loader = BGLLoader(filename=f"{full_data}/bgl/BGL.log")
    elif dataset == "hadoop":
        loader = HadoopLoader(filename=f"{full_data}/hadoop/",
                              filename_pattern="*.log",
                              labels_file_name=f"{full_data}/hadoop/abnormal_label_accurate.txt")
    return loader


for key, value in loglead_format.items():
    print(f"Processing: {key}, {value}")
    loglead_times = []
    for _ in range(10):
        print(f"r{_}", end=", ")
        time_start = time.time()
        loader = create_correct_loader(key)
        loader.load()
        if key == "hadoop":
            loader._merge_multiline_entries()
            loader._extract_process()
            # Store columns
            df_store_cols = loader.df.select(pl.col("seq_id", "seq_id_sub", "process", "row_nr", "column_1"))
            # This method splits the string and overwrite self.df
            loader._split_and_unnest(["date", "time", "level", "component", "m_message"])
            # Merge the stored columns back to self.df
            loader.df = pl.concat([loader.df, df_store_cols], how="horizontal")
        else:    
            loader._split_and_unnest(value)
        if key == "tb":
            loader._split_component_and_pid()
        loglead_times.append(time.time() - time_start)

    avg_loglead_time = sum(loglead_times) / len(loglead_times)
    print(f'LogLead {key} load average time over 10 runs: {avg_loglead_time:.2f} seconds. {loader.df.shape[0]} rows processed')


for key, value in logparser_format.items():
    print(f"Processing: {key}, {value}")
    logparser_times = []
    for _ in range(1):
        loader = create_correct_loader(key)
        time_start = time.time()
        df = log_to_dataframe(loader.filename, value)
        logparser_times.append(time.time() - time_start)

    avg_logparser_time = sum(logparser_times) / len(logparser_times)
    print(f'LogParser {key} load average time over 10 runs: {avg_logparser_time:.2f} seconds.{df.shape[0]} rows processed')
