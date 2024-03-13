#This file performs same actions in both LogLEAD and in LogParsers code. 
# LogParsers loading routine has been slightly modified. Modifications are explained in the separate file
 
#Separate demo files
import sys
import time
sys.path.append('..')
import polars as pl

import loglead.loaders.base as load
import demo.paper.t1_logparser_load as logparser

# Base directory for datasets
#full_data = "/home/ubuntu/Datasets"
full_data = "/home/mmantyla/Datasets"

logparser_format = {
    "hdfs" : "<Date> <Time> <Pid> <Level> <Component>: <Content>",
    "bgl" : "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
#    "tb" : "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>"
}

loglead_format = {
    "hdfs" : ["date", "time", "id", "level", "component", "m_message"],
    "bgl" : ["label", "timestamp", "date", "node", "time", "noderepeat", "type", "component", "level", "m_message"],
#    "tb" : ["label", "timestamp", "date", "userid", "month", "day", "time", "location", "component_pid", "m_message"],
    "hadoop" : ["date", "time","level", "component","m_message"]
}

def create_correct_loader(dataset):
    if dataset=="hdfs":
        loader = load.HDFSLoader(filename=f"{full_data}/hdfs/HDFS.log", 
                                            labels_file_name=f"{full_data}/hdfs/anomaly_label.csv")
    elif dataset=="tb":
        loader = load.ThunderbirdLoader(filename=f"{full_data}/thunderbird/Thunderbird.log") #Might take 2-3 minutes in HPC cloud. In desktop out of memory
    elif dataset=="bgl":
        loader = load.BGLLoader(filename=f"{full_data}/bgl/BGL.log")
    elif dataset=="hadoop":
        loader = load.HadoopLoader(filename=f"{full_data}/hadoop/",
                                                 filename_pattern  ="*.log",
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
        if (key == "hadoop"):
            loader._merge_multiline_entries()
            loader._extract_process()
            #Store columns 
            df_store_cols = loader.df.select(pl.col("seq_id","seq_id_sub", "process","row_nr", "column_1"))
            #This method splits the string and overwrite self.df
            loader._split_and_unnest(["date", "time","level", "component","m_message"])
            #Merge the stored columns back to self.df
            loader.df = pl.concat([loader.df, df_store_cols], how="horizontal")
        else:    
            loader._split_and_unnest(value)
        if (key == "tb"):
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
        df = logparser.log_to_dataframe(loader.filename, value)
        logparser_times.append(time.time() - time_start)

    avg_logparser_time = sum(logparser_times) / len(logparser_times)
    print(f'LogParser {key} load average time over 10 runs: {avg_logparser_time:.2f} seconds.{df.shape[0]} rows processed')
          

