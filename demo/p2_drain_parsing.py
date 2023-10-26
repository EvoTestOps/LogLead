#This file performs same actions in both LogLEAD and in LogParsers code. 
# LogParsers loading routine has been slightly modified. Modifications are explained in the separate file
 
#Separate demo files
import sys
import time
sys.path.append('~/Development/mika/LogLEAD/')
import polars as pl

import loglead.loader as load
import loglead.enricher as er
import demo.p1_logparser_load as logparser

full_data = "/home/ubuntu/Datasets"
dataset ="hdfs"

datasets = ["hdfs", "hadoop", "bgl"]

def create_correct_loader(dataset):
    if dataset=="hdfs":
        loader = load.HDFSLoader(filename=f"{full_data}/hdfs/HDFS.log", 
                                            labels_file_name=f"{full_data}/hdfs/anomaly_label.csv")
    elif dataset=="tb":
        loader = load.ThunderbirdLoader(filename=f"{full_data}/thunderbird/Thunderbird.log") #Might take 2-3 minutes in HPC cloud. In desktop out of memory
    elif dataset=="bgl":
        loader = load.BGLLoader(filename=f"{full_data}/bgl/BGL.log")
    elif(dataset=="hadoop"):    
        loader = load.HadoopLoader(filename=f"{full_data}/hadoop/",
                                        filename_pattern  ="*.log",
                                        labels_file_name=f"{full_data}/hadoop/abnormal_label_accurate.txt")    
        
        
    return loader

loader = create_correct_loader("tb")
df = loader.execute()


enricher = er.EventLogEnricher(df)

# List to store execution times for each method
times_without_masking = []
times_with_masking = []

# Running without drain_masking
for _ in range(1):
    enricher_copy = er.EventLogEnricher(df)  # Assuming you want to use a fresh copy for each iteration
    time_start = time.time()
    df_temp = enricher_copy.normalize()
    df_temp = enricher_copy.parse_drain(reparse=True)
    elapsed_time = time.time() - time_start
    times_without_masking.append(elapsed_time)

# Running with drain_masking
for _ in range(1):
    enricher_copy = er.EventLogEnricher(df)  # Assuming you want to use a fresh copy for each iteration
    time_start = time.time()
    df_temp = enricher_copy.parse_drain(drain_masking=True, reparse=True)
    elapsed_time = time.time() - time_start
    times_with_masking.append(elapsed_time)

# Calculate average times
avg_time_without_masking = sum(times_without_masking) / len(times_without_masking)
avg_time_with_masking = sum(times_with_masking) / len(times_with_masking)

# Print out results
print(f'Without Drainmasking: LogLead  average time over 10 runs: {avg_time_without_masking:.2f} seconds. {df.shape[0]} rows processed')
print(f'With Drainmasking: LogLead  average time over 10 runs: {avg_time_with_masking:.2f} seconds. {df.shape[0]} rows processed')

