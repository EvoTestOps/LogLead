#
#Separate demo files
import sys
import time
sys.path.append('~/Development/mika/LogLEAD/')
import polars as pl

import loglead.loader as load
import demo.p1_logparser_load as logparser

full_data = "/home/ubuntu/Datasets"

#One runs---------------------------------------------------
#Execute the same actions for both. In Loglead this means excluding timestamp parsing, 
time_start = time.time()
loader = load.HDFSLoader(filename=f"{full_data}/hdfs/HDFS.log", labels_file_name=f"{full_data}/hdfs/anomaly_label.csv")
loader.load()
loader._split_and_unnest(["date", "time", "id", "level", "component", "m_message"])
print(f'LogLead HDFS load total time: {time.time()-time_start:.2f} seconds')
time_start = time.time()
df= logparser.log_to_dataframe(f"{full_data}/hdfs/HDFS.log")
print(f'LogParser HDFS load total time: {time.time()-time_start:.2f} seconds')



#10 Runs------------------------------------------------------------
# For LogLead HDFS
loglead_times = []
for _ in range(10):
    time_start = time.time()
    loader = load.HDFSLoader(filename=f"{full_data}/hdfs/HDFS.log", labels_file_name=f"{full_data}/hdfs/anomaly_label.csv")
    loader.load()
    loader._split_and_unnest(["date", "time", "id", "level", "component", "m_message"])
    loglead_times.append(time.time() - time_start)

avg_loglead_time = sum(loglead_times) / len(loglead_times)
print(f'LogLead HDFS load average time over 10 runs: {avg_loglead_time:.2f} seconds')

# For LogParser HDFS

logparser_times = []
for _ in range(10):
    time_start = time.time()
    df = logparser.log_to_dataframe(f"{full_data}/hdfs/HDFS.log")
    logparser_times.append(time.time() - time_start)

avg_logparser_time = sum(logparser_times) / len(logparser_times)
print(f'LogParser HDFS load average time over 10 runs: {avg_logparser_time:.2f} seconds')


logparser.generate_logformat_regex('<Date> <Time> <Pid> <Level> <Component>: <Content>')