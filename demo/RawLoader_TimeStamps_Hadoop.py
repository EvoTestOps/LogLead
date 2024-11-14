#This file demonstrates working with RawLoader when you want get timestamps.
#This also exspl

import polars as pl
from loglead.loaders import RawLoader
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
full_data = os.getenv("LOG_DATA_PATH")
if not full_data:
    print("WARNING!: LOG_DATA_PATH is not set. This will most likely fail")

# Ensure this always gets executed in the same location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


hadoop_time_stamp_pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})'
hadoop_chrono_format = "%Y-%m-%d%H:%M:%S,%3f"
hadoop_path = os.path.join(full_data, "hadoop/")

#Keep strategy for traces
loader = RawLoader(hadoop_path, filename_pattern='*.log', strip_full_data_path=hadoop_path,
    timestamp_pattern=hadoop_time_stamp_pattern,  missing_timestamp_action="keep",
    timestamp_format = hadoop_chrono_format)
df = loader.execute()
print (f"Keep strategy total row count: {df.height}")
print (f"Keep strategy has normal log lines {df.filter(pl.col('m_timestamp').is_not_null())[0:2]}")
print (f"Keep strategy has traces as log lines {df.filter(pl.col('m_timestamp').is_null())[0:2]}")

#Fill lastseen strategy for traces
loader = RawLoader(hadoop_path, filename_pattern='*.log', strip_full_data_path=hadoop_path,
    timestamp_pattern=hadoop_time_stamp_pattern, missing_timestamp_action="fill-lastseen",
    timestamp_format = hadoop_chrono_format)
df = loader.execute()
print (f"Fill lastseen strategy total row count: {df.height}")
print (f"Fill lastseen strategy has normal log lines {df[0:2]}")
print (f"Fill lastseen strategy has traces as log lines with filled time stamp {df.filter(pl.col('m_message').str.starts_with('Container '))[0:2]}")

#Drop strategy for traces
loader = RawLoader(hadoop_path, filename_pattern='*.log', strip_full_data_path=hadoop_path,
    timestamp_pattern=hadoop_time_stamp_pattern, missing_timestamp_action="drop",
    timestamp_format = hadoop_chrono_format)
df = loader.execute()
print (f"Drop strategy total row count: {df.height}")
print (f"Drop strategy only has normal log lines {df.filter(pl.col('m_timestamp').is_not_null())[0:2]}")

#Merge strategy for traces
loader = RawLoader(hadoop_path, filename_pattern='*.log', strip_full_data_path=hadoop_path,
    timestamp_pattern=hadoop_time_stamp_pattern, missing_timestamp_action="merge",
    timestamp_format = hadoop_chrono_format)
df = loader.execute()
print (f"Merge strategy total row count: {df.height}")
print (f"Merge strategy has normal log lines {df.filter(pl.col('trace')=='')[0:2]}")
print (f"Merge strategy has log lines where trace is extra column {df.filter(pl.col('trace')!='')[0:2]}")

