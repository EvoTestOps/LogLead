#RawLoader with Mufano data.

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

mufano_time_stamp_pattern = r'^(\d{2}:\d{2}:\d{2}.\d{3})'
mufano_chrono_format = "%H:%M:%S.%3f" #16:39:25.295

mufano_path = os.path.join(full_data, "mufano/")
mufano_path = os.path.join(full_data, "mufano/light-oauth2-data-1719592986/")

#Keep strategy for traces
loader = RawLoader(mufano_path, filename_pattern='*.log', strip_full_data_path=mufano_path,
    timestamp_pattern=mufano_time_stamp_pattern,  missing_timestamp_action="keep",
    timestamp_format = mufano_chrono_format, date_from_files=True)
df = loader.execute()

df.filter(pl.col("m_timestamp").is_null()) #See time stamp null rows
df
#df.filter(pl.col("m_timestamp").cast(pl.Time) > pl.time(hour=23, minute=30)) # see things after certain time point
