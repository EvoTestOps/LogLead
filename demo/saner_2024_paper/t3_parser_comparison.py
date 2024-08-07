#This file performs same actions in both LogLEAD and in LogParsers code. 
# LogParsers loading routine has been slightly modified. Modifications are explained in the separate file
import copy
import time
import statistics

from dotenv import dotenv_values

from loglead.enhancers import EventLogEnhancer
from loglead.loaders import HDFSLoader

# Adjust full data source
envs = dotenv_values()
full_data = envs.get("LOG_DATA_PATH")


loader = HDFSLoader(filename=f"{full_data}/hdfs/HDFS.log",
                    labels_file_name=f"{full_data}/hdfs/anomaly_label.csv")


loader.execute()

df50k = copy.deepcopy(loader).reduce_dataframes(0.005)
df100k = copy.deepcopy(loader).reduce_dataframes(0.01)
df200k = copy.deepcopy(loader).reduce_dataframes(0.02)

datas = [df50k, df100k, df200k]
# parsers = ["drain", "spell", "lenma", "neural"]
# parsers = ["neural"]
parsers = ["drain", "spell", "lenma"]

for data in datas:
    df = data
    for parser in parsers:
        elapsed_times = []
        for _ in range(10):
            enricher = EventLogEnhancer(df)
            start_time = time.time()
            df = enricher.normalize()
            if parser == "drain":
                df_parsers = enricher.parse_drain()
            elif parser == "spell":
                df_parsers = enricher.parse_spell()
            elif parser == "lenma":
                # LenMa needs words not just the message
                df = enricher.words(column="e_message_normalized")
                df_parsers = enricher.parse_lenma()
            elif parser == "neural":
                df_parsers = enricher.create_neural_emb()
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_times.append(elapsed_time)
            print(".", end="")
        print ("Done.")
        mean_time = statistics.mean(elapsed_times)
        median_time = statistics.median(elapsed_times)

        elapsed_str = ', '.join([f"{time:.2f}" for time in elapsed_times])
        print(f"Individual run times for {parser} (seconds): {elapsed_str}")
        print(f"Mean time over 10 runs for {parser}: {mean_time:.2f} seconds")
        print(f"Median time over 10 runs for {parser}: {median_time:.2f} seconds")
        print("-------------------------------------------------------------")
