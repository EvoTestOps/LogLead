#This file performs same actions in both LogLead and in LogParsers code. 
#LogParsers loading routine has been slightly modified. Modifications are explained in the separate file
 
import sys
import time
sys.path.append('..')
import polars as pl

import loglead.loaders.base as load
import loglead.enhancer as er

#full_data = "/home/ubuntu/Datasets"
full_data = "/home/mmantyla/Datasets"

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


datasets = ["hadoop","hdfs", "bgl"]
#datasets = ["tb"]
# Assuming the function 'create_correct_loader' takes a string and returns an appropriate loader.
# And 'execute' method of the loader returns a DataFrame-like object.

# Dictionary to store execution times for each dataset

# Dictionary to store execution times and row counts for each dataset
dataset_info = {
    "without_masking": {dataset: {'times': [], 'row_count': None} for dataset in datasets},
    "with_masking": {dataset: {'times': [], 'row_count': None} for dataset in datasets},
}

for dataset_name in datasets:
    loader = create_correct_loader(dataset_name)
    df = loader.execute()
    
    # Store the row count for the current dataset
    row_count = df.shape[0]
    dataset_info["without_masking"][dataset_name]['row_count'] = row_count
    dataset_info["with_masking"][dataset_name]['row_count'] = row_count
    
    enhancer = er.EventLogEnhancer(df)

    # Running without drain_masking
    for _ in range(10):
        enhancer_copy = er.EventLogEnhancer(df)  # Use a fresh copy for each iteration
        time_start = time.time()
        df_temp = enhancer_copy.normalize()
        df_temp = enhancer_copy.parse_drain(reparse=True)
        elapsed_time = time.time() - time_start
        dataset_info["without_masking"][dataset_name]['times'].append(elapsed_time)

    # Running with drain_masking
    for _ in range(10):
        enhancer_copy = er.EventLogEnhancer(df)  # Use a fresh copy for each iteration
        time_start = time.time()
        df_temp = enhancer_copy.parse_drain(drain_masking=True, reparse=True)
        elapsed_time = time.time() - time_start
        dataset_info["with_masking"][dataset_name]['times'].append(elapsed_time)

# Process the times and print out results for each dataset
for dataset_name in datasets:
    avg_time_without_masking = sum(dataset_info["without_masking"][dataset_name]['times']) / len(dataset_info["without_masking"][dataset_name]['times'])
    avg_time_with_masking = sum(dataset_info["with_masking"][dataset_name]['times']) / len(dataset_info["with_masking"][dataset_name]['times'])

    row_count_without_masking = dataset_info["without_masking"][dataset_name]['row_count']
    row_count_with_masking = dataset_info["with_masking"][dataset_name]['row_count']

    print(f'{dataset_name} - Without Drainmasking: average time: {avg_time_without_masking:.2f} seconds. {row_count_without_masking} rows processed')
    print(f'{dataset_name} - With Drainmasking: average time: {avg_time_with_masking:.2f} seconds. {row_count_with_masking} rows processed')

 
    
    

