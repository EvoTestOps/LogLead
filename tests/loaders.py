#This file performs same actions in both LogLEAD and in LogParsers code. 
# LogParsers loading routine has been slightly modified. Modifications are explained in the separate file
 
#Separate demo files
import sys
import time

import polars as pl
import psutil
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append('..')
import loglead.loader as load
import loglead.nezha_loader as nezha_loader

# Base directory for datasets
full_data = "/home/mmantyla/Datasets"
full_data = "/home/ubuntu/Datasets"

memory_limit_TB = 40

print ("Loaders test starting.")
# Dictionary to store file paths for each dataset
# If you are missing any datasets. Comment them out. 
data_files = {
    "hadoop": {
        "log_dir": f"{full_data}/hadoop/",
        "filename_pattern": "*.log",
        "labels_file": f"{full_data}/hadoop/abnormal_label_accurate.txt"
    },
    "bgl": {
        "log_file": f"{full_data}/bgl/BGL.log"
    },
    "pro":{
        "log_file": f"{full_data}/profilence/*.txt"
    },
    "hdfs": {
        "log_file": f"{full_data}/hdfs/HDFS.log",
        "labels_file": f"{full_data}/hdfs/anomaly_label.csv"
    },
    "tb": {
        "log_file": f"{full_data}/thunderbird/Thunderbird.log",
        "log_file_2k": f"{full_data}/thunderbird/Thunderbird_2k.log"
    },
    "nezha": {
        "log_file":f"{full_data}/nezha/"
        }
}

def create_correct_loader(dataset):
    if dataset == "hdfs":
        loader = load.HDFSLoader(filename=data_files["hdfs"]["log_file"], 
                                 labels_file_name=data_files["hdfs"]["labels_file"])
    elif dataset == "tb": #Must have gbs for TB
        if memory > memory_limit_TB:
            # Might take 4-6 minutes in HPC VM. In desktop out of memory
            loader = load.ThunderbirdLoader(filename=data_files["tb"]["log_file"]) 
        else:
            loader = load.ThunderbirdLoader(filename=data_files["tb"]["log_file_2k"]) 
    elif dataset == "bgl":
        loader = load.BGLLoader(filename=data_files["bgl"]["log_file"])
    elif dataset=="pro":
       loader = load.ProLoader(filename=data_files["pro"]["log_file"])
    elif dataset == "hadoop":
        loader = load.HadoopLoader(filename=data_files["hadoop"]["log_dir"],
                                   filename_pattern=data_files["hadoop"]["filename_pattern"],
                                   labels_file_name=data_files["hadoop"]["labels_file"]) 
    elif dataset == "nezha":
        loader = nezha_loader.NezhaLoader(filename= data_files["nezha"]["log_file"],) 
    return loader


def check_and_save(dataset, loader):
    #Create a test datafolder
    test_data_path = os.path.join(full_data, "test_data")
    os.makedirs(test_data_path, exist_ok=True)
    if dataset == "hdfs":
        if len(loader.df) != 11175629:
            print(f"MISMATCH! hdfs expected 11175629 was {len(loader.df)}")
        loader.reduce_dataframes(frac=0.01) #Reduce all to roughly 100k rows
        
    elif dataset == "tb": #Must have gbs for TB
        if memory > memory_limit_TB:
            if len(loader.df) != 211212192:
                print(f"MISMATCH tb expected 211212192 was {len(loader.df)}")
            loader.reduce_dataframes(frac=0.0005)
        else:
            if len(loader.df) != 2000:
                print(f"MISMATCH tb expected 2000 was {len(loader.df)}")    
    elif dataset == "bgl":
        if len(loader.df) != 4747963:
            print(f"MISMATCH bgl expected 4747963 was {len(loader.df)}")
        loader.reduce_dataframes(frac=0.02) #Reduce all to roughly 100k rows
    elif dataset == "pro":
        if len(loader.df) != 5203599:
            print(f"MISMATCH pro expected 5203599 was {len(loader.df)}")
        loader.reduce_dataframes(frac=0.02) #Reduce all to roughly 100k rows
    elif dataset == "hadoop":
        if len(loader.df) != 177592:
            print(f"MISMATCH hadoop expected 177592 was {len(loader.df)}")
        loader.reduce_dataframes(frac=0.56) #Reduce all to roughly 100k rows
    elif dataset == "nezha":
        if len(loader.df) != 4230473:
            print(f"MISMATCH nezha expected 4230473 was {len(loader.df)}")
        loader.reduce_dataframes(frac=0.02) #Reduce all to roughly 100k rows
    else:
        print(f"Invalid dataset {dataset}")
        return
    #Save the data used in enhancer tests. 
    loader.df.write_parquet(f"{test_data_path}/{dataset}_lo.parquet") 
    if any(sub in dataset for sub in ["hdfs", "pro", "hadoop"]):
        loader.df_seq.write_parquet(f"{test_data_path}/{dataset}_lo_seq.parquet")  


for key, value in data_files.items():
    memory = psutil.virtual_memory().available / (1024 ** 3)
    print(f"Processing: {key}")
    loader = create_correct_loader(key)
    loader.execute()
    check_and_save(key, loader)

print ("Loading test complete.")