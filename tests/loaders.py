
 
#Separate demo files
import sys

import polars as pl
import psutil
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append('..')
import loglead.loaders.base as load
import loglead.loaders.supercomputers as load_sc
import loglead.loaders.hdfs as load_hdfs
import loglead.loaders.hadoop as load_hadoop
import loglead.loaders.pro as load_pro
import loglead.loaders.nezha as load_nezha

home_directory = os.path.expanduser('~')
full_data = os.path.join(home_directory, "Datasets")

memory_limit_TB = 50
memory_limit_NEZHA_WS = 30


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
    "spirit": {
        "log_file": f"{full_data}/spirit/spirit2.log",
    },
    "liberty": {
        "log_file": f"{full_data}/liberty/liberty2.log",
    },
    "nezha_tt": {
        "log_file":f"{full_data}/nezha/",
        "system": "TrainTicket"
        },
   "nezha_ws": {
        "log_file":f"{full_data}/nezha/",
        "system": "WebShop"
        }
    
}

def create_correct_loader(dataset):
    loader = None
    if dataset == "hdfs":
        loader = load_hdfs.HDFSLoader(filename=data_files["hdfs"]["log_file"], 
                                 labels_file_name=data_files["hdfs"]["labels_file"])
    elif dataset == "tb": #Must have gbs for TB
        if memory > memory_limit_TB:
            # Might take 4-6 minutes in HPC VM. In desktop out of memory
            loader = load_sc.ThuSpiLibLoader(filename=data_files["tb"]["log_file"]) 
        else:
            loader = load_sc.ThuSpiLibLoader(filename=data_files["tb"]["log_file_2k"]) 
    elif dataset == "spirit": 
        if memory > memory_limit_TB:
            # Might take 4-6 minutes in HPC VM. In desktop out of memory
            loader = load_sc.ThuSpiLibLoader(filename=data_files["spirit"]["log_file"]) 
        else:
            print("Skipping spirit due to memory limit") 
    elif dataset == "liberty": 
        if memory > memory_limit_TB:
            # Might take 4-6 minutes in HPC VM. In desktop out of memory
            loader = load_sc.ThuSpiLibLoader(filename=data_files["liberty"]["log_file"], split_component=False) 
        else:
            print("Skipping liberty due to memory limit") 
    elif dataset == "bgl":
        loader = load_sc.BGLLoader(filename=data_files["bgl"]["log_file"])
    elif dataset=="pro":
       loader = load_pro.ProLoader(filename=data_files["pro"]["log_file"])
    elif dataset == "hadoop":
        loader = load_hadoop.HadoopLoader(filename=data_files["hadoop"]["log_dir"],
                                   filename_pattern=data_files["hadoop"]["filename_pattern"],
                                   labels_file_name=data_files["hadoop"]["labels_file"]) 
    elif dataset == "nezha_tt":
        loader = load_nezha.NezhaLoader(filename= data_files["nezha_tt"]["log_file"],
                                        system = data_files["nezha_tt"]["system"]) 
    elif dataset == "nezha_ws":
        if memory > memory_limit_NEZHA_WS:
            loader = load_nezha.NezhaLoader(filename= data_files["nezha_ws"]["log_file"],
                                            system = data_files["nezha_ws"]["system"])
        else:
            print("Skipping liberty due to nezha_ws limit")      
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
    elif dataset == "spirit" or dataset == "liberty": #Must have gbs for TB
        loader.reduce_dataframes(frac=0.0005)
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
    elif dataset == "nezha_tt":
        if len(loader.df) != 272270:
            print(f"MISMATCH nezha_tt expected 272270 was {len(loader.df)}")
        loader.reduce_dataframes(frac=0.33) #Reduce all to roughly 100k rows
    elif dataset == "nezha_ws":
        if len(loader.df) != 3958203:
            print(f"MISMATCH nezha_ws expected 3 958 203 was {len(loader.df)}")
        loader.reduce_dataframes(frac=0.025) #Reduce all to roughly 100k rows
    else:
        print(f"Invalid dataset {dataset}")
        return
    #Save the data used in enhancer tests. 
    loader.df.write_parquet(f"{test_data_path}/{dataset}_lo.parquet") 
    if any(sub in dataset for sub in ["hdfs", "pro", "hadoop"]):
        loader.df_seq.write_parquet(f"{test_data_path}/{dataset}_lo_seq.parquet")  


for key, value in data_files.items():
    memory = psutil.virtual_memory().available / (1024 ** 3)
    print(f"Loading: {key}")
    loader = create_correct_loader(key)
    if (loader == None):
        continue
    loader.execute()
    check_and_save(key, loader)

print ("Loading test complete.")