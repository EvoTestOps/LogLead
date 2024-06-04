import sys
import psutil
import os
import yaml
import argparse
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
LOGLEAD_PATH = os.environ.get("LOGLEAD_PATH")
sys.path.append(os.environ.get("LOGLEAD_PATH"))
from loglead.loaders import BGLLoader, ThuSpiLibLoader, HDFSLoader, HadoopLoader, ProLoader, NezhaLoader

# Set up argument parser
parser = argparse.ArgumentParser(description='Dataset Loader Configuration')
parser.add_argument('--config', type=str, default='datasets.yml', help='Path to the YAML file containing dataset information. Default is datasets.yml.')
args = parser.parse_args()

# Read the configuration file
config_file = args.config
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

full_data_path = os.path.expanduser(config['root_folder'])

memory_limit_TB = 50
memory_limit_NEZHA_WS = 17
memory = psutil.virtual_memory().available / (1024 ** 3)
memory = round(memory, 2)  

print(f"Loaders test starting. Memory available: {memory}GB. Data folder: {full_data_path}")
def create_correct_loader(dataset_name, data, system=""):
    loader = None
    if 'log_file' in data:
        default_path = os.path.join(full_data_path,dataset_name, data['log_file'])
    else:
        default_path = os.path.join(full_data_path,dataset_name)

    if dataset_name == "hdfs":
        loader = HDFSLoader(filename=default_path,
                            labels_file_name=os.path.join(full_data_path,dataset_name, data['labels_file']))
    elif dataset_name == "tb":  # Must have gbs for TB
        if memory > memory_limit_TB:
            loader = ThuSpiLibLoader(filename=default_path)
        else:
             print("Skipping Thunderbird due to memory limit")
    elif dataset_name == "spirit": 
        if memory > memory_limit_TB:
            loader = ThuSpiLibLoader(filename=default_path)
        else:
            print("Skipping Spirit due to memory limit") 
    elif dataset_name == "liberty": 
        if memory > memory_limit_TB:
            loader = ThuSpiLibLoader(filename=default_path, split_component=False)
        else:
            print("Skipping Liberty due to memory limit") 
    elif dataset_name == "bgl":
        loader = BGLLoader(filename=default_path)
    elif dataset_name == "profilence":
        loader = ProLoader(filename=default_path)
    elif dataset_name == "hadoop":
        loader = HadoopLoader(filename=default_path,
                              filename_pattern=data['filename_pattern'],
                              labels_file_name=os.path.join(full_data_path, dataset_name, data['labels_file']))
    elif dataset_name == "nezha":
        if system == "WebShop" and memory <memory_limit_NEZHA_WS:
            print("Skipping Nezha WebShop due to memory limit")   
        else:
            loader = NezhaLoader(filename=default_path,
                                system=system)
        
    return loader

def check_and_save(dataset, loader, system=""):
    # Create a test data folder
    test_data_path = os.path.join(full_data_path, "test_data")
    os.makedirs(test_data_path, exist_ok=True)
    if dataset == "hdfs":
        if len(loader.df) != 11175629:
            print(f"MISMATCH! hdfs expected 11175629 was {len(loader.df)}. Perhaps old version of data?")
        loader.reduce_dataframes(frac=0.01)  # Reduce all to roughly 100k rows
        
    elif dataset == "tb":  # Must have gbs for TB
        if memory > memory_limit_TB:
            if len(loader.df) != 211212192:
                print(f"MISMATCH tb expected 211212192 was {len(loader.df)}. Perhaps old version of data?")
            loader.reduce_dataframes(frac=0.0005)
        else:
            if len(loader.df) != 2000:
                print(f"MISMATCH tb expected 2000 was {len(loader.df)}. Perhaps old version of data?")
    elif dataset == "spirit" or dataset == "liberty":  # Must have gbs for TB
        loader.reduce_dataframes(frac=0.0005)
    elif dataset == "bgl":
        if len(loader.df) != 4747963:
            print(f"MISMATCH bgl expected 4747963 was {len(loader.df)}. Perhaps old version of data?")
        loader.reduce_dataframes(frac=0.02)  # Reduce all to roughly 100k rows
    elif dataset == "profilence":
        if len(loader.df) != 5203599:
            print(f"MISMATCH pro expected 5203599 was {len(loader.df)}. Perhaps old version of data?")
        loader.reduce_dataframes(frac=0.02)  # Reduce all to roughly 100k rows
    elif dataset == "hadoop":
        if len(loader.df) != 180897:
            print(f"MISMATCH hadoop expected 180897 was {len(loader.df)}. Perhaps old version of data?")
        loader.reduce_dataframes(frac=0.56)  # Reduce all to roughly 100k rows
    elif dataset == "nezha" and system == "TrainTicket":
        if len(loader.df) != 272270:
            print(f"MISMATCH nezha_tt expected 272270 was {len(loader.df)}. Perhaps old version of data?")
        loader.reduce_dataframes(frac=0.33)  # Reduce all to roughly 100k rows
        dataset = dataset +"_tt"
    elif dataset == "nezha" and system == "WebShop":
        if len(loader.df) != 3958203:
            print(f"MISMATCH nezha_ws expected 3958203 was {len(loader.df)}. Perhaps old version of data?")
        loader.reduce_dataframes(frac=0.025)  # Reduce all to roughly 100k rows
        dataset = dataset +"_ws"
    else:
        print(f"Invalid dataset {dataset}")
        return
    # Save the data for enhancer tests.
    loader.df.write_parquet(f"{test_data_path}/{dataset}_lo.parquet") 
    if any(sub in dataset for sub in ["hdfs", "profilence", "hadoop"]):
        loader.df_seq.write_parquet(f"{test_data_path}/{dataset}_lo_seq.parquet")  

# Loop through the datasets in the configuration file
for dataset in config['datasets']:
    dataset_name = dataset['name']
    memory = psutil.virtual_memory().available / (1024 ** 3)
    print(f"Loading: {dataset_name}")
    if dataset_name == "nezha":
        for system in dataset['systems']:
            print(f"System: {system}")
            loader = create_correct_loader(dataset_name, dataset, system)
            if loader is None:
                continue
            loader.execute()
            check_and_save(dataset_name, loader, system)
    else:
        loader = create_correct_loader(dataset_name, dataset)
        if loader is None:
            continue
        loader.execute()
        check_and_save(dataset_name, loader)

print("Loading test complete.")
