import sys
import psutil
import os
import yaml
import argparse
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
LOGLEAD_PATH = os.environ.get("LOGLEAD_PATH")
sys.path.append(os.environ.get("LOGLEAD_PATH"))

from loglead.loaders import BGLLoader, ThuSpiLibLoader, HDFSLoader, HadoopLoader, ProLoader, NezhaLoader, ADFALoader, AWSCTDLoader

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
        if system == "WebShop" and memory < memory_limit_NEZHA_WS:
            print("Skipping Nezha WebShop due to memory limit")   
        else:
            loader = NezhaLoader(filename=default_path,
                                system=system)
    elif dataset_name == "adfa":
        default_path = default_path + "/ADFA-LD"
        loader = ADFALoader(filename=default_path)
    elif dataset_name == "awsctd":
        loader = AWSCTDLoader(filename=default_path+"/CSV")
        
    return loader

def check_and_save(dataset, loader, config, system=""):
    # Create a test data folder
    test_data_path = os.path.join(full_data_path, "test_data")
    os.makedirs(test_data_path, exist_ok=True)

    # Find the dataset configuration
    dataset_config = next((d for d in config['datasets'] if d['name'] == dataset), None)
    if not dataset_config:
        print(f"Invalid dataset {dataset}")
        return

    # Check system-specific configurations for Nezha
    if dataset == "nezha":
        if system == "TrainTicket":
            expected_length = dataset_config['train_ticket']['expected_length']
            reduction_fraction = dataset_config['train_ticket']['reduction_fraction']
            dataset = f"{dataset}_tt"
        elif system == "WebShop":
            expected_length = dataset_config['web_shop']['expected_length']
            reduction_fraction = dataset_config['web_shop']['reduction_fraction']
            dataset = f"{dataset}_ws"
        else:
            print(f"Invalid system for dataset {dataset}")
            return
    else:
        expected_length = dataset_config.get('expected_length')
        reduction_fraction = dataset_config.get('reduction_fraction')

    # Check and print mismatch if any
    if expected_length and len(loader.df) != expected_length and expected_length != 0:
        print(f"MISMATCH! {dataset} expected {expected_length} was {len(loader.df)}. Perhaps old version of data?")

    # Reduce data if reduction_fraction is specified
    if reduction_fraction:
        loader.reduce_dataframes(frac=reduction_fraction)

    # Save the data used for anomaly_detectors tests.
    loader.df.write_parquet(f"{test_data_path}/{dataset}_lo.parquet") 
    if any(sub in dataset for sub in ["hdfs", "profilence", "hadoop", "adfa", "awsctd"]):
        loader.df_seq.write_parquet(f"{test_data_path}/{dataset}_lo_seq.parquet")  

# Loop through the datasets in the configuration file
for dataset in config['datasets']:
    dataset_name = dataset['name']
    memory = psutil.virtual_memory().available / (1024 ** 3)

    skip_loader = not dataset.get('load', True)
    if skip_loader:
        print(f'Skipping loader for {dataset_name}.')
        continue

    print(f"Loading: {dataset_name}")
    if dataset_name == "nezha":
        for system in dataset['systems']:
            print(f"System: {system}")
            loader = create_correct_loader(dataset_name, dataset, system)
            if loader is None:
                continue
            loader.execute()
            check_and_save(dataset_name, loader, config, system)
    else:
        loader = create_correct_loader(dataset_name, dataset)
        if loader is None:
            continue
        loader.execute()
        check_and_save(dataset_name, loader, config)

print("Loading test complete.")
