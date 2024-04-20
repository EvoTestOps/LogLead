import gc
import time
import yaml
import sys
import polars as pl
import os
print(os.getcwd())
#sys.path.append('../..')
import sys
sys.path.append('/home/mmantyla/LogLead')
import loglead.loaders as loaders  # Assuming loaders are correctly organized
#from loglead.loaders.hadoop import HadoopLoader
import loglead.enhancer as er
import loglead.anomaly_detection as ad
import argparse

# Base path for datasets
full_data = "/home/mmantyla/Datasets"
# Load the configuration
default_config = '/home/mmantyla/LogLead/demo/parser_benchmark/ano_detection_config.yml'
#default_config = '/home/mmantyla/LogLead/demo/p1-fiplom_wip/config_missing.yml'
default_threshold = 600 #How many seconds is the threshold after which remaining runs for the parser are skipped
#config_path = '/home/mmantyla/LogLead/demo/p1-fiplom_wip/config_fiplom.yml'

# Adding argparse for command-line argument parsing
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-c', dest='config_path', type=str, default=default_config,
                    help='Configuration file path')
args = parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config(args.config_path)

def proportion_to_float(proportion_str):
    numerator, denominator = map(int, proportion_str.split('/'))
    return numerator / denominator

for dataset_name, dataset_info in config['datasets'].items():
    gc.collect()
    print(f"_________________________________________________________________________")
    print(f"Processing dataset: {dataset_name}")
    data_proportion = dataset_info['proportion']
    data_repeats = dataset_info['repeats']
    data_predict = dataset_info['predict']
    data_test_fraction = dataset_info['test_fraction']
    
    # Dynamic dataset loading based on configuration
    loader_class = getattr(loaders, dataset_info['loader'])
    loader_args = {key: full_data + value if key.endswith('name') else value 
                   for key, value in dataset_info.items() if key == 'filename' or key =='filename_pattern' or key =='labels_file_name'}
    loader = loader_class(**loader_args)
    time_start = time.time()
    loader.execute()
    time_elapsed = time.time() - time_start
    print(f'Data {dataset_name} loaded in {time_elapsed:.2f} seconds')
    proportion = proportion_to_float(data_proportion)
    if proportion < 1: 
        df = loader.reduce_dataframes(frac=proportion)
    else:
        df = loader.df
    df_seq = loader.df_seq

    #TODO might be a problem as we remove null after sequence creation. Well not really. Sequences are created in the enhancer
    #if dataset_name =="Bgl":
    #Kill nulls if they still exist
    df = df.filter(pl.col("m_message").is_not_null())

    enhancer = er.EventLogEnhancer(df)
    time_start = time.time()
    df = enhancer.normalize()
    df = enhancer.words(column="e_message_normalized")
    time_elapsed = time.time() - time_start
    print(f'Data normalized and split to words rows:{df.height}, time {time_elapsed:.2f} seconds')
    import logging
    logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    #Parse all for this dataset
    for parser in config['parsers']:
        parser_name = parser['name']
        parser_call = parser['call']
        parser_field = parser['field']
        # Parse
        enhancer = er.EventLogEnhancer(df)
        time_start = time.time()
        method_to_call = getattr(enhancer, parser_call)
        df = method_to_call()
        time_elapsed = time.time() - time_start
        print(f'Data parsed parser:{parser_name}, rows:{df.height}, time:{time_elapsed}s')
        #Aggregate to sequences
        if data_predict == 'seq':
            seq_enhancer = er.SequenceEnhancer(enhancer.df, df_seq)
            df_seq = seq_enhancer.events(parser_field)
            df_to_predict = df_seq
        else:
            df_to_predict = df
    
    
    sad = ad.AnomalyDetection(store_scores=True, print_scores=False, auc_roc=True)
        #print (f"Columns in df_to_predict: {df_to_predict.columns}")
    for algo in config['algos']:
        algo_call = algo['call']
        algo_name = algo['name']
        #sad = ad.AnomalyDetection(store_scores=True, print_scores=False)
        time_start = time.time()
        for i in range (data_repeats):
            first = True
            for parser in config['parsers']:
                    parser_field = parser['field']      
                    sad.item_list_col = parser_field
                    if first:
                        sad.test_train_split (df_to_predict, test_frac=data_test_fraction)
                    else: # We keep the existing split but prepare a new parserfield
                        sad.prepare_train_test_data()

                    method_to_call = getattr(sad, algo_call)
                    result = method_to_call()
                    sad.predict()
        time_elapsed = time.time() - time_start
        print(f'Predicted with algo:{algo_name} repeats:{data_repeats} fraction for testing:{data_test_fraction}, time:{time_elapsed}')

    print(f"Inspecting results. Averages of runs:")
    print(sad.storage.calculate_average_scores(score_type="auc_roc", metric="median").to_latex())
           