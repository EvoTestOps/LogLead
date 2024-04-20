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
default_config = '/home/mmantyla/LogLead/demo/parser_benchmark/speed_config.yml'
#default_config = '/home/mmantyla/LogLead/demo/parser_speed_tests/config_missing.yml'
default_threshold = 600 #How many seconds is the threshold after which remaining runs for the parser are skipped
#config_path = '/home/mmantyla/LogLead/demo/parser_speed_tests/config_fiplom.yml'

# Adding argparse for command-line argument parsing
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-c', dest='config_path', type=str, default=default_config,
                    help='Configuration file path')
parser.add_argument('-t', dest='threshold_seconds', type=int, default=default_threshold,
                    help='Threshold in seconds')
args = parser.parse_args()



def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config(args.config_path)

def proportion_to_float(proportion_str):
    numerator, denominator = map(int, proportion_str.split('/'))
    return numerator / denominator

def is_excluded(parser, dataset, proportion, exclusions):
    # Check if the proportion is a string and convert it to a float if necessary
    if isinstance(proportion, str):
        prop_numer, prop_denom = map(int, proportion.split('/'))
        proportion = prop_numer / prop_denom
    # If proportion is not a string, it's assumed to be a float already

    for exclusion in exclusions:
        if exclusion['parser'] == parser and exclusion['dataset'] == dataset:
            # Convert the exclusion proportion to a numerical value for comparison
            excl_numer, excl_denom = map(int, exclusion['proportion'].split('/'))
            excl_value = excl_numer / excl_denom

            # Compare the input proportion (now a float) with the exclusion value
            if proportion >= excl_value:
                return True
    return False

# Example usage remains the same



# Main processing loop
for dataset_name, dataset_info in config['datasets'].items():
    gc.collect()
    print(f"_________________________________________________________________________")
    print(f"Processing dataset: {dataset_name}")
    
    # Dynamic dataset loading based on configuration
    loader_class = getattr(loaders, dataset_info['loader'])
    loader_args = {key: full_data + value if key.endswith('name') else value 
                   for key, value in dataset_info.items() if key != 'loader'}
    loader = loader_class(**loader_args)
    time_start = time.time()

    loader.execute()
    time_elapsed = time.time() - time_start
    print(f'Data {dataset_name} loaded in {time_elapsed:.2f} seconds')
    df = loader.df

    #if dataset_name =="Bgl":
    #Kill nulls if they still exist
    df = df.filter(pl.col("m_message").is_not_null())
    #if dataset_name == "Nezha-Shop":
    #    #Select correct Nezha system
    #    df = df.filter(pl.col("system_name") == "GShop")
    #elif dataset_name == "Nezha-TrainTicket":
    #    df = df.filter(pl.col("system_name") == "TrainTicket")
    df = df.select(pl.col("m_message"))
    
    enhancer = er.EventLogEnhancer(df)
    time_start = time.time()
    df = enhancer.normalize()
    df = enhancer.words(column="e_message_normalized")
    time_elapsed = time.time() - time_start
    print(f'Data normalized and split to words {time_elapsed:.2f} seconds')
    import logging
    logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)


    for parser in config['parsers']:
        parser_name = parser['name']
        parser_call = parser['call']
        parser_field = parser['field']
        for proportion_str in config['proportions']:
            proportion = proportion_to_float(proportion_str)
            
            if is_excluded(parser_name, dataset_name, proportion, config.get('exclusions', [])):
                print(f"Skipping excluded combination: {parser_name}, {dataset_name}, {proportion_str}")
                continue
            
            # Reduce the dataframe according to the current fraction
            # gc.collect() might be necessary if memory management is a concern
            loader_class = getattr(loaders, "BaseLoader")
            loader2 = loader_class(df=df, filename="Fake")
            df_reduced = loader2.reduce_dataframes(frac=proportion)
            
            # Parse
            enhancer_reduced = er.EventLogEnhancer(df_reduced)
            time_start = time.time()
            
            method_to_call = getattr(enhancer_reduced, parser_call)
            result = method_to_call()
            
            time_elapsed = time.time() - time_start
              
            print(f'### data:{dataset_name} parser:{parser_name} fraction:{proportion_str} rows:{df_reduced.shape[0]} time:{time_elapsed:.2f} events:{len(enhancer_reduced.df[parser_field].unique())}')
           
            if time_elapsed > args.threshold_seconds:
                print(f'Skipping remaining proportions as the last run exceeded the threshold of {args.threshold_seconds} seconds.')
                break  # Exit the loop over proportions

            # New addition: Select up to three unique templates from the parser_field column.
            #unique_templates = enhancer_reduced.df[parser_field].unique()
            #templates_to_show = unique_templates[:3]  # This slices the first three elements, if they exist.
            # Printing the templates.
            # Join the templates with a separator (e.g., ", ") to make them a single string for printing.
            #templates_str = ", ".join(map(str, templates_to_show))  # Convert each template to string in case they are not.
            #print(f'Templates: {templates_str}')
    



        # for proportion_str in config['proportions']:
        #     proportion = proportion_to_float(proportion_str)
            
        #     if is_excluded(parser_name, dataset_name, proportion, config.get('exclusions', [])):
        #         print(f"Skipping excluded combination: {parser_name}, {dataset_name}, {proportion_str}")
        #         continue
            
        #     #print(f"Applying parser {parser_name} on dataset {dataset_name} with proportion {proportion_str}")
            
        #     # Reduce the dataframe according to the current fraction
        #     #gc.collect()
        #     loader_class = getattr(loaders, "BaseLoader")
        #     loader2 = loader_class(df=df, filename="Fake")
        #     df_reduced = loader2.reduce_dataframes(frac=proportion)
        #     # Parse
        #     enhancer_reduced = er.EventLogEnhancer(df_reduced)
        #     time_start = time.time()
        #     # Example: enhancer_reduced.parse_brain()  
        #     # Replaced with dynamic parser call
        #     method_to_call = getattr(enhancer_reduced, parser_call)
        #     result = method_to_call()
        #     time_elapsed = time.time() - time_start
        #     print(f'{dataset_name} {parser_name} for fraction {proportion_str}: {time_elapsed:.2f} seconds found {len(enhancer_reduced.df[parser_field].unique())} events ')
