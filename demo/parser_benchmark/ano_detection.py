import gc
import time
import os
import sys
import argparse
import datetime
import random

from dotenv import dotenv_values
import polars as pl
import yaml

from loglead import AnomalyDetector
from loglead.loaders import *
from loglead.enhancers import EventLogEnhancer, SequenceEnhancer


#import warnings
#from sklearn.exceptions import ConvergenceWarning
#warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Ensure this always gets executed in the same location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Base path for datasets
envs = dotenv_values()
full_data = envs.get("LOG_DATA_PATH")
if not full_data:
    print("WARNING!: LOG_DATA_PATH is not set. This will most likely fail")

# Load the configuration
default_config = os.path.join(script_dir, 'ano_detection_config.yml')


# Adding argparse for command-line argument parsing
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-c', dest='config_path', type=str, default=default_config,
                    help='Configuration file path')
args = parser.parse_args()


def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found. Please provide a valid configuration file path using the '-c' option.")
        sys.exit(1)


config = load_config(args.config_path)


def proportion_to_float(proportion_str):
    numerator, denominator = map(int, proportion_str.split('/'))
    return numerator / denominator


def print_latex_table(storage, data_name, data_proportion, redraws, metric, score_type, repeats, chrono_order,
                      normalize, rows, anos, test_frac):
    # Print the modified LaTeX table
    # Get the current timestamp and format it as a string
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # Create the filename using the timestamp
    filename = f"Table_{data_name}_{timestamp}_latex.txt"
    # Open the file in write mode
    with open(filename, "w") as f:
        chrono_order_str = "chronological order" if chrono_order else "not chronological order"
        normalized_str = "normalized" if normalize else "not normalized"
        for score in score_type:
            for meter in metric:
                latex = storage.calculate_average_scores(score_type=score, metric=meter).to_latex()
                # Write the LaTeX table to the file
                f.write(f"\\begin{{table*}}[]\n")
                f.write(f"\\caption{{Anomaly detection data:{data_name}, score:{score}, metric:{meter}, data-proportion:{data_proportion}, data-rows:{rows}, anos:{anos} redraws:{redraws}, train-test-repeats:{repeats}, test-fraction: {test_frac}, chono-order:{chrono_order}, normalize:{normalize}}}\n")
                f.write("\\centering\n")
                f.write(latex)
                f.write("\\end{table*}\n")
                f.write("\n")
                # Print a message indicating that the table was written to the file
    print(f"LaTeX tables written to {filename}")


def handle_test_fraction(test_fraction):
    values_to_process = []

    if isinstance(test_fraction, dict):
        # Generate a range of values from min to max with the specified step
        min_value = test_fraction['min']
        max_value = test_fraction['max']
        step = test_fraction['step']
        current_value = min_value
        while current_value < max_value + 0.001:  #sometimes values read from disk or added come as 0.9499999 when desired  is 0.95
            rounded_value = round(current_value, 2)
            values_to_process.append(rounded_value)
            current_value += step

    elif isinstance(test_fraction, (float, int)):
        # Add the single value to the list
        values_to_process.append(test_fraction)
    else:
        print("ERROR: Unexpected format for test_fraction")

    return values_to_process


for dataset_name, dataset_info in config['datasets'].items():
    gc.collect()
    print(f"_________________________________________________________________________")
    print(f"Processing dataset: {dataset_name}")
    data_proportion = dataset_info['proportion']
    data_redraws = dataset_info['proportion_redraws']
    data_repeats = dataset_info['train_test_repeats']
    data_predict = dataset_info['predict']
    data_test_fraction = handle_test_fraction(dataset_info['test_fraction'])
    data_chrono_order = dataset_info['chronological_order']
    data_normalize = dataset_info['normalize']
 
    # Dynamic dataset loading based on configuration
    loader_class = eval(f"{dataset_info['loader']}")
    loader_args = {key: full_data + value if key.endswith('name') else value 
                   for key, value in dataset_info.items() if key == 'filename' or key =='filename_pattern' or key =='labels_file_name'}
    loader = loader_class(**loader_args)
    time_start = time.time()
    loader.execute()
    time_elapsed = time.time() - time_start
    print(f'Data {dataset_name} loaded in {time_elapsed:.2f} seconds')
    # In Hadoop anomaly is majority class. Flip it to keep F1-binary equally difficult.
    if dataset_name == "Hadoop":
        loader.df_seq = loader.df_seq.with_columns(~pl.col("anomaly"))
    sad = AnomalyDetector(store_scores=True, print_scores=False, auc_roc=True)
    for i in range(data_redraws):
        proportion = proportion_to_float(data_proportion)
        if proportion < 1:  # Create a new baseloader for reducing dataframes.
            loader_redu = BaseLoader(filename="Dummy", df=loader.df, df_seq=loader.df_seq)
            loader_redu.reduce_dataframes(frac=proportion,  random_state=random.randint(0, 100000))
        else:
            loader_redu = loader
        df = loader_redu.df
        df_seq = loader_redu.df_seq

        #TODO might be a problem as we remove null after sequence creation. Well not really. Sequences are created in the enhancer
        #if dataset_name =="Bgl":
        #Kill nulls if they still exist
        df = df.filter(pl.col("m_message").is_not_null())

        enhancer = EventLogEnhancer(df)
        time_start = time.time()
        parser_params = None
        # In parsing do we normalize the input with regular expression or no
        if data_normalize:
            df = enhancer.normalize()
            df = enhancer.words(column="e_message_normalized")
            params = {'field': 'e_message_normalized'}
        else:
            df = enhancer.words(column="m_message")
            params = {'field': 'm_message'}
        time_elapsed = time.time() - time_start
        print(f'Data normalized and split to words redraws:{i+1}/{data_redraws}, proportion:{data_proportion}, rows:{df.height}, time {time_elapsed:.2f} seconds')
        import logging
        logging.basicConfig(level=logging.ERROR,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        logger = logging.getLogger(__name__)
        # Parse all for this dataset
        for parser in config['parsers']:
            parser_name = parser['name']
            parser_call = parser['call']
            parser_field = parser['field']
            # Parse
            enhancer = EventLogEnhancer(df)
            time_start = time.time()
            method_to_call = getattr(enhancer, parser_call)
            df = method_to_call(**params)
            time_elapsed = time.time() - time_start
            print(f'Data parsed parser:{parser_name}, rows:{df.height}, time:{time_elapsed}s')
            # Aggregate to sequences
            if data_predict == 'seq':
                seq_enhancer = SequenceEnhancer(enhancer.df, df_seq)
                df_seq = seq_enhancer.events(parser_field)
                df_to_predict = df_seq
            else:
                df_to_predict = df
        
        #print (f"Columns in df_to_predict: {df_to_predict.columns}")
        print(f'Predicting repeats:{data_repeats} data_rows:{df_to_predict.height}, anomalies:{df_to_predict["anomaly"].sum()} fraction for testing:{data_test_fraction}, chrono order:{data_chrono_order}')    
        for tf in data_test_fraction:
            print(f'Test fraction:{tf}')
            for j in range(data_repeats):
                print(f"Run:{j}", end=" ")
                time_start = time.time()
                first = True
                for parser in config['parsers']:
                    parser_name = parser['name']
                    parser_field = parser['field']
                    print(f"Parser:{parser_name}", end=" ")
                    sad.item_list_col = parser_field
                    if first:
                        sad.test_train_split(df_to_predict, test_frac=tf, shuffle=not data_chrono_order)
                        first = False
                    else:  # We keep the existing split but prepare a new parserfield
                        sad.prepare_train_test_data()
                    for algo in config['algos']:
                        algo_call = algo['call']
                        algo_name = algo['name']
                        print(f".", end="")
                        sys.stdout.flush()
                        #sad = AnomalyDetection(store_scores=True, print_scores=False)
                        time_algo_start = time.time()
                        method_to_call = getattr(sad, algo_call)
                        result = method_to_call()
                        sad.predict()
                        time_algo_start = time.time() - time_algo_start
                        print(f'algo train + predict:{algo_call} algo_time:{time_algo_start}')
                    #print(f" done")
                time_elapsed = time.time() - time_start
                print(f' time:{time_elapsed}')

    print_latex_table(sad.storage, dataset_name, data_proportion, score_type=["auc-roc", "f1"],
                      metric=["median", "mean"], redraws=data_redraws, repeats=data_repeats,
                      chrono_order=data_chrono_order, normalize=data_normalize, rows=df.height,
                      anos=df_to_predict["anomaly"].sum(), test_frac=data_test_fraction)
           