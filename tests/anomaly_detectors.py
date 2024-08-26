import glob
import os
import yaml
import polars as pl
import argparse

from loglead import AnomalyDetector

# Set up argument parser
parser = argparse.ArgumentParser(description='Dataset Loader Configuration')
parser.add_argument('--config', type=str, default='datasets.yml', help='Path to the YAML file containing dataset information. Default is datasets.yml.')
args = parser.parse_args()

# Read the configuration file
config_file = args.config
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

test_data_path = os.path.expanduser(config['root_folder'])
test_data_path = os.path.join(test_data_path, "test_data") 
 
# Get all .parquet files in the directory
all_files = glob.glob(os.path.join(test_data_path, "*.parquet"))

print("Anomaly detectors test starting.")
print(f"Found files {all_files}")

# Extract unique dataset names with '_eh_' , excluding '_seq' files
datasets = set()
for f in all_files:
    basename = os.path.basename(f)
    if "_lo" in basename and "_eh" in basename and "_seq" not in basename:
        dataset_name = basename.replace(".parquet", "")
        datasets.add(dataset_name)

cols_event = ["m_message", "e_words", "e_event_drain_id", "e_trigrams", "e_event_tip_id", "e_event_lenma_id", "e_bert_emb"] 
numeric_cols = ["seq_len", "eve_len_max", "duration_sec", "eve_len_over1", "nep_prob_nmax_avg", "nep_prob_nmax_min"]

def run_anomaly_detectors(df, cols_event, numeric_cols, test_frac):
    for col in cols_event:
        disabled_methods = set()
        disabled_methods.add("train_LOF")  # Too slow. Disabled to make tests run faster
        disabled_methods.add("train_OneClassSVM")  # Too slow. Disabled to make tests run faster
        if col == "m_message":
            disabled_methods.add("train_OOVDetector")
        if ((col == "e_trigrams" or col == "m_message") and "nezha_tt" in df.columns):
            disabled_methods.add("train_RF")  # Disable this slow combo.
        if col in df.columns and "anomaly" in df.columns and df["normal"].sum() > 9 and df["anomaly"].sum() > 9:
            print(f"Running anomaly detectors with {col}")
            sad = AnomalyDetector(item_list_col=col, print_scores=False, store_scores=True)
            sad.test_train_split(df, test_frac=test_frac) 
            sad.evaluate_all_ads(disabled_methods=disabled_methods)
        else:
            print (f"Skipped column {col}. Missing, no anomaly label or too low number (<10) of normal ({df['normal'].sum()}) or anomaly ({df['anomaly'].sum()}) instances")    


    # Because this requires all the determined columns, it inherently only runs with sequenced dfs
    all_columns_exist = all(column in df.columns for column in numeric_cols)
    if "anomaly" in df.columns and df["normal"].sum() > 9 and df["anomaly"].sum() > 9 and all_columns_exist:
        print(f"Running seqeuence anomaly detectors with numeric columns {numeric_cols}")
        disabled_methods = {"train_RarityModel", "train_OOVDetector"}
        sad = AnomalyDetector(numeric_cols=numeric_cols, print_scores=False, store_scores=True)
        sad.test_train_split(df, test_frac=test_frac) 
        sad.evaluate_all_ads(disabled_methods=disabled_methods)

for dataset in datasets:
    dataset_config = next((d for d in config['datasets'] if d['name'] in dataset), None)

    if dataset_config and not dataset_config.get('anomaly_detection', True):
        print(f"Skipping anomaly detection for dataset: {dataset}")
        continue

    # Load the event level data
    primary_file = os.path.join(test_data_path, f"{dataset}.parquet")
    seq_file = primary_file.replace(f"{dataset}.parquet", f"{dataset}_seq.parquet")

    if os.path.exists(seq_file):
        df_seq = pl.read_parquet(seq_file)
        print(f"Running sequence anomaly detectors with {seq_file}")
        run_anomaly_detectors(df_seq, cols_event, numeric_cols, test_frac=0.2)
    else:
        df = pl.read_parquet(primary_file)
        print(f"Running event anomaly detectors with {primary_file}")
        run_anomaly_detectors(df, cols_event, numeric_cols, test_frac=0.5)

print("Anomaly detectors test complete.")