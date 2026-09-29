import psutil
import os
import sys
import time
import traceback
import yaml
import argparse

import fingerprint as fp

from loglead.loaders import (AccessLogLoader, AutoLoader, BGLLoader, ThuSpiLibLoader, HDFSLoader,
                             HadoopLoader, ProLoader, NezhaLoader, ADFALoader, AWSCTDLoader,
                             DelimitedLoader, JsonLoader, LogfmtLoader, SyslogLoader, LO2Loader,
                             LO2v2Loader)

# Set up argument parser
parser = argparse.ArgumentParser(description='Dataset Loader Configuration')
parser.add_argument('--config', type=str, default='datasets_mid_labels.yml', help='Path to the YAML file containing dataset information. Default is datasets_mid_labels.yml.')
parser.add_argument('--baseline', choices=('check', 'capture', 'off'), default='check',
                    help='check: compare each loaded frame with tests/baselines/<config>.json (default). '
                         'capture: record the fingerprints as the new baseline. off: neither.')
parser.add_argument('--only', nargs='+', metavar='NAME', default=None,
                    help='Load only these datasets from the config.')
parser.add_argument('--keep-full', action='store_true',
                    help='Keep the full <name>_full.parquet a streamed dataset is written to '
                         '(deleted after sampling by default).')
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
baseline_file = fp.baseline_path(config_file)
baselines = fp.load_baselines(baseline_file)
failures = []
def create_correct_loader(dataset_name, data, system=""):
    loader = None
    if 'log_file' in data:
        default_path = os.path.join(full_data_path,dataset_name, data['log_file'])
    else:
        default_path = os.path.join(full_data_path,dataset_name)

    if dataset_name == "hdfs":
        loader = HDFSLoader(filename=default_path,
                            labels_file_name=os.path.join(full_data_path,dataset_name, data['labels_file']))
    elif dataset_name in ("thunderbird", "spirit"):  # Streamed when memory is short, see load_one()
        loader = ThuSpiLibLoader(filename=default_path)
    elif dataset_name == "liberty":
        loader = ThuSpiLibLoader(filename=default_path, split_component=False)
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
    elif dataset_name == "lo2":
        # LO2Loader chooses which test cases to read itself, so the entry's keys are just its
        # constructor arguments. single_error_type is the one that matters for a test: without it
        # the loader samples a different error case per run on every call and the row count is not
        # reproducible. It also forces dup_errors=True and errors_per_run=1 inside the loader.
        # The archive unpacks one level deep, and the loader wants the folder that directly
        # contains the run directories, not the one containing that.
        loader = LO2Loader(filename=os.path.join(default_path, data.get('folder', '')),
                           n_runs=data.get('n_runs', 53),
                           errors_per_run=data.get('errors_per_run', 1),
                           dup_errors=data.get('dup_errors', True),
                           single_error_type=data.get('single_error_type'),
                           single_service=data.get('single_service', ''))
    elif dataset_name == "lo2v2":
        # Reads the same unpacked archive as the lo2 entry, so the path is lo2's, not the entry name's.
        loader = LO2v2Loader(filename=os.path.join(full_data_path, "lo2", data.get('folder', '')),
                             test_cases=data.get('test_cases'),
                             services=data.get('services'),
                             continuation_lines=data.get('continuation_lines', 'fill-lastseen'))
    #TODO from here on we no longer test dataset_name but something else. 
    #Code created by idiot Claude and cannot even fix it. 
    #Needs to be alligend later with the dataset_name above.
    elif data.get('loader') == 'access_log':
        loader = AccessLogLoader(filename=default_path, format=data['format'],
                                filename_pattern=data.get('filename_pattern'))
    elif data.get('loader') == 'logfmt':
        loader = LogfmtLoader(filename=default_path,
                              filename_pattern=data.get('filename_pattern'))
    elif data.get('loader') == 'syslog':
        loader = SyslogLoader(filename=default_path,
                              filename_pattern=data.get('filename_pattern'))
    elif data.get('loader') == 'delimited':
        loader = DelimitedLoader(filename=default_path, format=data['format'],
                                 filename_pattern=data.get('filename_pattern'))
    # Before the bare 'format' branch below, which would otherwise swallow this: an auto entry may
    # legitimately carry a format key describing what it is expected to detect.
    elif data.get('loader') == 'auto':
        loader = AutoLoader(filename=default_path,
                            filename_pattern=data.get('filename_pattern'),
                            system=system or None)

    elif 'format' in data:
        loader = JsonLoader(filename=default_path, container=data['format'],
                            filename_pattern=data.get('filename_pattern'), flatten=True)
    else:
        print(f"ERROR did not find dataset: {dataset_name}")
        
    return loader

def check_baseline(key, frame):
    """Check that loading still gives the rows recorded earlier, or record them with --baseline capture."""
    if args.baseline == "off":
        return
    actual = fp.fingerprint(frame)
    if args.baseline == "capture":
        baselines[key] = actual
        print(f"Baseline captured for {key}: {actual['height']} rows, {len(actual['schema'])} columns")
        return
    expected = baselines.get(key)
    if expected is None:
        print(f"No baseline for {key} in {baseline_file}; run with --baseline capture to record one.")
        return
    problems = fp.compare(expected, actual)
    if problems:
        print(f"MISMATCH! {key} differs from its baseline: " + "; ".join(problems))
        failures.append(f"{key}: baseline")
    else:
        note = "" if fp.hashes_comparable(expected, actual) else \
            f" (content hashes not compared: baseline from polars {expected.get('polars')})"
        print(f"Baseline OK for {key}{note}")


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
        failures.append(f"{dataset}: row count")

    check_baseline(dataset, loader.df)
    if loader.df_seq is not None:
        # Sequence frames come out of group_by/unique, whose row order varies from run to run.
        df_seq = loader.df_seq.sort("seq_id") if "seq_id" in loader.df_seq.columns else loader.df_seq
        check_baseline(f"{dataset}_seq", df_seq)

    # Reduce data if reduction_fraction is specified
    if reduction_fraction:
        loader.reduce_dataframes(frac=reduction_fraction)

    # Save the data used for anomaly_detectors tests.
    loader.df.write_parquet(f"{test_data_path}/{dataset}_lo.parquet") 
    if any(sub in dataset for sub in ["hdfs", "profilence", "hadoop", "adfa", "awsctd", "lo2"]):
        loader.df_seq.write_parquet(f"{test_data_path}/{dataset}_lo_seq.parquet")  

def stream_and_save(dataset_name, dataset, loader):
    """Load a dataset too big for memory: sink it to parquet, check it, keep a sample for later stages.

    Memory stays bounded by a batch, so this handles datasets that do not fit in RAM. The row
    count and baseline checks run as streaming aggregations over the written file; the sample
    that enhancers.py and anomaly_detectors.py get is drawn with loglead.streaming.sample.
    """
    from loglead.streaming import count_rows, sample

    test_data_path = os.path.join(full_data_path, "test_data")
    os.makedirs(test_data_path, exist_ok=True)
    full = os.path.join(test_data_path, f"{dataset_name}_full.parquet")
    seq_full = os.path.join(test_data_path, f"{dataset_name}_full_seq.parquet")
    loader.sink(full, seq_path=seq_full)
    rows = count_rows(full)
    expected_length = dataset.get('expected_length')
    if expected_length and rows != expected_length:
        print(f"MISMATCH! {dataset_name} expected {expected_length} was {rows}. Perhaps old version of data?")
        failures.append(f"{dataset_name}: row count")
    check_baseline(dataset_name, full)
    if loader.df_seq is not None:
        check_baseline(f"{dataset_name}_seq", loader.df_seq.sort("seq_id"))
    fraction = dataset.get('reduction_fraction') or 1.0
    reduced = sample(full, fraction=fraction, seed=42)
    reduced.write_parquet(f"{test_data_path}/{dataset_name}_lo.parquet")
    print(f"Streamed {rows} rows of {dataset_name} to {full}; kept a {fraction} sample of "
          f"{reduced.height} rows as {dataset_name}_lo.parquet")
    if not args.keep_full:
        os.remove(full)
        if os.path.exists(seq_full):
            os.remove(seq_full)


def use_streaming(dataset_name, dataset, loader):
    """Stream when the config asks for it, or when an eager load would not fit in memory."""
    if not getattr(loader, "supports_streaming", False):
        return False
    if dataset.get('stream'):
        return True
    return dataset_name in ("thunderbird", "spirit", "liberty") and memory <= memory_limit_TB


def load_one(dataset_name, dataset, system=""):
    label = f"{dataset_name} ({system})" if system else dataset_name
    loader = create_correct_loader(dataset_name, dataset, system)
    if loader is None:
        return
    start_time = time.time()
    try:
        if use_streaming(dataset_name, dataset, loader):
            print(f"Streaming {label} (stream: {bool(dataset.get('stream'))}, memory available {memory:.1f}GB)")
            stream_and_save(dataset_name, dataset, loader)
            print(f"Loading {label} took {time.time() - start_time:.2f}s")
            return
        loader.execute()
        print(f"Loading {label} took {time.time() - start_time:.2f}s")
        check_and_save(dataset_name, loader, config, system)
    except Exception:
        print(f"FAIL loading {label}:")
        traceback.print_exc(file=sys.stdout)
        failures.append(f"{label}: raised")


# Loop through the datasets in the configuration file
for dataset in config['datasets']:
    dataset_name = dataset['name']
    memory = psutil.virtual_memory().available / (1024 ** 3)
    if args.only and dataset_name not in args.only:
        continue

    skip_loader = not dataset.get('load', True)
    if skip_loader:
        print(f'Skipping loader for {dataset_name}.')
        continue

    print(f"Loading: {dataset_name}")
    if dataset_name == "nezha":
        for system in dataset['systems']:
            print(f"System: {system}")
            load_one(dataset_name, dataset, system)
    else:
        load_one(dataset_name, dataset)

if args.baseline == "capture":
    fp.save_baselines(baseline_file, baselines)
    print(f"Baselines written to {baseline_file}")
if failures:
    print(f"Loading test complete. {len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("Loading test complete.")
