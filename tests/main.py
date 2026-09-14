import argparse
import runpy
import os
import shutil
import sys

import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(
    description='Run the full LogLead test suite: download data, then load, enhance and '
                'anomaly-detect it.')
parser.add_argument('--config', type=str, default=os.path.join(script_dir, 'datasets_mid_labels.yml'),
                     help='Path to the YAML file containing dataset information. '
                          'Default is tests/datasets_mid_labels.yml.')
parser.add_argument('--clean-test-data', action='store_true',
                     help='Delete the test_data folder (intermediate parquet files produced by '
                          'loaders.py/enhancers.py/anomaly_detectors.py) after the suite finishes. '
                          'Does not touch the downloaded raw datasets, which are kept for reuse.')
args = parser.parse_args()

config_path = os.path.abspath(args.config)

os.chdir(script_dir)
sys.path.insert(0, os.path.join(script_dir, '..', 'downloader'))

from download_data import main as download_data_main

download_data_main(None, config_path)
print ("___________________________________________________")

for stage in ('loaders.py', 'enhancers.py', 'anomaly_detectors.py'):
    sys.argv = [stage, '--config', config_path]
    runpy.run_path(stage)
    print ("___________________________________________________")

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
test_data_path = os.path.join(os.path.expanduser(config['root_folder']), "test_data")

if args.clean_test_data:
    shutil.rmtree(test_data_path, ignore_errors=True)
    print(f"All tests executed. Deleted test_data folder: {test_data_path}")
else:
    print(f"All tests executed. Test data folder: {test_data_path}. "
          f"Pass --clean-test-data to delete it (downloaded raw datasets are kept).")