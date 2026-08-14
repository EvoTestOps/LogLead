import argparse
import runpy
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(
    description='Run the full LogLead test suite: download data, then load, enhance and '
                'anomaly-detect it.')
parser.add_argument('--config', type=str, default=os.path.join(script_dir, 'datasets_mid_labels.yml'),
                     help='Path to the YAML file containing dataset information. '
                          'Default is tests/datasets_mid_labels.yml.')
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
print("All tests executed. Consider deleting contents of test_data folder if you do not need it.")