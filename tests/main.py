import runpy
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, os.path.join(script_dir, '..', 'downloader'))

from download_data import main as download_data_main

download_data_main(None, 'datasets.yml')
print ("___________________________________________________")
runpy.run_path('loaders.py')
print ("___________________________________________________")
runpy.run_path('enhancers.py')
print ("___________________________________________________")
runpy.run_path('anomaly_detectors.py')
print ("___________________________________________________")
print("All tests executed. Consider deleting contents of test_data folder if you do not need it.")