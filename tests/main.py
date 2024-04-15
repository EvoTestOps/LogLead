import runpy
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


runpy.run_path('loaders.py')
print ("___________________________________________________")
runpy.run_path('enhancers.py')
print ("___________________________________________________")
runpy.run_path('anomaly_detectors.py')
print ("___________________________________________________")
print("All tests executed. Consider deleting contents of test_data folder if you do not need it.")