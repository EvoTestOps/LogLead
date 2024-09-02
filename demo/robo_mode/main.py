import os
import yaml
from dotenv import load_dotenv, find_dotenv
from loglead.enhancers import EventLogEnhancer 
from log_analysis_functions import (read_folders, similarity_run_file, 
                                    similarity_run_content, similarity_file_content,
                                    similarity_line_content, anomaly_file_content, anomaly_line_content,
                                    anomaly_run, masking_patterns_myllari)

load_dotenv(find_dotenv())

import warnings
# Suppress specific warnings based on the message content
warnings.filterwarnings(
    "ignore", 
    "WARNING! data has no labels. Only unsupervised methods will work.", 
    UserWarning
)

full_data = os.getenv("LOG_DATA_PATH")
if not full_data:
    print("WARNING!: LOG_DATA_PATH is not set. This will most likely fail")

# Ensure this always gets executed in the same location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
output_folder = "output"

with open(os.path.join(script_dir, 'config.yml'), 'r') as file:
    config = yaml.safe_load(file)

df, _ = read_folders(os.path.join(full_data, "comp_ws", "all_data_10_percent"))
#Normalize. Takes 60s. 
enhancer = EventLogEnhancer(df)
print(f"Normalizing data")
df = enhancer.normalize(regexs=masking_patterns_myllari)
#Could also use Drain that is default. 
#df = enhancer.normalize()

# Extract configurations for each step
steps = config.get('steps', {})


# Processing run_file_similarity step
configs = steps.get('similarity_run_file', [])
for config in configs:
    similarity_run_file(
        df,
        config.get('target_run'),
        config.get('comparison_runs', 'ALL')
    )

# Processing run_content_similarity step
configs = steps.get('similarity_run_content', [])
for config in configs:
    similarity_run_content(
        df,
        config.get('target_run'),
        config.get('comparison_runs', 'ALL'),
        config.get('normalize_content', False)
    )

# Processing file_content_similarity step
configs = steps.get('similarity_file_content', [])
for config in configs:
    similarity_file_content(
        df,
        config.get('target_run'),
        config.get('comparison_runs', 'ALL'),
        config.get('normalize_content', False)
    )
# Processing file_content_similarity step
configs = steps.get('similarity_line_content', [])
for config in configs:
    similarity_line_content(
        df,
        config.get('target_run'),
        config.get('comparison_runs', 'ALL'),
        config.get('target_files', 'ALL'),
        config.get('normalize_content', False)
    )

configs = steps.get('anomaly_run_file', [])
for config in configs:
    anomaly_run(
        df,
        config.get('target_run'),
        config.get('comparison_runs', 'ALL'),
        file = True
    )

configs = steps.get('anomaly_run_content', [])
for config in configs:
    anomaly_run(
        df,
        config.get('target_run'),
        config.get('comparison_runs', 'ALL'),
        file = False,
        normalize=config.get('normalize_content', False)
    )


# Processing line_anomaly step
configs = steps.get('anomaly_file_content', [])
for config in configs:
    anomaly_file_content(
        df,
        config.get('target_run'),
        config.get('comparison_runs', 'ALL'),
        config.get('target_files', 'ALL'),
        config.get('normalize_content', False)
    )

# Processing file_anomaly step
configs = steps.get('anomaly_line_content', [])
for config in configs:
    anomaly_line_content(
        df,
        config.get('target_run'),
        config.get('comparison_runs', 'ALL'),
        config.get('target_files', 'ALL'),
        config.get('normalize_content', False)
    )

