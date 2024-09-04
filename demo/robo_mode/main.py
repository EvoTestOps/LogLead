import os
import yaml
import warnings
from dotenv import load_dotenv, find_dotenv
from loglead.enhancers import EventLogEnhancer
from log_analysis_functions import (
    set_output_folder, read_folders, similarity_run_file, similarity_run_content,
    similarity_file_content, similarity_line_content, anomaly_file_content, anomaly_line_content,
    anomaly_run, masking_patterns_myllari
)
from data_specific_preprocessing import preprocess_files

def load_config(config_path):
    """Load configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(config_path):
    # Load configuration
    config = load_config(config_path)
    print(f"Starting loaded config: {config_path}")

    # Set output folder
    output_folder = config.get('output_folder')
    set_output_folder(output_folder)

    # Set input data folder
    input_data_folder = config.get('input_data_folder')
    if not input_data_folder:
        input_data_folder = os.getenv("LOG_DATA_PATH")
        if not input_data_folder:
            print("WARNING!: LOG_DATA_PATH is not set. This will most likely fail")
        input_data_folder = os.path.join(input_data_folder, "comp_ws", "all_data_10_percent")

    # Read data
    df, _ = read_folders(input_data_folder)

    # Normalize data
    enhancer = EventLogEnhancer(df)
    print("Normalizing data")
    df = enhancer.normalize(regexs=masking_patterns_myllari)
    df = enhancer.words()

    # Data-specific preprocessing
    df = preprocess_files(df, config.get('preprocessing_steps', []))

    steps = config.get('steps', {})

    # Define a mapping of step types to their respective functions and additional parameters
    step_functions = {
        'similarity_run_file': {
            'func': similarity_run_file,
            'params': ['target_run', 'comparison_runs']
        },
        'similarity_run_content': {
            'func': similarity_run_content,
            'params': ['target_run', 'comparison_runs', 'normalize_content']
        },
        'similarity_file_content': {
            'func': similarity_file_content,
            'params': ['target_run', 'comparison_runs', 'normalize_content']
        },
        'similarity_line_content': {
            'func': similarity_line_content,
            'params': ['target_run', 'comparison_runs', 'target_files', 'normalize_content']
        },
        'anomaly_run_file': {
            'func': anomaly_run,
            'params': ['target_run', 'comparison_runs', 'detectors'],
            'fixed_args': {'file': True}
        },
        'anomaly_run_content': {
            'func': anomaly_run,
            'params': ['target_run', 'comparison_runs', 'detectors',  'normalize_content'],
            'fixed_args': {'file': False}
        },
        'anomaly_file_content': {
            'func': anomaly_file_content,
            'params': ['target_run', 'comparison_runs', 'target_files','detectors', 'normalize_content']
        },
        'anomaly_line_content': {
            'func': anomaly_line_content,
            'params': ['target_run', 'comparison_runs', 'target_files','detectors', 'normalize_content']
        },
    }

    # Iterate over each step type in the config
    for step_type, config_data in step_functions.items():
        configs = steps.get(step_type, [])
        func = config_data['func']
        params = config_data['params']
        #fixed_args = config_data.get('fixed_args', {})

        for config in configs:
            # Default values. comparison_runs=ALL others False
            kwargs = {param: config.get(param, 'ALL' if 'comparison_runs' in param else False) for param in params}
            # If 'detectors' is missing, and if so, assign the default value ["KMeans"]
            if 'detectors' in params and 'detectors' not in config:
                kwargs['detectors'] = ["KMeans"]
            #kwargs.update(fixed_args)
            kwargs['df'] = df  # Common argument across all function calls

            # Call the function with the prepared keyword arguments
            func(**kwargs)
    print(f"Done! See output in folder: {output_folder}")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv(find_dotenv())

    # Set working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Suppress specific warnings
    warnings.filterwarnings("ignore", "WARNING! data has no labels. Only unsupervised methods will work.", UserWarning)

    # Determine if running in an IPython environment
    try:
        __IPYTHON__
        ipython_env = True
    except NameError:
        ipython_env = False

    if ipython_env:
        # Running in IPython/Jupyter
        config_path ="config.yml"
    else:
        # Parse command-line arguments
        import argparse
        parser = argparse.ArgumentParser(description="LogLead RoboMode")
        parser.add_argument(
            "-c", "--config",
            default="config.yml",
            help="Path to the configuration file (default: config.yml)"
        )
        args = parser.parse_args()
        config_path = args.config

    # Run main process
    main(config_path)
