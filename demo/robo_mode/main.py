import os
import yaml
import warnings
from dotenv import load_dotenv, find_dotenv
from loglead.enhancers import EventLogEnhancer
from log_analysis_functions import (
    set_output_folder, read_folders, distance_run_file, distance_run_content,
    distance_file_content, distance_line_content,
    plot_run, plot_file_content,
    anomaly_file_content, anomaly_line_content,
    anomaly_run, masking_patterns_myllari, masking_patterns_myllari2
)
from data_specific_preprocessing import preprocess_files
import inspect

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
    # Different options for normalization. Should be in a config
    # df = enhancer.normalize(regexs=masking_patterns_myllari)
    # df = enhancer.normalize()
    df = enhancer.normalize(regexs=masking_patterns_myllari2)
    print("Parsing event templates")
    df = enhancer.parse_tip()

    # Data-specific preprocessing
    df = preprocess_files(df, config.get('preprocessing_steps', []))

    steps = config.get('steps', {})

    # Map step types that need to be handled differently
    special_cases = {
        'plot_run_file': {'func_name': 'plot_run', 'fixed_args': {'file': True, 'content_format':'File'}},
        'plot_run_content': {'func_name': 'plot_run', 'fixed_args': {'file': False}},
        'anomaly_run_file': {'func_name': 'anomaly_run', 'fixed_args': {'file': True}},
        'anomaly_run_content': {'func_name': 'anomaly_run', 'fixed_args': {'file': False}},
    }

    # Import the module where functions are defined
    import log_analysis_functions

    for step_type, configs in steps.items():
        for config_item in configs:
            # Determine function to call
            if step_type in special_cases:
                func_name = special_cases[step_type]['func_name']
                fixed_args = special_cases[step_type]['fixed_args']
            else:
                func_name = step_type
                fixed_args = {}

            # Get the function from the module
            func = getattr(log_analysis_functions, func_name, None)

            if func is None:
                print(f"Function {func_name} not found")
                continue

            # Get function parameters
            func_params = inspect.signature(func).parameters

            # Build kwargs
            kwargs = {k: v for k, v in config_item.items() if k in func_params}

            # Add fixed args
            kwargs.update(fixed_args)

            # Add df
            kwargs['df'] = df

            # Call the function
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
        config_path = "config.yml"
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
