import os
import polars as pl
import inspect
import datetime
from loglead.loaders import RawLoader
from loglead import LogDistance, AnomalyDetector

# Ensure this always gets executed in the same location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
output_folder = None

def set_output_folder(folder_path):
    """
    Set the global output folder path for the module.
    
    Parameters:
        folder_path (str): The path to the output folder.
    """
    global output_folder  # Declare that we're modifying the global variable
    output_folder = folder_path
    print(f"Output folder set to: {output_folder}")

def read_folders(folder, filename_pattern= "*.log"):
    loader = RawLoader(folder, filename_pattern=filename_pattern, strip_full_data_path=folder)
    df = loader.execute()
    df = df.filter(pl.col("m_message").is_not_null())

    df = df.with_columns([
        # Extract the first part of the path and create the 'run' column
        pl.col("file_name").str.extract(r'^/([^/]+)', 1).alias("run"),
        # Remove the first part of the path to keep the rest in 'file_name'
        pl.col("file_name").str.replace(r'^/[^/]+/', '', literal=False).alias("file_name")
    ])
    unique_runs = len(df.select("run").unique().to_series().to_list())
    print (f"Loaded {unique_runs} runs (folders) with {df.height} rows from folder {folder}")
    return df, unique_runs

def prepare_runs(df, target_run, comparison_runs="ALL"):
    """
    Prepares and validates the base and comparison runs from the dataframe.

    Parameters:
    - df: Polars DataFrame containing the data with a 'run' column.
    - base_run_name: The name of the base run to compare against other runs.
    - comparison_runs: List of comparison run names, 'ALL' to compare against all other runs, or an integer specifying the number of comparison runs (default is 'ALL').

    Returns:
    - run1: DataFrame containing the data for the base run.
    - comparison_runs: List of comparison run names to be compared with the base run.

    Raises:
    - ValueError: If the base run name or any comparison run name is not found in the dataframe.
    """
    
    # Extract unique runs
    unique_runs = df.select("run").unique().sort("run").to_series().to_list()
    
    # Validate base run name
    if target_run not in unique_runs:
        raise ValueError(f"Base run name '{target_run}' not found in the dataframe. Please provide a valid run name.")
    
    # Get the data for the base run
    run1 = df.filter(pl.col("run") == target_run)
    
    # Determine comparison runs
    if comparison_runs == "ALL":
        # Use all other runs except the base run
        comparison_runs = [run for run in unique_runs if run != target_run]
    elif isinstance(comparison_runs, int):
        # Ensure the number is valid
        if comparison_runs < 1 or comparison_runs > len(unique_runs) - 1:
            raise ValueError(f"Number of comparison runs must be between 1 and {len(unique_runs) - 1}.")
        # Exclude the base run and select the specified number of runs
        comparison_runs = [run for run in unique_runs if run != target_run][:comparison_runs]
    else:
        # Assume comparison_runs is a list and validate all provided comparison runs
        if not all(run in unique_runs for run in comparison_runs):
            invalid_runs = [run for run in comparison_runs if run not in unique_runs]
            raise ValueError(f"Comparison run names {invalid_runs} not found in the dataframe. Please provide valid run names.")
    
    return run1, comparison_runs

def check_multiple_target_runs(df, base_runs):

    unique_runs = df.select("run").unique().sort("run").to_series().to_list()

    if base_runs == "ALL":
        base_runs = unique_runs 
    elif isinstance(base_runs, int):
        # Ensure the number is valid
        if base_runs < 1 or base_runs > len(unique_runs) - 1:
            raise ValueError(f"Number of base runs must be between 1 and {len(unique_runs) - 1}.")
        # Exclude the base run and select the specified number of runs
        base_runs = unique_runs[:base_runs]
    else:
        # Assume comparison_runs is a list and validate all provided comparison runs
        if not all(run in unique_runs for run in base_runs):
            invalid_runs = [run for run in base_runs if run not in unique_runs]
            raise ValueError(f"Comparison run names {invalid_runs} not found in the dataframe. Please provide valid run names.")
    return base_runs

def distance_run_file(df, target_run, comparison_runs="ALL"):
    """
    Measure distances between one run and specified other runs in the dataframe and save the results as a CSV file.
    
    The output CSV filename will include the name of the base run.

    Parameters:
    - df: Polars DataFrame containing the data with a 'run' column.
    - target_run: Name of the run to compare against others.
    - comparison_runs: Optional list of run names to compare against. If None, compares against all other runs.
    """
    # Extract unique runs 
    run1, comparison_run_names = prepare_runs(df, target_run, comparison_runs) 
    results = []
    print(
        f"Executing {inspect.currentframe().f_code.co_name} with target run '{target_run}' and {len(comparison_run_names)} comparison runs"
        + (f": {comparison_run_names}" if len(comparison_run_names) < 6 else "")
    )
    # Compare the base run to each specified comparison run
    for other_run in comparison_run_names:
        run2 = df.filter(pl.col("run") == other_run)
        
        # Extract unique file names from each run
        file_names_run1 = run1.select("file_name").unique()
        file_names_run2 = run2.select("file_name").unique()
        # Find file names that are only in run1
        only_in_run1_count = file_names_run1.filter(~pl.col("file_name").is_in(file_names_run2.get_column("file_name"))).height
        # Find file names that are only in run2
        only_in_run2_count = file_names_run2.filter(~pl.col("file_name").is_in(file_names_run1.get_column("file_name"))).height
        # Find the intersection of file names between run1 and run2
        intersection_count = file_names_run1.filter(pl.col("file_name").is_in(file_names_run2.get_column("file_name"))).height
        # Find the union of file names between run1 and run2
        union_count = pl.concat([file_names_run1, file_names_run2]).unique().height
        jaccard_dist = 1 - (intersection_count / union_count)
        overlap_dist = 1 - (intersection_count / min(file_names_run1.height, file_names_run1.height))

        # Append results to the list
        results.append({
            "target_run": target_run,
            "comparison_run": other_run,
            "files only in target": only_in_run1_count,
            "files only in comparison": only_in_run2_count,
            "union": union_count,
            "intersection": intersection_count,
            "jaccard distance": jaccard_dist,
            "overlap distance": overlap_dist
        })
        # Print a dot to indicate progress
        print(".", end="", flush=True)

    print()  # Newline after progress dots
    # Create a Polars DataFrame from the results
    results_df = pl.DataFrame(results)
    write_dataframe_to_csv(results_df, analysis="dis", level=1, target_run=target_run, comparison_run="Many")

def distance_run_content(df, target_run, comparison_runs="ALL", normalize_content=False):
    """
    Measure distances between one run and specified other runs in the dataframe and save the results as a CSV file.
    
    The output CSV filename will include the name of the base run.

    Parameters:
    - df: Polars DataFrame containing the data with a 'run' column.
    - base_run_name: Name of the run to compare against others.
    - comparison_runs: Optional list of run names to compare against. If None, compares against all other runs.
    """
    field = "e_message_normalized" if normalize_content else "m_message"
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # Extract unique runs 
    run1, comparison_run_names = prepare_runs(df, target_run, comparison_runs) 
    results = []
    print(
        f"Executing {inspect.currentframe().f_code.co_name} with target run '{target_run}' and {len(comparison_run_names)} comparison runs"
        + (f": {comparison_run_names}" if len(comparison_run_names) < 6 else "")
    )
    # Compare the base run to each specified comparison run
    for other_run in comparison_run_names:
        run2 = df.filter(pl.col("run") == other_run)
        
        # Initialize LogSimilarity class for each pair of runs
        distance = LogDistance(run1, run2, field=field)

        # Measure distances between the base run and the current run
        cosine = distance.cosine()
        jaccard = distance.jaccard()
        compression = distance.compression()
        containment = distance.containment()

        # Append results to the list
        results.append({
            "target_run": target_run,
            "comparison_run": other_run,
            "cosine": cosine,
            "jaccard": jaccard,
            "compression": compression,
            "containment": containment
        })
        # Print a dot to indicate progress
        print(".", end="", flush=True)

    print()  # Newline after progress dots
    results_df = pl.DataFrame(results)
    write_dataframe_to_csv(results_df, analysis="dis", level=2, target_run=target_run, comparison_run="Many", norm=normalize_content)

def distance_file_content(df, target_run, comparison_runs="ALL", normalize_content=False):
    """
    Measure distances between one run and specified other runs in the dataframe and save the results as a CSV file.
    
    The output CSV filename will include the name of the base run.

    Parameters:
    - df: Polars DataFrame containing the data with a 'run' column.
    - base_run_name: Name of the run to compare against others.
    - comparison_runs: Optional list of run names to compare against. If None, compares against all other runs.
    """
    # Extract unique runs
    field = "e_message_normalized" if normalize_content else "m_message"

    run1, comparison_run_names = prepare_runs(df, target_run, comparison_runs) 
    results = []

    print(
        f"Executing {inspect.currentframe().f_code.co_name} with target run '{target_run}' and {len(comparison_run_names)} comparison runs"
        + (f": {comparison_run_names}" if len(comparison_run_names) < 6 else "")
    )
    # Compare the base run to each specified comparison run
    for other_run in comparison_run_names:
        run2 = df.filter(pl.col("run") == other_run)
        file_names_run1 = run1.select("file_name").unique()
        file_names_run2 = run2.select("file_name").unique()
        matching_file_names = file_names_run1.filter(pl.col("file_name").is_in(file_names_run2.get_column("file_name")))
        matching_file_names_list = matching_file_names.get_column("file_name").to_list()

        print(f"Comparing against {other_run} with {len(matching_file_names_list)} matching files")

        for file_name in matching_file_names_list:
            run1_file = run1.filter(pl.col("file_name") == file_name)
            run2_file = run2.filter(pl.col("file_name") == file_name)

            # Calculate the distances
            # Initialize LogSimilarity class for each pair of runs
            distance = LogDistance(run1_file, run2_file, field=field)
            # Measure distances between the base run and the current run
            cosine = distance.cosine()
            jaccard = distance.jaccard()
            compression = distance.compression()
            containment = distance.containment()
            #Too slow
            #same, changed, deleted, added = similarity.diff_lines() 
            
            # Create a dictionary to store results
            result = {
                'file_name': file_name,
                'target_run': target_run,
                'comparison_run': other_run,
                'cosine': cosine,
                'jaccard': jaccard,
                'compression': compression,
                'containment': containment,
                'target_size': distance.size1,
                'comparison_lines': distance.size2, 
            }
            results.append(result)
            # Print a dot to indicate progress
            print(".", end="", flush=True)
        print()  # Newline after progress dots

    # Create a Polars DataFrame from the results
    results_df = pl.DataFrame(results)
    write_dataframe_to_csv(results_df, analysis="dis", level=3, target_run=target_run, comparison_run="Many", norm=normalize_content)

def distance_line_content(df, target_run, comparison_runs="ALL", target_files="ALL", normalize_content=False):
    """
    Measure distances between one run and specified other runs in the dataframe and save the results as a CSV file.
    
    The output CSV filename will include the name of the base run.

    Parameters:
    - df: Polars DataFrame containing the data with a 'run' column.
    - base_run_name: Name of the run to compare against others.
    - comparison_runs: Optional list of run names to compare against. If None, compares against all other runs.
    """    
    field = "e_message_normalized" if normalize_content else "m_message"
    # Extract unique runs and files
    df_run1, comparison_run_names = prepare_runs(df, target_run, comparison_runs) 
    target_files = prepare_files(df_run1, target_files)
    df_other_runs = df.filter(pl.col("run").is_in(comparison_run_names))
    print(
        f"Executing {inspect.currentframe().f_code.co_name} with target run '{target_run}' and {len(comparison_run_names)} comparison runs"
        + (f": {comparison_run_names}" if len(comparison_run_names) < 6 else "")
    )
    # Compare the base run to each specified comparison run
    for other_run in comparison_run_names:
        for file_name in target_files:
            df_run1_file =  df_run1.filter(pl.col("file_name") == file_name)
            sanitized_file_name = file_name.replace('/', '_').replace('\\', '_')
            df_other_run_file = df_other_runs.filter(pl.col("run") == other_run) #Filter one run
            df_other_run_file = df_other_run_file.filter(pl.col("file_name") == file_name) #Filter one file
            distance = LogDistance(df_run1_file, df_other_run_file, field=field)
            diff = distance.diff_lines()
            
            write_dataframe_to_csv(diff, analysis="dis", level=4, target_run=target_run, comparison_run=other_run, norm=normalize_content, file=sanitized_file_name)
            print(".", end="", flush=True) #Progress on screen
    print()  # Newline after progress dots


def prepare_files(df_run1, files="ALL"):
    """
    Prepares and validates the files from the base run data based on the provided configuration.

    Parameters:
    - df_run1: Polars DataFrame containing the data for the base run with a 'file_name' column.
    - files: List of file names, 'ALL' to use all files in the base run, an integer specifying the number of files to use, 
             or a list of file names to be processed (default is 'ALL').

    Returns:
    - files: List of file names to be used in the analysis.

    Raises:
    - ValueError: If no valid files are found or if the input list has no matching files in the base run.
    """
    
    # Extract available files from the base run
    available_files = df_run1.select("file_name").unique().to_series().to_list()

    if isinstance(files, list):
        # Check if each specified file exists in the base run data
        missing_files = [file for file in files if file not in available_files]
        if missing_files:
            print(f"Warning: The following files do not exist in the base run: {missing_files}")
            # Remove missing files from the list to avoid processing them
            files = [file for file in files if file in available_files]
        
        if not files:
            raise ValueError("No valid files found in the provided list for processing.")
    
    elif files == "ALL":
        # Use all unique files from the base run
        files = available_files
    
    elif isinstance(files, int):
        # Validate that the number is within the range of available files
        if files < 1 or files > len(available_files):
            raise ValueError(f"Number of files must be between 1 and {len(available_files)}.")
        # Select the specified number of files
        files = available_files[:files]
    
    else:
        raise ValueError("Invalid type for 'files'. It must be 'ALL', a list, or an integer.")
    
    return files

def anomaly_line_content(df, target_run, comparison_runs="ALL", target_files="ALL", detectors=["KMeans"], normalize_content=False):
    """
    Measure distances between one run and specified other runs in the dataframe and save the results as a CSV file.
    
    The output CSV filename will include the name of the base run.

    Parameters:
    - df: Polars DataFrame containing the data with a 'run' column.
    - base_run_name: Name of the run to compare against others.
    - comparison_runs: Optional list of run names to compare against. If ALL, compares against all other runs.
    """
    # Extract unique runs
    field = "e_message_normalized" if normalize_content else "m_message"
 
    df_run1, comparison_run_names = prepare_runs(df, target_run, comparison_runs) 
    print(
        f"Executing {inspect.currentframe().f_code.co_name} with target run '{target_run}' and {len(comparison_run_names)} comparison runs"
        + (f": {comparison_run_names}" if len(comparison_run_names) < 6 else "")
    )
    target_files = prepare_files(df_run1, target_files)
    df_other_runs = df.filter(pl.col("run").is_in(comparison_run_names))
    print(f"Predicting {len(target_files)} files: {target_files}")
    # Loop over each file first
    for file_name in target_files:
        sanitized_file_name = file_name.replace('/', '_').replace('\\', '_')
        df_run1_files =  df_run1.filter(pl.col("file_name") == file_name)
        df_other_runs_files = df_other_runs.filter(pl.col("file_name") == file_name)
        if df_other_runs_files.height == 0:
            print(f"Found no files matching files in comparisons runs for file: {file_name}")
            continue

        df_anos = run_anomaly_detection(df_run1_files,df_other_runs_files, field, detectors=detectors, drop_input=False)

        df_anos = df_anos.with_row_index("line_number")
        #write_dataframe_to_csv(df_anos, output_csv)
        write_dataframe_to_csv(df_anos, analysis="ano", level=4, target_run=target_run, comparison_run="Many", norm=normalize_content, file=sanitized_file_name)
        print(".", end="", flush=True) #Progress on screen
    print()  # Newline after progress dots

def anomaly_file_content(df, target_run, comparison_runs="ALL", target_files="ALL", detectors=["KMeans"], normalize_content=False):
    """
    Measure distances between one run and specified other runs in the dataframe and save the results as a CSV file.
    
    The output CSV filename will include the name of the base run.

    Parameters:
    - df: Polars DataFrame containing the data with a 'run' column.
    - base_run_name: Name of the run to compare against others.
    - comparison_runs: Optional list of run names to compare against. If ALL, compares against all other runs.
    """
    field = "e_message_normalized" if normalize_content else "m_message"

    # Extract unique runs 
    df_run1, comparison_run_names = prepare_runs(df, target_run, comparison_runs) 
    # Generate output CSV file name based on base run
    print(
        f"Executing {inspect.currentframe().f_code.co_name} with target run '{target_run}' and {len(comparison_run_names)} comparison runs"
        + (f": {comparison_run_names}" if len(comparison_run_names) < 6 else "")
    )
    target_files = prepare_files(df_run1, target_files)
    print(f"Predicting {len(target_files)} files: {target_files}")
    df_other_runs = df.filter(pl.col("run").is_in(comparison_run_names))
    df_anos_merge = pl.DataFrame()

    for file_name in target_files:
        
        df_run1_files =  df_run1.filter(pl.col("file_name") == file_name)
        df_run1_files = df_run1_files.group_by('file_name').agg(pl.col(field).alias(field))

        df_other_runs_files = df_other_runs.filter(pl.col("file_name") == file_name)
        df_other_runs_files = df_other_runs.group_by('file_name').agg(pl.col(field).alias(field))

        if df_other_runs_files.height == 0:
            print(f"Found no files matching files in comparisons runs for file: {file_name}")
            continue

        df_anos = run_anomaly_detection(df_run1_files,df_other_runs_files,detectors=detectors, field= field)

        df_anos = df_anos.with_columns(pl.lit(file_name).alias("file_name"))
        df_anos_merge = df_anos_merge.vstack(df_anos)
        print(".", end="", flush=True) #Progress on screen
    print()  # Newline after progress dots
    write_dataframe_to_csv(df_anos_merge, analysis="ano", level=3, target_run=target_run, comparison_run="Many", norm=normalize_content)

def anomaly_run(df, target_run, comparison_runs="ALL", file = False, detectors=["KMeans"], normalize_content=False):
    """
    Detect anomalies at the run level.
    
    Parameters:
    - df: Polars DataFrame containing the data with a 'run' column.
    - base_run_name: Name of the run to analyze.
    - comparison_runs: Optional list of run names to compare against. If ALL, compares against all other runs.
    """
    if file: 
        field = "file_name"
    elif normalize_content: 
        field = "e_message_normalized"
    else:
        field = "m_message"
    df_anos_merge = pl.DataFrame()
    target_run_names = check_multiple_target_runs(df, target_run)
    
    print(f"Executing {inspect.currentframe().f_code.co_name} with {'file' if file else 'content'} anomalies of {len(target_run_names)} target runs with {comparison_runs} comparison runs")
    print(f"Target runs: {target_run_names}")
    print(f"Comparison runs: {comparison_runs}")
    for target_run_name in target_run_names:
        df_run1, comparison_run_names = prepare_runs(df, target_run_name, comparison_runs)
        
        df_run1 = df_run1.group_by("run").agg(pl.col(field).alias(field))
        df_other_runs = df.filter(pl.col("run").is_in(comparison_run_names)).group_by("run").agg(pl.col(field).alias(field))
        df_anos = run_anomaly_detection(df_run1, df_other_runs, detectors=detectors, field= field)
        comparison_runs_out = " ".join(comparison_run_names)
        df_anos = df_anos.with_columns(pl.lit(comparison_runs_out).alias("comparison_runs"))
        df_anos_merge = df_anos_merge.vstack(df_anos)
        print(".", end="", flush=True)
    print()  # Newline after progress dots
    write_dataframe_to_csv(df_anos_merge, analysis="ano", level=1 if file else 2, target_run="Many", comparison_run="Many", norm=normalize_content)


def run_anomaly_detection(df_run1_files,df_other_runs_files, field, detectors=["KMeans", "RarityModel"], drop_input=True):
    """
    Run anomaly detection using specified models.
    
    Parameters:
    - df_other_runs_files: DataFrame for training data.
    - df_run1_files: DataFrame for testing data.
    - field: Column name used by the AnomalyDetector as the item list column.
    - detectors: List of detector names to run (e.g., ["KMeans", "IsolationForest", "RarityModel"]).
                 If None, all detectors are run.
                 
    Returns:
    - DataFrame containing the predictions from the specified anomaly detectors.
    """
    # Initialize the AnomalyDetector
    sad = AnomalyDetector(item_list_col=field, print_scores=False, auc_roc=True)
    
    # Set the training and testing data
    sad.train_df = df_other_runs_files
    sad.test_df = df_run1_files
    
    # Prepare the data
    sad.prepare_train_test_data()
    
    # Initialize the output DataFrame
    df_anos = None
    
    # Run specified detectors or all if none are specified
    if detectors is None or "KMeans" in detectors:
        sad.train_KMeans()
        df_anos = sad.predict()
        df_anos = df_anos.rename({"pred_ano_proba": "kmeans_pred_ano_proba"})
    
    if detectors is None or "IsolationForest" in detectors:
        sad.train_IsolationForest()
        predictions = sad.predict().select("pred_ano_proba").rename({"pred_ano_proba": "IF_pred_ano_proba"})
        if df_anos is not None:
            df_anos = df_anos.with_columns(predictions)
        else:
            df_anos = predictions
    
    if detectors is None or "RarityModel" in detectors:
        sad.train_RarityModel()
        predictions = sad.predict().select("pred_ano_proba").rename({"pred_ano_proba": "RM_pred_ano_proba"})
        if df_anos is not None:
            df_anos = df_anos.with_columns(predictions)
        else:
            df_anos = predictions

    #Not working at the moment
    if detectors is None or "OOVDetector" in detectors:    
        #sad.X_train=None
        #sad.labels_train = None
        #sad.train_OOVDetector(filter_anos=False) #This just creates the object. No training for OOVD needed
        sad.train_OOVDetector() 
        predictions = sad.predict().select("pred_ano_proba").rename({"pred_ano_proba": "OOVD_pred_ano_proba"})
        if df_anos is not None:
            df_anos = df_anos.with_columns(predictions)
        else:
            df_anos = predictions

    #CSV does not support nested columns. Data needs to be dropped
    if drop_input: 
        if "m_message" in df_anos.columns:
            df_anos = df_anos.drop("m_message")
        if "e_message_normalized" in df_anos.columns:
            df_anos = df_anos.drop("e_message_normalized")
        if "file_name" in df_anos.columns:
            df_anos = df_anos.drop("file_name")
    return df_anos



def write_dataframe_to_csv(df, analysis, level=0, target_run="", comparison_run="", file="", norm=False, separator='\t', quote_style='always'):
    """
    Construct the file name and write a Polars DataFrame to a CSV file, creating directories if they don't exist.

    Parameters:
    - df: The Polars DataFrame to write.
    - analysis: A string indicating the type of analysis ('dis' for distance, 'ano' for another type).
    - level: An integer representing the level (default is 0).
    - target_run: A string representing the target run.
    - comparison_run: A string representing the comparison run.
    - file: Additional file information or identifier to include in the file name.
    - separator: The separator to use in the CSV file (default is '\t' for tab-separated).
    - quote_style: The quote style for writing the CSV file (default is 'always').
    """

    # Specific substring to remove from the end of run names
    suffix_to_remove = ""
    
    # Remove specific suffix from run names if present
    if comparison_run.endswith(suffix_to_remove):
        comparison_run = comparison_run[:-len(suffix_to_remove)]
    if target_run.endswith(suffix_to_remove):
        target_run = target_run[:-len(suffix_to_remove)]
    
    # Start constructing the output file name with the analysis type
    output_csv = analysis
    
    # Append the level to the file name
    output_csv += f"_L{level}"

    output_csv += f"_norm_{norm}"

    # Append the target and comparison run identifiers if they are provided
    if target_run:
        output_csv += f"_{target_run}"
    if comparison_run:
        output_csv += f"_vs_{comparison_run}"
    
    # Append the additional file identifier if provided
    if file:
        output_csv += f"_{file}"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_csv += f"_{timestamp}"
    # Finalize the file name with the CSV extension
    output_csv += ".xlsx"
    
    # Combine script_dir and output_folder to get the full directory path
    global output_folder
    output_directory = os.path.join(script_dir, output_folder)
    
    # Ensure the directory exists; if not, create it
    os.makedirs(output_directory, exist_ok=True)
    
    # Construct the full path for the CSV file
    output_path = os.path.join(output_directory, output_csv)
    df.write_excel(output_path)
    #print(f"Results saved to {output_path}")
    
masking_patterns_myllari = [
    ("${start}<QUOTED_ALPHANUMERIC>${end}", r"(?P<start>[^A-Za-z0-9-_]|^)'[a-zA-Z0-9-_]{16,}'(?P<end>[^A-Za-z0-9-_]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9/]|^)\d{2}/\d{2}/\d{4}(?P<end>[^0-9/]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9-]|^)\d{4}-\d{2}-\d{2}(?P<end>[^0-9-]|$)"),
    ("${start}<DATE_XX>${end}", r"(?P<start>[^A-Za-z0-9_]|^)DATE_\d{2}(?P<end>[^A-Za-z0-9_]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^A-Za-z]|^)\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), \d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b(?P<end>[^A-Za-z]|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:.]|^)\d{2}:\d{2}(?::\d{2}(?:\.\d{3})?)?(?P<end>[^0-9:.]|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:]|^)\d{2}:\d{2}(?P<end>[^0-9:]|$)"),
    ("${start}<DATETIME>${end}", r"(?P<start>[^0-9.]|^)\d{1,2}\.\d{1,2}\.\d{4} \d{1,2}\.\d{1,2}\.\d{2}(?P<end>[^0-9.]|$)"),
    ("${start}<VERSION>${end}", r"(?P<start>[^0-9.]|^)\d{1,5}(?:\.\d{1,3}){1,4}(?P<end>[^0-9.]|$)"),
    ("${start}<URL>${end}", r"(?P<start>[^A-Za-z0-9:/]|^)(https?://[^\s]+)(?P<end>[^A-Za-z0-9:/]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^A-Za-z]|^)\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{2}| \d) \d{4}\b(?P<end>[^A-Za-z]|$)"),
    ("${start}<TXID>${end}", r"(?P<start>[^0-9A-Fa-f-]|^)\d{4}-[0-9A-Fa-f]{16}(?P<end>[^0-9A-Fa-f-]|$)"),
    ("${start}<FILEPATH>${end}", r"(?P<start>[^A-Za-z0-9:\\]|^)[A-Za-z]:\\(?:[^\\\n]+\\)*[^\\\n]+(?P<end>[^A-Za-z0-9:\\]|$)"),
    ("${start}<APIKEY>${end}", r"(?P<start>[^A-Za-z0-9\"]|^)\"x-apikey\":\s\"[^\"]+\"(?P<end>[^A-Za-z0-9\"]|$)"),
    ("${start}<TIMEMS>${end}", r"(?P<start>[^0-9ms]|^)\b\d+\s+ms\b(?P<end>[^0-9ms]|$)"),
    ("${start}<SECONDS>${end}", r"(?P<start>[^0-9s-]|^)-?\d{1,4}s(?P<end>[^0-9s-]|$)"),
    ("${start}<HEXBLOCKS>${end}", r"(?P<start>[^0-9A-Fa-f-]|^)(?:[0-9A-Fa-f]{4,}-)+[0-9A-Fa-f]{4,}(?P<end>[^0-9A-Fa-f-]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^0-9A-Fa-f]|^)0x[0-9A-Fa-f]+(?P<end>[^0-9A-Fa-f]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^0-9A-Fa-f]|^)([0-9A-Fa-f]{6,})(?P<end>[^0-9A-Fa-f]|$)"),
    ("${start}<LARGEINT>${end}", r"(?P<start>[^0-9]|^)\d{4,}(?P<end>[^0-9]|$)")
    # Additional patterns can be added here in the same format
]
