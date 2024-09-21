import os
import polars as pl
import inspect
import datetime
from loglead.loaders import RawLoader
from loglead import LogDistance, AnomalyDetector
import umap
import plotly.express as px
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from loglead.enhancers import EventLogEnhancer

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

def _prepare_runs(df, target_run, comparison_runs="ALL"):
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
        # Remove the target run if it's present in the comparison runs
        comparison_runs = [run for run in comparison_runs if run != target_run]
        # Assume comparison_runs is a list and validate all provided comparison runs
        if not all(run in unique_runs for run in comparison_runs):
            invalid_runs = [run for run in comparison_runs if run not in unique_runs]
            raise ValueError(f"Comparison run names {invalid_runs} not found in the dataframe. Please provide valid run names.")
    
    return run1, comparison_runs

def _check_multiple_target_runs(df, base_runs):

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
        # If base_runs is a single string, convert it to a list
        if isinstance(base_runs, str):
            base_runs = [base_runs]
        # Assume comparison_runs is a list and validate all provided comparison runs
        if not all(run in unique_runs for run in base_runs):
            invalid_runs = [run for run in base_runs if run not in unique_runs]
            raise ValueError(f"Comparison run names {invalid_runs} not found in the dataframe. Please provide valid run names.")
    return base_runs

def plot_run(df: pl.DataFrame, target_run: str, comparison_runs="ALL", file=True, random_seed=None, group_by_indices=None, normalize_content=False, content_format="Words", vectorizer="Count"):
    """
    Create a UMAP plot based on a document-term matrix of file names and save it as an interactive HTML file.
    
    Parameters:
    - df: Polars DataFrame containing the data with a 'run' column and 'file_name' column.
    - target_run: Name of the target run to highlight.
    - comparison_runs: List of comparison run names to include in the plot.
    - file: Flag to indicate do use file names (True) or file contents (False)
    - random_seed: Random seed for reproducibility.
    - group_by_indices: List of integers indicating which parts of the 'run' string to group by.
    """
    # Apply the grouping by indices if specified
    if group_by_indices:
        df = _plot_group_runs_by_indices(df, group_by_indices)

    _, comparison_run_names = _prepare_runs(df, target_run, comparison_runs)
    print(
        f"Executing {inspect.currentframe().f_code.co_name} with {'file' if file else 'content'} norm={normalize_content} with {content_format} and Vectorizer:{vectorizer} target run '{target_run}' and {len(comparison_run_names)} comparison runs"
        + (f": {comparison_run_names}" if len(comparison_run_names) < 6 else "")
    )

    # Filter out target run and comparison runs from the DataFrame
    runs_to_include = [target_run] + comparison_run_names
    filtered_df = df.filter(pl.col("run").is_in(runs_to_include))
    filtered_df, field = _plot_prepare_content(filtered_df, normalize_content, content_format=content_format)
    #print (f"field: {field}")
    run_file_groups, documents = _plot_aggregate_run_file_groups(filtered_df, field, content_format, group_by_indices)
    embeddings_2d, num_unique_words_per_file = _plot_create_dtm_and_umap(documents, content_format, vectorizer, random_seed=None)
    
    #Prepare simple plot lines X unique_terms
    line_count = filtered_df.group_by("run").agg([pl.count().alias("line_count")]).sort("run")
    line_count_values = line_count.select("line_count").to_numpy().ravel()
    num_unique_words_per_file = np.asarray(num_unique_words_per_file).ravel()  # Ensure it's a 1D array
    line_count_values = np.asarray(line_count_values).ravel()  # Ensure it's a 1D array
    combined_data = np.column_stack((embeddings_2d, num_unique_words_per_file, line_count_values))

    fig1, fig2 = _plot_create_umap_plot(combined_data, run_file_groups, group_by_indices, target_run, file)
    
    #fig = _plot_create_umap_plot(embeddings_2d, run_file_groups, group_by_indices, target_run, file)
    write_dataframe_to_csv(fig1, analysis="plot_umap", level=1 if file else 2, norm=normalize_content, target_run=target_run, comparison_run="Many", content_format=content_format, vectorizer=vectorizer)
    write_dataframe_to_csv(fig2, analysis="plot_simple", level=1 if file else 2, norm=normalize_content, target_run=target_run, comparison_run="Many", content_format=content_format, vectorizer=vectorizer)




def _plot_group_runs_by_indices(df: pl.DataFrame, group_by_indices: list[int]) -> pl.DataFrame:
    """
    Groups the 'run' column based on specified indices.

    Parameters:
    - df: Polars DataFrame containing the 'run' column.
    - group_by_indices: List of integer indices specifying which parts to group by.

    Returns:
    - Polars DataFrame with an added 'group' column.
    """

    #S1: Split run string to multiple parts 1_33_44 becomes [1, 33, 44]
    df = df.with_columns(
        pl.col("run").str.split("_").alias("run_parts")
    )
    #S2:  Loop over and select the parts we want to keep for groups to a new df
    df_new = None
    for item in group_by_indices:
        if df_new is None:
           df_new = df.select(pl.col("run_parts").list.get(item, null_on_oob=True).alias(f"part_{item}"))
           df_new = df_new.with_columns(pl.col(f"part_{item}").fill_null(""))
        else:
           df_new = df_new.with_columns(df.select(pl.col("run_parts").list.get(item, null_on_oob=True).alias(f"part_{item}")))
           df_new = df_new.with_columns(pl.col(f"part_{item}").fill_null(""))

    #S3: New df contains only correct columns we merge 
    df_new = df_new.with_columns(pl.concat_str(pl.col("*"), separator="_",).alias("group"),)
    #S4: Lose the extra columns
    df_new = df_new.select(pl.col("group"))
    df = df.drop("run_parts")
    #S5 add the group column to df
    df = df.with_columns(df_new)
    return df

def _plot_prepare_content(df, normalize_content, content_format):
    """
    Function to process content (words, trigrams, etc.)  if content format SKLearn or not specified 
    """
    field = "e_message_normalized" if normalize_content else "m_message"
    enhancer = EventLogEnhancer(df)
    if content_format == "Words":
        df = enhancer.words(field)
        return df, "e_words"
    elif content_format == "3grams":
        df = enhancer.trigrams(field)
        return df, "e_trigrams"
    elif content_format == "Parse":
        df = enhancer.parse_tip(field)
        return df, "e_event_tip_id"
    elif content_format == "File":
        return df, "file_name"
    elif content_format == "Sklearn":
        return df, field
    else:
        print(f"Unrecognized content format: {content_format}")
        raise ValueError(f"Unrecognized content format: {content_format}")

def _plot_aggregate_run_file_groups(filtered_df_file, field, content_format, group_by_indices):
    """
    Process the DataFrame and prepare the run file groups based on content format and grouping by indices.

    Parameters:
    - filtered_df: The full DataFrame (used for non-file-specific operations).
    - filtered_df_file: The filtered DataFrame for the file being processed.
    - field: The column name to aggregate.
    - content_format: The format of content (e.g., 'Sklearn', 'Parse').
    - group_by_indices: Whether to group by indices or not.

    Returns:
    - run_file_groups: Aggregated and grouped data for the runs and their content.
    - documents: List of concatenated file name strings for the runs.
    """

    # Handle 'Sklearn' or 'Parse' content formats (no exploding, just aggregation)
    if content_format == "Sklearn" or content_format == "Parse":
        if group_by_indices:
            run_file_groups = filtered_df_file.select("run", field, "group").group_by("run").agg(
                pl.col(field),  # Keep the string column intact for later use in CountVectorizer or if Parse
                pl.col("group").first()  # First value of the group
            )
        else:
            run_file_groups = filtered_df_file.select("run", field).group_by("run").agg(
                pl.col(field)  # Keep the string column intact
            )
    elif content_format == "File":
        if group_by_indices:
            run_file_groups = filtered_df_file.select("run", field, "group").group_by("run").agg(
                pl.col(field).unique(), 
                pl.col("group").first()
            )
        else: 
            run_file_groups = filtered_df_file.select("run", field, "group").group_by("run").agg(
                pl.col("file_name").unique()
            )
    else:
        if group_by_indices:
            run_file_groups = filtered_df_file.select("run", field, "group").explode(field).group_by("run").agg(
                pl.col(field), pl.col("group").first()
            )
        else:
            run_file_groups = filtered_df_file.select("run", field).explode(field).group_by("run").agg(
                pl.col(field)
            )
    
    run_file_groups = run_file_groups.sort("run") 
    column_data = run_file_groups.select(pl.col(field))
    if content_format == "Sklearn":
        column_data = column_data.with_columns(
            pl.col(field).list.join(' ')  # Concatenate the list elements with a space separator
        )
        documents = column_data.select(pl.col(field)).to_series().to_list()
    else:
        documents = column_data.to_series().to_list()

    return run_file_groups, documents

def _plot_create_dtm_and_umap(documents, content_format, vectorizer_type, random_seed=None):
    """
    Create a document-term matrix (DTM) and perform UMAP dimensionality reduction.

    Parameters:
    - documents: List of document strings to be vectorized.
    - content_format: The format of the content ('Sklearn' or others).
    - vectorizer_type: Type of vectorizer ('Count' or 'Tfidf').
    - random_seed: Optional seed for UMAP to ensure reproducibility.

    Returns:
    - embeddings_2d: UMAP-reduced embeddings in 2D space.
    """

    # Set vectorizer parameters based on the content format
    vectorizer_params = {
        'tokenizer': lambda x: x,
        'preprocessor': None,
        'token_pattern': None,
        'lowercase': False
    } if content_format != "Sklearn" else {}

    # Create the vectorizer (Count or Tfidf)
    if vectorizer_type == "Count":
        vect = CountVectorizer(**vectorizer_params)
    elif vectorizer_type == "Tfidf":
        vect = TfidfVectorizer(**vectorizer_params)
    else:
        raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")

    # Fit the vectorizer to the documents and create the document-term matrix
    dtm = vect.fit_transform(documents)

    # Initialize UMAP with or without a random seed
    reducer = umap.UMAP(random_state=random_seed) if isinstance(random_seed, int) else umap.UMAP()

    # Perform UMAP dimensionality reduction on the document-term matrix
    embeddings_2d = reducer.fit_transform(dtm.toarray())
    unique_terms_per_document = (dtm > 0).sum(axis=1)
    return embeddings_2d, unique_terms_per_document

def _plot_create_umap_plot(embeddings_2d, run_file_groups, group_by_indices, target_run, file):
    """
    Create a UMAP plot using Plotly based on the provided embeddings and run group information.

    Parameters:
    - embeddings_2d: UMAP-reduced 2D embeddings.
    - run_file_groups: DataFrame containing run and group information.
    - group_by_indices: Flag indicating whether grouping by indices is applied.
    - target_run: The target run for highlighting in the plot.
    - file: The file name for which the UMAP plot is being created.

    Returns:
    - fig: Plotly scatter plot figure object.
    """

    if  isinstance(file, str):
        title = f"{file} distances - Target run: {target_run} with diamond"
    elif file:
        title = f"Run (file) distances - Target run: {target_run} with diamond"    
    else:
        title = f"Run (content) distances - Target run: {target_run} with diamond"    


    # Extract run labels from the run_file_groups DataFrame
    run_labels = run_file_groups["run"].to_list()

    # Determine group labels and color map based on grouping by indices
    if group_by_indices:
        group_labels = run_file_groups.select(pl.col("group")).to_series().to_list()
        unique_groups = sorted(set(group_labels))
        color_discrete_map = {group: f'rgba({(i*50)%255}, {(i*100)%255}, {(i*150)%255}, 1)' for i, group in enumerate(unique_groups)}
    else:
        group_labels = ['all'] * len(run_labels)
        color_discrete_map = {'all': 'blue'}

    # Create a Polars DataFrame for plotting
    plot_df = pl.DataFrame({
        "UMAP1": embeddings_2d[:, 0],
        "UMAP2": embeddings_2d[:, 1],
        "run": run_labels,
        "group": group_labels  # Add group labels for coloring
    })
    # Convert Polars DataFrame to dictionary format for Plotly plotting
    plot_dict = plot_df.to_dict(as_series=False)
    # Define marker symbols: 'cross' for target run, 'circle' for others
    marker_symbols = ['circle' if run != target_run else 'cross' for run in run_labels]
    # Create the interactive UMAP plot using Plotly
    fig1 = px.scatter(
        plot_dict,
        x="UMAP1",
        y="UMAP2",
        color="group",  # Color based on groups or a single color
        symbol=marker_symbols,  # Set custom marker symbols
        color_discrete_map=color_discrete_map,  # Set custom colors based on groups or a single color
        hover_data={"run": True, "UMAP1": False, "UMAP2": False},  # Only show the run, hide UMAP coordinates
        title=title
    )
     # Create a Polars DataFrame for plotting

    if file == True:
        x_title = "Files"
    else:
        x_title = "Unique terms"


    plot_df = pl.DataFrame({
        x_title: embeddings_2d[:, 2],
        "Lines": embeddings_2d[:, 3],
        "run": run_labels,
        "group": group_labels  # Add group labels for coloring
    })
    # Convert Polars DataFrame to dictionary format for Plotly plotting
    plot_dict = plot_df.to_dict(as_series=False)
    # Define marker symbols: 'cross' for target run, 'circle' for others
    marker_symbols = ['circle' if run != target_run else 'cross' for run in run_labels]
    # Create the interactive UMAP plot using Plotly
    fig2 = px.scatter(
        plot_dict,
        x=x_title,
        y="Lines",
        color="group",  # Color based on groups or a single color
        symbol=marker_symbols,  # Set custom marker symbols
        color_discrete_map=color_discrete_map,  # Set custom colors based on groups or a single color
        hover_data={"run": True, x_title: False, "Lines": False},  # Only show the run, hide UMAP coordinates
        title=title,
        log_y=True  # Set the Y-axis to log scale
    )


    return fig1, fig2


def plot_file_content(df: pl.DataFrame, target_run: str, comparison_runs="ALL", target_files="ALL", random_seed=None, group_by_indices=None, normalize_content=False, content_format="Words", vectorizer="Count"):
    """
    Create a UMAP plot based on a document-term matrix of file names and save it as an interactive HTML file.
    
    Parameters:
    - df: Polars DataFrame containing the data with a 'run' column and 'file_name' column.
    - target_run: Name of the target run to highlight.
    - comparison_runs: List of comparison run names to include in the plot.
    - random_state: If True, UMAP is randomized; if a number, it's used as the seed for reproducibility.
    
    The function will create a document-term matrix from the file names for each run.
    """
    #field = "e_message_normalized" if normalize_content else "m_message"
    # Apply the grouping by indices if specified
    if group_by_indices:
        df = _plot_group_runs_by_indices(df, group_by_indices)

    df_run1, comparison_run_names = _prepare_runs(df, target_run, comparison_runs)
    print(
        f"Executing {inspect.currentframe().f_code.co_name} with norm={normalize_content}, {content_format} and Vectorizer:{vectorizer} on target run '{target_run}', target file {target_files} and {len(comparison_run_names)} comparison runs"
        + (f": {comparison_run_names}" if len(comparison_run_names) < 6 else "")
    )

    # Filter out target run and comparison runs from the DataFrame
    runs_to_include = [target_run] + comparison_run_names
    filtered_df = df.filter(pl.col("run").is_in(runs_to_include))
    target_files = _prepare_files(df_run1, target_files)

    filtered_df, field = _plot_prepare_content(filtered_df, normalize_content, content_format=content_format)

    for file in target_files:
        filtered_df_file = filtered_df.filter(pl.col("file_name") == file).sort("file_name")
        run_file_groups, documents = _plot_aggregate_run_file_groups(filtered_df_file, field, content_format, group_by_indices)
        embeddings_2d, num_unique_words_per_file = _plot_create_dtm_and_umap(documents=documents, content_format=content_format, vectorizer_type=vectorizer, random_seed=random_seed)
        
        #fig = _plot_create_umap_plot(embeddings_2d, run_file_groups, group_by_indices, target_run, file)
        
        #num_unique_words_per_file = (dtm > 0).sum(axis=1).A1
        #Prepare simple plot lines X unique_terms
        line_count = filtered_df_file.group_by("run").agg([pl.count().alias("line_count")]).sort("run")
        line_count_values = line_count.select("line_count").to_numpy().ravel()
        num_unique_words_per_file = np.asarray(num_unique_words_per_file).ravel()  # Ensure it's a 1D array
        line_count_values = np.asarray(line_count_values).ravel()  # Ensure it's a 1D array
        combined_data = np.column_stack((embeddings_2d, num_unique_words_per_file, line_count_values))

        fig1, fig2 = _plot_create_umap_plot(combined_data, run_file_groups, group_by_indices, target_run, file)
        write_dataframe_to_csv(fig1, analysis="plot_umap", level=3, target_run=target_run, comparison_run="Many", file=file, norm=normalize_content, content_format=content_format, vectorizer=vectorizer)
        write_dataframe_to_csv(fig2, analysis="plot_simple", level=3, target_run=target_run, comparison_run="Many", file=file, norm=normalize_content, content_format=content_format, vectorizer=vectorizer)





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
    run1, comparison_run_names = _prepare_runs(df, target_run, comparison_runs) 
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
    run1, comparison_run_names = _prepare_runs(df, target_run, comparison_runs) 
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
            "target_lines": distance.size1,
            "comparison_lines": distance.size2,
            "cosine": cosine,
            "jaccard": jaccard,
            "compression": compression,
            "containment": containment
        })
        # Print a dot to indicate progress
        print(".", end="", flush=True)

    #Z-score Normalization + Sum of Distances to get one score 
    results = calculate_zscore_sum(results)

    print()  # Newline after progress dots
    results_df = pl.DataFrame(results)
    write_dataframe_to_csv(results_df, analysis="dis", level=2, target_run=target_run, comparison_run="Many", norm=normalize_content)

def calculate_zscore_sum(results):
    import numpy as np
    from scipy.stats import zscore
    """
    This function normalizes the distance measures in the results using Z-scores,
    sums the normalized values for each comparison run, and appends the zscore_sum
    to the respective result dictionaries.

    Args:
    results (list of dicts): Each dictionary contains distance measures (cosine, jaccard, compression, containment)
                             for each comparison run.

    Returns:
    list of dicts: Updated results with an additional 'zscore_sum' key for each run.
    """
    
    # Create the distance matrix from the results
    distance_matrix = np.array([
        [result["cosine"], result["jaccard"], result["compression"], result["containment"]]
        for result in results
    ])
    
    # Normalize each distance column using z-scores
    normalized_distances = np.apply_along_axis(zscore, axis=0, arr=distance_matrix)
    
    # Sum the normalized distances for each comparison run
    zscore_sum = normalized_distances.sum(axis=1)
    
    # Append the z-score sum to each result
    for idx, result in enumerate(results):
        result['zscore_sum'] = zscore_sum[idx]
    
    return results

def distance_file_content(df, target_run, comparison_runs="ALL", target_files=False, normalize_content=False):
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

    run1, comparison_run_names = _prepare_runs(df, target_run, comparison_runs) 
    results = []
    if target_files:
        target_files = _prepare_files(run1, target_files)

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

        if target_files:
            matching_file_names_list = list(set(target_files).intersection(set(matching_file_names_list)))
        if len(matching_file_names_list) == 0:
            continue
        print(
            f"Comparing against '{other_run}' with {len(matching_file_names_list)} matching files"
            + (f":  {matching_file_names_list}" if len(matching_file_names_list) < 6 else "")
            )
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
                'target_lines': distance.size1,
                'comparison_lines': distance.size2, 
                'cosine': cosine,
                'jaccard': jaccard,
                'compression': compression,
                'containment': containment,
            }
            results.append(result)
            # Print a dot to indicate progress
            print(".", end="", flush=True)
        results = calculate_zscore_sum(results)
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
    df_run1, comparison_run_names = _prepare_runs(df, target_run, comparison_runs) 
    target_files = _prepare_files(df_run1, target_files)
    df_other_runs = df.filter(pl.col("run").is_in(comparison_run_names))
    print(
        f"Executing {inspect.currentframe().f_code.co_name} with target run '{target_run}' and {len(comparison_run_names)} comparison runs"
        + (f": {comparison_run_names}" if len(comparison_run_names) < 6 else "")
    )
    # Compare the base run to each specified comparison run
    for other_run in comparison_run_names:
        for file_name in target_files:
            df_run1_file =  df_run1.filter(pl.col("file_name") == file_name)
            df_other_run_file = df_other_runs.filter(pl.col("run") == other_run) #Filter one run
            df_other_run_file = df_other_run_file.filter(pl.col("file_name") == file_name) #Filter one file
            distance = LogDistance(df_run1_file, df_other_run_file, field=field)
            diff = distance.diff_lines()
            
            write_dataframe_to_csv(diff, analysis="dis", level=4, target_run=target_run, comparison_run=other_run, norm=normalize_content, file=file_name)
            print(".", end="", flush=True) #Progress on screen
    print()  # Newline after progress dots


def _prepare_files(df_run1, files="ALL"):
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
    elif isinstance(files, str) and "*" in files:
        # Convert the glob-like pattern to a regex and filter available files
        import re
        pattern = re.compile(files.replace(".", r"\.").replace("*", ".*"))
        matched_files = [file for file in available_files if pattern.match(file)]
        if not matched_files:
            raise ValueError(f"No files matched the pattern: {files}")
        files = matched_files
    
    else:
        raise ValueError(f"Invalid type for 'files': {files}. It must be 'ALL', a list, an integer, or a pattern.")

    
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
 
    df_run1, comparison_run_names = _prepare_runs(df, target_run, comparison_runs) 
    print(
        f"Executing {inspect.currentframe().f_code.co_name} with target run '{target_run}' and {len(comparison_run_names)} comparison runs"
        + (f": {comparison_run_names}" if len(comparison_run_names) < 6 else "")
    )
    target_files = _prepare_files(df_run1, target_files)
    df_other_runs = df.filter(pl.col("run").is_in(comparison_run_names))
    print(f"Predicting {len(target_files)} files: {target_files}")
    # Loop over each file first
    for file_name in target_files:
        df_run1_files =  df_run1.filter(pl.col("file_name") == file_name)
        df_other_runs_files = df_other_runs.filter(pl.col("file_name") == file_name)
        if df_other_runs_files.height == 0:
            print(f"Found no files matching files in comparisons runs for file: {file_name}")
            continue

        df_anos = _run_anomaly_detection(df_run1_files,df_other_runs_files, field, detectors=detectors, drop_input=False)
        #Add moving averages. 
        df_anos_100 = _calculate_moving_average_all_numeric(df_anos, 10)
        df_anos_1000 = _calculate_moving_average_all_numeric(df_anos, 100)
        df_anos = df_anos.with_columns(df_anos_100)
        df_anos = df_anos.with_columns(df_anos_1000)
        df_anos = df_anos.with_row_index("line_number")
        #write_dataframe_to_csv(df_anos, output_csv)
        write_dataframe_to_csv(df_anos, analysis="ano", level=4, target_run=target_run, comparison_run="Many", norm=normalize_content, file=file_name)
        print(".", end="", flush=True) #Progress on screen
    print()  # Newline after progress dots

def _calculate_moving_average_all_numeric(df: pl.DataFrame, window_size: int) -> pl.DataFrame:
    """
    Internal function to calculate the moving average for all numeric columns in a Polars DataFrame.

    Args:
        df (pl.DataFrame): The Polars DataFrame containing the data.
        window_size (int): The size of the window over which to calculate the moving average.

    Returns:
        pl.DataFrame: A new DataFrame with only the moving averages for each numeric column.
    """
    # Get all numeric columns
    numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]

    if not numeric_cols:
        raise ValueError("No numeric columns found in the DataFrame")

    # Create a new DataFrame with the moving averages
    moving_avg_df = pl.DataFrame()

    # Add moving average for each numeric column to the new DataFrame
    for column in numeric_cols:
        moving_avg_column = f"moving_avg_{window_size}_{column}"
        moving_avg_df = moving_avg_df.hstack(
            df.select(pl.col(column).rolling_mean(window_size).alias(moving_avg_column))
        )

    return moving_avg_df

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

    target_run_names = _check_multiple_target_runs(df, target_run)
    # Extract unique runs
    df_anos_merge = pl.DataFrame() 
    for target_run in target_run_names:
        df_run1, comparison_run_names = _prepare_runs(df, target_run, comparison_runs) 
        # Generate output CSV file name based on base run
        print(
            f"Executing {inspect.currentframe().f_code.co_name} with target run '{target_run}' and {len(comparison_run_names)} comparison runs"
            + (f": {comparison_run_names}" if len(comparison_run_names) < 6 else "")
        )
        target_files = _prepare_files(df_run1, target_files)
        print(f"Predicting {len(target_files)} files: {target_files}")
        df_other_runs = df.filter(pl.col("run").is_in(comparison_run_names))
        #df_anos_merge = pl.DataFrame()

        for file_name in target_files:
            
            df_run1_files =  df_run1.filter(pl.col("file_name") == file_name)
            df_run1_files = df_run1_files.group_by('file_name').agg(pl.col(field).alias(field))

            df_other_runs_files = df_other_runs.filter(pl.col("file_name") == file_name)
            df_other_runs_files = df_other_runs.group_by('file_name').agg(pl.col(field).alias(field))

            if df_other_runs_files.height == 0:
                print(f"Found no files matching files in comparisons runs for file: {file_name}")
                continue

            df_anos = _run_anomaly_detection(df_run1_files,df_other_runs_files,detectors=detectors, field= field)

            df_anos = df_anos.with_columns(pl.lit(file_name).alias("file_name"))
            df_anos = df_anos.with_columns(pl.lit(target_run).alias("target_run"))
            df_anos = df_anos.with_columns(pl.lit(" ".join(comparison_run_names)).alias("comparison_runs"))
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
    - file: Flag to indicate do use file names (True) or file contents (False)
    """
    if file: 
        field = "file_name"
    elif normalize_content: 
        field = "e_message_normalized"
    else:
        field = "m_message"
    df_anos_merge = pl.DataFrame()
    target_run_names = _check_multiple_target_runs(df, target_run)
    
    print(f"Executing {inspect.currentframe().f_code.co_name} with {'file' if file else 'content'} anomalies of {len(target_run_names)} target runs with {comparison_runs} comparison runs")
    print(f"Target runs: {target_run_names}")
    print(f"Comparison runs: {comparison_runs}")
    for target_run_name in target_run_names:
        df_run1, comparison_run_names = _prepare_runs(df, target_run_name, comparison_runs)
        
        df_run1 = df_run1.group_by("run").agg(pl.col(field).alias(field))
        df_other_runs = df.filter(pl.col("run").is_in(comparison_run_names)).group_by("run").agg(pl.col(field).alias(field))
        df_anos = _run_anomaly_detection(df_run1, df_other_runs, detectors=detectors, field= field)
        comparison_runs_out = " ".join(comparison_run_names)
        df_anos = df_anos.with_columns(pl.lit(comparison_runs_out).alias("comparison_runs"))
        df_anos_merge = df_anos_merge.vstack(df_anos)
        print(".", end="", flush=True)
    print()  # Newline after progress dots
    write_dataframe_to_csv(df_anos_merge, analysis="ano", level=1 if file else 2, target_run="Many", comparison_run="Many", norm=normalize_content)


def _run_anomaly_detection(df_run1_files,df_other_runs_files, field, detectors=["KMeans", "RarityModel"], drop_input=True):
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

def write_dataframe_to_csv(df, analysis, level=0, target_run="", comparison_run="", file="", norm=False, content_format="", vectorizer="", separator='\t', quote_style='always'):
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
        sanitized_file_name = file.replace('/', '_').replace('\\', '_')
        output_csv += f"_{sanitized_file_name}"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_csv += f"_{timestamp}"
    # Finalize the file name with the CSV extension
    if content_format:
        output_csv += f"_{content_format}"

    if vectorizer:
        output_csv += f"_{vectorizer}"

    if isinstance(df, pl.DataFrame):
        output_csv += ".xlsx"
    else:
        output_csv += ".html"
    # Combine script_dir and output_folder to get the full directory path
    global output_folder
    output_directory = os.path.join(script_dir, output_folder)
    
    # Ensure the directory exists; if not, create it
    os.makedirs(output_directory, exist_ok=True)
    
    # Construct the full path for the CSV file
    output_path = os.path.join(output_directory, output_csv)
    if isinstance(df, pl.DataFrame):
        df.write_excel(output_path)
    else:
        df.write_html(output_path)
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

masking_patterns_myllari2 = [
    ("${start}<QUOTED_ALPHANUMERIC>${end}", r"(?P<start>[^A-Za-z0-9-_]|^)'[a-zA-Z0-9-_]{16,}'(?P<end>[^A-Za-z0-9-_]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9/]|^)\d{2}/\d{2}/\d{4}(?P<end>[^0-9/]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^0-9-]|^)\d{4}-\d{2}-\d{2}(?P<end>[^0-9-]|$)"),
    ("${start}<DATE_XX>${end}", r"(?P<start>[^A-Za-z0-9_]|^)DATE_\d{2}(?P<end>[^A-Za-z0-9_]|$)"),
    ("${start}<DATE>${end}", r"(?P<start>[^A-Za-z]|^)\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), \d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b(?P<end>[^A-Za-z]|$)"),
    ("${start}<TIME>${end}", r"(?P<start>[^0-9:]|^)\d{2}:\d{2}:\d{2},\d{3}(?P<end>[^0-9:]|$)"),  # New pattern for HH:MM:SS,MMM format
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
    ("${start}<LARGEINT>${end}", r"(?P<start>[^0-9]|^)\d{4,}(?P<end>[^0-9]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^A-Za-z0-9]|^)(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)([\-\+]?[1-9]\d+)(?P<end>[^A-Za-z0-9]|$)")
]

