#Define here any data specific preprocessing

import polars as pl

def preprocess_files(df, preprocessing_steps):
    for step in preprocessing_steps:
        function_name = step['name']
        args = step.get('args', [])
        
        function_to_call = globals().get(function_name)
        if function_to_call:
            # Call the function with the DataFrame and specified arguments
            df = function_to_call(df, *args)
        else:
            print(f"Function {function_name} not found in custom_preprocessing.")
        
    return df




def remove_run_name_from_file_names(df):
    """
    Adjusts the container log file names in a Hadoop environment by removing 
    the application ID part from the file names. This is necessary for making 
    file names consistent across different runs, enabling comparison.
    
    For example, in a run with application ID 'application_1445062781478_0011':
    
    - 'container_1445062781478_0011_01_000001.log' becomes 'container_0011_01_000001.log'
    - 'container_1445062781478_0011_01_000002.log' becomes 'container_0011_01_000002.log'
    
    This renaming ensures that log file names match across all runs after preprocessing.
    """
    # Extract the <Common_part> by removing the 'application' prefix from 'run'
    # Remove all chars before number. My_run_123_2 has common part 123_2
    df = df.with_columns(
        pl.col("run").str.replace_all(r"^[^\d]+", "").alias("common_part")
    )
    print(f"Running hadoop preprocessing remove_run_name_from_file_names")
    
    # Get unique <Common_part> values
    unique_parts = df["common_part"].unique()
    # Loop over unique <Common_part> values to replace them in the 'file_name' column
    for part in unique_parts:
        df = df.with_columns([
            pl.col("file_name").str.replace(part, "").alias("file_name")
        ])

    df = df.drop("common_part")

    return df