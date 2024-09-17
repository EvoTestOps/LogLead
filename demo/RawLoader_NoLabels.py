# This script provides a demonstration of how to proceeed when you do not have a specific loader nor labels for your data.
# The only requirements are two log files:
# 1. A training log file, typically from a successful run, used to build the anomaly detection models.
# 2. An analysis log file, often from a failed or problematic run, which will be scored using the models.
# The script trains models on the training log and then scores each log line in the analysis file.
# A higher score indicates a higher likelihood of an anomaly in the corresponding log line.
# The output is CSV-file for examing scores and lines


import os
import polars as pl
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import numpy as np
import zlib
from loglead.loaders import *
from loglead import AnomalyDetector, OOV_detector
from loglead.enhancers import EventLogEnhancer, SequenceEnhancer
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
full_data = os.getenv("LOG_DATA_PATH")
if not full_data:
    print("WARNING!: LOG_DATA_PATH is not set. This will most likely fail")

# Ensure this always gets executed in the same location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def write_to_str(df: pl.DataFrame, name: str):
    quote_char = "\a"
    csv_string = df.write_csv(separator=" ", quote_char=quote_char, include_header=False)
    csv_string = csv_string.replace(quote_char, '')   # Remove the placeholder quote_char
    with open(name, 'w') as f:
        f.write(csv_string)

def create_unlabeled_BGL(data: str, reduce_df_ratio: float = 0.1, test_ratio: float = 0.03, drop_columns: list = ["normal", "m_timestamp", "anomaly"]):
    """
    Function to create an unlabeled dataset from a BGL data file.
    You may skip this if you have your own data. 
    
    Parameters:
    - data (str): Path to the BGL data file.
    - reduce_df_ratio (float): The ratio by which to reduce the size of the DataFrame for efficient processing. Default is 0.1 (i.e., 10% of the original size).
    - test_ratio (float): The ratio of the data to be set aside as test data. Default is 0.03 (i.e., 3% of the original size).
    - drop_columns (list): List of columns to drop from the DataFrame. Default is ["normal", "m_timestamp", "anomaly"].
    
    Writes:
    - "train.log"
    - "train.label" Not mandotory
    - "test.log"
    - "test.label" Not mandatory

    """
    
    # Load BGL data
    loader = BGLLoader(data)
    df = loader.execute()
    
    # Reduce dataframe. #NEP will not work. 
    df_redu = loader.reduce_dataframes(frac=reduce_df_ratio)
    df_redu = df_redu.sort('m_timestamp')
    
    # Determine test size and split data
    test_size = int(test_ratio * df_redu.shape[0])
    train_df = df_redu.head(-test_size)
    test_df = df_redu.tail(test_size)
    
    # Split dataframe into features and labels
    def split_dataframe(df: pl.DataFrame, label_col: str, drop: list):
        df_label = df.select(pl.col(label_col))
        df_keeped = df.drop(label_col)
        for d in drop: 
            df_keeped = df_keeped.drop(d)
        return df_keeped, df_label

    train_df_no_labels, train_df_label = split_dataframe(train_df, "label", drop_columns)
    test_df_no_labels, test_df_label = split_dataframe(test_df, "label", drop_columns)
    
    write_to_str(train_df_no_labels, "train.log")
    write_to_str(train_df_label, "train.label")
    write_to_str(test_df_no_labels, "test.log")
    write_to_str(test_df_label, "test.label")
    print (f"Unlabeled BGL data subsets created to files: train.log and test.log. Labels are in files: train.label and test.label.")


def create_seq_in_file_HDFS(reduce_df_ratio: float = 0.0005, test_ratio: float = 0.5):
    loader = loader = HDFSLoader(filename=os.path.join(full_data,"hdfs", "HDFS.log"),
                                labels_file_name=os.path.join(full_data,"hdfs", "preprocessed/anomaly_label.csv"))
    df = loader.execute()
        
    # Reduce dataframe.  
    df_redu = loader.reduce_dataframes(frac=reduce_df_ratio)
    df_redu = df_redu.sort('m_timestamp')
    df_redu_seq = loader.df_seq#We operate on sequences

    # Determine test size and split data
    test_size = int(test_ratio * df_redu_seq.shape[0])
    train_df = df_redu_seq.head(-test_size)
    test_df = df_redu_seq.tail(test_size)

    def write_seqs_to_files(df_redu_seq, output_dir):
        for row in df_redu_seq.iter_rows(named=True):
            seq_id = row["seq_id"]
            df_redu_file = df_redu.filter(pl.col("seq_id") == seq_id)
            df_redu_file = df_redu_file.drop("seq_id", "m_timestamp")
            start = "A_" if row["anomaly"] else "N_" 
            file_path = os.path.join(output_dir, f"{start}{seq_id}.log")
            write_to_str(df_redu_file, file_path)

    # Create a directory to store the files
    output_train_dir = "HDFS_train"
    os.makedirs(output_train_dir, exist_ok=True)
    write_seqs_to_files(train_df, output_train_dir)

    output_test_dir = "HDFS_test"
    os.makedirs(output_test_dir, exist_ok=True)
    write_seqs_to_files(test_df, output_test_dir)
    print (f"HDFS data with one sequence at one file created to folders: {output_train_dir} and {output_test_dir}.")
    print (f"Anomaly files start with filename: A_ . Normal files start with filename: N_")





def load_and_enhance (file: str, pattern = None):
    """
    Loads raw data from a file and enhances it by normalizing and splitting to words.

    Parameters:
    - file (str): Path to the data file to be loaded.

    Returns:
    - pl.DataFrame: The enhanced DataFrame with normalized and processed event log messages.
    """
    loader = RawLoader(file, filename_pattern=pattern)
    df = loader.execute()
    #Normalize data. 
    enhancer = EventLogEnhancer(df)
    df = enhancer.normalize()
    df = enhancer.words("e_message_normalized")
    return df

def train_line_models(df):
    """
    Trains multiple anomaly detection models using the provided DataFrame and saves the models and vectorizer.

    Parameters:
    - df (pl.DataFrame): The input DataFrame containing the data to be used for training.

    Returns:
    - None: The function saves the trained models and vectorizer to files.
    """
    #Create anomaly detector and set what to use for training. 
    sad = AnomalyDetector(item_list_col="e_words")
    #Creat train data and vectorizer and save it
    sad.X_train, sad.labels_train, sad.vectorizer = sad._prepare_data(df)
    joblib.dump(sad.vectorizer, 'vectorizer.joblib')
    #Create unsupervised models and save them
    sad.train_KMeans()
    joblib.dump(sad.model, 'kmeans_model.joblib')
    sad.train_IsolationForest()
    joblib.dump(sad.model, 'IF_model.joblib')
    sad.train_RarityModel(filter_anos=False)
    joblib.dump(sad.model, 'RM_model.joblib')
    print (f"Log line anomaly detectors created. Isolation Forrest in: IF_model.joblib, KMeans in: kmeans.joblib, RarityModel in RM_model.joblib")
    #sad.train_OOVDetector() #OOV detector does not need training. Vectorizer is enough
    #joblib.dump(sad.model, 'OOV_model.joblib')

def analyse_with_line_models(df):
    """
    Analyzes log data using multiple anomaly detection models and outputs the predictions to a CSV file.

    Parameters:
    - df (pl.DataFrame): The input DataFrame containing the log data to be analyzed.

    Returns:
    - None: The function saves the predictions to a CSV file.
    """

    #Prepare anomaly detector
    vectorizer = joblib.load('vectorizer.joblib')
    sad = AnomalyDetector(item_list_col="e_words", print_scores=False, auc_roc=True)
    sad.test_df = df
    #Hack. AnomalyDetector was not really built to for this type of pipeline
    sad.X_test, _, _ = sad._prepare_data(df, vectorizer)
    # Run model
    sad.model = joblib.load('kmeans_model.joblib')
    df_anos = sad.predict()
    df_anos = df_anos.rename({"pred_ano_proba": "kmeans_pred_ano_proba"})
    #df_anos= df_anos.with_columns(predictions)

    sad.model = joblib.load('IF_model.joblib')
    predictions = sad.predict().select("pred_ano_proba").rename({"pred_ano_proba": "IF_pred_ano_proba"})
    df_anos = df_anos.with_columns(predictions)

    sad.model = joblib.load('RM_model.joblib')
    predictions = sad.predict().select("pred_ano_proba").rename({"pred_ano_proba": "RM_pred_ano_proba"})
    df_anos = df_anos.with_columns(predictions)
    #OOVDetector
    #We need to set these to None. OOVD does not use them
    sad.X_train=None
    sad.labels_train = None
    sad.train_OOVDetector(filter_anos=False) #This just creates the object. No training for OOVD needed 
    predictions = sad.predict().select("pred_ano_proba").rename({"pred_ano_proba": "OOVD_pred_ano_proba"})
    df_anos = df_anos.with_columns(predictions)
    df_anos.drop("e_words").write_csv("test_log_predicted.csv", quote_style="always", separator='\t')

    print (f"Log line anomaly detectors executed. Scored logfile is in file: test_log_predicted.csv with columns separated with \\t")

def containment_similarity(v_binary1, v_binary2):
    """Containment Similarity: Intersection divided by the smaller vector's size"""
    intersection = np.logical_and(v_binary1, v_binary2).sum()
    return intersection / min(v_binary1.sum(), v_binary2.sum()) if min(v_binary1.sum(), v_binary2.sum()) > 0 else 0


def measure_distance (df_train, df_analyze, field = "m_message", vectorizer = CountVectorizer, print_values=True):
    #setup
    s_train = df_train.select(pl.col(field).str.concat(" ")).item()
    s_analyze = df_analyze.select(pl.col(field).str.concat(" ")).item()
    vectorizer =  vectorizer()
    # Combine both texts to ensure the vectorizer includes all unique words from both
    combined_texts = [s_train, s_analyze]
    # Initialize and fit vectorizer on combined texts
    v_combined = vectorizer.fit_transform(combined_texts)
    # Split the combined vectorized text back into individual vectors
    v_train = v_combined[0]
    v_analyse = v_combined[1]
    # Calculate the cosine similarity
    cosine_sim = float(cosine_similarity(v_train, v_analyse)[0][0])
    # Jaccard
    v_binary1 = (v_train > 0).astype(int)
    v_binary2 = (v_analyse > 0).astype(int)
    #print (v_binary1)
    jaccard = float(jaccard_score(v_binary1, v_binary2,  average="samples"))
    #containment (smaller is subset of larger)
    intersection = v_binary1.multiply(v_binary2).sum()
    containment = float(intersection / min(v_binary1.sum(), v_binary2.sum()) if min(v_binary1.sum(), v_binary2.sum()) > 0 else 0)

    #compression distance
    len1 = len(zlib.compress(s_train.encode()))
    len2 = len(zlib.compress(s_analyze.encode()))
    combined_len = len(zlib.compress((s_train + s_analyze).encode()))
    compression =  combined_len / (len1 + len2)
    if print_values: 
        print(f"Distance of column {field} is Cosine: {cosine_sim}, Jaccard: {jaccard}, Compression: {compression}, Containment: {containment} ")

    return cosine_sim, jaccard, compression, containment

def measure_distance_random_files(df_train, df_analyze, field="e_message_normalized", sample_size=10):
    # Select unique file names and sample 10 from each DataFrame
    unique_train = df_train.select(pl.col("file_name")).unique().sample(sample_size)
    unique_analyze = df_analyze.select(pl.col("file_name")).unique().sample(sample_size)
    
    # Loop through the samples and calculate metrics
    print (f"File distances of random file pairs: ")
    for i in range(sample_size):
        df1 = df_train.filter(pl.col("file_name") == unique_train[i, 0])
        df2 = df_analyze.filter(pl.col("file_name") == unique_analyze[i, 0])
        
        # Calculate the distances
        cos, jaccard, compression, containment = measure_distance(df1, df2, field=field, print_values=False)
        
        # Print the file names and metrics
        print(f"{unique_train[i, 0]} - {unique_analyze[i, 0]}, "
              f"Cosine: {cos:.4f}, Jaccard: {jaccard:.4f}, Compression: {compression:.4f}, Containment: {containment:.4f}")



#BGL
create_unlabeled_BGL(os.path.join(full_data, "bgl/", "BGL.log"))
df_train = load_and_enhance("train.log") 
train_line_models(df_train)
df_analyze = load_and_enhance("test.log")
analyse_with_line_models(df_analyze)
measure_distance(df_train, df_analyze, field="e_message_normalized")
measure_distance(df_train, df_analyze, field="m_message")

#HDFS
create_seq_in_file_HDFS()
df_train = load_and_enhance("HDFS_train", pattern="*.log")
train_line_models(df_train)
df_analyze = load_and_enhance("HDFS_test", pattern="*.log")
analyse_with_line_models(df_analyze)
measure_distance_random_files(df_train, df_analyze, field="e_message_normalized")
measure_distance_random_files(df_train, df_analyze, field="m_message")

