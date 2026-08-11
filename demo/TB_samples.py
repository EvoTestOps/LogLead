# This is an example using TB (Thunderbird) data from samples folder. See also similar HDFS_samples.py
# The files have been loaded from raw and processed to parquet file format for efficient storage.
# This file demonstrates how to work after you have completed the loader.

# ______________________________________________________________________________
# Part 1 load libraries and setup paths.
import os

import polars as pl

from loglead.enhancers import EventLogEnhancer
from loglead import AnomalyDetector

# Ensure this always gets executed in the same location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Location of our sample data
sample_data = os.path.join(script_dir, 'samples', 'tb_0125percent.parquet')

# _________________________________________________________________________________
# Part 2 load data from sample file
# Load TB from sample data
df = pl.read_parquet(sample_data)
print(f"Read TB 0.125% sample. Numbers of events: {len(df)}")
ano_count = df["anomaly"].sum()
print(f"Anomaly count {ano_count}. Anomaly percentage in Events {ano_count/len(df)*100:.2f}%")


# _________________________________________________________________________________
# Part 3 add enhanced representations
print(f"\nStarting enhancing all log events:")
enhancer = EventLogEnhancer(df)


# For nicer printing a function to format series of elements as a list-like string
def format_as_list(series):
    elements = [str(item) for item in series]
    return '[' + ', '.join(elements) + ']'


# Pick a random line
row_index = 1
# row_index = random.randint(0, len(df) - 1)


print(f"Original log message: {df['m_message'][row_index]}")
# Create some enhanced representations
df = enhancer.normalize()
print(f"as normalized:        {df['e_message_normalized'][row_index]}")
df = enhancer.words()
print(f"as words:             {format_as_list(df['e_words'][row_index])}")
df = enhancer.trigrams()
print(f"as trigrams:          {format_as_list(df['e_trigrams'][row_index])}")
df = enhancer.parse_drain()
print(f"as Drain event id:    {df['e_event_drain_id'][row_index]}")
df = enhancer.parse_tip()
print(f"as Tipping event id:    {df['e_event_tip_id'][row_index]}")
# Spell parser takes a bit too long for the video
# df = enhancer.parse_spell()
# print(f"as Spell event id:    {df['e_event_spell_id'][row_index]}")
df = enhancer.length()
print(f"event length chars:   {df['e_chars_len'][row_index]}")
print(f"event length lines:   {df['e_lines_len'][row_index]}")
print(f"event length words:   {df['e_words_len'][row_index]}")

# _________________________________________________________________________________________
# Part 5 we do some anomaly detection. No part 4 here as TB is not labeled on sequence level. See HDFS_samples.py
print(f"\nStarting anomaly detection of TB Events")
numeric_cols = ["e_chars_len",  "e_lines_len",]
sad = AnomalyDetector()
# Using 10% for training 90% for testing
sad.numeric_cols = numeric_cols
sad.test_train_split(df, test_frac=0.90)
print(f"using 10% for training and 90% for testing")
print(f"Predicting with sequence length and duration ")

# Logistic Regression
sad.train_LR()
df_seq = sad.predict()
# Use Decision Tree
sad.train_DT()
df_seq = sad.predict()

print(f"Predicting with words")
sad.item_list_col = "e_words"
sad.numeric_cols = None  # Important otherwise we use both numeric_col and item_list_col for predicting
sad.prepare_train_test_data()  # Data needs to prepared after changing the predictor columns
# Logistic Regression
sad.train_LR()
df_seq = sad.predict()
# Use Decision Tree
sad.train_DT()
df_seq = sad.predict()

print(f"Predicting with Drain parsing results")
sad.item_list_col = "e_event_drain_id"
sad.prepare_train_test_data()
# Logistic Regression
sad.train_LR()
df_seq = sad.predict()
# Use Decision Tree
sad.train_DT()
df_seq = sad.predict()

# Predict from the fields the log arrived with, instead of from its message text. Thunderbird's
# loader keeps component, userid, location and the date parts as their own columns, and those
# carry real signal on their own. They are text, so they cannot go in numeric_cols - sklearn would
# get an object array and the fit would fail - which is what categorical_cols is for: it one-hot
# encodes them into the same sparse matrix the word/parser representations use.
print(f"Predicting with the log's own categorical fields, no message text")
sad.item_list_col = None
sad.categorical_cols = ["month", "day", "component", "date", "userid"]
sad.prepare_train_test_data()
# Logistic Regression
sad.train_LR()
df_seq = sad.predict()
# Use Decision Tree
sad.train_DT()
df_seq = sad.predict()
# Beware of columns that give the answer away: TB's own "label" column *is* the target here
# (anomaly == label != "-"), so predicting from it would score perfectly and prove nothing.
# loglead.select_predictors() profiles a dataframe and rejects such columns, along with
# identifiers and columns with no variation. Uncomment to see what it picks and what it drops:
# from loglead import select_predictors, print_predictor_report
# numeric, categorical, profile = select_predictors(df)
# print_predictor_report(profile, "Thunderbird")
sad.categorical_cols = []  # Reset so the runs below use only what they set themselves

# ____________________________________________________________
# Part 6 run all anomaly detectors and store scores
print(f"Running all anomaly detectors, excluding OneClassSVM and LOF, with Words and Trigrams and storing results")
print(f"We run everything two times - Adjust as needed")

sad = AnomalyDetector(store_scores=True, print_scores=False)
for i in range(2):  # We do just two loops in this demo
    sad.item_list_col = "e_words"
    sad.test_train_split(df, test_frac=0.90)
    sad.evaluate_all_ads(disabled_methods=["train_OneClassSVM", "train_LOF"])
    
    # We keep existing split but need to prepare data for trigram data
    sad.item_list_col = "e_trigrams"
    sad.prepare_train_test_data()
    sad.evaluate_all_ads(disabled_methods=["train_OneClassSVM", "train_LOF"])


print(f"Inspecting results. Averages of runs:")
print(sad.storage.calculate_average_scores(score_type="accuracy"))
print(f"Confusion matrices can also be inspected")
sad.storage.print_confusion_matrices("LogisticRegression", "e_trigrams")
