#This is an example using HDFS data from samples folder. Demonstrates the functionalities
# of the explainer module.

#______________________________________________________________________________
#Part 1 load libraries and setup paths. 
import sys
import os
#Ensure this always gets executed in the same location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append('..')
import loglead.loaders.hdfs as load
import loglead.enhancer as er
import loglead.anomaly_detection as ad
import polars as pl
import random

#our teams imports
import pickle
import numpy as np
import shap
import time
import resource
import matplotlib.pyplot as plt
import loglead.explainer as ex

#Location of our sample data
sample_data="../samples"

#_________________________________________________________________________________
#Part 2 load data from sample file
#Load HDFS from sample data
df = pl.read_parquet(f"{sample_data}/hdfs_events_2percent.parquet")
df_seqs = pl.read_parquet(f"{sample_data}/hdfs_seqs_2percent.parquet")
print(f"Read HDFS 2% sample. Numbers of events is: {len(df)} and number of sequences is {len(df_seqs)}")
ano_count = df_seqs["anomaly"].sum()
print(f"Anomaly count {ano_count}. Anomaly percentage in Sequences {ano_count/len(df_seqs)*100:.2f}%")


#_________________________________________________________________________________
#Part 3 add enhanced reprisentations 
#print(f"\nStarting enhancing all log events:")
enhancer = er.EventLogEnhancer(df)

# For nicer printing a function to format series of elements as a list-like string
def format_as_list(series):
    elements = [str(item) for item in series]
    return '[' + ', '.join(elements) + ']'
#Pick a random line
row_index = random.randint(0, len(df) - 1)

#Create some enhanced representations
df = enhancer.normalize()
df = enhancer.words()



#_________________________________________________________________________________
#Part 4 Aggregate event level data to sequence level
#print(f"\nStarting aggregating log event info to log sequences:")

seqs_row_index = random.randint(0, len(df_seqs) - 1)
seq_id = df_seqs['seq_id'][seqs_row_index]
print(f"Sequence level dataframe without aggregated info: {df_seqs.filter(pl.col('seq_id') == seq_id)}")
seq_enhancer = er.SequenceEnhancer(df = df, df_seq = df_seqs) 

# event_col can be changed
df_seqs = seq_enhancer.events(event_col="e_message_normalized") 
df_seqs = seq_enhancer.tokens(token="e_words")

# destroy df to save RAM
del df

#_________________________________________________________________________________________
#Part 5 Do some anomaly detection

sad = ad.AnomalyDetection()
sad.test_train_split (seq_enhancer.df_seq, test_frac=0.90)
sad.numeric_cols = None

# ============================================
# Choose the data what you use to train the model
# ==============================================
 
sad.item_list_col =  "e_message_normalized"
sad.prepare_train_test_data()

# ============================
# Use only one model at a time
# ============================   

# SHAPExplainer requires only the model, NNexplainer needs model and predictions
# SHAPExplainer supported models:
# LogisticRegression,LinearSVC, IsolationForest,DecisionTreeClassifier,
# RandomForestClassifier, XGBClassifier
sad.train_LR()
df_seq = sad.predict()



# Create a ShapExplainer object with trained anomaly detection object
# By default doesn't run if too large dataset or too many feature names
# This can be prevented by argumnet ignore_warning=True
# When creating the object one can determine the length of truncated featurenames
# unfortunately due to implementation this cannot be changed later.
ex1 = ex.ShapExplainer(sad, ignore_warning=True, plot_featurename_len=18)


# ShapExplainer can calculate shap values using calc_shapvalues method
# This has two optional arguments: data, custom_slice
# by default uses test data from anomaly detection object
# can be done inplace
# returns shap values
ex1.calc_shapvalues()

# After calculation the feature names can be sorted by shap importance
sfeaturenames = ex1.sorted_featurenames()

# If the shap values were not saved at calculation they can be extracterd.
# The values are in same order as features in vectorizer
shapvalues = ex1.shap_values  # shape (data, feature)
# The features of vectorizer can be also extracted by
featurenames = ex1.feature_names # shape (feature,)

# Plotting does not require to calculate shapvalues before hand.
# However, if data doesn't change and values are already calculated they are not 
# calculated again.

# The plot function has three plots: "summary", "bar", "beeswarm", default:"summary"
# The plot can be given vectorized data to plot with, defaults to anomaly detection test data
# if no existing data nor calculated shapvalues
ex1.plot(plottype="summary")

# A custom slice can be given to plot.
# Slice calculates shap values again.
ex1.plot(plottype="bar", custom_slice=slice(19,20))

# After slice the data needs to be reset due to shapvalues also being sliced.
# However, the values are much faster to calculate with smaller data so
# sliced plots should be left last.
X_test, labels_test = sad.test_data

# The number of displayed features can be changed.
# These n features are also printed in terminal in correct order.
ex1.plot(data=X_test,plottype="beeswarm", displayed=10)

# To get the shap values in the order of most impactful to least (same order as sorted_featurenames)
testvals = ex1.sorted_shapvalues()
# Prints the shapvalues of First feature
print(testvals[0])
# dimension feature x data
print(testvals.shape)

# Get the test data for NNexplainer
X_test, labels_test = sad.test_data

# Initialize NNExplainer
nn_explainer = ex.NNExplainer(df=df_seq, X=X_test, id_col="seq_id", pred_col="pred_normal")

# Plot the logs with their predictions in 2D scatter plot, might take a while
nn_explainer.plot_features_in_two_dimensions(ground_truth_col="anomaly")

# Print features of anomalous and the closest normal instances, features can be specified as list to feature_cols
nn_explainer.print_features_from_nn_mapping(feature_cols=["e_message_normalized"])

# Prints the log content of the anomalous and the closest normal instances in the mapping. Only use if e_words column is specified
# The logs can be long
nn_explainer.print_log_content_from_nn_mapping()

# NNexplainer can be used to create a mapping from anomalous instances to the closest normal instances
mapping = nn_explainer.mapping
ids = mapping.select(pl.col("anomalous_id")).to_numpy().flatten()
id_col="seq_id"

# We can then use the mapping for only looking at the anomalous instances
df = df_seq.with_columns(pl.when(pl.col(id_col).is_in(ids)).then(True).otherwise(False).alias("include"))
mask = df.select(pl.col("include")).to_numpy().flatten()

# now the used data can be masked and plotted using shap
maskeddata = X_test[mask]
ex1.plot(maskeddata, plottype="summary")
