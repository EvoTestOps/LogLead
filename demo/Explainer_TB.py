# This is an example using TB (Thunderbird) data from samples folder. Demonstrates the functionalities
# of the explainer module.

#______________________________________________________________________________
#Part 1 load libraries and setup paths. 
import sys
import os
#Ensure this always gets executed in the same location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append('..')
import loglead.loaders.supercomputers as tbload
#supercomputers.py
from loglead.enhancers import EventLogEnhancer, SequenceEnhancer
from loglead import AnomalyDetector
import polars as pl
import random
import loglead.explainer as ex

# Ensure this always gets executed in the same location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
# Location of our sample data
sample_data = os.path.join(script_dir, 'samples')


#_________________________________________________________________________________
#Part 2 load data from sample file
#Load TB from sample data
df = pl.read_parquet(f"{sample_data}/tb_0125percent.parquet")
print(f"Read TB 0.125% sample. Numbers of events: {len(df)}")
ano_count = df["anomaly"].sum()
print(f"Anomaly count {ano_count}. Anomaly percentage in Events {ano_count/len(df)*100:.2f}%")

# Due to limitations in RAM we use only some of the data
# If you have extensive adjust accordingly 
df = df.head(100000)


#_________________________________________________________________________________
#Part 3 add enhanced reprisentations 
print(f"\nStarting enhancing all log events:")
enhancer = EventLogEnhancer(df)

# For nicer printing a function to format series of elements as a list-like string
def format_as_list(series):
    elements = [str(item) for item in series]
    return '[' + ', '.join(elements) + ']'

#Create some enhanced representations
df = enhancer.normalize()
df = enhancer.words()


#_________________________________________________________________________________________
#Part 5 we do some anomaly detection. No part 4 here as TB is not labeled on sequence level. See HDFS_samples.py
sad = AnomalyDetector()
#Using 10% for training 90% for testing
sad.test_train_split (df, test_frac=0.90)

sad.item_list_col = "e_message_normalized"
sad.numeric_cols = None #Important otherwise we use both numeric_col and item_list_col for predicting
sad.prepare_train_test_data() #Data needs to prepared after changing the predictor columns

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


# create id column
df_seq = df_seq.with_columns(pl.Series(name="id", values=[i for i in range(df_seq.height)]))

# Get the test data for NNexplainer
X_test, labels_test = sad.test_data

# Initialize NNExplainer
nn_explainer = ex.NNExplainer(df=df_seq, X=X_test, id_col="id", pred_col="pred_ano")

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
id_col="id"

# We can then use the mapping for only looking at the anomalous instances
df = df_seq.with_columns(pl.when(pl.col(id_col).is_in(ids)).then(True).otherwise(False).alias("include"))
mask = df.select(pl.col("include")).to_numpy().flatten()

# now the used data can be masked and plotted using shap
maskeddata = X_test[mask]
ex1.plot(maskeddata, plottype="summary")
