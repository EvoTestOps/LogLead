#This is an example using TB (Thunderbird) data from samples folder. See also similar HDFS_samples.py  
#The files have been loaded from raw and processed to parquet file format for efficient storage.
#This file demonstrates how to work after you have completed the loader.

#______________________________________________________________________________
#Part 1 load libraries and setup paths. 
import sys
import os
#Ensure this always gets executed in the same location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append('..')
import loglead.loaders.base as load
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


#Location of our sample data
sample_data="../samples"

#_________________________________________________________________________________
#Part 2 load data from sample file
#Load TB from sample data
df = pl.read_parquet(f"{sample_data}/tb_0125percent.parquet")
#print(f"Read TB 0.125% sample. Numbers of events: {len(df)}")
#ano_count = df["anomaly"].sum()
#print(f"Anomaly count {ano_count}. Anomaly percentage in Events {ano_count/len(df)*100:.2f}%")

#amount = int(sys.argv[1])
#loader = load.HDFSLoader(filename="../.data/HDFS.log"
#    ,labels_file_name="../.data/preprocessed/anomaly_label.csv")
#df = loader.execute()
#df = loader.reduce_dataframes(frac=amount/100)
#df_seqs = loader.df_seq
#__________________________________



#_________________________________________________________________________________
#Part 3 add enhanced reprisentations 
#print(f"\nStarting enhancing all log events:")
enhancer = er.EventLogEnhancer(df)

# For nicer printing a function to format series of elements as a list-like string
def format_as_list(series):
    elements = [str(item) for item in series]
    return '[' + ', '.join(elements) + ']'
#Pick a random line

#Create some enhanced representations
df = enhancer.normalize()

#Spell parser takes a bit too long for the video  
#df = enhancer.parse_spell()
#print(f"as Spell event id:    {df['e_event_spell_id'][row_index]}")

#_________________________________________________________________________________________
#Part 5 we do some anomaly detection. No part 4 here as TB is not labeled on sequence level. See HDFS_samples.py
sad = ad.AnomalyDetection()
#Using 10% for training 90% for testing


#"e_message_normalized"#"e_template"#

sad.test_train_split (df, test_frac=0.90)

sad.item_list_col = "e_message_normalized"
sad.numeric_cols = None #Important otherwise we use both numeric_col and item_list_col for predicting
sad.prepare_train_test_data() #Data needs to prepared after changing the predictor columns
#Logistic Regression
sad.train_LR()
#df_seq = sad.predict()
#Use Decision Tree
#sad.train_DT()
#df_seq = sad.predict()



modelout = sad.get_model

# Get out the train data and the test data
X_train , labels_train = sad.train_data
X_test, labels_test = sad.test_data

# Get out the vectorizer
vect = sad.vec

voc = sad.voc


# use e_message_normalized WORKS
#e_template

#unsuper algo

# for shap I needed the model and both train and test data
r1 = resource.getrusage(resource.RUSAGE_SELF)

t1 = time.time()
explainer_ebm = shap.LinearExplainer(modelout, X_train, labels=vect.get_feature_names_out())
shap_values = explainer_ebm(X_test)
t2 = time.time()
r2  = resource.getrusage(resource.RUSAGE_SELF)

print(t2-t1)
print(r2.ru_maxrss-r1.ru_maxrss)
#shap.summary_plot(shap_values, X_test,feature_names=vect.get_feature_names_out(), max_display=16)
#shap.plots.bar(shap_values)
