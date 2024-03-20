#This is an example using HDFS data from samples folder. See also similar TB_samples.py 
#The files have been loaded from raw and processed to parquet file format for efficient storage.
#This demonstrates how to work after you have completed the loader.

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
#import resource

#Location of our sample data
sample_data="../samples"

#_________________________________________________________________________________
#Part 2 load data from sample file
#Load HDFS from sample data
df = pl.read_parquet(f"{sample_data}/hdfs_events_2percent.parquet")
df_seqs_all = pl.read_parquet(f"{sample_data}/hdfs_seqs_2percent.parquet")

df_seqs = df_seqs_all.head(1000)
#print(f"Read HDFS 2% sample. Numbers of events is: {len(df)} and number of sequences is {len(df_seqs)}")
#ano_count = df_seqs["anomaly"].sum()
#print(f"Anomaly count {ano_count}. Anomaly percentage in Sequences {ano_count/len(df_seqs)*100:.2f}%")


# amount is a command line argument
# meaning python3 Shap_HDFS.py amount
# should be between above 0 and max 100
if False:
    amount = int(sys.argv[1])
    loader = load.HDFSLoader(filename="../.data/HDFS.log"
        ,labels_file_name="../.data/preprocessed/anomaly_label.csv")
    df = loader.execute()
    df = loader.reduce_dataframes(frac=amount/100)
    df_seqs = loader.df_seq
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


#might do a loop here to do the code with different enhancers


#print(f"Original log message: {df['m_message'][row_index]}")
#Create some enhanced representations
df = enhancer.normalize()
#print(f"as normalized:        {df['e_message_normalized'][row_index]}")
#df = enhancer.words()
#print(f"as words:             {format_as_list(df['e_words'][row_index])}")
#df = enhancer.trigrams()
#print(f"as trigrams:          {format_as_list(df['e_trigrams'][row_index])}")
#df = enhancer.parse_drain()
#print(f"as Drain event id:    {df['e_event_id'][row_index]}")
#df = enhancer.parse_spell()
#print(f"as Spell event id:    {df['e_event_spell_id'][row_index]}")
#df = enhancer.length()
#print(f"event length chars:   {df['e_chars_len'][row_index]}")
#print(f"event length lines:   {df['e_lines_len'][row_index]}")
#print(f"event length words:   {df['e_words_len'][row_index]}")




#_________________________________________________________________________________
#Part 4 Aggregate event level data to sequence level
#print(f"\nStarting aggregating log event info to log sequences:")

seqs_row_index = random.randint(0, len(df_seqs) - 1)
seq_id = df_seqs['seq_id'][seqs_row_index]
#print(f"Sequence level dataframe without aggregated info: {df_seqs.filter(pl.col('seq_id') == seq_id)}")
seq_enhancer = er.SequenceEnhancer(df = df, df_seq = df_seqs) # onko tarpeelölinen
#df_seqs = seq_enhancer.events()
#added normalized message ?
df_seqs = seq_enhancer.events(event_col="e_message_normalized") #e_template
#df_seqs = seq_enhancer.events(event_col="e_template")
#print(f"list of events in a sequence: {format_as_list(df_seqs.filter(pl.col('seq_id') == seq_id)['e_event_id'][0])}")
#df_seqs = seq_enhancer.next_event_prediction()
#print(f"Predicted events ngram:       {format_as_list(df_seqs.filter(pl.col('seq_id') == seq_id)['nep_predict'][0])}")
#df_seqs = seq_enhancer.eve_len()
#df_seqs = seq_enhancer.seq_len()
#print(f"sequence length in events: {df_seqs.filter(pl.col('seq_id') == seq_id)['seq_len'][0]}")
#df_seqs = seq_enhancer.start_time()
#df_seqs = seq_enhancer.end_time()
#df_seqs = seq_enhancer.duration()
#print(f"sequence duration: {df_seqs.filter(pl.col('seq_id') == seq_id)['duration'][0]}")
#df_seqs = seq_enhancer.tokens(token="e_trigrams")
#df_seqs = seq_enhancer.tokens(token="e_words")
#df_seqs = seq_enhancer.tokens(token="e_message_normalized")
# THis crashes for me due to ram
#print(f"Sequence level dataframe with aggregated info: {df_seqs.filter(pl.col('seq_id') == seq_id)}")

# destroy df
del df

#_________________________________________________________________________________________
#Part 5 Do some anomaly detection

sad = ad.AnomalyDetection()
sad.test_train_split (seq_enhancer.df_seq, test_frac=0.90)
sad.numeric_cols = None

# ============================================
# Choose the data what you use to train the model
# ==============================================

if False:
    print(f"Predicting with words")
    sad.item_list_col = "e_words"
    sad.numeric_cols = None #Important otherwise we use both numeric_col and item_list_col for predicting
    sad.prepare_train_test_data() #Data needs to prepared after changing the predictor columns


    # ============================
    # Use only one model at a time
    # ============================

    #Logistic Regression
    sad.train_LR()
    #df_seq = sad.predict()     # we don't need the predictions now so they are removed

    #Use Decision Tree
    #sad.train_DT()
    #df_seq = sad.predict()

if True:
    #print(f"Predicting with Drain parsing results")
    sad.item_list_col =  "e_message_normalized"#"e_template"#""#e_event_id
    sad.prepare_train_test_data()

    # ============================
    # Use only one model at a time
    # ============================   
     
    #Logistic Regression
    #sad.train_LR()
    #df_seq = sad.predict()
    
    #Use Decision Tree
    #sad.train_DT()
    #df_seq = sad.predict()


    # laita tähän se modeli jota olet testaamassa
    # muoto sad.modeli()
   
    #sad.train_RF()
    sad.train_XGB()

# Above choose one model to be trained with wanted data

# Gives the trained model
modelout = sad.get_model

# Get out the train data and the test data
X_train , labels_train = sad.train_data
X_test, labels_test = sad.test_data

# Get out the vectorizer
vect = sad.vec

voc = sad.voc

##
# use e_message_normalized WORKS
#e_template

#unsuper algo
##

# for shap I needed the model and both train and test data
#r1 = resource.getrusage(resource.RUSAGE_SELF)
#print(r1)
t1 = time.time()



#explainer_ebm = shap.LinearExplainer(modelout, X_train)
#explainer = shap.TreeExplainer(modelout, X_train.toarray()) ### RF
explainer = shap.Explainer(modelout) ### XGB

shap_values = explainer(X_test.toarray())




t2 = time.time()
#r2  = resource.getrusage(resource.RUSAGE_SELF)
#print(f"shap took {t2-t1} s")
print(t2-t1)
#print(r2.ru_maxrss-r1.ru_maxrss)
shap.summary_plot(shap_values, X_test,feature_names=vect.get_feature_names_out(), max_display=16)
#shap.plots.bar(shap_values)
#shap.force_plot(explainer_ebm.expected_value[1], shap_values[1], X_test[1])