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
#Location of our sample data
sample_data="../samples"

#_________________________________________________________________________________
#Part 2 load data from sample file
#Load TB from sample data
df_all = pl.read_parquet(f"{sample_data}/tb_0125percent.parquet")
df = df_all.head(10000)
print(f"Read TB 0.125% sample. Numbers of events: {len(df)}")
ano_count = df["anomaly"].sum()
print(f"Anomaly count {ano_count}. Anomaly percentage in Events {ano_count/len(df)*100:.2f}%")


#_________________________________________________________________________________
#Part 3 add enhanced reprisentations 
print(f"\nStarting enhancing all log events:")
enhancer = er.EventLogEnhancer(df)

# For nicer printing a function to format series of elements as a list-like string
def format_as_list(series):
    elements = [str(item) for item in series]
    return '[' + ', '.join(elements) + ']'
#Pick a random line
row_index = 1 #random.randint(0, len(df) - 1)


print(f"Original log message: {df['m_message'][row_index]}")
#Create some enhanced representations
df = enhancer.normalize()
print(f"as normalized:        {df['e_message_normalized'][row_index]}")
df = enhancer.words()
print(f"as words:             {format_as_list(df['e_words'][row_index])}")
df = enhancer.trigrams()
print(f"as trigrams:          {format_as_list(df['e_trigrams'][row_index])}")
df = enhancer.parse_drain()
print(f"as Drain event id:    {df['e_event_id'][row_index]}")
#Spell parser takes a bit too long for the video  
#df = enhancer.parse_spell()
#print(f"as Spell event id:    {df['e_event_spell_id'][row_index]}")
df = enhancer.length()
print(f"event length chars:   {df['e_chars_len'][row_index]}")
print(f"event length lines:   {df['e_lines_len'][row_index]}")
print(f"event length words:   {df['e_words_len'][row_index]}")

#_________________________________________________________________________________________
#Part 5 we do some anomaly detection. No part 4 here as TB is not labeled on sequence level. See HDFS_samples.py
print(f"\nStarting anomaly detection of TB Events")
numeric_cols = ["e_chars_len",  "e_lines_len",]
sad = ad.AnomalyDetection()
#Using 10% for training 90% for testing
sad.numeric_cols = numeric_cols
sad.test_train_split (df, test_frac=0.90)
print(f"using 10% for training and 90% for testing")
print(f"Predicting with sequence length and duration ")

#Logistic Regression
sad.train_LR()
df_seq = sad.predict()
df_seq.write_parquet('C:\DS Project\LogLeadFork\sequence_LR.parquet')
#Use Decision Tree
sad.train_DT()
df_seq = sad.predict()
df_seq.write_parquet('C:\DS Project\LogLeadFork\sequence_DT.parquet')

print(f"Predicting with words")
sad.item_list_col = "e_words"
sad.numeric_cols =  None #Important otherwise we use both numeric_col and item_list_col for predicting
sad.prepare_train_test_data() #Data needs to prepared after changing the predictor columns
#Logistic Regression
sad.train_LR()
df_seq = sad.predict()
df_seq.write_parquet('C:\DS Project\LogLeadFork\words_LR.parquet')

#Use Decision Tree
sad.train_DT()
df_seq = sad.predict()
df_seq.write_parquet('C:\DS Project\LogLeadFork\words_DT.parquet')


print(f"Predicting with Drain parsing results")
sad.item_list_col = "e_event_id"
sad.prepare_train_test_data()
#Logistic Regression
sad.train_LR()
df_seq = sad.predict()
#Use Decision Tree
sad.train_DT()
df_seq = sad.predict()

