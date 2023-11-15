#This is an example using using HDFS from samples folder data 
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
import loglead.loader as load
import loglead.enhancer as er
import loglead.anomaly_detection as ad
import polars as pl
import random
#Location of our sample data
sample_data="../samples"

#_________________________________________________________________________________
#Part 2 load data from sample file
#Load HDFS from sample data
df = pl.read_parquet(f"{sample_data}/hdfs_events_2percent.parquet")
df_seqs = pl.read_parquet(f"{sample_data}/hdfs_seqs_2percent.parquet")
print(f"Read HDFS 2% sample numbers of events is: {len(df)} and number of sequences is {len(df_seqs)}")


#_________________________________________________________________________________
#Part 3 add enhanced represrations 
print(f"\nStarting enhancing all log events:")
enhancer = er.EventLogEnhancer(df)

# For nicer printing a function to format series of elements as a list-like string
def format_as_list(series):
    elements = [str(item) for item in series]
    return '[' + ', '.join(elements) + ']'
#Pick a random line
row_index = random.randint(0, len(df) - 1)


print(f"Original log message: {df['m_message'][row_index]}")
#Create some enhanced representations
df = enhancer.normalize()
print(f"as normalized:        {df['e_message_normalized'][row_index]}")
df = enhancer.words()
print(f"as words:             {format_as_list(df['e_words'][row_index])}")
df = enhancer.trigrams()
print(f"as trigrams:          {format_as_list(df['e_cgrams'][row_index])}")
df = enhancer.parse_drain()
print(f"as Drain event id:    {df['e_event_id'][row_index]}")
df = enhancer.parse_spell()
print(f"as Spell event id:    {df['e_event_spell_id'][row_index]}")
df = enhancer.length()
print(f"event length chars:   {df['e_message_len_char'][row_index]}")
print(f"event length lines:   {df['e_message_len_lines'][row_index]}")
print(f"event length words:   {df['e_message_len_words_ws'][row_index]}")



#_________________________________________________________________________________
#Part 4 Aggregate enchanced representations to sequence lvel
print(f"\nStarting aggregating log event info to log sequences:")

seqs_row_index = random.randint(0, len(df_seqs) - 1)
seq_id = df_seqs['seq_id'][seqs_row_index]
print(f"Sequence level dataframe without aggregated info: {df_seqs.filter(pl.col('seq_id') == seq_id)}")
seq_enhancer = er.SequenceEnhancer(df = df, df_seq = df_seqs)
df_seqs = seq_enhancer.events()
print(f"list of events in a sequence: {format_as_list(df_seqs.filter(pl.col('seq_id') == seq_id)['e_event_id'][0])}")
#df_seqs = seq_enhancer.eve_len()
df_seqs = seq_enhancer.seq_len()
print(f"sequence length in events: {df_seqs.filter(pl.col('seq_id') == seq_id)['seq_len'][0]}")
df_seqs = seq_enhancer.start_time()
df_seqs = seq_enhancer.end_time()
df_seqs = seq_enhancer.duration()
print(f"sequence duration: {df_seqs.filter(pl.col('seq_id') == seq_id)['duration'][0]}")
df_seqs = seq_enhancer.tokens()
print(f"Sequence level dataframe without aggregated info: {df_seqs.filter(pl.col('seq_id') == seq_id)}")

#_________________________________________________________________________________________
#Part 5 we do some anomaly detection
print(f"\nStarting anomaly detection of HDFS Sequences")
numeric_cols = ["seq_len",  "duration_sec",]
sad = ad.AnomalyDetection()
#Using 10% for training 90% for testing
sad.numeric_cols = numeric_cols
sad.test_train_split (seq_enhancer.df_seq, test_frac=0.90)
print(f"using 10% for training and 90% for testing")
print(f"Predicting with sequence length and duration ")

#Logistic Regression
sad.train_LR()
df_seq = sad.predict()
#Use Decision Tree
sad.train_DT()
df_seq = sad.predict()

print(f"Predicting with words")
sad.item_list_col = "e_words"
sad.numeric_cols = None #Important otherwise we use both numeric and item list for predicting
sad.prepare_train_test_data() #Data needs to prepared after chaning predictor columns
#Logistic Regression
sad.train_LR()
df_seq = sad.predict()
#Use Decision Tree
sad.train_DT()
df_seq = sad.predict()

print(f"Predicting with Drain parsing results")
sad.item_list_col = "e_event_id"
sad.prepare_train_test_data()
#Logistic Regression
sad.train_LR()
df_seq = sad.predict()
#Use Decision Tree
sad.train_DT()
df_seq = sad.predict()
