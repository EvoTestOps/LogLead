#Sequence levels prediction
import sys
sys.path.append('/home/ubuntu/Development/mika/LogLEAD')
import loglead.loader as load, loglead.enricher as er, loglead.anomaly_detection as ad
import time

full_data = "/home/ubuntu/Datasets"
private_data ="../private_data"
dataset = "hdfs_s_parq" #hdfs, pro, hadoop, tb, tb-small

df = None
df_seq = None
loader = None

if dataset=="hadoop":
       loader = load.HadoopLoader(filename=f"{full_data}/hadoop/",
                                                 filename_pattern  ="*.log",
                                                 labels_file_name=f"{full_data}/hadoop/abnormal_label_accurate.txt")
elif dataset=="hdfs":
       loader = load.HDFSLoader(filename=f"{full_data}/hdfs/HDFS.log", 
                                          labels_file_name=f"{full_data}/hdfs/anomaly_label.csv")
elif dataset=="pro":
       loader = load.ProLoader(filename=f"{full_data}/profilence/*.txt")
elif dataset=="tb":
       loader = load.ThunderbirdLoader(filename=f"{full_data}/thunderbird/Thunderbird.log") #Might take 2-3 minutes in HPC cloud. In desktop out of memory
elif dataset=="tb-small":
       loader = load.ThunderbirdLoader(filename=f"{full_data}/thunderbird/Thunderbird_2k.log") #Only 2k lines
elif dataset=="hdfs_s_parq":
       import polars as pl
       df = pl.read_parquet(f"{private_data}/hdfs_events_02.parquet")
       df_seq = pl.read_parquet(f"{private_data}/hdfs_seqs_02.parquet")

if loader != None:
       df = loader.execute()
       if (dataset != "hadoop"):
              df = loader.reduce_dataframes(frac=0.2)
       df_seq = loader.df_sequences       
       if (dataset == "hdfs"):
              df.write_parquet(f"{private_data}/hdfs_events_02.parquet")
              df_seq.write_parquet(f"{private_data}/hdfs_seqs_02.parquet")
              

#df = loader.execute()
#if dataset!="hadoop":
#    df = loader.reduce_dataframes(frac=0.02)
#df_seq = loader.df_sequences

  
#-Event enrichment----------------------------------------------
#Parsing in event level
enricher = er.EventLogEnricher(df)
df = enricher.length()
#df = enricher.parse_drain()
df = enricher.words()
df = enricher.alphanumerics()

#Collect events to sequence level as list[str]
seq_enricher = er.SequenceEnricher(df = df, df_sequences = df_seq)
#seq_enricher.events()
seq_enricher.eve_len()
seq_enricher.start_time()
seq_enricher.end_time()
seq_enricher.seq_len()
seq_enricher.duration()
seq_enricher.tokens()


sad = ad.SupervisedAnomalyDetection(item_list_col="e_words")
sad.test_train_split (seq_enricher.df_sequences, test_frac=0.95)
# Suppress ConvergenceWarning
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
sad.evaluate_all_ads()

#OLD-WAY__________________________________________________________________________
#Split
df_seq_train, df_seq_test = ad.test_train_split(seq_enricher.df_sequences, test_frac=0.95)

#Anomaly detection with Logstic Regression and DT----------------------------------------------

#Using tokens(words) from each sequence 
sad = ad.SupervisedAnomalyDetection(item_list_col="e_words")
sad.dep_evaluate_all_ads(df_seq_train, df_seq_test)

# sad.train_LR(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_DT(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_SVM(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_IsolationForest(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_RF(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_XGB(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)


# AD using only numeric columns:
#Defining numeric columns to be inclded
numeric_cols = ["seq_len", "eve_len_max", "duration_sec", "eve_len_over1"]
sad = ad.SupervisedAnomalyDetection(numeric_cols=numeric_cols)
sad.evaluate_all_ads(df_seq_train, df_seq_test)

# sad.train_LR(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_DT(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_SVM(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_IsolationForest(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_RF(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_XGB(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)


# AD using only event column:
sad =  ad.SupervisedAnomalyDetection(item_list_col="events")
sad.evaluate_all_ads(df_seq_train, df_seq_test)

# sad.train_LR(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_DT(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_SVM(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_IsolationForest(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_RF(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_XGB(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)


# AD using both
sad =  ad.SupervisedAnomalyDetection(item_list_col="events", numeric_cols=numeric_cols)
sad.evaluate_all_ads(df_seq_train, df_seq_test)

# sad.train_LR(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_DT(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_SVM(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_IsolationForest(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_RF(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
# sad.train_XGB(df_seq_train)
# df_seq_test = sad.predict(df_seq_test, print_scores = True)
