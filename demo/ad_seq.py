#Sequence levels prediction
import sys
sys.path.append('..')
import loglead.loader as load, loglead.enhancer as er, loglead.anomaly_detection as ad
import time

full_data = "/home/ubuntu/Datasets"
private_data ="../private_data"
dataset = "hdfs" #hdfs, pro, hadoop, tb, tb-small

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
       df = pl.read_parquet(f"{private_data}/hdfs_events_002.parquet")
       df_seq = pl.read_parquet(f"{private_data}/hdfs_seqs_002.parquet")

if loader != None:
       df = loader.execute()
       if (dataset != "hadoop"):
              df = loader.reduce_dataframes(frac=0.02)
       df_seq = loader.df_sequences       
       if (dataset == "hdfs"):
              df.write_parquet(f"{private_data}/hdfs_events_002.parquet")
              df_seq.write_parquet(f"{private_data}/hdfs_seqs_002.parquet")
              

#df = loader.execute()
#if dataset!="hadoop":
#    df = loader.reduce_dataframes(frac=0.02)
#df_seq = loader.df_sequences

  
#-Event enrichment----------------------------------------------
#Parsing in event level
enhancer = er.EventLogEnhancer(df)
df = enhancer.length()
df = enhancer.parse_drain()
df = enhancer.words()
df = enhancer.alphanumerics()

#Collect events to sequence level as list[str]
seq_enhancer = er.SequenceEnhancer(df = df, df_sequences = df_seq)
seq_enhancer.events()
seq_enhancer.eve_len()
seq_enhancer.start_time()
seq_enhancer.end_time()
seq_enhancer.seq_len()
seq_enhancer.duration()
seq_enhancer.tokens()


# Suppress ConvergenceWarning
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

#Using tokens(words) from each sequence 
sad = ad.SupervisedAnomalyDetection(item_list_col="e_words")
sad.test_train_split (seq_enhancer.df_sequences, test_frac=0.95)
sad.evaluate_all_ads()

# AD using only numeric columns:
#Defining numeric columns to be inclded
numeric_cols = ["seq_len", "eve_len_max", "duration_sec", "eve_len_over1"]
sad = ad.SupervisedAnomalyDetection(numeric_cols=numeric_cols)
sad.test_train_split (seq_enhancer.df_sequences, test_frac=0.95)
sad.evaluate_all_ads(disabled_methods=["train_RarityModel"])#Rarity model not working here for some reason. 

# AD using only event column:
sad =  ad.SupervisedAnomalyDetection(item_list_col="e_event_id")
sad.test_train_split (seq_enhancer.df_sequences, test_frac=0.95)
sad.evaluate_all_ads()

#Events + Numeric
sad =  ad.SupervisedAnomalyDetection(item_list_col="e_event_id", numeric_cols=numeric_cols)
sad.test_train_split (seq_enhancer.df_sequences, test_frac=0.95)
sad.evaluate_all_ads()