
import sys
import time
sys.path.append('..')
import polars as pl
import loglead.loaders.base as load, loglead.enhancer as er, loglead.anomaly_detection as ad
full_data = "/home/ubuntu/Datasets"
private_data ="../private_data"

loader = load.HDFSLoader(filename=f"{full_data}/hdfs/HDFS.log", 
                                            labels_file_name=f"{full_data}/hdfs/anomaly_label.csv")
loader.execute()

#Reduce as needed.
#df = loader.reduce_dataframes(0.001)
#df = loader.reduce_dataframes(0.1)
df = loader.df
df_seq =  loader.df_sequences

enchancer = er.EventLogEnhancer(df)
df = enchancer.normalize()
df = enchancer.words(column="e_message_normalized")
df = enchancer.parse_drain()
df = enchancer.parse_spell()
df = enchancer.parse_lenma()
df = enchancer.create_neural_emb()

#Perform an intermediate save here. 
#df.write_parquet(f"{private_data}/P4_Parsed_hdfs_events_01.parquet")
#df_seq.write_parquet(f"{private_data}/P4_Parsed_hdfs_seqs_01.parquet")
#df = pl.read_parquet(f"{private_data}/P4_Parsed_hdfs_events_01percent.parquet")
#df_seq = pl.read_parquet(f"{private_data}/P4_Parsed_hdfs_seqs_01percent.parquet")
#Continue by loading from file
df = pl.read_parquet(f"{private_data}/P4_Parsed_hdfs_events_01.parquet")
df_seq = pl.read_parquet(f"{private_data}/P4_Parsed_hdfs_seqs_01.parquet")


seq_enhancer = er.SequenceEnhancer(df = df, df_seq = df_seq)
seq_enhancer.tokens("e_words")
seq_enhancer.events("e_event_id")
seq_enhancer.events("e_event_lenma_id")
seq_enhancer.events("e_event_spell_id")
seq_enhancer.embeddings("e_bert_emb")

#Using tokens(words) from each sequence 
# Suppress ConvergenceWarning
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
#Disable unsupervised
disable = ["train_IsolationForest", "train_LOF","train_KMeans", "train_OneClassSVM", "train_RarityModel"]
#Need a loop to run these multiple times
sad = ad.AnomalyDetection(store_scores=True, print_scores=False)

for i in range(10):
    print (f"{i}", end ="")
    sad.item_list_col = "e_words"
    sad.test_train_split (seq_enhancer.df_seq, test_frac=0.95)
    sad.evaluate_all_ads(disabled_methods=disable)
    print(".", end="")
    
    sad.item_list_col = "e_event_id"
    sad.prepare_train_test_data()
    sad.evaluate_all_ads(disabled_methods=disable)
    print(".", end="")
    
    sad.item_list_col ="e_event_lenma_id"
    sad.prepare_train_test_data()
    sad.evaluate_all_ads(disabled_methods=disable)
    print(".", end="")
    
    sad.item_list_col ="e_event_spell_id"
    sad.prepare_train_test_data()
    sad.evaluate_all_ads(disabled_methods=disable)
    print(".")
    sad.item_list_col =None
    
    sad.emb_list_col = "e_bert_emb"
    sad.prepare_train_test_data()
    sad.evaluate_all_ads(disabled_methods=disable)
    sad.emb_list_col = None

#-----------------------------------------------
print(sad.storage.calculate_average_scores(score_type="f1").to_csv())

