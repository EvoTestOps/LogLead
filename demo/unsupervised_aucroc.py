#
#Separate demo files
import sys
sys.path.append('..')
# Suppress ConvergenceWarning
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import loglead.loader as load, loglead.enhancer as er, loglead.anomaly_detection as ad
import polars as pl

full_data = "/home/ubuntu/Datasets"


loader = load.BGLLoader(filename=f"{full_data}/bgl/BGL.log")
df = loader.execute()
df = loader.reduce_dataframes(frac=0.05)
df = df.filter(pl.col("m_message").is_not_null())
enricher =  er.EventLogEnhancer(df)

regexs = [('0','\d'),('0','0+')]
df = enricher.normalize(regexs, to_lower=True)
df = enricher.trigrams("e_message_normalized")

sad =  ad.AnomalyDetection(item_list_col="e_cgrams")
sad.test_train_split (df, test_frac=0.7, vec_name="TfidfVectorizer",oov_analysis=True)
sad.evaluate_all_ads(disabled_methods=["train_LOF", "train_OneClassSVM","train_LR","train_DT","train_LSVM","train_RF","train_XGB"])




loader = load.HadoopLoader(filename=f"{full_data}/hadoop/",
                                            filename_pattern  ="*.log",
                                            labels_file_name=f"{full_data}/hadoop/abnormal_label_accurate.txt")
df = loader.execute()
df = loader.reduce_dataframes(frac=1)
df_seq = loader.df_seq       

df = df.filter(pl.col("m_message").is_not_null())
enricher =  er.EventLogEnhancer(df)

regexs = [('0','\d'),('0','0+')]
df = enricher.normalize(regexs, to_lower=True)
df = enricher.trigrams("e_message_normalized")


seq_enricher = er.SequenceEnhancer(df = df, df_seq = df_seq)
seq_enricher.tokens(token="e_cgrams")
sad =  ad.AnomalyDetection(item_list_col="e_cgrams")
sad.test_train_split (seq_enricher.df_seq, test_frac=0.7, vec_name="TfidfVectorizer",oov_analysis=True)
print(sad.test_df)
sad.evaluate_all_ads(disabled_methods=["train_LOF", "train_OneClassSVM","train_LR","train_DT","train_LSVM","train_RF","train_XGB"])




loader = load.HDFSLoader(filename=f"{full_data}/hdfs/HDFS.log", 
                                    labels_file_name=f"{full_data}/hdfs/anomaly_label.csv")
df = loader.execute()
df = loader.reduce_dataframes(frac=0.05)
df_seq = loader.df_seq       

df = df.filter(pl.col("m_message").is_not_null())
enricher =  er.EventLogEnhancer(df)

regexs = [('0','\d'),('0','0+')]
df = enricher.normalize(regexs, to_lower=True)
df = enricher.trigrams("e_message_normalized")


seq_enricher = er.SequenceEnhancer(df = df, df_seq = df_seq)
seq_enricher.tokens(token="e_cgrams")
sad =  ad.AnomalyDetection(item_list_col="e_cgrams")
sad.test_train_split (seq_enricher.df_seq, test_frac=0.7, vec_name="TfidfVectorizer",oov_analysis=True)
print(sad.test_df)
sad.evaluate_all_ads(disabled_methods=["train_LOF", "train_OneClassSVM","train_LR","train_DT","train_LSVM","train_RF","train_XGB"])
