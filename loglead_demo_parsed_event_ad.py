#Sequence levels prediction
import loader as load, enricher as er, anomaly_detection as ad

# Loading HDFS Logs----------------------------------------------------------------
hdfs_processor = load.HDFSLoader(filename="../../../Datasets/hdfs/HDFS.log", 
                                     labels_file_name="../../../Datasets/hdfs/anomaly_label.csv")
df_hdfs = hdfs_processor.execute()
#smaller hdfs for faster running. Parsing of whole HDFS takes about 11min
df_hdfs_s = hdfs_processor.reduce_dataframes(frac=0.02)
  
#-Event enrichment----------------------------------------------
#Parsing in event level
enricher_hdfs = er.EventLogEnricher(df_hdfs_s)
df_hdfs_s = enricher_hdfs.parse_drain()
#Collect events to sequence level as list[str]
seq_enricher = er.SequenceEnricher(df = df_hdfs_s, df_sequences = hdfs_processor.df_sequences)
seq_enricher.enrich_sequence_events()

#Anomaly detection with Logstic Regression----------------------------------------------
#Split
df_seq_train, df_seq_test = ad.test_train_split(seq_enricher.df_sequences, test_frac=0.9)
#Train, test and print results
ad_seq = ad.SeqAnomalyDetection(df_seq_train)
ad_seq.train_LR()
df_seq_test = ad_seq.predict_LR(df_seq_test, print_scores = True)
