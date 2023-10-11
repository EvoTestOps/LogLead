#Sequence levels prediction
import loader as load, enricher as er, anomaly_detection as ad

#MM File name should be loglead_demo_neural_parser or loglead_demo_embedding_parser

#Which one to run. Only one true.
b_hadoop = True
b_hdfs = False
b_profilence = False


# Loading HDFS Logs----------------------------------------------------------------
#hdfs_processor = load.HDFSLoader(filename="../../../Datasets/hdfs/HDFS.log",
#                                     labels_file_name="../../../Datasets/hdfs/anomaly_label.csv")
#df = hdfs_processor.execute()
#smaller hdfs for faster running. Parsing of whole HDFS takes about 11min
#df = hdfs_processor.reduce_dataframes(frac=0.2)

df = None
df_seq = None
preprocessor = None

if (b_hadoop):
       preprocessor = load.HadoopLoader(filename="../../../Datasets/hadoop/",
                                                 filename_pattern  ="*.log",
                                                 labels_file_name="../../../Datasets/hadoop/abnormal_label_accurate.txt")

elif (b_hdfs):
       preprocessor = load.HDFSLoader(filename="../../../Datasets/hdfs/HDFS.log",
                                          labels_file_name="../../../Datasets/hdfs/anomaly_label.csv")

elif (b_profilence):
       preprocessor = load.ProLoader(filename="../../../Datasets/profilence/*.txt")

df = preprocessor.execute()
if (not b_hadoop):
    df = preprocessor.reduce_dataframes(frac=0.02)
df_seq = preprocessor.df_sequences

#df = df.head(7000)
#print(df.head(3))
#print(df.select(df['m_message']).head(3))
#-Event enrichment----------------------------------------------
#Parsing in event level
enricher_hdfs = er.EventLogEnricher(df)
enricher_hdfs.create_neural_emb()