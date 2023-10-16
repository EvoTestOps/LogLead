#
#Separate demo files
import loader as load, enricher as er, anomaly_detection as ad

dataset = "tb_s_parq" #hdfs, pro, hadoop, tb, tb_small (has no anomalies), tb_s_parq
df = None
df_seq = None
loader = None
if dataset=="hdfs":
       loader = load.HadoopLoader(filename="../../../Datasets/hadoop/",
                                                 filename_pattern  ="*.log",
                                                 labels_file_name="../../../Datasets/hadoop/abnormal_label_accurate.txt")
elif dataset=="hdfs":
       loader = load.HDFSLoader(filename="../../../Datasets/hdfs/HDFS.log", 
                                          labels_file_name="../../../Datasets/hdfs/anomaly_label.csv")
elif dataset=="pro":
       loader = load.ProLoader(filename="../../../Datasets/profilence/*.txt")
elif dataset=="tb":
       loader = load.ThunderbirdLoader(filename="../../../Datasets/thunderbird/Thunderbird.log") #Might take 2-3 minutes in HPC cloud. In desktop out of memory
elif dataset=="tb_small":
       loader = load.ThunderbirdLoader(filename="../../../Datasets/thunderbird/Thunderbird_2k.log") #Only 2k lines
elif dataset=="tb_s_parq":
       import polars as pl
       df = pl.read_parquet("tb_002.parquet")

if loader != None:
       df = loader.execute()
       if (dataset != "hadoop"):
              df = loader.reduce_dataframes(frac=0.02)
       df_seq = loader.df_sequences 
       if (dataset == "tb"):
              df.write_parquet("tb_002.parquet")
#Null should be handeled in the loader. However, if they exist they get killed here
df = df.filter(pl.col("m_message").is_not_null())
#Enrichment-------------------------------------------------
enricher =  er.EventLogEnricher(df)
df = enricher.words()
df = enricher.alphanumerics()
df = enricher.trigrams() #carefull might be slow

#Add anomaly scores for each line.
event_ad = ad.EventAnomalyDetection(df)
df = event_ad.compute_ano_score("e_words", 100)
df = event_ad.compute_ano_score("e_alphanumerics", 100)
#df = event_ad.compute_ano_score("e_cgrams", 100)

#Predict line anomalousness with given input, e.g. words
#Supervised classical ML based AD
sad = ad.SupervisedAnomalyDetection("e_words")
df_eve_train, df_eve_test = ad.test_train_split(df, test_frac=0.95)
res = sad.train_LR(df_eve_train)
res = sad.predict(df_eve_test)
res = sad.train_DT(df_eve_train)
res = sad.predict(df_eve_test)
#Does not work in some environment
#res = sad.train_IsolationForest(df_eve_train, contamination=0.1)
#res = sad.predict(df_eve_test)
res = sad.train_LOF(df_eve_train, filter_anos=True, contamination=0.1)
res = sad.predict(df_eve_test)
res = sad.train_RF(df_eve_train)
res = sad.predict(df_eve_test)
res = sad.train_XGB(df_eve_train)
res = sad.predict(df_eve_test)
