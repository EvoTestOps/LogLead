#
#Separate demo files
import loader as load, enricher as er, anomaly_detection as ad

dataset = "tb_s_emb" #hdfs, pro, hadoop, tb, tb_small (has no anomalies), tb_s_parq tb_s_emb
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
elif dataset=="hadoop_emb":
       import polars as pl
       df = pl.read_parquet("hadoop_events_bert_emb.parquet")      
       df_seq = pl.read_parquet("hadoop_seqs_bert_emb.parquet") 
elif dataset=="tb_s_emb":
       import polars as pl
       df = pl.read_parquet("tb_s_emb_events_bert_emb.parquet")      


if loader != None:
       df = loader.execute()
       if (dataset != "hadoop"):
              df = loader.reduce_dataframes(frac=0.02)
       df_seq = loader.df_sequences 
       if (dataset == "tb"):
              df.write_parquet("tb_002.parquet")
              
#Null should be handeled in the loader. However, if they exist they get killed here
#TODO does not work df_seqs like hdfs if they have null
df = df.filter(pl.col("m_message").is_not_null()) 
#Create bert embeddings
enricher = er.EventLogEnricher(df)
df = enricher.create_neural_emb("e_basebert_emb") #238second / whole thing 4:17 hadoop

#Save the file so no need to rerun later bert
df.write_parquet(f"{dataset}_events_bert_emb.parquet")
if dataset in ("hadoop", "hdfs", "profilence"):      
       df_seq.write_parquet(f"{dataset}_seqs_bert_emb.parquet")


df_eve_train, df_eve_test = ad.test_train_split(df, test_frac=0.98)
#-------------------------------------------------------------
#Predict using embeddings
sad = ad.SupervisedAnomalyDetection(None, None, "e_basebert_emb")
res = sad.train_LR(df_eve_train)
res = sad.predict(df_eve_test)
res = sad.train_DT(df_eve_train)
res = sad.predict(df_eve_test)
res = sad.train_SVM(df_eve_train, max_iter=300)
res = sad.predict(df_eve_test)
res = sad.train_IsolationForest(df_eve_train,filter_anos=False)
res = sad.predict(df_eve_test)
res = sad.train_LOF(df_eve_train, filter_anos=True, contamination=0.1)
res = sad.predict(df_eve_test)
res = sad.train_RF(df_eve_train)
res = sad.predict(df_eve_test)
res = sad.train_XGB(df_eve_train)
res = sad.predict(df_eve_test)

sad = ad.SupervisedAnomalyDetection(None, None, "e_bert_emb")
res = sad.train_LR(df_eve_train)
res = sad.predict(df_eve_test)
res = sad.train_DT(df_eve_train)
res = sad.predict(df_eve_test)
res = sad.train_IsolationForest(df_eve_train,filter_anos=False)
res = sad.predict(df_eve_test)
res = sad.train_LOF(df_eve_train, filter_anos=True, contamination=0.1)
res = sad.predict(df_eve_test)
res = sad.train_RF(df_eve_train)
res = sad.predict(df_eve_test)
res = sad.train_XGB(df_eve_train)
res = sad.predict(df_eve_test)