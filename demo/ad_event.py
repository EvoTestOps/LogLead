#
#Separate demo files
import sys
sys.path.append('..')

import loglead.loader as load, loglead.enhancer as er, loglead.anomaly_detection as ad
import polars as pl

full_data = "/home/ubuntu/Datasets"

dataset = "bgl" #hdfs, pro, hadoop, tb, tb_small (has no anomalies), tb_s_parq
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
elif dataset=="tb_small":
       loader = load.ThunderbirdLoader(filename=f"{full_data}/thunderbird/Thunderbird_2k.log") #Only 2k lines
elif dataset=="tb_s_parq":
       df = pl.read_parquet("tb_002.parquet")
elif dataset=="bgl":
       loader = load.BGLLoader(filename=f"{full_data}/bgl/BGL.log")

if loader != None:
       df = loader.execute()
       if (dataset != "hadoop"):
              df = loader.reduce_dataframes(frac=0.02)
       df_seq = loader.df_sequences 
       if (dataset == "tb"):
              df.write_parquet("tb_002.parquet")
#Null should be handeled in the loader. However, if they exist they get killed here
df = df.filter(pl.col("m_message").is_not_null())
#Enhancement-------------------------------------------------
enhancer =  er.EventLogEnhancer(df)

#Normalization before 
normalize = True
column = "m_message" #the default
if normalize:
       regexs = [('0','\d'),('0','0+')]
       df = enhancer.normalize(regexs, to_lower=True)
       column="e_message_normalized"

#df = enhancer.words(column)
#df = enhancer.alphanumerics(column)
#df = enhancer.trigrams(column) #carefull might be slow

#Split to words in sklearn
sad =  ad.SupervisedAnomalyDetection(item_list_col="m_message")
sad.test_train_split (df, test_frac=0.95)
sad.evaluate_all_ads()


#Split to words in polars.
df = enhancer.words(column)
sad =  ad.SupervisedAnomalyDetection(item_list_col="e_words")
sad.test_train_split (df, test_frac=0.95)
sad.evaluate_all_ads()

# res = sad.dep_train_LR(df_eve_train)
# res = sad.dep_predict(df_eve_test)
# res = sad.dep_train_DT(df_eve_train)
# res = sad.dep_predict(df_eve_test)
# res = sad.dep_train_LSVM(df_eve_train, max_iter=300)
# res = sad.dep_predict(df_eve_test)
# res = sad.dep_predict(df_eve_test)
# res = sad.dep_train_RF(df_eve_train)
# res = sad.dep_predict(df_eve_test)
# res = sad.dep_train_XGB(df_eve_train)
# res = sad.dep_predict(df_eve_test)
# #Unsupervised
# res = sad.dep_train_IsolationForest(df_eve_train,filter_anos=False)
# res = sad.dep_predict(df_eve_test)
# res = sad.dep_train_LOF(df_eve_train, filter_anos=True, contamination=0.1)
# res = sad.dep_predict(df_eve_test)
# res = sad.dep_train_KMeans(df_eve_train)
# res = sad.dep_predict(df_eve_test)

# #Custom plot enabled
# res = sad.dep_train_RarityModel(df_eve_train, threshold=800)
# res = sad.dep_predict(df_eve_test, print_scores = True, custom_plot = True)
