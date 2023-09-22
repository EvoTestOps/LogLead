#
#Separate demo files
import loader as load, enricher as er, anomaly_detection as ad

# Loading Hadoop Logs--------------------------------------------------------------------
hadoop_processor = load.HadoopLoader(filename="../../../Datasets/hadoop/",
                                            filename_pattern  ="*.log",
                                     labels_file_name="../../../Datasets/hadoop/abnormal_label_accurate.txt")
df_hadoop = hadoop_processor.execute()

# Loading Thunderbird Logs----------------------------------------------------------------------
thunderbird_processor = load.ThunderbirdLoader(filename="../../../Datasets/thunderbird/Thunderbird.log") #Might take 2-3 minutes in HPC cloud. In desktop out of memory
thunderbird_processor = load.ThunderbirdLoader(filename="../../../Datasets/thunderbird/Thunderbird_2k.log") #Only 2k lines
df_thu = thunderbird_processor.execute()

# Loading Profilence Logs-------------------------------------------------------------------------------
pro_processor = load.ProLoader(filename="../../../Datasets/profilence/*.txt")
df_pro = pro_processor.execute()

# Loading HDFS Logs----------------------------------------------------------------
hdfs_processor = load.HDFSLoader(filename="../../../Datasets/hdfs/HDFS.log", 
                                     labels_file_name="../../../Datasets/hdfs/anomaly_label.csv")
df_hdfs = hdfs_processor.execute()


#Enrichment with thunderbird-------------------------------------------------
enricher_thu =  er.EventLogEnricher(df_thu)
df_thu = enricher_thu.parse_drain()
df_thu = enricher_thu.words()
enricher_thu.tm.drain.print_tree()

#-Event enrichment HDFS----------------------------------------------
enricher_hdfs = er.EventLogEnricher(df_hdfs[0:20000]) #small hdfs for faster running
df_hdfs_s = enricher_hdfs.words()
df_hdfs_s = enricher_hdfs.alphanumerics()
df_hdfs_s = enricher_hdfs.trigrams()
df_hdfs_s = enricher_hdfs.parse_drain()

#Add anomaly scores for each line.
event_anomaly_detection = ad.EventAnomalyDetection(df_hdfs_s)
df_hdfs_s = event_anomaly_detection.compute_ano_score("e_words", 100)
df_hdfs_s = event_anomaly_detection.compute_ano_score("e_alphanumerics", 100)
df_hdfs_s = event_anomaly_detection.compute_ano_score("e_cgrams", 100)