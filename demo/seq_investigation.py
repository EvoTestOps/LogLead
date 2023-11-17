#Separate demo files
import loglead.loader as load, loglead.enhancer as er, anomaly_detection as ad
import polars as pl

#Which one to run. Only one true. 
b_hadoop = True
b_hdfs = False
b_profilence = False


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
df_seq = preprocessor.df_seq

#Eventmax length--------------------------------
event_enhancer = er.EventLogEnhancer(df)
df = event_enhancer.length()
seq_enhancer = er.SequenceEnhancer(df = df, df_seq = df_seq)
seq_enhancer.eve_len()
seq_enhancer.start_time()
seq_enhancer.end_time()
seq_enhancer.seq_len()
seq_enhancer.duration()


df_seq = None
if (b_hadoop):
       df_seq = seq_enhancer.df_seq.filter(pl.col("app_name")=="WordCount") #"PageRank" to change app

df_seq = seq_enhancer.df_seq.select(
              "normal",
              "eve_len_max",
              "eve_len_min",
              "eve_len_over1",
              "seq_len",
              "duration"
       )
       
#Take max values of normal runs as boundaries
normal_max = df_seq.filter((pl.col("normal")==True)).max()
normal_min = df_seq.filter((pl.col("normal")==True)).min()

#normal_max = df_seq.filter((pl.col("normal")==True)).quantile(0.995)
#normal_min = df_seq.filter((pl.col("normal")==True)).quantile(0.005)

#Ano count sequences
#Hadoop PageRank 21 
#Hadoop WordCount 22
#HDFS 16838
#Pro 2
df_seq.filter((pl.col("normal")==False)).shape[0]

#Normal
#Hadoop PageRank 8 
#Hadoop Pagerank 3
#HDFS 558223
#Pro 226
df_seq.filter((pl.col("normal")==True)).shape[0]




#df_seq_page_rank.filter((pl.col("normal")==False) &
#                        (pl.col("eve_len_max")>normal_max["eve_len_max"][0]))

#Hadoop PageRank Max 16/21TP Max 0-FP
#Hadoop WordCount Max 19/22TP Max 0-FP
#Hadoop All Max 33/43 Max 0_
#HDFS Max-6245/16838 TP | Max-0FP
#HDFS 1%-9424 TP | 1%-7327 FP 
#Pro Max 1/2TP
df_seq.filter((pl.col("normal")==False) &
                        (
                        (pl.col("eve_len_max")>normal_max["eve_len_max"][0]) |
                        (pl.col("eve_len_min")<normal_min["eve_len_min"][0]) |
                        (pl.col("eve_len_over1")>normal_max["eve_len_over1"][0])|
                        (pl.col("seq_len")>normal_max["seq_len"][0]) |
                        (pl.col("seq_len")<normal_min["seq_len"][0]) | 
                        (pl.col("duration")>normal_max["duration"][0]) |
                        (pl.col("duration")<normal_min["duration"][0])
                        )).shape[0]
# Getting the initial count when only "normal" is False


#Analysis which rules impact and how much
conditions = {
    "eve_len_max > normal_max": pl.col("eve_len_max") > normal_max["eve_len_max"][0],
    "eve_len_min < normal_min": pl.col("eve_len_min") < normal_min["eve_len_min"][0],
    "eve_len_over1 > normal_max": pl.col("eve_len_over1") > normal_max["eve_len_over1"][0],
    "seq_len > normal_max": pl.col("seq_len") > normal_max["seq_len"][0],
    "seq_len < normal_min": pl.col("seq_len") < normal_min["seq_len"][0],
    "duration > normal_max": pl.col("duration") > normal_max["duration"][0],
    "duration < normal_min": pl.col("duration") < normal_min["duration"][0]
}

filtered_counts = {}
for condition_name, condition in conditions.items():
    count = df_seq.filter((pl.col("normal") == False) & condition).shape[0]
    filtered_counts[condition_name] = count

for condition_name, count in filtered_counts.items():
    print(f"{condition_name}: {count} rows")



#Exploration of data. -------------------------------------------------------------
df_seq = seq_enhancer.df_seq

if (b_hadoop):
       df_seq.filter((pl.col("normal")==False) & 
                                 (pl.col("app_name")=="PageRank")).mean().sort("app_name")
       
df_seq.group_by("normal").mean().select(
       "seq_len", "duration","eve_len_over1", "eve_len_max", "eve_len_min", 
       "eve_len_avg", "eve_len_med", "normal").sort("normal")

df_seq.group_by("normal").quantile(0.99).select(
       "seq_len", "duration","eve_len_over1", "eve_len_max", "eve_len_min", 
       "eve_len_avg", "eve_len_med", "normal").sort("normal")

df_seq.group_by("normal").quantile(0.01).select(
       "seq_len", "duration","eve_len_over1", "eve_len_max", "eve_len_min", 
       "eve_len_avg", "eve_len_med", "normal").sort("normal")

#More exploration SEQ lenths and time---------------------------------------------

#Observe! How do normal compare to anomalies in terms of length and tine
df_seq.group_by("normal").mean().select("seq_len", "duration", "normal")
df_seq.group_by("normal").median().select("seq_len", "duration", "normal")

if (b_hadoop):
       df_seq.group_by("app_name").mean().select("seq_len", "duration", "normal", "app_name")
       df_seq.group_by("app_name").median().select("seq_len", "duration", "normal", "app_name")
       df_seq.group_by("Label").mean().select("seq_len", "duration", "normal", "Label")
       df_seq.group_by("Label").median().select("seq_len", "duration", "normal", "Label")



#More exploration - Compare normal and ano side by side
df_normal = df_seq.filter(pl.col("normal"))
df_ano = df_seq.filter(pl.col("normal") == False)
#HDFS execute this
if (b_hdfs):
       powers_of_two = [2**i for i in range(10) if 2**i <= 512]
if(b_hadoop):
       powers_of_two = [2**i for i in range(9, 16)]

ano_hist = df_ano.select("seq_len").to_series().hist(bins = powers_of_two)
normal_hist = df_normal.select("seq_len").to_series().hist(bins = powers_of_two)
normal_hist = normal_hist.with_columns(percent = 100* pl.col("seq_len_count") / pl.col("seq_len_count").sum())
ano_hist = ano_hist.with_columns(percent = 100* pl.col("seq_len_count") / pl.col("seq_len_count").sum())
#Observe differences. 
#Normal on the right side anomalies on the left. 
normal_hist.join(ano_hist, on='break_point')