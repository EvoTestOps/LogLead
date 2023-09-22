#Separate demo files
import loader as load, enricher as er, anomaly_detection as ad
import polars as pl

#Which one to run. Only one true. 
b_hadoop = False
b_hdfs = True
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
df_seq = preprocessor.df_sequences

#Eventmax length--------------------------------
event_enricher = er.EventLogEnricher(df)
df = event_enricher.length()
seq_enricher = er.SequenceEnricher(df = df, df_sequences = df_seq)
seq_enricher.enrich_event_length()
seq_enricher.enrich_start_time()
seq_enricher.enrich_end_time()
seq_enricher.enrich_sequence_length()
seq_enricher.enrich_sequence_duration()


df_seq = None
if (b_hadoop):
       df_seq = seq_enricher.df_sequences.filter(pl.col("app_name")=="WordCount") #"PageRank" to change app

df_seq = seq_enricher.df_sequences.select(
              "normal",
              "event_max_len",
              "event_min_len",
              "events_over_1_line",
              "seq_len",
              "seq_time"
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
#                        (pl.col("event_max_len")>normal_max["event_max_len"][0]))

#Hadoop PageRank Max 16/21TP Max 0-FP
#Hadoop WordCount Max 19/22TP Max 0-FP
#HDFS Max-6245/16838 TP | Max-0FP
#HDFS 1%-9424 TP | 1%-7327 FP 
#Pro Max 1/2TP
df_seq.filter((pl.col("normal")==False) &
                        (
                        (pl.col("event_max_len")>normal_max["event_max_len"][0]) |
                        (pl.col("event_min_len")<normal_min["event_min_len"][0]) |
                        (pl.col("events_over_1_line")>normal_max["events_over_1_line"][0])|
                        (pl.col("seq_len")>normal_max["seq_len"][0]) |
                        (pl.col("seq_len")<normal_min["seq_len"][0]) | 
                        (pl.col("seq_time")>normal_max["seq_time"][0]) |
                        (pl.col("seq_time")<normal_min["seq_time"][0])
                        )).shape[0]
# Getting the initial count when only "normal" is False


#Analysis which rules impact and how much
conditions = {
    "event_max_len > normal_max": pl.col("event_max_len") > normal_max["event_max_len"][0],
    "event_min_len < normal_min": pl.col("event_min_len") < normal_min["event_min_len"][0],
    "events_over_1_line > normal_max": pl.col("events_over_1_line") > normal_max["events_over_1_line"][0],
    "seq_len > normal_max": pl.col("seq_len") > normal_max["seq_len"][0],
    "seq_len < normal_min": pl.col("seq_len") < normal_min["seq_len"][0],
    "seq_time > normal_max": pl.col("seq_time") > normal_max["seq_time"][0],
    "seq_time < normal_min": pl.col("seq_time") < normal_min["seq_time"][0]
}

filtered_counts = {}
for condition_name, condition in conditions.items():
    count = df_seq.filter((pl.col("normal") == False) & condition).shape[0]
    filtered_counts[condition_name] = count

for condition_name, count in filtered_counts.items():
    print(f"{condition_name}: {count} rows")



#Exploration of data. -------------------------------------------------------------
df_seq = seq_enricher.df_sequences

if (b_hadoop):
       df_seq.filter((pl.col("normal")==False) & 
                                 (pl.col("app_name")=="PageRank")).mean().sort("app_name")
       
df_seq.group_by("normal").mean().select(
       "seq_len", "seq_time","events_over_1_line", "event_max_len", "event_min_len", 
       "event_avg_len", "event_med_len", "normal").sort("normal")

df_seq.group_by("normal").quantile(0.99).select(
       "seq_len", "seq_time","events_over_1_line", "event_max_len", "event_min_len", 
       "event_avg_len", "event_med_len", "normal").sort("normal")

df_seq.group_by("normal").quantile(0.01).select(
       "seq_len", "seq_time","events_over_1_line", "event_max_len", "event_min_len", 
       "event_avg_len", "event_med_len", "normal").sort("normal")

#More exploration SEQ lenths and time---------------------------------------------

#Observe! How do normal compare to anomalies in terms of length and tine
df_seq.group_by("normal").mean().select("seq_len", "seq_time", "normal")
df_seq.group_by("normal").median().select("seq_len", "seq_time", "normal")

if (b_hadoop):
       df_seq.group_by("app_name").mean().select("seq_len", "seq_time", "normal", "app_name")
       df_seq.group_by("app_name").median().select("seq_len", "seq_time", "normal", "app_name")
       df_seq.group_by("Label").mean().select("seq_len", "seq_time", "normal", "Label")
       df_seq.group_by("Label").median().select("seq_len", "seq_time", "normal", "Label")



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