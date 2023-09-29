#Sequence levels prediction
import loader as load, enricher as er, anomaly_detection as ad
import polars as pl
import math
from collections import Counter
from itertools import chain
import re


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
if (not b_hadoop):
    df = preprocessor.reduce_dataframes(frac=0.02)
df_seq = preprocessor.df_sequences

### Normalizing before everything now 
def normalize_message(value):
    line = re.sub(r'\d', '0', value)
    line = line.lower()
    line = re.sub('0+', '0', line)
    return line

df = df.with_columns(
    pl.col("m_message").apply(normalize_message)
)
  
#-Event enrichment----------------------------------------------
#Parsing in event level
enricher_hdfs = er.EventLogEnricher(df)
df = enricher_hdfs.length()



df = enricher_hdfs.trigrams()


# Adding trigrams to sequence level as an array (event) of arrays (trigrams)
agg_df = (
    df
    .groupby('seq_id')
    .agg(pl.col('e_cgrams'))
    .sort('seq_id')  # Optional: sorting by 'seq_id'
)

df_seq = df_seq.join(agg_df, on='seq_id', how='left')


#Split
normal_data = df_seq.filter(df_seq['normal'])
anomalous_data = df_seq.filter(~df_seq['normal'])

df_train, df_normal_test = ad.test_train_split(normal_data, test_frac=0.5)
df_test = pl.concat([df_normal_test, anomalous_data], how="vertical")


# Collect the data from the e_cgrams column to Python
e_cgrams_data = df_train['e_cgrams'].to_list()
# Flatten the list of lists into a single list of trigrams
all_trigrams = list(chain.from_iterable(chain.from_iterable(e_cgrams_data)))
# Count the occurrences of each trigram
trigram_counts = Counter(all_trigrams)
# Create a new DataFrame with the trigrams and their counts
df_ngram_counts = pl.DataFrame({
    'ngram': list(trigram_counts.keys()),
    'count': list(trigram_counts.values())
})


def rarity_score(ngram, train_ngrams_counter, total_ngrams, threshold = 0.05):
    ngram_freq = train_ngrams_counter.get(ngram, 0)
    # If the ngram doesn't appear in the training set, assign a high rarity score.
    if ngram_freq == 0:
        return 5
    normalized_freq = ngram_freq / total_ngrams
    if normalized_freq > threshold:
        return 0  # Common ngram, rarity score is 0
    
    return -math.log(normalized_freq)


frequent_ngrams_count=1000

# Count the frequency of ngrams in the training set
total_ngrams = df_ngram_counts['count'].sum()
train_ngrams_set = set(df_ngram_counts.sort(pl.col('count')).limit(frequent_ngrams_count)['ngram'])

unmatched_list = []
num_events_list = []
max_absolute_list = []
max_prop_list = []
total_ano_score_list = []
max_ano_score_list = []

# Process the test set
for index, row in enumerate(df_test.iter_rows()):
       unmatched_ngrams_total = 0
       total_ngrams_event = 0
       max_unmatched_in_event = 0
       max_proportion_in_event = 0
       total_ano_score = 0
       max_ano_score_in_event = 0
       
       for event in row[2]:
              event_ngrams_set = set(event)
              unmatched_ngrams = len([ngram for ngram in event_ngrams_set if ngram not in train_ngrams_set])
              event_ano_scores = [rarity_score(ngram, trigram_counts, total_ngrams) for ngram in event_ngrams_set]

              event_total_ano_score = sum(event_ano_scores)
              total_ano_score += event_total_ano_score

              proportion = unmatched_ngrams / len(event_ngrams_set) if len(event_ngrams_set) > 0 else 0
              max_unmatched_in_event = max(max_unmatched_in_event, unmatched_ngrams)
              max_proportion_in_event = max(max_proportion_in_event, proportion)
              max_ano_score_in_event = max(max_ano_score_in_event, event_total_ano_score)

              unmatched_ngrams_total += unmatched_ngrams
              total_ngrams_event += len(event_ngrams_set)

       unmatched_list.append(unmatched_ngrams_total)
       num_events_list.append(total_ngrams_event)
       max_absolute_list.append(max_unmatched_in_event)
       max_prop_list.append(max_proportion_in_event)
       total_ano_score_list.append(total_ano_score)
       max_ano_score_list.append(max_ano_score_in_event)
       
# Create new columns in df_test based on the computed values
df_test = df_test.with_columns(
    pl.lit(unmatched_list).alias("unmatched"),
    pl.lit(num_events_list).alias("num_events"),
    pl.lit(max_absolute_list).alias("max_absolute"),
    pl.lit(max_prop_list).alias("max_prop"),
    pl.lit(total_ano_score_list).alias("total_ano_score"),
    pl.lit(max_ano_score_list).alias("max_ano_score")
)

import pandas as pd
# convert to pandas
pddf = pd.DataFrame(df_test.select(['normal', 'max_prop']))

# perform the groupby and agg operations on the Pandas DataFrame
grouped_stats = pddf.groupby(0).agg(
    count=pd.NamedAgg(column=1, aggfunc='count'),
    mean=pd.NamedAgg(column=1, aggfunc='mean'),
    median=pd.NamedAgg(column=1, aggfunc='median'),
    std=pd.NamedAgg(column=1, aggfunc='std'),
    min=pd.NamedAgg(column=1, aggfunc='min'),
    max=pd.NamedAgg(column=1, aggfunc='max')
)
print(grouped_stats)
