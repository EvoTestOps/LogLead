#Sequence levels prediction
import loader as load, enricher as er, anomaly_detection as ad
import polars as pl
import math
from collections import Counter
from itertools import chain
import re
import time; 

prevtime = time.time() 


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
    df = preprocessor.reduce_dataframes(frac=1)
df_seq = preprocessor.df_sequences

print(f'Time preprocess: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()

### Normalizing before everything now 
def normalize_message(value):
    line = re.sub(r'\d', '0', value)
    line = line.lower()
    line = re.sub('0+', '0', line)
    return line

df = df.with_columns(
    pl.col("m_message").apply(normalize_message)
)

print(f'Time normalize: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()
  
#-Event enrichment----------------------------------------------
#Parsing in event level
enricher_hdfs = er.EventLogEnricher(df)
df = enricher_hdfs.length()
df = enricher_hdfs.trigrams()

print(f'Time enrich: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()

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

print(f'Time manage dfs for ad: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()

def rarity_score(ngram, train_ngrams_counter, total_ngrams, threshold = 0.05):
    ngram_freq = train_ngrams_counter.get(ngram, 0)
    # If the ngram doesn't appear in the training set, assign a high rarity score.
    if ngram_freq == 0:
        return 5
    normalized_freq = ngram_freq / total_ngrams
    if normalized_freq > threshold:
        return 0  # Common ngram, rarity score is 0
    
    return -math.log(normalized_freq)


frequent_ngrams_count=400

# Count the frequency of ngrams in the training set
total_ngrams = df_ngram_counts['count'].sum()
train_ngrams_set = set(df_ngram_counts.sort(pl.col('count')).limit(frequent_ngrams_count)['ngram'])

unmatched_list = []
num_trigrams_list = []
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
       num_trigrams_list.append(total_ngrams_event)
       max_absolute_list.append(max_unmatched_in_event)
       max_prop_list.append(max_proportion_in_event)
       total_ano_score_list.append(total_ano_score)
       max_ano_score_list.append(max_ano_score_in_event)
       
# Create new columns in df_test based on the computed values
df_test = df_test.with_columns(
    pl.lit(unmatched_list).alias("unmatched"),
    pl.lit(num_trigrams_list).alias("num_trigrams"),
    pl.lit(max_absolute_list).alias("max_absolute"),
    pl.lit(max_prop_list).alias("max_prop"),
    pl.lit(total_ano_score_list).alias("total_ano_score"),
    pl.lit(max_ano_score_list).alias("max_ano_score")
)

print(f'Time ad: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()

grouped_stats = (
    df_test.groupby("normal")
    .agg(
        count=pl.count("max_ano_score"),
        mean=pl.mean("max_ano_score"),
        median=pl.median("max_ano_score"),
        std=pl.std("max_ano_score"),
        min=pl.min("max_ano_score"),
        max=pl.max("max_ano_score")
    )
)

grouped_stats


import polars as pl

def evaluate(df, threshold=0.05, mode="max_ano_score", length_limit=200, header=True):
    # Define conditions based on the mode
    if mode == "total_avg":
        condition = (df['unmatched'] / df['num_trigrams']) > threshold
    else:
        condition = df[mode] > threshold

    # Override the condition if num_events is below the length_limit
    condition = condition | (df['num_trigrams'] < length_limit)

    # Create new columns to hold the predicted and actual anomaly status
    df = df.with_columns([
        pl.when(condition).then(1).otherwise(0).alias('is_anomaly_pred'),
        pl.when(df['normal']).then(0).otherwise(1).alias('is_anomaly_actual')
    ])

    # Define expressions to compute the confusion matrix elements
    tp_expr = (pl.col('is_anomaly_pred').cast(bool) & pl.col('is_anomaly_actual').cast(bool))
    fp_expr = (pl.col('is_anomaly_pred').cast(bool) & ~pl.col('is_anomaly_actual').cast(bool))
    tn_expr = (~pl.col('is_anomaly_pred').cast(bool) & ~pl.col('is_anomaly_actual').cast(bool))
    fn_expr = (~pl.col('is_anomaly_pred').cast(bool) & pl.col('is_anomaly_actual').cast(bool))


    # Calculate the elements of the confusion matrix
    tp = df.filter(tp_expr).shape[0]
    fp = df.filter(fp_expr).shape[0]
    tn = df.filter(tn_expr).shape[0]
    fn = df.filter(fn_expr).shape[0]

    # Calculate precision, recall, accuracy, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print the header
    if header:
        header = f"{'Mode':<15}{'Thresh.':<10}{'TP':<6}{'FP':<6}{'TN':<10}{'FN':<6}{'Precision':<10}{'Recall':<10}{'Accuracy':<10}{'F1 Score':<10}"
        print(header)
        print('-' * len(header))

    # Print results in a formatted manner
    results = f"{mode:<15}{threshold:<10.3f}{tp:<6}{fp:<6}{tn:<10}{fn:<6}{precision:<10.4f}{recall:<10.4f}{accuracy:<10.4f}{f1_score:<10.4f}"
    print(results)

# Assuming df_test is your Polars DataFrame
evaluate(df_test, threshold=0.05, mode="max_prop", length_limit=0)

print(f'Time eval: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()

first = True #For header
for thr_loop in range(470,560,5):
    evaluate(df_test, threshold=thr_loop, mode="max_ano_score", length_limit=200, header=first)
    first = False    


"""

Full HDFS
Time preprocess: 23.64 seconds
Time normalize: 131.44 seconds
Time enrich: 353.89 seconds
Time manage dfs for ad: 221.34 seconds
Time ad: 263.08 seconds
Time eval: 5.01 seconds

Note: A lot of the functions are from previous implementation and there might be faster polars options available.

Example output:

Mode           Thresh.   TP    FP    TN        FN    Precision Recall    Accuracy  F1 Score  
---------------------------------------------------------------------------------------------
max_ano_score  470.000   16838 2791110         0     0.0569    1.0000    0.0569    0.1077    
max_ano_score  475.000   16838 2791110         0     0.0569    1.0000    0.0569    0.1077    
max_ano_score  480.000   16838 2791110         0     0.0569    1.0000    0.0569    0.1077    
max_ano_score  485.000   16838 2791110         0     0.0569    1.0000    0.0569    0.1077    
max_ano_score  490.000   16523 26185217259     315   0.0594    0.9813    0.1141    0.1119    
max_ano_score  495.000   16523 26185217259     315   0.0594    0.9813    0.1141    0.1119    
max_ano_score  500.000   15518 19230186810     1320  0.0747    0.9216    0.3458    0.1381    
max_ano_score  505.000   14753 138985140126    2085  0.0960    0.8762    0.5233    0.1730    
max_ano_score  510.000   14753 138985140126    2085  0.0960    0.8762    0.5233    0.1730    
max_ano_score  515.000   14753 138985140126    2085  0.0960    0.8762    0.5233    0.1730    
max_ano_score  520.000   12720 86295 192816    4118  0.1285    0.7554    0.6945    0.2196    
max_ano_score  525.000   11784 77530 201581    5054  0.1319    0.6998    0.7210    0.2220    
max_ano_score  530.000   10803 33260 245851    6035  0.2452    0.6416    0.8672    0.3548    
max_ano_score  535.000   10650 24904 254207    6188  0.2995    0.6325    0.8949    0.4066    
max_ano_score  540.000   10584 16784 262327    6254  0.3867    0.6286    0.9222    0.4788    
max_ano_score  545.000   10413 8288  270823    6425  0.5568    0.6184    0.9503    0.5860    
max_ano_score  550.000   10346 181   278930    6492  0.9828    0.6144    0.9775    0.7561    
max_ano_score  555.000   10346 181   278930    6492  0.9828    0.6144    0.9775    0.7561    

"""

#Trigram prediction for BGL. This is a first version with a simple approach:
#1. Preprocess (incl. normalize)
#2. Split between training (normal) and test (ano + normal) sets
#3. Create a training vocabulary of most common trigrams 
#4. Calculate ano score with the rarity function for each trigram
#5. Compare scores of maximum score trigram of the event for normal and ano events
#6. Results:
#Statistics for scores_norm:
#Mean: 10.378, Median: 10.322, Standard Deviation: 1.647, Min: 6.016, Max: 18.458
#Statistics for scores_ano:
#Mean: 16.288, Median: 16.260, Standard Deviation: 1.536, Min: 6.981, Max: 18.458



prevtime =  time.time()

#PREPROCESS

preprocessor = load.BGLLoader(filename="../../../Datasets/bgl/BGL.log")

df = preprocessor.execute()
enricher = er.EventLogEnricher(df)
df = enricher.trigrams()
df = df.with_columns(
    pl.when(df['label'] == "-").then(True).otherwise(False).alias("normal")
)

print(f'Time preprocess: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()
 
def normalize_message(value):
    line = re.sub(r'\d', '0', value)
    line = line.lower()
    line = re.sub('0+', '0', line)
    return line

df = df.with_columns(
    pl.col("m_message").apply(normalize_message)
)

print(f'Time normalize: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()

#SPLIT

normal_data = df.filter(df['normal'])
anomalous_data = df.filter(~df['normal'])

df_train, df_normal_test = ad.test_train_split(normal_data, test_frac=0.5)
df_test = pl.concat([df_normal_test, anomalous_data], how="vertical")

print(f'Time split: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()

#CREATE NORMAL COUNTER AND DF

df_train = df_train.with_columns(pl.col("e_cgrams").fill_null([]))
e_cgrams_data = df_train["e_cgrams"] # if null values inside lists: [[item if item is not None else "" for item in sublist] for sublist in e_cgrams_data]
# Now flatten the list
all_trigrams = list(chain.from_iterable(e_cgrams_data))

trigram_counts = Counter(all_trigrams)
# Create a new DataFrame with the trigrams and their counts
df_ngram_counts = pl.DataFrame({
    'ngram': list(trigram_counts.keys()),
    'count': list(trigram_counts.values())
})

frequent_ngrams_count=400

# Count the frequency of ngrams in the training set
total_ngrams = df_ngram_counts['count'].sum()
train_ngrams_set = set(df_ngram_counts.sort(pl.col('count')).limit(frequent_ngrams_count)['ngram'])

print(f'Time create normal set: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()

#TEST

# Replace None sublists with empty lists
df_test = df_test.with_columns(pl.col("e_cgrams").fill_null([]))

# Replace None elements with empty strings within each sublist
e_cgrams_data = [[item if item is not None else "" for item in sublist] for sublist in e_cgrams_data]

scores = []
for event_ngrams in df_test["e_cgrams"]:
    event_ngrams_set = set(event_ngrams)
    event_ano_scores = [rarity_score(ngram, trigram_counts, total_ngrams) for ngram in event_ngrams_set]
    scores.append(event_ano_scores)
    
max_scores = [max(score_list) for score_list in scores if score_list]

#the scores are arranged so that normal ones comes first
scores_norm = max_scores[:2199751]
scores_ano = max_scores[2199751:]

print(f'Time test: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()

# ROUGH EVALUATION

import numpy as np

scores_norm_np = np.array(scores_norm)
scores_ano_np = np.array(scores_ano)

mean_norm = np.mean(scores_norm_np)
median_norm = np.median(scores_norm_np)
std_dev_norm = np.std(scores_norm_np)
min_value_norm = np.min(scores_norm_np)
max_value_norm = np.max(scores_norm_np)

mean_ano = np.mean(scores_ano_np)
median_ano = np.median(scores_ano_np)
std_dev_ano = np.std(scores_ano_np)
min_value_ano = np.min(scores_ano_np)
max_value_ano = np.max(scores_ano_np)

print(f'Statistics for scores_norm:')
print(f'Mean: {mean_norm:.3f}, Median: {median_norm:.3f}, Standard Deviation: {std_dev_norm:.3f}, Min: {min_value_norm:.3f}, Max: {max_value_norm:.3f}')

print(f'\nStatistics for scores_ano:')
print(f'Mean: {mean_ano:.3f}, Median: {median_ano:.3f}, Standard Deviation: {std_dev_ano:.3f}, Min: {min_value_ano:.3f}, Max: {max_value_ano:.3f}')

