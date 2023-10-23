#Sequence levels prediction
import loglead.loader as load, loglead.enricher as er, anomaly_detection as ad
import polars as pl
import math
from collections import Counter
from itertools import chain
import re
import time; 

prevtime = time.time() 


def rarity_score(ngram, train_ngrams_counter, total_ngrams, threshold = 0.05):
    ngram_freq = train_ngrams_counter.get(ngram, 0)
    # If the ngram doesn't appear in the training set, assign a high rarity score.
    if ngram_freq == 0:
        return 5
    normalized_freq = ngram_freq / total_ngrams
    if normalized_freq > threshold:
        return 0  # Common ngram, rarity score is 0
    
    return -math.log(normalized_freq)


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
#DONT USE APPLY, use replace_all from polars
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

#First version time
#Time preprocess: 132.10 seconds
#Time normalize: 23.15 seconds
#Time split: 16.43 seconds
#Time create normal set: 1773.85 seconds
#Time test: 2077.55 seconds

#2nd version for create normal set is now polars only and a 500x faster, few seconds
#Tried 4 different versions to handle the test, but they are all over 30min

#Event levels prediction
import loglead.loader as load, loglead.enricher as er, anomaly_detection as ad
import polars as pl
import math
from collections import Counter
from itertools import chain
import re
import time
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



prevtime =  time.time()

#PREPROCESS

preprocessor = load.BGLLoader(filename="../../../Datasets/bgl/BGL.log")

df = preprocessor.execute()

#df = df.sample(fraction=0.5)

print(f'Time preprocess: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()

enricher = er.EventLogEnricher(df)

regexs = [('0','\d'),('0','0+')]
df = enricher.normalize(regexs, to_lower=True)

print(f'Time normalize: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()

df = enricher.trigrams(column="e_message_normalized")
df = df.with_columns(
    pl.when(df['label'] == "-").then(True).otherwise(False).alias("normal")
)

print(f'Time create ngrams: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()



#SPLIT

normal_data = df.filter(df['normal'])
anomalous_data = df.filter(~df['normal'])

df_train, df_normal_test = ad.test_train_split(normal_data, test_frac=0.5)
df_test = pl.concat([df_normal_test, anomalous_data], how="vertical")
# Replace None sublists with empty lists
df_test = df_test.with_columns(pl.col("e_cgrams").fill_null([]))

print(f'Time split: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()

#CREATE NORMAL COUNTER AND DF

df_train = df_train.with_columns(pl.col("e_cgrams").fill_null([]))
#e_cgrams_data = df_train["e_cgrams"] # if null values inside lists: [[item if item is not None else "" for item in sublist] for sublist in e_cgrams_data]
# Now flatten the list
#all_trigrams = list(chain.from_iterable(e_cgrams_data))

flattened_train = df_train.select(pl.col("e_cgrams").explode())
# Assuming df_train is your DataFrame
df_ngram_counts = (
    flattened_train.groupby('e_cgrams')
    .agg(pl.col('e_cgrams').count().alias('count'))
    .sort('count', descending=True)
)


#trigram_counts = Counter(flattened_df)
# Create a new DataFrame with the trigrams and their counts
#df_ngram_counts = pl.DataFrame({
#    'ngram': list(trigram_counts.keys()),
#    'count': list(trigram_counts.values())
#})

#frequent_ngrams_count=9999

# Count the frequency of ngrams in the training set
total_ngrams = df_ngram_counts['count'].sum()
train_ngrams_set = set(df_ngram_counts.sort(pl.col('count'))['e_cgrams'])

print(f'Time create normal set: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()



def rarity_score(freq, total_ngrams, common_threshold = 0.01):
            normalized_freq = freq / total_ngrams
            if normalized_freq > common_threshold:
                return 0  # Common ngram, rarity score is 0     
            score = -math.log(normalized_freq) ** 3
            if freq == 0:
                return (-math.log(1/total_ngrams) ** 3 )*2
            return score
#TEST


### Precalc score dot count vector
#This is because we precalculate only based on the training data, not sure what happens to OOV later. 

#Precalculated score vector
train_trigrams = df_ngram_counts['e_cgrams'].to_list()
counts = df_ngram_counts['count'].to_list()
# Create a dictionary from the trigrams and counts, applying the rarity_score function to each count
score_dict = dict(zip(train_trigrams, [rarity_score(count, total_ngrams) for count in counts]))

# Get the trigrams
test_trigrams = df_test['e_cgrams'].to_list()

print(f'Time to list: {time.time() - prevtime:.2f} seconds')
prevtime = time.time()

# Disable the analyzer
vectorizer = CountVectorizer(analyzer=lambda x: x)

# Now fit the vectorizer and transform the data
X = vectorizer.fit_transform(test_trigrams)

feature_names = vectorizer.get_feature_names()

# Create the score_vector, here 32 is hardcoded OOV score
score_vector = np.array([score_dict.get(feature, 32) for feature in feature_names])

X_csr = X.tocsr()
score_matrix = X_csr.dot(score_vector)
df_test = df_test.with_columns(pl.Series(name="score_matrix", values=score_matrix))
#Divide by length
df_test = df_test.with_columns(df_test['e_cgrams'].apply(lambda x: len(x)).alias('e_cgrams_length'))
df_test = df_test.with_columns((df_test['score_matrix'] / df_test['e_cgrams_length']).alias('scores'))


scores_norm = df_test.filter(df_test['label'] == '-')['scores'].to_list()
scores_ano = df_test.filter(df_test['label'] != '-')['scores'].to_list()

#Add scores to the exploded test_df
print(f'Time test vectors: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()


#The polars way

df_test = df_test.explode("e_cgrams") #53 seconds
#This needs to be edited. The unique() was initially for just the trigrams, but now it's pretty pointless.
#test_ngrams_set = df_test.unique()
joined_df = df_test.join(df_ngram_counts, on='e_cgrams', how='left')
joined_df = joined_df.with_columns(pl.col('count').fill_null(0)).sort('count', descending=True)
df_test = joined_df.with_columns(
    pl.col("count").map_elements(lambda value: rarity_score(value, total_ngrams), return_dtype=pl.Float64).alias("rarity_score")
)
#Add scores to the exploded test_df
print(f'Time calc scores: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()


#Aggregate the list to message level again

#We can calculate the max directly if it's the only thing we want
#max_rarity_df = (
#    df_test.group_by('m_message')
#    .agg(
#        max_rarity_score=pl.col('rarity_score').max(),
#        label=pl.col('label').first(),
#    )
#)

def collect_list(s: pl.Series) -> pl.Series:
    return s.to_list()

#I haven't verified that aggregated matches the original exactly
aggregated_df = (
    df_test.group_by('m_message', 'timestamp')
    .agg(
        e_cgrams_list=pl.col('e_cgrams').map_elements(collect_list),
        rarity_score_list=pl.col('rarity_score').map_elements(collect_list),
        label=pl.col('label').first()
    )
)

scores_norm = aggregated_df.filter(aggregated_df['label'] == '-')['rarity_score_list'].to_list()
scores_ano = aggregated_df.filter(aggregated_df['label'] != '-')['rarity_score_list'].to_list()


print(f'Time aggregate and filter: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()


def evaluate(scores_norm, scores_ano, threshold = 15):

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    def auc_roc_analysis(labels, preds, plot=True):
        # Compute the ROC curve
        fpr, tpr, thresholds = roc_curve(labels, preds)
        # Compute the AUC from the points of the ROC curve
        roc_auc = auc(fpr, tpr)

        if plot:
            # Plot the ROC curve
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            plt.show()
        
        return roc_auc

    scores_norm_np = np.array(scores_norm)
    scores_norm_np = scores_norm_np[~np.isnan(scores_norm_np)]  # This will remove nan values

    scores_ano_np = np.array(scores_ano)
    scores_ano_np = scores_ano_np[~np.isnan(scores_ano_np)]  
    
    # Concatenate the scores and create a corresponding labels array
    all_scores = np.concatenate((scores_norm_np, scores_ano_np))
    labels = np.concatenate((np.zeros(len(scores_norm_np)), np.ones(len(scores_ano_np))))  # 0 for normal, 1 for anomalous

    # Call auc_roc_analysis with the labels and scores
    print("auc roc: ", auc_roc_analysis(labels, all_scores))

    tp = sum(score > threshold for score in scores_ano_np)
    fp = sum(score > threshold for score in scores_norm_np)
    fn = sum(score <= threshold for score in scores_ano_np)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Prec {precision:.3f}, recall {recall:.3f}, f1 {f1:.3f}")

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

    # figure
    plt.figure(figsize=(10, 5))
    plt.hist(scores_norm, bins=50, color='blue', alpha=0.5, label='scores_norm')
    plt.hist(scores_ano, bins=50, color='red', alpha=0.5, label='scores_ano')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of scores_norm and scores_ano')
    plt.legend(loc='upper right')  # Add legend to identify which color corresponds to which distribution
    plt.tight_layout()
    plt.show()

#max_scores = [max(score_list) for score_list in scores if score_list] #This is around 5 seconds
#avg_scores = [sum(score_list)/len(score_list) for score_list in scores if score_list] 
#weighted_multi = [sum(score_list)*max(score_list) / len(score_list) for score_list in scores if score_list]
#weighted_square = [sum(score_list)**2 / len(score_list)**2 for score_list in scores if score_list]

#evaluate(max_scores,labels,30)
#evaluate(avg_scores,labels,11)
#evaluate(weighted_multi,labels,200)

scores_norm = [max(score_list) for score_list in scores_norm if score_list] #This is around 5 seconds
scores_ano = [max(score_list) for score_list in scores_ano if score_list]
evaluate(scores_norm,scores_ano, 10)



print(f'Time evaluate: {time.time() - prevtime:.2f} seconds')
prevtime =  time.time()



###  COUNT VECTORIZER supervised

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Assume df is your DataFrame
df = df.with_columns(pl.col("e_cgrams").fill_null([]))

prevtime = time.time()

# Convert labels to binary: 0 for normal, 1 for anomalous
labels = [(0 if label == '-' else 1) for label in df['label'].to_list()]

# Get the trigrams
trigrams = df['e_cgrams'].to_list()

print(f'Time to list: {time.time() - prevtime:.2f} seconds')
prevtime = time.time()

# Disable the analyzer
vectorizer = CountVectorizer(analyzer=lambda x: x)

# Now fit the vectorizer and transform the data
X = vectorizer.fit_transform(trigrams)

print(f'Time vectorize: {time.time() - prevtime:.2f} seconds')
prevtime = time.time()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Create and train the Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)
print(f'Time train: {time.time() - prevtime:.2f} seconds')
prevtime = time.time()

y_pred = clf.predict(X_test)
print(f'Time predict: {time.time() - prevtime:.2f} seconds')
prevtime = time.time()

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Precision, Recall, F1 Score with 'micro' averaging:
precision = precision_score(y_test, y_pred, average='micro')
print(f'Micro-averaged Precision: {precision * 100:.2f}%')

recall = recall_score(y_test, y_pred, average='micro')
print(f'Micro-averaged Recall: {recall * 100:.2f}%')

f1 = f1_score(y_test, y_pred, average='micro')
print(f'Micro-averaged F1 Score: {f1 * 100:.2f}%')

print(f'Time to eval: {time.time() - prevtime:.2f} seconds')



"""
Test versions:

#Version 1, mostly python functions

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



Test version 2:
#Trying alternative approach for testing by precalcuting counts into polars df and fetching them for each ngram. 
#The row-by-row had to be done with python code though, so it was terribly slow. Over 2 hours for full BGL.

scores = []
# Iterate through each list of trigrams in df_test["e_cgrams"]
for event_ngrams in df_test["e_cgrams"]:
    # Convert the list of trigrams to a set to remove duplicates
    event_ngrams_set = set(event_ngrams)
    
    # Convert the set to a list so we can index into it
    event_ngrams_list = list(event_ngrams_set)
    
    # Extract the counts for each trigram from joined_df
    event_counts_df = joined_df.filter(joined_df['e_cgrams'].is_in(event_ngrams_set))
    event_counts = event_counts_df['count'].to_list()
    
    # If event_counts is empty, skip to the next iteration
    if not event_counts:
        scores.append(None)
        continue
    
    # Find the trigram with the lowest count
    min_count = min(event_counts)
    
    # Calculate the rarity score for the trigram with the lowest count
    score = rarity_score(min_count, total_ngrams)
    
    # Append the score to the scores list
    scores.append(score)

# Convert the scores list to a Polars Series
scores_series = pl.Series('scores', scores)

# If needed, add the scores_series to your DataFrame
df_test_with_scores = df_test.hstack(scores_series)


Test version 3:
#Get individual counts with pandas filter, loop inside loop. Even slower, 3 hours. 

test_ngrams_set = df_test.select(pl.col("e_cgrams").explode().unique())
joined_df = test_ngrams_set.join(df_ngram_counts, on='e_cgrams', how='left')
joined_df = joined_df.with_columns(pl.col('count').fill_null(0)).sort('count', descending=True)

scores = []
for event_ngrams in df_test["e_cgrams"]:
    event_ano_scores = []
    event_ngrams_set = set(event_ngrams)
    for trigram in event_ngrams_set:
        count = joined_df.filter(joined_df['e_cgrams']==trigram)['count'][0]
        event_ano_scores.append(rarity_score(count, total_ngrams))
    scores.append(event_ano_scores)
    
max_scores = [max(score_list) for score_list in scores if score_list]


#version 4:
count_dict = dict(zip(joined_df['e_cgrams'].to_list(), joined_df['count'].to_list()))

scores = []
for event_ngrams in df_test["e_cgrams"]:
    event_ano_scores = [rarity_score(count_dict.get(trigram, 0), total_ngrams) for trigram in set(event_ngrams)]
    scores.append(event_ano_scores)

    """