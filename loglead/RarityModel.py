import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import time
import polars as pl
import numpy as np


class RarityModel:
    def __init__(self, threshold = 10, common_threshold = 0.01):
        self.threshold = threshold
        self.score_vector = None
        self.scores = None
        self.is_norm = None
        self.common_threshold = common_threshold
        
    
    def fit(self, X_train, labels=None):  
        def rarity_score(freq, total_ngrams):
            normalized_freq = freq / total_ngrams
            if normalized_freq > self.common_threshold:
                return 0  #common ngram, rarity score is 0     
            score = -math.log(normalized_freq) ** 3
            #currently, there shouldn't be any OOV term inputs (freq==0) 
            #but if there is, this would give a score that's twice as rare as one with 1 occurence
            if freq == 0: 
                return (-math.log(1/total_ngrams) ** 3 )*2
            return score
        
        total_ngrams = X_train.sum()
        train_counts = np.array(X_train.sum(axis=0))[0]
        self.score_vector = np.array([rarity_score(count, total_ngrams) for count in train_counts])
        
    
    def predict(self, X_test):
        X_test_csr = X_test.tocsr()
        #Getting the count of non-zero elements along axis 1 (columns) for each instance
        non_zero_counts = np.array(X_test_csr.getnnz(axis=1), dtype=np.float64)  #Convert to float64 here
        non_zero_counts[non_zero_counts == 0] = 1  #ensuring no divisions by 0
        self.scores = X_test_csr.dot(self.score_vector)
        #Ensuring self.scores is a float array
        self.scores = self.scores.astype(np.float64)
        #Dividing the scores by the count of non-zero elements
        self.scores /= non_zero_counts
        #Comparing the scores to the threshold
        self.is_ano = (self.scores > self.threshold).astype(int)
        return self.is_ano
    
    def custom_plot(self, labels, x_axis_scale=1.0):
        # Double the font size
        #mpl.rcParams.update({'font.size': mpl.rcParams['font.size']*1.5})
        
        labels_bool = np.array(labels).astype(bool)
        scores_norm = self.scores[~labels_bool]
        scores_ano = self.scores[labels_bool]
        
        plt.figure(figsize=(8, 6))  # 4:3 aspect ratio
        plt.hist(scores_norm, bins=50, color='blue', alpha=0.5, label='Normal')
        plt.hist(scores_ano, bins=50, color='red', alpha=0.5, label='Anomaly')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')

        # Adjust x-axis limit based on the parameter
        max_score = max(np.max(scores_norm), np.max(scores_ano))
        plt.xlim([0, max_score * x_axis_scale])

        plt.tight_layout()
        plt.show()
