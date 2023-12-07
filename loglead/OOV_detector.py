

import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import polars as pl
import numpy as np

class OOV_detector:
    def __init__(self, len_col, test_df, threshold = 1):
        self.len_col = len_col
        self.test_df = test_df
        self.scores = 0
        self.threshold = threshold
            
    def fit(self, X_train = None, labels = None):
        #The "training set" to compare against comes inherently from test data's sparse matrix
        return
    
    def predict(self, X_test):
        #Give array of 0s if the needed length column is lacking in the df        
        if self.len_col not in self.test_df.columns:
            print("Column not found for OOVD")
            return np.zeros(self.test_df.select(pl.count()).item())
        else:
            msglen = self.test_df[self.len_col]
            test_word_count_np = np.array(X_test.tocsr().sum(axis=1)).squeeze()
            test_word_count_series = pl.Series(test_word_count_np)
            self.scores = np.array(msglen - test_word_count_series)
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

