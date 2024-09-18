import polars as pl
import numpy as np


__all__ = ['OOV_detector']


class OOV_detector:
    def __init__(self, len_col, item_list_col, test_df, threshold=1):
        self.len_col = len_col
        self.test_df = test_df
        self.item_list_col = item_list_col
        self.scores = 0
        self.threshold = threshold
            
    def fit(self, X_train=None, labels=None):
        # The "training set" to compare against comes inherently from test data's sparse matrix
        return
    

    #To enable pickling
    def identity_function(self,x):
        return x
    
    def predict(self, X_test):
        if self.len_col not in self.test_df.columns:
            # Length column not found, reconstructing and counting from the vectorizer.
            from sklearn.feature_extraction.text import CountVectorizer
            column_data = self.test_df.select(pl.col(self.item_list_col))             
            events = column_data.to_series().to_list()
            if column_data.dtypes[0]  == pl.datatypes.Utf8: #We get strs -> Use SKlearn Tokenizer
                vectorizer = CountVectorizer() 
            elif column_data.dtypes[0]  == pl.datatypes.List(pl.datatypes.Utf8): #We get list of str, e.g. words -> Do not use Skelearn Tokinizer 
                vectorizer = CountVectorizer(analyzer=self.identity_function)
            X = vectorizer.fit_transform(events)
            msglen = np.array(X.tocsr().sum(axis=1)).squeeze()
        else:
            msglen = self.test_df[self.len_col]
        test_word_count_np = np.array(X_test.tocsr().sum(axis=1)).squeeze()
        test_word_count_series = pl.Series(test_word_count_np)
        self.scores = np.array(msglen - test_word_count_series)
        self.is_ano = (self.scores > self.threshold).astype(int)
        return self.is_ano
    
    def custom_plot(self, labels, x_axis_scale=1.0):
        # Double the font size
        # mpl.rcParams.update({'font.size': mpl.rcParams['font.size']*1.5})
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise ImportError("Error importing matplotlib") from e

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