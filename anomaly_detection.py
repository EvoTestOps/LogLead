import polars as pl
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



class EventAnomalyDetection:
    def __init__(self, df):
        self.df = df

    def compute_ano_score(self, col_name, model_size):
        """
        :param col_name: A string that should be "e_words", "e_alphanumerics", or "e_cgrams"
        """
        exported = self.df.select(pl.col(col_name).reshape([-1])).to_series().to_list()
        counts = Counter(exported)  # 13s all hdfs
        top_tokens = [word for word, count in counts.most_common(model_size)]
        top_tokens = pl.Series(top_tokens, dtype=pl.Utf8)
        top_tokens = top_tokens.reshape([1, len(top_tokens)])

        self.df = self.df.with_columns(
            not_in_top_t=pl.col(col_name).list.set_difference(top_tokens),
            token_len=pl.col(col_name).list.lengths(),
        )
        self.df = self.df.with_columns(
            not_in_len=pl.col("not_in_top_t").list.lengths(),
        )
        
        final_col_name = f"as_{model_size}_{col_name}"
        self.df = self.df.with_columns(
            (pl.col("not_in_len") / pl.col("token_len")).alias(final_col_name),
        )
        
        #self.df = self.df.with_columns(
        #    (pl.col("not_in_len") / pl.col("token_len")).alias("as_" + col_name),
        #)
        return self.df


def test_train_split(df, test_frac):
    # Shuffle the DataFrame
    df = df.sample(fraction = 1.0, shuffle=True)
    # Split ratio
    test_size = int(test_frac * df.shape[0])

    # Split the DataFrame using head and tail
    test_df = df.head(test_size)
    train_df = df.tail(-test_size)
    return train_df, test_df

class SeqAnomalyDetection:
    def __init__(self, df_seq):
        self.df_train = df_seq
        #Preparate training data
        self.events, self.labels = self._prepare_data(self.df_train)

    def _prepare_data(self, df):
        events = df.select(pl.col("event_list")).to_series().to_list()
        labels = df.select(pl.col("normal")).to_series().to_list()
        # Convert lists of events to space-separated strings
        events = [' '.join(e) for e in events]
        return events, labels
        
    def train_LR (self):
        self.lr_vectorizer = CountVectorizer()
        events_vec =  self.lr_vectorizer.fit_transform(self.events)
        self.lr = LogisticRegression(max_iter=1000)
        # Train the model
        self.lr.fit(events_vec, self.labels)
        
    def predict_LR (self, df_seq, print_scores = True):
        events, labels = self._prepare_data(df_seq)
        events_vec =  self.lr_vectorizer.transform(events)
        predictions = self.lr.predict(events_vec)
        # Attach predictions to the df_seq Polars DataFrame
        df_seq = df_seq.with_columns(pl.Series(name="pred_normal", values=predictions.tolist()))
        if (print_scores):
            self._print_evaluation_scores(labels, predictions)
        return df_seq
    
    def _print_evaluation_scores(self, y_test, y_pred):
        # Evaluate the model's performance
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

        # Compute the confusion matrix--------------------------------------------------
        from sklearn.metrics import confusion_matrix
        #import matplotlib.pyplot as plt
        #import seaborn as sns
        cm = confusion_matrix(y_test, y_pred)
        # Print the confusion matrix
        print("Confusion Matrix:")
        print(cm)
        #F1--------------------------------------------------------
        from sklearn.metrics import f1_score
        # Compute the F1 score
        f1 = f1_score(y_test, y_pred)
        # Print the F1 score
        print(f"F1 Score: {f1:.2f}")
            