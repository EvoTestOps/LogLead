import polars as pl
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack


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
    def __init__(self, event_col=None, numeric_cols=None):
        self.event_col = event_col
        self.numeric_cols = numeric_cols if numeric_cols else []
        #self.events, self.labels, self.additional_features = self._prepare_data(self.df_train)

    def _prepare_data(self, df):
        # Extract events
        events = []
        if self.event_col:
            events = df.select(pl.col(self.event_col)).to_series().to_list()
            events = [' '.join(e) for e in events]

        labels = df.select(pl.col("normal")).to_series().to_list()
        
        # Extract additional predictors
        additional_features = []
        if self.numeric_cols:
            additional_features = df.select(self.numeric_cols).to_pandas().values
        
        return events, labels, additional_features
    
    def sklearn_vectorization (self, train, df_seq): 
        events, labels, additional_features = self._prepare_data(df_seq)
        X = None
        if self.event_col:
            if train:
                self.vectorizer = CountVectorizer() 
                events_vec = self.vectorizer.fit_transform(events)
            else: 
                events_vec = self.vectorizer.transform(events)
            X = events_vec
        
        if self.numeric_cols:
            X = hstack([X, additional_features]) if X is not None else additional_features
        
        return X, labels
    
    def train_LR(self, df_seq):
        X_train, labels = self.sklearn_vectorization(train=True, df_seq=df_seq)
        self.lr = LogisticRegression(max_iter=1000)
        self.lr.fit(X_train, labels)
        
    def predict_LR(self, df_seq, print_scores=True):
        X_test, labels = self.sklearn_vectorization(train=False, df_seq=df_seq)
        predictions = self.lr.predict(X_test)
        df_seq = df_seq.with_columns(pl.Series(name="pred_normal", values=predictions.tolist()))
        if print_scores:
            self._print_evaluation_scores(labels, predictions, self.lr)
        return df_seq
    
    def train_DT(self, df_seq):
        X_train, labels = self.sklearn_vectorization(train=True, df_seq=df_seq)
        self.dt = tree.DecisionTreeClassifier()
        self.dt.fit(X_train, labels)

    def predict_DT(self, df_seq, print_scores=True):
        X_test, labels = self.sklearn_vectorization(train=False, df_seq=df_seq)
        predictions = self.dt.predict(X_test)
        df_seq = df_seq.with_columns(pl.Series(name="pred_normal", values=predictions.tolist()))
        if print_scores:
            self._print_evaluation_scores(labels, predictions, self.dt)
        return df_seq

    def _print_evaluation_scores(self, y_test, y_pred, model):
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

        # Print feature importance
        if hasattr(self, 'vectorizer') and self.vectorizer:
            event_features = self.vectorizer.get_feature_names_out()
            event_features = list(event_features)
        else:
            event_features = []

        all_features = event_features + self.numeric_cols
        if isinstance(model, LogisticRegression):
            feature_importance = abs(model.coef_[0])
        elif isinstance(model, tree.DecisionTreeClassifier):
            feature_importance = model.feature_importances_
        else:
            raise ValueError("Model type not supported for feature importance extraction")
        sorted_idx = feature_importance.argsort()[::-1]  # Sort in descending order

        print("\nTop Important Predictors:")
        for i in range(min(10, len(sorted_idx))):  # Print top 10 or fewer
            print(f"{all_features[sorted_idx[i]]}: {feature_importance[sorted_idx[i]]:.4f}")


            