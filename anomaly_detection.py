import polars as pl
import numpy as np
from collections import Counter
#Faster sklearn enabled. See https://intel.github.io/scikit-learn-intelex/latest/
# Causes problems in RandomForrest. We have to use older version due to tensorflow numpy combatibilities
# from sklearnex import patch_sklearn
#patch_sklearn()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import OneClassSVM

from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
import scipy.sparse


class EventAnomalyDetection:
    def __init__(self, df):
        self.df = df

    def compute_ano_score(self, col_name, model_size):
        """
        :param col_name: A string that should be "e_words", "e_alphanumerics", or "e_cgrams"
        """
        #Export the column out from polars
        exported = self.df.select(pl.col(col_name).reshape([-1])).to_series().to_list()
        #Do a token count for whole data set
        counts = Counter(exported)  # 13s all hdfs
        #Create model consisting of top n (=model_tize) tokens
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

class SupervisedAnomalyDetection:
    def __init__(self, item_list_col=None, numeric_cols=None, emb_list_col=None, label_col="normal"):
        self.item_list_col = item_list_col
        self.numeric_cols = numeric_cols if numeric_cols else []
        self.label_col = label_col
        self.emb_list_col = emb_list_col
        #self.events, self.labels, self.additional_features = self._prepare_data(self.df_train)

    def _prepare_data(self, train, df_seq):
        X = None
        labels = df_seq.select(pl.col(self.label_col)).to_series().to_list()

        # Extract events
        if self.item_list_col:
            events = df_seq.select(pl.col(self.item_list_col)).to_series().to_list()
            events = [' '.join(e) for e in events]
            # We are training
            if train:
                self.vectorizer = CountVectorizer()
                X = self.vectorizer.fit_transform(events)
            # We are predicting
            else:
                X = self.vectorizer.transform(events)

        # Extract lists of embeddings
        if  self.emb_list_col:
            emb_list = df_seq.select(pl.col(self.emb_list_col)).to_series().to_list()
            
            # Convert lists of floats to a matrix
            #emb_matrix = np.array(emb_list)
            emb_matrix = np.vstack(emb_list)
            # Stack with X
            X = hstack([X, emb_matrix]) if X is not None else emb_matrix

        # Extract additional predictors
        if self.numeric_cols:
            additional_features = df_seq.select(self.numeric_cols).to_pandas().values
            X = hstack([X, additional_features]) if X is not None else additional_features

        return X, labels

    #Overwrites previous model
    def train_model(self, df_seq, model):
        X_train, labels = self._prepare_data(train=True, df_seq=df_seq)
        self.model = model
        self.model.fit(X_train, labels)
        
    def predict(self, df_seq, print_scores=True):
        X_test, labels = self._prepare_data(train=False, df_seq=df_seq)
        predictions = self.model.predict(X_test)
        #IsolationForrest does not give binary predictions. Convert
        if isinstance(self.model, (IsolationForest, LocalOutlierFactor,KMeans, OneClassSVM)):
            predictions = np.where(predictions > 0, 1, 0)
        df_seq = df_seq.with_columns(pl.Series(name="pred_normal", values=predictions.tolist()))
        if print_scores:
            self._print_evaluation_scores(labels, predictions, self.model)
        return df_seq

    def train_LR(self, df_seq, max_iter=1000):
        self.train_model (df_seq, LogisticRegression(max_iter=max_iter))
        
    def train_DT(self, df_seq):
        self.train_model (df_seq, DecisionTreeClassifier())

    def train_LSVM(self, df_seq,  penalty='l1', tol=0.1, C=1, dual=False, class_weight=None, max_iter=500):
        self.train_model (df_seq, LinearSVC(
            penalty=penalty, tol=tol, C=C, dual=dual, class_weight=class_weight, max_iter=max_iter))

    def train_IsolationForest(self, df_seq, n_estimators=100,  max_samples='auto', contamination="auto",filter_anos=False):
        if filter_anos:
            df_seq = df_seq.filter(pl.col(self.label_col))
        self.train_model (df_seq, IsolationForest(
            n_estimators=n_estimators, max_samples=max_samples, contamination=contamination))
                          
    def train_LOF(self, df_seq, n_neighbors=20, max_samples='auto', contamination="auto", filter_anos=True):
        #LOF novelty=True model needs to be trained without anomalies
        #If we set novelty=False then Predict is no longer available for calling.
        #It messes up our general model prediction routine
        if filter_anos:
            df_seq = df_seq.filter(pl.col(self.label_col))
        self.train_model (df_seq, LocalOutlierFactor(
            n_neighbors=n_neighbors,  contamination=contamination, novelty=True))
    
    def train_KMeans(self, df_seq):
        self.train_model(df_seq, KMeans(n_init="auto",n_clusters=2))

    def train_OneClassSVM(self, df_seq):
        self.train_model(df_seq, OneClassSVM(max_iter=500))

    def train_RF(self, df_seq):
        self.train_model(df_seq, RandomForestClassifier())

    def train_XGB(self, df_seq):
        self.train_model(df_seq, XGBClassifier())
        
    def evaluate_all_ads(self, train_df, test_df):
        for method_name in sorted(dir(self)):
            if method_name.startswith("train_") and not  method_name.startswith("train_model") :
                method = getattr(self, method_name)
                if callable(method):
                    method(train_df)
                    self.predict(test_df)

    def _print_evaluation_scores(self, y_test, y_pred, model, f_importance = False):
        print(f"Results from model: {type(model).__name__}")
        # Evaluate the model's performance
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        
        from sklearn.metrics import f1_score
        # Compute the F1 score
        f1 = f1_score(y_test, y_pred)
        # Print the F1 score
        print(f"F1 Score: {f1:.2f}")


        # Compute the confusion matrix--------------------------------------------------
        from sklearn.metrics import confusion_matrix
        #import matplotlib.pyplot as plt
        #import seaborn as sns
        cm = confusion_matrix(y_test, y_pred)
        # Print the confusion matrix
        print("Confusion Matrix:")
        print(cm)
        #F1--------------------------------------------------------

        # Print feature importance
        if (f_importance):
            if hasattr(self, 'vectorizer') and self.vectorizer:
                event_features = self.vectorizer.get_feature_names_out()
                event_features = list(event_features)
            else:
                event_features = []

            all_features = event_features + self.numeric_cols
            if isinstance(model, (LogisticRegression, LinearSVC)):
                feature_importance = abs(model.coef_[0])
            elif isinstance(model, DecisionTreeClassifier):
                feature_importance = model.feature_importances_
            else:
                print("Model type not supported for feature importance extraction")
                return
            sorted_idx = feature_importance.argsort()[::-1]  # Sort in descending order

            print("\nTop Important Predictors:")
            for i in range(min(10, len(sorted_idx))):  # Print top 10 or fewer
                print(f"{all_features[sorted_idx[i]]}: {feature_importance[sorted_idx[i]]:.4f}")
                