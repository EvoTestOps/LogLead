import time
from inspect import isclass

import polars as pl
import numpy as np
import pandas as pd
#Faster sklearn enabled. See https://intel.github.io/scikit-learn-intelex/latest/
# Causes problems in RandomForrest. We have to use older version due to tensorflow numpy combatibilities
# from sklearnex import patch_sklearn
#patch_sklearn()
from scipy.sparse import hstack
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import zlib
import difflib
import warnings

from .RarityModel import RarityModel
from .OOV_detector import OOV_detector

__all__ = ['AnomalyDetector', 'LogDistance']


class LogDistance:
    def __init__(self, df_train, df_analyze, field="m_message", vectorizer=CountVectorizer):
        """
        Initialize the LogDistannce class with data for distance measurements.

        Parameters:
        df_train (DataFrame): The training DataFrame containing the text data to be used as the reference.
        df_analyze (DataFrame): The analysis DataFrame containing the text data to be compared with the reference.
        field (str): The column name in both DataFrames that contains the text data to analyze. Defaults to "m_message".
        vectorizer (type): The vectorization method to use for converting text data into numerical vectors. 
                           It should be a scikit-learn vectorizer class. Defaults to CountVectorizer.
        """
        self.s_train = df_train.select(pl.col(field).str.concat(" ")).item()
        self.s_analyze = df_analyze.select(pl.col(field).str.concat(" ")).item()

        self.size1 = df_train.height
        self.size2 = df_analyze.height
        self.vectorizer = vectorizer()

        combined_texts = [self.s_train, self.s_analyze]
        try:
            self.v_combined = self.vectorizer.fit_transform(combined_texts)
        except ValueError as e:
            if "empty vocabulary" in str(e):
                #print(f"Empty vocabulary for field {field}. Skipping calculation.")
                self.v_train = None
                self.v_analyze = None
            else:
                raise
        else:
            self.v_train = self.v_combined[0]
            self.v_analyze = self.v_combined[1]
        
        #For diffing. Process only if diffing is called
        self.df_train = df_train
        self.df_analyze = df_analyze
        self.field = field

    def diff_lines(self):
        """
        Calculate the differences between the training and analysis data and return
        the result as a Polars DataFrame.

        Returns:
        pl.DataFrame: A DataFrame containing the line number, difference indicator,
        original line content, and modified line content.
        """
        # Add line numbers to both DataFrames
        df_train = self.df_train.with_row_count(name="line_number")
        df_analyze = self.df_analyze.with_row_count(name="line_number")

        # Extract content as lists for comparison
        lines_train = df_train.select(pl.col(self.field)).to_series().to_list()
        lines_analyze = df_analyze.select(pl.col(self.field)).to_series().to_list()

        # Use difflib to compute the differences
        d = difflib.Differ()
        diff = list(d.compare(lines_train, lines_analyze))

        # Create a DataFrame with a single column containing the diff output
        df_diff = pl.DataFrame({"diff_output": diff})

        # Split the 'diff_output' column into 'difference' and 'content'
        df_diff = df_diff.with_columns([
            pl.col("diff_output").str.slice(0, 1).alias("difference"),  # First character as difference indicator
            pl.col("diff_output").str.slice(2).alias("content")         # Content starting from the third character
        ])

        # Add line numbers for display purposes
        df_diff = df_diff.with_row_count(name="line_number")

        # Drop the intermediate 'diff_output' column
        df_diff = df_diff.drop("diff_output")
        return df_diff



    def cosine(self):
        """Calculate cosine distance between the training and analysis data."""
        if self.v_train is None or self.v_analyze is None:
            return None
        cosine_sim = float(cosine_similarity(self.v_train, self.v_analyze)[0][0])
        cosine_dist = 1 - cosine_sim
        return cosine_dist

    def jaccard(self):
        """Calculate Jaccard distance between the training and analysis data."""
        if self.v_train is None or self.v_analyze is None:
            return None
        v_binary1 = (self.v_train > 0).astype(int)
        v_binary2 = (self.v_analyze > 0).astype(int)
        jaccard_sim = float(jaccard_score(v_binary1, v_binary2, average="samples"))
        jaccard_dist = 1 - jaccard_sim
        return jaccard_dist


    def compression(self):
        """Calculate compression distance between the training and analysis data."""
        import bz2 as compressor
        if self.v_train is None or self.v_analyze is None:
            return None
        len1 = len(compressor.compress(self.s_train.encode()))
        len2 = len(compressor.compress(self.s_analyze.encode()))
        combined_len = len(compressor.compress((self.s_train + self.s_analyze).encode()))
        compression = (combined_len - min(len1, len2)) / max(len1, len2)
        return compression

    def containment(self):
        """Calculate containment distance between the training and analysis data."""
        if self.v_train is None or self.v_analyze is None:
            return None
        v_binary1 = (self.v_train > 0).astype(int)
        v_binary2 = (self.v_analyze > 0).astype(int)
        intersection = v_binary1.multiply(v_binary2).sum()

        # Calculate containment measure
        containment_measure = float(intersection / min(v_binary1.sum(), v_binary2.sum()) if min(v_binary1.sum(), v_binary2.sum()) > 0 else 0)
        
        # Calculate containment distance
        containment_distance = 1 - containment_measure
        return containment_distance


    def measure_all_distances(self, print_values=True):
        """Calculate all distance measures and print them if specified."""
        cosine_sim = self.cosine()
        jaccard = self.jaccard()
        compression = self.compression()
        containment = self.containment()
        if print_values:
            print(f"Distance of column is Cosine: {cosine_sim}, Jaccard: {jaccard}, Compression: {compression}, Containment: {containment}")

        return cosine_sim, jaccard, compression, containment, self.size1, self.size2

class AnomalyDetector:
    def __init__(self, item_list_col=None, numeric_cols=None, emb_list_col=None, label_col="anomaly", 
                 store_scores=False, print_scores=True, auc_roc=False):
        self.item_list_col = item_list_col
        self.numeric_cols = numeric_cols if numeric_cols else []
        self.label_col = label_col
        self.emb_list_col = emb_list_col
        self.store_scores = store_scores
        self.storage = _ModelResultsStorage()
        self.print_scores=print_scores
        self.train_vocabulary = None #TODO Is this used anywhere?
        self.auc_roc = auc_roc
        self.filter_anos = False

    def test_train_split(self, df, test_frac=0.9, shuffle=True, vectorizer_class=CountVectorizer):
        # Shuffle the DataFrame
        if shuffle:
            df = df.sample(fraction = 1.0, shuffle=True)
        elif 'start_time' in df.columns:
            df = df.sort('start_time')
        #Do we need this or sequence based quaranteed to be in correct order
        elif "m_timestamp" in df.columns:
            df = df.sort('m_timestamp')   
        # Split ratio
        test_size = int(test_frac * df.shape[0])

        # Split the DataFrame using head and tail
        self.train_df = df.head(-test_size) #Returns all rows expect last abs(-test_size)
        self.test_df = df.tail(test_size) #Returns the last test_size rows
        self.prepare_train_test_data(vectorizer_class=vectorizer_class)
        
    def prepare_train_test_data(self, vectorizer_class=CountVectorizer):
        #Prepare all data for running
        self.X_train, self.labels_train, self.vectorizer  = self._prepare_data(self.train_df, vectorizer_class)
        self.X_test, self.labels_test, _ = self._prepare_data(self.test_df, self.vectorizer)
        #No anomalies dataset is used for some unsupervised algos.
        if self.label_col in self.train_df.columns:
            self.X_train_no_anos, _, self.vectorizer_no_anos = self._prepare_data(self.train_df.filter(pl.col(self.label_col).not_()),
                                                     vectorizer_class)
            self.X_test_no_anos, self.labels_test_no_anos, _ = self._prepare_data(self.test_df, self.vectorizer_no_anos)
        else: #As we have no labels there is no difference in anos vs no_anos case
            self.X_train_no_anos, self.vectorizer_no_anos = self.X_train, self.vectorizer
            self.X_test_no_anos, self.labels_test_no_anos, = self.X_test, self.labels_test
     
    # added a way to get the test data
    @property
    def test_data(self):
        return self.X_test, self.labels_test

    # added a way to get the vectorizer
    @property
    def vec(self):
        return self.vectorizer

    @property #TODO used?
    def voc(self):
        return self.train_vocabulary

    #To enable pickling
    def identity_function(self,x):
        return x

    # did some changes so the vectorizer does not get overwritten by anos 
    def _prepare_data(self, df_seq, vectorizer_class=CountVectorizer):
        X = None
        if self.label_col in df_seq.columns:
            labels = df_seq.select(pl.col(self.label_col)).to_series().to_list()
        else:
            labels = []
            warnings.warn("WARNING! data has no labels. Only unsupervised methods will work.",
                            category=UserWarning
                            ) 
        vectorizer = None

        # Extract events
        if self.item_list_col:
            # Extract the column
            column_data = df_seq.select(pl.col(self.item_list_col))             
            events = column_data.to_series().to_list()
            # We are training because vectorizer is a class
#            if train:
            if isinstance(vectorizer_class, type):
                # Check the datatype  
                if column_data.dtypes[0]  == pl.datatypes.Utf8: #We get strs -> Use SKlearn Tokenizer
                    vectorizer = vectorizer_class() 
                elif column_data.dtypes[0]  == pl.datatypes.List(pl.datatypes.Utf8): #We get list of str, e.g. words -> Do not use Skelearn Tokinizer 
                    vectorizer = vectorizer_class(analyzer=self.identity_function)
                X = vectorizer.fit_transform(events)
                self.train_vocabulary = vectorizer.vocabulary_ #Needed?
            # We are predicting because vectorizer_class is instance of the previously created vectorizer. 
            elif isinstance(vectorizer_class, object):
                  vectorizer = vectorizer_class
                  X = vectorizer.transform(events)

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

        return X, labels, vectorizer    
        
    def train_model(self, model,  /, *, filter_anos=False, **model_kwargs):
        X_train_to_use = self.X_train_no_anos if filter_anos else self.X_train
        #Store the current the model and whether it uses ano data or no
        if isclass(model):
            self.model = model(**model_kwargs)
        else:
            self.model = model  # Backwards compatibility with previous implementation
        self.filter_anos = filter_anos
        self.model.fit(X_train_to_use, self.labels_train)

    def predict(self, custom_plot=False):
        #Binary scores
        X_test_to_use = self.X_test_no_anos if self.filter_anos else self.X_test
        predictions = self.model.predict(X_test_to_use)
        #Unsupervised modeles give predictions between -1 and 1. Convert to 0 and 1
        if isinstance(self.model, (IsolationForest, LocalOutlierFactor,KMeans, OneClassSVM)):
            predictions = np.where(predictions < 0, 1, 0)
        df_seq = self.test_df.with_columns(pl.Series(name="pred_ano", values=predictions.tolist()))
        
        #Continuous scores
        predictions_proba = None
        if self.auc_roc:
            if isinstance(self.model, (IsolationForest, LocalOutlierFactor, OneClassSVM)):
                # Unsupervised models give anomaly scores or decision function values
                predictions_proba = 1- self.model.decision_function(X_test_to_use)
            elif isinstance(self.model, (KMeans)):
                from sklearn.metrics.pairwise import pairwise_distances
                predictions_proba = np.min(pairwise_distances(X_test_to_use, self.model.cluster_centers_), axis=1)
            elif isinstance(self.model, LinearSVC):
                # LinearSVC does not have predict_proba method by default
                # Use decision_function method to obtain confidence scores
                # and convert them to probabilities using Platt scaling
                from sklearn.calibration import CalibratedClassifierCV
                X_train_to_use = self.X_train_no_anos if  self.filter_anos else self.X_train
                calibrated_model = CalibratedClassifierCV(self.model, cv='prefit')
                calibrated_model.fit(X_train_to_use, self.labels_train)
                predictions_proba = calibrated_model.predict_proba(X_test_to_use)[:, 1]
            elif isinstance(self.model, (OOV_detector, RarityModel)):
                predictions_proba = self.model.scores    
            else:
                # Supervised models give probabilities using predict_proba method
                predictions_proba = self.model.predict_proba(X_test_to_use)[:, 1]
            df_seq = self.test_df.with_columns(pl.Series(name="pred_ano_proba", values=predictions_proba.tolist()))      

        if self.print_scores:
            self._print_evaluation_scores(self.labels_test, predictions,predictions_proba, self.model)
        if custom_plot:
            self.model.custom_plot(self.labels_test)
        if self.store_scores:
            self.storage.store_test_results(self.labels_test, predictions,predictions_proba, type(self.model).__name__, 
                                            self.item_list_col, self.numeric_cols, self.emb_list_col)
        return df_seq 
       
    def train_LR(self, max_iter=4000, tol=0.0003):
        self.train_model(LogisticRegression, max_iter=max_iter, tol=tol)
    
    def train_DT(self):
        self.train_model(DecisionTreeClassifier)

    def train_LSVM(self, penalty='l1', tol=0.1, C=1, dual=False, class_weight=None, max_iter=4000):
        self.train_model(LinearSVC, penalty=penalty, tol=tol, C=C, dual=dual, class_weight=class_weight,
                         max_iter=max_iter)

    def train_IsolationForest(self, n_estimators=100,  max_samples='auto', contamination="auto",filter_anos=False):
        self.train_model(IsolationForest, filter_anos=filter_anos,
                         n_estimators=n_estimators, max_samples=max_samples, contamination=contamination)
                          
    def train_LOF(self, n_neighbors=20, contamination="auto", filter_anos=True):
        #LOF novelty=True model needs to be trained without anomalies
        #If we set novelty=False then Predict is no longer available for calling.
        #It messes up our general model prediction routine
        self.train_model(LocalOutlierFactor, filter_anos=filter_anos, n_neighbors=n_neighbors,
                         contamination=contamination, novelty=True)
    
    def train_KMeans(self, n_clusters=2, filter_anos = False):
        self.train_model(KMeans, filter_anos=filter_anos, n_init="auto", n_clusters=n_clusters)

    def train_OneClassSVM(self):
        self.train_model(OneClassSVM, max_iter=1000)

    def train_RF(self):
        self.train_model(RandomForestClassifier)

    def train_XGB(self):
        self.train_model(XGBClassifier)

    # stop

    def train_RarityModel(self, filter_anos=True, threshold=250):
        self.train_model(RarityModel, filter_anos=filter_anos, threshold=threshold)
        
    def train_OOVDetector(self, len_col=None, filter_anos=True, threshold=1):
        if len_col == None: 
            if "event" in self.item_list_col:
                len_col = "e_event_id_len" #item list col has the parser name when using events, but length doesn't
            else:
                len_col = self.item_list_col+"_len"
        self.train_model(OOV_detector, filter_anos=filter_anos, len_col=len_col, item_list_col=self.item_list_col, test_df=self.test_df, threshold=threshold)
        
    def evaluate_all_ads(self, disabled_methods=None):
        if disabled_methods is None:
            disabled_methods = set()
        train_methods = {getattr(self, m) for m in dir(self) if m.startswith('train_') and m not in disabled_methods
                         and callable(getattr(self, m))}
        train_methods.discard(self.train_model)
        for method in train_methods:
            if not self.print_scores:
                print(f"Running {method}")
            time_start = time.process_time()
            method()
            self.predict()
            if self.print_scores:
                print(f'Total time: {time.process_time()-time_start:.2f} seconds')

    @property
    def get_model(self):
        return self.model


    def evaluate_with_params(self, models_dict):
        for func_name, params in models_dict.items():
            func_name = "train_"+func_name
            method = getattr(self, func_name)
            method(**params)
            self.predict()

    def _print_evaluation_scores(self, y_test, y_pred, y_pred_proba, model, f_importance=False, auc_roc=True, f1optimize=False):
        print(f"Results from model: {type(model).__name__}")
        # Evaluate the model's performance
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Compute the F1 score
        f1 = f1_score(y_test, y_pred)
        # Print the F1 score
        print(f"F1 Score: {f1:.4f}")
        if self.auc_roc:
            # Compute the AUC-ROC score
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"AUC-ROC Score: {auc:.4f}")
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
                
        #AUC-ROC and optimal F1 search for selected unsupervised models
        if auc_roc or f1optimize:   
            from skopt import gp_minimize
            from skopt.space import Real
            from skopt.utils import use_named_args
            def bayesian_optimization(true_labels, anomaly_scores, max_iterations=20):
                t_start = time.time()
                space = [Real(min(anomaly_scores), max(anomaly_scores), name='threshold')]
                
                @use_named_args(space)
                def objective(**params):
                    threshold = params['threshold']
                    predicted_labels = (anomaly_scores >= threshold).astype(int)
                    return -f1_score(true_labels, predicted_labels)

                # Suppress the specific UserWarning about objective re-evaluation
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    res_gp = gp_minimize(objective, space, n_calls=max_iterations, random_state=0)
                print(f"F1 optimization time taken {(time.time() - t_start):.4f}")
                return res_gp.x[0], -res_gp.fun
            
            titlestr = type(self.model).__name__ + " ROC" # for plot (if it's on)
            X_test_to_use = self.X_test_no_anos if self.filter_anos else self.X_test
            if isinstance(self.model, (IsolationForest)):
                y_pred = 1 - model.score_samples(X_test_to_use) #lower = anomalous
                if auc_roc: print(f"AUCROC: {self._auc_roc_analysis(y_test, y_pred, titlestr):.4f}")
                if f1optimize: print(f"Optimal F1: {bayesian_optimization(y_test, y_pred)[1]:.4f}")
            if isinstance(self.model, KMeans):
                y_pred = np.min(model.transform(X_test_to_use), axis=1) #Shortest distance from the cluster to be used as ano score
                if auc_roc: print(f"AUCROC: {self._auc_roc_analysis(y_test, y_pred, titlestr):.4f}")
                if f1optimize: print(f"Optimal F1: {bayesian_optimization(y_test, y_pred)[1]:.4f}")
            if isinstance(self.model, (RarityModel, OOV_detector)):
                if auc_roc: print(f"AUCROC: {self._auc_roc_analysis(y_test, model.scores, titlestr):.4f}")
                if f1optimize: print(f"Optimal F1: {bayesian_optimization(y_test, model.scores)[1]:.4f}")


    @staticmethod
    def _auc_roc_analysis(labels, preds, titlestr ="ROC", plot=False):
        # Compute the ROC curve
        fpr, tpr, thresholds = roc_curve(labels, preds)
        # Compute the AUC from the points of the ROC curve
        roc_auc = auc(fpr, tpr)

        if plot:
            try:
                import matplotlib.pyplot as plt
            except Exception as e:
                raise ImportError("Error import matplotlib") from e
            # Plot the ROC curve
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(titlestr)
            plt.legend(loc="lower right")
            plt.show()

        return roc_auc


class _ModelResultsStorage:
    def __init__(self):
        self.test_results = []

    def _create_input_signature(self, item_list_col, numeric_cols, emb_list_col):
        # Create the input signature by concatenating all input types
        input_parts = (
            item_list_col if item_list_col is not None else [],
            numeric_cols if numeric_cols is not None else [],
            emb_list_col if emb_list_col is not None else [],
        )
        input_signature = ''.join(str(item) for sublist in input_parts for item in sublist)
        return input_signature

    def store_test_results(self, y_test, y_pred, y_pred_proba, model_name, item_list_col=None, numeric_cols=None, emb_list_col=None):
        input_signature = self._create_input_signature(item_list_col, numeric_cols, emb_list_col)

        result = {
            #'model': model,
            'model':model_name,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'input_signature': input_signature,
        }
        self.test_results.append(result)

    def calculate_average_scores(self, score_type='accuracy', metric='mean', mark_model_supervision = True):
        if score_type not in ['accuracy', 'f1', 'auc-roc']:
            raise ValueError("score_type must be 'accuracy', 'f1', 'auc-roc'.")
        #if metric not in ['mean', 'median']:
        #    raise ValueError("metric must be 'mean' or 'median'.")
        # Create a list to store each row of the DataFrame
        data_for_df = []

        # Iterate over stored results and calculate scores
        for result in self.test_results:
            #model_name = type(result['model']).__name__
            model_name = result['model']
            if mark_model_supervision:
                unsupervised = ["KMeans", "IsolationForest", "OneClassSVM", "LocalOutlierFactor", "OOV_detector", "RarityModel"]
                if model_name in unsupervised:
                    model_name = "us-"+model_name
                else:
                    model_name = "su-"+model_name
            input_signature = result['input_signature']
            
            # Calculate scores
            acc = accuracy_score(result['y_test'], result['y_pred'])
            f1 = f1_score(result['y_test'], result['y_pred'])
#            auc-roc = roc_auc_score(result['y_test'], result['y_pred_proba'])
            if 'y_pred_proba' in result and result['y_test'] is not None and result['y_pred_proba'] is not None:
                aucroc = roc_auc_score(result['y_test'], result['y_pred_proba'])
            else:
                aucroc = 0  # 
            # Append row to the list
            data_for_df.append({
                'Model': model_name,
                'Input Signature': input_signature,
                'Accuracy': acc,
                'F1 Score': f1,
                'AUC-ROC': aucroc
            })

        # Convert the list to a DataFrame
        df = pd.DataFrame(data_for_df)

        #  # Calculate average scores
        # if score_type == 'accuracy':
        #     df_average = df.groupby(['Model', 'Input Signature']).agg({'Accuracy': 'mean'}).reset_index()
        # elif  score_type == 'f1':
        #     df_average = df.groupby(['Model', 'Input Signature']).agg({'F1 Score': 'mean'}).reset_index()
        # else:
        #     df_average = df.groupby(['Model', 'Input Signature']).agg({'AUC-ROC': 'mean'}).reset_index()
        # Calculate average scores
        if score_type == 'accuracy':
            score_column = 'Accuracy'
        elif score_type == 'f1':
            score_column = 'F1 Score'
        else:
            score_column = 'AUC-ROC'
        
        df_average = df.groupby(['Model', 'Input Signature']).agg({score_column: metric}).reset_index()

        # # Pivot the DataFrame to get Input Signatures as columns
        score_column = 'Accuracy' if score_type == 'accuracy' else 'F1 Score' if score_type == 'f1' else 'AUC-ROC'
#        df_pivot = df_average.pivot(index='Model', columns='Input Signature', values=score_column).reset_index()
        df_pivot = df_average.pivot(index='Model', columns='Input Signature', values=score_column)
        df_pivot.columns = df_pivot.columns.map(lambda x: x.replace('_', '-'))
        # Fill NaN values with 0 or an appropriate value
        df_pivot.fillna(0, inplace=True)  # fill NaN values with 0

        df_pivot = df_pivot.apply(pd.to_numeric, errors='coerce')
        # Calculate and add column averages at the bottom of the DataFrame
        df_pivot.loc['Column Average'] = df_pivot.mean(numeric_only=True)

        # Calculate and add row averages as a new column
        df_pivot['Row Average'] = df_pivot.mean(axis=1, numeric_only=True)
        
        df_pivot = df_pivot.map(lambda x: f"{x:.3f}")
        return df_pivot
        
    def print_confusion_matrices(self, model_filter=None, signature_filter=None):
        for result in self.test_results:
            #model_name = type(result['model']).__name__
            model_name = result['model']
            input_signature = result['input_signature']
            cm = confusion_matrix(result['y_test'], result['y_pred'])

            # Check if model_name matches model_filter, if provided
            if model_filter and model_name != model_filter:
                continue

            # Check if input_signature matches signature_filter, if provided
            if signature_filter and input_signature != signature_filter:
                continue

            print(f"Model: {model_name}")
            print(f"Input Signature: {input_signature}")
            print("Confusion Matrix:")
            print(cm)
            
    def print_scores(self, model_filter=None, signature_filter=None, score_type='all'):
        # Check for valid score_type
        valid_score_types = ['accuracy', 'f1', 'all']
        if score_type not in valid_score_types:
            raise ValueError(f"score_type must be one of {valid_score_types}")

        for result in self.test_results:
            #model_name = type(result['model']).__name__
            model_name = result['model']
            input_signature = result['input_signature']
            acc = accuracy_score(result['y_test'], result['y_pred'])
            f1 = f1_score(result['y_test'], result['y_pred'])

            # Filter results
            if model_filter and model_name != model_filter:
                continue
            if signature_filter and input_signature != signature_filter:
                continue

            # Print results based on score_type
            print(f"Model: {model_name}")
            print(f"Input Signature: {input_signature}")
            if score_type in ['accuracy', 'all']:
                print(f"Accuracy: {acc:.4f}")
            if score_type in ['f1', 'all']:
                print(f"F1 Score: {f1:.4f}")


