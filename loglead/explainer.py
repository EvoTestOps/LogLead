from sklearn.metrics.pairwise import cosine_similarity
import polars as pl
import numpy as np
import umap
import plotly.express as px

import shap 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class NNExplainer:
    """A class for explaining the anomaly detection results using nearest neighbour search.
    With the class the user can find the closest normal instance to each anomalous instance
    in the vector space and visualize the instances in 2D UMAP space for further interactive
    exploration.
    """
    def __init__(self, df: pl.DataFrame, X: np.ndarray, id_col: str, pred_col: str):
        """Initializes the NNExplainer class with the given DataFrame, the feature matrix X,
        the column name for the instance id, and the column name for the prediction result.

        Args:
            df (pl.DataFrame): The dataframe produced by the anomaly detection module.
            X (np.ndarray): The feature matrix used for predicting the anomalies.
            id_col (str): The column name of the instance id. Should be unique for each instance.
            pred_col (str): The column name of the prediction result. Should be binary with 1 indicating an anomaly.
        """
        self.df = df
        self.X = X
        self.id_column = id_col
        self.prediction_column = pred_col
        self.mapping = self._get_normal_mapping()


    def _get_normal_mapping(self) -> pl.DataFrame:
        """Finds the closest normal instance (column indicating predictions set as False) to 
        each anomalous instance (column indicating predictions set as True) and returns the
        corresponding mapping as a Polars DataFrame with anomalous_id column indicating the
        id of the anomalous instance and the normal_id column indicating the nearest instance 
        to the anomalous instance in the vector space measured with cosine similarity.

        Returns:
            pl.DataFrame: The mapping of the anomalous instances to the nearest normal instances.
        """
        non_anomalous_ids = self.df.filter(pl.col(self.prediction_column) != 1).select(pl.col(self.id_column).alias("normal_id"))
        non_anomalies = self.X[~self.df.select(pl.col(self.prediction_column)).to_series()]
        anomalous_ids = self.df.filter(pl.col(self.prediction_column) == 1).select(pl.col(self.id_column).alias("anomalous_id"))
        anomalies = self.X[self.df.select(pl.col(self.prediction_column)).to_series()]
        similarities = cosine_similarity(anomalies, non_anomalies).argmax(axis=1)
        similarity_mapping = pl.concat([anomalous_ids, non_anomalous_ids[similarities]], how="horizontal")
        return similarity_mapping


    def print_log_content_from_nn_mapping(self) -> None:
        """Prints the log content of the anomalous and the closest normal instances in the mapping.
        The content is defined to be the list in the column e_words of the Polars DataFrame.
        """
        assert "e_words" in self.df.columns, "The column e_words is not present in the DataFrame."
        assert self.df.select(pl.col("e_words")).dtypes[0].is_nested(), "The column e_words is not nested data type."

        for anomaly, normal in self.mapping.rows():
            anomaly_words = self.df.filter(pl.col(self.id_column) == anomaly).select(pl.col("e_words")).to_series().to_list()[0]
            print(f"Anomaly sequence:{' '*8}{' '.join(anomaly_words)}")

            normal_words = self.df.filter(pl.col(self.id_column) == normal).select(pl.col("e_words")).to_series().to_list()[0]
            print(f"Closest normal sequence: {' '.join(normal_words)}\n")


    def print_features_from_nn_mapping(self, feature_cols: list[str]) -> None:
        """Prints the given features of the anomalous and the closest normal instances.

        Args:
            feature_cols (list[str]): The list of feature columns to be printed.
        """
        for anomaly, normal in self.mapping.rows():
            print(f"Features of anomaly {anomaly}: {self.df.filter(pl.col(self.id_column) == anomaly).select(pl.col(feature_cols)).to_pandas().values}")
            print(f"Features of closest normal {normal}: {self.df.filter(pl.col(self.id_column) == normal).select(pl.col(feature_cols)).to_pandas().values}")
            print("\n"*2)


    def print_false_positive_content(self, ground_truth_col: str):
        """Prints the content of the false positive instances in the log data. The false positive
        instances are the instances that are predicted to be anomalous but are not according to
        the ground truth labels.

        Args:
            ground_truth_col (str): The column name for the ground truth labels.
        """
        false_positives = self.df.filter((pl.col(self.prediction_column) == True) & (pl.col(ground_truth_col) == False)).select(pl.col(self.id_column), pl.col("e_words"))
        print("False positive sequences:")
        for row in false_positives.rows():
            print(f"{row[0]}: {' '.join(row[1])}")

    
    def print_false_negative_content(self, ground_truth_col: str):
        """Prints the content of the false negative instances in the log data. The false negative
        instances are the instances that are predicted to be normal but are anomalous according to
        the ground truth labels.

        Args:
            ground_truth_col (str): The column name for the ground truth labels.
        """
        false_negatives = self.df.filter((pl.col(self.prediction_column) == False) & (pl.col(ground_truth_col) == True)).select(pl.col(self.id_column), pl.col("e_words"))
        print("False negative sequences:")
        for row in false_negatives.rows():
            print(f"{row[0]}: {' '.join(row[1])}")


    def plot_features_in_two_dimensions(self, ground_truth_col: str = None) -> None:
        """Plots the features of the instances in 2D UMAP space. The instances are colored by whether
        they are predicted to be anomalous or not. If ground_truth_col is provided, the instances are
        also symbolized by the ground truth labels. The visualization is interactive and can be used to
        explore the instances in the 2D space.

        Args:
            ground_truth_col (str, optional): The column name for the ground truth labels. Defaults to None.
        """
        embeddings = umap.UMAP().fit_transform(self.X)
        df_vis = pl.DataFrame(embeddings, schema=["UMAP-1", "UMAP-2"])
        df_vis = df_vis.with_columns(
            self.df.select(pl.col(self.id_column)).to_series().alias(self.id_column),
            self.df.select(pl.col(self.prediction_column)).to_series().alias(self.prediction_column)
        )
        if ground_truth_col:
            symbol_col = "ground_truth"
            df_vis = df_vis.with_columns(ground_truth=self.df.select(pl.col(ground_truth_col)).to_series())
        else:
            symbol_col = None
        
        df_vis = df_vis.join(self.mapping, left_on=self.id_column, right_on="anomalous_id", how="left")
        df_vis = df_vis.with_columns(pl.when(pl.col("normal_id").is_null()).then(pl.lit("None")).otherwise(pl.col("normal_id")).alias("nearest_normal"))

        fig = px.scatter(
            data_frame=df_vis, 
            color=self.prediction_column, 
            x="UMAP-1", y="UMAP-2", 
            hover_data=[self.id_column, "nearest_normal"],
            title="Logs visualized in 2D UMAP space", 
            symbol=symbol_col,
            symbol_map={True: "cross", False: "circle"},)
        fig.show()


class ShapExplainer:
    """A class for explaining the anomaly detection results using SHapley Additive exPlanations.
    With the class the user can calculate the SHAP values for the features of the instances and
    visualize the SHAP values in different plots to understand the importance of the features
    in the anomaly detection model. The class currently supports the following anomaly detection
    models: Logistic Regression, Linear Support Vector Classifier, Decision Tree Classifier,
    Random Forest Classifier, Isolation Forest, and XGBoost Classifier.
    """
    def __init__(self, sad, ignore_warning=False, plot_featurename_len=16): 
        """Initializes the ShapExplainer class with the given anomaly detection object. The anomaly
        detection object should have the following attributes: model, X_train, X_test, and vectorizer.

        Args:
            sad (AnomalyDetection): The anomaly detection object used for predicting the anomalies.
            ignore_warning (bool, optional): Are warning about large dataset ignored. Defaults to False.
            plot_featurename_len (int, optional): Sets the lenght of truncated featurename in plots. Defaults to 16.
        """
        self.model = sad.model
        self.X_train = sad.X_train
        self.X_test = sad.X_test
        self.vec = sad.vectorizer
        self.warn = not ignore_warning # Should the program warn if large dataset
        self.Svals = None # SHAP values
        self.expl = None # Shap explainer
        self.istree =  False # Do we have tree model
        self.truncatelen = plot_featurename_len # variable for lengt of truncated name
        self.func = self._scuffmapping() # Do the mapping of model in init
        self.shapdata = None # Contains the data used to calc shapvalues
        self.threshold = 1500 # How many features before warning, can be changed if needed
        self.index = None # Sorted Indexes

    def linear(self):
        """Creates a Linear ShapExplainer object with given train data.
        """
        self.expl = shap.LinearExplainer(self.model, self.X_train, feature_names=self._truncatefn(self.truncatelen))
        return self.expl

    # should XGBoost be also a tree?
    def tree(self):
        """Creates a Tree ShapExplainer object with given train data.
        """
        self.expl  = shap.TreeExplainer(self.model, data=self.X_train.toarray(), feature_names=self._truncatefn(self.truncatelen))
        return self.expl


    def kernel(self):
        """Creates a Kernel ShapExplainer object with given train data.
        """
        self.expl = shap.KernelExplainer(self.model.predict, self.X_train, feature_names=self._truncatefn(self.truncatelen))
        return self.expl


    def plain(self):
        """Creates a Standard ShapExplainer object with given anomaly detector.
        """
        self.expl = shap.Explainer(self.model, feature_names=self._truncatefn(self.truncatelen))
        return self.expl


    # a function for mapping, could be changed to cases in python 3.10
    def _scuffmapping(self):
        """Maps the anomaly detection model to the correct ShapExplainer function.

        Returns:
            shap.Explainer: The ShapExplainer object for the anomaly detection model.
        """
        if isinstance(self.model, (LogisticRegression,LinearSVC)):
            return self.linear
        elif isinstance(self.model, (IsolationForest,DecisionTreeClassifier,RandomForestClassifier)):
            self.istree = True
            return self.tree
        elif isinstance(self.model, (XGBClassifier)):
            return self.plain
        else:
            # maybe this is good?
            raise NotImplementedError


    # should this sample the default test data?
    # should this return the shap values?
    def calc_shapvalues(self, test_data:np.ndarray=None, custom_slice:slice=None):
        """This function creates the SHAP values for a given vectorized dataset. The data 
        should be vectorized by a vectorizer of the trained anomaly detection model.

        Args:
            test_data (np.ndarray, optional): The data used to calculate the SHAP values. If not given, the function uses the test data of the anomaly detection object.
            custom_slice (slice, optional): The used dataset can be sliced by a Python slice object to select only a sample of the data to be used. Defaults to None.

        Raises:
            ResourceWarning: If ignore_warning not set true when creating the ShapExplainer, stops running if the computation is too resource intensive.

        Returns:
            np.ndarray: The calculated SHAP-values.
        """
        if test_data == None:
            test_data = self.X_test 
        
        if custom_slice:
            test_data = test_data[custom_slice]
        
        featureamount = self.vec.get_feature_names_out().shape[0]
        dataamount = test_data.size
        
        # the actual threshold could be tweaked but it was found that both data and feature
        # amount matters, so now the warning looks for both of them.
        # Could be changed to be something better if needed, now just a warning.
        if (dataamount >= 1000*self.threshold or featureamount >= self.threshold) and self.warn:
            print("Using large data set / many features, calculating shapvalues can be resource intensive!")
            raise ResourceWarning

        self.shapdata = test_data
        expl = self.func()
        if self.istree:
            self.Svals = expl(test_data.toarray())      
 
            if isinstance(self.model, IsolationForest):
                pass
            else:
            # some tree models gives two sets of values which are "mirrored"
            # when 1 the anomaly should have positive shap value
                self.Svals = self.Svals[:,:,1]
        else:
            self.Svals = expl(test_data)
        return self.Svals
        

    @property
    def shap_values(self):
        """
        Returns:
            np.ndarray: The stored SHAP-values.
        """
        return self.Svals


    @property
    def feature_names(self):
        """
        Returns:
            ndarray of str objects: Stored feature names.
        """
        return self.vec.get_feature_names_out()
        

    def sorted_shapvalues(self):
        """Can be used to get a sorted array of shap values. Sorted by feature importance.
        Does not contain base value.

        Returns:
            ndarray: ndarray of sorted shap values from most important to least.
        """
        if self.Svals == None:
            return None
        if self.index is not None:
            val = self.index
        else:
            val = np.argsort(np.sum(np.abs(self.Svals.values), axis=0))
        return np.array([self.Svals.values[:,i] for i in val][::-1])


    def sorted_featurenames(self):
        """
        Returns:
            list: Sorted feature names by SHAP importance.
        """
        val =  np.argsort(np.sum(np.abs(self.Svals.values), axis=0))
        self.index = val 
        fn = self.vec.get_feature_names_out()
        return [fn[i] for i in val][::-1]


    def _truncatefn(self,length:int):
        """Truncates the feature names to the given length.

        Args:
            length (int): The maximum length of the feature names.

        Returns:
            ndarray of str objects: The truncated feature names.
        """
        return self.vec.get_feature_names_out().astype(f'<U{length}')

    def plot(self, data:np.ndarray=None,plottype="summary", custom_slice:slice=None, displayed=16):
        """Plots the SHAP values in different plots to understand the importance of the features
        in the anomaly detection model. The plottype currently implemented are "summary", "bar", 
        and "beeswarm". The data to be plotted can be given directly to the function or the function
        uses the test data of the anomaly detection object. The custom_slice can be used to select
        only a sample of the data to be plotted.

        Args:
            data (np.ndarray, optional): Data used to calculate the SHAP values and creating the plot. Defaults to None. If no data given
            function uses test data included in anomaly detection object.
            plottype (str, optional): The plottype used for the plot, "summary", "bar", or "beeswarm". Defaults to "summary".
            slice (slice, optional): The slice used to select only a sample of the data to be plotted. Defaults to None.
            displayed (int, optional): The number of displayed features in the plot. Defaults to 16.
        """
        if data != None or self.Svals == None or custom_slice:
            self.calc_shapvalues(data, custom_slice)
            plotdata = self.shapdata
        elif custom_slice:
            plotdata = self.shapdata[custom_slice]
        else:
            plotdata = self.shapdata

        fullnames = self.sorted_featurenames()
        print("====================================")
        for i in range(displayed):
            print(fullnames[i])
        print("====================================")
     
        if plottype == "summary":
            shap.summary_plot(self.Svals, plotdata, max_display=displayed)
        elif plottype == "bar":
            shap.plots.bar(self.Svals, max_display=displayed)

        # create new elif for a new plot
        elif plottype == "beeswarm":
            # add shap plot with needed args, 
            # usually shap values from shap explainer
            # also usually max_display
            # depending on plot may have other requirements
            # for example summary_plot uses the dataset in some examples
            shap.plots.beeswarm(self.Svals, max_display=displayed)
            
