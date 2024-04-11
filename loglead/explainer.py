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
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
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
    def __init__(self, sad, ignore_warning=False): 
        """_summary_

        Args:
            sad (_type_): _description_
            ignore (bool, optional): Are warning about large dataset ignored. Defaults to False.
        """
        # Takend from sad
        self.model = sad.model
        self.X_train = sad.X_train
        self.X_test = sad.X_test
        self.vec = sad.vectorizer

        # Should the program warn if large dataset
        self.warn = not ignore_warning
        # Shap values
        self.Svals = None
        # Shap explainer
        self.expl = None
        # Do we have tree model
        self.istree =  False 
        # Do the mapping of model in init
        self.func = self._scuffmapping()
        # Contains the data used to calc shapvalues
        self.shapdata = None
        # How many features before warning, can be changed if needed
        self.threshold = 1500


    # Different shap explainers
    def linear(self) :
        """Creates Linear ShapExplainer object with given train data.

        """
        self.expl = shap.LinearExplainer(self.model, self.X_train, feature_names=self._truncatefn(16))
        return self.expl

    # shjould xgb be a tree?
    def tree(self):
        """Creates Linear ShapExplainer object with given train data.

        """
        self.expl  = shap.TreeExplainer(self.model, data=self.X_train.toarray(), feature_names=self._truncatefn(16))
        return self.expl

    def kernel(self):
        """Creates Linear ShapExplainer object with given train data.
        """
        self.expl = shap.KernelExplainer(self.model.predict, self.X_train, feature_names=self._truncatefn(16))
        return self.expl


    def plain(self):
        """Creates Linear ShapExplainer object with given train data.

        """
        self.expl = shap.Explainer(self.model, feature_names=self._truncatefn(16))
        return self.expl


    # a function for mapping, could be changed to cases in python 3.10
    def _scuffmapping(self):
        """Used to determine what model is used by anomaly detection. Then
        create a corresponding Shap explainer object.

        Returns:
            _type_: _description_
        """
        # linear
        if isinstance(self.model, (LogisticRegression,LinearSVC)):
            return self.linear
        #tree
        elif isinstance(self.model, (IsolationForest,DecisionTreeClassifier,RandomForestClassifier)):
            self.istree = True
            return self.tree

        elif isinstance(self.model, (XGBClassifier)):
            return self.plain
        else:
            # maybe this is good?
            raise NotImplementedError


    # sample default test data??
    # should this return?
    def calc_shapvalues(self, test_data=None, custom_slice:slice=None):
        """This function creates shap values for given vectorized dataset. The data should be vectorized
        by trained anomaly detection models vectorizer.

        Args:
            test_data (_type_, optional): Shap values are calculated for given data.
            Defaults to test data of anomaly detection object.

            custom_slice (slice, optional): The used dataset can be sliced by python slice object.
            Defaults to None

        Raises:
            ResourceWarning: If ignore_warning not set true in ShapExplainer init stops running if calculation
            too resource intensive.

        Returns:
            _type_: Calculated Shap values
        """
        if test_data == None:
            test_data = self.X_test 
        
        if custom_slice:
            test_data = test_data[custom_slice]
        

        featureamount = self.vec.get_feature_names_out().shape[0]
        dataamount = test_data.size
        
        # the actual threshold could be tweaked but I found that both data and feature
        # amount matters, so now the warning looks for both of them.
        # Could be changed to be something better if needed now just a warning.
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
            _type_: Stored Shap values.
        """
        return self.Svals


    @property
    def feature_names(self):
        """
        Returns:
            _type_: Stored feature names.
        """
        return self.vec.get_feature_names_out()


    def sorted_featurenames(self):
        """
        Returns:
            list: Sorted feature names by shap importance.
        """
        val =  np.argsort(np.sum(np.abs(self.Svals.values), axis=0))
        fn = self.vec.get_feature_names_out()
        return [fn[i] for i in val][::-1]


    def _truncatefn(self,length):
        return self.vec.get_feature_names_out().astype(f'<U{length}')

    def plot(self, data:np.ndarray=None,plottype="summary", custom_slice:slice=None, displayed=16):
        """_summary_

        Args:
            data (np.ndarray, optional): Data used to calculate the shap values and creating the plot. Defaults to None. If no data given
            function uses test data included in anomaly detection object.
            plottype (str, optional): _description_. Defaults to "summary".
            slice (_type_, optional): _description_. Defaults to None.
            displayed (int, optional): _description_. Defaults to 16.
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

        # create new elif for new plot
        elif plottype == "beeswarm":
            # add shap plot with needed args, 
            # usually shap values from shap explainer
            # also usually max_display
            # depending on plot may have other requirements
            # for example summary_plot uses the dataset in some examples
            shap.plots.beeswarm(self.Svals, max_display=displayed)
            
