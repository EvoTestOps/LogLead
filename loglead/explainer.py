from sklearn.metrics.pairwise import cosine_similarity
import polars as pl
import numpy as np
import umap
import plotly.express as px


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
