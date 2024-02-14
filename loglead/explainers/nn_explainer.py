from sklearn.metrics.pairwise import cosine_similarity
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import umap


def get_closest_normal_mapping_seq(df_test: pl.DataFrame, X_test: np.array) -> pl.DataFrame:
    non_anomalous_ids = df_test.filter(pl.col("pred_normal") != 1).select(pl.col("seq_id").alias("normal_seq_id"))
    non_anomalies = X_test[~df_test.select(pl.col("pred_normal")).to_series()]

    anomalous_ids = df_test.filter(pl.col("pred_normal") == 1).select(pl.col("seq_id").alias("anomalous_seq_id"))
    anomalies = X_test[df_test.select(pl.col("pred_normal")).to_series()]
    
    similarities = cosine_similarity(anomalies, non_anomalies).argmax(axis=1)
    similarity_mapping = pl.concat([anomalous_ids, non_anomalous_ids[similarities]], how="horizontal")
    return similarity_mapping


def print_log_content_from_nn_mapping_seq(mapping: pl.DataFrame, df_test: pl.DataFrame) -> None:
    for anomaly, normal in mapping.rows():
        anomaly_words = df_test.filter(pl.col("seq_id") == anomaly).select(pl.col("e_words")).to_series().to_list()[0]
        print(f"Anomaly sequence:{' '*8}{' '.join(anomaly_words)}")

        normal_words = df_test.filter(pl.col("seq_id") == normal).select(pl.col("e_words")).to_series().to_list()[0]
        print(f"Closest normal sequence: {' '.join(normal_words)}\n")


def plot_features_in_two_dimensions(df_test: pl.DataFrame, X_test: pl.DataFrame) -> None:
    embeddings = umap.UMAP().fit_transform(X_test)
    normal_embeddings = embeddings[~df_test.select(pl.col("pred_normal")).to_series()]
    anomaly_embeddings = embeddings[df_test.select(pl.col("pred_normal")).to_series()]

    plt.scatter(normal_embeddings[:, 0], normal_embeddings[:, 1], color="blue", marker="x", label="Normal")
    plt.scatter(anomaly_embeddings[:, 0], anomaly_embeddings[:, 1], color="red", marker="x", label="Anomaly")
    plt.legend()