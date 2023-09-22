import polars as pl
from collections import Counter

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
