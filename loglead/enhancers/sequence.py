import polars as pl

from loglead import NextEventPredictionNgram

__all__ = ['SequenceEnhancer']


class SequenceEnhancer:
    def __init__(self, df, df_seq):
        self.df = df
        self.df_seq = df_seq

    def start_time(self):
        df_temp = self.df.group_by('seq_id').agg(pl.col('m_timestamp').min().alias('start_time'))
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')
        return self.df_seq

    def end_time(self):
        df_temp = self.df.group_by('seq_id').agg(pl.col('m_timestamp').max().alias('end_time'))
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')
        return self.df_seq

    def time_stamp(self):
        # We used median as time stamp for sequence
        df_temp = self.df.group_by('seq_id').agg(pl.col('m_timestamp').median().alias('time_stamp'))
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')
        return self.df_seq

    def seq_len(self):
        # Count the number of rows for each seq_id
        df_temp = self.df.group_by('seq_id').agg(pl.len().alias('seq_len'))
        # Join this result with df_sequences on seq_id
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')
        # Add an alias  that is compatible with the token len naming.
        self.df_seq = self.df_seq.with_columns(self.df_seq['seq_len'].alias('e_event_id_len'))

        return self.df_seq

    def events(self, event_col="e_event_drain_id"):
        # Aggregate event ids into a list for each seq_id
        df_temp = self.df.group_by('seq_id').agg(pl.col(event_col).alias(event_col))
        # Join this result with df_sequences on seq_id
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')
        return self.df_seq

    def tokens(self, token="e_words"):
        # df_temp = self.df.group_by('seq_id').agg(pl.col(token).flatten().alias(token))
        # Same as above but the above crashes due to out of memory problems. We might need this fix also in other rows
        df_temp = self.df.select("seq_id", token).explode(token).group_by('seq_id').agg(pl.col(token))
        # Join this result with df_sequences on seq_id
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')

        # lengths
        df_temp = self.df.group_by('seq_id').agg(pl.col(token+"_len").sum().alias(token+"_len"))
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')

        return self.df_seq

    def duration(self):
        # Calculate the sequence duration for each seq_id as the difference between max and min timestamps
        df_temp = self.df.group_by('seq_id').agg(
            (pl.col('m_timestamp').max() - pl.col('m_timestamp').min()).alias('duration'),
            (pl.col('m_timestamp').max() - pl.col('m_timestamp').min()).dt.total_seconds().alias('duration_sec')
        )
        # Join this result with df_sequences on seq_id
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')
        return self.df_seq

    def eve_len(self):
        # Count the number of rows for each seq_id
        df_temp = self.df.group_by('seq_id').agg(
            eve_len_max=pl.col('e_chars_len').max(),
            eve_len_min=pl.col('e_chars_len').min(),
            eve_len_avg=pl.col('e_chars_len').mean(),
            eve_len_med=pl.col('e_chars_len').median(),
            eve_len_over1=(pl.col('e_chars_len') > 1).sum()
        )
        # Join this result with df_sequences on seq_id
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')
        return self.df_seq

    def embeddings(self, embedding_column="e_bert_emb"):
        # Aggregate by averaging the embeddings for each sequence (seq_id)
        df_temp = self.df.select(pl.col("seq_id"), pl.col(embedding_column).list.to_struct()).unnest(embedding_column)
        df_temp = df_temp.group_by('seq_id').mean()
        df_temp = df_temp.select(pl.col("seq_id"), pl.concat_list(pl.col("*").exclude("seq_id")).alias(embedding_column))
        # Join this result with df_sequences on seq_id
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')
        return self.df_seq

    def next_event_prediction(self, event_col="e_event_drain_id"):
        if event_col not in self.df_seq.columns:  # Ensure events are present otherwise no nep can be done
            self.events(event_col)
        nepn = NextEventPredictionNgram()
        # Create model and score with same data.
        # This follows the enhancer logic that there is no splitting. In AD we operate with splits
        # Also sequence parsing is not done with splits so this follows the same logic
        seq_data = self.df_seq[event_col].to_list()
        nepn.create_ngram_model(seq_data)
        # predicted events
        preds, correct,  s_abs, spn_sum, spn_max = nepn.predict_list(seq_data)
        self.df_seq = self.df_seq.with_columns(nep_predict=pl.Series(preds),
                                               nep_corr=pl.Series(correct),
                                               nep_abs=pl.Series(s_abs),
                                               nep_prob_nsum=pl.Series(spn_sum),
                                               nep_prob_nmax=pl.Series(spn_max))
        # Add average, max, and min columns for nep_corr
        # This info is already in nep_prob_nmax with 1 being correct and <1 being zeros
        # self.df_seq = self.df_seq.with_columns(pl.col("nep_corr").list.mean().alias("nep_corr_avg"))
        # self.df_seq = self.df_seq.with_columns(pl.col("nep_corr").list.max().alias("nep_corr_max"))
        # self.df_seq = self.df_seq.with_columns(pl.col("nep_corr").list.min().alias("nep_corr_min"))
        # Add average, max, and min columns for nep_abs
        self.df_seq = self.df_seq.with_columns(pl.col("nep_abs").list.mean().alias("nep_abs_avg"))
        self.df_seq = self.df_seq.with_columns(pl.col("nep_abs").list.max().alias("nep_abs_max"))
        self.df_seq = self.df_seq.with_columns(pl.col("nep_abs").list.min().alias("nep_abs_min"))
        # Add average, max, and min columns for nep_prob_nsum
        # For prediction it looks as nep_prob_nmax is superior
        # self.df_seq = self.df_seq.with_columns(pl.col("nep_prob_nsum").list.mean().alias("nep_prob_nsum_avg"))
        # self.df_seq = self.df_seq.with_columns(pl.col("nep_prob_nsum").list.max().alias("nep_prob_nsum_max"))
        # self.df_seq = self.df_seq.with_columns(pl.col("nep_prob_nsum").list.min().alias("nep_prob_nsum_min"))
        # Add average, max, and min columns for nep_prob_nmax
        self.df_seq = self.df_seq.with_columns(pl.col("nep_prob_nmax").list.mean().alias("nep_prob_nmax_avg"))
        self.df_seq = self.df_seq.with_columns(pl.col("nep_prob_nmax").list.max().alias("nep_prob_nmax_max"))
        self.df_seq = self.df_seq.with_columns(pl.col("nep_prob_nmax").list.min().alias("nep_prob_nmax_min"))

        self._perplexity(probab_col="nep_prob_nmax")
        return self.df_seq

    def _perplexity(self, probab_col="nep_prob_nmax"):
        self.df_seq = self.df_seq.with_columns(pl.col(probab_col).list.eval(pl.element().log()) #Log of probabilties in a list
                                    .list.mean() #Average of probs
                                    .mul(-1).exp().alias(probab_col + "_perp")) #flip sign and exponent
