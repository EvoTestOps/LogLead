import polars as pl
import drain3 as dr
from bertembedding import BertEmbeddings

# Drain.ini default regexes
# No lookahead or lookbedinde so reimplemented with capture groups
masking_patterns_drain = [
    ("${start}<ID>${end}", r"(?P<start>[^A-Za-z0-9]|^)(([0-9a-f]{2,}:){3,}([0-9a-f]{2,}))(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^A-Za-z0-9]|^)(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<SEQ>${end}", r"(?P<start>[^A-Za-z0-9]|^)([0-9a-f]{6,} ?){3,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<SEQ>${end}", r"(?P<start>[^A-Za-z0-9]|^)([0-9A-F]{4} ?){4,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(0x[a-f0-9A-F]+)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)([\-\+]?\d+)(?P<end>[^A-Za-z0-9]|$)"),
    ("${cmd}<CMD>", r"(?P<cmd>executed cmd )(\".+?\")")
]


class EventLogEnricher:
    def __init__(self, df):
        self.df = df

    # Helper function to check if all prerequisites exist
    def _prerequisites_exist(self, prerequisites):
        return all([col in self.df.columns for col in prerequisites])

    # Helper function to handle prerequisite check and raise exception if missing
    def _handle_prerequisites(self, prerequisites):
        if not self._prerequisites_exist(prerequisites):
            raise ValueError(f"Missing prerequisites for enrichment: {', '.join(prerequisites)}")

    # Function-based enricher to split messages into words
    def words(self, column="m_message"):
        self._handle_prerequisites([column])
        if "e_words" not in self.df.columns:
            self.df = self.df.with_columns(pl.col(column).str.split(by=" ").alias("e_words"))
        return self.df

    # Function-based enricher to extract alphanumeric tokens from messages
    def alphanumerics(self, column="m_message"):
        self._handle_prerequisites([column])
        if "e_alphanumerics" not in self.df.columns:
            self.df = self.df.with_columns(
                pl.col(column).str.extract_all(r"[a-zA-Z\d]+").alias("e_alphanumerics")
            )
        return self.df

    # Function-based enricher to create trigrams from messages
    # Trigrams enrichment is slow 1M lines in 40s.
    # Trigram flag to be removed after this is fixed.
    # https://github.com/pola-rs/polars/issues/10833
    # https://github.com/pola-rs/polars/issues/10890
    def trigrams(self):
        self._handle_prerequisites(["m_message"])
        if "e_cgrams" not in self.df.columns:
            self.df = self.df.with_columns(
                pl.col("m_message").map_elements(
                    lambda mes: self._create_cngram(message=mes, ngram=3)).alias("e_cgrams")
            )
        return self.df

    def _create_cngram(self, message, ngram=3):
        if ngram <= 0:
            return []
        return [message[i:i + ngram] for i in range(len(message) - ngram + 1)]

    # Enrich with drain parsing results
    def parse_drain(self, drain_masking=False, reparse=False):
        self._handle_prerequisites(["m_message"])
        if reparse or "e_event_id" not in self.df.columns:
            # Drain returns dict
            # {'change_type': 'none',
            # 'cluster_id': 1,
            # 'cluster_size': 2,
            # 'template_mined': 'session closed for user root',
            # 'cluster_count': 1}
            # we store template for later use.

            # We might have multiline log message, i.e. log_message + stack trace.
            # Use only first line of log message for parsing
            if drain_masking:
                dr.template_miner.config_filename = 'drain3.ini'
                self.tm = dr.TemplateMiner()
                self.df = self.df.with_columns(
                    message_trimmed=pl.col("m_message").str.split("\n").list.first()
                )
                self.df = self.df.with_columns(
                    drain=pl.col("message_trimmed").map_elements(lambda x: self.tm.add_log_message(x)))
            else:
                if "e_message_normalized" not in self.df.columns:
                    self.normalize()
                dr.template_miner.config_filename = 'drain3_no_masking.ini'
                self.tm = dr.TemplateMiner()
                self.df = self.df.with_columns(
                    drain=pl.col("e_message_normalized").map_elements(lambda x: self.tm.add_log_message(x)))

            self.df = self.df.with_columns(
                e_event_id=pl.lit("e") + pl.col("drain").struct.field("cluster_id").cast(pl.Utf8),
                # extra thing ensure we e1 e2 instead of 1 2
                e_template=pl.col("drain").struct.field("template_mined"))
            self.df = self.df.drop(
                "drain")  # Drop the dictionary produced by drain. Event_id and template are the most important.

            # tm.drain.print_tree()

        return self.df

    def create_neural_emb(self):
        self._handle_prerequisites(["m_message"])
        if "e_bert_emb" not in self.df.columns:
            # We might have multiline log message, i.e. log_message + stack trace.
            # Use only first line of log message for parsing

            #            self.df = self.df.with_columns(
            #                message_trimmed=pl.col("m_message").str.split("\n").list.first()
            #                .str.to_lowercase()
            #                .str.replace_all(r"[0-9\W_]", " ")
            #                .str.replace_all(r"\s+", " ")
            #                .str.replace_all(r"^\s+|\s+$", "")
            #            )
            if "e_message_normalized" not in self.df.columns:
                self.normalize()
            # create a BertEmbeddings class instance
            # MM: Do we need the generator later?
            # For example if predictions need to be done much after training
            # e.g. we train and then go to production where prediction take place.
            # if so then this should be
            # YQ: It depends on the situation. E.g., for my fusion model, I define it in '__main__',
            # to save the processing time, i.e., define one time and use it in all places
            self.bert_emb_gen = BertEmbeddings(bertmodel="albert")
            # bert_emb_gen = BertEmbeddings()
            # obtain bert sentence embedding
            # MM: Is it possible to do this map or map_elements as in line 70?
            # MM: Or is there too much performance hit?
            # MM: This makes unnessary copies from and to polars dataframe.
            # YQ: map or map_elements do the operation on "each element" of a column in the DataFrame.
            # YQ: We can create Bert embedding for "each element" and map it to a column, but it takes heavy computing resources
            # YQ: Because it fetch model and generate output again and again
            # YQ: You can see from the current bertembedding class, I feed all values on self.df['message_trimmed'] to Bert
            # YQ: This means we only fetch model and generate output one time.
            message_trimmed_list = self.df['e_message_normalized'].to_list()
            message_trimmed_emb_tensor = self.bert_emb_gen.create_bert_emb(message_trimmed_list)
            # Convert the eager tensor to a NumPy array
            message_trimmed_emb_list = message_trimmed_emb_tensor.numpy()
            bert_emb_col_df = pl.DataFrame({
                'e_bert_emb': message_trimmed_emb_list
            })

            self.df = self.df.hstack(bert_emb_col_df)
            # print(self.df["e_bert_emb"][1])
            # Albert, b_hadoop, 177592 entries, Time taken: 375.30 seconds
            # Base Bert, b_hadoop, 177592 entries, Time taken: 601.45 seconds
        return self.df

    def length(self):
        self._handle_prerequisites(["m_message"])
        if "e_message_len" not in self.df.columns:
            self.df = self.df.with_columns(
                e_message_len_char=pl.col("m_message").str.n_chars(),
                e_message_len_lines=pl.col("m_message").str.count_matches(r"(\n|\r|\r\n)")
            )
        return self.df

    def normalize(self, regexs=masking_patterns_drain, to_lower=False):

        # base_code = 'self.df = self.df.with_columns(e_message_normalized = pl.col("m_message").str.split("\\n").list.first()'
        base_code = 'self.df.with_columns(e_message_normalized = pl.col("m_message").str.split("\\n").list.first()'

        if to_lower:
            base_code += '.str.to_lowercase()'

        # Generate the replace_all chain
        for key, pattern in regexs:
            replace_code = f'.str.replace_all(r"{pattern}", "{key}")'
            base_code += replace_code

        base_code += ')'
        self.df = eval(base_code)
        return self.df
        # print (base_code)
        # return base_code


class SequenceEnricher:
    def __init__(self, df, df_sequences):
        self.df = df
        self.df_sequences = df_sequences

    def start_time(self):
        df_temp = self.df.group_by('seq_id').agg(pl.col('m_timestamp').min().alias('start_time'))
        self.df_sequences = self.df_sequences.join(df_temp, on='seq_id')
        return self.df_sequences

    def end_time(self):
        df_temp = self.df.group_by('seq_id').agg(pl.col('m_timestamp').max().alias('end_time'))
        self.df_sequences = self.df_sequences.join(df_temp, on='seq_id')
        return self.df_sequences

    def seq_len(self):
        # Count the number of rows for each seq_id
        df_temp = self.df.group_by('seq_id').agg(pl.count().alias('seq_len'))
        # Join this result with df_sequences on seq_id
        self.df_sequences = self.df_sequences.join(df_temp, on='seq_id')
        return self.df_sequences

    def events(self):
        # Aggregate event ids into a list for each seq_id
        df_temp = self.df.group_by('seq_id').agg(events=pl.col('e_event_id'))
        # Join this result with df_sequences on seq_id
        self.df_sequences = self.df_sequences.join(df_temp, on='seq_id')
        return self.df_sequences

    def tokens(self, token="e_words"):
        df_temp = self.df.group_by('seq_id').agg(pl.col(token).flatten().alias(token))
        # Join this result with df_sequences on seq_id
        self.df_sequences = self.df_sequences.join(df_temp, on='seq_id')
        return self.df_sequences

    def duration(self):
        # Calculate the sequence duration for each seq_id as the difference between max and min timestamps
        df_temp = self.df.group_by('seq_id').agg(
            (pl.col('m_timestamp').max() - pl.col('m_timestamp').min()).alias('duration'),
            (pl.col('m_timestamp').max() - pl.col('m_timestamp').min()).dt.seconds().alias('duration_sec')
        )
        # Join this result with df_sequences on seq_id
        self.df_sequences = self.df_sequences.join(df_temp, on='seq_id')
        return self.df_sequences

    def eve_len(self):
        # Count the number of rows for each seq_id
        df_temp = self.df.group_by('seq_id').agg(
            eve_len_max=pl.col('e_message_len_char').max(),
            eve_len_min=pl.col('e_message_len_char').min(),
            eve_len_avg=pl.col('e_message_len_char').mean(),
            eve_len_med=pl.col('e_message_len_char').median(),
            eve_len_over1=(pl.col('e_message_len_char') > 1).sum()
        )
        # Join this result with df_sequences on seq_id
        self.df_sequences = self.df_sequences.join(df_temp, on='seq_id')
        return self.df_sequences    
