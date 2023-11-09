import polars as pl
import drain3 as dr
import parsers.lenma.lenma_template as lmt
from parsers.pyspell.spell import lcsmap, lcsobj
import hashlib
#Lazy import inside the method. 
#from .bertembedding import BertEmbeddings
import os

# Drain.ini default regexes
# No lookahead or lookbedinde so reimplemented with capture groups. Still problem with overlaps See
# https://docs.rs/regex/latest/regex/
# https://stackoverflow.com/questions/57497045/how-to-get-overlapping-regex-captures-in-rust
# Orig:     BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_200811092030_0001_m_000590_0/part-00590. blk_-1727475099218615100
# After 1st BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_<NUM>_0001_m_<NUM>_0/part-<NUM>. blk_<SEQ>'
# After 2nd BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_<NUM>_<NUM>_m_<NUM>_<NUM>/part-<NUM>. blk_<SEQ>'
masking_patterns_drain = [
    ("${start}<ID>${end}", r"(?P<start>[^A-Za-z0-9]|^)(([0-9a-f]{2,}:){3,}([0-9a-f]{2,}))(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<IP>${end}", r"(?P<start>[^A-Za-z0-9]|^)(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<SEQ>${end}", r"(?P<start>[^A-Za-z0-9]|^)([0-9a-f]{6,} ?){3,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<SEQ>${end}", r"(?P<start>[^A-Za-z0-9]|^)([0-9A-F]{4} ?){4,}(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(0x[a-f0-9A-F]+)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)([\-\+]?\d+)(?P<end>[^A-Za-z0-9]|$)"),
    ("${cmd}<CMD>", r"(?P<cmd>executed cmd )(\".+?\")")
]


class EventLogEnhancer:
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
    def trigrams(self, column="m_message"):
        self._handle_prerequisites([column])
        if "e_cgrams" not in self.df.columns:
            self.df = self.df.with_columns(
                pl.col(column).map_elements(
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
            current_script_path = os.path.abspath(__file__)
            current_script_directory = os.path.dirname(current_script_path)
            drain3_ini_location =  os.path.join(current_script_directory, '../parsers/drain3/')
            if drain_masking:
                dr.template_miner.config_filename = os.path.join(drain3_ini_location,'drain3.ini') #TODO fix the path relative
                self.tm = dr.TemplateMiner()
                self.df = self.df.with_columns(
                    message_trimmed=pl.col("m_message").str.split("\n").list.first()
                )
                self.df = self.df.with_columns(
                    drain=pl.col("message_trimmed").map_elements(lambda x: self.tm.add_log_message(x)))
            else:
                if "e_message_normalized" not in self.df.columns:
                    self.normalize()
                dr.template_miner.config_filename =os.path.join(drain3_ini_location, 'drain3_no_masking.ini') #drain3_no_masking.ini'  #TODO fix the path relative
                self.tm = dr.TemplateMiner()
                self.df = self.df.with_columns(
                    drain=pl.col("e_message_normalized").map_elements(lambda x: self.tm.add_log_message(x)))

            self.df = self.df.with_columns(
                e_event_id=pl.lit("e") + pl.col("drain").struct.field("cluster_id").cast(pl.Utf8),
                # extra thing ensure we e1 e2 instead of 1 2
                e_template=pl.col("drain").struct.field("template_mined"))
            self.df = self.df.drop("drain")  # Drop the dictionary produced by drain. Event_id and template are the most important.
            # tm.drain.print_tree()
        return self.df
    
    #https://github.com/keiichishima/templateminer
    def parse_lenma(self, masking=True, reparse=False):
        self._handle_prerequisites(["e_words"])
        if reparse or "e_event_lenma_id" not in self.df.columns:

            self.lenma_tm = lmt.LenmaTemplateManager(threshold=0.9)
            self.df = self.df.with_row_count()
            self.df = self.df.with_columns(
                lenma_obj=pl.struct(["e_words", "row_nr"])
                .map_elements(lambda x: self.lenma_tm.infer_template(x["e_words"], x["row_nr"])))
            def extract_id(obj):
                template_str = " ".join(obj.words)
                eid = hashlib.md5(template_str.encode("utf-8")).hexdigest()[0:8]   
                return {'eid':eid, 'template_str':template_str}

            self.df = self.df.with_columns(
                lenma_info= pl.col("lenma_obj").map_elements(lambda x:extract_id(x))
            )
            self.df = self.df.with_columns(
                e_event_lenma_id = pl.col("lenma_info").struct.field("eid"),
                e_template_lenma = pl.col("lenma_info").struct.field("template_str"))
            self.df = self.df.drop(["lenma_obj", "lenma_info", "row_nr"])
        return self.df

    #https://github.com/bave/pyspell/
    def parse_spell(self, masking=True, reparse=False):
        self._handle_prerequisites(["m_message"])
        if reparse or "e_event_spell_id" not in self.df.columns:
            if "e_message_normalized" not in self.df.columns:
                self.normalize()
            self.spell = lcsmap(r'\s+')
            self.df = self.df.with_columns(
                spell_obj=pl.col("e_message_normalized")
                .map_elements(lambda x: self.spell.insert(x)))

            def extract_id(obj):
                template_str = " ".join(obj._lcsseq)
                eid = hashlib.md5(template_str.encode("utf-8")).hexdigest()[0:8]   
                return {'eid':eid, 'template_str':template_str}

            self.df = self.df.with_columns(
                 spell_info= pl.col("spell_obj").map_elements(lambda x:extract_id(x))
            )
            self.df = self.df.with_columns(
                e_event_spell_id = pl.col("spell_info").struct.field("eid"),
                e_template_spell = pl.col("spell_info").struct.field("template_str"))
            self.df = self.df.drop(["spell_obj", "spell_info"])
        return self.df

    def create_neural_emb(self):
        self._handle_prerequisites(["m_message"])
        if "e_bert_emb" not in self.df.columns:
            #Lazy import only if needed
            from parsers.bert.bertembedding import BertEmbeddings
            if "e_message_normalized" not in self.df.columns:
                self.normalize()
            self.bert_emb_gen = BertEmbeddings(bertmodel="albert")
            message_trimmed_list = self.df['e_message_normalized'].to_list()
            message_trimmed_emb_tensor = self.bert_emb_gen.create_bert_emb(message_trimmed_list)
            # Convert the eager tensor to a NumPy array
            message_trimmed_emb_list = message_trimmed_emb_tensor.numpy()
            bert_emb_col_df = pl.DataFrame({
                'e_bert_emb': message_trimmed_emb_list
            })

            self.df = self.df.hstack(bert_emb_col_df)
        return self.df

    def length(self):
        self._handle_prerequisites(["m_message"])
        if "e_message_len" not in self.df.columns:
            self.df = self.df.with_columns(
                e_message_len_char=pl.col("m_message").str.n_chars(),
                e_message_len_lines=pl.col("m_message").str.count_matches(r"(\n|\r|\r\n)")
            )
        return self.df

    def normalize(self, regexs=masking_patterns_drain, to_lower=False, twice=True):

        # base_code = 'self.df = self.df.with_columns(e_message_normalized = pl.col("m_message").str.split("\\n").list.first()'
        base_code = 'self.df.with_columns(e_message_normalized = pl.col("m_message").str.split("\\n").list.first()'

        if to_lower:
            base_code += '.str.to_lowercase()'

        # Generate the replace_all chain
        # TODO We need to duplicate everything otherwise we get only every other replacement in 
        #"Folder_0012_2323_2324" -> After first replacement we get Folder Folder_<NUM>_2323_<NUM>
        #After second replacement we get  Folder_<NUM>_<NUM>_<NUM>. This is ugly but due to Crate limitations
        # https://docs.rs/regex/latest/regex/
        # https://stackoverflow.com/questions/57497045/how-to-get-overlapping-regex-captures-in-rust
        # Orig:     BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_200811092030_0001_m_000590_0/part-00590. blk_-1727475099218615100
        # After 1st BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_<NUM>_0001_m_<NUM>_0/part-<NUM>. blk_<SEQ>'
        # After 2nd BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_<NUM>_<NUM>_m_<NUM>_<NUM>/part-<NUM>. blk_<SEQ>'
        #Longer explanation Overlapping Matches: The regex crate does not find overlapping matches by default. If your text has numbers that are immediately adjacent to each other with only a non-alphanumeric separator (which is consumed by the start or end group), the regex engine won't match the second number because the separator is already consumed by the first match.
        for key, pattern in regexs:
            replace_code = f'.str.replace_all(r"{pattern}", "{key}")'
            base_code += replace_code
            if twice:
                base_code += replace_code

        base_code += ')'
        self.df = eval(base_code)
        return self.df
        # print (base_code)
        # return base_code


class SequenceEnhancer:
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

    def events(self, event_col = "e_event_id"):
        # Aggregate event ids into a list for each seq_id
        df_temp = self.df.group_by('seq_id').agg(pl.col(event_col).alias(event_col))
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
    
    def embeddings(self, embedding_column="e_bert_emb"):
        # Aggregate by averaging the embeddings for each sequence (seq_id)
        df_temp = self.df.select(pl.col("seq_id"), pl.col(embedding_column).list.to_struct()).unnest(embedding_column)
        df_temp = df_temp.group_by('seq_id').mean()
        df_temp = df_temp.select(pl.col("seq_id"),pl.concat_list(pl.col("*").exclude("seq_id")).alias(embedding_column))
        # Join this result with df_sequences on seq_id
        self.df_sequences = self.df_sequences.join(df_temp, on='seq_id')
        return self.df_sequences
