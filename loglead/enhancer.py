import polars as pl


import hashlib
#Lazy import inside the method. 
#from .bertembedding import BertEmbeddings
import os
import loglead.next_event_prediction as nep

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
            self.df = self.df.with_columns(
                e_words_len = pl.col("e_words").list.lengths(),
            )
        else:
            print("e_words already found")
        return self.df

    # Function-based enricher to extract alphanumeric tokens from messages
    def alphanumerics(self, column="m_message"):
        self._handle_prerequisites([column])
        if "e_alphanumerics" not in self.df.columns:
            self.df = self.df.with_columns(
                pl.col(column).str.extract_all(r"[a-zA-Z\d]+").alias("e_alphanumerics")
            )
            self.df = self.df.with_columns(
                e_alphanumerics_len = pl.col("e_alphanumerics").list.lengths(),
            )
        return self.df

    # Function-based enricher to create trigrams from messages
    # Trigrams enrichment is slow 1M lines in 40s.
    # Trigram flag to be removed after this is fixed.
    # https://github.com/pola-rs/polars/issues/10833
    # https://github.com/pola-rs/polars/issues/10890
    def trigrams(self, column="m_message"):
        self._handle_prerequisites([column])
        if "e_trigrams" not in self.df.columns:
            self.df = self.df.with_columns(
                pl.col(column).map_elements(
                    lambda mes: self._create_cngram(message=mes, ngram=3), return_dtype=pl.List(pl.Utf8)).alias("e_trigrams")
            )
            self.df = self.df.with_columns(
                e_trigrams_len = pl.col("e_trigrams").list.lengths()
            )
        return self.df

    def _create_cngram(self, message, ngram=3):
        if ngram <= 0:
            return []
        return [message[i:i + ngram] for i in range(len(message) - ngram + 1)]

    # Enrich with drain parsing results
    def parse_drain(self, field = "e_message_normalized", drain_masking=False, reparse=False):
        self._handle_prerequisites([field])
        if reparse or "e_event_drain_id" not in self.df.columns:
            import drain3 as dr
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
            return_dtype = pl.Struct([
                pl.Field("change_type", pl.Utf8),
                pl.Field("cluster_id", pl.Int64),
                pl.Field("cluster_size", pl.Int64),
                pl.Field("template_mined", pl.Utf8),
                pl.Field("cluster_count", pl.Int64)
            ])
            if drain_masking:
                dr.template_miner.config_filename = os.path.join(drain3_ini_location,'drain3.ini') #TODO fix the path relative
                tm = dr.TemplateMiner()
                self.df = self.df.with_columns(
                    message_trimmed=pl.col("m_message").str.split("\n").list.first()
                )
                self.df = self.df.with_columns(
                    drain=pl.col("message_trimmed").map_elements(lambda x: tm.add_log_message(x), return_dtype=return_dtype))
            else:
                #if "e_message_normalized" not in self.df.columns:
                #    self.normalize()
                dr.template_miner.config_filename =os.path.join(drain3_ini_location, 'drain3_no_masking.ini') #drain3_no_masking.ini'  #TODO fix the path relative
                tm = dr.TemplateMiner()
                self.df = self.df.with_columns(
                    drain=pl.col(field).map_elements(lambda x: tm.add_log_message(x), return_dtype=return_dtype))

            self.df = self.df.with_columns(
                # extra letter to ensure we e1 e2 instead of 1 2
                e_event_drain_id=pl.lit("e") + pl.col("drain").struct.field("cluster_id").cast(pl.Utf8),
                e_template=pl.col("drain").struct.field("template_mined"))
            self.df = self.df.drop("drain")  # Drop the dictionary produced by drain. Event_id and template are the most important.
            # tm.drain.print_tree()
        return self.df 
    
        
    def parse_brain(self, field = "e_message_normalized",  masking=True, reparse=False):
        self._handle_prerequisites([field])
        if reparse or "e_event_brain_id" not in self.df.columns:
            if "e_event_brain_id" in self.df.columns:
                self.df = self.df.drop("e_event_brain_id")
            import parsers.Brain.Brain as brain
            brain_parser = brain.LogParser(messages=self.df[field])
            brain_parser.parse() 
            df_new = brain_parser.df_log.select(pl.col("EventId").alias("e_event_brain_id"))
            self.df = pl.concat([self.df, df_new], how="horizontal")
        return self.df

    def parse_ael(self,field = "e_message_normalized",  masking=True, reparse=False):
        self._handle_prerequisites([field])
        if reparse or "e_event_ael_id" not in self.df.columns:
            if "e_event_ael_id" in self.df.columns:
                self.df = self.df.drop("e_event_ael_id")
            import parsers.AEL.AEL as ael
            ael_parser = ael.LogParser(messages=self.df[field])
            ael_parser.parse() 
            df_new = ael_parser.df_log.select(pl.col("EventId").alias("e_event_ael_id"))
            self.df = pl.concat([self.df, df_new], how="horizontal")
        return self.df

    #New parser not yet released to public. Coming early 2024
    def parse_tip(self, field = "e_message_normalized",  masking=True, reparse=False):
        self._handle_prerequisites([field])
        if reparse or "e_event_tip_id" not in self.df.columns:
            if "e_event_tip_id" in self.df.columns:
                self.df = self.df.drop("e_event_tip_id")
            import tipping as tip #See https://pypi.org/project/tipping/
            if "row_nr" in self.df.columns:
                self.df = self.df.drop("row_nr")
            self.df = self.df.with_row_count()
            tipping_clusters = tip.parse(self.df[field], return_templates=False, return_masks=False)
            df_new = pl.DataFrame(
                {
                    "e_event_tip_id": tipping_clusters[0],
                }
            )
            df_new = df_new.with_columns(
                e_event_tip_id=pl.when(pl.col("e_event_tip_id").is_null())
                .then(pl.lit("e_null"))
                .otherwise(pl.lit("e") + pl.col("e_event_tip_id").cast(pl.Utf8))
            )
            self.df = pl.concat([self.df, df_new], how="horizontal")
        return self.df
    
    def parse_iplom(self, field = "e_message_normalized",  masking=True, reparse=False, CT=0.35, PST=0, lower_bound=0.1):
        self._handle_prerequisites([field])
        if reparse or "e_event_iplom_id" not in self.df.columns:
            if "e_event_iplom_id" in self.df.columns:
                self.df = self.df.drop("e_event_iplom_id")
            import parsers.iplom.IPLoM as iplom
            if "row_nr" in self.df.columns:
                self.df = self.df.drop("row_nr")
            self.df = self.df.with_row_count()
            #TODO Storing each parser in self might eat a lot of memeory 
            iplom_parser = iplom.LogParser(messages=self.df[field], CT=CT,PST=PST,lowerBound=lower_bound)#FST not implemented
            iplom_parser.parse()
            df_output = pl.DataFrame({
                "row_nr": [row[0] for row in iplom_parser.output],
                "e_event_iplom_id": [row[1] for row in iplom_parser.output]
            })
            #Trying to prevent nulls in iplom. There should not be any TODO check if this is needed
            df_output = df_output.with_columns(
                e_event_iplom_id=pl.when(pl.col("e_event_iplom_id").is_null())
                .then(pl.lit("e_null"))
                .otherwise(pl.col("e_event_iplom_id"))
            )
            #print(f'Iplom NULL count {df_output["e_event_iplom_id"].null_count()}')
            df_output = df_output.with_columns(df_output.get_column("row_nr").cast(pl.UInt32).alias("row_nr"))
            self.df = self.df.join(df_output, on="row_nr", how="left")
        return self.df


    #Faster version of IPLoM coming in 2024
    def parse_pliplom(self, field = "e_message_normalized",  masking=True, reparse=False, CT=0.35, FST=0, PST=0,lower_bound=0.1, single_outlier_event=True):
        self._handle_prerequisites(["e_words"]) #Check word split method https://github.com/logpai/logparser/blob/main/logparser/IPLoM/IPLoM.py#L154
        if reparse or "e_event_plimplom_id" not in self.df.columns:
            if "e_event_plimplom_id" in self.df.columns:
                self.df = self.df.drop("e_event_pliplom_id")
            if "row_nr" in self.df.columns:
                self.df = self.df.drop("row_nr")
            self.df = self.df.with_row_count()
            import parsers.pl_iplom.pl_iplom as pl_iplom
            pliplom_parser = pl_iplom.IPLoM(self.df, CT=CT, FST=FST, PST=PST,lower_bound=lower_bound, single_outlier_event=single_outlier_event)
            df_new = pliplom_parser.parse()
            #df_new = plimplom_parser.merge_partitions_to_dataframe()
            df_new = df_new.select([
                pl.col("row_nr"),
                pl.col("event_id").alias("e_event_pliplom_id")
            ])
            self.df = self.df.join(df_new, on="row_nr", how="left")
        return self.df

    #https://github.com/keiichishima/templateminer
    def parse_lenma(self, field = "e_message_normalized",  masking=True, reparse=False):
        self._handle_prerequisites(["e_words"])
        if reparse or "e_event_lenma_id" not in self.df.columns:
            import parsers.lenma.lenma_template as lmt
            lenma_tm = lmt.LenmaTemplateManager(threshold=0.9)
            if "row_nr" in self.df.columns:
                self.df = self.df.drop("row_nr")
            self.df = self.df.with_row_count()
            self.df = self.df.with_columns(
                lenma_obj=pl.struct(["e_words", "row_nr"])
                .map_elements(lambda x:lenma_tm.infer_template(x["e_words"], x["row_nr"]), return_dtype=pl.Object))
            def extract_id(obj):
                template_str = " ".join(obj.words)
                eid = hashlib.md5(template_str.encode("utf-8")).hexdigest()[0:8]   
                return {'eid':eid, 'template_str':template_str}
            return_dtype = pl.Struct([
                pl.Field("eid", pl.Utf8),
                pl.Field("template_str", pl.Utf8)
            ])
            self.df = self.df.with_columns(
                lenma_info= pl.col("lenma_obj").map_elements(lambda x:extract_id(x), return_dtype=return_dtype)
            )
            self.df = self.df.with_columns(
                e_event_lenma_id = pl.col("lenma_info").struct.field("eid"),
                e_template_lenma = pl.col("lenma_info").struct.field("template_str"))
            self.df = self.df.drop(["lenma_obj", "lenma_info", "row_nr"])
        return self.df

    #https://github.com/bave/pyspell/
    def parse_spell(self, field = "e_message_normalized",  masking=True, reparse=False):
        self._handle_prerequisites([field])
        if reparse or "e_event_spell_id" not in self.df.columns:
            from parsers.pyspell.spell import lcsmap
            #if "e_message_normalized" not in self.df.columns:
            #    self.normalize()
            spell = lcsmap(r'\s+')
            self.df = self.df.with_columns(
                spell_obj=pl.col(field)
                    .map_elements(lambda x: spell.insert(x), return_dtype=pl.Object))

            def extract_id(obj):
                template_str = " ".join(obj._lcsseq)
                eid = hashlib.md5(template_str.encode("utf-8")).hexdigest()[0:8]   
                return {'eid':eid, 'template_str':template_str}

            return_dtype = pl.Struct([
                pl.Field("eid", pl.Utf8),
                pl.Field("template_str", pl.Utf8)
            ])
            self.df = self.df.with_columns(
                 spell_info= pl.col("spell_obj").map_elements(lambda x:extract_id(x),return_dtype=return_dtype)
            )
            self.df = self.df.with_columns(
                e_event_spell_id = pl.col("spell_info").struct.field("eid"),
                e_template_spell = pl.col("spell_info").struct.field("template_str"))
            self.df = self.df.drop(["spell_obj", "spell_info"])
        return self.df

    def create_neural_emb(self, field="e_message_normalized"):
        self._handle_prerequisites([field])
        if "e_bert_emb" not in self.df.columns:
            #Lazy import only if needed
            from parsers.bert.bertembedding import BertEmbeddings
            #if "e_message_normalized" not in self.df.columns:
            #    self.normalize()
            self.bert_emb_gen = BertEmbeddings(bertmodel="albert")
            message_trimmed_list = self.df[field].to_list()
            message_trimmed_emb_tensor = self.bert_emb_gen.create_bert_emb(message_trimmed_list)
            # Convert the eager tensor to a NumPy array
            message_trimmed_emb_list = message_trimmed_emb_tensor.numpy()
            bert_emb_col_df = pl.DataFrame({
                'e_bert_emb': message_trimmed_emb_list
            })

            self.df = self.df.hstack(bert_emb_col_df)
        return self.df

    def length(self, column="m_message"):
        self._handle_prerequisites(["m_message"])
        if "e_chars_len" not in self.df.columns:
            self.df = self.df.with_columns(
                e_chars_len=pl.col(column).str.n_chars(),
                e_lines_len=pl.col(column).str.count_matches(r"(\n|\r|\r\n)"),
                e_event_id_len = 1 #Messages are always one event. Added to simplify code later on. 
 
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
        
    
    def item_cumsum(self, column="e_message_normalized", chronological_order=True, ano_only=True, unique_only=True):
        self._handle_prerequisites([column, 'm_timestamp'])
        if ano_only:
            self._handle_prerequisites(['anomaly'])

        if chronological_order:
            self.df = self.df.sort('m_timestamp')

        # Initial condition by unique_only
        condition = pl.col(column).is_first_distinct() if unique_only else pl.lit(True)

        # Take in ano_only if required
        if ano_only:
            condition = condition & pl.col('anomaly')

        # Generate the dynamic column name based on the parameters
        column_name = 'cumu_items_'
        column_name += 'un' if unique_only else ''
        column_name += 'an' if ano_only else ''

        # In my tests cumsum needs a column in the table
        self.df = self.df.with_columns(condition.cast(pl.Int32).alias('count_support'))
        self.df = self.df.with_columns(
            pl.col('count_support').cumsum().alias(column_name)
        )
        self.df = self.df.drop('count_support')

        return self.df

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
        #We used median as time stamp for sequence
        df_temp = self.df.group_by('seq_id').agg(pl.col('m_timestamp').median().alias('time_stamp'))
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')
        return self.df_seq


    def seq_len(self):
        # Count the number of rows for each seq_id
        df_temp = self.df.group_by('seq_id').agg(pl.count().alias('seq_len'))
        # Join this result with df_sequences on seq_id
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')
        #Add an alias  that is compatible with the token len naming. 
        #TODO Should be in events() function
        self.df_seq = self.df_seq.with_columns(self.df_seq['seq_len'].alias('e_event_id_len'))

        return self.df_seq

    def events(self, event_col = "e_event_drain_id"):
        # Aggregate event ids into a list for each seq_id
        df_temp = self.df.group_by('seq_id').agg(pl.col(event_col).alias(event_col))
        # Join this result with df_sequences on seq_id
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')
        return self.df_seq


    def tokens(self, token="e_words"):
        #df_temp = self.df.group_by('seq_id').agg(pl.col(token).flatten().alias(token))
        #Same as above but the above crashes due to out of memory problems. We might need this fix also in other rows
        df_temp = self.df.select("seq_id", token).explode(token).group_by('seq_id').agg(pl.col(token))
        # Join this result with df_sequences on seq_id
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')

        #lengths
        df_temp = self.df.group_by('seq_id').agg(pl.col(token+"_len").sum().alias(token+"_len"))
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')

        return self.df_seq
    
    def duration(self):
        # Calculate the sequence duration for each seq_id as the difference between max and min timestamps
        df_temp = self.df.group_by('seq_id').agg(
            (pl.col('m_timestamp').max() - pl.col('m_timestamp').min()).alias('duration'),
            (pl.col('m_timestamp').max() - pl.col('m_timestamp').min()).dt.seconds().alias('duration_sec')
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
        df_temp = df_temp.select(pl.col("seq_id"),pl.concat_list(pl.col("*").exclude("seq_id")).alias(embedding_column))
        # Join this result with df_sequences on seq_id
        self.df_seq = self.df_seq.join(df_temp, on='seq_id')
        return self.df_seq
        
    def next_event_prediction (self, event_col = "e_event_drain_id"):
        if event_col not in self.df_seq.columns:#Ensure events are present otherwise no nep can be done
            self.events(event_col)
        nepn = nep.NextEventPredictionNgram()
        #Create model and score with same data.
        #This follows the enchancer logic that there is no splitting. In AD we operate with splits  
        #Also sequence parsing is not done with splits so this follows the same logic
        seq_data = self.df_seq[event_col].to_list()
        nepn.create_ngram_model(seq_data)
        #predicted events
        preds, correct,  s_abs, spn_sum, spn_max = nepn.predict_list(seq_data)
        self.df_seq = self.df_seq.with_columns(nep_predict = pl.Series(preds), 
                                    nep_corr= pl.Series(correct),
                                    nep_abs = pl.Series(s_abs),
                                    nep_prob_nsum = pl.Series(spn_sum),
                                    nep_prob_nmax = pl.Series(spn_max))
        # Add average, max, and min columns for nep_corr
        # This info is already in nep_prob_nmax with 1 being correct and <1 being zeros
        #self.df_seq = self.df_seq.with_columns(pl.col("nep_corr").list.mean().alias("nep_corr_avg"))
        #self.df_seq = self.df_seq.with_columns(pl.col("nep_corr").list.max().alias("nep_corr_max"))
        #self.df_seq = self.df_seq.with_columns(pl.col("nep_corr").list.min().alias("nep_corr_min"))
        # Add average, max, and min columns for nep_abs
        self.df_seq = self.df_seq.with_columns(pl.col("nep_abs").list.mean().alias("nep_abs_avg"))
        self.df_seq = self.df_seq.with_columns(pl.col("nep_abs").list.max().alias("nep_abs_max"))
        self.df_seq = self.df_seq.with_columns(pl.col("nep_abs").list.min().alias("nep_abs_min"))
        # Add average, max, and min columns for nep_prob_nsum
        # For prediction it looks as nep_prob_nmax is superior
        #self.df_seq = self.df_seq.with_columns(pl.col("nep_prob_nsum").list.mean().alias("nep_prob_nsum_avg"))
        #self.df_seq = self.df_seq.with_columns(pl.col("nep_prob_nsum").list.max().alias("nep_prob_nsum_max"))
        #self.df_seq = self.df_seq.with_columns(pl.col("nep_prob_nsum").list.min().alias("nep_prob_nsum_min"))
        # Add average, max, and min columns for nep_prob_nmax
        self.df_seq = self.df_seq.with_columns(pl.col("nep_prob_nmax").list.mean().alias("nep_prob_nmax_avg"))
        self.df_seq = self.df_seq.with_columns(pl.col("nep_prob_nmax").list.max().alias("nep_prob_nmax_max"))
        self.df_seq = self.df_seq.with_columns(pl.col("nep_prob_nmax").list.min().alias("nep_prob_nmax_min"))

        self._perplexity(probab_col = "nep_prob_nmax")
        return self.df_seq

    def _perplexity(self, probab_col = "nep_prob_nmax"): 
        self.df_seq = self.df_seq.with_columns(pl.col(probab_col).list.eval(pl.element().log()) #Log of probabilties in a list
                                    .list.mean() #Average of probs
                                    .mul(-1).exp().alias(probab_col + "_perp")) #flip sign and exponent
