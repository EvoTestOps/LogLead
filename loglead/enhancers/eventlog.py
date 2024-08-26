import hashlib

import polars as pl

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
#   ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)([a-f0-9A-F]+)(?P<end>[^A-Za-z0-9]|$)"),
#   ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(0x[a-f0-9A-F]+|[a-f0-9A-F]+)(?P<end>[^A-Za-z0-9]|$)"),
#   ("${start}<HEX>${end}", r"(?P<start>[^A-Za-z0-9]|^)(0x[a-f0-9A-F]{2,}(?:[a-f0-9A-F]{2})*|[a-f0-9A-F]{2}(?:[a-f0-9A-F]{2})*)(?P<end>[^A-Za-z0-9]|$)"),
    ("${start}<NUM>${end}", r"(?P<start>[^A-Za-z0-9]|^)([\-\+]?\d+)(?P<end>[^A-Za-z0-9]|$)"),
    ("${cmd}<CMD>", r"(?P<cmd>executed cmd )(\".+?\")")
]

__all__ = ['EventLogEnhancer']


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
                e_words_len = pl.col("e_words").list.len(),
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
                e_alphanumerics_len = pl.col("e_alphanumerics").list.len(),
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
                e_trigrams_len = pl.col("e_trigrams").list.len()
            )
        return self.df

    @staticmethod
    def _create_cngram(message, ngram=3):
        if ngram <= 0:
            return []
        return [message[i:i + ngram] for i in range(len(message) - ngram + 1)]

    # Enrich with drain parsing results
    def parse_drain(self, field = "e_message_normalized", drain_masking=False, reparse=False, templates=False):
        self._handle_prerequisites([field])
        if reparse or "e_event_drain_id" not in self.df.columns:
            # Drain returns dict
            # {'change_type': 'none',
            # 'cluster_id': 1,
            # 'cluster_size': 2,
            # 'template_mined': 'session closed for user root',
            # 'cluster_count': 1}
            # we store template for later use.

            # We might have multiline log message, i.e. log_message + stack trace.
            # Use only first line of log message for parsing
            return_dtype = pl.Struct([
                pl.Field("change_type", pl.Utf8),
                pl.Field("cluster_id", pl.Int64),
                pl.Field("cluster_size", pl.Int64),
                pl.Field("template_mined", pl.Utf8),
                pl.Field("cluster_count", pl.Int64)
            ])
            if drain_masking:
                from loglead.parsers import DrainTemplateMiner as tm
                self.df = self.df.with_columns(
                    message_trimmed=pl.col("m_message").str.split("\n").list.first()
                )
                self.df = self.df.with_columns(
                    drain=pl.col("message_trimmed").map_elements(lambda x: tm.add_log_message(x), return_dtype=return_dtype))
            else:
                #if "e_message_normalized" not in self.df.columns:
                #    self.normalize()
                from loglead.parsers import DrainTemplateMinerNoMasking as tm
                self.df = self.df.with_columns(
                    drain=pl.col(field).map_elements(lambda x: tm.add_log_message(x), return_dtype=return_dtype))

            if templates:
                self.df = self.df.with_columns(
                    # extra letter to ensure we get e1 e2 instead of 1 2
                    e_event_drain_id=pl.lit("e") + pl.col("drain").struct.field("cluster_id").cast(pl.Utf8),
                    e_event_drain_template=pl.col("drain").struct.field("template_mined"))
            else:
                self.df = self.df.with_columns(
                    # extra letter to ensure we get e1 e2 instead of 1 2
                    e_event_drain_id=pl.lit("e") + pl.col("drain").struct.field("cluster_id").cast(pl.Utf8))    
            self.df = self.df.drop("drain")  # Drop the dictionary produced by drain. Event_id and template are the most important.
            # tm.drain.print_tree()
        return self.df 
    
    def parse_brain(self, field = "e_message_normalized", reparse=False):
        self._handle_prerequisites([field])
        if reparse or "e_event_brain_id" not in self.df.columns:
            if "e_event_brain_id" in self.df.columns:
                self.df = self.df.drop("e_event_brain_id")

            from loglead.parsers import BrainParser
            brain_parser = BrainParser(messages=self.df[field])
            brain_parser.parse() 
            df_new = brain_parser.df_log.select(pl.col("EventId").alias("e_event_brain_id"))
            self.df = pl.concat([self.df, df_new], how="horizontal")
        return self.df

    def parse_ael(self,field = "e_message_normalized",  reparse=False):
        self._handle_prerequisites([field])
        if reparse or "e_event_ael_id" not in self.df.columns:
            if "e_event_ael_id" in self.df.columns:
                self.df = self.df.drop("e_event_ael_id")

            from loglead.parsers import AELParser
            ael_parser = AELParser(messages=self.df[field])
            ael_parser.parse() 
            df_new = ael_parser.df_log.select(pl.col("EventId").alias("e_event_ael_id"))
            self.df = pl.concat([self.df, df_new], how="horizontal")
        return self.df

    #See https://pypi.org/project/tipping/
    #and https://arxiv.org/abs/2408.00645 
    def parse_tip(self, field = "e_message_normalized", reparse=False, templates=False):
        self._handle_prerequisites([field])
        if reparse or "e_event_tip_id" not in self.df.columns:
            if "e_event_tip_id" in self.df.columns:
                self.df = self.df.drop("e_event_tip_id")
            import tipping as tip #See https://pypi.org/project/tipping/
            if "row_nr" in self.df.columns:
                self.df = self.df.drop("row_nr")
            self.df = self.df.with_row_index("row_nr", )
            tipping_clusters, tipping_masks, tipping_templates = tip.parse(self.df[field], return_templates=templates, return_masks=False)
            if templates:
                df_new = pl.DataFrame(
                    {
                        "e_event_tip_id": tipping_clusters,
                    }
                )
                #convert sets to lists
                tipping_templates = [list(s)[0] if s else None for s in tipping_templates]
                df_templates = pl.DataFrame({
                    "e_event_tip_id": range(len(tipping_templates)),
                    "e_event_tip_template": tipping_templates
                })
                df_new = df_new.join(df_templates, on="e_event_tip_id", how="left")
                df_new = df_new.with_columns(pl.col("e_event_tip_template").cast(pl.Utf8))
            else: 
                df_new = pl.DataFrame(
                    {
                        "e_event_tip_id": tipping_clusters,
                    }
                )
            df_new = df_new.with_columns(
                e_event_tip_id=pl.when(pl.col("e_event_tip_id").is_null())
                .then(pl.lit("e_null"))
                .otherwise(pl.lit("e") + pl.col("e_event_tip_id").cast(pl.Utf8))
            )
            self.df = pl.concat([self.df, df_new], how="horizontal")
        return self.df
    
    def parse_iplom(self, field = "e_message_normalized", reparse=False, CT=0.35, PST=0, lower_bound=0.1):
        self._handle_prerequisites([field])
        if reparse or "e_event_iplom_id" not in self.df.columns:
            if "e_event_iplom_id" in self.df.columns:
                self.df = self.df.drop("e_event_iplom_id")
            if "row_nr" in self.df.columns:
                self.df = self.df.drop("row_nr")
            self.df = self.df.with_row_index("row_nr", )
            from loglead.parsers import IPLoMParser
            #TODO Storing each parser in self might eat a lot of memeory
            iplom_parser = IPLoMParser(messages=self.df[field], CT=CT, PST=PST, lowerBound=lower_bound)#FST not implemented
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
    def parse_pliplom(self, field = "e_message_normalized",  reparse=False, CT=0.35, FST=0, PST=0,lower_bound=0.1, single_outlier_event=True):
        self._handle_prerequisites(["e_words"]) #Check word split method https://github.com/logpai/logparser/blob/main/logparser/IPLoM/IPLoM.py#L154
        if reparse or "e_event_plimplom_id" not in self.df.columns:
            if "e_event_plimplom_id" in self.df.columns:
                self.df = self.df.drop("e_event_pliplom_id")
            if "row_nr" in self.df.columns:
                self.df = self.df.drop("row_nr")
            self.df = self.df.with_row_index("row_nr", )
            from loglead.parsers import PL_IPLoMParser
            pliplom_parser = PL_IPLoMParser(self.df, CT=CT, FST=FST, PST=PST, lower_bound=lower_bound, single_outlier_event=single_outlier_event)
            df_new = pliplom_parser.parse()
            #df_new = plimplom_parser.merge_partitions_to_dataframe()
            df_new = df_new.select([
                pl.col("row_nr"),
                pl.col("event_id").alias("e_event_pliplom_id")
            ])
            self.df = self.df.join(df_new, on="row_nr", how="left")
        return self.df

    #https://github.com/keiichishima/templateminer
    def parse_lenma(self, field = "e_message_normalized",  reparse=False):
        self._handle_prerequisites(["e_words"])
        if reparse or "e_event_lenma_id" not in self.df.columns:
            from loglead.parsers import LenmaTemplateManager

            lenma_tm = LenmaTemplateManager(threshold=0.9)
            if "row_nr" in self.df.columns:
                self.df = self.df.drop("row_nr")
            self.df = self.df.with_row_index("row_nr", )
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
    def parse_spell(self, field = "e_message_normalized",  reparse=False):
        self._handle_prerequisites([field])
        if reparse or "e_event_spell_id" not in self.df.columns:
            from loglead.parsers import SpellParser
            #if "e_message_normalized" not in self.df.columns:
            #    self.normalize()
            spell = SpellParser(r'\s+')
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
            from loglead.parsers import BertEmbeddings
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
                e_chars_len=pl.col(column).str.len_chars(),
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

    def item_cumsum2(self, column="e_message_normalized", chronological_order=1, ano_only=True, unique_only=True, out_column=None):
        if out_column is None:
            out_column = column + "_cumsum"
        column_name = out_column
        self._handle_prerequisites([column, 'm_timestamp'])
        if ano_only:
            self._handle_prerequisites(['anomaly'])

        if chronological_order == 1:
            self.df = self.df.sort('m_timestamp')
        elif chronological_order ==-1: 
            self.df = self.df.sort('m_timestamp', descending = True)    

        # Initial condition by unique_only
        condition = pl.col(column).is_first_distinct() if unique_only else pl.lit(True)

        # Take in ano_only if required
        if ano_only:
            condition = condition & pl.col('anomaly')

        # In my tests cumsum needs a column in the table
        self.df = self.df.with_columns(condition.cast(pl.Int32).alias('count_support'))
        self.df = self.df.with_columns(
            pl.col('count_support').cum_sum().alias(column_name)
        )
        self.df = self.df.drop('count_support')

        return self.df

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
            pl.col('count_support').cum_sum().alias(column_name)
        )
        self.df = self.df.drop('count_support')

        return self.df

