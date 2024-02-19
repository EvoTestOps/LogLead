
import polars as pl 
import logging
logger = logging.getLogger(__name__)
#Bugs in logparser IPLOM
#No check that partitionL has IDs that are long enough. Default is 200. So if event is 201 long this will fail
#https://github.com/logpai/logparser/blob/main/logparser/IPLoM/IPLoM.py#L163
#This code is never executed. Should be self.para.PST != 0 not ==. If == then outlier block will always be empty
#https://github.com/logpai/logparser/blob/main/logparser/IPLoM/IPLoM.py#L437C12-L437C31 
# https://github.com/logpai/logparser/blob/main/logparser/IPLoM/IPLoM.py#L477  

class Partition:
    def __init__(self, df, len, split_trace):
        self.df = df
        #self.has_subpartitions = False
        self.subpartitions = []
        self.template = ""
        self.len = len
        self.split_trace = split_trace 

    def create_template(self):
        template_parts = []
        for col in self.df.columns:
            unique_values = self.df.select(col).unique()
            if len(unique_values) == 1:
                # If there's only one unique value, add it to the template
                value = unique_values[col][0]
                template_parts.append(str(value))
            else:
                # If there are multiple unique values, add a '*' to the template
                template_parts.append('<*>')
        
        # Join the template parts with spaces or any other delimiter as needed
        self.template = " ".join(template_parts)    

#TODO 
# 1) Log Template creation, DONE
# 2) check why we get no trace S1 S3. We should have atleast some        
# 2) Log Id Creation, 
# 3) Rolling log IDs and templates back to Enhancer
# how it is done on tipping
        
class IPLoM:
    def __init__(self, df, CT = 0.35, PST=0, FST=0,lower_bound = 0.1, single_outlier_event = True):
        self.df = df
        self.CT = CT #Cluster goodness threshold
        self.PST = PST #Partition Support Threshold
        self.partitions = []
        self.outlier_partitions = []
        self.FST = FST #File Support Threshold
        self.single_outlier_event = single_outlier_event
        self.lower_bound = lower_bound 

    def print_cluster_info(self):
        # Start analyzing from the top level, with depth 0
        self._print_cluster_info_recursive(self.partitions, depth=0)
        self._print_outlier_clusters()

    def _print_outlier_clusters(self):
        print ("Printing outlier clusters") 
        for outlier in self.outlier_partitions:
             outlier.create_template()
             print(f"Outlier cluster with length {outlier.len} with trace {outlier.split_trace} has {outlier.df.shape[0]} rows and template {outlier.template}")

    def _print_cluster_info_recursive(self, partitions, depth):
        # Loop through each partition at the current depth
        for partition in partitions:
            # Print details of the current partition
            prefix = "  " * depth  # Indentation to visualize depth
            if partition.df is None:
                print(f"{prefix}Cluster at depth {depth} with length {partition.len} with trace {partition.split_trace} has no data (df is null)")
            else:
                #print(f"{prefix}Cluster at depth {depth} has {partition.df.shape[0]} rows")
                partition.create_template()
                print(f"{prefix}Cluster at depth {depth} with length {partition.len} with trace {partition.split_trace} has {partition.df.shape[0]} rows and template {partition.template}")
            
            # If the partition has subpartitions, recursively analyze them
            if partition.subpartitions:
                print(f"{prefix}  It has {len(partition.subpartitions)} subpartitions:")
                self._print_cluster_info_recursive(partition.subpartitions, depth + 1)

    def merge_partitions_to_dataframe(self):
        df_list = []  # List to accumulate DataFrames

        def traverse_and_concat(partitions, parent_id):
            for i, partition in enumerate(partitions, start=1):
                current_id = f"{parent_id}e{i}"  # Construct a unique event_id for each partition
                if partition.df is not None:
                    partition.create_template()
                    # Create a dataframe with template and event_id columns for the current partition
                    df_parsed = partition.df.select("row_nr")
                    df_parsed = df_parsed.with_columns([
                        pl.lit(partition.template).alias('template'),
                        pl.lit(current_id).alias('event_id'),
                        pl.lit(partition.len).alias('event_len')
                    ])
                    df_list.append(df_parsed)  # Append to list instead of concatenating immediately
                # Recursively process subpartitions
                if partition.subpartitions:
                    traverse_and_concat(partition.subpartitions, current_id)

        def process_outlier_partitions():
            for i, outlier in enumerate(self.outlier_partitions, start=1):
                current_id = f"outlier_e" if self.single_outlier_event else f"outlier_e{i}"
                if outlier.df is not None:
                    outlier.create_template()
                    df_parsed = outlier.df.select("row_nr")
                    df_parsed = df_parsed.with_columns([
                        pl.lit("outlier").alias('template'),
                        pl.lit(current_id).alias('event_id'),
                        pl.lit(outlier.len).alias('event_len')
                    ])
                    df_list.append(df_parsed)  # Append to list

        # Reset self.acc_df to ensure it's empty before starting
        self.acc_df = pl.DataFrame()
        # Process regular and outlier partitions
        traverse_and_concat(self.partitions, "")
        process_outlier_partitions()  # Process outliers after handling regular partitions

        # Concatenate all DataFrames in the list if it's not empty
        if df_list:
            self.acc_df = pl.concat(df_list)
        return self.acc_df

    def parse(self):
        #Step 1
        self.s1_clust_by_message_length()
        #Step 2
        for i in range(len(self.partitions)):
            #Step 2
            logger.debug ("\n")
            self.s2_clust_by_token_pos(self.partitions[i])
            #Step 3
            self.s3_clust_by_bijection(self.partitions[i])


    def s1_clust_by_message_length(self):
        logger.debug ("s1 start")
        #STEP1 - Cluster (aggragete) logs based on word length
        df_aggre_s1 = self.df.select(pl.col("e_words_len")).unique()
        #Add events for each cluster
        #df_temp = self.df.select("e_words_len", "e_words").group_by('e_words_len').agg(pl.col("e_words").alias("events"))
        df_temp = self.df.select("e_words_len", "e_words", "row_nr").group_by('e_words_len').agg(
            pl.col("e_words").alias("events"),
            pl.col("row_nr").alias("row_nr"))

        df_aggre_s1 = df_aggre_s1.join(df_temp, on='e_words_len')
        # There is alternivate to count length from reduced cluster directly. 
        #self.df_iplom = self.df_iplom.get_column("events").list.len()
        df_temp = self.df.group_by('e_words_len').agg(pl.count().alias('part_len'))
        df_aggre_s1 = df_aggre_s1.join(df_temp, on='e_words_len')
        logger.debug(f"s1 end found {df_aggre_s1.shape[0]} len clusters")
        #Create dataframes for each partition
        #We iterate over rows of Dataframe.  
        for i in range(len(df_aggre_s1)):
            df_part = df_aggre_s1[i] 
            len_words = df_part['e_words_len'][0]
            part_len = df_part['part_len'][0]
            logger.debug (f"\nCreation parittions with {len_words} words and {part_len} events")
            df_part = df_aggre_s1[i].with_columns(pl.col("events", "row_nr")).explode("events", "row_nr")
            df_part = df_part.with_columns(pl.col("events").list.to_struct()).unnest("events")
            df_part = df_part.drop("e_words_len", "part_len")
            self.add_partition(Partition(df = df_part, len=len_words, split_trace="S1 "))
        return df_aggre_s1 #For easier debugging
    
    def add_partition(self, partition, parent_partition = None):
        if self.FST > 0 and partition.df.shape[0] / self.df.shape[0] < self.FST:
            self.outlier_partitions.append(partition)
        else:
            if parent_partition:
                if self.PST > 0 and partition.df.shape[0] / parent_partition.df.shape[0] < self.PST:
                    self.outlier_partitions.append(partition)
                else:
                    parent_partition.subpartitions.append(partition)
            else:        
                self.partitions.append(partition)
        #FST check
        
    def s2_clust_by_token_pos(self, partition):

        unique_counts = [len(partition.df[col].unique()) for col in partition.df.columns] # Also faster approx_n_unique() could be used if we need more speed
        #Min col counts
        min_count = min(unique_counts)
        min_unique_count, min_column_index = min((count, idx) for idx, count in enumerate(unique_counts))

        if (min_count>1):#Split based on word with fewest unique values must be greater than 1
            #partition.has_subpartitions = True
            row_dict = partition.df.partition_by(partition.df.columns[min_column_index], as_dict=True)
            #self.df_s2_dict = row_dict
            for key, dataframe in row_dict.items():
                logger.debug(f"s2 least frequent for word dataframe Appending: {key}")
                #partition.subpartitions.append(Partition(dataframe, len = partition.len,  split_trace=partition.split_trace+" S2"))
                self.add_partition(Partition(dataframe, len = partition.len,  split_trace=partition.split_trace+"S2 "), parent_partition=partition)
            partition.df = None #To save memory
        else: 
            self.df_s2_dict = None
        logger.debug (f"s2 end found min unique count is {min_count} and is found in col index {min_column_index} ")
        #return self.df_s2_dict
        #TODO ADD PST Stuff
    
    def _get_p1_p2 (self, unique_counts):
        if (len(unique_counts) > 2):
            from collections import Counter
            freqs = Counter(unique_counts)

            # Get the list of most common elements and remove 1 if present
            common_items = [(num, count) for num, count in freqs.most_common() if num != 1]

            # Check if there is a tie for the most frequent number
            if len(common_items) >= 2 and common_items[0][1] == common_items[1][1]:
                # There is a tie, select the two smallest numbers
                two_smallest_numbers = sorted(common_items, key=lambda x: x[0])[:2]

                # Extract numbers and their positions
                most_freq_num1, most_freq_num2 = two_smallest_numbers[0][0], two_smallest_numbers[1][0]
                positions1 = [i for i, num in enumerate(unique_counts) if num == most_freq_num1]
                positions2 = [i for i, num in enumerate(unique_counts) if num == most_freq_num2]
                # Assign p1 and p2 based on the count of positions1
                if len(positions1) >= 2:
                    p1 = positions1[0]
                    p2 = positions1[1]
                else:
                    p1 = positions1[0] if positions1 else None
                    p2 = positions2[0] if positions2 else None
                # Display the results for the two smallest numbers
                logger.debug("Two smallest most frequent numbers:", most_freq_num1, most_freq_num2)
                most_freq_count = common_items[0][1]
                logger.debug("Count:", most_freq_count)
                logger.debug("Positions of first number:", positions1)
                logger.debug("Positions of second number:", positions2)
            elif common_items:
                # No tie, or not enough elements for a tie. Select the most frequent number
                most_freq_num, most_freq_count = common_items[0]
                positions = [i for i, num in enumerate(unique_counts) if num == most_freq_num]
                p1 = positions[0]
                p2 = positions[1]
                # Display the results
                logger.debug("Most Frequent Number:", most_freq_num)
                logger.debug("Count:", most_freq_count)
                logger.debug("Positions:", positions)
            else:
                logger.debug("All elements are 1, no other frequent numbers.")
        elif (len(unique_counts) == 2):
            p1 = 0
            p2 = 1
        else:
            p1 = -1 
            p2 = 0
        logger.debug (f"P1 is {p1} P2 is {p2}")
        return p1, p2

    def s3_clust_by_bijection (self, partition):
        if partition.subpartitions:  # Checks if the subpartitions list is not empty
            logger.debug (f"S3 Found subpartition with {len(partition.subpartitions)} dataframes")
            for subpartition in partition.subpartitions:
                self.s3_clust_by_bijection(subpartition)
            return 
        else:
            logger.debug (f"S3 processing df with rows: {partition.df.shape[0]} with length {partition.len} with trace {partition.split_trace}")
        #S3.1 figure which columns to select for P1 and P2
        part_df = partition.df
        unique_counts = [len(part_df[col].unique()) for col in part_df.columns]
        number_of_ones = unique_counts.count(1)
        cluster_goodness = number_of_ones / len(unique_counts)
        logger.debug (f"Cluster goodness {cluster_goodness} threshold {self.CT}")
        if (cluster_goodness > self.CT): #Skip S3 as our cluster is good enough
            return
        p1, p2 = self._get_p1_p2 (unique_counts)
        if (p1 == -1): #Check is this from existing implementations
            return
        col_p1 = part_df.columns[p1]
        col_p2 = part_df.columns[p2]
        #S3.2 Check relationship and split
        # Create a DataFrame with all unique pairs from col_p1 and col_p2

        
        p1_part_dict = {}
        p2_part_dict = {}

        unique_pairs = part_df.select([col_p1, col_p2]).unique()
        unique_pairs = unique_pairs.select(pl.concat_list(pl.col([col_p1, col_p2])).alias("unique_pairs"))
        
        logger.debug (f"Found unique pairs {unique_pairs}")
        for row in unique_pairs.to_dicts():
            pair = row["unique_pairs"]
            value_p1, value_p2 = pair
            logger.debug(f"P1 value is {value_p1} P2 value is {value_p2}")
            count_p1 = part_df.filter(pl.col(col_p1) == value_p1).select(pl.col(col_p2)).unique().shape[0]
            count_p2 = part_df.filter(pl.col(col_p2) == value_p2).select(pl.col(col_p1)).unique().shape[0]
            logger.debug(f"P1 count is {count_p1} P2 is count {count_p2}")
            #Get split position for each type
            if count_p1 > 1 and count_p2 > 1:
                logger.debug(f"{pair}: M-M (Many-to-Many)")
                #Journal paper says move to own partition. Earlier paper split based on lower cardinality if arrive from Step1
                #Decision we keep it in current partition which in the end will be its own parittion. 
                #return 4
                split_pos = 0
            elif count_p1 > 1:
                logger.debug(f"{pair}: 1-M (1-to-Many)")
                #temp_df = part_df.filter(pl.col(col_p1) == value_p1).select(pl.col(col_p1))
                s_temp_df = part_df.filter((pl.col(col_p1) == value_p1) & (pl.col(col_p2) == value_p2)).select(pl.col(col_p1))
                len_s_temp_df = s_temp_df.shape[0]
                cardinality_s_temp_df = s_temp_df.with_columns(pl.col(col_p1) == value_p1).unique().shape[0]
                split_pos = self._get_rank_position(len_s_temp_df, cardinality_s_temp_df, True)
                #logger.debug (f"s_temp is {s_temp_df}")
                logger.debug (f"s_temp len: {len_s_temp_df} card: {cardinality_s_temp_df}")
                #Do we want all lines where M exits or just the ones that part of this pair
                # Based on counts: 
                #Counts + SET: This seems to use whole set of tokens https://github.com/fluency03/iplom-java/blob/master/src/iplom/IPLoM.java#L591
                #Counts + Set supprt: Here it seems to only tokens in the pair. https://github.com/logpai/logparser/blob/main/logparser/IPLoM/IPLoM.py#L355
                #The conf paper suggests this as well. Journal paper is vague. 
                #self.get_rank_position(len_s_temp_df, cardinality_s_temp_df)
                #return 3
            elif count_p2 > 1:
                logger.debug(f"{pair}: M-1 (Many-to-1)")
                s_temp_df = part_df.filter((pl.col(col_p1) == value_p1) & (pl.col(col_p2) == value_p2)).select(pl.col(col_p2))
                len_s_temp_df = s_temp_df.shape[0]
                cardinality_s_temp_df = s_temp_df.with_columns(pl.col(col_p2) == value_p2).unique().shape[0]
                split_pos = self._get_rank_position(len_s_temp_df, cardinality_s_temp_df, False)
                #logger.debug (f"s_temp is {s_temp_df}")
                logger.debug (f"s_temp len: {len_s_temp_df} card: {cardinality_s_temp_df}")
                #return 2
            elif count_p1 == 1 and count_p2 == 1:
                logger.debug(f"{pair}: 1-1 (1-to-1)")
                split_pos = 1
            else:
                logger.warning(f"ERROR undefined relantionship !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                #return -1
            #Split   
            if split_pos==1:
                logger.debug(f"Split is 1. Size of part_df: {part_df.shape[0]}")
                #new_df = part_df.filter(pl.col(col_p1) == value_p1)
                #part_df = part_df.filter(~(pl.col(col_p1) == value_p1))
                new_df = part_df.filter((pl.col(col_p1) == value_p1) & (pl.col(col_p2) == value_p2))
                part_df = part_df.filter(~((pl.col(col_p1) == value_p1) & (pl.col(col_p2) == value_p2)))
                #logger.debug (f"new df: {new_df}")
                #logger.debug (f"part_df df:  {part_df}")
                if value_p1 in p1_part_dict:
                    # Append new_df to the existing DataFrame for this key
                    #Dictionary exits appending.
                    logger.debug (f"Dictionary exits appending")  
                    p1_part_dict[value_p1] = pl.concat([p1_part_dict[value_p1], new_df])
                else:
                    # If the key does not exist, simply add new_df to the dictionary
                    p1_part_dict[value_p1] = new_df

            elif split_pos==2:
                logger.debug(f"Split is 2. Size of part_df: {part_df.shape[0]}")
                #new_df = part_df.filter(pl.col(col_p2) == value_p2)
                #part_df = part_df.filter(~(pl.col(col_p2) == value_p2))
                new_df = part_df.filter((pl.col(col_p1) == value_p1) & (pl.col(col_p2) == value_p2))
                part_df = part_df.filter(~((pl.col(col_p1) == value_p1) & (pl.col(col_p2) == value_p2)))
                #logger.debug (f"new df: {new_df}")
                #logger.debug (f"part_df df:  {part_df}")
                if value_p2 in p2_part_dict:
                    # Append new_df to the existing DataFrame for this key
                    logger.debug (f"Dictionary exits appending") 
                    p2_part_dict[value_p2] = pl.concat([p2_part_dict[value_p2], new_df])
                else:
                    # If the key does not exist, simply add new_df to the dictionary
                    p2_part_dict[value_p2] = new_df

        for key, dataframe in p1_part_dict.items():
            logger.debug(f"S3P1 dict with key Appending: {key}")
            #partition.subpartitions.append(Partition(dataframe, len = partition.len, split_trace=partition.split_trace+"S3"))
            self.add_partition(Partition(dataframe, len = partition.len,  split_trace=partition.split_trace+"S3 "), parent_partition=partition)
        for key, dataframe in p2_part_dict.items():
            logger.debug(f"S3P2 dict with key Appending: {key}")
            #partition.subpartitions.append(Partition(dataframe, len = partition.len, split_trace=partition.split_trace+"S3"))
            self.add_partition(Partition(dataframe, len = partition.len,  split_trace=partition.split_trace+"S3 "), parent_partition=partition)
        #part_df is reduced in the process. If we do assign here we the same log rows in multiple levels
        if (part_df.shape[0] == 0):
            partition.df = None 
        else:
            partition.df = part_df


    def _get_rank_position(self, len, card, one_to_m):
        dist = card / len
        if dist < self.lower_bound:
            if one_to_m:
                return 2
            else:
                return 1
        else:
            if one_to_m:
                return 1
            else:
                return 2

              
        
    def _relation_type(self, df_part, p1, p2):
        # Get the column names from indices
        col_p1 = df_part.columns[p1]
        col_p2 = df_part.columns[p2]

        # Create a DataFrame with all unique pairs from col_p1 and col_p2
        unique_pairs = df_part.select([col_p1, col_p2]).unique()
        unique_pairs = unique_pairs.select(pl.concat_list(pl.col([col_p1, col_p2])).alias("unique_pairs"))
        
        logger.debug (f"Found unique pairs {unique_pairs}")
        for row in unique_pairs.to_dicts():
            pair = row["unique_pairs"]
            value_p1, value_p2 = pair
            logger.debug(f"P1 value is {value_p1} P2 value is {value_p2}")
            count_p1 = df_part.filter(pl.col(col_p1) == value_p1).select(pl.col(col_p2)).unique().shape[0]
            count_p2 = df_part.filter(pl.col(col_p2) == value_p2).select(pl.col(col_p1)).unique().shape[0]
            logger.debug(f"P1 count is {count_p1} P2 is count {count_p2}")

            if count_p1 > 1 and count_p2 > 1:
                logger.debug(f"{pair}: M-M (Many-to-Many)")
                return 4
            elif count_p1 > 1:
                logger.debug(f"{pair}: M-1 (Many-to-1)")
                return 3
            elif count_p2 > 1:
                logger.debug(f"{pair}: 1-M (1-to-Many)")
                return 2
            elif count_p1 == 1 and count_p2 == 1:
                logger.debug(f"{pair}: 1-1 (1-to-1)")
                return 1
            else:
                logger.warning(f"ERROR undefined relantionship")
                return -1
