import time

import polars as pl
from dotenv import dotenv_values
from loglead.loaders import BGLLoader
from loglead.loaders import ThuSpiLibLoader
from loglead.loaders import HadoopLoader
from loglead.loaders import HDFSLoader
from loglead.loaders import RawLoader

from loglead.enhancers import EventLogEnhancer, SequenceEnhancer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from loglead import AnomalyDetector

# Suppress ConvergenceWarning
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Adjust full data source
envs = dotenv_values()
full_data = envs.get("LOG_DATA_PATH") #If this doesn't work, add it manually, e.g: full_data = "G:/Datasets"

#List the representations (column names) for anomaly detection
items = ["e_words", "e_trigrams","e_event_drain_id"] # "e_trigrams","e_alphanumerics"

shuffle_train_l = [True]#,False]
filter_anos_l = [True]#,False]

#### CHECKLIST: models, representations, filter_anos, 
###             shuffle, CV/tfidf, data split fractions

# Notes:
### F1 scores for these models don't tell much unless you use the optimizer.
### Each dataset has their own section where, e.g., you can determine how much data to use. 
### While the code is mostly the same, event and sequence based datasets have some differences 
### Whether the model uses completely unfiltered data (i.e. anomalies in training) can be adjusted with parameter "filter_anos"
### Here normalization is simply turning to lowercase, all numbers to 0s and multiple subsequent 0s to single 0
### The TruncatedSVD is there i

for filter_anos in filter_anos_l:
    for shuffle_train in shuffle_train_l:
        print("f"+str(int(filter_anos))+"s"+str(int(shuffle_train))) #tags to find from the output

        print("---------- HDFS ----------")
        frac_data = 0.01
        test_frac = 0.95
        stime = time.time()
        loader = HDFSLoader(filename=f"{full_data}/hdfs/HDFS.log", 
                                            labels_file_name=f"{full_data}/hdfs/anomaly_label.csv")
        df = loader.execute()
        df = loader.reduce_dataframes(frac=frac_data)
        df_seq = loader.df_seq       
        print("time loaded", time.time()-stime)

        df = df.filter(pl.col("m_message").is_not_null())
        #df = df.unique(subset=['m_message'])

        enhancer = EventLogEnhancer(df)
        df = enhancer.length()


        regexs = [('0','\d'),('0','0+')]
        df = enhancer.normalize(regexs, to_lower=True)
        print("time normalized", time.time()-stime)
        stime = time.time()
        df = enhancer.trigrams_unarranged("e_message_normalized")
        print("time trigrams", time.time()-stime)
        stime = time.time()
        df = enhancer.words("e_message_normalized")
        print("time words", time.time()-stime)
        stime = time.time()
        df = enhancer.parse_drain()
        print("time parse", time.time()-stime)
        stime = time.time()

        seq_enhancer =SequenceEnhancer(df = df, df_seq = df_seq)
        # Print for info
        #print("ano", len(seq_enhancer.df_seq.filter(seq_enhancer.df_seq["normal"]==False)))
        #print("normal", len(seq_enhancer.df_seq.filter(seq_enhancer.df_seq["normal"]==True)))
        seq_enhancer.seq_len()
        seq_enhancer.start_time()

        sad = AnomalyDetector()
        for item in items:
            models_dict = {
                "IsolationForest": {"filter_anos":filter_anos},
                "KMeans": {"filter_anos":filter_anos},
                "RarityModel": {"filter_anos":filter_anos},
                #"OOVDetector": {"filter_anos":filter_anos}, # Done later with CountVectorizer
            }
            print("-----", item, "-----")
            if "event" in item:
                seq_enhancer.events(item)
            elif item != "m_message":
                seq_enhancer.tokens(item)
            sad.item_list_col = item

            stime = time.time()
            sad.test_train_split(seq_enhancer.df_seq, test_frac=test_frac, shuffle=shuffle_train, vectorizer_class=TfidfVectorizer)
            print("time split and prepare:", time.time()-stime)
            sad.evaluate_with_params(models_dict)
            models_dict = {
                "OOVDetector": {"filter_anos":filter_anos},
            }
            stime = time.time()
            sad.test_train_split (seq_enhancer.df_seq, test_frac=test_frac, shuffle=shuffle_train, vectorizer_class=CountVectorizer)
            print("time split and prepare:", time.time()-stime)
            sad.evaluate_with_params(models_dict)


        print("---------- BGL ----------")
        frac_data = 0.05
        test_frac = 0.95
        stime = time.time()
        loader = BGLLoader(filename=f"{full_data}/bgl/BGL.log")
        df = loader.execute()
        ##print("ano", len(df.filter(df["normal"]==False)))
        ##print("normal", len(df.filter(df["normal"]==True)))
        df = loader.reduce_dataframes(frac=frac_data)
        df = df.filter(pl.col("m_message").is_not_null())
        #df = df.unique(subset=['m_message'])
        # print("time loaded", time.time()-stime)

        enhancer = EventLogEnhancer(df)
        stime = time.time()
        regexs = [('0','\d'),('0','0+')]
        df = enhancer.normalize(regexs, to_lower=True)
        print("time normalized", time.time()-stime)
        stime = time.time()
        df = enhancer.trigrams_unarranged("e_message_normalized")
        print("time trigrams", time.time()-stime)
        stime = time.time()
        df = enhancer.words("e_message_normalized")
        print("time words", time.time()-stime)
        stime = time.time()
        df = enhancer.parse_drain()
        print("time parse", time.time()-stime)
        stime = time.time()

        df = enhancer.length("e_message_normalized")

        sad = AnomalyDetector() 
        for item in items:
            models_dict = {
                "IsolationForest": {"filter_anos":filter_anos},
                "KMeans": {"filter_anos":filter_anos},
                "RarityModel": {"filter_anos":filter_anos},
                #"OOVDetector": {"filter_anos":filter_anos}, # Done later with CountVectorizer
            }
            print("-----", item, "-----")
            sad.item_list_col = item
            stime = time.time()
            sad.test_train_split (df, test_frac=test_frac, shuffle=shuffle_train, vectorizer_class=TfidfVectorizer)
            print("time split and prepare:", time.time()-stime)
            sad.evaluate_with_params(models_dict)
            models_dict = {
                "OOVDetector": {"filter_anos":filter_anos},
            }
            stime = time.time()
            sad.test_train_split (df, test_frac=test_frac, shuffle=shuffle_train, vectorizer_class=CountVectorizer)
            print("time split and prepare:", time.time()-stime)
            sad.evaluate_with_params(models_dict)


        print("---------- Thunderbird ----------")
        frac_data = 0.001
        test_frac = 0.95
        stime = time.time()
        loader = ThuSpiLibLoader(filename=f"{full_data}/thunderbird/tbird2.log", split_component=False)
        df = loader.execute()
        #print("ano", len(df.filter(df["normal"]==False)))
        #print("normal", len(df.filter(df["normal"]==True)))
        df = loader.reduce_dataframes(frac=frac_data)
        df = df.filter(pl.col("m_message").is_not_null())
        #df = df.unique(subset=['m_message'])

        enhancer = EventLogEnhancer(df)
        print("time loaded", time.time()-stime)


        enhancer = EventLogEnhancer(df)
        stime = time.time()
        regexs = [('0','\d'),('0','0+')]
        df = enhancer.normalize(regexs, to_lower=True)
        print("time normalized", time.time()-stime)
        stime = time.time()
        df = enhancer.trigrams_unarranged("e_message_normalized")
        print("time trigrams", time.time()-stime)
        stime = time.time()
        df = enhancer.words("e_message_normalized")
        print("time words", time.time()-stime)
        stime = time.time()
        df = enhancer.parse_drain()
        print("time parse", time.time()-stime)
        stime = time.time()

        df = enhancer.length("e_message_normalized")

        sad = AnomalyDetector() 
        for item in items:
            models_dict = {
                "IsolationForest": {"filter_anos":filter_anos},
                "KMeans": {"filter_anos":filter_anos},
                "RarityModel": {"filter_anos":filter_anos},
                #"OOVDetector": {"filter_anos":filter_anos}, # Done later with CountVectorizer
            }
            print("-----", item, "-----")
            sad.item_list_col = item
            
            stime = time.time()
            sad.test_train_split (df, test_frac=test_frac, vectorizer_class=TfidfVectorizer)
            print("time split and prepare:", time.time()-stime)
            
            sad.evaluate_with_params(models_dict)
            models_dict = {
                "OOVDetector": {"filter_anos":filter_anos},
            }
            stime = time.time()
            sad.test_train_split (df, test_frac=test_frac, shuffle=shuffle_train, vectorizer_class=CountVectorizer)
            print("time split and prepare:", time.time()-stime)
            sad.evaluate_with_params(models_dict)


        print("---------- spirit ----------")
        frac_data = 0.001
        test_frac = 0.95
        stime = time.time()
        loader = ThuSpiLibLoader(filename=f"{full_data}/spirit/spirit2.log", split_component=False)
        df = loader.execute()
        #print("ano", len(df.filter(df["normal"]==False)))
        #print("normal", len(df.filter(df["normal"]==True)))
        df = loader.reduce_dataframes(frac=frac_data)
        df = df.filter(pl.col("m_message").is_not_null())
        enhancer = EventLogEnhancer(df)
        print("time loaded", time.time()-stime)


        enhancer = EventLogEnhancer(df)
        stime = time.time()
        regexs = [('0','\d'),('0','0+')]
        df = enhancer.normalize(regexs, to_lower=True)
        print("time normalized", time.time()-stime)
        stime = time.time()
        df = enhancer.trigrams_unarranged("e_message_normalized")
        print("time trigrams", time.time()-stime)
        stime = time.time()
        df = enhancer.words("e_message_normalized")
        print("time words", time.time()-stime)
        stime = time.time()
        df = enhancer.parse_drain()
        print("time parse", time.time()-stime)
        stime = time.time()

        df = enhancer.length("e_message_normalized")

        sad = AnomalyDetector() 
        for item in items:
            models_dict = {
                "IsolationForest": {"filter_anos":filter_anos},
                "KMeans": {"filter_anos":filter_anos},
                "RarityModel": {"filter_anos":filter_anos},
                #"OOVDetector": {"filter_anos":filter_anos}, # Done later with CountVectorizer
            }
            print("-----", item, "-----")
            sad.item_list_col = item
            
            stime = time.time()
            sad.test_train_split (df, test_frac=test_frac, shuffle=shuffle_train, vectorizer_class=TfidfVectorizer)
            print("time split and prepare:", time.time()-stime)
            
            sad.evaluate_with_params(models_dict)
            models_dict = {
                "OOVDetector": {"filter_anos":filter_anos},
            }
            stime = time.time()
            sad.test_train_split (df, test_frac=test_frac, shuffle=shuffle_train, vectorizer_class=CountVectorizer)
            print("time split and prepare:", time.time()-stime)
            sad.evaluate_with_params(models_dict)

        print("---------- liberty ----------")
        frac_data = 0.001
        test_frac = 0.95
        stime = time.time()
        loader = ThuSpiLibLoader(filename=f"{full_data}/liberty/liberty2.log", split_component=False)
        df = loader.execute()
        #print("ano", len(df.filter(df["normal"]==False)))
        #print("normal", len(df.filter(df["normal"]==True)))
        df = loader.reduce_dataframes(frac=frac_data)
        df = df.filter(pl.col("m_message").is_not_null())
        enhancer = EventLogEnhancer(df)
        print("time loaded", time.time()-stime)


        enhancer = EventLogEnhancer(df)
        stime = time.time()
        regexs = [('0','\d'),('0','0+')]
        df = enhancer.normalize(regexs, to_lower=True)
        print("time normalized", time.time()-stime)
        stime = time.time()
        df = enhancer.trigrams_unarranged("e_message_normalized")
        print("time trigrams", time.time()-stime)
        stime = time.time()
        df = enhancer.words("e_message_normalized")
        print("time words", time.time()-stime)
        stime = time.time()
        df = enhancer.parse_drain()
        print("time parse", time.time()-stime)
        stime = time.time()

        df = enhancer.length("e_message_normalized")

        sad = AnomalyDetector() 
        for item in items:
            models_dict = {
                "IsolationForest": {"filter_anos":filter_anos},
                "KMeans": {"filter_anos":filter_anos},
                "RarityModel": {"filter_anos":filter_anos},
                #"OOVDetector": {"filter_anos":filter_anos}, # Done later with CountVectorizer
            }
            print("-----", item, "-----")
            sad.item_list_col = item
            
            stime = time.time()
            sad.test_train_split (df, test_frac=test_frac, shuffle=shuffle_train, vectorizer_class=TfidfVectorizer)
            print("time split and prepare:", time.time()-stime)
            sad.evaluate_with_params(models_dict)
            models_dict = {
                "OOVDetector": {"filter_anos":filter_anos},
            }
            stime = time.time()
            sad.test_train_split (df, test_frac=test_frac, shuffle=shuffle_train, vectorizer_class=CountVectorizer)
            print("time split and prepare:", time.time()-stime)
            sad.evaluate_with_params(models_dict)

        print("---------- Hadoop ----------")
        frac_data = 1
        test_frac = 0.5
        stime = time.time()
        loader = HadoopLoader(filename=f"{full_data}/hadoop/",
                                                    filename_pattern  ="*.log",
                                                    labels_file_name=f"{full_data}/hadoop/abnormal_label_accurate.txt")
        df = loader.execute()
        df = loader.reduce_dataframes(frac=frac_data)

        df_seq = loader.df_seq       
        print("time loaded", time.time()-stime)
        df = df.filter(pl.col("m_message").is_not_null())

        enhancer = EventLogEnhancer(df)
        df = enhancer.length()

        regexs = [('0','\d'),('0','0+')]
        df = enhancer.normalize(regexs, to_lower=True)
        print("time normalized", time.time()-stime)
        stime = time.time()
        df = enhancer.trigrams_unarranged("e_message_normalized")
        print("time trigrams", time.time()-stime)
        stime = time.time()
        df = enhancer.words("e_message_normalized")
        print("time words", time.time()-stime)
        stime = time.time()
        df = enhancer.parse_drain()
        print("time parse", time.time()-stime)
        stime = time.time()

        seq_enhancer =SequenceEnhancer(df = df, df_seq = df_seq)
        #print("ano", len(seq_enhancer.df_seq.filter(seq_enhancer.df_seq["normal"]==False)))
        #print("normal", len(seq_enhancer.df_seq.filter(seq_enhancer.df_seq["normal"]==True)))
        seq_enhancer.seq_len() #OOVD uses data from the df for faster calculations
        seq_enhancer.start_time()

        sad = AnomalyDetector()
        for item in items:
            models_dict = {
                "IsolationForest": {"filter_anos":filter_anos},
                "KMeans": {"filter_anos":filter_anos},
                "RarityModel": {"filter_anos":filter_anos},
                #"OOVDetector": {"filter_anos":filter_anos}, # Done later with CountVectorizer
            }
            print("-----", item, "-----")
            if item != "e_event_drain_id":
                seq_enhancer.tokens(item)
            else:
                seq_enhancer.events(item)
            sad.item_list_col = item

            stime = time.time()
            sad.test_train_split (seq_enhancer.df_seq, test_frac=test_frac, shuffle=shuffle_train, vectorizer_class=TfidfVectorizer)
            print("time split and prepare:", time.time()-stime)

            sad.evaluate_with_params(models_dict)
            #if filter_anos:
            models_dict = {
                "OOVDetector": {"filter_anos":filter_anos},
            }
            stime = time.time()
            sad.test_train_split (seq_enhancer.df_seq, test_frac=test_frac, shuffle=shuffle_train, vectorizer_class=CountVectorizer)
            print("time split and prepare:", time.time()-stime)
            sad.evaluate_with_params(models_dict)
