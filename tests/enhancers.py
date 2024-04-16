import polars as pl
import sys
import polars as pl
import glob
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append('..')
import loglead.enhancer as eh
import loglead.loaders.base as load

home_directory = os.path.expanduser('~')
test_data_path = os.path.join(home_directory, "Datasets", "test_data") 

# Get all .parquet files in the directory
all_files = glob.glob(os.path.join(test_data_path, "*.parquet"))
print ("Enhancers test starting.")
print (f"Found files {all_files}")
# Extract unique dataset names, excluding '_seq' files
datasets = set()
for f in all_files:
    basename = os.path.basename(f)
    #Only use main files (no _seq) that exist after loader (no _eh)
    if "_lo" in basename and "_seq" not in basename and "_eh" not in basename:
        dataset_name = basename.replace(".parquet", "")
        datasets.add(dataset_name)

# Loop through each dataset and enhance
for dataset in datasets:
    # Load the event level data
    primary_file = os.path.join(test_data_path, f"{dataset}.parquet")
    print(f"\nLoading {primary_file}")
    df = pl.read_parquet(primary_file)
    #Kill nulls if they still exist
    df = df.filter(pl.col("m_message").is_not_null())
    # Enhance the event data
    print ("Enhancing data:", end=": ")
    enhancer = eh.EventLogEnhancer(df)
    print ("event lengths",  end=", ")
    df = enhancer.length()
    print ("normalizing",   end=", ")
    df = enhancer.normalize()
    print ("spliting to words",   end=", ")
    df = enhancer.words()
    print ("spliting to alphanumerics",   end=", ")
    df = enhancer.alphanumerics()
    print ("spliting to trigrams",   end=", ")
    df = enhancer.trigrams()
    print ("Drain parsing",   end=", ")
    df = enhancer.parse_drain()
    print ("Tipping parsing",   end=", ")
    df = enhancer.parse_tip()

    # Enhance / Aggregate sequence level
    loader = load.BaseLoader(filename=None, df=None, df_seq = None)
    seq_file = primary_file.replace(f"{dataset}.parquet", f"{dataset}_seq.parquet")
    if os.path.exists(seq_file):
        df_seq = pl.read_parquet(seq_file)
        loader.df_seq = df_seq
        enhancer_seq = eh.SequenceEnhancer(df = df, df_seq = df_seq)
        print ("\nAggregating drain parsing results",   end=", ")
        df_seq = enhancer_seq.events()
        print ("\nCreating next-event-prediction results from Drain events",   end=", ")
        df_seq = enhancer_seq.next_event_prediction()
        print ("Aggregating tokens / words",   end=", ")
        df_seq = enhancer_seq.tokens()
        df_seq = enhancer_seq.tokens("e_trigrams")
        print ("Aggregating event lengths",   end=", ") 
        df_seq = enhancer_seq.eve_len()
        print ("Enhancing sequence duration",   end=", ")
        df_seq = enhancer_seq.start_time()
        df_seq = enhancer_seq.end_time()
        df_seq = enhancer_seq.duration()
        print ("Enhancing sequence length in events")
        df_seq = enhancer_seq.seq_len()
        #Preparing loader for addition reduction
        loader.df_seq = df_seq
    loader.df = df
    #Save the data used for anomaly_detectors tests. 
    loader.df.write_parquet(f"{test_data_path}/{dataset}_eh.parquet") 
    if os.path.exists(seq_file):
        loader.df_seq.write_parquet(f"{test_data_path}/{dataset}_eh_seq.parquet")
    #Test the remaining slow parsers.   
    df = loader.reduce_dataframes(frac=0.01)#Reducing the ~100k -> 1k
    df_seq = loader.df_seq
    enhancer.df = df
    print(f"Remaining parsers with lines {len(df)}", end=": ")
    print ("IpLom",   end=", ")
    df = enhancer.parse_iplom()
    print ("Pl-Iplom parsing",   end=", ")
    df = enhancer.parse_pliplom()
    print ("AEL parsing",   end=", ")
    df = enhancer.parse_ael()
    print ("Brain parsing",   end=", ")
    df = enhancer.parse_brain()
    print ("Spell parsing",   end=", ")
    df = enhancer.parse_spell()
    print ("LenMa parsing",  end=", ")
    df = enhancer.parse_lenma()
    try:
        import tensorflow as tf
        # Perform actions with TensorFlow
        print("\nTensorFlow is available. Creating embeddings...")
        df = enhancer.create_neural_emb()
    except ImportError:
        # TensorFlow is not available
        print("\nTensorFlow is not installed. Embedding creation not tested")
        # Add any alternative code here if TensorFlow is not available


print ("Enhancers test complete.")