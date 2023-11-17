import polars as pl
import sys
import polars as pl
import glob
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append('..')
import loglead.enhancer as eh
import loglead.loader as load

# Set your directory
test_data = "/home/mmantyla/Datasets/test_data"  # Replace with the path to your folder

# Get all .parquet files in the directory
all_files = glob.glob(os.path.join(test_data, "*.parquet"))

# Extract unique dataset names, excluding '_seq' files
datasets = set()
for f in all_files:
    basename = os.path.basename(f)
    if "_seq" not in basename:
        dataset_name = basename.replace(".parquet", "")
        datasets.add(dataset_name)

# Loop through each dataset and enhance
for dataset in datasets:
    # Load the event level data
    primary_file = os.path.join(test_data, f"{dataset}.parquet")
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
    # Enhance / Aggregate sequence level
    seq_file = primary_file.replace(f"{dataset}.parquet", f"{dataset}_seq.parquet")
    loader = load.BaseLoader(filename=None, df=None, df_seq = None)
    if os.path.exists(seq_file):
        df_seq = pl.read_parquet(seq_file)
        loader.df_seq = df_seq
        enhancer_seq = eh.SequenceEnhancer(df = df, df_seq = df_seq)
        print ("Aggregating drain parsing results",   end=", ")
        enhancer_seq.events()
        print ("Aggregating tokens / words",   end=", ")
        enhancer_seq.tokens()
        print ("Aggregating event lengths",   end=", ") 
        enhancer_seq.eve_len()
        print ("Enhancing sequence duration",   end=", ")
        enhancer_seq.start_time()
        enhancer_seq.end_time()
        enhancer_seq.duration()
        print ("Enhancing sequence length in events")
        enhancer_seq.seq_len()
        #Preparing loader for addition reduction
        loader.df_seq = df_seq
    loader.df = df
    df = loader.reduce_dataframes(frac=0.01)#Reducing the reduced 100k -> 1k
    df_seq = loader.df_seq
    print(len(df))
    print ("Spell parsing",   end=", ")
    enhancer.df = df
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


