import polars as pl
import sys
import polars as pl
import glob
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append('..')
import loglead.enhancer as eh

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

# Loop through each dataset
for dataset in datasets:
    # Load the event level data
    primary_file = os.path.join(test_data, f"{dataset}.parquet")
    print(f"\nLoading {primary_file}")
    df = pl.read_parquet(primary_file)
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
    print ("Spell parsing",   end=", ")
    df = enhancer.parse_spell()
    print ("Enhancing data:",  end=", ")

    #Loader object and reduce.
    #df = enhancer.parse_lenma()
#Check for tf existance. 
    #df = enhancer.parse_spell()

