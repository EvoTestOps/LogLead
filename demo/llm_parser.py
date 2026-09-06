# This demo shows how to use IPLoM LLM Parser, a log parsing pipeline that combines fast
# syntactic clustering (IPLoM) with LLM-based template extraction without labeled data.
#
# Requires Python 3.11+ and the optional `iplom_llm` extra:
#   uv run --extra iplom_llm demo/llm_parser.py
#
# The parser requires either a locally hosted LLM, or an OpenRouter API key.
# See https://github.com/EvoTestOps/iplom-llm-parser/blob/main/README.md for details about configuration.

# ______________________________________________________________________________
# Part 1 load libraries and setup paths.
import os
import random

import polars as pl
from dotenv import find_dotenv, load_dotenv
from iplom_llm_parser import Config, IPLoMConfig, LLMConfig, PipelineConfig
from iplom_llm_parser.llm_client import LLMConnectionError, check_llm_connection

from loglead.enhancers import EventLogEnhancer

# Ensure this always gets executed in the same location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Reads API_KEY from a .env file at the repo root
# see .env.sample
load_dotenv(find_dotenv())

# Location of our sample data
sample_data = os.path.join(script_dir, "samples")

# Parser configuration
pipeline_config = PipelineConfig(template_correction=False)
iplom_config = IPLoMConfig()


# Option 1: OpenRouter
# Model ids and pricing: https://openrouter.ai/models
# Set the API key to .env file (see .env.sample)
api_key = os.environ.get("API_KEY") or os.environ.get("OPENROUTER_API_KEY")

llm_config = LLMConfig(
    provider="openrouter",
    model="google/gemma-4-26b-a4b-it",
    api_key=api_key,
    max_concurrent=1,
)

# Option 2: locally hosted OpenAI-compatible server (e.g. with LM Studio)
# llm_config = LLMConfig(
#     provider="local",
#     model="google/gemma-4-26b-a4b-qat",
#     base_url="http://localhost:1234/v1",
#     max_concurrent=1,
# )

config = Config(pipeline=pipeline_config, iplom=iplom_config, llm=llm_config)

# Alternatively use config file:
# config_path = os.path.join(script_dir, "config.toml")

# LLMClient() also runs this check internally before making any requests,
# so this call is not strictly required, but catches bad endpoint/key/model immeadiately.
try:
    check_llm_connection(llm_config)
except LLMConnectionError as e:
    raise SystemExit(f"LLM check failed: {e}")


# For nicer printing a function to format series of elements as a list-like string
def format_as_list(series):
    elements = [str(item) for item in series]
    return "[" + ", ".join(elements) + "]"


# _________________________________________________________________________________
# Part 2 load data from sample file
df = pl.read_parquet(os.path.join(sample_data, "hdfs_events_2percent.parquet"))
print(f"Read HDFS 2% sample. Number of events is: {len(df)}")

# Test with a small subset
demo_size = 500
df_small = df.head(demo_size)
print(f"Using a {demo_size}-row subset for this demo to limit LLM calls\n")

# _________________________________________________________________________________
# Part 3 run IPLoM-LLM-Parser via EventLogEnhancer
print("Starting IPLoM-LLM parsing:")
enhancer = EventLogEnhancer(df_small)

df_small = enhancer.parse_iplom_llm(
    field="m_message",
    # config_path=config_path,
    config=config,
)

stats = enhancer.iplom_llm_stats
print(f"LLM calls made:      {stats.llm_calls}")
print(f"Input tokens used:   {stats.input_tokens}")
print(f"Output tokens used:  {stats.output_tokens}")
print(f"Total time (s):      {stats.total_time:.2f}")
print(f"Rows matched:        {stats.matched}")
print(f"Rows left as noise:  {stats.noise}\n")

# Pick a random line and show the parsed result
row_index = random.randint(0, len(df_small) - 1)
print(f"Original log message: {df_small['m_message'][row_index]}")
print(f"IPLoM-LLM event id: {df_small['e_event_iplom_llm_id'][row_index]}")
print(f"IPLoM-LLM event template: {df_small['e_event_iplom_llm_template'][row_index]}")
print(f"Params: {format_as_list(df_small['e_event_iplom_llm_params'][row_index])}")
print(
    f"LLM generated slot types: {format_as_list(df_small['e_event_iplom_llm_slot_types'][row_index])}"
)

# _________________________________________________________________________________
# Part 4 inspect templates found across the whole subset
event_summary = (
    df_small.group_by("e_event_iplom_llm_id")
    .agg(
        pl.col("e_event_iplom_llm_template").first().alias("template"),
        pl.len().alias("count"),
    )
    .sort("count", descending=True)
)
print(f"\nDistinct templates found: {len(event_summary)}")
print(event_summary)

# Inspect noise rows
noise_df = df_small.filter(pl.col("e_event_iplom_llm_template").is_null())
if len(noise_df) > 0:
    print(f"\n{len(noise_df)} rows left unmatched as noise, e.g.:")
    print(noise_df.select("m_message").head(5))
else:
    print("\nNo noise rows - every line matched a template.")
