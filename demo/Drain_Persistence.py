import os
import polars as pl

from loglead.enhancers import EventLogEnhancer


def run_experiment(n_parts=10):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # Location of our sample data
    sample_data = os.path.join(script_dir, 'samples')

    # Load TB (Thunderbird) data from sample data
    df_full = pl.read_parquet(os.path.join(sample_data, "tb_0125percent.parquet"))
    print(f"Loaded dataset with {df_full.shape[0]} rows")

    # Split into n parts
    chunk_size = df_full.shape[0] // n_parts
    chunks = [df_full[i*chunk_size:(i+1)*chunk_size] for i in range(n_parts)]

    # Start template mining
    print("=" * 50)
    print("NORMAL MODE")
    print("=" * 50)
    # --- Normal mode ---
    for i, chunk in enumerate(chunks):
        df_chunk = chunk.clone()

        enhancer = EventLogEnhancer(df_chunk)
        df_chunk = enhancer.normalize()
        df_chunk = enhancer.parse_drain(reparse=True, templates=True)
        templates_per_id = df_chunk.group_by("e_event_drain_id").agg(pl.col("e_event_drain_template"))
        templates_per_id_dict = templates_per_id.to_dict(as_series=False)
        template_id_mapping = dict(zip(templates_per_id_dict["e_event_drain_id"], templates_per_id_dict["e_event_drain_template"]))

        print("-" * 50)
        print(f"Log chunk {i+1}")
        print("-" * 50)
        print(f"Number of clusters: {len(template_id_mapping.keys())}")
        print(f"Templates for the first five clusters (e1, e2, e3, e4, and e5):")
        # In normal mode, each chunk starts with a fresh miner.
        # Because the size of log chunk is not that large,
        # clusters from 'e1' to 'e5' always exist,
        # but their content might chance between runs.
        try:
            print(max(template_id_mapping['e1'], key=lambda t: t.count("*")))
        
        except Exception: KeyError (
            print("Templates not found. Cluster 'e1' have been pruned.")
        )
        try:
            print(max(template_id_mapping['e2'], key=lambda t: t.count("*")))
        
        except Exception: KeyError (
            print("Templates not found. Cluster 'e2' have been pruned.")
        )        
        try:
            print(max(template_id_mapping['e3'], key=lambda t: t.count("*")))
        
        except Exception: KeyError (
            print("Templates not found. Cluster 'e3' have been pruned.")
        )        
        try:
            print(max(template_id_mapping['e4'], key=lambda t: t.count("*")))
        
        except Exception: KeyError (
            print("Templates not found. Cluster 'e4' have been pruned.")
        )        
        try:
            print(max(template_id_mapping['e5'], key=lambda t: t.count("*")))
        
        except Exception: KeyError (
            print("Templates not found. Cluster 'e5' have been pruned.")
        )
    print("=" * 50)
    print("PERSISTENCE MODE")
    print("=" * 50)

    # --- Persistence mode ---
    for i, chunk in enumerate(chunks):
        df_chunk = chunk.clone()

        enhancer = EventLogEnhancer(df_chunk)
        df_chunk = enhancer.normalize()
        df_chunk = enhancer.parse_drain(reparse=True, templates=True, persistence=True)
        templates_per_id = df_chunk.group_by("e_event_drain_id").agg(pl.col("e_event_drain_template"))
        templates_per_id_dict = templates_per_id.to_dict(as_series=False)
        template_id_mapping = dict(zip(templates_per_id_dict["e_event_drain_id"], templates_per_id_dict["e_event_drain_template"]))
        
        print("-" * 50)
        print(f"Log chunk {i+1}")
        print("-" * 50)
        print(f"Number of clusters: {len(template_id_mapping.keys())}")
        print(f"Templates for the first five clusters (e1, e2, e3, e4, and e5):")
        # Miner state is carried across chunks.
        # If the same log pattern shows up later,
        # it’s matched to its original cluster (e1, e2, e3, etc.).
        # In persistence mode, log processing runs longer, which increases
        # the possibility of pruning or merging clusters.
        try:
            print(max(template_id_mapping['e1'], key=lambda t: t.count("*")))
        
        except Exception: KeyError (
            print("Templates not found. Cluster 'e1' have been pruned.")
        )
        try:
            print(max(template_id_mapping['e2'], key=lambda t: t.count("*")))
        
        except Exception: KeyError (
            print("Templates not found. Cluster 'e2' have been pruned.")
        )        
        try:
            print(max(template_id_mapping['e3'], key=lambda t: t.count("*")))
        
        except Exception: KeyError (
            print("Templates not found. Cluster 'e3' have been pruned.")
        )        
        try:
            print(max(template_id_mapping['e4'], key=lambda t: t.count("*")))
        
        except Exception: KeyError (
            print("Templates not found. Cluster 'e4' have been pruned.")
        )        
        try:
            print(max(template_id_mapping['e5'], key=lambda t: t.count("*")))
        
        except Exception: KeyError (
            print("Templates not found. Cluster 'e5' have been pruned.")
        )
    
    # Clean up
    if os.path.exists("drain3_state_no_masking.json"):
        os.remove("drain3_state_no_masking.json")

    return


if __name__ == "__main__":
    # If the state file exists, remove it to prevent
    # any unexpected side effects to demo.
    if os.path.exists("drain3_state_no_masking.json"):
        os.remove("drain3_state_no_masking.json")
    run_experiment()
