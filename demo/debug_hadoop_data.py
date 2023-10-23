#Separate demo files
import loglead.loader as load, loglead.enricher as er, anomaly_detection as ad
import polars as pl

hadoop_processor = load.HadoopLoader(filename="../../../Datasets/hadoop/",
                                            filename_pattern  ="*.log",
                                     labels_file_name="../../../Datasets/hadoop/abnormal_label_accurate.txt")
df_hadoop = hadoop_processor.execute()
df_hadoop.filter(df_hadoop['m_message'].is_null())
df_hadoop.filter(pl.col('m_message').is_null())

df_hadoop.select(pl.col('m_message'))

df_hadoop.with_columns(pl.col('m_message'))


for col_name in df_hadoop.columns:
    null_count = df_hadoop.filter(pl.col(col_name).is_null()).shape[0]
    print(f"Number of null values in {col_name} column: {null_count}")

df_hadoop = df_hadoop.with_columns(pl.col("m_message").fill_null("<EMPTY LOG MESSAGE>"))



df_hadoop.filter(pl.col("group2").is_null())
df_hadoop.filter(pl.col("m_message").is_null())["seq_id_sub"][0]
df_hadoop.filter(pl.col("m_timestamp").is_null())["seq_id_sub"][0]

df_hadoop.filter(pl.col("m_message").is_null())["column_1"][0]
df_hadoop.filter((pl.col("m_timestamp").is_null()) & (
    pl.col("seq_id_sub").str.starts_with('container_1445062781478_0011_01_000001.log')))["column_1"][0]

df_hadoop = df_hadoop.select(pl.col("row_nr", "group", "seq_id_sub", "m_timestamp", "column_1"))

# Find the index of the matched line
matched_index = df_hadoop.filter((pl.col("m_timestamp").is_null()) & 
                                 (pl.col("seq_id_sub").str.starts_with('container_1445062781478_0011_01_000001.log')))["row_nr"][0]
# Retrieve the surrounding lines
start_index = max(0, matched_index - 5)  # Line before
end_index = min(len(df_hadoop) - 1, matched_index + 5)  # Line after
surrounding_lines = df_hadoop.filter((pl.col("row_nr") >= start_index) & (pl.col("row_nr") <= end_index)
                                     & (pl.col("seq_id_sub").str.starts_with('container_1445062781478_0011_01_000001.log')))["column_1"]


surrounding_lines = df_hadoop.filter((pl.col("row_nr") >= start_index) & (pl.col("row_nr") <= end_index)
                                     & (pl.col("seq_id_sub").str.starts_with('container_1445062781478_0011_01_000001.log')))

print(surrounding_lines)
for i in range(len(surrounding_lines)):
    print(surrounding_lines[i])




# Find the index of the matched line
index = df_hadoop.filter((pl.col("m_timestamp").is_null()) & 
                         (pl.col("seq_id_sub").str.starts_with('container_1445062781478_0011_01_000001.log'))).collect().index[0]

# Retrieve the surrounding lines
start_index = max(0, index - 1)  # Line before
end_index = min(len(df_hadoop) - 1, index + 1)  # Line after

surrounding_lines = df_hadoop.slice(start_index, end_index - start_index + 1)
print(surrounding_lines)


from functools import reduce
# Create a function to sum expressions
def sum_exprs(a, b):
    return a + b
# Create a new column 'null_count' that counts the number of null values in each row
null_counts = [pl.col(col).is_null().cast(pl.Int32) for col in df_hadoop.columns]
total_nulls = reduce(sum_exprs, null_counts)
df_hadoop = df_hadoop.with_columns(total_nulls.alias("null_count"))
# Sort the dataframe by 'null_count' in descending order
sorted_df = df_hadoop.sort("null_count", descending=True)
# Display the top rows with the highest count of null values
top_null_rows = sorted_df.head(10)  # Adjust the number to display more or fewer rows
print(top_null_rows)

print(top_null_rows["seq_id"][0])
print(top_null_rows["seq_id_sub"][0])
print(top_null_rows["date"][0])

hadoop_processor = load.HadoopLoader(filename="../../../Datasets/hadoop/",
                                            filename_pattern  ="*.log",
                                     labels_file_name="../../../Datasets/hadoop/abnormal_label_accurate.txt")
df_hadoop = hadoop_processor.load()
df_hadoop = hadoop_processor._merge_multiline_entries()

