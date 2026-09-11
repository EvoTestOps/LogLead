# MCP tool performance

Cells: seconds. Columns: the four data fractions.

What "5%" means differs by log root: for `hadoop_renamed` and `hdfs_balanced_5k` it is 5% of the
**log folders** (and their files); for `bgl_split_10` it is 5% of `BGL.log`'s **log lines**, taken
first and then split into 10 slices, so the folder count is always 10 and what varies is the text
inside each one.

Tables A1-A4 are **cold** (first call, nothing cached). Tables B1-B4 are the same grid **warm**
(repeated call, same process).

## Log roots at each fraction

| log root | unit varied | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| hadoop_renamed | log folders | 3 folders / 54 files | 6 folders / 108 files | 28 folders / 509 files | 55 folders / 978 files |
| hdfs_balanced_5k | log folders | 250 folders / 250 files | 500 folders / 500 files | 2,500 folders / 2,500 files | 5,000 folders / 5,000 files |
| bgl_split_10 | log lines | 237,398 lines (10 x 23,739) | 474,796 lines (10 x 47,479) | 2,373,982 lines (10 x 237,398) | 4,747,963 lines (10 x 474,796) |

# Part A -- cold (first call)

## Table A1 -- Auxiliary tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| peek_log_root | hadoop_renamed | 0.121 | 0.156 | 0.211 | 0.206 |
| peek_log_root | hdfs_balanced_5k | 0.198 | 0.182 | 0.232 | 0.214 |
| peek_log_root | bgl_split_10 | 0.090 | 0.078 | 0.086 | 0.041 |
| open_log_root | hadoop_renamed | 1.176 | 1.403 | 3.481 | 5.490 |
| open_log_root | hdfs_balanced_5k | 1.358 | 1.605 | 2.926 | 5.219 |
| open_log_root | bgl_split_10 | 1.291 | 2.387 | 12.1 | 26.5 |
| list_log_roots | hadoop_renamed | 0.016 | 0.011 | 0.018 | 0.021 |
| list_log_roots | hdfs_balanced_5k | 0.009 | 0.015 | 0.016 | 0.024 |
| list_log_roots | bgl_split_10 | 0.023 | 0.037 | 0.149 | 0.256 |
| describe_log_root | hadoop_renamed | 0.023 | 0.015 | 0.023 | 0.046 |
| describe_log_root | hdfs_balanced_5k | 0.016 | 0.021 | 0.027 | 0.051 |
| describe_log_root | bgl_split_10 | 0.046 | 0.068 | 0.351 | 0.522 |
| set_folder_names | hadoop_renamed | 0.036 | 0.086 | 0.110 | 0.192 |
| set_folder_names | hdfs_balanced_5k | 0.024 | 0.036 | 0.123 | 0.144 |
| set_folder_names | bgl_split_10 | 0.109 | 0.155 | 0.947 | 2.319 |
| read_log_lines | hadoop_renamed | 0.003 | 0.002 | 0.003 | 0.004 |
| read_log_lines | hdfs_balanced_5k | 0.001 | 0.003 | 0.003 | 0.004 |
| read_log_lines | bgl_split_10 | 0.008 | 0.005 | 0.019 | 0.025 |
| search_log_lines | hadoop_renamed | 0.013 | 0.006 | 0.017 | 0.018 |
| search_log_lines | hdfs_balanced_5k | 0.006 | 0.007 | 0.011 | 0.015 |
| search_log_lines | bgl_split_10 | 0.027 | 0.033 | 0.092 | 0.102 |
| read_log_lines (new tokens) | hadoop_renamed | 0.094 | 0.184 | 0.307 | 0.656 |
| read_log_lines (new tokens) | hdfs_balanced_5k | 0.048 | 0.054 | 0.136 | 0.231 |
| read_log_lines (new tokens) | bgl_split_10 | 0.077 | 0.147 | 0.431 | 4.153 |
| new_tokens | hadoop_renamed | 0.103 | 0.093 | 0.119 | 0.171 |
| new_tokens | hdfs_balanced_5k | 0.051 | 0.051 | 0.065 | 0.083 |
| new_tokens | bgl_split_10 | 0.107 | 0.130 | 0.356 | 0.958 |
| query_result | hadoop_renamed | 0.003 | 0.002 | 0.003 | 0.003 |
| query_result | hdfs_balanced_5k | 0.002 | 0.003 | 0.008 | 0.005 |
| query_result | bgl_split_10 | 0.003 | 0.003 | 0.003 | 0.003 |
| split_log_file | hadoop_renamed | 0.003 | 0.003 | 0.033 | 0.074 |
| split_log_file | hdfs_balanced_5k | 0.001 | 0.001 | 0.001 | 0.001 |
| split_log_file | bgl_split_10 | 0.023 | 0.034 | 0.154 | 0.433 |
| close_log_root | hadoop_renamed | 0.011 | 0.015 | 0.015 | 0.038 |
| close_log_root | hdfs_balanced_5k | 0.007 | 0.009 | 0.015 | 0.014 |
| close_log_root | bgl_split_10 | 0.029 | 0.045 | 0.157 | 0.264 |
| run_config | hadoop_renamed | 0.739 | 1.611 | 7.130 | 15.6 |
| run_config | hdfs_balanced_5k | 1.945 | 3.693 | 18.9 | 38.7 |
| run_config | bgl_split_10 | 12.2 | 12.2 | 57.3 | 180.3 |

## Table A2 -- Distance tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| distance_folder_filename | hadoop_renamed | 0.066 | 0.074 | 0.402 | 0.715 |
| distance_folder_filename | hdfs_balanced_5k | 1.659 | 4.297 | 19.4 | 47.9 |
| distance_folder_filename | bgl_split_10 | 0.123 | 0.173 | 0.323 | 0.431 |
| distance_folder_content | hadoop_renamed | 0.881 | 1.412 | 7.916 | 15.4 |
| distance_folder_content | hdfs_balanced_5k | 1.814 | 3.960 | 20.1 | 41.5 |
| distance_folder_content | bgl_split_10 | 5.805 | 11.5 | 56.0 | 147.7 |
| distance_file_content | hadoop_renamed | 0.775 | 1.621 | 8.271 | 16.9 |
| distance_file_content | hdfs_balanced_5k | 0.019 | 0.025 | 0.021 | 0.023 |
| distance_file_content | bgl_split_10 | 0.034 | 0.052 | 0.167 | 0.275 |
| distance_line_content | hadoop_renamed | 1.388 | 0.102 | 0.359 | 1.284 |
| distance_line_content | hdfs_balanced_5k | 0.724 | 1.588 | 8.571 | 16.2 |
| distance_line_content | bgl_split_10 | 0.089 | 0.117 | 0.246 | 0.327 |

## Table A3 -- Anomaly tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename | hadoop_renamed | 0.214 | 0.218 | 0.199 | 0.233 |
| anomaly_folder_filename | hdfs_balanced_5k | 0.159 | 0.173 | 0.194 | 0.260 |
| anomaly_folder_filename | bgl_split_10 | 0.242 | 0.343 | 0.931 | 1.559 |
| anomaly_folder_content | hadoop_renamed | 0.213 | 0.230 | 0.363 | 0.786 |
| anomaly_folder_content | hdfs_balanced_5k | 0.161 | 0.178 | 0.314 | 0.528 |
| anomaly_folder_content | bgl_split_10 | 0.318 | 0.544 | 1.865 | 8.044 |
| anomaly_file_content | hadoop_renamed | err | 2.572 | 2.404 | 3.285 |
| anomaly_file_content | hdfs_balanced_5k | 0.021 | 0.022 | 0.028 | 0.036 |
| anomaly_file_content | bgl_split_10 | 0.032 | 0.041 | 0.100 | 0.261 |
| anomaly_line_content | hadoop_renamed | 0.584 | 0.329 | 0.379 | 0.389 |
| anomaly_line_content | hdfs_balanced_5k | 0.017 | 0.016 | 0.026 | 0.032 |
| anomaly_line_content | bgl_split_10 | 0.032 | 0.056 | 0.150 | 0.269 |

## Table A4 -- Plot tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| plot_folder_filename | hadoop_renamed | 0.047 | 0.067 | 0.067 | 0.071 |
| plot_folder_filename | hdfs_balanced_5k | 0.119 | 0.117 | 0.140 | 0.185 |
| plot_folder_filename | bgl_split_10 | 0.127 | 0.133 | 0.292 | 0.371 |
| plot_folder_content (scatter) | hadoop_renamed | 0.083 | 0.130 | 0.350 | 0.607 |
| plot_folder_content (scatter) | hdfs_balanced_5k | 0.075 | 0.100 | 0.190 | 0.317 |
| plot_folder_content (scatter) | bgl_split_10 | 0.208 | 0.421 | 1.590 | 6.933 |
| plot_folder_content (scatter+umap) | hadoop_renamed | err | 16.3 | 8.078 | 10.9 |
| plot_folder_content (scatter+umap) | hdfs_balanced_5k | 8.490 | 9.033 | 16.9 | 27.8 |
| plot_folder_content (scatter+umap) | bgl_split_10 | 7.730 | 8.101 | 9.375 | 14.7 |
| plot_file_content (scatter) | hadoop_renamed | 0.077 | 0.066 | 0.072 | 0.087 |
| plot_file_content (scatter) | hdfs_balanced_5k | 0.012 | 0.011 | 0.018 | 0.022 |
| plot_file_content (scatter) | bgl_split_10 | 0.027 | 0.038 | 0.073 | 0.225 |
| plot_file_content (scatter+umap) | hadoop_renamed | err | 0.114 | 0.148 | 0.241 |
| plot_file_content (scatter+umap) | hdfs_balanced_5k | 0.010 | 0.011 | 0.014 | 0.016 |
| plot_file_content (scatter+umap) | bgl_split_10 | 0.024 | 0.030 | 0.103 | 0.215 |

# Part B -- warm (repeated call)

## Table B1 -- Auxiliary tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| peek_log_root | hadoop_renamed | 0.096 | 0.102 | 0.119 | 0.137 |
| peek_log_root | hdfs_balanced_5k | 0.134 | 0.128 | 0.167 | 0.186 |
| peek_log_root | bgl_split_10 | 0.021 | 0.021 | 0.025 | 0.027 |
| open_log_root | hadoop_renamed | 0.031 | 0.029 | 0.043 | 0.063 |
| open_log_root | hdfs_balanced_5k | 0.019 | 0.027 | 0.064 | 0.094 |
| open_log_root | bgl_split_10 | 0.074 | 0.104 | 0.444 | 1.075 |
| list_log_roots | hadoop_renamed | 0.016 | 0.010 | 0.013 | 0.020 |
| list_log_roots | hdfs_balanced_5k | 0.007 | 0.011 | 0.013 | 0.018 |
| list_log_roots | bgl_split_10 | 0.022 | 0.034 | 0.139 | 0.227 |
| describe_log_root | hadoop_renamed | 0.025 | 0.015 | 0.027 | 0.050 |
| describe_log_root | hdfs_balanced_5k | 0.017 | 0.023 | 0.035 | 0.040 |
| describe_log_root | bgl_split_10 | 0.050 | 0.064 | 0.286 | 0.467 |
| set_folder_names | hadoop_renamed | 0.036 | 0.070 | 0.110 | 0.342 |
| set_folder_names | hdfs_balanced_5k | 0.022 | 0.036 | 0.123 | 0.144 |
| set_folder_names | bgl_split_10 | 0.124 | 0.251 | 0.963 | 2.224 |
| read_log_lines | hadoop_renamed | 0.003 | 0.002 | 0.004 | 0.005 |
| read_log_lines | hdfs_balanced_5k | 0.002 | 0.002 | 0.003 | 0.004 |
| read_log_lines | bgl_split_10 | 0.006 | 0.008 | 0.026 | 0.027 |
| search_log_lines | hadoop_renamed | 0.011 | 0.007 | 0.015 | 0.019 |
| search_log_lines | hdfs_balanced_5k | 0.006 | 0.007 | 0.010 | 0.015 |
| search_log_lines | bgl_split_10 | 0.021 | 0.030 | 0.067 | 0.105 |
| read_log_lines (new tokens) | hadoop_renamed | 0.040 | 0.036 | 0.038 | 0.041 |
| read_log_lines (new tokens) | hdfs_balanced_5k | 0.027 | 0.029 | 0.031 | 0.033 |
| read_log_lines (new tokens) | bgl_split_10 | 0.052 | 0.065 | 0.136 | 0.240 |
| new_tokens | hadoop_renamed | 0.077 | 0.063 | 0.086 | 0.079 |
| new_tokens | hdfs_balanced_5k | 0.048 | 0.044 | 0.044 | 0.046 |
| new_tokens | bgl_split_10 | 0.084 | 0.089 | 0.162 | 0.214 |
| query_result | hadoop_renamed | 0.003 | 0.003 | 0.003 | 0.003 |
| query_result | hdfs_balanced_5k | 0.002 | 0.002 | 0.005 | 0.007 |
| query_result | bgl_split_10 | 0.003 | 0.003 | 0.002 | 0.003 |
| split_log_file | hadoop_renamed | 0.000 | 0.000 | 0.000 | 0.000 |
| split_log_file | hdfs_balanced_5k | 0.000 | 0.000 | 0.000 | 0.000 |
| split_log_file | bgl_split_10 | 0.000 | 0.000 | 0.000 | 0.000 |
| close_log_root | hadoop_renamed | 0.011 | 0.015 | 0.015 | 0.038 |
| close_log_root | hdfs_balanced_5k | 0.007 | 0.009 | 0.015 | 0.014 |
| close_log_root | bgl_split_10 | 0.029 | 0.045 | 0.157 | 0.264 |
| run_config | hadoop_renamed | 0.740 | 2.102 | 7.378 | 16.8 |
| run_config | hdfs_balanced_5k | 1.913 | 3.808 | 26.9 | 37.5 |
| run_config | bgl_split_10 | 7.259 | 11.3 | 56.2 | 159.7 |

## Table B2 -- Distance tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| distance_folder_filename | hadoop_renamed | 0.061 | 0.102 | 0.323 | 0.786 |
| distance_folder_filename | hdfs_balanced_5k | 1.629 | 3.754 | 19.1 | 38.4 |
| distance_folder_filename | bgl_split_10 | 0.115 | 0.145 | 0.282 | 0.447 |
| distance_folder_content | hadoop_renamed | 0.668 | 1.383 | 8.024 | 16.0 |
| distance_folder_content | hdfs_balanced_5k | 1.838 | 4.051 | 19.4 | 43.1 |
| distance_folder_content | bgl_split_10 | 5.896 | 10.6 | 55.0 | 155.1 |
| distance_file_content | hadoop_renamed | 0.912 | 1.665 | 7.704 | 16.7 |
| distance_file_content | hdfs_balanced_5k | 0.015 | 0.019 | 0.019 | 0.022 |
| distance_file_content | bgl_split_10 | 0.032 | 0.050 | 0.162 | 0.273 |
| distance_line_content | hadoop_renamed | 1.280 | 0.110 | 0.404 | 1.131 |
| distance_line_content | hdfs_balanced_5k | 0.761 | 1.558 | 8.185 | 16.6 |
| distance_line_content | bgl_split_10 | 0.108 | 0.140 | 0.250 | 0.378 |

## Table B3 -- Anomaly tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename | hadoop_renamed | 0.159 | 0.186 | 0.162 | 0.200 |
| anomaly_folder_filename | hdfs_balanced_5k | 0.144 | 0.148 | 0.177 | 0.225 |
| anomaly_folder_filename | bgl_split_10 | 0.205 | 0.293 | 0.856 | 1.433 |
| anomaly_folder_content | hadoop_renamed | 0.198 | 0.223 | 0.345 | 0.687 |
| anomaly_folder_content | hdfs_balanced_5k | 0.156 | 0.177 | 0.305 | 0.443 |
| anomaly_folder_content | bgl_split_10 | 0.306 | 0.501 | 1.677 | 6.760 |
| anomaly_file_content | hadoop_renamed | err | 2.543 | 2.472 | 3.045 |
| anomaly_file_content | hdfs_balanced_5k | 0.021 | 0.019 | 0.029 | 0.039 |
| anomaly_file_content | bgl_split_10 | 0.030 | 0.045 | 0.137 | 0.263 |
| anomaly_line_content | hadoop_renamed | 0.470 | 0.250 | 0.283 | 0.282 |
| anomaly_line_content | hdfs_balanced_5k | 0.017 | 0.017 | 0.023 | 0.033 |
| anomaly_line_content | bgl_split_10 | 0.034 | 0.056 | 0.155 | 0.261 |

## Table B4 -- Plot tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| plot_folder_filename | hadoop_renamed | 0.051 | 0.067 | 0.084 | 0.070 |
| plot_folder_filename | hdfs_balanced_5k | 0.066 | 0.062 | 0.092 | 0.129 |
| plot_folder_filename | bgl_split_10 | 0.078 | 0.096 | 0.220 | 0.309 |
| plot_folder_content (scatter) | hadoop_renamed | 0.078 | 0.120 | 0.253 | 0.542 |
| plot_folder_content (scatter) | hdfs_balanced_5k | 0.070 | 0.088 | 0.196 | 0.322 |
| plot_folder_content (scatter) | bgl_split_10 | 0.202 | 0.368 | 1.440 | 6.124 |
| plot_folder_content (scatter+umap) | hadoop_renamed | err | 0.157 | 0.308 | 0.697 |
| plot_folder_content (scatter+umap) | hdfs_balanced_5k | 0.423 | 1.028 | 8.575 | 8.919 |
| plot_folder_content (scatter+umap) | bgl_split_10 | 0.209 | 0.389 | 1.548 | 5.723 |
| plot_file_content (scatter) | hadoop_renamed | 0.083 | 0.071 | 0.075 | 0.108 |
| plot_file_content (scatter) | hdfs_balanced_5k | 0.010 | 0.012 | 0.015 | 0.018 |
| plot_file_content (scatter) | bgl_split_10 | 0.025 | 0.030 | 0.085 | 0.213 |
| plot_file_content (scatter+umap) | hadoop_renamed | err | 0.108 | 0.148 | 0.220 |
| plot_file_content (scatter+umap) | hdfs_balanced_5k | 0.012 | 0.014 | 0.016 | 0.018 |
| plot_file_content (scatter+umap) | bgl_split_10 | 0.025 | 0.037 | 0.102 | 0.212 |

# Detailed breakdowns (per detector / per measure)

The tables above run every anomaly tool with all four detectors, and `distance_folder_content`/`distance_file_content` with all four measures, at once. Part C/D below break the same figure down per detector / per measure run in isolation (`detectors=["<name>"]` / `measures=["<name>"]`), so the cost of narrowing either is visible on its own rather than folded into the combined call. `distance_folder_filename` (jaccard/overlap distance over file names only) and `distance_line_content` (a text diff, no measures) are not broken down further -- neither computes multiple vectorized measures in one pass.

# Part C -- cold (first call)

## Table C1 -- Anomaly tools detailed

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename (KMeans) | hadoop_renamed | 0.079 | 0.078 | 0.138 | 0.315 |
| anomaly_folder_filename (KMeans) | hdfs_balanced_5k | 0.086 | 0.085 | 0.097 | 0.148 |
| anomaly_folder_filename (KMeans) | bgl_split_10 | 0.157 | 0.255 | 0.911 | 1.984 |
| anomaly_folder_filename (IsolationForest) | hadoop_renamed | 0.199 | 0.206 | 0.351 | 0.376 |
| anomaly_folder_filename (IsolationForest) | hdfs_balanced_5k | 0.452 | 0.201 | 0.263 | 0.290 |
| anomaly_folder_filename (IsolationForest) | bgl_split_10 | 0.272 | 0.376 | 0.901 | 1.738 |
| anomaly_folder_filename (RarityModel) | hadoop_renamed | 0.030 | 0.036 | 0.061 | 0.137 |
| anomaly_folder_filename (RarityModel) | hdfs_balanced_5k | 0.062 | 0.039 | 0.063 | 0.091 |
| anomaly_folder_filename (RarityModel) | bgl_split_10 | 0.106 | 0.179 | 0.804 | 1.455 |
| anomaly_folder_filename (OOVDetector) | hadoop_renamed | 0.044 | 0.061 | 0.060 | 0.142 |
| anomaly_folder_filename (OOVDetector) | hdfs_balanced_5k | 0.056 | 0.034 | 0.070 | 0.181 |
| anomaly_folder_filename (OOVDetector) | bgl_split_10 | 0.114 | 0.200 | 0.696 | 1.604 |
| anomaly_folder_content (KMeans) | hadoop_renamed | 0.095 | 0.123 | 0.922 | 2.167 |
| anomaly_folder_content (KMeans) | hdfs_balanced_5k | 0.067 | 0.090 | 0.230 | 0.350 |
| anomaly_folder_content (KMeans) | bgl_split_10 | 0.259 | 0.515 | 2.656 | 10.6 |
| anomaly_folder_content (IsolationForest) | hadoop_renamed | 0.236 | 0.241 | 0.561 | 1.096 |
| anomaly_folder_content (IsolationForest) | hdfs_balanced_5k | 0.200 | 0.263 | 0.387 | 0.474 |
| anomaly_folder_content (IsolationForest) | bgl_split_10 | 0.380 | 0.586 | 2.207 | 10.5 |
| anomaly_folder_content (RarityModel) | hadoop_renamed | 0.064 | 0.099 | 0.341 | 1.263 |
| anomaly_folder_content (RarityModel) | hdfs_balanced_5k | 0.074 | 0.054 | 0.185 | 0.305 |
| anomaly_folder_content (RarityModel) | bgl_split_10 | 0.218 | 0.506 | 2.190 | 16.8 |
| anomaly_folder_content (OOVDetector) | hadoop_renamed | 0.087 | 0.235 | 0.256 | 1.500 |
| anomaly_folder_content (OOVDetector) | hdfs_balanced_5k | 0.087 | 0.051 | 0.170 | 0.318 |
| anomaly_folder_content (OOVDetector) | bgl_split_10 | 0.261 | 0.453 | 2.118 | OOM |
| anomaly_file_content (KMeans) | hadoop_renamed | err | 0.536 | 1.145 | 1.657 |
| anomaly_file_content (KMeans) | hdfs_balanced_5k | 0.032 | 0.041 | 0.046 | 0.043 |
| anomaly_file_content (KMeans) | bgl_split_10 | 0.043 | 0.054 | 0.107 | 0.231 |
| anomaly_file_content (IsolationForest) | hadoop_renamed | 2.657 | 2.853 | 5.588 | 5.619 |
| anomaly_file_content (IsolationForest) | hdfs_balanced_5k | 0.050 | 0.031 | 0.040 | 0.047 |
| anomaly_file_content (IsolationForest) | bgl_split_10 | 0.039 | 0.048 | 0.137 | 0.261 |
| anomaly_file_content (RarityModel) | hadoop_renamed | 0.326 | 0.413 | 0.832 | 1.228 |
| anomaly_file_content (RarityModel) | hdfs_balanced_5k | 0.050 | 0.029 | 0.043 | 0.045 |
| anomaly_file_content (RarityModel) | bgl_split_10 | 0.040 | 0.054 | 0.134 | 0.974 |
| anomaly_file_content (OOVDetector) | hadoop_renamed | 0.418 | 0.725 | 0.593 | 1.535 |
| anomaly_file_content (OOVDetector) | hdfs_balanced_5k | 0.053 | 0.027 | 0.039 | 0.048 |
| anomaly_file_content (OOVDetector) | bgl_split_10 | 0.043 | 0.045 | 0.139 | 0.403 |
| anomaly_line_content (KMeans) | hadoop_renamed | 0.359 | 0.267 | 0.442 | 0.469 |
| anomaly_line_content (KMeans) | hdfs_balanced_5k | 0.032 | 0.029 | 0.043 | 0.042 |
| anomaly_line_content (KMeans) | bgl_split_10 | 0.044 | 0.063 | 0.181 | 0.307 |
| anomaly_line_content (IsolationForest) | hadoop_renamed | 0.614 | 0.243 | 0.646 | 0.327 |
| anomaly_line_content (IsolationForest) | hdfs_balanced_5k | 0.036 | 0.026 | 0.032 | 0.044 |
| anomaly_line_content (IsolationForest) | bgl_split_10 | 0.045 | 0.061 | 0.180 | 0.333 |
| anomaly_line_content (RarityModel) | hadoop_renamed | 0.110 | 0.117 | 0.134 | 0.232 |
| anomaly_line_content (RarityModel) | hdfs_balanced_5k | 0.042 | 0.019 | 0.036 | 0.041 |
| anomaly_line_content (RarityModel) | bgl_split_10 | 0.050 | 0.066 | 0.190 | 0.265 |
| anomaly_line_content (OOVDetector) | hadoop_renamed | 0.098 | 0.112 | 0.126 | 0.384 |
| anomaly_line_content (OOVDetector) | hdfs_balanced_5k | 0.048 | 0.020 | 0.038 | 0.045 |
| anomaly_line_content (OOVDetector) | bgl_split_10 | 0.054 | 0.057 | 0.182 | 0.405 |

## Table C2 -- Distance tools detailed

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| distance_folder_content (cosine) | hadoop_renamed | 0.121 | 0.296 | 1.228 | 3.681 |
| distance_folder_content (cosine) | hdfs_balanced_5k | 1.713 | 2.905 | 14.6 | 29.8 |
| distance_folder_content (cosine) | bgl_split_10 | 0.861 | 1.418 | 5.690 | 21.0 |
| distance_folder_content (jaccard) | hadoop_renamed | 0.122 | 0.603 | 1.566 | 4.287 |
| distance_folder_content (jaccard) | hdfs_balanced_5k | 1.747 | 3.279 | 16.8 | 35.1 |
| distance_folder_content (jaccard) | bgl_split_10 | 0.774 | 1.255 | 5.571 | 18.5 |
| distance_folder_content (compression) | hadoop_renamed | 0.713 | 3.382 | 14.6 | 19.3 |
| distance_folder_content (compression) | hdfs_balanced_5k | 1.590 | 3.359 | 16.8 | 32.8 |
| distance_folder_content (compression) | bgl_split_10 | 7.818 | 14.7 | 80.1 | 209.1 |
| distance_folder_content (containment) | hadoop_renamed | 0.106 | 0.441 | 1.408 | 2.766 |
| distance_folder_content (containment) | hdfs_balanced_5k | 1.219 | 2.483 | 13.5 | 27.5 |
| distance_folder_content (containment) | bgl_split_10 | 0.716 | 1.381 | 6.927 | 16.7 |
| distance_file_content (cosine) | hadoop_renamed | 0.240 | 0.801 | 2.345 | 7.382 |
| distance_file_content (cosine) | hdfs_balanced_5k | 0.026 | 0.026 | 0.029 | 0.032 |
| distance_file_content (cosine) | bgl_split_10 | 0.055 | 0.062 | 0.225 | 0.540 |
| distance_file_content (jaccard) | hadoop_renamed | 0.282 | 1.031 | 3.026 | 7.701 |
| distance_file_content (jaccard) | hdfs_balanced_5k | 0.024 | 0.024 | 0.024 | 0.037 |
| distance_file_content (jaccard) | bgl_split_10 | 0.046 | 0.064 | 0.231 | 0.394 |
| distance_file_content (compression) | hadoop_renamed | 0.971 | 2.919 | 13.6 | 21.3 |
| distance_file_content (compression) | hdfs_balanced_5k | 0.020 | 0.026 | 0.028 | 0.028 |
| distance_file_content (compression) | bgl_split_10 | 0.045 | 0.065 | 0.226 | 0.339 |
| distance_file_content (containment) | hadoop_renamed | 0.234 | 1.004 | 4.454 | 4.910 |
| distance_file_content (containment) | hdfs_balanced_5k | 0.025 | 0.026 | 0.028 | 0.032 |
| distance_file_content (containment) | bgl_split_10 | 0.056 | 0.067 | 0.178 | 0.326 |

# Part D -- warm (repeated call)

## Table D1 -- Anomaly tools detailed

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename (KMeans) | hadoop_renamed | 0.046 | 0.052 | 0.090 | 0.190 |
| anomaly_folder_filename (KMeans) | hdfs_balanced_5k | 0.054 | 0.057 | 0.077 | 0.098 |
| anomaly_folder_filename (KMeans) | bgl_split_10 | 0.123 | 0.171 | 0.737 | 1.806 |
| anomaly_folder_filename (IsolationForest) | hadoop_renamed | 0.201 | 0.200 | 0.243 | 0.366 |
| anomaly_folder_filename (IsolationForest) | hdfs_balanced_5k | 0.188 | 0.211 | 0.226 | 0.219 |
| anomaly_folder_filename (IsolationForest) | bgl_split_10 | 0.249 | 0.296 | 0.787 | 1.403 |
| anomaly_folder_filename (RarityModel) | hadoop_renamed | 0.027 | 0.037 | 0.070 | 0.136 |
| anomaly_folder_filename (RarityModel) | hdfs_balanced_5k | 0.055 | 0.032 | 0.068 | 0.096 |
| anomaly_folder_filename (RarityModel) | bgl_split_10 | 0.099 | 0.178 | 0.578 | 1.365 |
| anomaly_folder_filename (OOVDetector) | hadoop_renamed | 0.037 | 0.095 | 0.060 | 0.128 |
| anomaly_folder_filename (OOVDetector) | hdfs_balanced_5k | 0.060 | 0.034 | 0.069 | 0.092 |
| anomaly_folder_filename (OOVDetector) | bgl_split_10 | 0.118 | 0.185 | 0.710 | 1.351 |
| anomaly_folder_content (KMeans) | hadoop_renamed | 0.081 | 0.111 | 0.438 | 1.291 |
| anomaly_folder_content (KMeans) | hdfs_balanced_5k | 0.065 | 0.085 | 0.201 | 0.317 |
| anomaly_folder_content (KMeans) | bgl_split_10 | 0.253 | 0.390 | 2.157 | 10.2 |
| anomaly_folder_content (IsolationForest) | hadoop_renamed | 0.210 | 0.228 | 0.596 | 0.935 |
| anomaly_folder_content (IsolationForest) | hdfs_balanced_5k | 0.219 | 0.245 | 0.367 | 0.478 |
| anomaly_folder_content (IsolationForest) | bgl_split_10 | 0.359 | 0.547 | 2.300 | 11.4 |
| anomaly_folder_content (RarityModel) | hadoop_renamed | 0.061 | 0.103 | 0.470 | 1.072 |
| anomaly_folder_content (RarityModel) | hdfs_balanced_5k | 0.076 | 0.057 | 0.176 | 0.305 |
| anomaly_folder_content (RarityModel) | bgl_split_10 | 0.200 | 0.499 | 1.990 | 11.5 |
| anomaly_folder_content (OOVDetector) | hadoop_renamed | 0.078 | 0.180 | 0.235 | 1.059 |
| anomaly_folder_content (OOVDetector) | hdfs_balanced_5k | 0.086 | 0.055 | 0.168 | 0.314 |
| anomaly_folder_content (OOVDetector) | bgl_split_10 | 0.254 | 0.388 | 2.046 | OOM |
| anomaly_file_content (KMeans) | hadoop_renamed | err | 0.521 | 1.121 | 1.315 |
| anomaly_file_content (KMeans) | hdfs_balanced_5k | 0.038 | 0.032 | 0.040 | 0.048 |
| anomaly_file_content (KMeans) | bgl_split_10 | 0.042 | 0.054 | 0.174 | 0.250 |
| anomaly_file_content (IsolationForest) | hadoop_renamed | 2.699 | 2.594 | 5.476 | 4.971 |
| anomaly_file_content (IsolationForest) | hdfs_balanced_5k | 0.053 | 0.032 | 0.036 | 0.048 |
| anomaly_file_content (IsolationForest) | bgl_split_10 | 0.040 | 0.054 | 0.175 | 0.345 |
| anomaly_file_content (RarityModel) | hadoop_renamed | 0.356 | 0.485 | 0.632 | 1.112 |
| anomaly_file_content (RarityModel) | hdfs_balanced_5k | 0.050 | 0.025 | 0.040 | 0.044 |
| anomaly_file_content (RarityModel) | bgl_split_10 | 0.038 | 0.059 | 0.189 | 0.308 |
| anomaly_file_content (OOVDetector) | hadoop_renamed | 0.376 | 0.504 | 0.611 | 1.365 |
| anomaly_file_content (OOVDetector) | hdfs_balanced_5k | 0.053 | 0.026 | 0.040 | 0.051 |
| anomaly_file_content (OOVDetector) | bgl_split_10 | 0.048 | 0.045 | 0.181 | 0.349 |
| anomaly_line_content (KMeans) | hadoop_renamed | 0.104 | 0.105 | 0.189 | 0.184 |
| anomaly_line_content (KMeans) | hdfs_balanced_5k | 0.039 | 0.027 | 0.039 | 0.044 |
| anomaly_line_content (KMeans) | bgl_split_10 | 0.047 | 0.066 | 0.177 | 0.314 |
| anomaly_line_content (IsolationForest) | hadoop_renamed | 0.238 | 0.243 | 0.505 | 0.594 |
| anomaly_line_content (IsolationForest) | hdfs_balanced_5k | 0.043 | 0.022 | 0.033 | 0.043 |
| anomaly_line_content (IsolationForest) | bgl_split_10 | 0.044 | 0.063 | 0.176 | 0.311 |
| anomaly_line_content (RarityModel) | hadoop_renamed | 0.107 | 0.140 | 0.118 | 0.284 |
| anomaly_line_content (RarityModel) | hdfs_balanced_5k | 0.037 | 0.021 | 0.033 | 0.042 |
| anomaly_line_content (RarityModel) | bgl_split_10 | 0.043 | 0.066 | 0.163 | 0.271 |
| anomaly_line_content (OOVDetector) | hadoop_renamed | 0.095 | 0.106 | 0.118 | 0.230 |
| anomaly_line_content (OOVDetector) | hdfs_balanced_5k | 0.042 | 0.019 | 0.036 | 0.046 |
| anomaly_line_content (OOVDetector) | bgl_split_10 | 0.053 | 0.055 | 0.170 | 0.343 |

## Table D2 -- Distance tools detailed

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| distance_folder_content (cosine) | hadoop_renamed | 0.114 | 0.310 | 1.229 | 3.510 |
| distance_folder_content (cosine) | hdfs_balanced_5k | 1.433 | 2.841 | 14.3 | 30.9 |
| distance_folder_content (cosine) | bgl_split_10 | 0.696 | 1.702 | 5.619 | 17.6 |
| distance_folder_content (jaccard) | hadoop_renamed | 0.127 | 0.478 | 1.255 | 4.086 |
| distance_folder_content (jaccard) | hdfs_balanced_5k | 1.701 | 3.179 | 16.6 | 34.8 |
| distance_folder_content (jaccard) | bgl_split_10 | 0.825 | 1.160 | 5.653 | 16.7 |
| distance_folder_content (compression) | hadoop_renamed | 0.708 | 1.936 | 14.9 | 20.3 |
| distance_folder_content (compression) | hdfs_balanced_5k | 1.515 | 3.207 | 16.2 | 32.6 |
| distance_folder_content (compression) | bgl_split_10 | 8.572 | 15.5 | 96.5 | 225.0 |
| distance_folder_content (containment) | hadoop_renamed | 0.121 | 0.374 | 1.431 | 2.953 |
| distance_folder_content (containment) | hdfs_balanced_5k | 1.216 | 2.686 | 12.9 | 26.0 |
| distance_folder_content (containment) | bgl_split_10 | 0.714 | 1.251 | 5.905 | 16.4 |
| distance_file_content (cosine) | hadoop_renamed | 0.260 | 0.968 | 2.473 | 7.492 |
| distance_file_content (cosine) | hdfs_balanced_5k | 0.023 | 0.021 | 0.026 | 0.031 |
| distance_file_content (cosine) | bgl_split_10 | 0.054 | 0.065 | 0.189 | 0.374 |
| distance_file_content (jaccard) | hadoop_renamed | 0.262 | 1.132 | 3.396 | 7.813 |
| distance_file_content (jaccard) | hdfs_balanced_5k | 0.021 | 0.023 | 0.026 | 0.036 |
| distance_file_content (jaccard) | bgl_split_10 | 0.045 | 0.067 | 0.172 | 0.366 |
| distance_file_content (compression) | hadoop_renamed | 0.792 | 2.251 | 12.7 | 19.2 |
| distance_file_content (compression) | hdfs_balanced_5k | 0.021 | 0.021 | 0.025 | 0.030 |
| distance_file_content (compression) | bgl_split_10 | 0.047 | 0.062 | 0.231 | 0.351 |
| distance_file_content (containment) | hadoop_renamed | 0.219 | 0.745 | 3.306 | 4.669 |
| distance_file_content (containment) | hdfs_balanced_5k | 0.026 | 0.025 | 0.027 | 0.028 |
| distance_file_content (containment) | bgl_split_10 | 0.057 | 0.062 | 0.200 | 0.329 |
