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
| peek_log_root | hadoop_renamed | 0.104 | 0.114 | 0.144 | 0.207 |
| peek_log_root | hdfs_balanced_5k | 0.163 | 0.183 | 0.192 | 0.305 |
| peek_log_root | bgl_split_10 | 0.071 | 0.084 | 0.096 | 0.124 |
| open_log_root | hadoop_renamed | 1.076 | 1.477 | 2.742 | 5.490 |
| open_log_root | hdfs_balanced_5k | 1.358 | 1.605 | 2.926 | 5.219 |
| open_log_root | bgl_split_10 | 1.291 | 2.387 | 12.1 | 26.5 |
| list_log_roots | hadoop_renamed | 0.010 | 0.012 | 0.016 | 0.021 |
| list_log_roots | hdfs_balanced_5k | 0.009 | 0.015 | 0.016 | 0.024 |
| list_log_roots | bgl_split_10 | 0.023 | 0.037 | 0.149 | 0.256 |
| describe_log_root | hadoop_renamed | 0.013 | 0.016 | 0.021 | 0.046 |
| describe_log_root | hdfs_balanced_5k | 0.016 | 0.021 | 0.027 | 0.051 |
| describe_log_root | bgl_split_10 | 0.046 | 0.068 | 0.351 | 0.522 |
| set_folder_names | hadoop_renamed | 0.034 | 0.039 | 0.110 | 0.192 |
| set_folder_names | hdfs_balanced_5k | 0.024 | 0.036 | 0.123 | 0.144 |
| set_folder_names | bgl_split_10 | 0.109 | 0.155 | 0.947 | 2.319 |
| read_log_lines | hadoop_renamed | 0.002 | 0.003 | 0.003 | 0.004 |
| read_log_lines | hdfs_balanced_5k | 0.001 | 0.003 | 0.003 | 0.004 |
| read_log_lines | bgl_split_10 | 0.008 | 0.005 | 0.019 | 0.025 |
| search_log_lines | hadoop_renamed | 0.007 | 0.009 | 0.011 | 0.018 |
| search_log_lines | hdfs_balanced_5k | 0.006 | 0.007 | 0.011 | 0.015 |
| search_log_lines | bgl_split_10 | 0.027 | 0.033 | 0.092 | 0.102 |
| query_result | hadoop_renamed | 0.004 | 0.003 | 0.003 | 0.003 |
| query_result | hdfs_balanced_5k | 0.002 | 0.003 | 0.008 | 0.005 |
| query_result | bgl_split_10 | 0.003 | 0.003 | 0.003 | 0.003 |
| split_log_file | hadoop_renamed | 0.003 | 0.003 | 0.022 | 0.039 |
| split_log_file | hdfs_balanced_5k | 0.001 | 0.001 | 0.001 | 0.001 |
| split_log_file | bgl_split_10 | 0.012 | 0.024 | 0.146 | 0.437 |
| close_log_root | hadoop_renamed | 0.009 | 0.011 | 0.015 | 0.038 |
| close_log_root | hdfs_balanced_5k | 0.007 | 0.009 | 0.015 | 0.014 |
| close_log_root | bgl_split_10 | 0.029 | 0.045 | 0.157 | 0.264 |
| run_config | hadoop_renamed | 0.823 | 1.579 | 7.130 | 15.6 |
| run_config | hdfs_balanced_5k | 1.945 | 3.693 | 18.9 | 38.7 |
| run_config | bgl_split_10 | 12.2 | 12.2 | 57.3 | 180.3 |

## Table A2 -- Distance tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| distance_folder_filename | hadoop_renamed | 0.035 | 0.074 | 0.316 | 0.715 |
| distance_folder_filename | hdfs_balanced_5k | 1.659 | 4.297 | 19.4 | 47.9 |
| distance_folder_filename | bgl_split_10 | 0.123 | 0.173 | 0.323 | 0.431 |
| distance_folder_content | hadoop_renamed | 0.615 | 1.388 | 6.821 | 15.4 |
| distance_folder_content | hdfs_balanced_5k | 1.814 | 3.960 | 20.1 | 41.5 |
| distance_folder_content | bgl_split_10 | 5.805 | 11.5 | 56.0 | 147.7 |
| distance_file_content | hadoop_renamed | 0.731 | 1.563 | 8.271 | 16.9 |
| distance_file_content | hdfs_balanced_5k | 0.019 | 0.025 | 0.021 | 0.023 |
| distance_file_content | bgl_split_10 | 0.034 | 0.052 | 0.167 | 0.275 |
| distance_line_content | hadoop_renamed | 0.031 | 0.080 | 0.359 | 1.284 |
| distance_line_content | hdfs_balanced_5k | 0.724 | 1.588 | 8.571 | 16.2 |
| distance_line_content | bgl_split_10 | 0.089 | 0.117 | 0.246 | 0.327 |

## Table A3 -- Anomaly tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename | hadoop_renamed | 0.158 | 0.200 | 0.199 | 0.233 |
| anomaly_folder_filename | hdfs_balanced_5k | 0.159 | 0.173 | 0.194 | 0.260 |
| anomaly_folder_filename | bgl_split_10 | 0.242 | 0.343 | 0.931 | 1.559 |
| anomaly_folder_content | hadoop_renamed | 0.234 | 0.298 | 0.363 | 0.786 |
| anomaly_folder_content | hdfs_balanced_5k | 0.161 | 0.178 | 0.314 | 0.528 |
| anomaly_folder_content | bgl_split_10 | 0.318 | 0.544 | 1.865 | 8.044 |
| anomaly_file_content | hadoop_renamed | err | 2.482 | 2.404 | 3.285 |
| anomaly_file_content | hdfs_balanced_5k | 0.021 | 0.022 | 0.028 | 0.036 |
| anomaly_file_content | bgl_split_10 | 0.032 | 0.041 | 0.100 | 0.261 |
| anomaly_line_content | hadoop_renamed | 0.420 | 0.332 | 0.379 | 0.389 |
| anomaly_line_content | hdfs_balanced_5k | 0.017 | 0.016 | 0.026 | 0.032 |
| anomaly_line_content | bgl_split_10 | 0.032 | 0.056 | 0.150 | 0.269 |

## Table A4 -- Plot tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| plot_folder_filename | hadoop_renamed | 0.057 | 0.062 | 0.067 | 0.071 |
| plot_folder_filename | hdfs_balanced_5k | 0.119 | 0.117 | 0.140 | 0.185 |
| plot_folder_filename | bgl_split_10 | 0.127 | 0.133 | 0.292 | 0.371 |
| plot_folder_content (scatter) | hadoop_renamed | 0.092 | 0.106 | 0.350 | 0.607 |
| plot_folder_content (scatter) | hdfs_balanced_5k | 0.075 | 0.100 | 0.190 | 0.317 |
| plot_folder_content (scatter) | bgl_split_10 | 0.208 | 0.421 | 1.590 | 6.933 |
| plot_folder_content (scatter+umap) | hadoop_renamed | err | 9.633 | 8.078 | 10.9 |
| plot_folder_content (scatter+umap) | hdfs_balanced_5k | 8.490 | 9.033 | 16.9 | 27.8 |
| plot_folder_content (scatter+umap) | bgl_split_10 | 7.730 | 8.101 | 9.375 | 14.7 |
| plot_file_content (scatter) | hadoop_renamed | 0.061 | 0.064 | 0.072 | 0.087 |
| plot_file_content (scatter) | hdfs_balanced_5k | 0.012 | 0.011 | 0.018 | 0.022 |
| plot_file_content (scatter) | bgl_split_10 | 0.027 | 0.038 | 0.073 | 0.225 |
| plot_file_content (scatter+umap) | hadoop_renamed | err | 0.103 | 0.148 | 0.241 |
| plot_file_content (scatter+umap) | hdfs_balanced_5k | 0.010 | 0.011 | 0.014 | 0.016 |
| plot_file_content (scatter+umap) | bgl_split_10 | 0.024 | 0.030 | 0.103 | 0.215 |

# Part B -- warm (repeated call)

## Table B1 -- Auxiliary tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| peek_log_root | hadoop_renamed | 0.094 | 0.126 | 0.108 | 0.117 |
| peek_log_root | hdfs_balanced_5k | 0.152 | 0.144 | 0.179 | 0.284 |
| peek_log_root | bgl_split_10 | 0.022 | 0.028 | 0.030 | 0.033 |
| open_log_root | hadoop_renamed | 0.017 | 0.024 | 0.049 | 0.063 |
| open_log_root | hdfs_balanced_5k | 0.019 | 0.027 | 0.064 | 0.094 |
| open_log_root | bgl_split_10 | 0.074 | 0.104 | 0.444 | 1.075 |
| list_log_roots | hadoop_renamed | 0.007 | 0.008 | 0.011 | 0.020 |
| list_log_roots | hdfs_balanced_5k | 0.007 | 0.011 | 0.013 | 0.018 |
| list_log_roots | bgl_split_10 | 0.022 | 0.034 | 0.139 | 0.227 |
| describe_log_root | hadoop_renamed | 0.011 | 0.015 | 0.020 | 0.050 |
| describe_log_root | hdfs_balanced_5k | 0.017 | 0.023 | 0.035 | 0.040 |
| describe_log_root | bgl_split_10 | 0.050 | 0.064 | 0.286 | 0.467 |
| set_folder_names | hadoop_renamed | 0.034 | 0.039 | 0.110 | 0.342 |
| set_folder_names | hdfs_balanced_5k | 0.022 | 0.036 | 0.123 | 0.144 |
| set_folder_names | bgl_split_10 | 0.124 | 0.251 | 0.963 | 2.224 |
| read_log_lines | hadoop_renamed | 0.001 | 0.001 | 0.003 | 0.005 |
| read_log_lines | hdfs_balanced_5k | 0.002 | 0.002 | 0.003 | 0.004 |
| read_log_lines | bgl_split_10 | 0.006 | 0.008 | 0.026 | 0.027 |
| search_log_lines | hadoop_renamed | 0.007 | 0.007 | 0.011 | 0.019 |
| search_log_lines | hdfs_balanced_5k | 0.006 | 0.007 | 0.010 | 0.015 |
| search_log_lines | bgl_split_10 | 0.021 | 0.030 | 0.067 | 0.105 |
| query_result | hadoop_renamed | 0.003 | 0.003 | 0.003 | 0.003 |
| query_result | hdfs_balanced_5k | 0.002 | 0.002 | 0.005 | 0.007 |
| query_result | bgl_split_10 | 0.003 | 0.003 | 0.002 | 0.003 |
| split_log_file | hadoop_renamed | 0.000 | 0.000 | 0.000 | 0.000 |
| split_log_file | hdfs_balanced_5k | 0.000 | 0.000 | 0.000 | 0.000 |
| split_log_file | bgl_split_10 | 0.000 | 0.000 | 0.000 | 0.000 |
| close_log_root | hadoop_renamed | 0.009 | 0.011 | 0.015 | 0.038 |
| close_log_root | hdfs_balanced_5k | 0.007 | 0.009 | 0.015 | 0.014 |
| close_log_root | bgl_split_10 | 0.029 | 0.045 | 0.157 | 0.264 |
| run_config | hadoop_renamed | 0.787 | 1.436 | 7.378 | 16.8 |
| run_config | hdfs_balanced_5k | 1.913 | 3.808 | 26.9 | 37.5 |
| run_config | bgl_split_10 | 7.259 | 11.3 | 56.2 | 159.7 |

## Table B2 -- Distance tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| distance_folder_filename | hadoop_renamed | 0.035 | 0.075 | 0.293 | 0.786 |
| distance_folder_filename | hdfs_balanced_5k | 1.629 | 3.754 | 19.1 | 38.4 |
| distance_folder_filename | bgl_split_10 | 0.115 | 0.145 | 0.282 | 0.447 |
| distance_folder_content | hadoop_renamed | 0.603 | 1.338 | 7.992 | 16.0 |
| distance_folder_content | hdfs_balanced_5k | 1.838 | 4.051 | 19.4 | 43.1 |
| distance_folder_content | bgl_split_10 | 5.896 | 10.6 | 55.0 | 155.1 |
| distance_file_content | hadoop_renamed | 0.743 | 1.813 | 7.704 | 16.7 |
| distance_file_content | hdfs_balanced_5k | 0.015 | 0.019 | 0.019 | 0.022 |
| distance_file_content | bgl_split_10 | 0.032 | 0.050 | 0.162 | 0.273 |
| distance_line_content | hadoop_renamed | 0.033 | 0.080 | 0.404 | 1.131 |
| distance_line_content | hdfs_balanced_5k | 0.761 | 1.558 | 8.185 | 16.6 |
| distance_line_content | bgl_split_10 | 0.108 | 0.140 | 0.250 | 0.378 |

## Table B3 -- Anomaly tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename | hadoop_renamed | 0.147 | 0.213 | 0.162 | 0.200 |
| anomaly_folder_filename | hdfs_balanced_5k | 0.144 | 0.148 | 0.177 | 0.225 |
| anomaly_folder_filename | bgl_split_10 | 0.205 | 0.293 | 0.856 | 1.433 |
| anomaly_folder_content | hadoop_renamed | 0.216 | 0.268 | 0.345 | 0.687 |
| anomaly_folder_content | hdfs_balanced_5k | 0.156 | 0.177 | 0.305 | 0.443 |
| anomaly_folder_content | bgl_split_10 | 0.306 | 0.501 | 1.677 | 6.760 |
| anomaly_file_content | hadoop_renamed | err | 2.399 | 2.472 | 3.045 |
| anomaly_file_content | hdfs_balanced_5k | 0.021 | 0.019 | 0.029 | 0.039 |
| anomaly_file_content | bgl_split_10 | 0.030 | 0.045 | 0.137 | 0.263 |
| anomaly_line_content | hadoop_renamed | 0.214 | 0.239 | 0.283 | 0.282 |
| anomaly_line_content | hdfs_balanced_5k | 0.017 | 0.017 | 0.023 | 0.033 |
| anomaly_line_content | bgl_split_10 | 0.034 | 0.056 | 0.155 | 0.261 |

## Table B4 -- Plot tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| plot_folder_filename | hadoop_renamed | 0.061 | 0.059 | 0.084 | 0.070 |
| plot_folder_filename | hdfs_balanced_5k | 0.066 | 0.062 | 0.092 | 0.129 |
| plot_folder_filename | bgl_split_10 | 0.078 | 0.096 | 0.220 | 0.309 |
| plot_folder_content (scatter) | hadoop_renamed | 0.097 | 0.105 | 0.253 | 0.542 |
| plot_folder_content (scatter) | hdfs_balanced_5k | 0.070 | 0.088 | 0.196 | 0.322 |
| plot_folder_content (scatter) | bgl_split_10 | 0.202 | 0.368 | 1.440 | 6.124 |
| plot_folder_content (scatter+umap) | hadoop_renamed | err | 0.145 | 0.308 | 0.697 |
| plot_folder_content (scatter+umap) | hdfs_balanced_5k | 0.423 | 1.028 | 8.575 | 8.919 |
| plot_folder_content (scatter+umap) | bgl_split_10 | 0.209 | 0.389 | 1.548 | 5.723 |
| plot_file_content (scatter) | hadoop_renamed | 0.064 | 0.066 | 0.075 | 0.108 |
| plot_file_content (scatter) | hdfs_balanced_5k | 0.010 | 0.012 | 0.015 | 0.018 |
| plot_file_content (scatter) | bgl_split_10 | 0.025 | 0.030 | 0.085 | 0.213 |
| plot_file_content (scatter+umap) | hadoop_renamed | err | 0.098 | 0.148 | 0.220 |
| plot_file_content (scatter+umap) | hdfs_balanced_5k | 0.012 | 0.014 | 0.016 | 0.018 |
| plot_file_content (scatter+umap) | bgl_split_10 | 0.025 | 0.037 | 0.102 | 0.212 |
