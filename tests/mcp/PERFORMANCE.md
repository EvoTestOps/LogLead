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
| peek_log_root | hadoop_renamed | 0.237 | 0.105 | 0.155 | 0.144 |
| peek_log_root | hdfs_balanced_5k | 0.136 | 0.125 | 0.160 | 0.177 |
| peek_log_root | bgl_split_10 | 0.084 | 0.081 | 0.080 | 0.086 |
| open_log_root | hadoop_renamed | 1.059 | 1.340 | 3.229 | 6.061 |
| open_log_root | hdfs_balanced_5k | 1.305 | 1.495 | 2.903 | 4.690 |
| open_log_root | bgl_split_10 | 1.508 | 2.696 | 14.0 | 36.2 |
| open_log_root (no parsers) | hadoop_renamed | 0.979 | 1.201 | 3.178 | 5.441 |
| open_log_root (no parsers) | hdfs_balanced_5k | 1.204 | 1.402 | 2.810 | 4.461 |
| open_log_root (no parsers) | bgl_split_10 | 1.172 | 2.076 | 11.2 | 28.8 |
| open_log_root (parse tip) | hadoop_renamed | 0.948 | 1.218 | 3.114 | 6.561 |
| open_log_root (parse tip) | hdfs_balanced_5k | 1.311 | 1.579 | 2.897 | 4.847 |
| open_log_root (parse tip) | bgl_split_10 | 1.462 | 2.644 | 13.7 | 33.6 |
| open_log_root (parse drain) | hadoop_renamed | 1.099 | 1.457 | 4.039 | 8.140 |
| open_log_root (parse drain) | hdfs_balanced_5k | 1.383 | 1.516 | 3.249 | 5.168 |
| open_log_root (parse drain) | bgl_split_10 | 2.450 | 4.485 | 21.5 | 49.7 |
| list_log_roots | hadoop_renamed | 0.007 | 0.007 | 0.013 | 0.019 |
| list_log_roots | hdfs_balanced_5k | 0.012 | 0.008 | 0.009 | 0.015 |
| list_log_roots | bgl_split_10 | 0.022 | 0.031 | 0.143 | 0.302 |
| describe_log_root | hadoop_renamed | 0.011 | 0.012 | 0.018 | 0.028 |
| describe_log_root | hdfs_balanced_5k | 0.018 | 0.019 | 0.022 | 0.026 |
| describe_log_root | bgl_split_10 | 0.028 | 0.060 | 0.294 | 0.386 |
| set_folder_names | hadoop_renamed | 0.024 | 0.041 | 0.088 | 0.600 |
| set_folder_names | hdfs_balanced_5k | 0.025 | 0.033 | 0.067 | 0.105 |
| set_folder_names | bgl_split_10 | 0.087 | 0.135 | 0.853 | 5.144 |
| read_log_lines | hadoop_renamed | 0.005 | 0.005 | 0.005 | 0.004 |
| read_log_lines | hdfs_balanced_5k | 0.005 | 0.004 | 0.004 | 0.006 |
| read_log_lines | bgl_split_10 | 0.004 | 0.007 | 0.011 | 0.019 |
| search_log_lines | hadoop_renamed | 0.008 | 0.008 | 0.014 | 0.015 |
| search_log_lines | hdfs_balanced_5k | 0.011 | 0.012 | 0.013 | 0.018 |
| search_log_lines | bgl_split_10 | 0.018 | 0.029 | 0.049 | 0.172 |
| filter_log_lines | hadoop_renamed | 0.051 | 0.041 | 0.057 | 0.293 |
| filter_log_lines | hdfs_balanced_5k | 0.038 | 0.043 | 0.055 | 0.074 |
| filter_log_lines | bgl_split_10 | 0.071 | 0.114 | 0.387 | 1.145 |
| new_tokens | hadoop_renamed | 0.063 | 0.060 | 0.082 | 0.173 |
| new_tokens | hdfs_balanced_5k | 0.050 | 0.053 | 0.069 | 0.085 |
| new_tokens | bgl_split_10 | 0.093 | 0.130 | 0.344 | 1.232 |
| query_result | hadoop_renamed | 0.003 | 0.003 | 0.003 | 0.002 |
| query_result | hdfs_balanced_5k | 0.003 | 0.003 | 0.007 | 0.005 |
| query_result | bgl_split_10 | 0.003 | 0.003 | 0.003 | 0.016 |
| split_log_file | hadoop_renamed | 0.003 | 0.003 | 0.030 | 0.035 |
| split_log_file | hdfs_balanced_5k | 0.001 | 0.001 | 0.001 | 0.002 |
| split_log_file | bgl_split_10 | 0.012 | 0.024 | 0.122 | 0.329 |
| close_log_root | hadoop_renamed | 0.009 | 0.009 | 0.011 | 0.024 |
| close_log_root | hdfs_balanced_5k | 0.008 | 0.009 | 0.011 | 0.014 |
| close_log_root | bgl_split_10 | 0.031 | 0.050 | 0.212 | 0.290 |
| run_config | hadoop_renamed | 0.243 | 0.434 | 1.289 | 2.282 |
| run_config | hdfs_balanced_5k | 1.408 | 2.552 | 12.1 | 25.6 |
| run_config | bgl_split_10 | 1.154 | 1.714 | 7.225 | 19.6 |

## Table A2 -- Distance tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| distance_folder_filename | hadoop_renamed | 0.047 | 0.100 | 0.386 | 0.714 |
| distance_folder_filename | hdfs_balanced_5k | 2.125 | 4.173 | 21.1 | 42.4 |
| distance_folder_filename | bgl_split_10 | 0.127 | 0.158 | 0.347 | 0.576 |
| distance_folder_filename (threshold=False) | hadoop_renamed | 0.040 | 0.081 | 0.354 | 0.627 |
| distance_folder_filename (threshold=False) | hdfs_balanced_5k | 2.120 | 4.176 | 21.0 | 42.7 |
| distance_folder_filename (threshold=False) | bgl_split_10 | 0.105 | 0.123 | 0.251 | 0.390 |
| distance_folder_content | hadoop_renamed | 0.102 | 0.317 | 1.141 | 2.102 |
| distance_folder_content | hdfs_balanced_5k | 2.029 | 3.882 | 19.1 | 39.0 |
| distance_folder_content | bgl_split_10 | 0.806 | 1.310 | 5.925 | 18.3 |
| distance_folder_content (threshold=False) | hadoop_renamed | 0.092 | 0.192 | 0.944 | 1.885 |
| distance_folder_content (threshold=False) | hdfs_balanced_5k | 1.915 | 3.713 | 19.6 | 39.9 |
| distance_folder_content (threshold=False) | bgl_split_10 | 0.525 | 0.866 | 4.080 | 11.0 |
| distance_file_content | hadoop_renamed | 0.246 | 0.831 | 2.974 | 5.592 |
| distance_file_content | hdfs_balanced_5k | 0.026 | 0.023 | 0.026 | 0.026 |
| distance_file_content | bgl_split_10 | 0.038 | 0.052 | 0.175 | 0.315 |
| distance_file_content (threshold=False) | hadoop_renamed | 0.209 | 0.512 | 2.328 | 5.014 |
| distance_file_content (threshold=False) | hdfs_balanced_5k | 0.026 | 0.022 | 0.030 | 0.033 |
| distance_file_content (threshold=False) | bgl_split_10 | 0.039 | 0.043 | 0.146 | 0.269 |
| log_line_clustering | hadoop_renamed | 0.104 | 0.083 | 0.103 | 0.322 |
| log_line_clustering | hdfs_balanced_5k | 0.033 | 0.033 | 0.044 | 0.048 |
| log_line_clustering | bgl_split_10 | 0.041 | 0.061 | 0.172 | 0.325 |

## Table A3 -- Anomaly tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename | hadoop_renamed | 0.423 | 0.797 | 1.524 | 2.846 |
| anomaly_folder_filename | hdfs_balanced_5k | 1.347 | 1.400 | 1.613 | 1.924 |
| anomaly_folder_filename | bgl_split_10 | 1.690 | 1.994 | 4.874 | 8.532 |
| anomaly_folder_filename (threshold=False) | hadoop_renamed | 0.133 | 0.146 | 0.150 | 0.322 |
| anomaly_folder_filename (threshold=False) | hdfs_balanced_5k | 0.137 | 0.137 | 0.160 | 0.198 |
| anomaly_folder_filename (threshold=False) | bgl_split_10 | 0.197 | 0.235 | 0.650 | 1.156 |
| anomaly_folder_content | hadoop_renamed | 0.506 | 1.018 | 3.632 | 7.082 |
| anomaly_folder_content | hdfs_balanced_5k | 1.500 | 1.661 | 2.764 | 4.063 |
| anomaly_folder_content | bgl_split_10 | 2.258 | 3.595 | 13.5 | 59.1 |
| anomaly_folder_content (threshold=False) | hadoop_renamed | 0.167 | 0.194 | 0.324 | 0.592 |
| anomaly_folder_content (threshold=False) | hdfs_balanced_5k | 0.152 | 0.162 | 0.281 | 0.404 |
| anomaly_folder_content (threshold=False) | bgl_split_10 | 0.280 | 0.444 | 1.608 | OOM |
| anomaly_file_content | hadoop_renamed | err | 10.9 | 23.9 | 27.4 |
| anomaly_file_content | hdfs_balanced_5k | 0.024 | 0.022 | 0.023 | 0.026 |
| anomaly_file_content | bgl_split_10 | 0.025 | 0.036 | 0.097 | 6.643 |
| anomaly_file_content (threshold=False) | hadoop_renamed | err | 2.046 | 2.289 | 2.653 |
| anomaly_file_content (threshold=False) | hdfs_balanced_5k | 0.018 | 0.020 | 0.022 | 0.026 |
| anomaly_file_content (threshold=False) | bgl_split_10 | 0.026 | 0.035 | 0.109 | 0.333 |
| anomaly_line_content | hadoop_renamed | 0.811 | 0.865 | 1.584 | 1.802 |
| anomaly_line_content | hdfs_balanced_5k | 0.018 | 0.018 | 0.020 | 0.024 |
| anomaly_line_content | bgl_split_10 | 0.025 | 0.039 | 0.110 | 0.249 |
| anomaly_line_content (threshold=False) | hadoop_renamed | 0.402 | 0.215 | 0.243 | 0.257 |
| anomaly_line_content (threshold=False) | hdfs_balanced_5k | 0.019 | 0.018 | 0.026 | 0.025 |
| anomaly_line_content (threshold=False) | bgl_split_10 | 0.026 | 0.037 | 0.109 | 0.254 |

## Table A4 -- Plot tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| plot_folder_filename | hadoop_renamed | 0.049 | 0.061 | 0.074 | 0.063 |
| plot_folder_filename | hdfs_balanced_5k | 0.112 | 0.111 | 0.145 | 0.189 |
| plot_folder_filename | bgl_split_10 | 0.121 | 0.140 | 0.221 | 0.952 |
| plot_folder_content (scatter) | hadoop_renamed | 0.081 | 0.118 | 0.345 | 0.535 |
| plot_folder_content (scatter) | hdfs_balanced_5k | 0.077 | 0.089 | 0.182 | 0.321 |
| plot_folder_content (scatter) | bgl_split_10 | 0.199 | 0.390 | 1.524 | 6.378 |
| plot_folder_content (scatter+umap) | hadoop_renamed | err | 13.1 | 13.7 | 14.1 |
| plot_folder_content (scatter+umap) | hdfs_balanced_5k | 13.5 | 14.2 | 20.9 | 30.8 |
| plot_folder_content (scatter+umap) | bgl_split_10 | 13.1 | 14.1 | 15.1 | 22.3 |
| plot_file_content (scatter) | hadoop_renamed | 0.092 | 0.060 | 0.075 | 0.069 |
| plot_file_content (scatter) | hdfs_balanced_5k | 0.018 | 0.015 | 0.018 | 0.017 |
| plot_file_content (scatter) | bgl_split_10 | 0.024 | 0.032 | 0.077 | 0.565 |
| plot_file_content (scatter+umap) | hadoop_renamed | err | 0.095 | 0.143 | 0.203 |
| plot_file_content (scatter+umap) | hdfs_balanced_5k | 0.020 | 0.013 | 0.021 | 0.020 |
| plot_file_content (scatter+umap) | bgl_split_10 | 0.024 | 0.033 | 0.106 | 0.531 |

## Table A5 -- Sequence tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| sequence_line_event_prediction | hadoop_renamed | 0.215 | 0.093 | 0.117 | 0.301 |
| sequence_line_event_prediction | hdfs_balanced_5k | 0.030 | 0.026 | 0.029 | 0.038 |
| sequence_line_event_prediction | bgl_split_10 | 0.028 | 0.044 | 0.116 | 0.615 |

# Part B -- warm (repeated call)

## Table B1 -- Auxiliary tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| peek_log_root | hadoop_renamed | 0.089 | 0.078 | 0.091 | 0.102 |
| peek_log_root | hdfs_balanced_5k | 0.103 | 0.098 | 0.129 | 0.150 |
| peek_log_root | bgl_split_10 | 0.024 | 0.024 | 0.025 | 0.025 |
| open_log_root | hadoop_renamed | 0.017 | 0.018 | 0.040 | 0.052 |
| open_log_root | hdfs_balanced_5k | 0.016 | 0.019 | 0.040 | 0.060 |
| open_log_root | bgl_split_10 | 0.058 | 0.091 | 0.402 | 0.845 |
| open_log_root (no parsers) | hadoop_renamed | 0.014 | 0.019 | 0.035 | 0.049 |
| open_log_root (no parsers) | hdfs_balanced_5k | 0.017 | 0.018 | 0.035 | 0.050 |
| open_log_root (no parsers) | bgl_split_10 | 0.060 | 0.089 | 0.374 | 0.743 |
| open_log_root (parse tip) | hadoop_renamed | 0.013 | 0.018 | 0.042 | 0.048 |
| open_log_root (parse tip) | hdfs_balanced_5k | 0.015 | 0.022 | 0.036 | 0.058 |
| open_log_root (parse tip) | bgl_split_10 | 0.056 | 0.084 | 0.337 | 0.643 |
| open_log_root (parse drain) | hadoop_renamed | 0.016 | 0.016 | 0.036 | 0.048 |
| open_log_root (parse drain) | hdfs_balanced_5k | 0.021 | 0.019 | 0.036 | 0.052 |
| open_log_root (parse drain) | bgl_split_10 | 0.052 | 0.074 | 0.349 | 0.604 |
| list_log_roots | hadoop_renamed | 0.006 | 0.007 | 0.012 | 0.016 |
| list_log_roots | hdfs_balanced_5k | 0.009 | 0.007 | 0.009 | 0.013 |
| list_log_roots | bgl_split_10 | 0.019 | 0.029 | 0.162 | 0.280 |
| describe_log_root | hadoop_renamed | 0.012 | 0.013 | 0.018 | 0.030 |
| describe_log_root | hdfs_balanced_5k | 0.018 | 0.016 | 0.021 | 0.028 |
| describe_log_root | bgl_split_10 | 0.030 | 0.051 | 0.212 | 0.378 |
| set_folder_names | hadoop_renamed | 0.026 | 0.032 | 0.093 | 0.655 |
| set_folder_names | hdfs_balanced_5k | 0.022 | 0.029 | 0.071 | 0.117 |
| set_folder_names | bgl_split_10 | 0.093 | 0.177 | 0.887 | 4.712 |
| read_log_lines | hadoop_renamed | 0.004 | 0.004 | 0.005 | 0.004 |
| read_log_lines | hdfs_balanced_5k | 0.004 | 0.004 | 0.003 | 0.005 |
| read_log_lines | bgl_split_10 | 0.004 | 0.006 | 0.013 | 0.020 |
| search_log_lines | hadoop_renamed | 0.008 | 0.008 | 0.015 | 0.013 |
| search_log_lines | hdfs_balanced_5k | 0.011 | 0.012 | 0.012 | 0.013 |
| search_log_lines | bgl_split_10 | 0.019 | 0.032 | 0.054 | 0.107 |
| filter_log_lines | hadoop_renamed | 0.029 | 0.026 | 0.029 | 0.035 |
| filter_log_lines | hdfs_balanced_5k | 0.026 | 0.030 | 0.030 | 0.032 |
| filter_log_lines | bgl_split_10 | 0.044 | 0.057 | 0.114 | 0.197 |
| new_tokens | hadoop_renamed | 0.051 | 0.045 | 0.049 | 0.052 |
| new_tokens | hdfs_balanced_5k | 0.043 | 0.045 | 0.042 | 0.049 |
| new_tokens | bgl_split_10 | 0.077 | 0.077 | 0.129 | 0.228 |
| query_result | hadoop_renamed | 0.002 | 0.002 | 0.002 | 0.002 |
| query_result | hdfs_balanced_5k | 0.002 | 0.002 | 0.005 | 0.005 |
| query_result | bgl_split_10 | 0.002 | 0.002 | 0.003 | 0.003 |
| split_log_file | hadoop_renamed | 0.000 | 0.000 | 0.000 | 0.000 |
| split_log_file | hdfs_balanced_5k | 0.000 | 0.000 | 0.000 | 0.000 |
| split_log_file | bgl_split_10 | 0.000 | 0.000 | 0.000 | 0.000 |
| close_log_root | hadoop_renamed | 0.009 | 0.009 | 0.011 | 0.024 |
| close_log_root | hdfs_balanced_5k | 0.008 | 0.009 | 0.011 | 0.014 |
| close_log_root | bgl_split_10 | 0.031 | 0.050 | 0.212 | 0.290 |
| run_config | hadoop_renamed | 0.209 | 0.404 | 1.224 | 2.296 |
| run_config | hdfs_balanced_5k | 1.310 | 2.486 | 12.2 | 25.3 |
| run_config | bgl_split_10 | 0.960 | 1.534 | 6.478 | 18.8 |

## Table B2 -- Distance tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| distance_folder_filename | hadoop_renamed | 0.041 | 0.082 | 0.356 | 0.626 |
| distance_folder_filename | hdfs_balanced_5k | 2.065 | 4.181 | 20.8 | 42.2 |
| distance_folder_filename | bgl_split_10 | 0.104 | 0.122 | 0.250 | 0.375 |
| distance_folder_filename (threshold=False) | hadoop_renamed | 0.042 | 0.081 | 0.349 | 0.616 |
| distance_folder_filename (threshold=False) | hdfs_balanced_5k | 2.097 | 4.146 | 21.1 | 42.3 |
| distance_folder_filename (threshold=False) | bgl_split_10 | 0.099 | 0.121 | 0.244 | 0.374 |
| distance_folder_content | hadoop_renamed | 0.092 | 0.200 | 0.931 | 1.855 |
| distance_folder_content | hdfs_balanced_5k | 1.922 | 3.781 | 19.4 | 39.2 |
| distance_folder_content | bgl_split_10 | 0.526 | 0.859 | 3.970 | 11.0 |
| distance_folder_content (threshold=False) | hadoop_renamed | 0.090 | 0.191 | 0.947 | 1.844 |
| distance_folder_content (threshold=False) | hdfs_balanced_5k | 1.924 | 3.880 | 19.2 | 39.0 |
| distance_folder_content (threshold=False) | bgl_split_10 | 0.522 | 0.875 | 4.088 | 11.0 |
| distance_file_content | hadoop_renamed | 0.234 | 0.872 | 3.006 | 5.709 |
| distance_file_content | hdfs_balanced_5k | 0.028 | 0.026 | 0.027 | 0.030 |
| distance_file_content | bgl_split_10 | 0.035 | 0.049 | 0.166 | 0.247 |
| distance_file_content (threshold=False) | hadoop_renamed | 0.213 | 0.473 | 2.248 | 4.962 |
| distance_file_content (threshold=False) | hdfs_balanced_5k | 0.026 | 0.027 | 0.030 | 0.031 |
| distance_file_content (threshold=False) | bgl_split_10 | 0.035 | 0.055 | 0.146 | 0.300 |
| log_line_clustering | hadoop_renamed | 0.131 | 0.093 | 0.111 | 0.219 |
| log_line_clustering | hdfs_balanced_5k | 0.033 | 0.032 | 0.039 | 0.047 |
| log_line_clustering | bgl_split_10 | 0.040 | 0.061 | 0.200 | 0.337 |

## Table B3 -- Anomaly tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename | hadoop_renamed | 0.149 | 0.151 | 0.163 | 0.275 |
| anomaly_folder_filename | hdfs_balanced_5k | 0.147 | 0.154 | 0.174 | 0.197 |
| anomaly_folder_filename | bgl_split_10 | 0.206 | 0.260 | 0.734 | 1.282 |
| anomaly_folder_filename (threshold=False) | hadoop_renamed | 0.138 | 0.141 | 0.160 | 0.192 |
| anomaly_folder_filename (threshold=False) | hdfs_balanced_5k | 0.142 | 0.140 | 0.158 | 0.191 |
| anomaly_folder_filename (threshold=False) | bgl_split_10 | 0.179 | 0.242 | 0.647 | 1.145 |
| anomaly_folder_content | hadoop_renamed | 0.191 | 0.198 | 0.370 | 0.664 |
| anomaly_folder_content | hdfs_balanced_5k | 0.166 | 0.187 | 0.290 | 0.395 |
| anomaly_folder_content | bgl_split_10 | 0.295 | 0.459 | 1.741 | 10.4 |
| anomaly_folder_content (threshold=False) | hadoop_renamed | 0.173 | 0.197 | 0.338 | 0.662 |
| anomaly_folder_content (threshold=False) | hdfs_balanced_5k | 0.151 | 0.173 | 0.262 | 0.364 |
| anomaly_folder_content (threshold=False) | bgl_split_10 | 0.268 | 0.424 | 1.585 | OOM |
| anomaly_file_content | hadoop_renamed | err | 11.0 | 23.7 | 27.4 |
| anomaly_file_content | hdfs_balanced_5k | 0.019 | 0.020 | 0.024 | 0.026 |
| anomaly_file_content | bgl_split_10 | 0.025 | 0.034 | 0.101 | 0.320 |
| anomaly_file_content (threshold=False) | hadoop_renamed | err | 2.073 | 2.249 | 2.638 |
| anomaly_file_content (threshold=False) | hdfs_balanced_5k | 0.019 | 0.021 | 0.025 | 0.028 |
| anomaly_file_content (threshold=False) | bgl_split_10 | 0.028 | 0.038 | 0.120 | 0.287 |
| anomaly_line_content | hadoop_renamed | 0.423 | 0.224 | 0.243 | 0.273 |
| anomaly_line_content | hdfs_balanced_5k | 0.020 | 0.019 | 0.024 | 0.030 |
| anomaly_line_content | bgl_split_10 | 0.027 | 0.037 | 0.124 | 0.252 |
| anomaly_line_content (threshold=False) | hadoop_renamed | 0.398 | 0.207 | 0.233 | 0.240 |
| anomaly_line_content (threshold=False) | hdfs_balanced_5k | 0.018 | 0.019 | 0.022 | 0.028 |
| anomaly_line_content (threshold=False) | bgl_split_10 | 0.029 | 0.040 | 0.110 | 0.232 |

## Table B4 -- Plot tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| plot_folder_filename | hadoop_renamed | 0.053 | 0.057 | 0.076 | 0.066 |
| plot_folder_filename | hdfs_balanced_5k | 0.067 | 0.072 | 0.095 | 0.125 |
| plot_folder_filename | bgl_split_10 | 0.071 | 0.090 | 0.162 | 0.881 |
| plot_folder_content (scatter) | hadoop_renamed | 0.082 | 0.100 | 0.286 | 0.473 |
| plot_folder_content (scatter) | hdfs_balanced_5k | 0.074 | 0.082 | 0.179 | 0.280 |
| plot_folder_content (scatter) | bgl_split_10 | 0.175 | 0.326 | 1.398 | 5.955 |
| plot_folder_content (scatter+umap) | hadoop_renamed | err | 0.133 | 0.306 | 0.517 |
| plot_folder_content (scatter+umap) | hdfs_balanced_5k | 0.489 | 0.919 | 7.739 | 9.328 |
| plot_folder_content (scatter+umap) | bgl_split_10 | 0.191 | 0.352 | 1.415 | 5.828 |
| plot_file_content (scatter) | hadoop_renamed | 0.085 | 0.063 | 0.079 | 0.071 |
| plot_file_content (scatter) | hdfs_balanced_5k | 0.018 | 0.016 | 0.017 | 0.019 |
| plot_file_content (scatter) | bgl_split_10 | 0.023 | 0.032 | 0.095 | 0.465 |
| plot_file_content (scatter+umap) | hadoop_renamed | err | 0.095 | 0.147 | 0.195 |
| plot_file_content (scatter+umap) | hdfs_balanced_5k | 0.018 | 0.017 | 0.019 | 0.021 |
| plot_file_content (scatter+umap) | bgl_split_10 | 0.024 | 0.035 | 0.105 | 0.465 |

## Table B5 -- Sequence tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| sequence_line_event_prediction | hadoop_renamed | 0.198 | 0.095 | 0.131 | 0.181 |
| sequence_line_event_prediction | hdfs_balanced_5k | 0.027 | 0.026 | 0.031 | 0.035 |
| sequence_line_event_prediction | bgl_split_10 | 0.035 | 0.045 | 0.120 | 0.553 |

# Detailed breakdowns (per detector / per measure)

The tables above run every anomaly tool with all four detectors, `distance_folder_content`/`distance_file_content` with their three default measures (cosine, jaccard, containment; compression is opt-in), and `sequence_line_event_prediction` with both order detectors, at once; `log_line_clustering` defaults to its coarse pair (Prefix + Exact) in one pass. Part C/D below break the same figure down per detector / per measure run in isolation (`detectors=["<name>"]` / `measures=["<name>"]`), so the cost of narrowing either is visible on its own rather than folded into the combined call -- `log_line_clustering` included, run once per bucket measure (Exact, Prefix, Minhash) so they can be compared directly. `distance_folder_filename` (jaccard/overlap distance over file names only) is not broken down further -- it computes one measure, not a default pair. The anomaly rows pass `threshold=False`, so they show one detector's own cost without the extra fits of the clean range; the distance rows keep the default clean range.

# Part C -- cold (first call)

## Table C1 -- Anomaly tools detailed

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename (KMeans) | hadoop_renamed | 0.081 | 0.066 | 0.070 | 0.353 |
| anomaly_folder_filename (KMeans) | hdfs_balanced_5k | 0.055 | 0.058 | 0.082 | 0.100 |
| anomaly_folder_filename (KMeans) | bgl_split_10 | 0.103 | 0.167 | 0.668 | 1.153 |
| anomaly_folder_filename (IsolationForest) | hadoop_renamed | 0.158 | 0.142 | 0.151 | 0.321 |
| anomaly_folder_filename (IsolationForest) | hdfs_balanced_5k | 0.146 | 0.148 | 0.173 | 0.185 |
| anomaly_folder_filename (IsolationForest) | bgl_split_10 | 0.200 | 0.254 | 0.684 | 1.169 |
| anomaly_folder_filename (RarityDetector) | hadoop_renamed | 0.035 | 0.034 | 0.043 | 0.209 |
| anomaly_folder_filename (RarityDetector) | hdfs_balanced_5k | 0.037 | 0.038 | 0.055 | 0.081 |
| anomaly_folder_filename (RarityDetector) | bgl_split_10 | 0.080 | 0.136 | 0.539 | 1.012 |
| anomaly_folder_filename (OOVDetector) | hadoop_renamed | 0.036 | 0.043 | 0.052 | 0.217 |
| anomaly_folder_filename (OOVDetector) | hdfs_balanced_5k | 0.035 | 0.042 | 0.061 | 0.071 |
| anomaly_folder_filename (OOVDetector) | bgl_split_10 | 0.089 | 0.161 | 0.571 | 1.312 |
| anomaly_folder_content (KMeans) | hadoop_renamed | 0.090 | 0.117 | 0.305 | 0.629 |
| anomaly_folder_content (KMeans) | hdfs_balanced_5k | 0.052 | 0.064 | 0.173 | 0.297 |
| anomaly_folder_content (KMeans) | bgl_split_10 | 0.192 | 0.382 | 1.709 | 6.467 |
| anomaly_folder_content (IsolationForest) | hadoop_renamed | 0.165 | 0.176 | 0.357 | 0.645 |
| anomaly_folder_content (IsolationForest) | hdfs_balanced_5k | 0.144 | 0.167 | 0.263 | 0.404 |
| anomaly_folder_content (IsolationForest) | bgl_split_10 | 0.260 | 0.431 | 1.670 | 6.978 |
| anomaly_folder_content (RarityDetector) | hadoop_renamed | 0.074 | 0.096 | 0.271 | 0.547 |
| anomaly_folder_content (RarityDetector) | hdfs_balanced_5k | 0.048 | 0.061 | 0.147 | 0.269 |
| anomaly_folder_content (RarityDetector) | bgl_split_10 | 0.182 | 0.340 | 1.595 | 6.304 |
| anomaly_folder_content (OOVDetector) | hadoop_renamed | 0.078 | 0.099 | 0.263 | 0.495 |
| anomaly_folder_content (OOVDetector) | hdfs_balanced_5k | 0.045 | 0.055 | 0.145 | 0.250 |
| anomaly_folder_content (OOVDetector) | bgl_split_10 | 0.183 | 0.393 | 1.724 | 6.656 |
| anomaly_file_content (KMeans) | hadoop_renamed | err | 0.514 | 0.688 | 0.994 |
| anomaly_file_content (KMeans) | hdfs_balanced_5k | 0.022 | 0.021 | 0.023 | 0.026 |
| anomaly_file_content (KMeans) | bgl_split_10 | 0.032 | 0.037 | 0.091 | 0.222 |
| anomaly_file_content (IsolationForest) | hadoop_renamed | 1.966 | 1.825 | 2.117 | 2.439 |
| anomaly_file_content (IsolationForest) | hdfs_balanced_5k | 0.016 | 0.019 | 0.022 | 0.026 |
| anomaly_file_content (IsolationForest) | bgl_split_10 | 0.028 | 0.035 | 0.098 | 0.185 |
| anomaly_file_content (RarityDetector) | hadoop_renamed | 0.377 | 0.418 | 0.595 | 0.858 |
| anomaly_file_content (RarityDetector) | hdfs_balanced_5k | 0.026 | 0.022 | 0.023 | 0.025 |
| anomaly_file_content (RarityDetector) | bgl_split_10 | 0.029 | 0.038 | 0.090 | 0.248 |
| anomaly_file_content (OOVDetector) | hadoop_renamed | 0.403 | 0.425 | 0.615 | 0.932 |
| anomaly_file_content (OOVDetector) | hdfs_balanced_5k | 0.025 | 0.023 | 0.024 | 0.025 |
| anomaly_file_content (OOVDetector) | bgl_split_10 | 0.030 | 0.033 | 0.094 | 0.194 |
| anomaly_line_content (KMeans) | hadoop_renamed | 0.448 | 0.195 | 0.230 | 0.235 |
| anomaly_line_content (KMeans) | hdfs_balanced_5k | 0.018 | 0.019 | 0.023 | 0.024 |
| anomaly_line_content (KMeans) | bgl_split_10 | 0.026 | 0.036 | 0.116 | 0.251 |
| anomaly_line_content (IsolationForest) | hadoop_renamed | 0.347 | 0.168 | 0.195 | 0.212 |
| anomaly_line_content (IsolationForest) | hdfs_balanced_5k | 0.018 | 0.020 | 0.020 | 0.026 |
| anomaly_line_content (IsolationForest) | bgl_split_10 | 0.024 | 0.032 | 0.107 | 0.222 |
| anomaly_line_content (RarityDetector) | hadoop_renamed | 0.179 | 0.081 | 0.098 | 0.104 |
| anomaly_line_content (RarityDetector) | hdfs_balanced_5k | 0.019 | 0.020 | 0.024 | 0.023 |
| anomaly_line_content (RarityDetector) | bgl_split_10 | 0.025 | 0.038 | 0.113 | 0.238 |
| anomaly_line_content (OOVDetector) | hadoop_renamed | 0.164 | 0.073 | 0.117 | 0.106 |
| anomaly_line_content (OOVDetector) | hdfs_balanced_5k | 0.018 | 0.022 | 0.023 | 0.027 |
| anomaly_line_content (OOVDetector) | bgl_split_10 | 0.025 | 0.035 | 0.128 | 0.250 |

## Table C2 -- Distance tools detailed

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| distance_folder_content (cosine) | hadoop_renamed | 0.085 | 0.272 | 1.111 | 1.996 |
| distance_folder_content (cosine) | hdfs_balanced_5k | 1.593 | 3.150 | 16.0 | 32.2 |
| distance_folder_content (cosine) | bgl_split_10 | 0.756 | 1.299 | 5.834 | 18.6 |
| distance_folder_content (jaccard) | hadoop_renamed | 0.083 | 0.267 | 1.096 | 2.048 |
| distance_folder_content (jaccard) | hdfs_balanced_5k | 1.756 | 3.492 | 17.3 | 35.6 |
| distance_folder_content (jaccard) | bgl_split_10 | 0.782 | 1.262 | 5.848 | 17.5 |
| distance_folder_content (compression) | hadoop_renamed | 0.714 | 5.090 | 24.4 | 33.1 |
| distance_folder_content (compression) | hdfs_balanced_5k | 1.953 | 3.378 | 16.7 | 33.8 |
| distance_folder_content (compression) | bgl_split_10 | 20.7 | 38.8 | 223.7 | 839.5 |
| distance_folder_content (containment) | hadoop_renamed | 0.077 | 0.272 | 1.014 | 1.932 |
| distance_folder_content (containment) | hdfs_balanced_5k | 1.482 | 2.938 | 14.6 | 29.5 |
| distance_folder_content (containment) | bgl_split_10 | 0.715 | 1.267 | 5.783 | 18.3 |
| distance_file_content (cosine) | hadoop_renamed | 0.186 | 0.729 | 2.334 | 4.857 |
| distance_file_content (cosine) | hdfs_balanced_5k | 0.028 | 0.030 | 0.032 | 0.033 |
| distance_file_content (cosine) | bgl_split_10 | 0.033 | 0.048 | 0.150 | 0.368 |
| distance_file_content (jaccard) | hadoop_renamed | 0.208 | 0.757 | 2.592 | 5.181 |
| distance_file_content (jaccard) | hdfs_balanced_5k | 0.030 | 0.026 | 0.028 | 0.030 |
| distance_file_content (jaccard) | bgl_split_10 | 0.036 | 0.048 | 0.149 | 0.305 |
| distance_file_content (compression) | hadoop_renamed | 0.697 | 4.572 | 21.0 | 27.3 |
| distance_file_content (compression) | hdfs_balanced_5k | 0.025 | 0.028 | 0.025 | 0.032 |
| distance_file_content (compression) | bgl_split_10 | 0.034 | 0.047 | 0.144 | 0.331 |
| distance_file_content (containment) | hadoop_renamed | 0.175 | 0.707 | 2.187 | 4.287 |
| distance_file_content (containment) | hdfs_balanced_5k | 0.027 | 0.030 | 0.034 | 0.041 |
| distance_file_content (containment) | bgl_split_10 | 0.036 | 0.055 | 0.154 | 0.324 |
| log_line_clustering (Exact) | hadoop_renamed | 0.083 | 0.080 | 0.080 | 0.318 |
| log_line_clustering (Exact) | hdfs_balanced_5k | 0.031 | 0.032 | 0.045 | 0.046 |
| log_line_clustering (Exact) | bgl_split_10 | 0.037 | 0.056 | 0.172 | 0.422 |
| log_line_clustering (Prefix) | hadoop_renamed | 0.094 | 0.082 | 0.089 | 0.182 |
| log_line_clustering (Prefix) | hdfs_balanced_5k | 0.031 | 0.033 | 0.037 | 0.044 |
| log_line_clustering (Prefix) | bgl_split_10 | 0.040 | 0.065 | 0.203 | 0.345 |
| log_line_clustering (Minhash) | hadoop_renamed | 0.104 | 0.109 | 0.166 | 0.943 |
| log_line_clustering (Minhash) | hdfs_balanced_5k | 0.034 | 0.033 | 0.045 | 0.051 |
| log_line_clustering (Minhash) | bgl_split_10 | 0.043 | 0.060 | 0.187 | 0.332 |

## Table C3 -- Sequence tools detailed

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| sequence_line_event_prediction (NEP) | hadoop_renamed | 0.191 | 0.119 | 0.133 | 0.291 |
| sequence_line_event_prediction (NEP) | hdfs_balanced_5k | 0.024 | 0.028 | 0.029 | 0.034 |
| sequence_line_event_prediction (NEP) | bgl_split_10 | 0.034 | 0.046 | 0.132 | 0.242 |
| sequence_line_event_prediction (LAP) | hadoop_renamed | 0.154 | 0.087 | 0.118 | 0.296 |
| sequence_line_event_prediction (LAP) | hdfs_balanced_5k | 0.021 | 0.028 | 0.029 | 0.038 |
| sequence_line_event_prediction (LAP) | bgl_split_10 | 0.033 | 0.044 | 0.119 | 0.228 |

# Part D -- warm (repeated call)

## Table D1 -- Anomaly tools detailed

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename (KMeans) | hadoop_renamed | 0.045 | 0.050 | 0.057 | 0.099 |
| anomaly_folder_filename (KMeans) | hdfs_balanced_5k | 0.039 | 0.045 | 0.060 | 0.078 |
| anomaly_folder_filename (KMeans) | bgl_split_10 | 0.080 | 0.139 | 0.556 | 1.068 |
| anomaly_folder_filename (IsolationForest) | hadoop_renamed | 0.148 | 0.126 | 0.138 | 0.187 |
| anomaly_folder_filename (IsolationForest) | hdfs_balanced_5k | 0.134 | 0.138 | 0.158 | 0.174 |
| anomaly_folder_filename (IsolationForest) | bgl_split_10 | 0.186 | 0.241 | 0.631 | 1.148 |
| anomaly_folder_filename (RarityDetector) | hadoop_renamed | 0.036 | 0.035 | 0.046 | 0.106 |
| anomaly_folder_filename (RarityDetector) | hdfs_balanced_5k | 0.035 | 0.040 | 0.057 | 0.077 |
| anomaly_folder_filename (RarityDetector) | bgl_split_10 | 0.081 | 0.131 | 0.499 | 0.965 |
| anomaly_folder_filename (OOVDetector) | hadoop_renamed | 0.038 | 0.035 | 0.052 | 0.100 |
| anomaly_folder_filename (OOVDetector) | hdfs_balanced_5k | 0.042 | 0.044 | 0.055 | 0.075 |
| anomaly_folder_filename (OOVDetector) | bgl_split_10 | 0.089 | 0.157 | 0.498 | 1.141 |
| anomaly_folder_content (KMeans) | hadoop_renamed | 0.086 | 0.096 | 0.253 | 0.553 |
| anomaly_folder_content (KMeans) | hdfs_balanced_5k | 0.054 | 0.063 | 0.155 | 0.259 |
| anomaly_folder_content (KMeans) | bgl_split_10 | 0.176 | 0.332 | 1.529 | 6.077 |
| anomaly_folder_content (IsolationForest) | hadoop_renamed | 0.160 | 0.186 | 0.340 | 0.662 |
| anomaly_folder_content (IsolationForest) | hdfs_balanced_5k | 0.143 | 0.160 | 0.263 | 0.372 |
| anomaly_folder_content (IsolationForest) | bgl_split_10 | 0.247 | 0.405 | 1.558 | 6.133 |
| anomaly_folder_content (RarityDetector) | hadoop_renamed | 0.073 | 0.093 | 0.267 | 0.510 |
| anomaly_folder_content (RarityDetector) | hdfs_balanced_5k | 0.047 | 0.055 | 0.136 | 0.256 |
| anomaly_folder_content (RarityDetector) | bgl_split_10 | 0.168 | 0.342 | 1.441 | 6.567 |
| anomaly_folder_content (OOVDetector) | hadoop_renamed | 0.074 | 0.091 | 0.256 | 0.520 |
| anomaly_folder_content (OOVDetector) | hdfs_balanced_5k | 0.046 | 0.057 | 0.147 | 0.240 |
| anomaly_folder_content (OOVDetector) | bgl_split_10 | 0.188 | 0.345 | 1.497 | 6.557 |
| anomaly_file_content (KMeans) | hadoop_renamed | err | 0.523 | 0.680 | 1.031 |
| anomaly_file_content (KMeans) | hdfs_balanced_5k | 0.021 | 0.022 | 0.023 | 0.024 |
| anomaly_file_content (KMeans) | bgl_split_10 | 0.028 | 0.034 | 0.103 | 0.241 |
| anomaly_file_content (IsolationForest) | hadoop_renamed | 1.927 | 1.832 | 2.063 | 2.470 |
| anomaly_file_content (IsolationForest) | hdfs_balanced_5k | 0.019 | 0.019 | 0.024 | 0.025 |
| anomaly_file_content (IsolationForest) | bgl_split_10 | 0.025 | 0.035 | 0.099 | 0.204 |
| anomaly_file_content (RarityDetector) | hadoop_renamed | 0.385 | 0.415 | 0.592 | 0.875 |
| anomaly_file_content (RarityDetector) | hdfs_balanced_5k | 0.021 | 0.022 | 0.024 | 0.026 |
| anomaly_file_content (RarityDetector) | bgl_split_10 | 0.028 | 0.037 | 0.098 | 0.226 |
| anomaly_file_content (OOVDetector) | hadoop_renamed | 0.390 | 0.431 | 0.631 | 0.868 |
| anomaly_file_content (OOVDetector) | hdfs_balanced_5k | 0.021 | 0.025 | 0.025 | 0.028 |
| anomaly_file_content (OOVDetector) | bgl_split_10 | 0.027 | 0.034 | 0.121 | 0.227 |
| anomaly_line_content (KMeans) | hadoop_renamed | 0.187 | 0.076 | 0.097 | 0.096 |
| anomaly_line_content (KMeans) | hdfs_balanced_5k | 0.020 | 0.019 | 0.023 | 0.023 |
| anomaly_line_content (KMeans) | bgl_split_10 | 0.026 | 0.037 | 0.111 | 0.242 |
| anomaly_line_content (IsolationForest) | hadoop_renamed | 0.289 | 0.171 | 0.198 | 0.217 |
| anomaly_line_content (IsolationForest) | hdfs_balanced_5k | 0.017 | 0.019 | 0.022 | 0.024 |
| anomaly_line_content (IsolationForest) | bgl_split_10 | 0.025 | 0.033 | 0.116 | 0.208 |
| anomaly_line_content (RarityDetector) | hadoop_renamed | 0.151 | 0.082 | 0.099 | 0.106 |
| anomaly_line_content (RarityDetector) | hdfs_balanced_5k | 0.021 | 0.020 | 0.024 | 0.024 |
| anomaly_line_content (RarityDetector) | bgl_split_10 | 0.026 | 0.041 | 0.108 | 0.252 |
| anomaly_line_content (OOVDetector) | hadoop_renamed | 0.153 | 0.073 | 0.115 | 0.111 |
| anomaly_line_content (OOVDetector) | hdfs_balanced_5k | 0.018 | 0.022 | 0.022 | 0.026 |
| anomaly_line_content (OOVDetector) | bgl_split_10 | 0.026 | 0.034 | 0.124 | 0.247 |

## Table D2 -- Distance tools detailed

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| distance_folder_content (cosine) | hadoop_renamed | 0.084 | 0.179 | 0.885 | 1.788 |
| distance_folder_content (cosine) | hdfs_balanced_5k | 1.579 | 3.113 | 15.7 | 32.3 |
| distance_folder_content (cosine) | bgl_split_10 | 0.558 | 0.855 | 3.978 | 11.9 |
| distance_folder_content (jaccard) | hadoop_renamed | 0.083 | 0.178 | 0.904 | 1.847 |
| distance_folder_content (jaccard) | hdfs_balanced_5k | 1.873 | 3.394 | 17.4 | 35.5 |
| distance_folder_content (jaccard) | bgl_split_10 | 0.540 | 0.865 | 3.906 | 10.9 |
| distance_folder_content (compression) | hadoop_renamed | 0.729 | 1.666 | 8.553 | 17.9 |
| distance_folder_content (compression) | hdfs_balanced_5k | 2.010 | 3.185 | 16.4 | 33.6 |
| distance_folder_content (compression) | bgl_split_10 | 4.662 | 8.273 | 43.1 | 142.3 |
| distance_folder_content (containment) | hadoop_renamed | 0.082 | 0.185 | 0.872 | 1.771 |
| distance_folder_content (containment) | hdfs_balanced_5k | 1.439 | 2.889 | 14.5 | 29.6 |
| distance_folder_content (containment) | bgl_split_10 | 0.503 | 0.874 | 3.888 | 11.4 |
| distance_file_content (cosine) | hadoop_renamed | 0.181 | 0.735 | 2.414 | 4.596 |
| distance_file_content (cosine) | hdfs_balanced_5k | 0.026 | 0.028 | 0.029 | 0.031 |
| distance_file_content (cosine) | bgl_split_10 | 0.038 | 0.046 | 0.124 | 0.268 |
| distance_file_content (jaccard) | hadoop_renamed | 0.199 | 0.781 | 2.604 | 5.088 |
| distance_file_content (jaccard) | hdfs_balanced_5k | 0.029 | 0.025 | 0.028 | 0.030 |
| distance_file_content (jaccard) | bgl_split_10 | 0.037 | 0.048 | 0.122 | 0.312 |
| distance_file_content (compression) | hadoop_renamed | 0.691 | 4.604 | 20.8 | 27.5 |
| distance_file_content (compression) | hdfs_balanced_5k | 0.024 | 0.028 | 0.029 | 0.030 |
| distance_file_content (compression) | bgl_split_10 | 0.036 | 0.043 | 0.138 | 0.268 |
| distance_file_content (containment) | hadoop_renamed | 0.169 | 0.738 | 2.232 | 4.205 |
| distance_file_content (containment) | hdfs_balanced_5k | 0.027 | 0.027 | 0.031 | 0.033 |
| distance_file_content (containment) | bgl_split_10 | 0.036 | 0.055 | 0.152 | 0.262 |
| log_line_clustering (Exact) | hadoop_renamed | 0.086 | 0.079 | 0.082 | 0.188 |
| log_line_clustering (Exact) | hdfs_balanced_5k | 0.032 | 0.033 | 0.041 | 0.052 |
| log_line_clustering (Exact) | bgl_split_10 | 0.040 | 0.064 | 0.175 | 0.375 |
| log_line_clustering (Prefix) | hadoop_renamed | 0.088 | 0.081 | 0.091 | 0.331 |
| log_line_clustering (Prefix) | hdfs_balanced_5k | 0.032 | 0.033 | 0.042 | 0.045 |
| log_line_clustering (Prefix) | bgl_split_10 | 0.042 | 0.065 | 0.188 | 0.349 |
| log_line_clustering (Minhash) | hadoop_renamed | 0.100 | 0.109 | 0.146 | 0.761 |
| log_line_clustering (Minhash) | hdfs_balanced_5k | 0.031 | 0.035 | 0.045 | 0.049 |
| log_line_clustering (Minhash) | bgl_split_10 | 0.044 | 0.063 | 0.209 | 0.348 |

## Table D3 -- Sequence tools detailed

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| sequence_line_event_prediction (NEP) | hadoop_renamed | 0.172 | 0.103 | 0.132 | 0.199 |
| sequence_line_event_prediction (NEP) | hdfs_balanced_5k | 0.025 | 0.026 | 0.031 | 0.034 |
| sequence_line_event_prediction (NEP) | bgl_split_10 | 0.031 | 0.046 | 0.118 | 0.237 |
| sequence_line_event_prediction (LAP) | hadoop_renamed | 0.150 | 0.090 | 0.118 | 0.233 |
| sequence_line_event_prediction (LAP) | hdfs_balanced_5k | 0.026 | 0.027 | 0.030 | 0.034 |
| sequence_line_event_prediction (LAP) | bgl_split_10 | 0.033 | 0.048 | 0.118 | 0.231 |
