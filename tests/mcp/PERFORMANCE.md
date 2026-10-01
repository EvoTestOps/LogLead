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
| open_log_root (no parsers) | hadoop_renamed | 1.049 | 1.217 | 2.708 | 4.708 |
| open_log_root (no parsers) | hdfs_balanced_5k | 1.704 | 1.635 | 2.873 | 4.574 |
| open_log_root (no parsers) | bgl_split_10 | 1.084 | 1.729 | 8.304 | 21.9 |
| open_log_root (parse tip) | hadoop_renamed | 0.968 | 1.195 | 2.804 | 4.724 |
| open_log_root (parse tip) | hdfs_balanced_5k | 1.367 | 1.467 | 2.818 | 4.611 |
| open_log_root (parse tip) | bgl_split_10 | 1.173 | 2.038 | 9.481 | 25.2 |
| open_log_root (parse drain) | hadoop_renamed | 1.230 | 1.418 | 3.593 | 7.079 |
| open_log_root (parse drain) | hdfs_balanced_5k | 1.361 | 1.738 | 3.268 | 5.111 |
| open_log_root (parse drain) | bgl_split_10 | 2.358 | 4.684 | 26.2 | 54.9 |
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
| distance_folder_filename | hadoop_renamed | 0.033 | 0.106 | 0.330 | 0.698 |
| distance_folder_filename | hdfs_balanced_5k | 1.750 | 3.175 | 16.7 | 34.1 |
| distance_folder_filename | bgl_split_10 | 0.108 | 0.145 | 0.369 | 0.674 |
| distance_folder_filename (threshold=False) | hadoop_renamed | 0.035 | 0.070 | 0.311 | 0.662 |
| distance_folder_filename (threshold=False) | hdfs_balanced_5k | 1.609 | 3.234 | 17.0 | 34.2 |
| distance_folder_filename (threshold=False) | bgl_split_10 | 0.108 | 0.134 | 0.283 | 0.432 |
| distance_folder_content | hadoop_renamed | 0.116 | 0.266 | 1.148 | 2.238 |
| distance_folder_content | hdfs_balanced_5k | 1.292 | 3.040 | 13.5 | 28.2 |
| distance_folder_content | bgl_split_10 | 0.833 | 1.368 | 6.531 | 18.9 |
| distance_folder_content (threshold=False) | hadoop_renamed | 0.110 | 0.203 | 1.012 | 2.063 |
| distance_folder_content (threshold=False) | hdfs_balanced_5k | 1.336 | 2.586 | 13.7 | 28.2 |
| distance_folder_content (threshold=False) | bgl_split_10 | 0.558 | 0.955 | 4.312 | 11.9 |
| distance_file_content | hadoop_renamed | 0.359 | 0.842 | 2.954 | 7.494 |
| distance_file_content | hdfs_balanced_5k | 0.020 | 0.019 | 0.021 | 0.023 |
| distance_file_content | bgl_split_10 | 0.049 | 0.057 | 0.188 | 0.383 |
| distance_file_content (threshold=False) | hadoop_renamed | 0.234 | 0.538 | 2.593 | 6.230 |
| distance_file_content (threshold=False) | hdfs_balanced_5k | 0.023 | 0.019 | 0.021 | 0.025 |
| distance_file_content (threshold=False) | bgl_split_10 | 0.047 | 0.063 | 0.136 | 0.255 |
| distance_line_content | hadoop_renamed | 0.101 | 0.121 | 0.194 | 0.367 |
| distance_line_content | hdfs_balanced_5k | 0.034 | 0.036 | 0.062 | 0.063 |
| distance_line_content | bgl_split_10 | 0.069 | 0.085 | 0.250 | 0.449 |

## Table A3 -- Anomaly tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename | hadoop_renamed | 0.441 | 0.913 | 1.739 | 2.938 |
| anomaly_folder_filename | hdfs_balanced_5k | 1.461 | 1.541 | 1.726 | 2.156 |
| anomaly_folder_filename | bgl_split_10 | 1.800 | 2.273 | 5.619 | 10.3 |
| anomaly_folder_filename (threshold=False) | hadoop_renamed | 0.148 | 0.153 | 0.207 | 0.218 |
| anomaly_folder_filename (threshold=False) | hdfs_balanced_5k | 0.132 | 0.152 | 0.178 | 0.201 |
| anomaly_folder_filename (threshold=False) | bgl_split_10 | 0.207 | 0.274 | 0.735 | 1.346 |
| anomaly_folder_content | hadoop_renamed | 0.501 | 1.075 | 3.426 | 7.911 |
| anomaly_folder_content | hdfs_balanced_5k | 1.537 | 1.705 | 2.786 | 4.104 |
| anomaly_folder_content | bgl_split_10 | 2.537 | 4.090 | 18.8 | 69.8 |
| anomaly_folder_content (threshold=False) | hadoop_renamed | 0.194 | 0.194 | 0.330 | 0.719 |
| anomaly_folder_content (threshold=False) | hdfs_balanced_5k | 0.170 | 0.170 | 0.291 | 0.420 |
| anomaly_folder_content (threshold=False) | bgl_split_10 | 0.322 | 0.502 | 1.844 | 8.938 |
| anomaly_file_content | hadoop_renamed | err | 12.7 | 24.3 | 32.4 |
| anomaly_file_content | hdfs_balanced_5k | 0.018 | 0.020 | 0.023 | 0.041 |
| anomaly_file_content | bgl_split_10 | 0.029 | 0.044 | 0.114 | 0.177 |
| anomaly_file_content (threshold=False) | hadoop_renamed | err | 2.072 | 2.324 | 3.039 |
| anomaly_file_content (threshold=False) | hdfs_balanced_5k | 0.020 | 0.021 | 0.021 | 0.035 |
| anomaly_file_content (threshold=False) | bgl_split_10 | 0.029 | 0.038 | 0.144 | 0.272 |
| anomaly_line_content | hadoop_renamed | 0.622 | 1.833 | 1.699 | 1.945 |
| anomaly_line_content | hdfs_balanced_5k | 0.015 | 0.016 | 0.019 | 0.028 |
| anomaly_line_content | bgl_split_10 | 0.030 | 0.041 | 0.131 | 0.262 |
| anomaly_line_content (threshold=False) | hadoop_renamed | 0.219 | 0.447 | 0.240 | 0.267 |
| anomaly_line_content (threshold=False) | hdfs_balanced_5k | 0.017 | 0.015 | 0.018 | 0.029 |
| anomaly_line_content (threshold=False) | bgl_split_10 | 0.029 | 0.042 | 0.131 | 0.279 |

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

## Table A5 -- Sequence tools

Measured with `Parse-Drain` already built; the parse is priced on its own in Table A1/B1. See the
`**` note after Table C3.

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| sequence_line_event_prediction | hadoop_renamed | 0.115 | 0.084 | 0.118 | 0.482 |
| sequence_line_event_prediction | hdfs_balanced_5k | 0.015 | 0.019 | 0.031 | 0.042 |
| sequence_line_event_prediction | bgl_split_10 | 0.037 | 0.042 | 0.118 | 0.244 |

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
| open_log_root (no parsers) | hadoop_renamed | 0.017 | 0.022 | 0.039 | 0.049 |
| open_log_root (no parsers) | hdfs_balanced_5k | 0.024 | 0.017 | 0.033 | 0.053 |
| open_log_root (no parsers) | bgl_split_10 | 0.057 | 0.085 | 0.483 | 0.915 |
| open_log_root (parse tip) | hadoop_renamed | 0.020 | 0.035 | 0.048 | 0.054 |
| open_log_root (parse tip) | hdfs_balanced_5k | 0.021 | 0.021 | 0.047 | 0.072 |
| open_log_root (parse tip) | bgl_split_10 | 0.071 | 0.092 | 0.492 | 1.397 |
| open_log_root (parse drain) | hadoop_renamed | 0.020 | 0.021 | 0.040 | 0.059 |
| open_log_root (parse drain) | hdfs_balanced_5k | 0.017 | 0.024 | 0.041 | 0.056 |
| open_log_root (parse drain) | bgl_split_10 | 0.057 | 0.085 | 0.355 | 0.690 |
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
| distance_folder_filename | hadoop_renamed | 0.032 | 0.078 | 0.312 | 0.657 |
| distance_folder_filename | hdfs_balanced_5k | 1.619 | 3.218 | 18.1 | 35.3 |
| distance_folder_filename | bgl_split_10 | 0.112 | 0.138 | 0.272 | 0.404 |
| distance_folder_filename (threshold=False) | hadoop_renamed | 0.038 | 0.072 | 0.303 | 0.683 |
| distance_folder_filename (threshold=False) | hdfs_balanced_5k | 1.649 | 3.245 | 16.9 | 33.9 |
| distance_folder_filename (threshold=False) | bgl_split_10 | 0.103 | 0.132 | 0.283 | 0.394 |
| distance_folder_content | hadoop_renamed | 0.111 | 0.182 | 0.959 | 2.199 |
| distance_folder_content | hdfs_balanced_5k | 1.362 | 2.736 | 13.4 | 30.1 |
| distance_folder_content | bgl_split_10 | 0.578 | 0.953 | 4.346 | 12.4 |
| distance_folder_content (threshold=False) | hadoop_renamed | 0.108 | 0.189 | 0.990 | 2.058 |
| distance_folder_content (threshold=False) | hdfs_balanced_5k | 1.337 | 2.646 | 13.3 | 29.1 |
| distance_folder_content (threshold=False) | bgl_split_10 | 0.572 | 0.999 | 4.294 | 11.8 |
| distance_file_content | hadoop_renamed | 0.256 | 0.829 | 3.023 | 6.496 |
| distance_file_content | hdfs_balanced_5k | 0.019 | 0.018 | 0.022 | 0.024 |
| distance_file_content | bgl_split_10 | 0.044 | 0.060 | 0.167 | 0.280 |
| distance_file_content (threshold=False) | hadoop_renamed | 0.236 | 0.524 | 2.532 | 5.548 |
| distance_file_content (threshold=False) | hdfs_balanced_5k | 0.022 | 0.021 | 0.022 | 0.027 |
| distance_file_content (threshold=False) | bgl_split_10 | 0.043 | 0.063 | 0.148 | 0.248 |
| distance_line_content | hadoop_renamed | 0.101 | 0.090 | 0.205 | 0.358 |
| distance_line_content | hdfs_balanced_5k | 0.037 | 0.042 | 0.063 | 0.069 |
| distance_line_content | bgl_split_10 | 0.065 | 0.080 | 0.289 | 0.361 |

## Table B3 -- Anomaly tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename | hadoop_renamed | 0.163 | 0.178 | 0.183 | 0.317 |
| anomaly_folder_filename | hdfs_balanced_5k | 0.142 | 0.152 | 0.175 | 0.238 |
| anomaly_folder_filename | bgl_split_10 | 0.224 | 0.304 | 0.797 | 1.531 |
| anomaly_folder_filename (threshold=False) | hadoop_renamed | 0.135 | 0.148 | 0.158 | 0.282 |
| anomaly_folder_filename (threshold=False) | hdfs_balanced_5k | 0.129 | 0.150 | 0.163 | 0.207 |
| anomaly_folder_filename (threshold=False) | bgl_split_10 | 0.213 | 0.278 | 0.755 | 1.346 |
| anomaly_folder_content | hadoop_renamed | 0.205 | 0.218 | 0.326 | 0.754 |
| anomaly_folder_content | hdfs_balanced_5k | 0.160 | 0.180 | 0.278 | 0.417 |
| anomaly_folder_content | bgl_split_10 | 0.321 | 0.518 | 2.136 | 7.508 |
| anomaly_folder_content (threshold=False) | hadoop_renamed | 0.190 | 0.203 | 0.311 | 0.718 |
| anomaly_folder_content (threshold=False) | hdfs_balanced_5k | 0.156 | 0.178 | 0.270 | 0.401 |
| anomaly_folder_content (threshold=False) | bgl_split_10 | 0.302 | 0.477 | 1.829 | 7.820 |
| anomaly_file_content | hadoop_renamed | err | 11.2 | 24.8 | 30.3 |
| anomaly_file_content | hdfs_balanced_5k | 0.019 | 0.021 | 0.021 | 0.041 |
| anomaly_file_content | bgl_split_10 | 0.025 | 0.034 | 0.107 | 0.277 |
| anomaly_file_content (threshold=False) | hadoop_renamed | err | 2.090 | 2.339 | 2.922 |
| anomaly_file_content (threshold=False) | hdfs_balanced_5k | 0.018 | 0.022 | 0.023 | 0.033 |
| anomaly_file_content (threshold=False) | bgl_split_10 | 0.031 | 0.039 | 0.158 | 0.288 |
| anomaly_line_content | hadoop_renamed | 0.225 | 0.493 | 0.248 | 0.307 |
| anomaly_line_content | hdfs_balanced_5k | 0.015 | 0.015 | 0.017 | 0.027 |
| anomaly_line_content | bgl_split_10 | 0.027 | 0.036 | 0.123 | 0.297 |
| anomaly_line_content (threshold=False) | hadoop_renamed | 0.236 | 0.485 | 0.244 | 0.288 |
| anomaly_line_content (threshold=False) | hdfs_balanced_5k | 0.017 | 0.015 | 0.019 | 0.027 |
| anomaly_line_content (threshold=False) | bgl_split_10 | 0.031 | 0.043 | 0.128 | 0.296 |

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

## Table B5 -- Sequence tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| sequence_line_event_prediction | hadoop_renamed | 0.103 | 0.104 | 0.125 | 0.233 |
| sequence_line_event_prediction | hdfs_balanced_5k | 0.015 | 0.018 | 0.025 | 0.030 |
| sequence_line_event_prediction | bgl_split_10 | 0.033 | 0.047 | 0.139 | 0.252 |

# Detailed breakdowns (per detector / per measure)

The tables above run every anomaly tool with all four detectors, `distance_folder_content`/`distance_file_content` with their three default measures (cosine, jaccard, containment; compression is opt-in), and `sequence_line_event_prediction` with both order detectors, at once; `distance_line_content` defaults to its coarse pair (Prefix + Exact) in one pass. Part C/D below break the same figure down per detector / per measure run in isolation (`detectors=["<name>"]` / `measures=["<name>"]`), so the cost of narrowing either is visible on its own rather than folded into the combined call -- `distance_line_content` included, run once per bucket measure (Exact, Prefix, Minhash) so they can be compared directly. `distance_folder_filename` (jaccard/overlap distance over file names only) is not broken down further -- it computes one measure, not a default pair. The anomaly rows pass `threshold=False`, so they show one detector's own cost without the extra fits of the clean range; the distance rows keep the default clean range.

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
| distance_folder_content (cosine) | hadoop_renamed | 0.087 | 0.311 | 1.074 | 3.050 |
| distance_folder_content (cosine) | hdfs_balanced_5k | 0.942 | 1.775 | 8.873 | 19.6 |
| distance_folder_content (cosine) | bgl_split_10 | 0.746 | 1.482 | 7.341 | 23.6 |
| distance_folder_content (jaccard) | hadoop_renamed | 0.107 | 0.344 | 1.087 | 2.249 |
| distance_folder_content (jaccard) | hdfs_balanced_5k | 1.114 | 2.062 | 11.6 | 22.5 |
| distance_folder_content (jaccard) | bgl_split_10 | 0.730 | 1.513 | 6.393 | 25.1 |
| distance_folder_content (compression) | hadoop_renamed | 1.072 | 7.989 | 28.5 | 37.6 |
| distance_folder_content (compression) | hdfs_balanced_5k | 1.109 | 2.126 | 10.8 | 21.6 |
| distance_folder_content (compression) | bgl_split_10 | 24.6 | 46.8 | 275.1 | 905.8 |
| distance_folder_content (containment) | hadoop_renamed | 0.101 | 0.265 | 1.122 | 2.100 |
| distance_folder_content (containment) | hdfs_balanced_5k | 0.757 | 1.443 | 7.882 | 17.1 |
| distance_folder_content (containment) | bgl_split_10 | 0.732 | 1.310 | 6.143 | 19.7 |
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
| distance_line_content (Exact) | hadoop_renamed | 0.136 | 0.097 | 0.163 | 2.292* |
| distance_line_content (Exact) | hdfs_balanced_5k | 0.032 | 0.046 | 0.083 | 0.094 |
| distance_line_content (Exact) | bgl_split_10 | 0.073 | 0.086 | 0.275 | 0.513 |
| distance_line_content (Prefix) | hadoop_renamed | 0.085 | 0.073 | 0.147 | 0.365 |
| distance_line_content (Prefix) | hdfs_balanced_5k | 0.036 | 0.037 | 0.053 | 0.065 |
| distance_line_content (Prefix) | bgl_split_10 | 0.076 | 0.086 | 0.258 | 0.433 |
| distance_line_content (Minhash) | hadoop_renamed | 0.120 | 0.114 | 0.200 | 1.460 |
| distance_line_content (Minhash) | hdfs_balanced_5k | 0.034 | 0.035 | 0.058 | 0.062 |
| distance_line_content (Minhash) | bgl_split_10 | 0.072 | 0.087 | 0.242 | 0.513 |

\* `distance_line_content (Exact)` on `hadoop_renamed` at 100% looks like an outlier (2.292s cold
vs. 0.380s warm -- see Table D2), but it isn't about `Exact`. `Exact`
is the first of the three bucket measures run, so its cold call pays the one-time cost of
materializing the `Words` content column for the whole log root. `Prefix`'s and `Minhash`'s cold
cells then reuse that cached column, which is why their own cold numbers stay low. The fair
per-measure comparison is the warm numbers in Table D2, not the cold ones here. Maybe worth eliminating
this measurement-order artifact later (e.g. by resetting the content cache before each measure's
cold cell).

## Table C3 -- Sequence tools detailed

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| sequence_line_event_prediction (NEP) | hadoop_renamed | 0.343 | 0.247 | 0.292 | 0.661 |
| sequence_line_event_prediction (NEP) | hdfs_balanced_5k | 0.018 | 0.020 | 0.021 | 0.044 |
| sequence_line_event_prediction (NEP) | bgl_split_10 | 0.031 | 0.045 | 0.142 | 0.272 |
| sequence_line_event_prediction (LAP) | hadoop_renamed | 0.088 | 0.077 | 0.111 | 0.165 |
| sequence_line_event_prediction (LAP) | hdfs_balanced_5k | 0.019 | 0.022 | 0.021 | 0.039 |
| sequence_line_event_prediction (LAP) | bgl_split_10 | 0.033 | 0.053 | 0.164 | 0.280 |

\*\* **Parsing is priced on its own, and the sequence rows no longer absorb it.**
`sequence_line_event_prediction` defaults to `content_format="Parse-Drain"`, which this grid does
not pre-parse -- it opens with `parsers=["tip"]`. Until this was fixed, whichever sequence cell ran
first charged the entire log root's Drain parse to itself: NEP read **44.0s** on `bgl_split_10` at
100%, of which ~41.5s was parsing, while LAP and the combined call looked cheap only because the
column was already built by then. All three sequence cells now materialize `Parse-Drain` *before*
the timed call, so each measures its detector rather than the parse. NEP at that same cell is now
0.272s against LAP's 0.280s -- they cost about the same, which the old numbers hid completely.

The parse itself has no tool of its own: parsing arrives either through
`open_log_root(parsers=[...])` or lazily, inside whichever analysis first asks for a
`Parse-<Algorithm>` format. So a parser's cost is a **difference between two cold opens** in Table
A1. On `bgl_split_10` at 100% that is 25.2 - 21.9 ~= **3.3s for tip** and 54.9 - 21.9 ~= **33s for
Drain**; on `hadoop_renamed` and `hdfs_balanced_5k` both stay under ~2.4s at every fraction.

Read the **cold** column of those three rows only. `parsers` is not part of the parquet cache key
(`SessionStore._cache_key`), so every variant shares one cache file and `flush` writes whatever it
parsed into it -- a cached re-attach can be handed a frame an earlier open already parsed. That is
why all three cold calls pass `refresh=True`, and why their warm column is a re-attach rather than
a parse. The plain `open_log_root` row is the `parsers=["tip"]` case measured independently, so it
and `open_log_root (parse tip)` should agree.

Rows measured against an already-built representation, and so *not* paying for it: all of Table
A5/B5/C3/D3 (`Parse-Drain`, deliberately, as above), and `distance_line_content`'s `Prefix` and
`Minhash` cold cells (`Words`, as a side effect of `Exact` running first -- see the `*` note).

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
| distance_folder_content (cosine) | hadoop_renamed | 0.091 | 0.186 | 0.915 | 1.943 |
| distance_folder_content (cosine) | hdfs_balanced_5k | 0.880 | 1.774 | 9.749 | 18.9 |
| distance_folder_content (cosine) | bgl_split_10 | 0.542 | 1.042 | 4.416 | 12.7 |
| distance_folder_content (jaccard) | hadoop_renamed | 0.109 | 0.246 | 0.928 | 2.012 |
| distance_folder_content (jaccard) | hdfs_balanced_5k | 1.040 | 2.064 | 10.6 | 23.1 |
| distance_folder_content (jaccard) | bgl_split_10 | 0.560 | 1.208 | 4.625 | 13.3 |
| distance_folder_content (compression) | hadoop_renamed | 1.738 | 1.956 | 9.820 | 20.2 |
| distance_folder_content (compression) | hdfs_balanced_5k | 0.965 | 2.138 | 10.2 | 21.5 |
| distance_folder_content (compression) | bgl_split_10 | 5.361 | 9.848 | 50.5 | 139.3 |
| distance_folder_content (containment) | hadoop_renamed | 0.099 | 0.168 | 0.898 | 1.876 |
| distance_folder_content (containment) | hdfs_balanced_5k | 0.732 | 1.496 | 8.084 | 16.3 |
| distance_folder_content (containment) | bgl_split_10 | 0.617 | 0.941 | 4.298 | 12.2 |
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
| distance_line_content (Exact) | hadoop_renamed | 0.078 | 0.077 | 0.195 | 0.380 |
| distance_line_content (Exact) | hdfs_balanced_5k | 0.034 | 0.040 | 0.063 | 0.068 |
| distance_line_content (Exact) | bgl_split_10 | 0.070 | 0.089 | 0.253 | 0.443 |
| distance_line_content (Prefix) | hadoop_renamed | 0.084 | 0.090 | 0.125 | 0.313 |
| distance_line_content (Prefix) | hdfs_balanced_5k | 0.034 | 0.037 | 0.062 | 0.062 |
| distance_line_content (Prefix) | bgl_split_10 | 0.072 | 0.083 | 0.221 | 0.477 |
| distance_line_content (Minhash) | hadoop_renamed | 0.115 | 0.118 | 0.183 | 0.797 |
| distance_line_content (Minhash) | hdfs_balanced_5k | 0.032 | 0.039 | 0.053 | 0.063 |
| distance_line_content (Minhash) | bgl_split_10 | 0.069 | 0.082 | 0.226 | 0.420 |

## Table D3 -- Sequence tools detailed

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| sequence_line_event_prediction (NEP) | hadoop_renamed | 0.083 | 0.083 | 0.112 | 0.200 |
| sequence_line_event_prediction (NEP) | hdfs_balanced_5k | 0.019 | 0.023 | 0.022 | 0.045 |
| sequence_line_event_prediction (NEP) | bgl_split_10 | 0.033 | 0.044 | 0.155 | 0.255 |
| sequence_line_event_prediction (LAP) | hadoop_renamed | 0.082 | 0.076 | 0.097 | 0.249 |
| sequence_line_event_prediction (LAP) | hdfs_balanced_5k | 0.019 | 0.020 | 0.022 | 0.035 |
| sequence_line_event_prediction (LAP) | bgl_split_10 | 0.035 | 0.051 | 0.153 | 0.259 |

# When this data was measured

Taken from git history (`git blame`), not from re-running the grid -- so a group's date is when
its numbers last *changed* in this file. A cell whose value happened to round the same across two
runs still carries the older commit, so these dates are a lower bound on freshness, not an exact
measurement date.

- Log roots shape table: 2026-09-10, `e04752f` ("Performance test report") -- unchanged since.
- Tables A1-B4 (base grid: aux/distance/anomaly/plot, combined calls): 2026-09-10, `6f0238f`
  ("Memory measurements added") through `21235f9` ("Memory measurement improvements") -- except:
  - `read_log_lines (new tokens)` / `new_tokens` rows (Table A1/B1): 2026-09-11, `f2a5c50`
    ("Vocabulary analyzer added").
  - `distance_line_content` rows (Table A2/B2): 2026-09-13, `475a7ff` ("Benchmark updates").
- Tables C1-D2 (per-detector / per-measure detail, incl. OOVDetector): 2026-09-10, `0e072ec`
  ("Distance measures options added") -- except:
  - `distance_line_content (Exact/Prefix/Minhash)` rows (Table C2/D2): 2026-09-13, `475a7ff`
    ("Benchmark updates").
- Tables A5/B5/C3/D3 (`sequence_line_event_prediction`, NEP + LAP) and the three
  `open_log_root (no parsers|parse tip|parse drain)` rows in Table A1/B1: 2026-09-23, not yet
  committed -- the sequence rows re-measured with `Parse-Drain` pre-built, and the parse itself
  broken out into its own rows, so no analysis row is charged for it. Supersedes the first
  sequence measurement of 2026-09-22, where NEP absorbed the parse (see the `**` note).
- All `distance_folder_content` rows (Table A2/B2/C2/D2), including the new
  `distance_folder_content (threshold=False)` row: 2026-09-30, not yet committed. Re-measured
  after two changes, so they are not comparable with the older rows around them:
  - compression became opt-in (`bfcfb6f`), so the combined call runs cosine, jaccard and
    containment only. Most of the drop from the previous figures (e.g. 147.7s -> 18.9s on
    `bgl_split_10` at 100%) is that.
  - the call now also returns `clean_range` by default: how much the baseline folders differ
    from each other, from a sample of at most 10 of them (45 pairs). The
    `threshold=False` row is the same call without it; the gap between the two rows is the
    range's cost. Warm calls reuse a range cached in the session, which is why the cold-warm gap
    widened on `bgl_split_10`.
  - One OOM kill on `distance_folder_content (cosine)` at `bgl_split_10` 100% (the first cell
    after opening that log root) did not reproduce on a re-run; the re-run's figures are shown.
- The `distance_folder_filename`, `distance_file_content` and four `anomaly_*` rows (Table
  A2/B2/A3/B3), each with a new `(threshold=False)` row: 2026-09-30, not yet committed. All of
  them now return a clean range by default; the `(threshold=False)` row is the same call without
  it, so the gap between the two is the range's cost. `distance_file_content` also no longer
  includes compression (`bfcfb6f`).
  - distance tools: pairwise distances between at most 10 sampled baseline folders, per file
    for `distance_file_content`. Costs little beyond the call itself.
  - anomaly tools: each of up to 10 sampled baseline folders is scored like a target against
    the others, so up to 10 extra model fits -- per file for `anomaly_file_content` and
    `anomaly_line_content`. These rows score one target against all others, which is the
    costliest case; with `target_folder="ALL"` the fits the call already made are reused. Warm
    calls reuse the range cached in the session.
  - `anomaly_folder_content` at `bgl_split_10` 100% was OOM-killed once with the range on and
    measured 69.8s / 10.5 GB on the re-run shown, against 8.9s / 5.3 GB without it.
  - The per-detector anomaly rows (Table C1/D1) were not re-measured; the benchmark now runs them
    with `threshold=False`, which is what they measured before the range existed.
