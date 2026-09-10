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
| peek_log_root | hadoop_renamed | 0.108 | 0.115 | 0.131 | 0.200 |
| peek_log_root | hdfs_balanced_5k | 0.173 | 0.193 | 0.202 | 0.224 |
| peek_log_root | bgl_split_10 | 0.026 | 0.034 | 0.033 | 0.029 |
| open_log_root | hadoop_renamed | 1.232 | 1.215 | 2.506 | 4.756 |
| open_log_root | hdfs_balanced_5k | 1.558 | 2.130 | 2.567 | 3.932 |
| open_log_root | bgl_split_10 | 1.489 | 2.171 | 10.8 | 24.5 |
| list_log_roots | hadoop_renamed | 0.010 | 0.011 | 0.016 | 0.022 |
| list_log_roots | hdfs_balanced_5k | 0.006 | 0.012 | 0.013 | 0.017 |
| list_log_roots | bgl_split_10 | 0.036 | 0.039 | 0.161 | 0.274 |
| describe_log_root | hadoop_renamed | 0.013 | 0.014 | 0.020 | 0.047 |
| describe_log_root | hdfs_balanced_5k | 0.015 | 0.021 | 0.021 | 0.033 |
| describe_log_root | bgl_split_10 | 0.050 | 0.064 | 0.294 | 0.482 |
| set_folder_names | hadoop_renamed | 0.046 | 0.038 | 0.112 | 0.158 |
| set_folder_names | hdfs_balanced_5k | 0.024 | 0.048 | 0.113 | 0.110 |
| set_folder_names | bgl_split_10 | 0.088 | 0.161 | 1.018 | 2.647 |
| read_log_lines | hadoop_renamed | 0.011 | 0.001 | 0.002 | 0.004 |
| read_log_lines | hdfs_balanced_5k | 0.001 | 0.002 | 0.001 | 0.003 |
| read_log_lines | bgl_split_10 | 0.007 | 0.008 | 0.017 | 0.031 |
| search_log_lines | hadoop_renamed | 0.005 | 0.007 | 0.011 | 0.018 |
| search_log_lines | hdfs_balanced_5k | 0.005 | 0.007 | 0.008 | 0.010 |
| search_log_lines | bgl_split_10 | 0.029 | 0.028 | 0.066 | 0.155 |
| query_result | hadoop_renamed | 0.002 | 0.003 | 0.003 | 0.004 |
| query_result | hdfs_balanced_5k | 0.002 | 0.003 | 0.007 | 0.007 |
| query_result | bgl_split_10 | 0.002 | 0.003 | 0.003 | 0.003 |
| split_log_file | hadoop_renamed | 0.003 | 0.003 | 0.021 | 0.062 |
| split_log_file | hdfs_balanced_5k | 0.001 | 0.001 | 0.001 | 0.001 |
| split_log_file | bgl_split_10 | 0.011 | 0.023 | 0.167 | 0.338 |
| close_log_root | hadoop_renamed | 0.011 | 0.011 | 0.014 | 0.022 |
| close_log_root | hdfs_balanced_5k | 0.005 | 0.011 | 0.015 | 0.016 |
| close_log_root | bgl_split_10 | 0.027 | 0.036 | 0.137 | 0.253 |
| run_config | hadoop_renamed | 1.764 | 3.041 | 9.463 | 20.7 |
| run_config | hdfs_balanced_5k | 2.127 | 3.580 | 19.3 | 35.6 |
| run_config | bgl_split_10 | 6.015 | 10.2 | 60.7 | 150.1 |

## Table A2 -- Distance tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| distance_folder_filename | hadoop_renamed | 0.039 | 0.067 | 0.298 | 1.330 |
| distance_folder_filename | hdfs_balanced_5k | 1.565 | 4.366 | 20.3 | 33.2 |
| distance_folder_filename | bgl_split_10 | 0.196 | 0.169 | 0.298 | 0.449 |
| distance_folder_content | hadoop_renamed | 0.714 | 1.309 | 7.100 | 18.8 |
| distance_folder_content | hdfs_balanced_5k | 1.913 | 4.079 | 18.4 | 44.5 |
| distance_folder_content | bgl_split_10 | 8.613 | 10.4 | 55.8 | 152.8 |
| distance_file_content | hadoop_renamed | 0.737 | 1.555 | 7.996 | 17.7 |
| distance_file_content | hdfs_balanced_5k | 0.019 | 0.022 | 0.022 | 0.023 |
| distance_file_content | bgl_split_10 | 0.056 | 0.067 | 0.260 | 0.611 |
| distance_line_content | hadoop_renamed | 0.057 | 0.074 | 0.587 | 1.193 |
| distance_line_content | hdfs_balanced_5k | 0.635 | 1.188 | 6.275 | 12.7 |
| distance_line_content | bgl_split_10 | 0.093 | 0.102 | 0.266 | 0.373 |

## Table A3 -- Anomaly tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename | hadoop_renamed | 0.311 | 0.155 | 0.194 | 0.216 |
| anomaly_folder_filename | hdfs_balanced_5k | 0.156 | 0.161 | 0.199 | 0.192 |
| anomaly_folder_filename | bgl_split_10 | 0.228 | 0.337 | 0.999 | 1.458 |
| anomaly_folder_content | hadoop_renamed | 0.212 | 0.188 | 0.380 | 0.665 |
| anomaly_folder_content | hdfs_balanced_5k | 0.153 | 0.181 | 0.311 | 0.473 |
| anomaly_folder_content | bgl_split_10 | 0.298 | 0.485 | 1.912 | 6.408 |
| anomaly_file_content | hadoop_renamed | err | 2.008 | 2.339 | 2.727 |
| anomaly_file_content | hdfs_balanced_5k | 0.019 | 0.022 | 0.032 | 0.033 |
| anomaly_file_content | bgl_split_10 | 0.032 | 0.042 | 0.114 | 0.212 |
| anomaly_line_content | hadoop_renamed | 0.725 | 0.290 | 0.370 | 0.352 |
| anomaly_line_content | hdfs_balanced_5k | 0.015 | 0.015 | 0.020 | 0.028 |
| anomaly_line_content | bgl_split_10 | 0.033 | 0.043 | 0.145 | 0.257 |

## Table A4 -- Plot tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| plot_folder_filename | hadoop_renamed | 0.097 | 0.051 | 0.081 | 0.068 |
| plot_folder_filename | hdfs_balanced_5k | 0.112 | 0.098 | 0.145 | 0.148 |
| plot_folder_filename | bgl_split_10 | 0.119 | 0.121 | 0.253 | 0.422 |
| plot_folder_content (scatter) | hadoop_renamed | 0.118 | 0.102 | 0.305 | 0.516 |
| plot_folder_content (scatter) | hdfs_balanced_5k | 0.066 | 0.086 | 0.189 | 0.306 |
| plot_folder_content (scatter) | bgl_split_10 | 0.206 | 0.439 | 1.736 | 7.478 |
| plot_folder_content (scatter+umap) | hadoop_renamed | err | 7.759 | 8.854 | 8.135 |
| plot_folder_content (scatter+umap) | hdfs_balanced_5k | 8.238 | 8.849 | 18.6 | 27.6 |
| plot_folder_content (scatter+umap) | bgl_split_10 | 7.562 | 7.762 | 10.1 | 13.8 |
| plot_file_content (scatter) | hadoop_renamed | 0.061 | 0.054 | 0.064 | 0.122 |
| plot_file_content (scatter) | hdfs_balanced_5k | 0.014 | 0.013 | 0.017 | 0.017 |
| plot_file_content (scatter) | bgl_split_10 | 0.029 | 0.040 | 0.096 | 0.159 |
| plot_file_content (scatter+umap) | hadoop_renamed | err | 0.099 | 0.144 | 0.287 |
| plot_file_content (scatter+umap) | hdfs_balanced_5k | 0.013 | 0.012 | 0.014 | 0.017 |
| plot_file_content (scatter+umap) | bgl_split_10 | 0.031 | 0.030 | 0.106 | 0.210 |

# Part B -- warm (repeated call)

## Table B1 -- Auxiliary tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| peek_log_root | hadoop_renamed | 0.098 | 0.114 | 0.114 | 0.132 |
| peek_log_root | hdfs_balanced_5k | 0.166 | 0.172 | 0.173 | 0.203 |
| peek_log_root | bgl_split_10 | 0.028 | 0.028 | 0.032 | 0.030 |
| open_log_root | hadoop_renamed | 0.042 | 0.021 | 0.043 | 0.064 |
| open_log_root | hdfs_balanced_5k | 0.022 | 0.026 | 0.039 | 0.073 |
| open_log_root | bgl_split_10 | 0.104 | 0.108 | 0.443 | 0.982 |
| list_log_roots | hadoop_renamed | 0.009 | 0.009 | 0.013 | 0.022 |
| list_log_roots | hdfs_balanced_5k | 0.006 | 0.010 | 0.010 | 0.016 |
| list_log_roots | bgl_split_10 | 0.025 | 0.031 | 0.201 | 0.240 |
| describe_log_root | hadoop_renamed | 0.013 | 0.015 | 0.019 | 0.047 |
| describe_log_root | hdfs_balanced_5k | 0.014 | 0.024 | 0.023 | 0.026 |
| describe_log_root | bgl_split_10 | 0.046 | 0.055 | 0.275 | 0.414 |
| set_folder_names | hadoop_renamed | 0.033 | 0.039 | 0.097 | 0.188 |
| set_folder_names | hdfs_balanced_5k | 0.022 | 0.040 | 0.114 | 0.118 |
| set_folder_names | bgl_split_10 | 0.116 | 0.217 | 0.999 | 2.332 |
| read_log_lines | hadoop_renamed | 0.001 | 0.002 | 0.003 | 0.004 |
| read_log_lines | hdfs_balanced_5k | 0.001 | 0.001 | 0.001 | 0.002 |
| read_log_lines | bgl_split_10 | 0.009 | 0.008 | 0.023 | 0.032 |
| search_log_lines | hadoop_renamed | 0.006 | 0.005 | 0.012 | 0.019 |
| search_log_lines | hdfs_balanced_5k | 0.005 | 0.007 | 0.008 | 0.010 |
| search_log_lines | bgl_split_10 | 0.033 | 0.028 | 0.065 | 0.093 |
| query_result | hadoop_renamed | 0.002 | 0.003 | 0.003 | 0.004 |
| query_result | hdfs_balanced_5k | 0.002 | 0.002 | 0.005 | 0.005 |
| query_result | bgl_split_10 | 0.002 | 0.003 | 0.003 | 0.003 |
| split_log_file | hadoop_renamed | 0.000 | 0.000 | 0.000 | 0.000 |
| split_log_file | hdfs_balanced_5k | 0.000 | 0.000 | 0.000 | 0.000 |
| split_log_file | bgl_split_10 | 0.000 | 0.000 | 0.000 | 0.000 |
| close_log_root | hadoop_renamed | 0.011 | 0.011 | 0.014 | 0.022 |
| close_log_root | hdfs_balanced_5k | 0.005 | 0.011 | 0.015 | 0.016 |
| close_log_root | bgl_split_10 | 0.027 | 0.036 | 0.137 | 0.253 |
| run_config | hadoop_renamed | 0.776 | 1.533 | 7.804 | 15.8 |
| run_config | hdfs_balanced_5k | 1.893 | 3.647 | 18.3 | 36.0 |
| run_config | bgl_split_10 | 6.002 | 10.2 | 53.7 | 149.2 |

## Table B2 -- Distance tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| distance_folder_filename | hadoop_renamed | 0.035 | 0.062 | 0.283 | 1.124 |
| distance_folder_filename | hdfs_balanced_5k | 1.612 | 3.263 | 17.3 | 35.6 |
| distance_folder_filename | bgl_split_10 | 0.186 | 0.153 | 0.299 | 0.448 |
| distance_folder_content | hadoop_renamed | 0.908 | 1.258 | 8.526 | 18.5 |
| distance_folder_content | hdfs_balanced_5k | 1.728 | 4.006 | 19.1 | 36.5 |
| distance_folder_content | bgl_split_10 | 6.674 | 11.7 | 63.5 | 143.1 |
| distance_file_content | hadoop_renamed | 0.711 | 1.774 | 8.201 | 18.0 |
| distance_file_content | hdfs_balanced_5k | 0.017 | 0.022 | 0.024 | 0.025 |
| distance_file_content | bgl_split_10 | 0.052 | 0.061 | 0.165 | 0.351 |
| distance_line_content | hadoop_renamed | 0.065 | 0.066 | 0.538 | 1.124 |
| distance_line_content | hdfs_balanced_5k | 0.577 | 1.192 | 6.306 | 12.8 |
| distance_line_content | bgl_split_10 | 0.097 | 0.132 | 0.260 | 0.370 |

## Table B3 -- Anomaly tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename | hadoop_renamed | 0.172 | 0.137 | 0.158 | 0.189 |
| anomaly_folder_filename | hdfs_balanced_5k | 0.130 | 0.136 | 0.178 | 0.187 |
| anomaly_folder_filename | bgl_split_10 | 0.193 | 0.260 | 0.755 | 1.343 |
| anomaly_folder_content | hadoop_renamed | 0.214 | 0.173 | 0.328 | 0.622 |
| anomaly_folder_content | hdfs_balanced_5k | 0.152 | 0.172 | 0.295 | 0.370 |
| anomaly_folder_content | bgl_split_10 | 0.282 | 0.448 | 1.753 | 6.317 |
| anomaly_file_content | hadoop_renamed | err | 1.974 | 2.300 | 2.676 |
| anomaly_file_content | hdfs_balanced_5k | 0.018 | 0.019 | 0.029 | 0.031 |
| anomaly_file_content | bgl_split_10 | 0.032 | 0.038 | 0.112 | 0.224 |
| anomaly_line_content | hadoop_renamed | 0.372 | 0.213 | 0.266 | 0.273 |
| anomaly_line_content | hdfs_balanced_5k | 0.016 | 0.015 | 0.024 | 0.031 |
| anomaly_line_content | bgl_split_10 | 0.035 | 0.042 | 0.154 | 0.235 |

## Table B4 -- Plot tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| plot_folder_filename | hadoop_renamed | 0.087 | 0.054 | 0.085 | 0.065 |
| plot_folder_filename | hdfs_balanced_5k | 0.060 | 0.061 | 0.090 | 0.116 |
| plot_folder_filename | bgl_split_10 | 0.081 | 0.095 | 0.235 | 0.309 |
| plot_folder_content (scatter) | hadoop_renamed | 0.105 | 0.104 | 0.282 | 0.477 |
| plot_folder_content (scatter) | hdfs_balanced_5k | 0.071 | 0.082 | 0.193 | 0.283 |
| plot_folder_content (scatter) | bgl_split_10 | 0.213 | 0.354 | 1.606 | 9.221 |
| plot_folder_content (scatter+umap) | hadoop_renamed | err | 0.121 | 0.298 | 0.560 |
| plot_folder_content (scatter+umap) | hdfs_balanced_5k | 0.459 | 1.012 | 8.046 | 9.775 |
| plot_folder_content (scatter+umap) | bgl_split_10 | 0.207 | 0.396 | 1.809 | 5.720 |
| plot_file_content (scatter) | hadoop_renamed | 0.056 | 0.057 | 0.073 | 0.118 |
| plot_file_content (scatter) | hdfs_balanced_5k | 0.012 | 0.011 | 0.013 | 0.016 |
| plot_file_content (scatter) | bgl_split_10 | 0.027 | 0.033 | 0.092 | 0.187 |
| plot_file_content (scatter+umap) | hadoop_renamed | err | 0.104 | 0.128 | 0.273 |
| plot_file_content (scatter+umap) | hdfs_balanced_5k | 0.012 | 0.012 | 0.014 | 0.016 |
| plot_file_content (scatter+umap) | bgl_split_10 | 0.028 | 0.034 | 0.133 | 0.195 |
