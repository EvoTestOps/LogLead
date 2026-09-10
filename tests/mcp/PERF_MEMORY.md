# MCP tool memory

Cells: GB, peak resident memory sampled every 20ms while the call ran. Columns: the four
data fractions.

What "5%" means differs by log root: for `hadoop_renamed` and `hdfs_balanced_5k` it is 5% of the
**log folders** (and their files); for `bgl_split_10` it is 5% of `BGL.log`'s **log lines**, taken
first and then split into 10 slices, so the folder count is always 10 and what varies is the text
inside each one.

Tables A1-A4 are **cold** (first call, nothing cached). Tables B1-B4 are the same grid
**warm** -- the max across the repeats, same process, rather than the median: a peak that
only showed up once is still the one a client should plan for.

**A cell is the whole process's peak while the call ran, not the call's own allocation.**
Every cell of a block runs with the log root already open, so the session's frame is
resident underneath -- on bgl at 100% that is ~3GB before any tool is called, and it is why
the aux column climbs down a bgl column. Read a cell as what a machine running this call on
this log root needs, which is the number that OOM-kills.

`peek_log_root` and `split_log_file` are the exception: they are measured **before** the
block opens anything, since neither needs a session. Peek stats the files and reads a few
hundred lines; split streams its file through an 8MB buffer. Their floor is the server's own
imports (~0.31GB: polars, sklearn, plotly, the MCP SDK), paid once at startup and shared by
every tool. Measured *after* the open, as they were in the first grid to record memory, peek
read 5.14GB on bgl at 100% -- all but ~0.02 of it the frame sitting beside it. Neither
scales with the data: peek is ~20-30MB from 10 log folders to 5,000 and from 0.01GB to
0.74GB of logs, because it counts files and samples lines rather than reading them.

Cells recorded before `loglead.delta` stopped importing umap eagerly read ~0.2GB high. Clear
the cell directory to re-measure the grid from scratch.

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
| peek_log_root | hadoop_renamed | 0.314 | 0.315 | 0.316 | 0.315 |
| peek_log_root | hdfs_balanced_5k | 0.316 | 0.316 | 0.318 | 0.320 |
| peek_log_root | bgl_split_10 | 0.310 | 0.310 | 0.310 | 0.310 |
| open_log_root | hadoop_renamed | 0.403 | 0.442 | 0.852 | 1.50 |
| open_log_root | hdfs_balanced_5k | 0.578 | 0.606 | 0.822 | 1.08 |
| open_log_root | bgl_split_10 | 1.03 | 1.42 | 4.50 | 8.50 |
| list_log_roots | hadoop_renamed | 0.404 | 0.447 | 0.858 | 1.51 |
| list_log_roots | hdfs_balanced_5k | 0.579 | 0.608 | 0.830 | 1.10 |
| list_log_roots | bgl_split_10 | 1.09 | 1.53 | 5.07 | 9.72 |
| describe_log_root | hadoop_renamed | 0.405 | 0.447 | 0.858 | 1.51 |
| describe_log_root | hdfs_balanced_5k | 0.579 | 0.609 | 0.830 | 1.10 |
| describe_log_root | bgl_split_10 | 1.13 | 1.58 | 5.07 | 9.77 |
| set_folder_names | hadoop_renamed | 0.639 | 0.946 | 1.44 | 2.58 |
| set_folder_names | hdfs_balanced_5k | 0.773 | 0.841 | 1.29 | 1.62 |
| set_folder_names | bgl_split_10 | 1.67 | 2.27 | 6.52 | 12.80 |
| read_log_lines | hadoop_renamed | 0.405 | 0.447 | 0.859 | 1.51 |
| read_log_lines | hdfs_balanced_5k | 0.580 | 0.609 | 0.831 | 1.10 |
| read_log_lines | bgl_split_10 | 1.16 | 1.62 | 5.07 | 9.77 |
| search_log_lines | hadoop_renamed | 0.405 | 0.447 | 0.859 | 1.51 |
| search_log_lines | hdfs_balanced_5k | 0.580 | 0.609 | 0.831 | 1.10 |
| search_log_lines | bgl_split_10 | 1.16 | 1.62 | 5.08 | 9.78 |
| query_result | hadoop_renamed | 0.597 | 0.883 | 1.33 | 2.44 |
| query_result | hdfs_balanced_5k | 0.749 | 0.801 | 1.14 | 1.49 |
| query_result | bgl_split_10 | 1.42 | 1.96 | 5.93 | 12.30 |
| split_log_file | hadoop_renamed | 0.318 | 0.319 | 0.320 | 0.326 |
| split_log_file | hdfs_balanced_5k | 0.317 | 0.318 | 0.319 | 0.322 |
| split_log_file | bgl_split_10 | 0.313 | 0.313 | 0.330 | 0.330 |
| close_log_root | hadoop_renamed | 0.665 | 1.00 | 1.51 | 2.73 |
| close_log_root | hdfs_balanced_5k | 0.783 | 0.857 | 1.36 | 1.68 |
| close_log_root | bgl_split_10 | 1.70 | 2.30 | 6.53 | 12.88 |
| run_config | hadoop_renamed | 0.605 | 0.906 | 1.35 | 2.46 |
| run_config | hdfs_balanced_5k | 0.757 | 0.817 | 1.19 | 1.57 |
| run_config | bgl_split_10 | 1.52 | 2.11 | 6.38 | 13.13 |

## Table A2 -- Distance tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| distance_folder_filename | hadoop_renamed | 0.406 | 0.448 | 0.859 | 1.51 |
| distance_folder_filename | hdfs_balanced_5k | 0.581 | 0.610 | 0.832 | 1.10 |
| distance_folder_filename | bgl_split_10 | 1.17 | 1.63 | 5.08 | 9.79 |
| distance_folder_content | hadoop_renamed | 0.423 | 0.462 | 0.872 | 1.53 |
| distance_folder_content | hdfs_balanced_5k | 0.583 | 0.612 | 0.836 | 1.10 |
| distance_folder_content | bgl_split_10 | 1.18 | 1.64 | 5.16 | 10.38 |
| distance_file_content | hadoop_renamed | 0.424 | 0.470 | 1.08 | 1.54 |
| distance_file_content | hdfs_balanced_5k | 0.584 | 0.613 | 0.837 | 1.10 |
| distance_file_content | bgl_split_10 | 1.18 | 1.64 | 5.18 | 10.16 |
| distance_line_content | hadoop_renamed | 0.435 | 0.488 | 1.10 | 1.56 |
| distance_line_content | hdfs_balanced_5k | 0.585 | 0.617 | 0.852 | 1.11 |
| distance_line_content | bgl_split_10 | 1.18 | 1.65 | 5.26 | 10.20 |

## Table A3 -- Anomaly tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename | hadoop_renamed | 0.439 | 0.508 | 1.12 | 1.56 |
| anomaly_folder_filename | hdfs_balanced_5k | 0.590 | 0.624 | 0.881 | 1.12 |
| anomaly_folder_filename | bgl_split_10 | 1.19 | 1.67 | 5.40 | 10.26 |
| anomaly_folder_content | hadoop_renamed | 0.447 | 0.525 | 1.19 | 1.77 |
| anomaly_folder_content | hdfs_balanced_5k | 0.595 | 0.636 | 0.953 | 1.25 |
| anomaly_folder_content | bgl_split_10 | 1.23 | 1.77 | 5.67 | 12.35 |
| anomaly_file_content | hadoop_renamed | err | 0.549 | 1.25 | 2.38 |
| anomaly_file_content | hdfs_balanced_5k | 0.598 | 0.649 | 0.980 | 1.28 |
| anomaly_file_content | bgl_split_10 | 1.25 | 1.80 | 5.73 | 12.55 |
| anomaly_line_content | hadoop_renamed | 0.519 | 0.593 | 1.22 | 2.30 |
| anomaly_line_content | hdfs_balanced_5k | 0.601 | 0.650 | 0.980 | 1.28 |
| anomaly_line_content | bgl_split_10 | 1.25 | 1.80 | 5.73 | 12.56 |

## Table A4 -- Plot tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| plot_folder_filename | hadoop_renamed | 0.571 | 0.595 | 1.25 | 2.31 |
| plot_folder_filename | hdfs_balanced_5k | 0.627 | 0.672 | 0.991 | 1.31 |
| plot_folder_filename | bgl_split_10 | 1.27 | 1.82 | 5.73 | 12.61 |
| plot_folder_content (scatter) | hadoop_renamed | 0.590 | 0.585 | 1.29 | 2.44 |
| plot_folder_content (scatter) | hdfs_balanced_5k | 0.630 | 0.678 | 1.01 | 1.32 |
| plot_folder_content (scatter) | bgl_split_10 | 1.31 | 1.90 | 5.98 | 13.24 |
| plot_folder_content (scatter+umap) | hadoop_renamed | err | 0.915 | 1.38 | 2.53 |
| plot_folder_content (scatter+umap) | hdfs_balanced_5k | 0.757 | 0.805 | 1.14 | 1.52 |
| plot_folder_content (scatter+umap) | bgl_split_10 | 1.41 | 1.96 | 6.16 | 13.21 |
| plot_file_content (scatter) | hadoop_renamed | 0.595 | 0.897 | 1.35 | 2.46 |
| plot_file_content (scatter) | hdfs_balanced_5k | 0.748 | 0.800 | 1.14 | 1.49 |
| plot_file_content (scatter) | bgl_split_10 | 1.42 | 1.96 | 5.89 | 12.28 |
| plot_file_content (scatter+umap) | hadoop_renamed | err | 0.916 | 1.35 | 2.46 |
| plot_file_content (scatter+umap) | hdfs_balanced_5k | 0.748 | 0.800 | 1.14 | 1.49 |
| plot_file_content (scatter+umap) | bgl_split_10 | 1.42 | 1.96 | 5.89 | 12.29 |

# Part B -- warm (repeated call)

## Table B1 -- Auxiliary tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| peek_log_root | hadoop_renamed | 0.317 | 0.319 | 0.320 | 0.319 |
| peek_log_root | hdfs_balanced_5k | 0.317 | 0.318 | 0.319 | 0.321 |
| peek_log_root | bgl_split_10 | 0.313 | 0.313 | 0.314 | 0.313 |
| open_log_root | hadoop_renamed | 0.404 | 0.447 | 0.858 | 1.51 |
| open_log_root | hdfs_balanced_5k | 0.579 | 0.608 | 0.830 | 1.10 |
| open_log_root | bgl_split_10 | 1.09 | 1.53 | 5.09 | 9.77 |
| list_log_roots | hadoop_renamed | 0.404 | 0.447 | 0.858 | 1.51 |
| list_log_roots | hdfs_balanced_5k | 0.579 | 0.609 | 0.830 | 1.10 |
| list_log_roots | bgl_split_10 | 1.09 | 1.53 | 5.07 | 9.72 |
| describe_log_root | hadoop_renamed | 0.405 | 0.447 | 0.858 | 1.51 |
| describe_log_root | hdfs_balanced_5k | 0.580 | 0.609 | 0.831 | 1.10 |
| describe_log_root | bgl_split_10 | 1.16 | 1.62 | 5.07 | 9.77 |
| set_folder_names | hadoop_renamed | 0.662 | 0.993 | 1.50 | 2.71 |
| set_folder_names | hdfs_balanced_5k | 0.781 | 0.855 | 1.34 | 1.68 |
| set_folder_names | bgl_split_10 | 1.70 | 2.29 | 6.53 | 12.91 |
| read_log_lines | hadoop_renamed | 0.405 | 0.447 | 0.859 | 1.51 |
| read_log_lines | hdfs_balanced_5k | 0.580 | 0.609 | 0.831 | 1.10 |
| read_log_lines | bgl_split_10 | 1.16 | 1.62 | 5.07 | 9.77 |
| search_log_lines | hadoop_renamed | 0.405 | 0.447 | 0.859 | 1.51 |
| search_log_lines | hdfs_balanced_5k | 0.580 | 0.609 | 0.831 | 1.10 |
| search_log_lines | bgl_split_10 | 1.16 | 1.63 | 5.08 | 9.79 |
| query_result | hadoop_renamed | 0.597 | 0.883 | 1.33 | 2.44 |
| query_result | hdfs_balanced_5k | 0.749 | 0.801 | 1.14 | 1.49 |
| query_result | bgl_split_10 | 1.42 | 1.96 | 5.93 | 12.30 |
| split_log_file | hadoop_renamed | 0.318 | 0.319 | 0.320 | 0.319 |
| split_log_file | hdfs_balanced_5k | 0.317 | 0.318 | 0.319 | 0.322 |
| split_log_file | bgl_split_10 | 0.313 | 0.313 | 0.314 | 0.313 |
| close_log_root | hadoop_renamed | 0.665 | 1.00 | 1.51 | 2.73 |
| close_log_root | hdfs_balanced_5k | 0.783 | 0.857 | 1.36 | 1.68 |
| close_log_root | bgl_split_10 | 1.70 | 2.30 | 6.53 | 12.88 |
| run_config | hadoop_renamed | 0.626 | 0.916 | 1.39 | 2.52 |
| run_config | hdfs_balanced_5k | 0.766 | 0.830 | 1.26 | 1.59 |
| run_config | bgl_split_10 | 1.64 | 2.26 | 6.54 | 12.92 |

## Table B2 -- Distance tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| distance_folder_filename | hadoop_renamed | 0.406 | 0.448 | 0.860 | 1.51 |
| distance_folder_filename | hdfs_balanced_5k | 0.581 | 0.610 | 0.833 | 1.10 |
| distance_folder_filename | bgl_split_10 | 1.17 | 1.63 | 5.08 | 9.79 |
| distance_folder_content | hadoop_renamed | 0.424 | 0.463 | 0.878 | 1.54 |
| distance_folder_content | hdfs_balanced_5k | 0.584 | 0.613 | 0.837 | 1.10 |
| distance_folder_content | bgl_split_10 | 1.18 | 1.64 | 5.17 | 10.38 |
| distance_file_content | hadoop_renamed | 0.432 | 0.484 | 1.09 | 1.56 |
| distance_file_content | hdfs_balanced_5k | 0.584 | 0.613 | 0.837 | 1.10 |
| distance_file_content | bgl_split_10 | 1.18 | 1.65 | 5.26 | 10.20 |
| distance_line_content | hadoop_renamed | 0.438 | 0.500 | 1.11 | 1.56 |
| distance_line_content | hdfs_balanced_5k | 0.587 | 0.620 | 0.867 | 1.11 |
| distance_line_content | bgl_split_10 | 1.18 | 1.65 | 5.26 | 10.20 |

## Table B3 -- Anomaly tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename | hadoop_renamed | 0.439 | 0.508 | 1.12 | 1.57 |
| anomaly_folder_filename | hdfs_balanced_5k | 0.591 | 0.626 | 0.897 | 1.13 |
| anomaly_folder_filename | bgl_split_10 | 1.21 | 1.69 | 5.41 | 10.24 |
| anomaly_folder_content | hadoop_renamed | 0.459 | 0.548 | 1.24 | 2.24 |
| anomaly_folder_content | hdfs_balanced_5k | 0.598 | 0.649 | 0.994 | 1.31 |
| anomaly_folder_content | bgl_split_10 | 1.26 | 1.84 | 5.91 | 13.24 |
| anomaly_file_content | hadoop_renamed | err | 0.553 | 1.23 | 2.36 |
| anomaly_file_content | hdfs_balanced_5k | 0.600 | 0.649 | 0.980 | 1.28 |
| anomaly_file_content | bgl_split_10 | 1.25 | 1.80 | 5.73 | 12.56 |
| anomaly_line_content | hadoop_renamed | 0.575 | 0.576 | 1.23 | 2.32 |
| anomaly_line_content | hdfs_balanced_5k | 0.601 | 0.651 | 0.980 | 1.28 |
| anomaly_line_content | bgl_split_10 | 1.25 | 1.80 | 5.73 | 12.57 |

## Table B4 -- Plot tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| plot_folder_filename | hadoop_renamed | 0.586 | 0.596 | 1.25 | 2.31 |
| plot_folder_filename | hdfs_balanced_5k | 0.647 | 0.681 | 0.995 | 1.29 |
| plot_folder_filename | bgl_split_10 | 1.30 | 1.84 | 5.61 | 12.62 |
| plot_folder_content (scatter) | hadoop_renamed | 0.598 | 0.620 | 1.29 | 2.46 |
| plot_folder_content (scatter) | hdfs_balanced_5k | 0.650 | 0.681 | 1.01 | 1.33 |
| plot_folder_content (scatter) | bgl_split_10 | 1.33 | 1.91 | 6.12 | 13.23 |
| plot_folder_content (scatter+umap) | hadoop_renamed | err | 0.916 | 1.38 | 2.56 |
| plot_folder_content (scatter+umap) | hdfs_balanced_5k | 0.758 | 0.810 | 1.16 | 1.54 |
| plot_folder_content (scatter+umap) | bgl_split_10 | 1.43 | 2.01 | 6.20 | 13.20 |
| plot_file_content (scatter) | hadoop_renamed | 0.597 | 0.897 | 1.35 | 2.46 |
| plot_file_content (scatter) | hdfs_balanced_5k | 0.748 | 0.800 | 1.14 | 1.49 |
| plot_file_content (scatter) | bgl_split_10 | 1.42 | 1.96 | 5.89 | 12.29 |
| plot_file_content (scatter+umap) | hadoop_renamed | err | 0.917 | 1.35 | 2.45 |
| plot_file_content (scatter+umap) | hdfs_balanced_5k | 0.748 | 0.801 | 1.14 | 1.49 |
| plot_file_content (scatter+umap) | bgl_split_10 | 1.42 | 1.96 | 5.89 | 12.30 |
