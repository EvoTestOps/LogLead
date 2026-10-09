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
| peek_log_root | hadoop_renamed | 0.322 | 0.324 | 0.326 | 0.329 |
| peek_log_root | hdfs_balanced_5k | 0.322 | 0.321 | 0.323 | 0.325 |
| peek_log_root | bgl_split_10 | 0.324 | 0.323 | 0.324 | 0.325 |
| open_log_root | hadoop_renamed | 0.503 | 0.554 | 0.829 | 1.59 |
| open_log_root | hdfs_balanced_5k | 0.411 | 0.445 | 0.786 | 1.11 |
| open_log_root | bgl_split_10 | 0.850 | 1.18 | 3.98 | 11.26 |
| open_log_root (no parsers) | hadoop_renamed | 0.510 | 0.549 | 0.828 | 1.61 |
| open_log_root (no parsers) | hdfs_balanced_5k | 0.414 | 0.453 | 0.739 | 0.935 |
| open_log_root (no parsers) | bgl_split_10 | 0.842 | 1.21 | 3.62 | 7.06 |
| open_log_root (parse tip) | hadoop_renamed | 0.514 | 0.549 | 0.849 | 1.58 |
| open_log_root (parse tip) | hdfs_balanced_5k | 0.416 | 0.455 | 0.684 | 0.927 |
| open_log_root (parse tip) | bgl_split_10 | 0.862 | 1.19 | 3.59 | 7.02 |
| open_log_root (parse drain) | hadoop_renamed | 0.524 | 0.558 | 0.849 | 1.61 |
| open_log_root (parse drain) | hdfs_balanced_5k | 0.419 | 0.461 | 0.678 | 0.943 |
| open_log_root (parse drain) | bgl_split_10 | 1.03 | 1.46 | 4.52 | 9.08 |
| list_log_roots | hadoop_renamed | 0.519 | 0.557 | 0.841 | 1.61 |
| list_log_roots | hdfs_balanced_5k | 0.420 | 0.462 | 0.646 | 0.834 |
| list_log_roots | bgl_split_10 | 1.04 | 1.40 | 3.40 | 5.67 |
| describe_log_root | hadoop_renamed | 0.519 | 0.557 | 0.842 | 1.59 |
| describe_log_root | hdfs_balanced_5k | 0.420 | 0.462 | 0.636 | 0.834 |
| describe_log_root | bgl_split_10 | 1.00 | 1.29 | 2.39 | 3.77 |
| set_folder_names | hadoop_renamed | 0.813 | 0.875 | 1.06 | 2.18 |
| set_folder_names | hdfs_balanced_5k | 0.709 | 0.730 | 0.944 | 1.19 |
| set_folder_names | bgl_split_10 | 1.15 | 1.44 | 3.54 | 8.28 |
| read_log_lines | hadoop_renamed | 0.516 | 0.552 | 0.825 | 1.59 |
| read_log_lines | hdfs_balanced_5k | 0.418 | 0.460 | 0.633 | 0.833 |
| read_log_lines | bgl_split_10 | 0.961 | 1.11 | 1.96 | 3.71 |
| search_log_lines | hadoop_renamed | 0.513 | 0.552 | 0.825 | 1.57 |
| search_log_lines | hdfs_balanced_5k | 0.418 | 0.460 | 0.631 | 0.831 |
| search_log_lines | bgl_split_10 | 0.961 | 1.07 | 1.96 | 3.83 |
| filter_log_lines | hadoop_renamed | 0.514 | 0.554 | 0.840 | 1.87 |
| filter_log_lines | hdfs_balanced_5k | 0.419 | 0.460 | 0.652 | 0.847 |
| filter_log_lines | bgl_split_10 | 0.919 | 1.04 | 2.17 | 4.70 |
| new_tokens | hadoop_renamed | 0.511 | 0.537 | 0.820 | 1.56 |
| new_tokens | hdfs_balanced_5k | 0.417 | 0.457 | 0.634 | 0.825 |
| new_tokens | bgl_split_10 | 0.823 | 0.867 | 2.14 | 4.66 |
| query_result | hadoop_renamed | 0.808 | 0.841 | 0.897 | 1.48 |
| query_result | hdfs_balanced_5k | 0.704 | 0.720 | 0.893 | 1.14 |
| query_result | bgl_split_10 | 1.03 | 1.17 | 2.60 | 4.86 |
| split_log_file | hadoop_renamed | 0.328 | 0.333 | 0.339 | 0.345 |
| split_log_file | hdfs_balanced_5k | 0.323 | 0.323 | 0.324 | 0.327 |
| split_log_file | bgl_split_10 | 0.339 | 0.336 | 0.361 | 0.356 |
| close_log_root | hadoop_renamed | 0.845 | 0.924 | 1.14 | 2.21 |
| close_log_root | hdfs_balanced_5k | 0.723 | 0.759 | 1.03 | 1.27 |
| close_log_root | bgl_split_10 | 1.16 | 1.39 | 2.68 | 6.20 |
| run_config | hadoop_renamed | 0.822 | 0.856 | 0.967 | 1.63 |
| run_config | hdfs_balanced_5k | 0.707 | 0.729 | 0.937 | 1.19 |
| run_config | bgl_split_10 | 1.13 | 1.41 | 3.63 | 5.85 |

## Table A2 -- Distance tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| distance_folder_filename | hadoop_renamed | 0.504 | 0.531 | 0.718 | 1.41 |
| distance_folder_filename | hdfs_balanced_5k | 0.416 | 0.450 | 0.608 | 0.763 |
| distance_folder_filename | bgl_split_10 | 0.642 | 0.798 | 1.97 | 3.65 |
| distance_folder_filename (threshold=False) | hadoop_renamed | 0.498 | 0.498 | 0.558 | 1.08 |
| distance_folder_filename (threshold=False) | hdfs_balanced_5k | 0.391 | 0.409 | 0.513 | 0.623 |
| distance_folder_filename (threshold=False) | bgl_split_10 | 0.600 | 0.749 | 1.84 | 3.44 |
| distance_folder_content | hadoop_renamed | 0.492 | 0.489 | 0.561 | 1.22 |
| distance_folder_content | hdfs_balanced_5k | 0.389 | 0.409 | 0.513 | 0.624 |
| distance_folder_content | bgl_split_10 | 0.585 | 0.733 | 1.83 | 3.77 |
| distance_folder_content (threshold=False) | hadoop_renamed | 0.482 | 0.483 | 0.560 | 1.07 |
| distance_folder_content (threshold=False) | hdfs_balanced_5k | 0.389 | 0.408 | 0.510 | 0.623 |
| distance_folder_content (threshold=False) | bgl_split_10 | 0.574 | 0.723 | 1.80 | 3.77 |
| distance_file_content | hadoop_renamed | 0.482 | 0.497 | 0.633 | 2.35 |
| distance_file_content | hdfs_balanced_5k | 0.389 | 0.409 | 0.514 | 0.629 |
| distance_file_content | bgl_split_10 | 0.588 | 0.746 | 1.91 | 4.06 |
| distance_file_content (threshold=False) | hadoop_renamed | 0.490 | 0.499 | 0.639 | 2.29 |
| distance_file_content (threshold=False) | hdfs_balanced_5k | 0.390 | 0.410 | 0.517 | 0.634 |
| distance_file_content (threshold=False) | bgl_split_10 | 0.625 | 0.779 | 2.17 | 4.07 |
| log_line_clustering | hadoop_renamed | 0.490 | 0.496 | 0.578 | 1.81 |
| log_line_clustering | hdfs_balanced_5k | 0.391 | 0.411 | 0.519 | 0.637 |
| log_line_clustering | bgl_split_10 | 0.633 | 0.821 | 2.14 | 4.25 |

## Table A3 -- Anomaly tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename | hadoop_renamed | 0.482 | 0.489 | 0.587 | 2.15 |
| anomaly_folder_filename | hdfs_balanced_5k | 0.393 | 0.415 | 0.528 | 0.665 |
| anomaly_folder_filename | bgl_split_10 | 0.644 | 0.815 | 2.10 | 4.12 |
| anomaly_folder_filename (threshold=False) | hadoop_renamed | 0.475 | 0.483 | 0.575 | 1.54 |
| anomaly_folder_filename (threshold=False) | hdfs_balanced_5k | 0.393 | 0.415 | 0.530 | 0.657 |
| anomaly_folder_filename (threshold=False) | bgl_split_10 | 0.626 | 0.816 | 2.25 | 4.39 |
| anomaly_folder_content | hadoop_renamed | 0.516 | 0.565 | 1.47 | 3.24 |
| anomaly_folder_content | hdfs_balanced_5k | 0.414 | 0.456 | 0.721 | 1.14 |
| anomaly_folder_content | bgl_split_10 | 0.763 | 1.15 | 3.66 | 13.51 |
| anomaly_folder_content (threshold=False) | hadoop_renamed | 0.556 | 0.621 | 1.55 | 2.55 |
| anomaly_folder_content (threshold=False) | hdfs_balanced_5k | 0.423 | 0.465 | 0.743 | 1.28 |
| anomaly_folder_content (threshold=False) | bgl_split_10 | 0.809 | 1.27 | 4.34 | OOM |
| anomaly_file_content | hadoop_renamed | err | 0.683 | 1.26 | 3.29 |
| anomaly_file_content | hdfs_balanced_5k | 0.428 | 0.479 | 0.757 | 0.937 |
| anomaly_file_content | bgl_split_10 | 0.723 | 0.983 | 2.84 | 7.55 |
| anomaly_file_content (threshold=False) | hadoop_renamed | err | 0.561 | 0.976 | 2.14 |
| anomaly_file_content (threshold=False) | hdfs_balanced_5k | 0.428 | 0.477 | 0.757 | 0.855 |
| anomaly_file_content (threshold=False) | bgl_split_10 | 0.707 | 0.961 | 2.82 | 3.14 |
| anomaly_line_content | hadoop_renamed | 0.511 | 0.569 | 0.894 | 1.56 |
| anomaly_line_content | hdfs_balanced_5k | 0.428 | 0.477 | 0.753 | 0.783 |
| anomaly_line_content | bgl_split_10 | 0.701 | 0.960 | 2.82 | 3.11 |
| anomaly_line_content (threshold=False) | hadoop_renamed | 0.511 | 0.562 | 0.894 | 1.59 |
| anomaly_line_content (threshold=False) | hdfs_balanced_5k | 0.421 | 0.477 | 0.736 | 0.780 |
| anomaly_line_content (threshold=False) | bgl_split_10 | 0.698 | 0.960 | 2.81 | 3.10 |

## Table A4 -- Plot tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| plot_folder_filename | hadoop_renamed | 0.555 | 0.582 | 0.905 | 1.84 |
| plot_folder_filename | hdfs_balanced_5k | 0.453 | 0.504 | 0.756 | 0.786 |
| plot_folder_filename | bgl_split_10 | 0.711 | 0.975 | 2.91 | 5.32 |
| plot_folder_content (scatter) | hadoop_renamed | 0.540 | 0.615 | 1.07 | 1.89 |
| plot_folder_content (scatter) | hdfs_balanced_5k | 0.444 | 0.498 | 0.744 | 0.892 |
| plot_folder_content (scatter) | bgl_split_10 | 0.791 | 1.10 | 3.30 | 6.98 |
| plot_folder_content (scatter+umap) | hadoop_renamed | err | 0.812 | 1.11 | 1.92 |
| plot_folder_content (scatter+umap) | hdfs_balanced_5k | 0.724 | 0.734 | 0.908 | 1.13 |
| plot_folder_content (scatter+umap) | bgl_split_10 | 0.947 | 1.13 | 3.29 | 7.20 |
| plot_file_content (scatter) | hadoop_renamed | 0.810 | 0.852 | 1.05 | 1.69 |
| plot_file_content (scatter) | hdfs_balanced_5k | 0.703 | 0.720 | 0.892 | 1.19 |
| plot_file_content (scatter) | bgl_split_10 | 1.03 | 1.27 | 2.69 | 5.03 |
| plot_file_content (scatter+umap) | hadoop_renamed | err | 0.854 | 0.964 | 1.64 |
| plot_file_content (scatter+umap) | hdfs_balanced_5k | 0.704 | 0.719 | 0.892 | 1.19 |
| plot_file_content (scatter+umap) | bgl_split_10 | 1.03 | 1.23 | 2.50 | 5.22 |

## Table A5 -- Sequence tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| sequence_line_event_prediction | hadoop_renamed | 0.531 | 0.560 | 0.895 | 2.02 |
| sequence_line_event_prediction | hdfs_balanced_5k | 0.420 | 0.477 | 0.733 | 0.765 |
| sequence_line_event_prediction | bgl_split_10 | 0.697 | 0.951 | 2.81 | 4.54 |

# Part B -- warm (repeated call)

## Table B1 -- Auxiliary tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| peek_log_root | hadoop_renamed | 0.328 | 0.333 | 0.339 | 0.345 |
| peek_log_root | hdfs_balanced_5k | 0.323 | 0.323 | 0.324 | 0.327 |
| peek_log_root | bgl_split_10 | 0.339 | 0.337 | 0.345 | 0.340 |
| open_log_root | hadoop_renamed | 0.505 | 0.553 | 0.782 | 1.43 |
| open_log_root | hdfs_balanced_5k | 0.412 | 0.447 | 0.755 | 0.925 |
| open_log_root | bgl_split_10 | 0.879 | 1.25 | 4.21 | 10.98 |
| open_log_root (no parsers) | hadoop_renamed | 0.509 | 0.548 | 0.763 | 1.45 |
| open_log_root (no parsers) | hdfs_balanced_5k | 0.416 | 0.455 | 0.638 | 0.748 |
| open_log_root (no parsers) | bgl_split_10 | 0.861 | 1.24 | 3.74 | 7.10 |
| open_log_root (parse tip) | hadoop_renamed | 0.510 | 0.551 | 0.756 | 1.44 |
| open_log_root (parse tip) | hdfs_balanced_5k | 0.418 | 0.457 | 0.639 | 0.783 |
| open_log_root (parse tip) | bgl_split_10 | 0.888 | 1.24 | 3.96 | 7.49 |
| open_log_root (parse drain) | hadoop_renamed | 0.524 | 0.560 | 0.841 | 1.62 |
| open_log_root (parse drain) | hdfs_balanced_5k | 0.421 | 0.463 | 0.647 | 0.846 |
| open_log_root (parse drain) | bgl_split_10 | 1.07 | 1.50 | 3.83 | 7.50 |
| list_log_roots | hadoop_renamed | 0.519 | 0.557 | 0.842 | 1.58 |
| list_log_roots | hdfs_balanced_5k | 0.420 | 0.462 | 0.646 | 0.834 |
| list_log_roots | bgl_split_10 | 1.03 | 1.36 | 3.25 | 4.87 |
| describe_log_root | hadoop_renamed | 0.519 | 0.552 | 0.825 | 1.59 |
| describe_log_root | hdfs_balanced_5k | 0.420 | 0.462 | 0.637 | 0.834 |
| describe_log_root | bgl_split_10 | 1.00 | 1.22 | 2.17 | 3.78 |
| set_folder_names | hadoop_renamed | 0.837 | 0.909 | 1.14 | 2.31 |
| set_folder_names | hdfs_balanced_5k | 0.719 | 0.753 | 1.02 | 1.27 |
| set_folder_names | bgl_split_10 | 1.19 | 1.46 | 2.97 | 8.16 |
| read_log_lines | hadoop_renamed | 0.513 | 0.552 | 0.825 | 1.59 |
| read_log_lines | hdfs_balanced_5k | 0.418 | 0.460 | 0.632 | 0.831 |
| read_log_lines | bgl_split_10 | 0.961 | 1.07 | 1.91 | 3.71 |
| search_log_lines | hadoop_renamed | 0.513 | 0.552 | 0.820 | 1.57 |
| search_log_lines | hdfs_balanced_5k | 0.418 | 0.460 | 0.631 | 0.818 |
| search_log_lines | bgl_split_10 | 0.944 | 1.05 | 2.01 | 3.90 |
| filter_log_lines | hadoop_renamed | 0.513 | 0.545 | 0.822 | 1.75 |
| filter_log_lines | hdfs_balanced_5k | 0.419 | 0.459 | 0.652 | 0.847 |
| filter_log_lines | bgl_split_10 | 0.894 | 0.982 | 2.10 | 4.70 |
| new_tokens | hadoop_renamed | 0.509 | 0.537 | 0.797 | 1.56 |
| new_tokens | hdfs_balanced_5k | 0.417 | 0.457 | 0.634 | 0.823 |
| new_tokens | bgl_split_10 | 0.762 | 0.850 | 2.17 | 4.64 |
| query_result | hadoop_renamed | 0.808 | 0.841 | 0.897 | 1.48 |
| query_result | hdfs_balanced_5k | 0.704 | 0.720 | 0.893 | 1.12 |
| query_result | bgl_split_10 | 1.03 | 1.17 | 2.58 | 4.86 |
| split_log_file | hadoop_renamed | 0.328 | 0.333 | 0.339 | 0.345 |
| split_log_file | hdfs_balanced_5k | 0.323 | 0.323 | 0.324 | 0.327 |
| split_log_file | bgl_split_10 | 0.339 | 0.336 | 0.344 | 0.339 |
| close_log_root | hadoop_renamed | 0.845 | 0.924 | 1.14 | 2.21 |
| close_log_root | hdfs_balanced_5k | 0.723 | 0.759 | 1.03 | 1.27 |
| close_log_root | bgl_split_10 | 1.16 | 1.39 | 2.68 | 6.20 |
| run_config | hadoop_renamed | 0.826 | 0.867 | 0.995 | 1.66 |
| run_config | hdfs_balanced_5k | 0.710 | 0.735 | 0.960 | 1.21 |
| run_config | bgl_split_10 | 1.18 | 1.50 | 3.89 | 6.07 |

## Table B2 -- Distance tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| distance_folder_filename | hadoop_renamed | 0.504 | 0.523 | 0.588 | 1.23 |
| distance_folder_filename | hdfs_balanced_5k | 0.391 | 0.411 | 0.512 | 0.623 |
| distance_folder_filename | bgl_split_10 | 0.618 | 0.805 | 2.00 | 3.66 |
| distance_folder_filename (threshold=False) | hadoop_renamed | 0.498 | 0.498 | 0.558 | 1.07 |
| distance_folder_filename (threshold=False) | hdfs_balanced_5k | 0.390 | 0.409 | 0.513 | 0.624 |
| distance_folder_filename (threshold=False) | bgl_split_10 | 0.590 | 0.750 | 1.82 | 3.44 |
| distance_folder_content | hadoop_renamed | 0.490 | 0.485 | 0.562 | 1.22 |
| distance_folder_content | hdfs_balanced_5k | 0.389 | 0.409 | 0.511 | 0.623 |
| distance_folder_content | bgl_split_10 | 0.578 | 0.720 | 1.82 | 3.77 |
| distance_folder_content (threshold=False) | hadoop_renamed | 0.482 | 0.484 | 0.558 | 1.07 |
| distance_folder_content (threshold=False) | hdfs_balanced_5k | 0.389 | 0.408 | 0.512 | 0.624 |
| distance_folder_content (threshold=False) | bgl_split_10 | 0.577 | 0.721 | 1.81 | 3.77 |
| distance_file_content | hadoop_renamed | 0.488 | 0.498 | 0.639 | 2.32 |
| distance_file_content | hdfs_balanced_5k | 0.390 | 0.410 | 0.516 | 0.633 |
| distance_file_content | bgl_split_10 | 0.616 | 0.776 | 2.13 | 4.11 |
| distance_file_content (threshold=False) | hadoop_renamed | 0.492 | 0.505 | 0.640 | 2.41 |
| distance_file_content (threshold=False) | hdfs_balanced_5k | 0.390 | 0.411 | 0.517 | 0.634 |
| distance_file_content (threshold=False) | bgl_split_10 | 0.629 | 0.816 | 2.17 | 4.26 |
| log_line_clustering | hadoop_renamed | 0.486 | 0.496 | 0.579 | 1.79 |
| log_line_clustering | hdfs_balanced_5k | 0.391 | 0.412 | 0.522 | 0.643 |
| log_line_clustering | bgl_split_10 | 0.641 | 0.826 | 2.00 | 4.05 |

## Table B3 -- Anomaly tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename | hadoop_renamed | 0.478 | 0.483 | 0.572 | 1.73 |
| anomaly_folder_filename | hdfs_balanced_5k | 0.393 | 0.415 | 0.531 | 0.663 |
| anomaly_folder_filename | bgl_split_10 | 0.619 | 0.801 | 2.18 | 4.24 |
| anomaly_folder_filename (threshold=False) | hadoop_renamed | 0.475 | 0.483 | 0.576 | 1.54 |
| anomaly_folder_filename (threshold=False) | hdfs_balanced_5k | 0.393 | 0.415 | 0.530 | 0.659 |
| anomaly_folder_filename (threshold=False) | bgl_split_10 | 0.635 | 0.827 | 2.39 | 4.68 |
| anomaly_folder_content | hadoop_renamed | 0.551 | 0.605 | 1.63 | 2.52 |
| anomaly_folder_content | hdfs_balanced_5k | 0.421 | 0.459 | 0.739 | 1.23 |
| anomaly_folder_content | bgl_split_10 | 0.802 | 1.24 | 4.24 | 14.59 |
| anomaly_folder_content (threshold=False) | hadoop_renamed | 0.574 | 0.677 | 1.38 | 3.02 |
| anomaly_folder_content (threshold=False) | hdfs_balanced_5k | 0.429 | 0.481 | 0.776 | 1.32 |
| anomaly_folder_content (threshold=False) | bgl_split_10 | 0.806 | 1.19 | 3.51 | OOM |
| anomaly_file_content | hadoop_renamed | err | 0.595 | 1.19 | 2.67 |
| anomaly_file_content | hdfs_balanced_5k | 0.428 | 0.478 | 0.757 | 0.937 |
| anomaly_file_content | bgl_split_10 | 0.721 | 0.965 | 2.84 | 5.65 |
| anomaly_file_content (threshold=False) | hadoop_renamed | err | 0.581 | 0.993 | 2.16 |
| anomaly_file_content (threshold=False) | hdfs_balanced_5k | 0.428 | 0.477 | 0.757 | 0.849 |
| anomaly_file_content (threshold=False) | bgl_split_10 | 0.712 | 0.961 | 2.82 | 3.12 |
| anomaly_line_content | hadoop_renamed | 0.513 | 0.565 | 0.894 | 1.58 |
| anomaly_line_content | hdfs_balanced_5k | 0.428 | 0.477 | 0.752 | 0.783 |
| anomaly_line_content | bgl_split_10 | 0.701 | 0.960 | 2.82 | 3.10 |
| anomaly_line_content (threshold=False) | hadoop_renamed | 0.522 | 0.561 | 0.893 | 1.59 |
| anomaly_line_content (threshold=False) | hdfs_balanced_5k | 0.421 | 0.477 | 0.733 | 0.775 |
| anomaly_line_content (threshold=False) | bgl_split_10 | 0.698 | 0.951 | 2.82 | 3.13 |

## Table B4 -- Plot tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| plot_folder_filename | hadoop_renamed | 0.555 | 0.584 | 0.908 | 1.83 |
| plot_folder_filename | hdfs_balanced_5k | 0.464 | 0.495 | 0.747 | 0.795 |
| plot_folder_filename | bgl_split_10 | 0.729 | 0.984 | 2.85 | 4.95 |
| plot_folder_content (scatter) | hadoop_renamed | 0.560 | 0.630 | 1.06 | 1.93 |
| plot_folder_content (scatter) | hdfs_balanced_5k | 0.463 | 0.513 | 0.727 | 0.907 |
| plot_folder_content (scatter) | bgl_split_10 | 0.793 | 1.12 | 3.46 | 6.99 |
| plot_folder_content (scatter+umap) | hadoop_renamed | err | 0.831 | 1.08 | 1.87 |
| plot_folder_content (scatter+umap) | hdfs_balanced_5k | 0.740 | 0.737 | 0.971 | 1.33 |
| plot_folder_content (scatter+umap) | bgl_split_10 | 1.05 | 1.35 | 3.52 | 7.22 |
| plot_file_content (scatter) | hadoop_renamed | 0.835 | 0.847 | 1.05 | 1.67 |
| plot_file_content (scatter) | hdfs_balanced_5k | 0.704 | 0.720 | 0.892 | 1.19 |
| plot_file_content (scatter) | bgl_split_10 | 1.03 | 1.23 | 2.51 | 5.35 |
| plot_file_content (scatter+umap) | hadoop_renamed | err | 0.867 | 0.936 | 1.57 |
| plot_file_content (scatter+umap) | hdfs_balanced_5k | 0.704 | 0.720 | 0.892 | 1.19 |
| plot_file_content (scatter+umap) | bgl_split_10 | 1.03 | 1.23 | 2.49 | 5.37 |

## Table B5 -- Sequence tools

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| sequence_line_event_prediction | hadoop_renamed | 0.551 | 0.561 | 0.898 | 2.03 |
| sequence_line_event_prediction | hdfs_balanced_5k | 0.419 | 0.477 | 0.733 | 0.765 |
| sequence_line_event_prediction | bgl_split_10 | 0.697 | 0.951 | 2.82 | 4.70 |

# Detailed breakdowns (per detector / per measure)

The tables above run every anomaly tool with all four detectors, `distance_folder_content`/`distance_file_content` with their three default measures (cosine, jaccard, containment; compression is opt-in), and `sequence_line_event_prediction` with both order detectors, at once; `log_line_clustering` defaults to its coarse pair (Prefix + Exact) in one pass. Part C/D below break the same figure down per detector / per measure run in isolation (`detectors=["<name>"]` / `measures=["<name>"]`), so the cost of narrowing either is visible on its own rather than folded into the combined call -- `log_line_clustering` included, run once per bucket measure (Exact, Prefix, Minhash) so they can be compared directly. `distance_folder_filename` (jaccard/overlap distance over file names only) is not broken down further -- it computes one measure, not a default pair. The anomaly rows pass `threshold=False`, so they show one detector's own cost without the extra fits of the clean range; the distance rows keep the default clean range.

# Part C -- cold (first call)

## Table C1 -- Anomaly tools detailed

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename (KMeans) | hadoop_renamed | 0.349 | 0.362 | 0.442 | 1.33 |
| anomaly_folder_filename (KMeans) | hdfs_balanced_5k | 0.343 | 0.348 | 0.382 | 0.424 |
| anomaly_folder_filename (KMeans) | bgl_split_10 | 0.512 | 0.687 | 1.72 | 3.21 |
| anomaly_folder_filename (IsolationForest) | hadoop_renamed | 0.424 | 0.458 | 0.548 | 1.46 |
| anomaly_folder_filename (IsolationForest) | hdfs_balanced_5k | 0.364 | 0.383 | 0.508 | 0.619 |
| anomaly_folder_filename (IsolationForest) | bgl_split_10 | 0.600 | 0.795 | 2.08 | 7.86 |
| anomaly_folder_filename (RarityDetector) | hadoop_renamed | 0.443 | 0.467 | 0.516 | 1.58 |
| anomaly_folder_filename (RarityDetector) | hdfs_balanced_5k | 0.365 | 0.393 | 0.523 | 0.734 |
| anomaly_folder_filename (RarityDetector) | bgl_split_10 | 0.581 | 0.765 | 2.32 | 8.04 |
| anomaly_folder_filename (OOVDetector) | hadoop_renamed | 0.475 | 0.504 | 0.612 | 1.65 |
| anomaly_folder_filename (OOVDetector) | hdfs_balanced_5k | 0.376 | 0.414 | 0.614 | 0.863 |
| anomaly_folder_filename (OOVDetector) | bgl_split_10 | 0.680 | 0.922 | 3.12 | 12.71 |
| anomaly_folder_content (KMeans) | hadoop_renamed | 0.380 | 0.413 | 0.656 | 1.58 |
| anomaly_folder_content (KMeans) | hdfs_balanced_5k | 0.354 | 0.369 | 0.485 | 0.616 |
| anomaly_folder_content (KMeans) | bgl_split_10 | 0.606 | 0.884 | 2.51 | 5.87 |
| anomaly_folder_content (IsolationForest) | hadoop_renamed | 0.427 | 0.482 | 0.672 | 1.65 |
| anomaly_folder_content (IsolationForest) | hdfs_balanced_5k | 0.361 | 0.377 | 0.508 | 0.681 |
| anomaly_folder_content (IsolationForest) | bgl_split_10 | 0.608 | 0.929 | 3.13 | 7.19 |
| anomaly_folder_content (RarityDetector) | hadoop_renamed | 0.464 | 0.502 | 0.683 | 1.78 |
| anomaly_folder_content (RarityDetector) | hdfs_balanced_5k | 0.367 | 0.402 | 0.577 | 0.854 |
| anomaly_folder_content (RarityDetector) | bgl_split_10 | 0.637 | 0.969 | 3.26 | 10.77 |
| anomaly_folder_content (OOVDetector) | hadoop_renamed | 0.488 | 0.523 | 0.726 | 1.81 |
| anomaly_folder_content (OOVDetector) | hdfs_balanced_5k | 0.377 | 0.413 | 0.616 | 0.813 |
| anomaly_folder_content (OOVDetector) | bgl_split_10 | 0.682 | 0.992 | 3.25 | 11.07 |
| anomaly_file_content (KMeans) | hadoop_renamed | err | 0.472 | 0.915 | 2.29 |
| anomaly_file_content (KMeans) | hdfs_balanced_5k | 0.364 | 0.383 | 0.514 | 0.684 |
| anomaly_file_content (KMeans) | bgl_split_10 | 0.647 | 0.822 | 2.20 | 7.76 |
| anomaly_file_content (IsolationForest) | hadoop_renamed | 0.469 | 0.541 | 0.945 | 2.38 |
| anomaly_file_content (IsolationForest) | hdfs_balanced_5k | 0.365 | 0.393 | 0.563 | 0.733 |
| anomaly_file_content (IsolationForest) | bgl_split_10 | 0.613 | 0.813 | 2.27 | 7.94 |
| anomaly_file_content (RarityDetector) | hadoop_renamed | 0.497 | 0.569 | 0.986 | 2.49 |
| anomaly_file_content (RarityDetector) | hdfs_balanced_5k | 0.375 | 0.413 | 0.628 | 0.926 |
| anomaly_file_content (RarityDetector) | bgl_split_10 | 0.699 | 0.963 | 3.12 | 12.63 |
| anomaly_file_content (OOVDetector) | hadoop_renamed | 0.526 | 0.582 | 1.00 | 2.46 |
| anomaly_file_content (OOVDetector) | hdfs_balanced_5k | 0.379 | 0.412 | 0.691 | 0.800 |
| anomaly_file_content (OOVDetector) | bgl_split_10 | 0.696 | 0.978 | 3.23 | 12.75 |
| anomaly_line_content (KMeans) | hadoop_renamed | 0.446 | 0.463 | 0.666 | 1.36 |
| anomaly_line_content (KMeans) | hdfs_balanced_5k | 0.364 | 0.382 | 0.507 | 0.662 |
| anomaly_line_content (KMeans) | bgl_split_10 | 0.602 | 0.792 | 2.03 | 7.75 |
| anomaly_line_content (IsolationForest) | hadoop_renamed | 0.447 | 0.470 | 0.543 | 1.17 |
| anomaly_line_content (IsolationForest) | hdfs_balanced_5k | 0.365 | 0.393 | 0.555 | 0.734 |
| anomaly_line_content (IsolationForest) | bgl_split_10 | 0.603 | 0.766 | 2.26 | 7.94 |
| anomaly_line_content (RarityDetector) | hadoop_renamed | 0.497 | 0.518 | 0.683 | 1.40 |
| anomaly_line_content (RarityDetector) | hdfs_balanced_5k | 0.376 | 0.413 | 0.629 | 0.891 |
| anomaly_line_content (RarityDetector) | bgl_split_10 | 0.681 | 0.930 | 3.06 | 12.62 |
| anomaly_line_content (OOVDetector) | hadoop_renamed | 0.511 | 0.525 | 0.658 | 1.58 |
| anomaly_line_content (OOVDetector) | hdfs_balanced_5k | 0.379 | 0.412 | 0.682 | 0.794 |
| anomaly_line_content (OOVDetector) | bgl_split_10 | 0.685 | 0.977 | 3.18 | 12.76 |

## Table C2 -- Distance tools detailed

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| distance_folder_content (cosine) | hadoop_renamed | 0.492 | 0.506 | 0.591 | 1.34 |
| distance_folder_content (cosine) | hdfs_balanced_5k | 0.379 | 0.405 | 0.675 | 0.779 |
| distance_folder_content (cosine) | bgl_split_10 | 0.679 | 0.959 | 3.17 | 12.85 |
| distance_folder_content (jaccard) | hadoop_renamed | 0.472 | 0.484 | 0.556 | 1.46 |
| distance_folder_content (jaccard) | hdfs_balanced_5k | 0.362 | 0.368 | 0.568 | 0.724 |
| distance_folder_content (jaccard) | bgl_split_10 | 0.620 | 0.835 | 2.65 | 8.81 |
| distance_folder_content (compression) | hadoop_renamed | 0.472 | 0.480 | 0.557 | 1.42 |
| distance_folder_content (compression) | hdfs_balanced_5k | 0.356 | 0.368 | 0.568 | 0.724 |
| distance_folder_content (compression) | bgl_split_10 | 0.596 | 0.828 | 2.64 | 8.85 |
| distance_folder_content (containment) | hadoop_renamed | 0.454 | 0.467 | 0.539 | 1.34 |
| distance_folder_content (containment) | hdfs_balanced_5k | 0.357 | 0.368 | 0.569 | 0.725 |
| distance_folder_content (containment) | bgl_split_10 | 0.614 | 0.807 | 2.63 | 8.47 |
| distance_file_content (cosine) | hadoop_renamed | 0.488 | 0.497 | 0.614 | 2.79 |
| distance_file_content (cosine) | hdfs_balanced_5k | 0.361 | 0.367 | 0.564 | 0.719 |
| distance_file_content (cosine) | bgl_split_10 | 0.584 | 0.793 | 2.57 | 8.73 |
| distance_file_content (jaccard) | hadoop_renamed | 0.475 | 0.491 | 0.611 | 2.76 |
| distance_file_content (jaccard) | hdfs_balanced_5k | 0.355 | 0.367 | 0.564 | 0.720 |
| distance_file_content (jaccard) | bgl_split_10 | 0.578 | 0.770 | 2.49 | 8.58 |
| distance_file_content (compression) | hadoop_renamed | 0.457 | 0.481 | 0.613 | 2.78 |
| distance_file_content (compression) | hdfs_balanced_5k | 0.355 | 0.366 | 0.566 | 0.720 |
| distance_file_content (compression) | bgl_split_10 | 0.577 | 0.768 | 2.49 | 8.53 |
| distance_file_content (containment) | hadoop_renamed | 0.458 | 0.480 | 0.613 | 2.78 |
| distance_file_content (containment) | hdfs_balanced_5k | 0.355 | 0.367 | 0.566 | 0.719 |
| distance_file_content (containment) | bgl_split_10 | 0.573 | 0.754 | 2.43 | 8.39 |
| log_line_clustering (Exact) | hadoop_renamed | 0.463 | 0.468 | 0.561 | 1.78 |
| log_line_clustering (Exact) | hdfs_balanced_5k | 0.357 | 0.369 | 0.571 | 0.726 |
| log_line_clustering (Exact) | bgl_split_10 | 0.612 | 0.789 | 2.65 | 8.48 |
| log_line_clustering (Prefix) | hadoop_renamed | 0.458 | 0.463 | 0.551 | 1.63 |
| log_line_clustering (Prefix) | hdfs_balanced_5k | 0.358 | 0.371 | 0.576 | 0.737 |
| log_line_clustering (Prefix) | bgl_split_10 | 0.610 | 0.772 | 2.30 | 7.91 |
| log_line_clustering (Minhash) | hadoop_renamed | 0.467 | 0.489 | 0.631 | 2.82 |
| log_line_clustering (Minhash) | hdfs_balanced_5k | 0.358 | 0.371 | 0.576 | 0.741 |
| log_line_clustering (Minhash) | bgl_split_10 | 0.594 | 0.753 | 2.24 | 7.88 |

## Table C3 -- Sequence tools detailed

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| sequence_line_event_prediction (NEP) | hadoop_renamed | 0.488 | 0.524 | 0.660 | 2.81 |
| sequence_line_event_prediction (NEP) | hdfs_balanced_5k | 0.359 | 0.372 | 0.578 | 0.745 |
| sequence_line_event_prediction (NEP) | bgl_split_10 | 0.591 | 0.732 | 2.24 | 7.83 |
| sequence_line_event_prediction (LAP) | hadoop_renamed | 0.490 | 0.496 | 0.584 | 1.48 |
| sequence_line_event_prediction (LAP) | hdfs_balanced_5k | 0.358 | 0.372 | 0.579 | 0.749 |
| sequence_line_event_prediction (LAP) | bgl_split_10 | 0.588 | 0.724 | 2.21 | 7.78 |

# Part D -- warm (repeated call)

## Table D1 -- Anomaly tools detailed

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| anomaly_folder_filename (KMeans) | hadoop_renamed | 0.352 | 0.366 | 0.451 | 1.32 |
| anomaly_folder_filename (KMeans) | hdfs_balanced_5k | 0.345 | 0.351 | 0.391 | 0.440 |
| anomaly_folder_filename (KMeans) | bgl_split_10 | 0.525 | 0.699 | 1.85 | 3.51 |
| anomaly_folder_filename (IsolationForest) | hadoop_renamed | 0.420 | 0.450 | 0.524 | 1.44 |
| anomaly_folder_filename (IsolationForest) | hdfs_balanced_5k | 0.363 | 0.383 | 0.481 | 0.541 |
| anomaly_folder_filename (IsolationForest) | bgl_split_10 | 0.580 | 0.748 | 2.27 | 7.03 |
| anomaly_folder_filename (RarityDetector) | hadoop_renamed | 0.444 | 0.468 | 0.523 | 1.59 |
| anomaly_folder_filename (RarityDetector) | hdfs_balanced_5k | 0.365 | 0.394 | 0.523 | 0.724 |
| anomaly_folder_filename (RarityDetector) | bgl_split_10 | 0.579 | 0.775 | 2.45 | 8.36 |
| anomaly_folder_filename (OOVDetector) | hadoop_renamed | 0.475 | 0.502 | 0.604 | 1.62 |
| anomaly_folder_filename (OOVDetector) | hdfs_balanced_5k | 0.376 | 0.412 | 0.613 | 0.834 |
| anomaly_folder_filename (OOVDetector) | bgl_split_10 | 0.668 | 0.931 | 3.10 | 12.12 |
| anomaly_folder_content (KMeans) | hadoop_renamed | 0.416 | 0.472 | 0.863 | 2.05 |
| anomaly_folder_content (KMeans) | hdfs_balanced_5k | 0.364 | 0.382 | 0.528 | 0.737 |
| anomaly_folder_content (KMeans) | bgl_split_10 | 0.650 | 0.917 | 3.02 | 9.20 |
| anomaly_folder_content (IsolationForest) | hadoop_renamed | 0.464 | 0.535 | 0.906 | 2.09 |
| anomaly_folder_content (IsolationForest) | hdfs_balanced_5k | 0.365 | 0.394 | 0.589 | 0.794 |
| anomaly_folder_content (IsolationForest) | bgl_split_10 | 0.657 | 1.00 | 3.30 | 9.25 |
| anomaly_folder_content (RarityDetector) | hadoop_renamed | 0.497 | 0.561 | 0.921 | 2.26 |
| anomaly_folder_content (RarityDetector) | hdfs_balanced_5k | 0.375 | 0.413 | 0.628 | 0.950 |
| anomaly_folder_content (RarityDetector) | bgl_split_10 | 0.699 | 1.04 | 3.63 | 14.02 |
| anomaly_folder_content (OOVDetector) | hadoop_renamed | 0.523 | 0.574 | 0.951 | 2.28 |
| anomaly_folder_content (OOVDetector) | hdfs_balanced_5k | 0.379 | 0.414 | 0.691 | 0.831 |
| anomaly_folder_content (OOVDetector) | bgl_split_10 | 0.694 | 1.08 | 3.80 | 14.07 |
| anomaly_file_content (KMeans) | hadoop_renamed | err | 0.433 | 0.848 | 1.98 |
| anomaly_file_content (KMeans) | hdfs_balanced_5k | 0.364 | 0.383 | 0.514 | 0.663 |
| anomaly_file_content (KMeans) | bgl_split_10 | 0.647 | 0.817 | 2.22 | 7.77 |
| anomaly_file_content (IsolationForest) | hadoop_renamed | 0.429 | 0.484 | 0.627 | 1.75 |
| anomaly_file_content (IsolationForest) | hdfs_balanced_5k | 0.365 | 0.393 | 0.563 | 0.733 |
| anomaly_file_content (IsolationForest) | bgl_split_10 | 0.613 | 0.766 | 2.28 | 7.96 |
| anomaly_file_content (RarityDetector) | hadoop_renamed | 0.484 | 0.556 | 0.894 | 2.21 |
| anomaly_file_content (RarityDetector) | hdfs_balanced_5k | 0.376 | 0.413 | 0.629 | 0.926 |
| anomaly_file_content (RarityDetector) | bgl_split_10 | 0.695 | 0.964 | 3.12 | 12.63 |
| anomaly_file_content (OOVDetector) | hadoop_renamed | 0.508 | 0.549 | 0.895 | 2.19 |
| anomaly_file_content (OOVDetector) | hdfs_balanced_5k | 0.379 | 0.412 | 0.691 | 0.797 |
| anomaly_file_content (OOVDetector) | bgl_split_10 | 0.684 | 0.978 | 3.24 | 12.76 |
| anomaly_line_content (KMeans) | hadoop_renamed | 0.447 | 0.469 | 0.653 | 1.27 |
| anomaly_line_content (KMeans) | hdfs_balanced_5k | 0.364 | 0.382 | 0.508 | 0.619 |
| anomaly_line_content (KMeans) | bgl_split_10 | 0.602 | 0.788 | 2.03 | 7.75 |
| anomaly_line_content (IsolationForest) | hadoop_renamed | 0.459 | 0.470 | 0.524 | 1.19 |
| anomaly_line_content (IsolationForest) | hdfs_balanced_5k | 0.365 | 0.393 | 0.555 | 0.734 |
| anomaly_line_content (IsolationForest) | bgl_split_10 | 0.603 | 0.767 | 2.26 | 7.93 |
| anomaly_line_content (RarityDetector) | hadoop_renamed | 0.511 | 0.517 | 0.674 | 1.39 |
| anomaly_line_content (RarityDetector) | hdfs_balanced_5k | 0.376 | 0.413 | 0.629 | 0.891 |
| anomaly_line_content (RarityDetector) | bgl_split_10 | 0.681 | 0.914 | 3.06 | 12.68 |
| anomaly_line_content (OOVDetector) | hadoop_renamed | 0.506 | 0.526 | 0.632 | 1.56 |
| anomaly_line_content (OOVDetector) | hdfs_balanced_5k | 0.379 | 0.410 | 0.682 | 0.778 |
| anomaly_line_content (OOVDetector) | bgl_split_10 | 0.685 | 0.975 | 3.18 | 12.75 |

## Table D2 -- Distance tools detailed

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| distance_folder_content (cosine) | hadoop_renamed | 0.490 | 0.500 | 0.543 | 1.35 |
| distance_folder_content (cosine) | hdfs_balanced_5k | 0.361 | 0.366 | 0.561 | 0.714 |
| distance_folder_content (cosine) | bgl_split_10 | 0.638 | 0.871 | 2.48 | 12.74 |
| distance_folder_content (jaccard) | hadoop_renamed | 0.472 | 0.484 | 0.541 | 1.34 |
| distance_folder_content (jaccard) | hdfs_balanced_5k | 0.359 | 0.366 | 0.561 | 0.714 |
| distance_folder_content (jaccard) | bgl_split_10 | 0.582 | 0.757 | 2.41 | 8.34 |
| distance_folder_content (compression) | hadoop_renamed | 0.461 | 0.467 | 0.536 | 1.27 |
| distance_folder_content (compression) | hdfs_balanced_5k | 0.355 | 0.366 | 0.562 | 0.714 |
| distance_folder_content (compression) | bgl_split_10 | 0.564 | 0.750 | 2.39 | 8.25 |
| distance_folder_content (containment) | hadoop_renamed | 0.454 | 0.469 | 0.540 | 1.34 |
| distance_folder_content (containment) | hdfs_balanced_5k | 0.355 | 0.366 | 0.563 | 0.714 |
| distance_folder_content (containment) | bgl_split_10 | 0.581 | 0.738 | 2.33 | 8.13 |
| distance_file_content (cosine) | hadoop_renamed | 0.486 | 0.498 | 0.620 | 2.71 |
| distance_file_content (cosine) | hdfs_balanced_5k | 0.362 | 0.368 | 0.567 | 0.724 |
| distance_file_content (cosine) | bgl_split_10 | 0.619 | 0.835 | 2.66 | 8.95 |
| distance_file_content (jaccard) | hadoop_renamed | 0.475 | 0.490 | 0.614 | 2.74 |
| distance_file_content (jaccard) | hdfs_balanced_5k | 0.356 | 0.368 | 0.568 | 0.724 |
| distance_file_content (jaccard) | bgl_split_10 | 0.596 | 0.828 | 2.62 | 8.86 |
| distance_file_content (compression) | hadoop_renamed | 0.460 | 0.480 | 0.612 | 2.78 |
| distance_file_content (compression) | hdfs_balanced_5k | 0.356 | 0.368 | 0.569 | 0.725 |
| distance_file_content (compression) | bgl_split_10 | 0.609 | 0.809 | 2.65 | 8.60 |
| distance_file_content (containment) | hadoop_renamed | 0.467 | 0.481 | 0.617 | 2.75 |
| distance_file_content (containment) | hdfs_balanced_5k | 0.356 | 0.368 | 0.569 | 0.723 |
| distance_file_content (containment) | bgl_split_10 | 0.612 | 0.790 | 2.63 | 8.60 |
| log_line_clustering (Exact) | hadoop_renamed | 0.463 | 0.467 | 0.561 | 1.71 |
| log_line_clustering (Exact) | hdfs_balanced_5k | 0.358 | 0.370 | 0.575 | 0.734 |
| log_line_clustering (Exact) | bgl_split_10 | 0.612 | 0.778 | 2.54 | 8.05 |
| log_line_clustering (Prefix) | hadoop_renamed | 0.458 | 0.463 | 0.550 | 1.63 |
| log_line_clustering (Prefix) | hdfs_balanced_5k | 0.358 | 0.371 | 0.576 | 0.739 |
| log_line_clustering (Prefix) | bgl_split_10 | 0.611 | 0.772 | 2.25 | 7.88 |
| log_line_clustering (Minhash) | hadoop_renamed | 0.479 | 0.517 | 0.668 | 2.82 |
| log_line_clustering (Minhash) | hdfs_balanced_5k | 0.359 | 0.372 | 0.578 | 0.744 |
| log_line_clustering (Minhash) | bgl_split_10 | 0.591 | 0.750 | 2.25 | 7.88 |

## Table D3 -- Sequence tools detailed

| tool | log root | 5% | 10% | 50% | 100% |
|---|---|---|---|---|---|
| sequence_line_event_prediction (NEP) | hadoop_renamed | 0.508 | 0.508 | 0.646 | 1.98 |
| sequence_line_event_prediction (NEP) | hdfs_balanced_5k | 0.359 | 0.372 | 0.579 | 0.748 |
| sequence_line_event_prediction (NEP) | bgl_split_10 | 0.591 | 0.735 | 2.23 | 7.83 |
| sequence_line_event_prediction (LAP) | hadoop_renamed | 0.488 | 0.492 | 0.556 | 1.48 |
| sequence_line_event_prediction (LAP) | hdfs_balanced_5k | 0.359 | 0.372 | 0.579 | 0.749 |
| sequence_line_event_prediction (LAP) | bgl_split_10 | 0.573 | 0.732 | 2.21 | 7.78 |
