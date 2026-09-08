# What each MCP tool costs

The tools in `loglead/mcp/server.py` take anywhere from 4 milliseconds to 21 minutes. The tool
schema does not say which is which. This file gives the measured times, so a caller can pick
arguments before spending the time.

Sections: the test data, how to read a row, how to predict a call you have not made, how to choose
arguments, and the full table of measurements.

## The test data

Two log roots were measured. `tests/mcp/make_test_data.py` builds them.

| log root | log folders | log files | log lines | files per folder | lines per folder |
|---|---|---|---|---|---|
| **hadoop** (`hadoop_renamed`) | 55 | 978 | 180,897 | ~18 | ~3,300 |
| **hdfs** (`hdfs_balanced_5k`) | 5,000 | 5,000 | 90,862 | 1 | ~18 |

They have opposite shapes. Hadoop has few log folders and each one is large. HDFS has many log
folders and each one is tiny.

Shape drives cost more than total size does. Costs charged per log folder are worst on HDFS. Costs
charged per file or per line are worst on Hadoop.

All numbers come from one machine, with a warm parquet cache and `--repeat 3`. The seconds depend
on the hardware. The ratios between rows do not.

## How to read a row

Each row in the table is one call at one size.

- **first** is the cold call. **warm** is the median of three repeats.
- A gap between them is a column the session computed once and kept. It is paid once per session,
  not once per call. `Parse-Drain` takes 3.59s the first time and 0.18s after that.
- **class** is the cost band of the warm number: `instant` under 0.1s, `fast` under 1s, `slow`
  under 10s, `very slow` above 10s.
- **per unit** is the warm time divided by the size of that call. Use it to compare rows. Do not
  use it to predict a different size. Most tools do fixed work before the first unit, so the
  per-unit number changes with the size. Use the formulas below instead.

## How to predict a call

Every tool with a size argument was measured at two or three sizes. The measurements were fitted to
a straight line:

```
seconds = fixed + marginal * n
```

`fixed` is the work a call does no matter how large `n` is. This is loading, aggregating the text
of every log folder, or building a term matrix. `marginal` is what one more comparison folder,
target folder or file adds.

The split matters because it tells you whether a narrower selector will save you anything. If
`fixed` is large, it will not.

To predict a call, put your own `n` into the formula. `anomaly_folder_content` on Hadoop with 20
target folders is `0.09 + 0.498 * 20`, or about 10s. With all 55 folders it is about 28s, which is
what the table measured.

| tool | log root | fixed | marginal | per what |
|---|---|---|---|---|
| `distance_folder_filename` | hadoop | 0.04s | **0.015s** | comparison folder |
| `distance_folder_content` | hadoop | 0.09s | **0.285s** | comparison folder |
| `distance_file_content` | hadoop | **10.24s** | 0.551s | target file |
| `distance_line_content` | hadoop | 0.01s | 0.013s | diff |
| `anomaly_folder_filename` | hadoop | 0.02s | **0.163s** | target folder |
| `anomaly_folder_content` | hadoop | 0.09s | **0.498s** | target folder |
| `anomaly_file_content` | hadoop | 0.13s | 0.391s | target file |
| `plot_folder_content` | hadoop | 0.09s | 0.007s | comparison folder |
| `anomaly_folder_content` | hdfs | 0.04s | **0.247s** | target folder |
| `plot_folder_content` | hdfs | 0.06s | **0.000s** | comparison folder |

A rate from one log root does not apply to the other. `distance_folder_content` costs 0.285s per
comparison folder on Hadoop. That rate predicts 24 minutes for all 4,999 HDFS folders. The measured
time is 35s. Use the formula for the log root you have. If there is none, use a measured "ALL" row
from the table.

## Choosing arguments

These are the decisions that change what a call costs, in the order a caller makes them. Each step
says what to do, and the numbers behind it.

**1. Open the log root once, and name the format if you know it.** `open_log_root` is the single
most expensive call here: about 20s for Hadoop's 978 files, about 127s for HDFS's 5,000. Nearly all
of that is the `auto` format probe, which runs per file. Passing a `format` skips the probe.
Re-opening the same log root costs about 0.1s, because the session is cached to parquet. So this is
paid once and everything after it is cheap. That is why sessions exist.

**2. Name your targets. Do not leave `target_folder` at `"ALL"`.** The `anomaly_*` tools refit four
detectors for every single target log folder. That is 0.498s per folder on Hadoop and 0.247s on
HDFS. The default `"ALL"` therefore costs about 28s on Hadoop's 55 folders and about 21 minutes on
HDFS's 5,000. Nothing in the call signature hints at this. It is the most expensive mistake
available in the API.

**3. Multiply the rate by your real `n` before calling anything cheap.** A small marginal cost is
still large when `n` is large. `distance_folder_filename` has the smallest marginal cost of any
measure here, at 0.015s per comparison folder. It also produced one of the slowest calls measured,
at 41.7s, because `comparison_folders="ALL"` on HDFS means 4,999 pairs.

**4. Narrow the selector only where narrowing helps.** Whether a shorter `target_files` or
`comparison_folders` list saves time depends on how much of the call is fixed cost. Compare the
`fixed` and `marginal` columns in the table above:

| tool | fixed | marginal | does narrowing help? |
|---|---|---|---|
| `distance_file_content` | 10.24s | 0.551s per file | No. The aggregation runs before any file is read. |
| `anomaly_file_content` | 0.13s | 0.391s per file | Yes. Almost the entire cost is per file. |
| `distance_folder_content` | 0.09s | 0.285s per folder | Yes. |
| `anomaly_folder_content` | 0.09s | 0.498s per folder | Yes. This is step 2. |
| `plot_folder_content` (scatter) | 0.09s | 0.007s per folder | No, and it does not need to. See step 6. |

**5. Pick the cheapest `content_format` that answers the question.** The format changes cost by
60-80x on identical data. At 10 comparison folders on Hadoop: `Parse-Tip` 0.18s, `Parse-Drain`
0.18s after 3.59s to parse once, `Words` 2.78s, `Sklearn` 2.83s, `3grams` 11.35s. One template id
per line is a much smaller document than every word of the line. So scan with a parser format
first, then re-check only the survivors with `Words`.

**6. Ask for the UMAP only when you need the layout.** The default scatter of `plot_folder_content`
is cheap and stays cheap: 0.26s for all 4,999 HDFS folders, about the same as 50 of them. Adding
`plots=["umap"]` makes the same call 8.5s, roughly 32x more. Two parts of that cannot be tuned
away. The first UMAP in a process pays about 13s more for numba to compile. And `random_seed` makes
umap-learn single-threaded, which is 8.5s seeded against 3.2s unseeded. That is what a reproducible
layout costs.

**7. Follow up with `query_result`, not another analysis.** Filtering or re-sorting a result you
already have takes under 0.01s. Recomputing it takes 27-28s.

## Measurements

Every row, in the order the benchmark runs them.

This table is generated. The prose above it is written by hand. Running
`uv run tests/mcp/benchmark.py --markdown tests/mcp/COST.md` overwrites the whole file with the
table alone, so put the prose back afterwards and update its numbers to match.

| tool | log root | call | first (s) | warm (s) | class | per unit |
|---|---|---|---|---|---|---|
| `open_log_root` | hadoop | read + mask + parse tip (978 files) | 19.943 | 0.06 | instant | 0.0001 s/file |
| `list_log_roots` | hadoop | 1 open session | 0.024 | 0.022 | instant |  |
| `describe_log_root` | hadoop | 55 log folders | 0.038 | 0.038 | instant |  |
| `describe_log_root` | hadoop | include_files=True | 0.051 | 0.046 | instant |  |
| `read_log_lines` | hadoop | 100 lines | 0.004 | 0.004 | instant |  |
| `search_log_lines` | hadoop | regex over 181k lines | 0.015 | 0.013 | instant |  |
| `search_log_lines` | hadoop | literal over 181k lines | 0.025 | 0.022 | instant |  |
| `distance_folder_filename` | hadoop | comparison_folders=5 | 0.096 | 0.096 | instant | 0.0193 s/folder |
| `distance_folder_filename` | hadoop | comparison_folders=27 | 0.379 | 0.47 | fast | 0.0174 s/folder |
| `distance_folder_filename` | hadoop | comparison_folders=54 | 0.654 | 0.829 | fast | 0.0154 s/folder |
| `distance_folder_content` | hadoop | comparison_folders=5, Words | 1.871 | 1.493 | slow | 0.2986 s/folder |
| `distance_folder_content` | hadoop | comparison_folders=27, Words | 7.337 | 7.816 | slow | 0.2895 s/folder |
| `distance_folder_content` | hadoop | comparison_folders=54, Words | 15.714 | 15.467 | very slow | 0.2864 s/folder |
| `distance_file_content` | hadoop | target_files=1, comparison=ALL | 11.004 | 10.831 | very slow | 10.8309 s/file |
| `distance_file_content` | hadoop | target_files=3, comparison=ALL | 13.525 | 11.806 | very slow | 3.9353 s/file |
| `distance_file_content` | hadoop | target_files=5, comparison=ALL | 12.771 | 13.034 | very slow | 2.6067 s/file |
| `distance_line_content` | hadoop | 1 file x 1 comparison folders | 0.022 | 0.026 | instant | 0.0256 s/diff |
| `distance_line_content` | hadoop | 1 file x 5 comparison folders | 0.078 | 0.08 | instant | 0.016 s/diff |
| `anomaly_folder_filename` | hadoop | target_folder=1, comparison=ALL | 0.218 | 0.17 | fast | 0.1698 s/target |
| `anomaly_folder_filename` | hadoop | target_folder=5, comparison=ALL | 0.838 | 0.857 | fast | 0.1714 s/target |
| `anomaly_folder_filename` | hadoop | target_folder=55, comparison=ALL | 9.162 | 8.987 | slow | 0.1634 s/target |
| `anomaly_folder_content` | hadoop | target_folder=1, comparison=ALL | 0.607 | 0.533 | fast | 0.5332 s/target |
| `anomaly_folder_content` | hadoop | target_folder=5, comparison=ALL | 2.59 | 2.646 | slow | 0.5291 s/target |
| `anomaly_folder_content` | hadoop | target_folder=55, comparison=ALL | 28.56 | 27.481 | very slow | 0.4997 s/target |
| `anomaly_file_content` | hadoop | target_files=1, comparison=ALL | 0.537 | 0.528 | fast | 0.5278 s/file |
| `anomaly_file_content` | hadoop | target_files=3, comparison=ALL | 1.328 | 1.293 | slow | 0.4311 s/file |
| `anomaly_file_content` | hadoop | target_files=5, comparison=ALL | 2.157 | 2.093 | slow | 0.4187 s/file |
| `anomaly_line_content` | hadoop | 1 file, scores every line + writes a plot | 0.32 | 0.26 | fast |  |
| `distance_folder_content` | hadoop | comparison=10, Words | 2.901 | 2.776 | slow |  |
| `distance_folder_content` | hadoop | comparison=10, 3grams | 14.528 | 11.352 | very slow |  |
| `distance_folder_content` | hadoop | comparison=10, Sklearn | 2.777 | 2.833 | slow |  |
| `distance_folder_content` | hadoop | comparison=10, Parse-Tip | 0.176 | 0.176 | fast |  |
| `distance_folder_content` | hadoop | comparison=10, Parse-Drain | 3.588 | 0.185 | fast |  |
| `plot_folder_filename` | hadoop | plots=["scatter"] (default) | 0.07 | 0.066 | instant |  |
| `plot_folder_content` | hadoop | comparison_folders=5, plots=[scatter] | 0.116 | 0.112 | fast | 0.0223 s/folder |
| `plot_folder_content` | hadoop | comparison_folders=27, plots=[scatter] | 0.318 | 0.314 | fast | 0.0116 s/folder |
| `plot_folder_content` | hadoop | comparison_folders=54, plots=[scatter] | 0.482 | 0.469 | fast | 0.0087 s/folder |
| `plot_folder_content` | hadoop | plots=["umap","scatter"] | 8.016 | 0.514 | fast |  |
| `plot_file_content` | hadoop | 1 file, plots=["scatter"] | 0.065 | 0.07 | instant |  |
| `plot_file_content` | hadoop | 1 file, plots=["umap","scatter"] | 0.206 | 0.204 | fast |  |
| `query_result` | hadoop | filter + sort a 55-row result | 0.009 | 0.009 | instant |  |
| `open_log_root` | hdfs | read + mask + parse tip (5,000 files) | 126.781 | 0.085 | instant | 0.0 s/file |
| `describe_log_root` | hdfs | 5,000 log folders | 0.041 | 0.043 | instant |  |
| `describe_log_root` | hdfs | include_files=True (5,000 files) | 0.069 | 0.059 | instant |  |
| `search_log_lines` | hdfs | regex over 91k lines | 0.014 | 0.014 | instant |  |
| `distance_folder_filename` | hdfs | comparison_folders=ALL (4,999) | 44.082 | 41.651 | very slow | 0.0083 s/folder |
| `distance_folder_content` | hdfs | comparison_folders=ALL (4,999), Words | 38.151 | 34.599 | very slow | 0.0069 s/folder |
| `anomaly_folder_content` | hdfs | target_folder=1, comparison="Normal_*" | 0.265 | 0.26 | fast | 0.2596 s/target |
| `anomaly_folder_content` | hdfs | target_folder=5, comparison="Normal_*" | 1.272 | 1.3 | slow | 0.26 s/target |
| `anomaly_folder_content` | hdfs | target_folder=20, comparison="Normal_*" | 5.028 | 4.972 | slow | 0.2486 s/target |
| `plot_folder_filename` | hdfs | plots=["scatter"] (default) | 0.113 | 0.098 | instant |  |
| `plot_folder_content` | hdfs | comparison_folders=50, plots=[scatter] | 0.045 | 0.056 | instant | 0.0011 s/folder |
| `plot_folder_content` | hdfs | comparison_folders=500, plots=[scatter] | 0.08 | 0.081 | instant | 0.0002 s/folder |
| `plot_folder_content` | hdfs | comparison_folders=4999, plots=[scatter] (= ALL) | 0.279 | 0.261 | fast | 0.0001 s/folder |
| `plot_folder_content` | hdfs | plots=["umap","scatter"] | 21.596 | 8.457 | slow |  |
| `plot_folder_content` | hdfs | plots=["umap"], random_seed=None (threaded) | 4.177 | 3.186 | slow |  |
| `query_result` | hdfs | filter + sort a 5,000-row result | 0.004 | 0.004 | instant |  |
