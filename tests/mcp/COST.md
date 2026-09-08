# What each MCP tool costs
TODO: Very poor quality text by Claude. Read through and fix if possible

The tools in `loglead/mcp/server.py` span five orders of magnitude in cost. The cheapest measured
call is 2 ms; the most expensive completed one is 166 s; the default arguments of `anomaly_*` on a
log root of large log folders are not slow but fatal, taking 14.6 GB and dying on a 16 GB machine.
None of this is visible in the tool schema, so this file measures it: what a call costs, what the
cost is proportional to, and which arguments move it.

The seconds below are from one machine and are not portable. The *ratios* between rows, and the
split between fixed and per-unit cost, are properties of the code and do travel.

Sections: [the test data](#the-test-data), [how to read a row](#how-to-read-a-row), [how to predict
a call you have not made](#how-to-predict-a-call), [when memory rather than time is the
limit](#when-memory-not-time-is-the-limit), [how to choose arguments](#choosing-arguments), and
[the full table of measurements](#measurements).

## The test data

Three log roots were measured. `tests/mcp/make_test_data.py` builds the first two from public
loghub datasets; the third is a plain loghub download that `split_log_file` cuts into slices.

| log root | log folders | log files | log lines | files per folder | lines per folder |
|---|---|---|---|---|---|
| **hadoop** (`hadoop_renamed`) | 55 | 978 | 180,897 | ~18 | ~3,300 |
| **hdfs** (`hdfs_balanced_5k`) | 5,000 | 5,000 | 90,862 | 1 | ~18 |
| **bgl** (`BGL.log` split into 10) | 10 | 10 | 4,713,493 | 1 | **~471,000** |

(BGL's count is rows in the loaded frame; the file on disk is 4,747,963 lines, and the difference
is continuation lines loading as part of the event above them.)

The three are extreme on three different axes, which is the point of measuring all three. Hadoop has
few log folders, each moderately large, with many files inside each. HDFS has very many log folders,
each tiny, one file each. BGL has very few log folders, each enormous, one file each — 26x Hadoop's
lines in a fifth of the log folders.

Total log size predicts a call's cost badly. HDFS holds half of Hadoop's lines and yet produces the
single slowest completed call in this file (`distance_folder_filename` at 41.7 s), because that tool
is charged per log folder and HDFS has 5,000 of them. What predicts cost is the match between what a
tool is charged per and which axis the log root is large on: per-folder costs are worst on HDFS,
per-file and per-line costs are worst on Hadoop, and costs charged per *unit of text inside* one log
folder are worst on BGL. That last axis is the one the other two never exercise, and the gap is
large: one BGL comparison folder is 471,000 lines to vectorize, so `distance_folder_content` there
costs roughly 19.5 s per comparison folder against Hadoop's 0.285 s — 68x more, on a log root with a
fifth as many folders.

BGL is also the only log root here where a call fails rather than merely taking a long time; see
[when memory, not time, is the limit](#when-memory-not-time-is-the-limit).

Hadoop and HDFS were measured with `--repeat 3`. BGL's heavy rows are single measurements, because
one of them is two and a half minutes. All runs used a warm parquet cache.

## How to read a row

Each row in the measurement table is one call at one size.

**first** is the cold call and **warm** is the median of the repeats. A large gap between them means
the session computed a column once and kept it, and that the cost is paid once per session rather
than once per call: `Parse-Drain` is 3.59 s the first time and 0.18 s after. **class** bands the warm
number — `instant` under 0.1 s, `fast` under 1 s, `slow` under 10 s, `very slow` above.

**per unit** is the warm time divided by the size of that call. It is useful for comparing two rows
of the same tool and useless for predicting a third, because most tools do fixed work before the
first unit: `distance_file_content` reads as 10.83 s per file at one file and 2.61 s per file at
five, and neither number is the cost of a file. Use the formulas in the next section instead.

## How to predict a call

Every tool with a size argument was measured at two or three sizes, and the measurements fitted to a
straight line:

```
seconds = fixed + marginal * n
```

`fixed` is the work a call does regardless of `n` — loading, aggregating every log folder's text,
building a term matrix. `marginal` is what one more comparison folder, target folder or file adds.
The split is what tells you whether narrowing a selector will save anything: where `fixed` dominates,
it will not.

| tool | log root | fixed | marginal | per what |
|---|---|---|---|---|
| `distance_folder_filename` | hadoop | 0.04s | **0.015s** | comparison folder |
| `distance_folder_content` | hadoop | 0.09s | **0.285s** | comparison folder |
| `distance_file_content` | hadoop | **10.24s** | 0.551s | target file |
| `distance_line_content` | hadoop | 0.01s | 0.013s | diff |
| `anomaly_folder_filename` | hadoop | 0.02s | **0.163s** | target folder |
| `anomaly_folder_content` | hadoop | 0.09s | **0.498s** | target folder |
| `anomaly_file_content` | hadoop | 0.35s | 0.141s | target file |
| `plot_folder_content` | hadoop | 0.09s | 0.007s | comparison folder |
| `anomaly_folder_content` | hdfs | 0.04s | **0.247s** | target folder |
| `plot_folder_content` | hdfs | 0.06s | **0.000s** | comparison folder |
| `distance_folder_content` | bgl | 0.00s | **19.515s** | comparison folder |

Put your own `n` into the formula. `anomaly_folder_content` on Hadoop with 20 target folders is
`0.09 + 0.498 * 20`, about 10 s; with all 55 it predicts 27 s, which is what the table measured.

Two limits on this, both measured rather than assumed.

**The BGL fit is a slope, not a model.** Its three points are 13.4 s at one comparison folder, 41.0 s
at three and 166.3 s at nine — the per-unit cost rises from 13.4 s to 18.5 s across that range, so the
cost grows faster than linearly and the fitted 19.5 s per folder is a summary of the range measured,
not something to extrapolate past nine.

**A rate from one log root does not transfer to another**, and the three here disagree by two orders
of magnitude in both directions. `distance_folder_content` costs 0.285 s per comparison folder on
Hadoop. That rate predicts 24 minutes for HDFS's 4,999 folders; the measured time is 35 s, because an
HDFS log folder is 18 lines. The same rate predicts 2.6 s for BGL's nine; the measured time is 166 s,
because a BGL log folder is 471,000 lines. Use the formula for the log root you have; failing that,
use a measured "ALL" row from the table — and if the log root is a split single file, expect it to
behave like BGL rather than like either of the others.

## When memory, not time, is the limit

On Hadoop and HDFS every tool finishes and the only question is how long you wait. On BGL that stops
being true, because what a log folder costs is the text inside it, and BGL's hold 471,000 lines each.

Each of these was measured in its own fresh process on a 16 GB machine:

| call | log root | time | peak RSS |
|---|---|---|---|
| `open_log_root` (read + mask + parse tip) | bgl | 26s | **2.9 GB** |
| `distance_folder_content`, comparison=ALL, `Parse-Tip` | bgl | 13s | 2.9 GB |
| `distance_folder_content`, comparison=ALL, `Words` | bgl | 148s | 3.8 GB |
| `plot_folder_content`, `plots=["umap","scatter"]` | bgl | 23s | 7.6 GB |
| `distance_folder_content`, comparison=3, `3grams` | bgl | 187s | **11.6 GB** |
| `anomaly_folder_content`, `target_folder="ALL"` | bgl | 76s | **14.6 GB** |

The last two rows are the ones to know, and the qualifier "in its own fresh process" is doing the
work: each of them completes alone and each is an out-of-memory kill once anything substantial has
run before it. That is how both were found — by killing the benchmark. `target_folder` defaults to
`"ALL"`.

**The rows do not compose.** Two of them in one session is not 14.6 GB, it is an OOM kill. One
`anomaly_folder_content` target measured inside the benchmark, in a process already holding 6.6 GB,
took it to 12.0 GB; the 14.6 GB above is what all ten cost with nothing else resident, which is the
best case rather than the typical one. This non-composability is baked into how BGL is benchmarked:
`anomaly_folder_content` is measured at one size instead of swept (a sweep to four targets was killed
three times), and the cached open is measured by closing and reopening rather than by holding a second
session beside the first, which on this log root would be two 2.9 GB frames at once.

**The 2.9 GB left by `open_log_root` is a floor, not a one-off.** The session keeps the enhanced frame
by design, so every later call sits on top of it, leaving ~12 GB of headroom for everything else.
**And `close_log_root` does not give it back**: opening BGL cold, closing the session, collecting
garbage and reopening from the parquet cache leaves the process at 6.6 GB rather than the 2.9 GB of a
single fresh open, because Python's allocator keeps the freed arenas. Closing frees memory for the
next call to reuse; it does not restore headroom to a process that has already run out. Restarting
does.

The way out on a log root of this shape is the parser formats. `Parse-Tip` is flat in memory as well
as cheap in time — 2.9 GB at every comparison count measured — against `Words` climbing with each
folder added and `3grams` at 11.6 GB. Here they are not merely the fast option, they are the one that
fits.

## Choosing arguments

These are the decisions that change what a call costs, in the order a caller makes them.

**0. Peek before you open, and split a single file before you peek again.** `peek_log_root` stats the
files and samples a few hundred lines; it never parses, so a 743 MB log costs what a small one does
(5 ms warm on BGL). It is also what tells you a path holds one file rather than a set of log folders,
in which case nothing here has anything to compare and `split_log_file` cuts it into slices first.
Splitting BGL's 743 MB into ten takes 2.9 s, and asking for the same split again is free — the slices
and a manifest describing them are already on disk.

**1. Open the log root once; it is the only call whose cost the file count drives.** The `auto` format
probe is a read per file. Probing all of them is 16.5 s on Hadoop's 978 files and 102.7 s on HDFS's
5,000, nearly all of it probing. Sampling at most `max_detect_files` (50 by default) and applying that
answer to the rest produces the same frame in **4.3 s and 3.4 s** — 4x and 30x less — and the file
count almost stops mattering. Probing everything is still one argument away (`max_detect_files=0`).
Pay it when one odd file among thousands would have to be read differently; `peek_log_root`'s
`file_names` says how many distinct file-name shapes are down there, and a log root of one shape has
nothing for a fuller probe to find. Passing an explicit `format` skips detection altogether.

BGL shows what the remaining cost is: only 10 files to probe, and still 26 s, because that part is
masking and parsing 4.7M lines. Re-opening is 0.07 s on Hadoop and HDFS and 1.0 s on BGL, since the
session is cached to parquet. Opening is paid once and everything after is cheap, which is what
sessions are for.

**2. Name your targets; do not leave `target_folder` at `"ALL"`.** The `anomaly_*` tools refit four
detectors for every target log folder: 0.498 s per folder on Hadoop, 0.247 s on HDFS, about 7 s on
BGL. So the default costs 27 s on Hadoop's 55 folders, an extrapolated ~21 minutes on HDFS's 5,000 —
and on BGL's ten it is not a time cost at all, but 14.6 GB and an out-of-memory kill. Nothing in the
call signature hints at any of this, which makes it the costliest default in these tools and, on one
log root, an unrecoverable one.

**3. Multiply the rate by your real `n` before calling something cheap.** A small marginal cost is
still large when `n` is large. `distance_folder_filename` has the smallest marginal cost measured
here, 0.015 s per comparison folder, and it also produced the slowest completed call in the file at
41.7 s, because `comparison_folders="ALL"` on HDFS is 4,999 pairs.

**4. Narrow a selector only where narrowing helps.** Whether a shorter `target_files` or
`comparison_folders` list saves time depends on how much of the call is fixed:

| tool | fixed | marginal | does narrowing help? |
|---|---|---|---|
| `distance_file_content` | 10.24s | 0.551s per file | No. The aggregation runs before any file is read. |
| `anomaly_file_content` | 0.35s | 0.141s per file | Some. Two thirds of a five-file call is per file. |
| `distance_folder_content` | 0.09s | 0.285s per folder | Yes. |
| `anomaly_folder_content` | 0.09s | 0.498s per folder | Yes — this is step 2. |
| `plot_folder_content` (scatter) | 0.09s | 0.007s per folder | No, and it does not need to. See step 6. |

**5. Pick the cheapest `content_format` that answers the question.** On identical data, at 10
comparison folders on Hadoop: `Parse-Tip` 0.18 s, `Parse-Drain` 0.18 s (after 3.59 s to parse once),
`Words` 2.78 s, `Sklearn` 2.83 s, `3grams` 11.35 s — 64x from the cheapest to the dearest. One
template id per line is a much smaller document than every word of the line. Scan with a parser
format first, then re-check only the survivors with `Words`.

The ordering holds on BGL, where the format also decides memory. At 3 comparison folders: `Parse-Tip`
4.5 s at 2.9 GB, `Words` 38 s at 3.8 GB, `Sklearn` 41 s, `3grams` 187 s at 11.6 GB — 42x end to end,
and the dearest does not survive being run after anything else.

**6. Ask for the UMAP only when you need the layout.** The default scatter of `plot_folder_content` is
cheap and stays cheap: 0.26 s for all 4,999 HDFS folders, about what 50 of them cost. Adding
`plots=["umap"]` makes the same call 8.5 s, 32x more. Two parts of that cannot be tuned away: the
first UMAP in a process pays about 13 s more for numba to compile, and `random_seed` makes umap-learn
single-threaded, 8.5 s seeded against 3.2 s unseeded — the price of a reproducible layout.

The rule is about the number of points, not the size of the log. On BGL the UMAP is effectively free:
scatter and scatter-plus-UMAP both land in a 6-13 s range across runs, because the layout has ten
points to place and what both calls really pay for is the term matrix over 4.7M lines. Ask what the
layout has to lay out before assuming it is the expensive half.

**7. Follow up with `query_result`, not another analysis.** Filtering or re-sorting a result you
already have is under 0.01 s. Recomputing it is 27 s on Hadoop, and on BGL the difference is between
free and two and a half minutes.

**8. On a flat log root, L2 is the level.** A flat log root is one where each log folder is a single
file — what `split_log_file` produces, and what `hdfs_balanced_5k` already was. It is the right shape
whenever the file *is* the unit being compared (an HDFS block, a slice of one long log), since
wrapping each file in its own directory would add a level carrying no information.

The consequence is that there is no file level to drill into: "which file inside this unit is odd" is
not a question when the unit is one file. All four L3/L4 tools match a file against its namesake in
the other log folders, and on a flat log root no two units share a file name, so
`distance_file_content`, `anomaly_file_content`, `distance_line_content` and `anomaly_line_content`
return nothing — BGL's `distance_file_content` at `target_files=ALL, comparison=ALL` is 0.62 s of
finding no match. `distance_folder_content`, `anomaly_folder_content` and `plot_folder_content` are
the tools that answer "which unit looks wrong".

## Measurements

Every measured row, in the order the benchmark runs them.

**This table is generated and the prose above it is not.** Running
`uv run tests/mcp/benchmark.py --markdown tests/mcp/COST.md` overwrites the whole file with the table
alone, so put the prose back afterwards and update its numbers to match.

Three things about the table itself. The hadoop and hdfs rows come from one run and the bgl rows from
another, on the same machine; the four `open_log_root` rows for hadoop and hdfs are newer than the
rest of their run, re-measured when format detection began sampling files, which is the only change
that has moved a number here — everything else works on an already-open session. BGL's heavy rows are
single measurements rather than medians, so `first` and `warm` there are two samples of the same call
differing by ordinary noise: read the pair as a range, not as a cache effect (the one genuine cache
effect on BGL is `open_log_root`, 25.9 s against 1.0 s). And the peak-memory numbers in [when memory,
not time, is the limit](#when-memory-not-time-is-the-limit) are not from this table at all — each was
measured in its own process, because in one process they do not fit.

| tool | log root | call | first (s) | warm (s) | class | per unit |
|---|---|---|---|---|---|---|
| `open_log_root` | hadoop | read + mask + parse tip (978 files) | 4.308 | 0.066 | instant | 0.0001 s/file |
| `open_log_root` | hadoop | same, max_detect_files=0 (probe all 978) | 16.467 | 16.467 | very slow | 0.0168 s/file |
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
| `anomaly_file_content` | hadoop | target_files=1, comparison=ALL | 0.885 | 0.501 | fast | 0.5009 s/file |
| `anomaly_file_content` | hadoop | target_files=3, comparison=ALL | 0.871 | 0.764 | fast | 0.2548 s/file |
| `anomaly_file_content` | hadoop | target_files=5, comparison=ALL | 0.965 | 1.063 | slow | 0.2126 s/file |
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
| `open_log_root` | hdfs | read + mask + parse tip (5,000 files) | 3.442 | 0.072 | instant | 0.0 s/file |
| `open_log_root` | hdfs | same, max_detect_files=0 (probe all 5,000) | 102.670 | 102.670 | very slow | 0.0205 s/file |
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
| `peek_log_root` | bgl | the unsplit 743 MB file | 0.092 | 0.005 | instant |  |
| `split_log_file` | bgl | 743 MB -> 10 slices, by=lines | 2.914 | 0.0 | instant | 0.0 s/MB |
| `open_log_root` | bgl | read + mask + parse tip (10 files, 4.7M lines) | 25.923 | 1.016 | slow | 0.0 s/line |
| `describe_log_root` | bgl | 10 log folders | 0.606 | 0.45 | fast |  |
| `read_log_lines` | bgl | 100 lines out of 471k | 0.028 | 0.026 | instant |  |
| `search_log_lines` | bgl | regex over 4.7M lines | 0.055 | 0.054 | instant |  |
| `distance_folder_filename` | bgl | comparison_folders=ALL (9) | 0.465 | 0.492 | fast | 0.0547 s/folder |
| `distance_file_content` | bgl | target_files=ALL, comparison=ALL | 0.532 | 0.62 | fast |  |
| `anomaly_folder_content` | bgl | target_folder=1, comparison=ALL | 11.768 | 11.312 | very slow | 11.3119 s/target |
| `distance_folder_content` | bgl | comparison_folders=1, Words | 15.481 | 13.417 | very slow | 13.4171 s/folder |
| `distance_folder_content` | bgl | comparison_folders=3, Words | 39.664 | 40.966 | very slow | 13.6553 s/folder |
| `distance_folder_content` | bgl | comparison_folders=9, Words (= ALL) | 160.371 | 166.257 | very slow | 18.473 s/folder |
| `distance_folder_content` | bgl | comparison=3, Parse-Tip | 4.629 | 4.494 | slow |  |
| `distance_folder_content` | bgl | comparison=3, Words | 39.328 | 37.88 | very slow |  |
| `distance_folder_content` | bgl | comparison=3, Sklearn | 41.209 | 41.293 | very slow |  |
| `plot_folder_content` | bgl | comparison_folders=ALL (9), plots=["scatter"] | 8.678 | 12.689 | very slow | 1.4099 s/folder |
| `plot_folder_content` | bgl | plots=["umap","scatter"] | 18.571 | 6.036 | slow |  |
| `query_result` | bgl | filter + sort a 10-row result | 0.003 | 0.002 | instant |  |
