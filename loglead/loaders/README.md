# Loaders

A loader turns log-data into a Polars DataFrame that the rest of
LogLead (`EventLogEnhancer` → `SequenceEnhancer` → `AnomalyDetector`) can work with without knowing
which dataset it is looking at. Every loader implements `load()` (read the raw files into
`self.df`) and `preprocess()` (dataset-specific cleanup); `BaseLoader.execute()` drives
`load → preprocess → check_for_nulls_and_non_utf8 → check_mandatory_columns → add_ano_col`. See the
"Architecture" section of the top-level [`CLAUDE.md`](../../CLAUDE.md) for the full column-naming
convention (`m_*`/`e_*`/`seq_*`, `normal`/`anomaly`, `df` vs `df_seq`).

**If you don't know which one you need, use [`AutoLoader`](auto.py)** — it works out the format and
builds the right loader from the list below:

```python
AutoLoader(filename="mystery.log").execute()                     # one file
AutoLoader(filename="logs", filename_pattern="*.log").execute()  # a tree, detected per file
```

Four loader shapes live side by side here:

- **Detecting** — `AutoLoader`, which picks one of the others and builds it.
- **General-purpose** — `RawLoader`, the fallback for any line-oriented text file: one column,
  `m_message`, one row per line, no dataset-specific assumptions.
- **Dataset-specific** — one Python class per dataset (`HDFSLoader`, `BGLLoader`, ...). Most of the
  table below.
- **Spec-driven** — one class per *format family*, configured by a YAML spec instead of subclassed
  per dataset: `JsonLoader`, `AccessLogLoader`, `DelimitedLoader`, `LogfmtLoader`, `SyslogLoader`.
  Each of these can also read a dataset that has no shipped spec via inline keyword arguments or a
  spec file of your own — the keyword arguments *are* the spec keys.

Plus `BaseLoader`, the base class every loader above implements.

## At a glance

For the five spec-driven loaders, one representative dataset is shown here; each has more than one
shipped spec, listed with its own landing page in that loader's section below.

| Loader | Reads | Dataset / example dataset | Landing page |
|---|---|---|---|
| [`AutoLoader`](auto.py) | anything — detects the format, then builds the loader that reads it | — (any of the below) | — |
| [`RawLoader`](raw.py) | any file or directory tree, one event per line | — (your own data) | — |
| [`JsonLoader`](json.py) | JSON/NDJSON, one class + a spec per schema | nginx JSON access logs (+ 3 more, see below) | [github.com/elastic/examples/.../nginx_json_logs](https://github.com/elastic/examples/tree/master/Common%20Data%20Formats/nginx_json_logs) |
| [`AccessLogLoader`](access_log.py) | Apache/nginx access logs, one class + a spec per layout | Kaggle "Web Server Access Logs" | [kaggle.com/datasets/eliasdabbas/web-server-access-logs](https://www.kaggle.com/datasets/eliasdabbas/web-server-access-logs) |
| [`DelimitedLoader`](delimited.py) | CSV/TSV with a header, Zeek's `#fields` TSV, W3C/IIS `#Fields:` | loghub `*_structured.csv` (+ 4 more, see below) | [github.com/logpai/loghub](https://github.com/logpai/loghub) |
| [`LogfmtLoader`](logfmt.py) | `key=value` logfmt text | Grafana/Loki Drain test data | [github.com/grafana/loki](https://github.com/grafana/loki/tree/main/pkg/pattern/drain/testdata) |
| [`SyslogLoader`](syslog.py) | syslog, RFC 3164 / RFC 5424 | loghub Linux / Mac / OpenSSH | [github.com/logpai/loghub](https://github.com/logpai/loghub) |
| [`HDFSLoader`](hdfs.py) | one large log + a CSV label file | HDFS_v1 | [github.com/logpai/loghub/tree/master/HDFS](https://github.com/logpai/loghub/tree/master/HDFS#hdfs_v1) |
| [`HadoopLoader`](hadoop.py) | tree of `application_*/container_*/*.log` + a label file | Hadoop | [github.com/logpai/loghub/tree/master/Hadoop](https://github.com/logpai/loghub/tree/master/Hadoop) |
| [`BGLLoader`](bgl.py) | one file | BlueGene/L (BGL) | [github.com/logpai/loghub/tree/master/BGL](https://github.com/logpai/loghub/tree/master/BGL) |
| [`ThuSpiLibLoader`](supercomputers.py) | one file | Thunderbird / Spirit / Liberty | [usenix.org/cfdr-data#hpc4](https://www.usenix.org/cfdr-data#hpc4) |
| [`NezhaLoader`](nezha.py) | tree of CSVs + JSONs (logs, traces, metrics, fault labels) | Nezha (TrainTicket, WebShop) | [github.com/IntelligentDDS/Nezha](https://github.com/IntelligentDDS/Nezha) |
| [`LO2Loader`](lo2.py) | tree of runs/test-cases/services (text logs + JSON metrics) | LO2v2 (Light-OAuth2 microservice logs & metrics) | [zenodo.org/records/18937117](https://zenodo.org/records/18937117) |
| [`ADFALoader`](adfa.py) | directories of `.txt` (already-parsed syscall ids) | ADFA-LD | [github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset](https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset) |
| [`AWSCTDLoader`](awsctd.py) | directories of `.csv` (syscall names, one sequence per file) | AWSCTD | [github.com/DjPasco/AWSCTD](https://github.com/DjPasco/AWSCTD) |
| [`ProLoader`](pro.py) | directory of files | Profilence | not a public dataset (see below) |

## Detecting

### `AutoLoader` ([`auto.py`](auto.py))

The loader for when you do not know which loader you need. It samples a file, decides what format it
is in, and builds the loader that reads that format — so `AutoLoader(filename=...).execute()` works
on anything, and `detections()` tells you what it decided and how strong the evidence was.

Detection happens in two stages, most specific first:

1. **Dataset probe** — is this a public dataset that has its own loader? Recognized from what sits
   *next to* the log rather than only from the log itself, because those loaders need it:
   `HDFSLoader` needs its `anomaly_label.csv` and `HadoopLoader` its `abnormal_label.txt`, and the
   sibling file both confirms the dataset and supplies the argument. A dataset whose labels are
   missing is deliberately **not** recognized as that dataset — it falls through to stage 2 and loads unlabeled rather
   than crashing or silently marking everything normal. Covers HDFS, Hadoop, ADFA, AWSCTD, Nezha,
   LO2, Profilence, BGL and Thunderbird/Spirit/Liberty. Not every dataset keeps its labels in a
   file: LO2's label is the *name of the test-case directory* (`correct` vs the error injected), so
   there the `correct/` directory is what the probe looks for. **`dataset_probe=False`** skips this
   check, and you need that when a line has to carry the file it came from. These dataset loaders
   don't record one: they number events with a `seq_id` instead (Hadoop's frame has
   `seq_id`/`seq_id_sub`; Nezha keeps only the base name of the file). That is why the log-folder
   comparison in [`loglead/delta/`](../delta/) passes it. Only the check on the directory is skipped —
   a single BGL- or HDFS-shaped file inside the tree is still recognized below, and does get a file
   name.
2. **Format probe, per file** — a file that declares its own columns (Zeek's `#separator`, W3C's
   `#Fields:`) → JSON → web access log → syslog → logfmt → a delimited file with a header row →
   generic timestamped text → plain text. Each candidate is scored as a match rate over a sample of
   the file, and the best one above `min_match_rate` (default 0.5) wins. Specific before generic,
   which is what stops a logfmt line whose value happens to be an ISO timestamp from being read as
   generic timestamped text — and why the two delimited tests sit at opposite ends: a file that
   names its own columns has said what it is, while "the first line looks like a header" is the
   weakest evidence here and goes almost last.

Detection is per file because a folder holding several formats is the normal case rather than the
exception. When every file agrees, the whole tree goes to one loader and keeps its parallel
multi-file read; only a genuinely mixed folder pays for one loader per file, and those results are
stacked into a single frame.

Nothing is ever refused: an unrecognized file is read as plain text and said so. Files that did not
decode cleanly are reported separately from files whose format was guessed, since otherwise a
mis-decoded file and a mis-detected one look identical downstream.

Worked example: [`demo/AutoLoader_samples.py`](../../demo/AutoLoader_samples.py) — the first two
sections need no download.

## General-purpose

### `BaseLoader` ([`base.py`](base.py))

Not a loader for any dataset — the template-method base class every loader below subclasses.
Defines the `load → preprocess → check_for_nulls_and_non_utf8 → check_mandatory_columns →
add_ano_col` pipeline, the `_split_and_unnest()` helper used by the positional-text loaders, and the
`\a`-separator CSV-reader trick used to read a log file one whole line at a time.

### `RawLoader` ([`raw.py`](raw.py))

The starting point for any log file that doesn't have (or doesn't need) a dedicated loader: one
column, `m_message`, one row per line, no labels. Optional regex-based timestamp extraction
(`timestamp_pattern` + `timestamp_format`) with several strategies for lines that don't match
(`drop`/`keep`/`fill-lastseen`/`merge`). It is also what `loglead/delta/` and the MCP tools use when
asked for `format="raw"`, and what `AutoLoader`'s generic and plain-text branches build — log-folder
comparison works on arbitrary, unlabeled log trees, so plain text is always an acceptable answer
there.

## Spec-driven format families

Each of these is one Python class covering a whole format, configured either by a shipped YAML spec
(`format="name"`), a path to your own spec file (`format="./my_format.yml"`), or inline keyword
arguments — the three forms use the same keys, so a spec file is nothing more than a serialized
constructor call. See each module's docstring for the full keyword-argument reference.

### `JsonLoader` ([`json.py`](json.py))

One loader for JSON/NDJSON logs; what differs between JSON datasets is only the *mapping* (which
key is the timestamp, which is the message, which correlates records), so that mapping is the spec.
Shipped specs live in [`json_formats/`](json_formats/):

| Spec | Dataset | Landing page |
|---|---|---|
| [`nginx_json.yml`](json_formats/nginx_json.yml) | nginx JSON access logs (`elastic/examples`) | [github.com/elastic/examples/.../nginx_json_logs](https://github.com/elastic/examples/tree/master/Common%20Data%20Formats/nginx_json_logs) |
| [`otrf_winevent.yml`](json_formats/otrf_winevent.yml) | OTRF Security-Datasets (Windows event logs, one scenario per file) | [github.com/OTRF/Security-Datasets](https://github.com/OTRF/Security-Datasets) / [securitydatasets.com](https://securitydatasets.com) |
| [`nginx_plus_status.yml`](json_formats/nginx_plus_status.yml) | NGINX Plus status API snapshots (`elastic/examples`, `nginx_json_plus_logs`) | [github.com/elastic/examples/.../nginx_json_plus_logs](https://github.com/elastic/examples/tree/master/Common%20Data%20Formats/nginx_json_plus_logs) |
| [`gelf.yml`](json_formats/gelf.yml) | GELF (Logstash/Elasticsearch-style envelope) — a message format, not a dataset | [go2docs.graylog.org - GELF format](https://go2docs.graylog.org/current/getting_in_log_data/gelf_format.html) |

The AIT Alert Data Set (AIT-ADS) is a fourth JSON dataset LogLead downloads
(`tests/datasets_json.yml`) but does not yet ship a spec for — three IDS schemas
(AMiner/Wazuh/Suricata) in one directory, selected by `file_pattern`. Landing page:
[zenodo.org/records/8263181](https://zenodo.org/records/8263181).

### `AccessLogLoader` ([`access_log.py`](access_log.py))

One loader for Apache/nginx web access logs. A layout is positional text, so a spec is just a
regex — written as an nginx `log_format` string, which the loader compiles. Shipped specs live in
[`access_log_formats/`](access_log_formats/):

| Spec | Layout | Notes |
|---|---|---|
| [`common.yml`](access_log_formats/common.yml) | NCSA Common Log Format (CLF) | Apache's "common" / nginx's 7-field baseline |
| [`combined.yml`](access_log_formats/combined.yml) | CLF + referrer + user agent | Apache's "combined" / nginx's default; byte-identical between the two servers |
| [`combined_xff.yml`](access_log_formats/combined_xff.yml) | Combined + trailing `$http_x_forwarded_for` | matches the Kaggle dataset below |
| [`vhost_combined.yml`](access_log_formats/vhost_combined.yml) | Combined, prefixed with the virtual host | Apache's "vhost_combined" / Debian's multi-site default |

Dataset exercised in the test suite (`tests/datasets_access_log.yml`, format `combined_xff`):
Kaggle "Web Server Access Logs" (zanbil.ir, 10,365,152 lines) —
[kaggle.com/datasets/eliasdabbas/web-server-access-logs](https://www.kaggle.com/datasets/eliasdabbas/web-server-access-logs).
Kaggle only serves it to a logged-in account, so it must be downloaded by hand and pointed at via
`local_archive:` rather than a URL.

### `DelimitedLoader` ([`delimited.py`](delimited.py))

One loader for delimited text that names its own columns — CSV and TSV with a header row, plus the
self-describing variants. Polars reads the bytes, so what this loader adds is the mapping layer and
one question the other families never face: *where do the column names come from*. Three answers,
decided **per file**, so a directory holding more than one reads in a single call:

| Header style | Where the names come from | Also declared |
|---|---|---|
| `row` | the first line, as in any CSV | — (the delimiter is sniffed from it) |
| `zeek` | `#fields` | `#separator`, `#types`, `#empty_field`, `#unset_field` — delimiter, column types and null markers all come out of the file |
| `w3c` | a `#Fields:` directive, re-declared at every log rotation | — (space separated) |

Shipped specs live in [`delimited_formats/`](delimited_formats/):

| Spec | Format | Notes |
|---|---|---|
| [`zeek.yml`](delimited_formats/zeek.yml) | Zeek TSV, any log type | `ts` as an epoch timestamp, `uid` as `seq_id`; everything else is read from the file |
| [`zeek_labeled.yml`](delimited_formats/zeek_labeled.yml) | Zeek `conn.log.labeled` | IoT-23's appended `label`/`det_label`; renders an explicit message so the label does not leak into `m_message` |
| [`iis.yml`](delimited_formats/iis.yml) | Microsoft IIS, W3C extended | IIS's default field set as the fallback for a file that lost its `#Fields:` line |
| [`loghub.yml`](delimited_formats/loghub.yml) | loghub `*_structured.csv` | `Content` is the message; no timestamp, since each of the 16 systems splits it differently |
| [`loghub_labeled.yml`](delimited_formats/loghub_labeled.yml) | loghub `*_structured.csv` for BGL and Thunderbird | the two that also carry `Label` and an epoch `Timestamp` |

Datasets exercised in the test suite (`tests/datasets_csv_tsv.yml`), one per header style:

| Dataset | Style | Landing page |
|---|---|---|
| loghub `*_structured.csv`, 16 systems in one folder (32,000 rows, 15 distinct headers) | row | [github.com/logpai/loghub](https://github.com/logpai/loghub) |
| Brim's Zeek sample data — a whole 35-log-type output directory, 1,474,104 events | zeek | [github.com/brimdata/zed-sample-data](https://github.com/brimdata/zed-sample-data/tree/main/zeek-default) |
| IoT-23 `conn.log.labeled`, labelled per line (23,145 events, 21,222 malicious) | zeek | [stratosphereips.org/datasets-iot23](https://www.stratosphereips.org/datasets-iot23) |
| Splunk Attack Range IIS logs, incl. an Exchange 2016 server over the ProxyLogon window | w3c | [github.com/splunk/attack_data](https://github.com/splunk/attack_data) |

Two things worth knowing before extending it. Nulls are the designed outcome, not a defect: Zeek and
W3C write a marker for every field an event did not carry, and a folder of files with different
headers is sparse by construction (the Zeek directory above lands as 342 columns). And where a
format has no message column — Zeek and W3C both — the row is rendered as `name=value` text, since a
delimited row only becomes a log line once the header is put back in front of the values;
`label_field` is never part of that rendering, because a message stating the answer makes every
detector look perfect.

### `LogfmtLoader` ([`logfmt.py`](logfmt.py))

One loader for logfmt (`key=value key2="value with spaces"`), the convention Heroku popularized and
most of the Go ecosystem (Grafana, Loki, Prometheus, Docker, Consul) emits by default. Every line
carries its own field names and they're conventional (`ts`/`time`/`timestamp`/`t`,
`msg`/`message`, `level`/`lvl`/`severity`), so unlike JSON there is usually no mapping to configure
and no per-dataset spec directory.

Dataset exercised in the test suite (`tests/datasets_fmt.yml`): Grafana Labs' own production
logs from four services (grafana-ruler, agent, distributor, ingester), committed to `grafana/loki`
as Drain test data —
[github.com/grafana/loki/tree/main/pkg/pattern/drain/testdata](https://github.com/grafana/loki/tree/main/pkg/pattern/drain/testdata).

### `SyslogLoader` ([`syslog.py`](syslog.py))

One loader for syslog. Unlike access logs, syslog isn't one layout per site — RFC 3164 (old BSD
format) and RFC 5424 (structured successor) cover it, so both are built-in regexes rather than
spec files; which one applies is detected per file.

Dataset exercised in the test suite (`tests/datasets_syslog.yml`): three loghub files in one
directory — Linux `/var/log/messages`, macOS, and an OpenSSH auth log (full of visible brute-force
attempts) — [github.com/logpai/loghub](https://github.com/logpai/loghub) (see the `Linux`, `Mac`
and `OpenSSH` subdirectories). No labels.

## Dataset-specific loaders

### `HDFSLoader` ([`hdfs.py`](hdfs.py))

Hadoop Distributed File System logs, sequence-labeled by block id (`blk_...`, extracted with a
regex into `seq_id`); anomaly labels come from a separate CSV. Dataset: **HDFS_v1** —
[github.com/logpai/loghub/tree/master/HDFS](https://github.com/logpai/loghub/tree/master/HDFS#hdfs_v1),
full download via [Zenodo](https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1)
(courtesy of the [LogHub](https://github.com/logpai/loghub) team). Canonical worked example:
[`demo/HDFS_samples.py`](../../demo/HDFS_samples.py).

### `HadoopLoader` ([`hadoop.py`](hadoop.py))

A tree of `application_*/container_*/*.log` files, sequence-labeled at the application-id level;
handles multi-line entries (e.g. stack traces) by merging continuation lines into their parent
event. Dataset: **Hadoop** —
[github.com/logpai/loghub/tree/master/Hadoop](https://github.com/logpai/loghub/tree/master/Hadoop),
full download via [Zenodo](https://zenodo.org/records/8196385/files/Hadoop.zip?download=1).

### `BGLLoader` ([`bgl.py`](bgl.py))

Event-labeled only (no sequence grouping) — the first field starting with `-` means normal.
Dataset: **BlueGene/L (BGL)** —
[github.com/logpai/loghub/tree/master/BGL](https://github.com/logpai/loghub/tree/master/BGL),
full download via [Zenodo](https://zenodo.org/records/8196385/files/BGL.zip?download=1).

### `ThuSpiLibLoader` ([`supercomputers.py`](supercomputers.py))

Event-labeled only, same label convention as BGL, additionally splitting the embedded
`component[pid]` field. Covers three related supercomputer logs sharing one layout:

| Dataset | Landing page |
|---|---|
| Thunderbird | [usenix.org/cfdr-data#hpc4](https://www.usenix.org/cfdr-data#hpc4) (log excerpt: [loghub Thunderbird](https://github.com/logpai/loghub/tree/master/Thunderbird)) |
| Spirit | [usenix.org/cfdr-data#hpc4](https://www.usenix.org/cfdr-data#hpc4) |
| Liberty | [usenix.org/cfdr-data#hpc4](https://www.usenix.org/cfdr-data#hpc4) |

All three are hosted by USENIX's Computer Failure Data Repository (CFDR). Canonical worked example:
[`demo/TB_samples.py`](../../demo/TB_samples.py).

### `NezhaLoader` ([`nezha.py`](nezha.py))

The largest loader in the package — reads logs, traces, metrics and fault-injection ground truth
from a tree of CSVs and JSONs, and joins injected-fault time windows onto both events and metrics
to derive `anomaly`. Dataset: **Nezha** —
[github.com/IntelligentDDS/Nezha](https://github.com/IntelligentDDS/Nezha), the first microservice
log/trace/metric dataset, covering two systems (pass as the loader's `system` argument):

- `TrainTicket` — [github.com/FudanSELab/train-ticket](https://github.com/FudanSELab/train-ticket)
- `WebShop` — [github.com/GoogleCloudPlatform/microservices-demo](https://github.com/GoogleCloudPlatform/microservices-demo)

### `LO2Loader` ([`lo2.py`](lo2.py))

Reads a tree of `run/test_case/service` log files (plus optional Prometheus-style JSON metrics via
`load_metrics()`) from load-testing a Light-OAuth2 microservice deployment; `seq_id` is built from
`run__test_case__service`, and `test_case == "correct"` is normal. Dataset: **LO2v2 — An Improved
Microservice Dataset of Logs and Metrics** —
[Zenodo (record 18937117)](https://zenodo.org/records/18937117). The dataset's own runs test the
[Light-OAuth2](https://github.com/networknt/light-oauth2) services the loader's `single_service`
argument names (`client`, `code`, `key`, `refresh-token`, `service`, `token`, `user`).

Use **v2, not v1** ([record 14938118](https://zenodo.org/records/14938118),
[arXiv:2504.12067](https://arxiv.org/abs/2504.12067)). In v1 every run ran the "correct" test first
in a fixed order, so service startup lines occurred only in correct tests and a classifier could
score F1 0.976 on the Token service by learning `registered for` and `status.yml` — leakage, not
detection. v2 shuffles test order and randomizes durations; the same analysis drops to 0.623.

The test suite (`tests/datasets_lo2.yml`) reads only `light-oauth2-logs.zip` (2.9 GB), the reduced
log set the v2 paper's own analysis used — the full `LO2v2.zip` is 65.6 GB and is mostly metrics
and traces this loader does not read.

### `ADFALoader` ([`adfa.py`](adfa.py))

Reads directories of `.txt` files that already contain parsed Linux syscall **ids** rather than log
text (`m_message` is an event id, not a message), so `EventLogEnhancer` parsing is unneeded; the
label comes from the directory name. Dataset: **ADFA-LD** (labelled version) —
[github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset](https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset),
designed for host-based intrusion detection.

### `AWSCTDLoader` ([`awsctd.py`](awsctd.py))

Reads directories of `.csv` files, one sequence (one comma-separated line of syscall **names**) per
file; the label is the last item on the line. Dataset: **AWSCTD** (Attack-caused Windows System
Calls Traces Dataset) —
[github.com/DjPasco/AWSCTD](https://github.com/DjPasco/AWSCTD), also designed for intrusion
detection.

### `ProLoader` ([`pro.py`](pro.py))

Reads a directory of positional whitespace-separated text files; the anomaly label comes from
whether the file name starts with `success`. Dataset: **Profilence** — not a public dataset (see
the "Not open dataset" comment in the loader itself); included because the rest of the pipeline
(enhancers, anomaly detectors) is exercised against it internally, not because the data is
downloadable.
