# Datasets

`download_data.py` fetches every public log dataset LogLead knows about, as listed in
[`datasets.yml`](datasets.yml). This file describes what each of those datasets is.

```
uv run downloader/download_data.py                                    # everything in datasets.yml
uv run downloader/download_data.py --config tests/datasets_json.yml   # one test-specific set
uv run downloader/download_data.py --location /mnt/big/Datasets       # override root_folder
```

Each entry lands in `<root_folder>/<name>/` (default `~/Datasets`). Archives are deleted after
unpacking, an existing folder is never re-downloaded, and `download: false` on an entry skips it.
Prefer that flag over deleting an entry: this file is the record of where each dataset came from.

`datasets.yml` is **download-only**: it carries just `name`, `url`/`urls` (or `local_archive` +
`source_url`) and `download`. Everything about *reading* a dataset — `log_file`, `labels_file`,
`format`, `predictor_cols`, `expected_length` — lives in the `tests/datasets_*.yml` configs instead,
so a changed test expectation never touches this file. The top-level [`CLAUDE.md`](../CLAUDE.md)
explains how those configs are split.

## Disk space

The full set is **~124 GB unpacked**, and about **6.8 GB of transfer** to get there. Liberty, Spirit
and Thunderbird account for 97 GB of the unpacked total and LO2 for another 15 GB — set
`download: false` on those four and the rest fits in 12 GB.

Sizes are binary units (MB = MiB, GB = GiB), measured on disk after unpacking; download sizes are
the servers' `Content-Length`. Line counts are of the log files as downloaded.

| Dataset | Download | Unpacked | Lines |
|---|---|---|---|
| [bgl](#bgl) | 55 MB | 709 MB | 4,747,963 |
| [hadoop](#hadoop) | 3.3 MB | 47 MB | 394,308 |
| [hdfs](#hdfs) | 178 MB | 1.7 GB | 11,175,629 |
| [liberty](#liberty-spirit-thunderbird) | 641 MB | 29.5 GB | 265,569,231 |
| [spirit](#liberty-spirit-thunderbird) | 864 MB | 37.3 GB | 272,298,969 |
| [thunderbird](#liberty-spirit-thunderbird) | 1.9 GB | 29.7 GB | 211,212,192 |
| [nezha](#nezha) | 335 MB clone | 2.8 GB | 4,230,907 in the log files |
| [adfa](#adfa) | 2.3 MB | 8.8 MB | 5,951 traces, 2,747,550 syscall ids |
| [awsctd](#awsctd) | 9.7 MB | 559 MB | 592,505 traces, 174,847,810 syscall names |
| [nginx_json](#nginx_json) | 12 MB | 12 MB | 51,462 |
| [security_datasets](#security_datasets) | 13 KB | 180 KB | 118 |
| [ait_ads](#ait_ads) | 92 MB | 2.7 GB | 2,655,821 |
| [logfmt](#logfmt) | 22 MB | 22 MB | 56,100 |
| [syslog](#syslog) | 6.0 MB | 88 MB | 797,996 |
| [access_log](#access_log) | 267 MB, by hand | 3.3 GB | 10,365,152 |
| [loghub_csv](#loghub_csv) | 6.1 MB | 6.1 MB | 32,000 rows + 16 headers |
| [zeek](#zeek) | 44 MB | 215 MB | 1,474,104 records + 315 header lines |
| [iot23](#iot23) | 2.8 MB | 2.8 MB | 23,145 connections + 9 header lines |
| [iis](#iis) | 16 MB | 16 MB | 55,826 requests + 12 directive lines |
| [lo2](#lo2) | 2.7 GB | 15 GB | 103,140,992 |

---

## bgl

Console log of Lawrence Livermore's BlueGene/L supercomputer, 214.7 days of it, and the most-used
benchmark in log anomaly detection. Every line is independently labelled: a first field of `-` means
normal, anything else names an alert category, so the labels are per line and there is no grouping
into larger units. Distributed by [loghub](https://github.com/logpai/loghub/tree/master/BGL) via
[Zenodo](https://zenodo.org/records/8196385).

## hadoop

Hadoop MapReduce application logs from two jobs, WordCount and PageRank, run on a five-machine
cluster with faults injected deliberately — machine down, network disconnection, disk full. The tree
is `application_*/container_*/*.log`, 978 files across 55 applications, and `abnormal_label.txt`
labels each *application* rather than each line. Java stack traces span several lines here, so a
line and an event are not the same thing.
[loghub](https://github.com/logpai/loghub/tree/master/Hadoop) /
[Zenodo](https://zenodo.org/records/8196385).

## hdfs

38.7 hours of Hadoop Distributed File System logs from a 203-node EC2 cluster, the other canonical
anomaly-detection benchmark. Each line carries a block id (`blk_-1608999687919862906`) and
`preprocessed/anomaly_label.csv` labels each of the 575,061 blocks, 16,838 of them anomalous — so
the unit being labelled is the block, not the line. This is the HDFS_v1 variant;
[loghub](https://github.com/logpai/loghub/tree/master/HDFS#hdfs_v1) also publishes a larger v2 and a
TraceBench v3 that LogLead does not download.

## liberty, spirit, thunderbird

Three Sandia National Labs supercomputer syslogs from the 2004–2006 USENIX
[CFDR collection](https://www.usenix.org/cfdr-data#hpc4), gathered for Oliner and Stearley's DSN'07
paper *What Supercomputers Say*. They share BGL's layout and label convention — leading `-` is
normal, one label per line — but are two orders of magnitude larger, together roughly 750 million
lines and 97 GB unpacked. That size is why they are normally left gzipped on disk: the `.gz` files
they arrive as total 3.4 GB, and are readable as they stand.

## nezha

The first microservice dataset to publish logs, traces *and* metrics together with fault-injection
ground truth, collected from two systems — [TrainTicket](https://github.com/FudanSELab/train-ticket)
and [WebShop](https://github.com/GoogleCloudPlatform/microservices-demo) — over four days in 2022
and 2023. Fault injections are listed per day in `<date>-fault_list.json` as an injection timestamp,
a target pod and a fault type, so an event is anomalous by virtue of when and where it happened; the
two systems hold 272,270 and 3,958,203 events. It has no release archive, so it arrives as a clone of
[IntelligentDDS/Nezha](https://github.com/IntelligentDDS/Nezha) keeping `construct_data/` and
`rca_data/`, most of whose 2.8 GB is traces and metrics rather than logs.

## adfa

ADFA-LD, a host-based intrusion detection benchmark: 5,951 Linux syscall traces, each a file holding
one whitespace-separated line of syscall **ids** rather than log text. The directory name is the
label — 4,372 validation and 833 training traces are normal, and 746 attack traces cover six
techniques (adduser, hydra FTP and SSH brute force, java meterpreter, meterpreter, web shell).
Those 8.8 MB of ids amount to 2,747,550 individual syscalls.
[Labelled version on GitHub](https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset).

## awsctd

AWSCTD, the Attack-Caused Windows System Calls Traces Dataset: 592,505 traces of Windows syscall
**names**, one comma-separated sequence per line, with the malware family (or `Clean`) as the last
item on the line. The 66 CSVs are six overlapping packagings of the same material — `AllMalware`,
`MalwarePlusClean` and a second copy of each — so the folder as a whole holds 174,847,810 syscalls
and reading one subfolder is usually what you want. [DjPasco/AWSCTD](https://github.com/DjPasco/AWSCTD).

## nginx_json

51,462 nginx access-log records written as NDJSON, one JSON object per line, covering 17 May to
4 June 2015 on a small download server — almost every request is a `GET /downloads/product_1` or
`product_2`, mostly from Debian APT clients. Each record is flat and fixed (`time`, `remote_ip`,
`request`, `response`, `bytes`, `referrer`, `agent`) and there are no labels. Downloaded as a bare
file rather than an archive, from
[elastic/examples](https://github.com/elastic/examples/tree/master/Common%20Data%20Formats/nginx_json_logs).

## security_datasets

One scenario from the Open Threat Research Forge's
[Security-Datasets](https://github.com/OTRF/Security-Datasets): 118 Windows event-log records
captured while LSASS memory was dumped via direct syscalls, a credential-access technique. Records
are NDJSON carrying the Windows event schema — `EventID`, `Channel`, `Message` — with a different
field set per event type, so the file is small but structurally heterogeneous. Several hundred more
scenarios are catalogued at [securitydatasets.com](https://securitydatasets.com).

## ait_ads

The AIT Alert Data Set: intrusion-detection alerts produced by replaying the AIT Log Data Set V2
testbed, eight simulated enterprise networks (`fox`, `harrison`, `russellmitchell`, …) each attacked
over several days. Two detectors ran over every network, AMiner and Wazuh, giving 16 NDJSON files
with two very different schemas and 2,655,821 alerts in total. Ground truth comes from a separate
`labels.csv` downloaded alongside. [Zenodo record 8263181](https://zenodo.org/records/8263181).

## logfmt

Grafana Labs' own production logs from four services — grafana-ruler, agent, distributor and
ingester — committed to the Loki repository as test data for its Drain implementation. Lines are
logfmt (`ts=2024-04-16T15:10:44Z level=info msg="received file watcher event"`), and real Grafana
output routinely puts free text *before* the first `key=value` pair. Unlabelled, and the four files
differ enough in their key sets to be worth treating as four sources.
[grafana/loki testdata](https://github.com/grafana/loki/tree/main/pkg/pattern/drain/testdata).

## syslog

Three loghub corpora downloaded into one folder: a Linux `/var/log/messages` (25,567 lines), a macOS
system log (117,283) and an OpenSSH auth log (655,146, full of visible brute-force attempts). All
three are RFC 3164, the old BSD format, whose timestamps carry a month, day and time but **no year**.
Unlabelled; from [loghub](https://github.com/logpai/loghub)'s `Linux`, `Mac` and `OpenSSH` folders
via [Zenodo](https://zenodo.org/records/8196385).

## access_log

Kaggle's "Web Server Access Logs": 10,365,152 requests to the Iranian e-commerce site zanbil.ir over
a few days in 2019, in Apache combined format with a trailing `X-Forwarded-For` field. At 3.3 GB it
is the largest single log file here, and holding all of it in memory at once takes roughly 11 GB.
Kaggle serves it only to a logged-in account, so there is no URL to fetch: download it by hand from
[kaggle.com/datasets/eliasdabbas/web-server-access-logs](https://www.kaggle.com/datasets/eliasdabbas/web-server-access-logs),
put the zip where the entry's `local_archive:` points, and re-run — that archive is only read, never
deleted.

## loghub_csv

The `*_structured.csv` files loghub publishes alongside 16 of its raw logs — Android, Apache, BGL,
HDFS, Spark, Windows and ten more — each a 2,000-line sample already parsed into columns such as
`LineId`, `Content` and `EventTemplate`. The 16 land in one folder and carry 15 distinct headers
between them, since each system was parsed into whichever fields it has. BGL and Thunderbird are the
two that also carry a `Label` column and an epoch `Timestamp`.
[logpai/loghub](https://github.com/logpai/loghub).

## zeek

A complete Zeek output directory from Brim's sample data: all 35 log types (`conn`, `dns`, `http`,
`ssl`, `files`, …) from one network capture, 1,474,104 records in total, of which `conn.log` is more
than two thirds. Zeek TSV describes itself — `#separator`, `#fields`, `#types`, `#empty_field` and
`#unset_field` header lines declare the delimiter, column names, column types and null markers — so
the 35 files are 35 different schemas with very little overlap. Unlabelled.
[brimdata/zed-sample-data](https://github.com/brimdata/zed-sample-data/tree/main/zeek-default).

## iot23

One capture from the Stratosphere Lab's IoT-23 dataset: a Zeek `conn.log.labeled` of 23,145
connections from an IoT device infected with Mirai (capture CTU-IoT-Malware-Capture-34-1), 21,222 of
them malicious and 1,923 benign. It is ordinary Zeek TSV with two extra columns appended, `label` and
`det_label`, which makes it one of the few corpora here labelled per line. Small enough to load
whole. [stratosphereips.org/datasets-iot23](https://www.stratosphereips.org/datasets-iot23).

## iis

Three Microsoft IIS web-server logs from Splunk's Attack Range corpus: an Exchange 2016 server
captured over the ProxyLogon exploitation window, a WSUS server, and a PowerShell Web Access log.
They are W3C extended format, where a `#Fields:` directive names the columns and is re-declared at
every log rotation, so the column set can change partway through a file. Unlabelled, 55,826 requests.
[splunk/attack_data](https://github.com/splunk/attack_data).

## lo2

LO2v2, logs and metrics from load-testing a [Light-OAuth2](https://github.com/networknt/light-oauth2)
microservice deployment: 115 runs × ~54 test cases × 7 services, 43,078 log files and 103 million
lines. A test case named `correct` is normal and every other name is the error injected, so the
labels live in the directory tree rather than in a file or a column. Use **v2, not
[v1](https://zenodo.org/records/14938118)** — v1 ran the correct test first in a fixed order, so
service startup lines leaked into the normal class and inflated F1 from 0.623 to 0.976. This entry
takes `light-oauth2-logs.zip`, the reduced log set the v2 paper's own analysis used, rather than the
65.6 GB full record. [Zenodo record 18937117](https://zenodo.org/records/18937117).

---

## Adding a dataset

Add an entry with `name` plus `url`, `urls` (several files into one folder) or `local_archive` +
`source_url` (for anything behind a login). `.zip`, `.tar*`, `.gz` and `.7z` are unpacked
automatically and the archive removed; a GitHub tree URL is cloned and the named folder kept;
anything else is left as the plain file it is. Then add the reading side — `log_file`, `format`,
`expected_length` and the rest — to the relevant `tests/datasets_*.yml`, and add a section here.
