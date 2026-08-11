# Test data for LogLead's JSON support

**Status:** investigation / proposal. No code changes.
**Prerequisite:** [log-format-support.md](log-format-support.md) — this document picks up from its
§5 item 1 ("JSON lines (NDJSON), done properly") and §1.3 (why `GELFLoader` doesn't count as JSON
support today). Read that first for the loader architecture and the reasoning behind prioritizing
JSON.

**Scope:** a `JsonLoader` needs data to develop and regress against. This document lists five
candidate public datasets, chosen to exercise the specific ways JSON logs are harder than the
happy path, and proposes which to actually download into a new `downloader/datasets_json.yml`.

**Outcome:** the recommendation in §3 was taken — `downloader/datasets_json.yml` now ships
candidates 1–3. The loader those datasets are for is designed in
[log-format-json-loader.md](log-format-json-loader.md), which measures polars' actual behaviour
against each of the requirements in §1 below.

Every size, line count, and format claim below was checked by actually fetching the data or
querying the GitHub/Zenodo APIs — not copied from a paper or a README's prose. See §5 for exactly
what was run.

---

## 1. What the test data has to exercise

The requirements fall out of `log-format-support.md` §1.3 — they are exactly the places where
`GELFLoader` breaks today, and where a naive `read_ndjson` call would still be wrong:

| # | Property | Why it matters |
|---|---|---|
| a | NDJSON happy path at scale | the common case; must be vectorized, not a Python loop |
| b | **Heterogeneous keys across lines** | today's `pl.concat` of per-line frames *raises* here; schema unification is the whole point |
| c | **Nested objects** | forces the path-syntax decision (`log/logger` vs `.key[i]`) |
| d | **Embedded newlines inside a string field** | the record is one line, the message is multi-line — the opposite of the usual multi-line problem |
| e | **Mixed types for one key** across lines | a schema-inference trap; Polars must be told, not left to guess |
| f | **JSON that is not NDJSON** | a top-level array, or an object wrapping one (CloudTrail) |
| g | **Several JSON schemas in one directory tree** | the `loglead/delta/` log-folder case: one root, many folders, mixed shapes |
| h | Labels, at a stated granularity | line / time-window / file — needed for `AnomalyDetector`, nice-to-have for a loader test |
| i | **Dynamic keys used as data** (map-like objects, e.g. hostnames or pool names as the JSON *keys* of an object) | breaks fixed-field struct inference — the "columns" are data, not schema, so `pl.read_ndjson`'s usual struct expansion doesn't apply |

## 2. The five candidates

| # | Dataset | Format | Size | Labels | Licence |
|---|---|---|---|---|---|
| 1 | [nginx JSON access logs](https://github.com/elastic/examples/tree/master/Common%20Data%20Formats/nginx_json_logs) (`elastic/examples`) | NDJSON, flat, 8 keys | 12.1 MB, **51,462 lines** (measured) | none (but `response` status is a proxy signal) | Apache-2.0 |
| 2 | [OTRF Security-Datasets](https://github.com/OTRF/Security-Datasets) | NDJSON, **keys vary per line** | per-scenario ZIPs, 12 KB – 1 MB (measured: one scenario = 118 lines / 180 KB) | per-file, mapped to MITRE ATT&CK techniques | MIT |
| 3 | [AIT Alert Data Set (AIT-ADS)](https://zenodo.org/records/8263181) | JSON alerts, **three different schemas** (AMiner / Wazuh / Suricata) | `ait_ads.zip` 96.2 MB, ~2.66 M alerts | `labels.csv` (3.7 kB), time-window attack labels | CC-BY-4.0 |
| 4 | [AIT Log Data Set V2](https://zenodo.org/records/5789064) (one testbed) | mixed; includes Suricata `eve.json` | 8 testbed ZIPs, **7.1 – 26.5 GB each** (130.6 GB total, ~171 GB unpacked) | **per log line**, as JSON objects referencing line numbers | CC BY-**NC**-SA 4.0 |
| 5 | [flaws.cloud CloudTrail](https://summitroute.com/blog/2020/10/09/public_dataset_of_cloudtrail_logs_from_flaws_cloud/) | `{"Records":[…]}` — JSON, **not** NDJSON | 240 MB `.tar` of gzipped chunks (verified live, 251,688,960 bytes) | none formally; the environment is almost entirely attack traffic | public dataset, summitroute.com |
| 6 | [nginx_json_plus_logs](https://github.com/elastic/examples/tree/master/Common%20Data%20Formats/nginx_json_plus_logs) (`elastic/examples`) | NDJSON, deeply nested, **hostnames/pool names used as JSON object keys** | 2.78 MB, **500 lines** (measured) | none | Apache-2.0 |

Direct links (description page vs. actual file — check both before deciding):

| # | Description page | Raw data |
|---|---|---|
| 1 | [repo folder + README](https://github.com/elastic/examples/tree/master/Common%20Data%20Formats/nginx_json_logs) | [nginx_json_logs](https://raw.githubusercontent.com/elastic/examples/master/Common%20Data%20Formats/nginx_json_logs/nginx_json_logs) (raw file, 12.1 MB) |
| 2 | [repo](https://github.com/OTRF/Security-Datasets) / [docs site](https://securitydatasets.com) | e.g. [cmd_lsass_memory_dumpert_syscalls.zip](https://github.com/OTRF/Security-Datasets/blob/master/datasets/atomic/windows/credential_access/host/cmd_lsass_memory_dumpert_syscalls.zip) — one of many scenario ZIPs under [`datasets/atomic/windows/`](https://github.com/OTRF/Security-Datasets/tree/master/datasets/atomic/windows) |
| 3 | [Zenodo record 8263181](https://zenodo.org/records/8263181) · [AIT-ADS GitHub (processing scripts)](https://github.com/ait-aecid/alert-data-set) | `ait_ads.zip` linked from the Zenodo record page |
| 4 | [Zenodo record 5789064](https://zenodo.org/records/5789064) | eight testbed ZIPs (`fox.zip` … `wilson.zip`) linked from the Zenodo record page |
| 5 | [Summit Route blog post](https://summitroute.com/blog/2020/10/09/public_dataset_of_cloudtrail_logs_from_flaws_cloud/) | [flaws_cloudtrail_logs.tar](http://summitroute.com/downloads/flaws_cloudtrail_logs.tar) |
| 6 | [repo folder + README](https://github.com/elastic/examples/tree/master/Common%20Data%20Formats/nginx_json_plus_logs) | [nginxplus_json_logs](https://raw.githubusercontent.com/elastic/examples/master/Common%20Data%20Formats/nginx_json_plus_logs/nginxplus_json_logs) (raw file, 2.78 MB) |

### 1 — nginx JSON access logs

The easiest possible starting point and the one I would write the first unit test against. Flat
objects, one per line, `{"time", "remote_ip", "remote_user", "request", "response", "bytes",
"referrer", "agent"}`, 51,462 of them. It is also two priorities in one file: it is a **web access
log delivered as JSON**, so it exercises `log-format-support.md` §5 item 1 and previews item 4. No
labels, but HTTP status plus byte counts give something to detect on, and the `time` field is in
Apache's `17/May/2015:08:05:32 +0000` format — i.e. JSON does *not* save you from timestamp
parsing, which is a useful thing for the test suite to prove.

### 2 — OTRF Security-Datasets

Windows security events and Zeek network logs captured while a named attack technique was
executed, one scenario per ZIP, 1.8k GitHub stars, MIT. I downloaded one
(`cmd_lsass_memory_dumpert_syscalls`, 12.9 KB zipped → 118 JSON lines) and it hits requirements (b)
and (d) directly: **keys differ line to line** because the fields depend on the Windows EventID,
and the `Message` field contains embedded `\r\n` and tab characters. This is precisely the input
that makes today's `GELFLoader` raise. Small enough that one scenario could ship as a committed
test fixture; the folder-per-technique layout also makes it a natural fit for the `delta`/MCP
log-root shape, with the technique name as the log-folder name.

### 3 — AIT Alert Data Set (AIT-ADS)

My pick for the *labelled, realistic scale* slot. 2.66M JSON alerts (2.29M Wazuh, 307k Suricata,
56k AMiner) from eight attack scenarios, with a separate `labels.csv` giving attack-phase time
windows, CC-BY-4.0, and — the useful part — the three IDSs emit **completely different field
sets**, so a single directory contains three unrelated JSON schemas. That is requirement (g) as a
real dataset rather than a contrived one, and it maps directly onto the log-folder comparison in
`loglead/delta/`. 96 MB is a sane download.

### 4 — AIT Log Data Set V2

The gold standard for labels — ground truth is **per log line**, given as JSON objects that
reference line numbers and name the attack step (`escalate`, `attacker_change_user`, …) — and it
contains Suricata `eve.json` alongside Apache, audit, DNS, VPN and syslog, so it also covers the
non-JSON formats from `log-format-support.md` §5 items 3–5. The problem is size: the smallest
testbed is 7.1 GB and the set is 130.6 GB compressed. Also note the licence is CC BY-**NC**-SA:
non-commercial and share-alike. That constrains *users of the data*, not LogLead (we would only
ever list a URL, never redistribute), but it is worth stating in the YAML so nobody is surprised.
Recommendation: include one testbed with `download: false`, as an opt-in.

### 5 — flaws.cloud CloudTrail

3.5 years of real attack traffic against a deliberately vulnerable AWS account, chunked into
100,000-record files in the native CloudTrail shape — `{"Records": [ … ]}` — and gzipped. This is
the only candidate that covers requirement (f), and its records are deeply nested
(`userIdentity.sessionContext.attributes…`), which covers (c) better than anything else here. Two
frictions: it is a `.tar`, which the current downloader cannot extract (see §4), and
`{"Records":[…]}` is not a format `scan_ndjson` reads, so it only becomes relevant once the loader
grows a non-NDJSON branch. Defer it, but keep it on the list — it is the honest test of whether the
JSON support is "NDJSON support" or actually "JSON support".

### 6 — nginx_json_plus_logs

Lives in the same `elastic/examples` repo, right next to candidate 1, and easy to mistake for a bigger
version of it — it isn't. It's not an access log at all: it's a periodic dump of NGINX **Plus**'s
status API (the paid product's `/status` endpoint, "Live Activity Monitoring"), so each line is a
snapshot of the whole server's state rather than one client request. Verified directly (downloaded the
raw file and parsed every line): 500 lines, one every ~5.1s over ~42.5 minutes, single server, 2.78 MB
total. The 15 top-level keys (`connections`, `ssl`, `requests`, `server_zones`, `upstreams`, `caches`,
`stream`, …) are identical on every line — no heterogeneity at the top level, unlike candidate 2 — but
`server_zones` and `upstreams` are objects keyed by **hostname** and **upstream-pool name**
respectively, i.e. the object's keys are data, not a fixed schema. That's requirement (i), and nothing
else on this list exercises it: `pl.read_ndjson`'s struct-based inference assumes a bounded field set,
and here it isn't one. There's also lower-grade heterogeneity one level down — inside
`upstreams.*.peers[]`, some peer objects carry a `max_conns` key and sibling peers in the same array
don't (confirmed in the raw data). No labels, and only 500 lines, so it's not a candidate to replace
anything in §3 — it's a small, purpose-built fixture for the dynamic-key-as-map case if/when the
`JsonLoader` needs to prove it handles that.

## 3. Coverage and recommendation

| | a scale | b het. keys | c nested | d newlines-in-field | e mixed types | f non-NDJSON | g many schemas | h labels | i dynamic-key maps |
|---|---|---|---|---|---|---|---|---|---|
| 1 nginx | ✓ | | | | | | | — | |
| 2 Security-Datasets | | **✓** | ✓ | **✓** | ✓ | | ✓ | file-level | |
| 3 AIT-ADS | **✓** | ✓ | ✓ | | | | **✓** | time-window | |
| 4 AIT-LDSv2 | ✓ | ✓ | ✓ | | | | ✓ | **per line** | |
| 5 flaws CloudTrail | ✓ | | **✓** | | | **✓** | | (implicit) | |
| 6 nginx_json_plus | | (nested only) | ✓ | | | | | none | **✓** |

**Recommendation: take 1, 2 and 3 now.** Together they cover everything except (f), they total
under 110 MB, and all three are permissively licensed. Add 4 as an opt-in entry with
`download: false` — it is the only source of per-line labels, so it should be *listed* even if
nobody downloads it by default. Add 5 when the non-NDJSON branch is actually being built, not
before. Add 6 only once dynamic-key-as-map handling is actually being built and needs a fixture —
at 500 lines it's too small and unlabeled to be a general-purpose test dataset, its only job is
requirement (i).

## 4. `datasets_json.yml` — yes, and it works today

A separate `downloader/datasets_json.yml` alongside the existing `datasets.yml` is the right call,
and it needs **no code change to download**. `download_data.py` reads only `root_folder` and, per
dataset, `name`, `download`, and `url`/`urls` — everything else in the YAML is consumed by the test
scripts. `--config` already exists and only defaults to the sibling `datasets.yml`:

```
uv run downloader/download_data.py --config downloader/datasets_json.yml
```

`tests/datasets.yml` is the precedent: a second config with the same schema and a different
purpose.

Things to know before writing the file:

- **Archive support is `.zip`, `.gz`, `.7z` — not `.tar`.** Candidate 5 needs either a small
  addition to the extractor or manual handling. Candidates 1–4 are all fine (raw file, zip, zip,
  zip).
- **A bare, unarchived file is fine.** Candidate 1 is just a file; the downloader saves it and
  skips extraction.
- **GitHub folder URLs already work.** `transform_github_url` + `clone_github_repo` +
  `move_github_folder` clone the repo and move a subpath — this is how the Nezha entry pulls two
  subfolders today, so candidate 2 can be fetched the same way (or as individual scenario ZIPs).
- **Re-running is safe.** `main()` skips any dataset whose target folder already exists.
- **Zenodo `?download=1` URLs are handled** — the filename is taken before the query string.
- **The test-suite keys are inert for now.** `log_file`, `labels_file`, `filename_pattern`,
  `expected_length`, `reduction_fraction`, `load`/`enhance`/`anomaly_detection` are read by
  `tests/loaders.py`, whose `create_correct_loader()` is an `if/elif` chain **dispatching on the
  dataset name to a hard-coded loader class**. So a JSON dataset cannot enter the test suite until
  a `JsonLoader` exists *and* gets a branch there. Write the keys in anyway — the file should not
  need re-editing later.
- **Add a `loader:`/`format:` key to these entries.** This is where `log-format-support.md` §5
  item 7 (a format registry) first pays for itself: the entries would name their loader/format
  instead of relying on a name-matching `if/elif`, and that name-to-loader lookup is the smallest
  possible first step toward a real format registry.

Follow-up worth doing once a loader exists: commit a small slice of candidate 1 (or one
Security-Datasets scenario) under `demo/samples/`, matching the existing
`hdfs_events_2percent.parquet` convention, so there is a `demo/JSON_samples.py` smoke test that
needs no download at all.

## 5. How to verify any claim in this document

| Claim | Where to look |
|---|---|
| nginx JSON logs: 12.1 MB / 51,462 lines / 8 flat keys | downloaded [the raw file](https://raw.githubusercontent.com/elastic/examples/master/Common%20Data%20Formats/nginx_json_logs/nginx_json_logs) directly; repo licence Apache-2.0 confirmed via GitHub API (`GET /repos/elastic/examples`) |
| Security-Datasets: per-scenario ZIPs 12 KB–1 MB, NDJSON with per-line key variation and `\r\n` inside `Message` | downloaded and unzipped [`cmd_lsass_memory_dumpert_syscalls.zip`](https://github.com/OTRF/Security-Datasets/blob/master/datasets/atomic/windows/credential_access/host/cmd_lsass_memory_dumpert_syscalls.zip) (118 lines); repo licence MIT confirmed via GitHub API, 1,796 stars |
| AIT-ADS: 96.2 MB, ~2.66M alerts, `labels.csv`, CC-BY-4.0 | [Zenodo record 8263181](https://zenodo.org/records/8263181); cross-checked against [AIT-ADS GitHub README](https://github.com/ait-aecid/alert-data-set) |
| AIT-LDSv2: 8 ZIPs 7.1–26.5 GB, 130.6 GB total, per-line labels, CC BY-NC-SA 4.0 | [Zenodo record 5789064](https://zenodo.org/records/5789064) |
| flaws.cloud CloudTrail: 240 MB `.tar`, `{"Records":[…]}` gzipped chunks | `HEAD https://summitroute.com/downloads/flaws_cloudtrail_logs.tar` → `Content-Length: 251688960`; description at the [Summit Route blog post](https://summitroute.com/blog/2020/10/09/public_dataset_of_cloudtrail_logs_from_flaws_cloud/) |
| nginx_json_plus_logs: 2.78 MB, 500 lines, ~5.1s cadence over ~42.5 min, 15 fixed top-level keys, `server_zones`/`upstreams` keyed by hostname/pool name, `max_conns` present on some `peers[]` entries but not siblings | downloaded [the raw file](https://raw.githubusercontent.com/elastic/examples/master/Common%20Data%20Formats/nginx_json_plus_logs/nginxplus_json_logs) directly and parsed every line with Python (`wc -l`, `json.loads` per line, diffed `timestamp` deltas and top-level key sets) |
| Downloader reads only `root_folder` + `name`/`download`/`url`\|`urls` | `downloader/download_data.py:238-266`; `--config` at `:271` |
| Extractor handles `.zip`/`.gz`/`.7z` only | `downloader/download_data.py:88-143`, dispatch at `:225-232` |
| Test suite dispatches loader by dataset name | `tests/loaders.py` — `create_correct_loader()` if/elif chain |

**Not independently verified — treat as a lead, not a fact:** a related repository,
[`ait-aecid/anomaly-detection-log-datasets`](https://github.com/ait-aecid/anomaly-detection-log-datasets),
turned up in search as "analysis scripts for log data sets used in anomaly detection" and may be
useful alongside AIT-ADS/AIT-LDSv2, but I did not fetch it and it is not otherwise referenced above.
