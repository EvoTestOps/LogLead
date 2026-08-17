# Broadening LogLead's log format support

**Status:** living document — the original proposal plus a record of what has since been built
against it. Items 1–6 of §5 are delivered and are marked as such inline; the agreed shortlist is
complete, and what remains is item 8 (`from <field>`), the catalogue, and reaching `delta`/MCP.
§8 is the current scorecard and the answer to "what next".
**Question:** `lnav` and `angle-grinder` can read log formats that LogLead cannot — most obviously
JSON. What exactly do they support, and in what order should LogLead adopt the same capabilities?

**Evidence base:** local clones of both tools, read directly rather than from their websites:

| Tool | Local path | Files that define format support |
|---|---|---|
| lnav | `~/lnav` | `src/formats/*.json` (73 definitions), `src/log_format_impls.cc` (5 hard-coded formats), `src/file_format.hh`, `src/time_formats.am`, `docs/schemas/format-v1.schema.json`, `docs/source/formats.rst`, `ARCHITECTURE.md` |
| angle-grinder | `~/angle-grinder` | `src/lang.rs` (the `json`/`logfmt` inline operators), `src/operator/` (one file per aggregating operator), `aliases/*.toml`, `README.md` |

Both clones were re-read when this document was last revised (lnav `bb0005f9`, 2026-07-21;
angle-grinder `9c2fc88`, 2026-02-05). Counts below were re-verified against them rather than carried
forward; where a number has drifted since the first draft it is noted.

**Companion documents**, both picking up where §5 item 1 (JSON lines) leaves off:
[log-format-json-testdata.md](log-format-json-testdata.md) — candidate public datasets to develop
and test a `JsonLoader` against, with links to each dataset's description page;
[log-format-json-loader.md](log-format-json-loader.md) — the design for that loader, as one
config-driven class plus format specs as data (a first instance of §5 item 7).

---

## 1. What LogLead has today

### 1.1 How the loader layer actually works

LogLead is a three-stage pipeline — **Loader → Enhancer → AnomalyDetector** — joined by Polars
DataFrames and a column-naming convention. The loader's entire job is to turn "whatever this
dataset looks like on disk" into that convention, so that nothing downstream needs to know which
dataset it is looking at.

**The contract.** `BaseLoader` is a template-method base class. A subclass implements only two
methods, `load()` (read bytes into a DataFrame) and `preprocess()` (dataset-specific cleanup).
`execute()` then drives a fixed sequence:

```
load() → preprocess() → check_for_nulls_and_non_utf8() → check_mandatory_columns() → add_ano_col()
```

`check_for_nulls_and_non_utf8()` only *warns* (it prints a count of nulls and of `U+FFFD`
replacement characters per column); `check_mandatory_columns()` raises. `add_ano_col()` derives
whichever of the `normal`/`anomaly` boolean columns is missing from the other, so downstream code
can rely on both.

**Two frames, one convention.** A loader populates `self.df` (one row per log event) and, only for
sequence-labelled datasets, `self.df_seq` (one row per sequence). Columns are prefixed by origin:
`m_*` are mandatory/raw columns from the loader, `e_*` are added by `EventLogEnhancer`, `seq_*` by
`SequenceEnhancer`. The default mandatory set is `m_message` + `m_timestamp`; loaders whose data has
no clock override it (`ADFALoader` and `AWSCTDLoader` require only `m_message`, `RawLoader` requires
nothing).

**How lines get read.** There is one trick that explains most of the loader code. Every text loader
reads log files through Polars' *CSV* reader with the separator set to `\a` (the BEL control
character) — a byte chosen precisely because it will never occur in a log. The CSV engine therefore
never splits anything, and is used purely as a fast, parallel, memory-mapped line reader: each log
line arrives whole in a single column. The rest of the settings exist for the same reason —
`has_header=False`, `quote_char=None` (so a stray `"` cannot swallow the next 10,000 lines),
`infer_schema_length=0` or an explicit all-`String` schema, `encoding="utf8-lossy"`,
`truncate_ragged_lines=True`.

**How fields get extracted.** The only field-splitting facility in the base class is
`_split_and_unnest(field_names)`: split the line on single spaces into exactly *n* pieces, name them
positionally, unnest into columns. That is the whole parser. BGL, Thunderbird/Spirit/Liberty and
ProLoader are each one call to it with a different list of ten-ish field names. Anything beyond a
fixed positional layout is done ad hoc with Polars string expressions — `str.extract` with a regex
(HDFS block ids, Hadoop's `[process]`), `splitn` on a delimiter (`component[pid]`),
`str.replace_all` to collapse runs of whitespace.

**How multiple files get read.** `os.walk` + `glob` to enumerate, one lazy `pl.scan_csv` per file,
`pl.collect_all()` to run them in parallel, `pl.concat` to stack. `include_file_paths` adds a
`file_name` column, which is what the log-folder comparison code (`loglead/delta/`) later splits
into `folder` + `file_name`.

**How the awkward parts get handled.** Everything else is per-loader and hard-coded:

- *Timestamps* — each loader hard-codes its own `strptime` pattern, or converts an epoch integer.
  `RawLoader` is the only configurable one: the caller supplies a regex (`timestamp_pattern`, capture
  group 1) *and* a matching `timestamp_format`. Nothing is detected.
- *Multi-line records* (stack traces) — solved twice, differently. `RawLoader` has
  `missing_timestamp_action='merge'`, which flags lines that produced a timestamp, cumulative-sums
  the flag into group ids, and folds the untimestamped continuation lines into a `trace` column.
  `HadoopLoader` has its own `_merge_multiline_entries()`.
- *Labels* — from a separate label file (HDFS, Hadoop), from a marker in the first field
  (`-` means normal in BGL and Thunderbird), or from directory/file names (Pro, ADFA, AWSCTD, LO2).

### 1.2 The loaders

Fifteen loaders, in three groups rather than the two this section originally described:

- **Dataset-driven** (10) — one Python class per dataset: `HDFSLoader`, `HadoopLoader`, `BGLLoader`,
  `ThuSpiLibLoader`, `ProLoader`, `NezhaLoader`, `LO2Loader`, `ADFALoader`, `AWSCTDLoader`, plus
  `RawLoader` as the general-purpose fallback.
- **Format-family** (4) — one class per *format family* rather than per dataset. Two are
  **spec-driven**, configured by a YAML spec (`JsonLoader`, `AccessLogLoader`; §5 items 1 and 4);
  two ship **no spec directory on purpose** (`LogfmtLoader`, `SyslogLoader`; §5 items 2 and 5),
  because their field names are fixed by convention or by an RFC and so there is nothing per-dataset
  left to configure. That distinction is the useful one: a spec directory is what a family needs
  when the *mapping* varies between datasets, and logfmt and syslog are the cases where it does not.
- **Detecting** (1) — `AutoLoader` (§5 item 6), which picks one of the above and builds it.

All four format-family loaders, and `AutoLoader`, were delivered after this analysis was written.

| Loader | Reads | Physical format | Notes |
|---|---|---|---|
| `RawLoader` | any file or directory tree, one event per line | line-oriented plain text | The only general-purpose loader. Optional user-supplied timestamp regex; no labels. This is the loader behind `loglead/delta/` and therefore behind every MCP tool. |
| `HDFSLoader` | one large log + a separate CSV label file | positional whitespace text | `seq_id` extracted with a `blk_…` regex |
| `HadoopLoader` | tree of `application_*/container_*/*.log` | positional text + multiline merge | labels from a separate file |
| `BGLLoader` | one file | positional whitespace text, 10 fields | label = first field |
| `ThuSpiLibLoader` | one file (Thunderbird / Spirit / Liberty) | positional whitespace text, 10 fields | additionally splits `component[pid]` |
| `ProLoader` | directory of files | positional whitespace text | label from the file name |
| `NezhaLoader` | directory of CSVs + JSONs (logs, traces, metrics, ground truth) | header CSV + JSON | by far the largest loader (~560 lines); entirely dataset-specific |
| `LO2Loader` | tree of runs/services | text logs + JSON metrics | builds `seq_id` from run/test/service |
| `ADFALoader` | directories of `.txt` | whitespace-separated syscall **ids** | not log text; already "parsed" |
| `AWSCTDLoader` | directories of `.csv` | newline-separated syscall names, one sequence per file | not log text |
| `JsonLoader` | one file, or a directory tree via `filename_pattern` | JSON lines, or a top-level array / wrapped-object container | spec-driven — `loglead/loaders/json_formats/*.yml` supplies the field mapping; see §5 item 1 |
| `AccessLogLoader` | one file, or a directory tree via `filename_pattern` | positional text (Apache/nginx Common/Combined Log Format) | spec-driven — `loglead/loaders/access_log_formats/*.yml` is an nginx `log_format` string compiled to a regex; see §5 item 4 |
| `LogfmtLoader` | one file, or a directory tree via `filename_pattern` | `key=value` logfmt text | no spec directory — the key names are conventional, so the mapping is built in; see §5 item 2 |
| `SyslogLoader` | one file, or a directory tree via `filename_pattern` | syslog, RFC 3164 / RFC 5424, chosen **per file** | no spec directory — two layouts, both defined by an RFC; `multiline` handles continuation lines; see §5 item 5 |
| `AutoLoader` | anything the loaders above read | decided per file from a sample | detects the format and **builds one of the other loaders**; parses nothing itself. See §5 item 6 |

### 1.3 The gap, stated precisely

*As first written, this section said:* LogLead can read exactly two things — **text that can be
split positionally on spaces**, and **whatever a person has written a bespoke Python class for**.
Every format that dominates modern production logging — JSON lines, logfmt, CSV/TSV with a header,
syslog, web access logs, journald, Windows events — requires a new hand-written loader.

**That is no longer the gap.** JSON lines, logfmt, syslog, web access logs and delimited text with a
header each have a format-family loader, and `AutoLoader` picks between them without being told.
journald's JSON output falls out of `JsonLoader`, as item 5 predicted it would. What remains, stated
precisely:

- **Parsers cannot be re-run against a column** (§5 item 8). A Kubernetes JSON envelope wrapping an
  app line needs two passes, and every parser LogLead has takes a file, not a `Series`.
- **The catalogue is small.** The *mechanisms* now match the two reference tools; the inventory of
  known formats does not. Thirteen shipped specs against lnav's 73 definitions (§8).
- **None of it reaches `delta`/MCP** (§6, last bullet). `log_root.py` still calls `RawLoader`
  directly, so the 19 MCP tools remain plain-text-only regardless of what the loader layer can do.

One specific worth recording because it changes the priorities below:

- **Compression is already fine.** Polars transparently decompresses `.gz` in both `read_csv` and
  `scan_csv` (verified against the pinned polars 1.38.1), so gzipped logs already work today —
  provided the caller's `filename_pattern` matches the `.gz` extension. This is *not* a gap; don't
  spend effort here.

---

## 2. What lnav supports

lnav's stated design goal, first line of its `ARCHITECTURE.md`, is *"Don't make the user do
something that can be done automatically"*, with format detection given as the example. Its format
support has two layers.

### 2.1 The catalogue: 73 declarative + 5 built-in

**73 formats ship as JSON data files** in `src/formats/`, not as code. Of those, **59 are
regex-based text formats** and **14 are JSON-lines formats**. Only **5 formats needed to be written
in C++** (`src/log_format_impls.cc`), and they are exactly the ones a regex cannot express.

| Family | Examples from `src/formats/` |
|---|---|
| Web / CDN / load balancer (≈12) | `access_log` (Common Log Format), `error_log`, `alb_log`, `elb_log`, `s3_log`, `cloudflare`, `caddy`, `haproxy`, `uwsgi`, `page_log`, `web_robot` |
| Databases | `mysql_error`, `mysql_gen`, `mysql_slow`, `postgres`, `vpostgres`, `redis`, `mongodb` |
| App / language runtimes | `java_log`, `glog` (C++), `rails`, `laravel`, `openstack`, `zookeeper`, `nextflow`, `spdlog`, `env_logger`, `rust_tracing`, `zap_console`, `idea_log` |
| OS / infrastructure | `syslog`, `journald_json`, `dpkg`, `cups`, `sssd`, `strace`, `logcat` (Android), `macosuni` (macOS unified log), `esx_syslog`, `vmk`, `unifi`, `unifi_iptables` |
| Structured / observability | `ecs` (Elastic Common Schema), `otel_collector`, `otlp_python`, `bunyan`, `pino`, `github_events`, `pcap` |

**The 5 hard-coded formats are the interesting ones**, because they are the generic mechanisms
rather than vendor entries:

1. **`generic_log`** — the fallback. About fifteen ordered regexes that between them say little more
   than *"a leading timestamp-shaped token, optionally followed by a level word"*. If one matches,
   the file is a log; if none does, it is plain text. This is what makes lnav useful on a file
   nobody has ever written a format for.
2. **`logfmt_log`** — the `key=value` convention. Recognizes `timestamp`/`time`/`ts`/`t` as the time,
   `level`/`lvl` as the level, `message`/`msg` as the body; every other key goes into a `fields`
   JSON blob. Requires the *whole* line to be key/value pairs.
3. **`bro_log`** — Zeek/Bro TSV. **Self-describing**: lnav reads the `#fields` header line and
   derives the schema from the file itself. Zero configuration.
4. **`w3c_log`** — W3C Extended Log File Format (IIS et al). Also self-describing via its
   `#Fields:` directive.
5. **`piper_log`** — lnav's own capture format.

Above the log-format layer there is a **container/file-format layer** (`file_format.hh`):
`SQLITE_DB`, `ARCHIVE`, `MULTIPLEXED`, `REMOTE`, plus a **converter** hook. A format definition can
declare a magic-byte test over the first N bytes and a shell command to convert the file into
something readable — `pcap_log` sniffs pcap/pcapng/snoop/btsnoop magic and shells out to a
tshark-based script; `otel_collector_log` does the same. So binary support is delegated, not
implemented.

### 2.2 How auto-detection works

This is the part worth copying, and it is deliberately cheap:

1. **Container sniff.** Read the first bytes; decide SQLite / archive / pcap / remote / plain. Magic
   tests are declared *in the format file*, as an expression over a fixed-size header.
2. **Line match.** Read the first few lines and try every format's regex set until one matches.
   Once a format matches, it is **locked in for that whole file** — detection is not re-run per
   line. `max-unrecognized-lines` bounds how much leading noise is tolerated. No match → plain text.
3. **Timestamp parse.** All formats need a timestamp, so this must be fast. `strptime(3)` was too
   slow, so lnav *generates* a parser at compile time (the `ptimec` component) over a fixed list of
   **109 distinct strptime-style patterns** in `src/time_formats.am` (107 when this was first
   written — the list grows, which is itself the point: it is data, not code).
4. **Samples double as tests.** Every format definition carries a `sample` array of real log lines
   that must parse. Contributing a format is submitting data, not code.

### 2.3 What a format definition contains

From `docs/schemas/format-v1.schema.json` — note how much of it is *semantics*, not splitting:

- **Matching:** `regex` (named captures), `file-type` (`text`/`json`), `file-pattern` (restrict by
  filename), `multiline`, `max-unrecognized-lines`, `sample`.
- **Field semantics:** `timestamp-field`, `timestamp-format`, `level-field` + `level` map,
  `body-field`, `opid-field` (operation/correlation id), `thread-id-field`, `duration-field`,
  `src-file-field`/`src-line-field`, `subsecond-field`.
- **Typing:** a `value` object declaring the type of each extracted field.
- **JSON logs:** `line-format` — how to render the object back as a readable line — plus
  `hide-extra` and JSON-pointer paths (`log/logger`) for nested keys.
- **Other:** `converter`, `highlights`, `tags`, `partitions`, `search-table`, `url`, `description`.

The lesson: a format declares **which field is the time, which is the message, which is the level,
which correlates records** — precisely the mapping LogLead performs implicitly when it names a
column `m_timestamp`.

---

## 3. What angle-grinder supports

angle-grinder is a different kind of tool and it matters for how one reads its format list: it is a
**query language over one line-oriented stream** (stdin, or a single file — see `src/bin/agrind.rs`).
It has no format catalogue and no detection at all. Instead the user names the parser as the first
operator in a pipeline, and the tool's advertised first example is exactly a format question:
`agrind '* | json | count by log_level'`.

Its parsers, from `src/operator/`:

| Operator | What it does |
|---|---|
| `json [from <field>]` | Parse the line as JSON. Nested access with `.key[index]`, including negative indices. Non-JSON lines are dropped unless kept explicitly. |
| `logfmt [from <field>]` | Parse `key=value` pairs (the Heroku/Splunk convention). |
| `split [(<field>)] [on <sep>] [as <new>]` | Split on a separator (default `,`) into an array — i.e. CSV/TSV/whitespace, without a header. |
| `parse "* pat * pat *" as a,b,c [nodrop] [noconvert]` | Glob-style positional extraction. `*` ≈ `.*`. Whitespace in the pattern matches any whitespace. |
| `parse regex "(?P<name>…)" [from <field>] [nodrop]` | Rust-syntax regex, **named captures only**. |

Note that `json` and `logfmt` are not files in `src/operator/` — that directory holds the
aggregating operators. Both are `InlineOperator` variants defined in `src/lang.rs`, which is where
to look to check their behaviour.

Plus **aliases** (`aliases/*.toml`): named, reusable pipelines discoverable by keyword. Tellingly,
every shipped alias that is not a test fixture is a *format definition expressed as a `parse`
pattern* — `apache`, `nginx`, `k8s-ingress-nginx` — each one a positional extraction of a
Common/Combined Log Format line into named fields. (There are four files; the fourth,
`multi-operator.toml`, is a `json | count` test fixture keyed `testmultioperator`, not a format.)

Three transferable lessons:

1. **A handful of composable parsers covers most real logs.** JSON + logfmt + delimited + glob
   pattern + regex, with no per-vendor catalogue, is enough for a very popular tool.
2. **`from <field>` matters.** Real logs are layered: a Kubernetes JSON envelope around an app line,
   a syslog frame around a JSON payload, a JSON field containing logfmt. angle-grinder's documented
   example is literally `json | logfmt from nested_key`. Parsing must be re-runnable against an
   existing column, not only against the raw line.
3. **Extraction converts types.** Numbers become numbers, so aggregation works — and `noconvert` is
   an explicit opt-out, which tells you it is the default on purpose.

---

## 4. Where the two tools agree

The intersection is a short list, and it is the shortlist LogLead should work from.

| Format | lnav | angle-grinder | LogLead today |
|---|---|---|---|
| **JSON lines / NDJSON** | 14 built-in formats + declarative support | `json` operator (its headline feature) | ✅ `JsonLoader` (§5 item 1) |
| **logfmt** | built-in C++ format | `logfmt` operator | ✅ `LogfmtLoader` (§5 item 2) |
| **Delimited + header (CSV/TSV)** | Zeek TSV + W3C ELF, both self-describing | `split` | ✅ `DelimitedLoader` (§5 item 3), incl. both self-describing variants |
| **Web access logs (CLF/Combined)** | ~12 formats | all 3 shipped format aliases | ✅ `AccessLogLoader` (§5 item 4) |
| **Syslog** | built-in format | — | ✅ `SyslogLoader` (§5 item 5) |
| **Generic timestamped text** | `generic_log` fallback + 109 timestamp patterns | `parse` / `parse regex` | ✅ `AutoLoader`'s generic pass — 9 patterns tried automatically; `RawLoader` still takes a user-supplied regex |
| **Format auto-detection** | the whole design goal | none — the user names the parser | ✅ `AutoLoader` (§5 item 6), incl. a dataset probe neither tool has |
| **Parse from an existing field** | — | `json \| logfmt from <field>` | ❌ (§5 item 8) |
| **Binary containers** (pcap, SQLite, archives) | container layer + converter subprocesses | — | ❌ deliberately deprioritized |

---

## 5. Recommended order of adoption

Ordering criteria: (a) share of real-world log volume in that format, (b) how much of *LogLead's*
pipeline the format unlocks — does it hand us a timestamp, a level, a sequence id? — (c) cost
against the existing Polars architecture, (d) whether it reaches the `loglead/delta/` + MCP path,
which today only ever sees `RawLoader` output.

### 1. JSON lines (NDJSON), done properly — **highest value, low effort**

The default output of essentially every modern logging library and platform: Bunyan, pino, zap,
Serilog, `python-json-logger`, Docker's `json-file` driver, Kubernetes, Elastic ECS, OpenTelemetry,
GELF, `journalctl -o json`, and the log exports of Cloudflare/AWS. lnav ships 14 such formats;
angle-grinder's first documented example is a JSON one.

Build a `JsonLoader` on Polars' native `read_ndjson`/`scan_ndjson` (multithreaded, schema-unifying,
same lazy/`collect_all` pattern as the existing loaders), with:

- a **field mapping** argument: which key is the message, the timestamp, the level, the sequence id;
- a **nested-key path** syntax — lnav uses `/`-separated JSON pointers (`log/logger`), angle-grinder
  uses `.key[i]`; pick one and document it;
- a **policy for the remaining keys**: keep as columns, fold into `m_message`, or drop.

Payoff beyond the obvious: Kubernetes/OTel logs carry `trace_id`/`pod`/`container` fields, which
give `SequenceEnhancer` a *real* `seq_id` — something the current dataset-specific loaders have to
manufacture with regexes.

**Delivered as `JsonLoader`** (`loglead/loaders/json.py`), built on `scan_ndjson`/`read_ndjson`
exactly as proposed, plus the two other container shapes Polars gives for free: `array` (a
top-level `[...]`) and `wrapped` (an object holding the records, e.g. CloudTrail's
`{"Records": [...]}`). The field mapping is `timestamp_field`/`message_field`/`level_field`/
`seq_id_field`, with `line_format` covering records where no single key is the message (an access
log delivered as JSON). Nested-key path syntax landed as lnav's `/`-separated JSON pointers rather
than angle-grinder's `.key[i]` (`path_separator`, default `/`) — the choice left open above,
decided because a real dataset in the test data has literal dots inside its own keys. `extra_fields`
(`keep`/`drop`) is the remaining-keys policy. Format specs
(`loglead/loaders/json_formats/*.yml`: `nginx_json`, `otrf_winevent`, `nginx_plus_status`, `gelf`)
are exactly what item 7 below calls for — the loader's keyword arguments are the spec keys, so a
spec file is a serialized constructor call. Tested against three of the candidates in
[log-format-json-testdata.md](log-format-json-testdata.md) (nginx access logs, OTRF
Security-Datasets, AIT-ADS); design detail and measurements in
[log-format-json-loader.md](log-format-json-loader.md).

### 2. logfmt — **cheapest win on the list**

Both tools implement it as a first-class format. It is the Go/Heroku ecosystem's default and is what
Grafana/Loki, Docker, Consul, etcd and much of the Prometheus tooling emit. The grammar is strict
and trivially parseable — one regex plus Polars string operations, no new dependency. Better, the
key names are conventional, so lnav's mapping (`timestamp|time|ts|t` → time, `level|lvl` → level,
`message|msg` → body) can be adopted wholesale and the loader needs **zero configuration** in the
common case.

**Delivered as `LogfmtLoader`** (`loglead/loaders/logfmt.py`), and the "zero configuration" bet held:
lnav's mapping was adopted wholesale, which makes this the first family loader with **no spec
directory** — keyword arguments to override the mapping exist and read like the other two, but no
shipped dataset needs them. Two things the proposal did not anticipate, both from real Grafana
output: candidate keys are **coalesced rather than first-wins**, because one file routinely mixes
`t=` and `ts=` lines from two components; and each key becomes its own column, so a tree of files
lands wide-and-sparse the way heterogeneous JSON does. Tested against Grafana Labs' own production
logs from four services, committed to `grafana/loki` as its Drain test data
(`tests/datasets_fmt.yml`, 56,100 events). One result carries beyond logfmt: lnav's *anchored* logfmt
test turns out to be wrong on real data, and only measurement caught it — see item 6.

### 3. Delimited text with a header (CSV/TSV), including the self-describing variants

How exported observability data usually arrives: SIEM exports, audit logs, Zeek, W3C/IIS — and,
directly relevant to this project, the `*_structured.csv` files that loghub-style datasets ship as
their already-parsed form. LogLead already depends on Polars' CSV reader, so the work is the
*mapping layer*, not parsing. The self-describing variants are the cheapest auto-detection win
available anywhere on this list: Zeek's `#fields` line and W3C's `#Fields:` directive mean the
column names come out of the file, so there is nothing for the user to configure. Both are already
in lnav and can be copied nearly verbatim as specs.

**Delivered as `DelimitedLoader`** (`loglead/loaders/delimited.py`), the third spec-driven loader,
reusing JsonLoader's field-semantics vocabulary as AccessLogLoader did. The prediction that the work
is the mapping layer rather than the parsing held, with one addition this item did not anticipate:
the loader's real question is *where the column names come from*, and there are three answers, so
that is the `header` argument (`row`/`zeek`/`w3c`, decided per file, plus `none` and `auto`). The
self-describing variants did turn out to be the cheapest win claimed — Zeek's header declares not
only the names but the delimiter, the null markers **and** the column types, so `#types` is read as
well and duration/byte counts reach `AnomalyDetector(numeric_cols=...)` as numbers with nothing
configured. Shipped specs (`loglead/loaders/delimited_formats/*.yml`: `zeek`, `zeek_labeled`, `iis`,
`loghub`, `loghub_labeled`) only supply what a file cannot say about itself: which column is the
message, which is the timestamp and in what encoding, and which is a label.

Labels are the part that generalized beyond this item: this is the one format family that routinely
arrives labelled — loghub keeps the alert category in `Label`, IoT-23 appends `label` to Zeek's
conn.log — so `label_field`/`normal_values` produce the `normal` column directly, and the label
column is kept out of the rendered `m_message`, which otherwise hands the ground truth to every text
detector. Tested against four corpora in `tests/datasets_csv_tsv.yml`, one per header style: loghub's
16 `*_structured.csv` files in one folder (32,000 rows, 15 distinct headers, so 15 mappings rather
than one), Brim's 35-log-type Zeek output directory (1,474,104 events, 342 columns after the
diagonal concat), IoT-23's labelled conn.log (23,145 events), and Splunk Attack Range IIS logs
(55,826). Two measured results worth keeping: the raw-BGL cross-check — the same events read as a
structured CSV score F1 0.815 from `Type`/`Level`/`Component` against 0.844 through `BGLLoader` —
and the header-row detection rule, where an exact-field-count test scores 1.000 on loghub's Apache
CSV and **0.000** on its Hadoop one, whose `Content` always contains a comma, so the rule is "at
least as many delimiters as the header, and header cells that look like names". **Not done:** a
`#Fields:` redeclaration that changes the field set mid-file (the first one found locks the file,
lnav's rule), Zeek's JSON output (that is a `JsonLoader` spec), and reading `#types` beyond scalars —
a set or vector column stays raw text.

### 4. Web access logs: Common/Combined Log Format and relatives

The most common non-structured format on the internet, and the one place both tools invest most
(lnav: ~12 formats; angle-grinder: all three of its shipped format aliases). For LogLead specifically it is
attractive for a reason beyond ubiquity: access logs have a **natural sequence id** (client IP,
session, or request id) and a **natural anomaly signal** (status codes, byte counts, latency), so
they exercise the loader → enhancer → sequence → detector chain end to end. This is also the most
plausible route to a demo dataset that is neither a 15-year-old HDFS dump nor a supercomputer log.
Cover Apache access + error, nginx, and the AWS variants (ALB/ELB/S3/CloudFront) — all positional,
all expressible as one regex each.

**Delivered as `AccessLogLoader`** (`loglead/loaders/access_log.py`), the second spec-driven loader
after item 1's `JsonLoader`, reusing that loader's field-semantics vocabulary
(`timestamp_field`/`message_field`/`line_format`/`level_field`/`seq_id_field`) rather than
inventing a parallel one. Where a JSON format spec is a field mapping, an access-log format spec is
one regex: the shipped specs (`loglead/loaders/access_log_formats/*.yml` — `common`, `combined`,
`combined_xff`, `vhost_combined`) write it as an nginx `log_format` string
(`'$remote_addr - - [$time_local] "$request" $status $body_bytes_sent ...'`), which the loader
compiles to named captures rather than requiring a hand-written regex per format — the same "spec
is a serialized constructor call" property item 7 argues for generally. `$request` is additionally
split into `method`/`path`/`protocol`, and `status`/byte counts land as typed numbers, so the
natural anomaly signal this section opens with reaches `AnomalyDetector(numeric_cols=...)` with no
manual cast — the typed-columns point in §6 below, applied. Tested against the
[Kaggle "Web Server Access Logs"](https://www.kaggle.com/datasets/eliasdabbas/web-server-access-logs)
dataset (`tests/datasets_access_log.yml`): 10,365,152 lines, 100% matching the `combined_xff` spec.
Not yet done: Apache error logs and the AWS variants (ALB/ELB/S3/CloudFront) have no spec yet, since
nothing in the test data exercises them.

### 5. Syslog (RFC 3164 and RFC 5424) and journald

Still the substrate of Linux and network-device logging; lnav treats syslog as a core format. RFC
5424 additionally brings `procid`, `msgid` and structured-data elements, which map cleanly onto
LogLead columns and give another free correlation id. journald's JSON output falls out of item 1 for
free *once the field mapping is configurable* — which is an argument for doing item 1 properly
rather than hard-coding one JSON shape.

**Delivered as `SyslogLoader`** (`loglead/loaders/syslog.py`). Like item 2 it ships **no spec
directory**, but for the opposite reason: syslog has exactly two layouts and both are defined by an
RFC, so `rfc3164` and `rfc5424` are built-in regexes with `pattern=` as the escape hatch. Which one
applies is decided **per file** from its first lines — lnav's own rule — so a directory holding both
reads in one call; a load that mixes them runs two vectorized parses and coalesces, since no single
`strptime` covers both. Two consequences worth knowing before editing it. RFC 3164 carries **no
year**, so `m_timestamp` is built by prepending `year=` (current year by default) — the three test
files were recorded in 2005, 2016 and 2017, so no single year would be right for all of them and
only ordering within a file survives. And a non-matching line is normally the second line of a
multi-line message rather than garbage, so it is `multiline` that handles it
(`merge-message` default / `merge-add-column` / `keep` / `drop` / `raise`), with `min_match_rate` —
not the first bad line — as what catches a genuinely wrong format. Merging is not cosmetic: it takes
Drain from 901 templates to 647 on the macOS file, the difference being stack-trace fragments that
were being mined as if they were events. Tested against three loghub corpora in one directory
(`tests/datasets_syslog.yml`: Linux, macOS, OpenSSH — 797,997 lines in, 787,859 events out after
merging). **Not done:** journald specifically, which needs no syslog work — it is a `JsonLoader`
spec nobody has written, and there is no test corpus for it. RFC 5424 structured-data elements
(`[exampleSDID@32473 iut="3"]`) are captured as one raw string rather than exploded into columns.

### 6. A generic "timestamped line" detector — LogLead's own `generic_log`

The biggest usability win, placed here rather than first because it *depends* on the branches above
existing, and because it needs a design decision more than it needs a parser. Concretely: given an
unknown file, try in order — is it JSON lines? logfmt? delimited-with-header? does the leading token
parse as a timestamp under any of N known patterns? — set `m_timestamp`/`m_message` accordingly, and
fall back to today's `RawLoader` behaviour if nothing matches. Follow lnav's ordering: container
sniff → match on the first few lines → **lock the format for the file** → plain text on failure.

Two implementation constraints specific to LogLead:

- lnav needed a code-generated timestamp parser because repeated `strptime` attempts are expensive.
  The Polars equivalent is: **run the trial on a sample of lines, then apply one vectorized
  `strptime` to the entire column.** Never per row. `JsonLoader` and `AccessLogLoader` already
  follow this (`_pick_format()` in each), so a future detector can reuse the same helper.
- Detection is inherently **per file**, but LogLead's `delta`/MCP path loads a whole directory tree
  in one call. The detection result must therefore be resolved per `file_name` group, not once per
  load — a mixed-format log folder is the normal case, not the exception.

**Delivered as `AutoLoader`** (`loglead/loaders/auto.py`). It **builds one of the other loaders and
never parses anything itself**, which is what log-format-json-loader.md §2 asked for — detection is
a layer *on top of* the spec system that picks a spec name — and why `detect_format()` is importable
on its own: the decision is inspectable without loading. Both constraints above are met: candidates
are scored on a sample and the winner runs one vectorized pass, and detection is resolved per file,
with a tree whose files all agree delegated to a single loader (keeping its parallel multi-file
read) and a genuinely mixed folder loaded one file at a time and stacked with `diagonal_relaxed`.

The order shipped is a file that declares its own columns (Zeek's `#separator`, W3C's `#Fields:`) →
JSON → access log → syslog → logfmt → a delimited file with a header row → generic timestamped text
→ `RawLoader`, which differs from the list above in two ways. **Delimited-with-header is split in
two rather than tried once**, at opposite ends of the chain: a file that names its own columns has
stated what it is and is the most specific evidence available, while "the first line looks like a
header" is the weakest and goes almost last. And there is a **stage before all of it** that this
section did not anticipate — a *dataset probe* recognizing the public datasets that have their own
loader, from the label file and directory layout beside the log rather than from the log alone,
because `HDFSLoader` needs its `anomaly_label.csv` and `HadoopLoader` its `abnormal_label.txt`. That
is what makes auto-loading a labelled dataset yield a real `df_seq` instead of bare lines; a dataset
whose labels are absent is deliberately not claimed, and falls through to the format probe. It
covers HDFS, Hadoop, ADFA, AWSCTD, Nezha, LO2, Profilence, BGL and Thunderbird/Spirit/Liberty.
**Neither reference tool has this stage**, because neither has a reason to: lnav identifies a
*format* so it can display a file, while LogLead identifies a *dataset* so it can attach labels.
LO2 is the case that shows what that means in practice — its label is not in a file at all but in
the name of a test-case directory (`correct` versus the error injected), so the probe looks for the
directory layout and nothing in the log lines would ever reveal it.

Three things were measured rather than reasoned, and each contradicts the obvious implementation:

- **The logfmt test counts `>=2` `key=value` pairs per line and must not be anchored at `^`.**
  Anchoring looks like the right way to stop syslog lines (which do contain `key=value`) from being
  read as logfmt. On the shipped logfmt corpus it scores 0.996 at a 1,000-line sample and **0.199 at
  5,000** — real Grafana lines put free text before the pairs — and 0.000 on the very common
  `<timestamp> level=info msg="…"` shape. The syslog false positive it was meant to prevent is
  already harmless: unanchored, syslog scores 0.16–0.20, well under `min_match_rate`.
- **The sample is the file's head plus a chunk from its middle**, because that same file's head is
  not representative of it — which is what hid the bug above. A `seek` and one 256 KB read costs
  milliseconds even on a multi-gigabyte log.
- **`m_timestamp` is normalized to naive microseconds afterwards.** Polars takes the time unit from
  the format string, so a `%3f` pattern yields `Datetime('ms')` while every other loader yields
  `'us'`, and the two will not `pl.concat` — the same trap as timezone-awareness (D5 in
  log-format-json-loader.md), reached by a different route. A loader whose output dtype depends on
  which format happened to be detected would export that trap to its caller.

The `m_timestamp`-mandatory question from §6 is answered the way `JsonLoader` answered it: per
instance, `["m_message"]` only, since a plain-text fallback legitimately has no clock. §6's encoding
requirement is met by `detections()`, which reports the undecodable-character count per file so that
a mis-decoded file and a mis-detected format do not look alike. Not yet done: reaching `delta`/MCP,
which is blocked on `read_folders()` consuming `orig_file_name`, a column only `RawLoader` produces.

### 7. Declarative format definitions (a format registry) — the real architectural lesson

lnav supports 73 formats with 5 C++ classes because **its formats are data**: a JSON file with a
regex, a field-semantics mapping, timestamp formats, and sample lines that double as tests. LogLead
currently spends a Python class per format, which is why it has eleven loaders and ten of them are
dataset-specific.

A YAML/JSON format spec — name, file pattern, line regex or structured type, which capture becomes
`m_timestamp`/`m_message`/level/`seq_id`, candidate timestamp formats, sample lines — would let
LogLead absorb new formats without code, and let users add in-house formats without forking. Items
1–3 provide the structured branches; the registry covers the regex branch and everything after.
It also simplifies the MCP server: a format becomes a *parameter* instead of a code path.

### 8. Nested / embedded parsing (`from <field>`)

angle-grinder's `json | logfmt from nested_key` exists because real logs are layered. Once the
parsers from items 1–3 exist as callable pieces, letting them run against an existing column instead
of the raw line is a small addition with disproportionate reach. Note that LogLead already has the
right home for this: it is an **enhancer** operation producing `e_*` columns, not a loader concern —
which keeps the loader layer simple and makes the capability composable and chainable like the rest
of `EventLogEnhancer`.

### Explicitly deprioritized

- **Binary containers** — pcap, SQLite databases, archives, Windows EVTX. lnav supports these by
  sniffing magic bytes and shelling out to a converter (pcap → tshark). Each one drags in an external
  tool for little anomaly-detection value. If ever wanted, copy lnav's *converter* idea — declare a
  command that converts to an already-supported format — rather than writing readers.
- **The per-vendor long tail** — VMware, UniFi, Katello, SnapLogic, Proxifier, and so on. That
  catalogue exists because lnav is an operator's tool for whatever is on the box. LogLead's users
  bring their own logs; the answer is the registry (item 7), not sixty hand-written loaders.
- **angle-grinder's query and aggregation language** — not a format concern. Polars already covers it.

---

## 6. Cross-cutting issues to settle while doing the above

These are not formats, but they will each be hit repeatedly during items 1–6 and are cheaper to
decide once. Four of the six are now settled; the two that are not are called out as **open**.

- **Typed columns.** ✅ *Settled.* Both tools convert on extraction — angle-grinder documents
  `noconvert` as the *opt-out*, which shows it is deliberate. LogLead's older loaders keep everything
  as `Utf8` (`infer_schema_length=0`); the new family loaders do not. `AccessLogLoader` lands
  `status` and the byte counts as numbers, so the natural anomaly signal reaches
  `AnomalyDetector(numeric_cols=...)` with no manual cast. The old loaders were not retrofitted, so
  the convention holds going forward rather than across the board.
- **Centralize timestamp parsing.** ⚠️ **Open, and now half-done in a way worth finishing.** There
  are two lists: `_GENERIC_TIMESTAMPS` in `auto.py` (9 pattern/format pairs, used only by the
  detector's generic pass) and a hard-coded `strptime` string inside each dataset loader. So the
  shared list exists but only detection consumes it. lnav's `time_formats.am` has 109 entries
  against those 9 — the gap is the *inventory*, not the mechanism, and it is the cheapest way to
  make the generic pass cover more unknown files.
- **Unify multi-line handling.** ⚠️ **Open, and it got worse.** Implemented *three* times now:
  `RawLoader.missing_timestamp_action`, `HadoopLoader._merge_multiline_entries`, and
  `SyslogLoader.multiline`. The third deliberately named its `keep`/`drop` options after the first,
  which documents the duplication without removing it. lnav expresses this as a single `multiline`
  flag on the format. The shared continuation rule is still worth hoisting: *a line that does not
  start a new record belongs to the previous one.* `SyslogLoader`'s version is the most developed
  and is the natural thing to generalize — note it also learned that where the merged text lands
  matters (`merge-message` feeds it to `e_words`/`e_chars_len`; `merge-add-column` parks it in
  `trace`), which the other two do not distinguish.
- **Decide whether `m_timestamp` stays mandatory.** ✅ *Settled: per instance.* `JsonLoader`,
  `LogfmtLoader` and `AutoLoader` each set `self._mandatory_columns = ["m_message"]`, because a
  plain-text fallback legitimately has no clock. The mandatory set is therefore per-loader, not
  global, and detection is allowed to produce a timeless frame.
- **Encoding.** ✅ *Settled.* Loaders still read `utf8-lossy`, and `AutoLoader.detections()` reports
  the `U+FFFD` count per file, so a mis-decoded file and a mis-detected format no longer look alike.
- **Reach the MCP path.** ⚠️ **Open — and this is the one that makes all the other work invisible.**
  `loglead/delta/log_root.py` still calls `RawLoader` with a glob and nothing else, so the 19 MCP
  tools remain plain-text-only no matter what the loader layer can now read. The natural shape is a
  format argument on `open_log_root` defaulting to `"auto"`. The blocker is small and specific:
  `read_folders()` consumes an `orig_file_name` column that only `RawLoader` produces, and
  `mcp/session.py`'s `BASE_COLUMNS` requires it downstream.

---

## 7. How to verify any claim in this document

| Claim | Where to look |
|---|---|
| lnav format counts (73 external: 59 regex + 14 JSON-lines) | `~/lnav/src/formats/*.json`, key `file-type` (older files use legacy `"json": true`) |
| lnav's 5 hard-coded formats | `~/lnav/src/log_format_impls.cc` — `generic_log_format`, `logfmt_format`, `bro_log_format`, `w3c_log_format`, `piper_log_format` |
| Detection algorithm | `~/lnav/docs/source/formats.rst` §"Built-in Formats"; `~/lnav/ARCHITECTURE.md` §"File Monitoring" and §"Log Formats" |
| 109 timestamp patterns | `~/lnav/src/time_formats.am` — `grep -oE '"[^"]+"' src/time_formats.am \| wc -l` |
| Format spec fields | `~/lnav/docs/schemas/format-v1.schema.json` |
| Container formats and converters | `~/lnav/src/file_format.hh`; `~/lnav/src/formats/pcap_log.json` (`converter` block) |
| angle-grinder parsers | `~/angle-grinder/src/operator/{parse,split}.rs`; `json`/`logfmt` are `InlineOperator` variants in `~/angle-grinder/src/lang.rs`, not files under `src/operator/`. `README.md` §"Non Aggregate Operators" |
| angle-grinder input is stdin or one file | `~/angle-grinder/src/bin/agrind.rs` |
| angle-grinder's shipped format aliases | `~/angle-grinder/aliases/{apache,nginx,k8s-ingress-nginx}.toml`; `multi-operator.toml` is a test fixture, not a format |
| LogLead loader contract | `loglead/loaders/base.py` — `execute()`, `_split_and_unnest()`, `_csv_separator` |
| LogLead's loader inventory | `loglead/loaders/__init__.py`; per-loader detail in `loglead/loaders/README.md` |
| Which formats LogLead detects, and in what order | `loglead/loaders/auto.py` — `detect_dataset()` then `detect_format()`; `tests/log_file_detection.py` checks the choice against every dataset config |
| Shipped format specs (13) | `loglead/loaders/json_formats/*.yml` (4), `loglead/loaders/access_log_formats/*.yml` (4), `loglead/loaders/delimited_formats/*.yml` (5) |
| MCP path is still `RawLoader`-only | `loglead/delta/log_root.py` — the single `RawLoader(...)` call; `orig_file_name` in `loglead/mcp/session.py` `BASE_COLUMNS` |
| `.gz` already works | Polars 1.38.1 decompresses transparently in both `read_csv` and `scan_csv` |

JSON test-dataset candidates, their measured sizes, and the downloader/test-suite integration notes
have their own verification table in
[log-format-json-testdata.md](log-format-json-testdata.md)&nbsp;§5.

---

## 8. Where LogLead stands, and what to do next

### 8.1 Against angle-grinder: ahead, except on one axis

angle-grinder's whole format surface is five parsers — `json`, `logfmt`, `split`, `parse`,
`parse regex` — and no detection at all: the user names the parser. LogLead now has an equal or
better answer to four of the five, plus detection, which angle-grinder deliberately does not attempt.

Two things it still does that LogLead does not, and only one of them matters:

- **`from <field>`** (§5 item 8) — re-running a parser against a column instead of a file. This is
  the real gap, and angle-grinder's own documentation leads with it (`json | logfmt from
  nested_key`) because layered logs are ubiquitous: a Kubernetes JSON envelope around an app line, a
  syslog frame around a JSON payload.
- **`split` without a header** — a thin slice of §5 item 3, and the less useful half of it.
  `DelimitedLoader` covers it with `header='none'` plus `columns=[...]`, i.e. only if you can name
  the fields; splitting into positional, unnamed columns is still not offered, and nothing has
  wanted it.

### 8.2 Against lnav: the mechanisms match, the catalogue does not

This is the honest summary, and the distinction is worth being precise about, because the two halves
have very different costs.

**Mechanisms — LogLead is close to parity, and ahead in one place.** Of lnav's five hard-coded
formats, `piper_log` is lnav's own capture format and irrelevant here; of the remaining four,
LogLead now has an equivalent for each — `generic_log` (`AutoLoader`'s generic pass), `logfmt`
(`LogfmtLoader`), and `bro_log` and `w3c_log` (`DelimitedLoader`'s zeek and w3c header
styles, which read the same directives lnav's two classes do). lnav's detection
design — sniff, match on the first lines, **lock the format for the file**, fall back to plain text —
was adopted wholesale and is what `AutoLoader` does. LogLead is *ahead* on the dataset probe (§5
item 6), which lnav has no reason to want: lnav identifies a format so it can display a file;
LogLead identifies a dataset so it can attach labels and produce a `df_seq`.

**Catalogue — not close, and that is a deliberate choice rather than a backlog.**

| | lnav | LogLead |
|---|---|---|
| Format definitions shipped as data | 73 | 13 (4 JSON + 4 access log + 5 delimited) |
| Format families needing code | 5 C++ classes | 5 family loaders + 1 detector |
| Timestamp patterns tried automatically | 109 | 9 |

§5 item 7 already argues that closing the *definition* count is not the goal — "LogLead's users
bring their own logs; the answer is the registry, not sixty hand-written loaders" — and that holds.
The **109-vs-9 timestamp row does not fall under that argument**, though, and is the one number in
this table worth moving: it is a list of patterns, it costs nothing but transcription, and every
entry added widens what `AutoLoader`'s generic pass recognizes on files nobody wrote a spec for.

### 8.3 What to add next

In order. Both are small and unlock disproportionate value. The third entry that stood here — §5
item 3, delimited text with a header — is now built (`DelimitedLoader`), which completes the
original shortlist; what it recommended held up, including doing the self-describing variants
first, and the one thing it did not foresee is that a *generic* CSV needs less configuration than
predicted (`Content`-style message columns are conventional enough to guess) and a *labelled* one
needs more (the label column has to be kept out of the message).

1. **Expose the format layer to `delta`/MCP** (§6, last bullet). *Not a format at all*, which is why
   it is first: every loader built since this document was written is invisible through the 19 MCP
   tools, because `log_root.py` hard-codes `RawLoader`. A format argument on `open_log_root`
   defaulting to `"auto"` turns four delivered loaders and a detector into user-visible capability
   for what is, mechanically, a small change — the blocker is one column (`orig_file_name`) that
   only `RawLoader` currently produces. Nothing else on this list has that ratio.
2. **Grow the generic timestamp list** (§6, second bullet). Nine patterns against lnav's 109, in one
   already-central place (`_GENERIC_TIMESTAMPS` in `auto.py`), consumed by the one code path that
   handles files no spec covers. Pure transcription from `time_formats.am`, bounded, and directly
   measurable: the detection test already reports the match rate per file.
Then, if the layered-log case shows up in practice, **§5 item 8 (`from <field>`)** — noting the
design point that section already makes: it belongs in `EventLogEnhancer` producing `e_*` columns,
not in a loader.

**Still explicitly not worth doing:** the per-vendor long tail (§5's "explicitly deprioritized"), and
binary containers. Both arguments hold unchanged.

### 8.4 What this document should stop claiming

Superseded statements, kept visible so the diff is legible rather than silently rewritten: §1.3's
"every format that dominates modern production logging requires a new hand-written loader" (five
families now exist), §4's empty cells for logfmt, syslog and delimited-with-header, §6's
"implemented twice already" for multi-line handling — it is three now, and consolidating them is
still open — and, as of `DelimitedLoader`, this document's own repeated claim that LogLead "reads a
CSV only inside `NezhaLoader`" and that item 3 is the last unbuilt piece of the shortlist.
