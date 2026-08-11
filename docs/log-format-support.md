# Broadening LogLead's log format support

**Status:** investigation / proposal. No code changes.
**Question:** `lnav` and `angle-grinder` can read log formats that LogLead cannot — most obviously JSON.
What exactly do they support, and in what order should LogLead adopt the same capabilities?

**Evidence base:** local clones of both tools, read directly rather than from their websites:

| Tool | Local path | Files that define format support |
|---|---|---|
| lnav | `~/lnav` | `src/formats/*.json` (73 definitions), `src/log_format_impls.cc` (5 hard-coded formats), `src/file_format.hh`, `src/time_formats.am`, `docs/schemas/format-v1.schema.json`, `docs/source/formats.rst`, `ARCHITECTURE.md` |
| angle-grinder | `~/angle-grinder` | `src/operator/` (one file per operator), `aliases/*.toml`, `README.md` |

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

Eleven concrete loaders. Only the first is format-driven; the other ten are dataset-driven.

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
| `GELFLoader` | one JSON object per line | **JSON lines** | the codebase's only JSON reader — see below |

### 1.3 The gap, stated precisely

LogLead can read exactly two things: **text that can be split positionally on spaces**, and
**whatever a person has written a bespoke Python class for**. Every format that dominates modern
production logging — JSON lines, logfmt, CSV/TSV with a header, syslog, web access logs, journald,
Windows events — requires a new hand-written loader.

Two specifics worth recording because they change the priorities below:

- **`GELFLoader` is not a usable JSON implementation.** It loops in Python: `json.loads` per line,
  builds a **one-row `pl.DataFrame` per line**, then `pl.concat`s them all. That is one DataFrame
  construction per log event, so it will not survive a real GELF volume. It is also brittle: the
  default vertical `concat` requires identical schemas, so a single line with a missing or extra key
  raises rather than unifying. Polars' native `read_ndjson`/`scan_ndjson` does both correctly.
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
   **107 distinct strptime-style patterns** in `src/time_formats.am`.
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

Plus **aliases** (`aliases/*.toml`): named, reusable pipelines discoverable by keyword. Tellingly,
all three shipped aliases are *format definitions expressed as a `parse` pattern* — `apache`,
`nginx`, `k8s-ingress-nginx` — each one a positional extraction of a Common/Combined Log Format
line into named fields.

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
| **JSON lines / NDJSON** | 14 built-in formats + declarative support | `json` operator (its headline feature) | `GELFLoader` only, Python loop, not scalable |
| **logfmt** | built-in C++ format | `logfmt` operator | — |
| **Delimited + header (CSV/TSV)** | Zeek TSV + W3C ELF, both self-describing | `split` | only inside `NezhaLoader`, dataset-specific |
| **Web access logs (CLF/Combined)** | ~12 formats | all 3 shipped aliases | — |
| **Syslog** | built-in format | — | — |
| **Generic timestamped text** | `generic_log` fallback + 107 timestamp patterns | `parse` / `parse regex` | `RawLoader`, but the user must supply the regex *and* the format |
| **Binary containers** (pcap, SQLite, archives) | container layer + converter subprocesses | — | — |

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
angle-grinder's first documented example is a JSON one. LogLead has one JSON loader and it builds a
DataFrame per line.

Build a `JsonLoader` on Polars' native `read_ndjson`/`scan_ndjson` (multithreaded, schema-unifying,
same lazy/`collect_all` pattern as the existing loaders), with:

- a **field mapping** argument: which key is the message, the timestamp, the level, the sequence id;
- a **nested-key path** syntax — lnav uses `/`-separated JSON pointers (`log/logger`), angle-grinder
  uses `.key[i]`; pick one and document it;
- a **policy for the remaining keys**: keep as columns, fold into `m_message`, or drop.

Payoff beyond the obvious: `GELFLoader` becomes a three-line configuration of it rather than a
separate implementation; and Kubernetes/OTel logs carry `trace_id`/`pod`/`container` fields, which
give `SequenceEnhancer` a *real* `seq_id` — something the current dataset-specific loaders have to
manufacture with regexes.

### 2. logfmt — **cheapest win on the list**

Both tools implement it as a first-class format. It is the Go/Heroku ecosystem's default and is what
Grafana/Loki, Docker, Consul, etcd and much of the Prometheus tooling emit. The grammar is strict
and trivially parseable — one regex plus Polars string operations, no new dependency. Better, the
key names are conventional, so lnav's mapping (`timestamp|time|ts|t` → time, `level|lvl` → level,
`message|msg` → body) can be adopted wholesale and the loader needs **zero configuration** in the
common case.

### 3. Delimited text with a header (CSV/TSV), including the self-describing variants

How exported observability data usually arrives: SIEM exports, audit logs, Zeek, W3C/IIS — and,
directly relevant to this project, the `*_structured.csv` files that loghub-style datasets ship as
their already-parsed form. LogLead already depends on Polars' CSV reader, so the work is the
*mapping layer*, not parsing. The self-describing variants are the cheapest auto-detection win
available anywhere on this list: Zeek's `#fields` line and W3C's `#Fields:` directive mean the
column names come out of the file, so there is nothing for the user to configure. Both are already
in lnav and can be copied nearly verbatim as specs.

### 4. Web access logs: Common/Combined Log Format and relatives

The most common non-structured format on the internet, and the one place both tools invest most
(lnav: ~12 formats; angle-grinder: all three shipped aliases). For LogLead specifically it is
attractive for a reason beyond ubiquity: access logs have a **natural sequence id** (client IP,
session, or request id) and a **natural anomaly signal** (status codes, byte counts, latency), so
they exercise the loader → enhancer → sequence → detector chain end to end. This is also the most
plausible route to a demo dataset that is neither a 15-year-old HDFS dump nor a supercomputer log.
Cover Apache access + error, nginx, and the AWS variants (ALB/ELB/S3/CloudFront) — all positional,
all expressible as one regex each.

### 5. Syslog (RFC 3164 and RFC 5424) and journald

Still the substrate of Linux and network-device logging; lnav treats syslog as a core format. RFC
5424 additionally brings `procid`, `msgid` and structured-data elements, which map cleanly onto
LogLead columns and give another free correlation id. journald's JSON output falls out of item 1 for
free *once the field mapping is configurable* — which is an argument for doing item 1 properly
rather than hard-coding one JSON shape.

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
  `strptime` to the entire column.** Never per row — that is the mistake `GELFLoader` already makes.
- Detection is inherently **per file**, but LogLead's `delta`/MCP path loads a whole directory tree
  in one call. The detection result must therefore be resolved per `file_name` group, not once per
  load — a mixed-format log folder is the normal case, not the exception.

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
decide once:

- **Typed columns.** Both tools convert on extraction — angle-grinder documents `noconvert` as the
  *opt-out*, which shows it is deliberate. LogLead's loaders keep everything as `Utf8`
  (`infer_schema_length=0`). New structured loaders should land numeric fields as numbers so that
  `AnomalyDetector`'s `numeric_cols` works without a manual cast.
- **Centralize timestamp parsing.** It is currently a hard-coded `strptime` string per loader. The
  format work needs one shared list of candidate patterns and one "try these, vectorized" helper —
  effectively LogLead's version of `time_formats.am`.
- **Unify multi-line handling.** Implemented twice already (`RawLoader`'s merge action and
  `HadoopLoader._merge_multiline_entries`). lnav expresses this as a single `multiline` flag on the
  format. Hoist it to one shared continuation rule: *a line that does not start a new record belongs
  to the previous one.*
- **Decide whether `m_timestamp` stays mandatory.** It currently is, for all but three loaders. Plain
  application logs without dates, and syscall traces, legitimately have no clock. Either detection is
  allowed to produce a timeless frame, or the mandatory set becomes per-format.
- **Encoding.** Loaders read `utf8-lossy` and then warn about `U+FFFD`. That is a reasonable default,
  but a detection path should record *which files* degraded, since a mis-detected format and a
  mis-decoded file look the same downstream.
- **Reach the MCP path.** `loglead/delta/log_root.py` calls `RawLoader` with a glob and nothing else.
  Unless new format support is exposed there — most naturally as a format argument on
  `open_log_root`, with `"auto"` as the default once item 6 exists — the 19 MCP tools stay
  plain-text-only and none of this work is visible through them.

---

## 7. How to verify any claim in this document

| Claim | Where to look |
|---|---|
| lnav format counts (73 external: 59 regex + 14 JSON-lines) | `~/lnav/src/formats/*.json`, key `file-type` (older files use legacy `"json": true`) |
| lnav's 5 hard-coded formats | `~/lnav/src/log_format_impls.cc` — `generic_log_format`, `logfmt_format`, `bro_log_format`, `w3c_log_format`, `piper_log_format` |
| Detection algorithm | `~/lnav/docs/source/formats.rst` §"Built-in Formats"; `~/lnav/ARCHITECTURE.md` §"File Monitoring" and §"Log Formats" |
| 107 timestamp patterns | `~/lnav/src/time_formats.am` |
| Format spec fields | `~/lnav/docs/schemas/format-v1.schema.json` |
| Container formats and converters | `~/lnav/src/file_format.hh`; `~/lnav/src/formats/pcap_log.json` (`converter` block) |
| angle-grinder parsers | `~/angle-grinder/src/operator/{parse,split}.rs`, `~/angle-grinder/README.md` §"Non Aggregate Operators" |
| angle-grinder input is stdin or one file | `~/angle-grinder/src/bin/agrind.rs` |
| angle-grinder's shipped format aliases | `~/angle-grinder/aliases/{apache,nginx,k8s-ingress-nginx}.toml` |
| LogLead loader contract | `loglead/loaders/base.py` — `execute()`, `_split_and_unnest()`, `_csv_separator` |
| `GELFLoader`'s per-line DataFrame construction | `loglead/loaders/gelf.py` |
| `.gz` already works | Polars 1.38.1 decompresses transparently in both `read_csv` and `scan_csv` |

JSON test-dataset candidates, their measured sizes, and the downloader/test-suite integration notes
have their own verification table in
[log-format-json-testdata.md](log-format-json-testdata.md)&nbsp;§5.
