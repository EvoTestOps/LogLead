# LogLead
LogLead is designed to efficiently benchmark log anomaly detection algorithms and log representations. LogLead is also used as a backend for projects such as [LogDelta](https://github.com/EvoTestOps/LogDelta) and [VisualLogAnalyzer](https://github.com/EvoTestOps/VisualLogAnalyzer), which offer a more user-friendly approach to log analysis and log anomaly detection.

LogLead combines three independently swappable stages — **Loader → Enhancer → Anomaly Detector** — so the same enhancement and detection code applies to any log once it is loaded. It ships loaders for a dozen public datasets, format-detecting and spec-driven loaders for everyday log formats (JSON, syslog, logfmt, access logs, CSV/TSV), a dozen log representations (enhancers), and 11 classifiers. Everything runs on [Polars](https://www.pola.rs/) dataframes rather than Pandas.

If there's something you believe should be included, please submit a request for a dataset, enhancer, or classifier in the [issue tracker](https://github.com/EvoTestOps/LogLead/issues).

## What's new in 2.0

- **`AutoLoader`** — point it at a file or a directory and it detects the format and builds the right loader for you. No more picking a loader class by hand.
- **New format-family loaders** — `JsonLoader`, `SyslogLoader`, `LogfmtLoader`, `AccessLogLoader`, and `DelimitedLoader` read a format from a YAML spec instead of a bespoke Python class. Specs for GELF, nginx, Windows events, Zeek, IIS, OpenStack, and loghub ship with the package.
- **`loglead.delta`** — an unsupervised pipeline that compares many log *folders* against each other (distance, anomaly, and visualization, at folder-name, folder-content, file, and line granularity) rather than training on labels.
- **MCP server** — let an AI assistant drive log comparison conversationally. See [MCP server](#mcp-server) below.

**Breaking changes from 1.x** are listed in the [changelog](https://github.com/EvoTestOps/LogLead/blob/main/CHANGELOG.md). In short: Python 3.9 is no longer supported (3.10–3.13 now), `GELFLoader` is replaced by `JsonLoader(format="gelf")`, `EventLogEnhancer.normalize()` is renamed to `mask()` (the old name still works but warns), and the SHAP-based explainer module was removed.

## Installing LogLead

LogLead requires Python 3.10–3.13.

Install with [`uv`](https://docs.astral.sh/uv/):
```
uv add loglead
```
Or with `pip`:
```
python -m pip install loglead
```

Then clone the project, move to demo folder and run some demos
```
git clone https://github.com/EvoTestOps/LogLead.git
cd LogLead
uv run demo/HDFS_samples.py
uv run demo/TB_samples.py
```
Or with `pip` (after installing LogLead into your environment):
```
cd LogLead/demo
python HDFS_samples.py
python TB_samples.py
```
`uv run` syncs the environment from `pyproject.toml`/`uv.lock` on first use, so there's no separate install step before running anything.

### Loading your own logs

The easiest starting point is [`AutoLoader`](https://github.com/EvoTestOps/LogLead/blob/main/loglead/loaders/auto.py), which works out the format and builds the matching loader:

```python
from loglead.loaders import AutoLoader

loader = AutoLoader(filename="mystery.log")                      # one file
loader = AutoLoader(filename="logs", filename_pattern="*.log")   # a tree, detected per file
df = loader.execute()
```

Run [AutoLoader_samples.py](https://github.com/EvoTestOps/LogLead/blob/main/demo/AutoLoader_samples.py) to see it on the bundled sample data. If detection gets it wrong, or you want the dataset-specific cleanup a custom loader does, pick a loader explicitly — [loglead/loaders/README.md](https://github.com/EvoTestOps/LogLead/blob/main/loglead/loaders/README.md) lists every loader, the formats it reads, and the shipped format specs. [`RawLoader`](https://github.com/EvoTestOps/LogLead/blob/main/loglead/loaders/raw.py) is the no-assumptions fallback: one row per line, one `m_message` column.

To try [RawLoaderDemo](https://github.com/EvoTestOps/LogLead/blob/main/demo/RawLoader_NoLabels.py), get the original [BGL](https://zenodo.org/records/8196385/files/BGL.zip?download=1) and [HDFS](https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1) datasets, then point the demo at them via a ".env" file in your LogLead root (see [.env.sample](https://github.com/EvoTestOps/LogLead/blob/main/.env.sample)) or by editing the script directly:
```
uv run demo/RawLoader_NoLabels.py    # or: python RawLoader_NoLabels.py
```
Finally, you can download all public datasets with the [downloader](https://github.com/EvoTestOps/LogLead/blob/main/downloader/download_data.py) script — or, if you've cloned the repo and want to run the test suite too, point it at one of the [tests/datasets_*.yml](https://github.com/EvoTestOps/LogLead/tree/main/tests) configs instead (e.g. [tests/datasets_mid_labels.yml](https://github.com/EvoTestOps/LogLead/blob/main/tests/datasets_mid_labels.yml), the one `tests/main.py` uses by default):
```
uv run downloader/download_data.py                                            # or: python downloader/download_data.py
uv run downloader/download_data.py --config tests/datasets_mid_labels.yml     # or with pip, same --config flag
```
**Disk space:** downloading everything transfers roughly 7 GB and expands to about 104 GB unzipped — check you have **~110 GB free**. Liberty, Spirit, and Thunderbird account for most of it, at 30-38 GB each.

If you're short on space, edit the `datasets:` list in [downloader/datasets.yml](https://github.com/EvoTestOps/LogLead/blob/main/downloader/datasets.yml) (or the relevant `tests/datasets_*.yml` if you're using `--config tests/datasets_*.yml`) and set `download: false` for datasets you don't need.

| Dataset | Download size | Unzipped size |
|---|---|---|
| BGL | 58 MB | 709 MB |
| Hadoop | 3 MB | 49 MB |
| HDFS | 187 MB | 1.8 GB |
| Liberty | 672 MB | 30 GB |
| Spirit | 906 MB | 38 GB |
| Thunderbird | 2.0 GB | 30 GB |
| Nezha (git clone) | ~2.9 GB | 2.9 GB |
| ADFA-LD | 2.4 MB | 26 MB |
| AWSCTD | 10 MB | 559 MB |
| **Total** | **~6.7 GB** | **~104 GB** |

### Known issues

- If `scikit-learn` wheel fails to compile, check that you have `gcc` and `g++` installed.
- pip version does not have the `tensorflow` dependencies necessary for `BertEmbeddings`.
Install them manually (preferably in a conda enviroment).

## MCP server

LogLead ships an [MCP](https://modelcontextprotocol.io) server so an AI agent (Claude, Goose, etc.) can drive
log comparison and anomaly analysis conversationally. It loads, masks, and parses a log root once per session
and reuses that for every later question, exposing the [`loglead.delta`](#functional-overview) comparison
pipeline as tools.

```
uv add "loglead[mcp]"      # or: python -m pip install "loglead[mcp]"
loglead-mcp
```

Point your MCP client (Claude Code, Goose, Claude Desktop) at the `loglead-mcp` command this installs. Full
client setup is in the [GitHub README's MCP server section](https://github.com/EvoTestOps/LogLead#mcp-server),
and a full example session with screenshots is in
[MCP_client_demo_script.md](https://github.com/EvoTestOps/LogLead/blob/main/MCP_client_demo_script.md).

## Demos
Both demos below analyze different datasets but share most of their underlying code, showing how the same enhancement and detection logic carries across log formats.

### Thunderbird Supercomputer Log Demo
- **Script**: [TB_samples.py](https://github.com/EvoTestOps/LogLead/blob/main/demo/TB_samples.py)
- **Description**: This demo presents a Thunderbird supercomputer log, labeled at the line (event) level. A first column marked with “-” indicates normal behavior, while other markings represent anomalies.
- **Log Snapshot**: View the log [here](https://github.com/logpai/loghub/blob/master/Thunderbird/Thunderbird_2k.log_structured.csv).
- **Dataset**: The demo includes a parquet file containing a subset of 263,408 log events, with 21,955 anomalies.
- **Screencast**: For an overview of the demo, watch our [5-minute screencast on YouTube](https://www.youtube.com/watch?v=8stdbtTfJVo).
### Hadoop Distributed File System (HDFS) Log Demo

- **Script**: [HDFS_samples.py](https://github.com/EvoTestOps/LogLead/blob/main/demo/HDFS_samples.py)
- **Description**: This demo showcases logs from the Hadoop Distributed File System (HDFS), labeled at the sequence level (a sequence is a collection of multiple log events).
- **Log Snapshot**: View the log [here](https://github.com/logpai/loghub/blob/master/HDFS/HDFS_2k.log_structured.csv).
- **Anomaly Labels**: Provided in a separate file.
- **Dataset**: The demo includes a parquet file containing a subset of 222,579 log events, forming 11,501 sequences with 350 anomalies.

## Testing
The demos catch obvious errors quickly; the full test set takes longer (up to 30 minutes). With `pip`, `cd` into the script's directory and drop the `uv run` prefix.

```
uv run demo/HDFS_samples.py                          # basic demos
uv run demo/TB_samples.py
uv run demo/parser_benchmark/ano_detection.py         # parser benchmark
uv run demo/parser_benchmark/parsing_speed.py
uv run tests/main.py                                  # full test suite
```

## Example of Anomaly Detection results
Below you can see anomaly detection results (F1-Binary) trained on 0.5% subset of HDFS data. 
We use 5 different log message enhancement strategies: [Words](https://en.wikipedia.org/wiki/Bag-of-words_model), [Drain](https://github.com/logpai/Drain3), [LenMa](https://github.com/keiichishima/templateminer), [Spell](https://github.com/logpai/logparser/tree/main/logparser/Spell), and [BERT](https://github.com/google-research/bert) 

The enhancement strategies are tested with 5 different machine learning algorithms: DT (Decision Tree), SVM (Support Vector Machine), LR (Logistic Regression), RF (Random Forest), and XGB (eXtreme Gradient Boosting).

|         | Words  | Drain  | Lenma  | Spell  | Bert   | Average |
|---------|--------|--------|--------|--------|--------|---------|
| DT      | 0.9719 | 0.9816 | 0.9803 | 0.9828 | 0.9301 | 0.9693  |
| SVM     | 0.9568 | 0.9591 | 0.9605 | 0.9559 | 0.8569 | 0.9378  |
| LR      | 0.9476 | 0.8879 | 0.8900 | 0.9233 | 0.5841 | 0.8466  |
| RF      | 0.9717 | 0.9749 | 0.9668 | 0.9809 | 0.9382 | 0.9665  |
| XGB     | 0.9721 | 0.9482 | 0.9492 | 0.9535 | 0.9408 | 0.9528  |
|---------|--------|--------|--------|--------|--------|---------|
| Average | 0.9640 | 0.9503 | 0.9494 | 0.9593 | 0.8500 |         |

## Functional overview
LogLead is composed of distinct modules: the Loader, Enhancer, and Anomaly Detector, all on [Polars](https://www.pola.rs/) dataframes.

**Loader:** reads log files into a dataframe with the semi-mandatory fields later stages depend on. Point [`AutoLoader`](https://github.com/EvoTestOps/LogLead/blob/main/loglead/loaders/auto.py) at a file or directory and it detects the format and builds the right loader; [`RawLoader`](https://github.com/EvoTestOps/LogLead/blob/main/loglead/loaders/raw.py) is the no-assumptions fallback. Custom loaders (more accurate anomaly detection) exist for 10 systems:
* 3: [HDFS_v1](https://github.com/logpai/loghub/tree/master/HDFS#hdfs_v1), [Hadoop](https://github.com/logpai/loghub/tree/master/Hadoop), [BGL](https://github.com/logpai/loghub/tree/master/BGL), courtesy of the [LogHub team](https://github.com/logpai/loghub) ([Zenodo](https://zenodo.org/records/3227177) for full data).
* 3: [Spirit, Thunderbird and Liberty](https://www.usenix.org/cfdr-data#hpc4), from Usenix.
* 2: [Nezha](https://github.com/IntelligentDDS/Nezha) — the first microservice-based dataset, spanning [TrainTicket](https://github.com/FudanSELab/train-ticket) and the [Google Cloud Webshop demo](https://github.com/GoogleCloudPlatform/microservices-demo), with logs, traces, and metrics.
* 2: [ADFA](https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset) and [AWSCTD](https://github.com/DjPasco/AWSCTD), for intrusion detection.

Beyond those, five *spec-driven* loaders read a whole format family from a YAML spec instead of a bespoke class — [`JsonLoader`](https://github.com/EvoTestOps/LogLead/blob/main/loglead/loaders/json.py), [`SyslogLoader`](https://github.com/EvoTestOps/LogLead/blob/main/loglead/loaders/syslog.py), [`LogfmtLoader`](https://github.com/EvoTestOps/LogLead/blob/main/loglead/loaders/logfmt.py), [`AccessLogLoader`](https://github.com/EvoTestOps/LogLead/blob/main/loglead/loaders/access_log.py) and [`DelimitedLoader`](https://github.com/EvoTestOps/LogLead/blob/main/loglead/loaders/delimited.py) — with shipped specs for GELF, nginx, Windows events, Zeek, IIS, OpenStack and loghub. Full list in [loglead/loaders/README.md](https://github.com/EvoTestOps/LogLead/blob/main/loglead/loaders/README.md).

**Enhancer:** adds columns to the dataframe — event- or sequence-level — such as message/sequence length, [Duration](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.Duration.html), [Regex](https://crates.io/crates/regex), [Words](https://en.wikipedia.org/wiki/Bag-of-words_model), and [character n-grams](https://en.wikipedia.org/wiki/N-gram). Log parsers: [Drain](https://github.com/logpai/Drain3), [LenMa](https://github.com/keiichishima/templateminer), [Spell](https://github.com/bave/pyspell), [IPLoM](https://github.com/EvoTestOps/LogLead/tree/main/parsers/iplom), [AEL](https://github.com/EvoTestOps/LogLead/tree/main/parsers/AEL), [Brain](https://github.com/EvoTestOps/LogLead/tree/main/parsers/Brain), [Fast-IPLoM](https://github.com/EvoTestOps/LogLead/tree/main/parsers/fast_iplom), [Tipping](https://pypi.org/project/tipping/), and [BERT](https://github.com/google-research/bert). [NextEventPrediction](https://arxiv.org/abs/2202.09214) (with probabilities and perplexity) can run on top of any parser's output.

**Anomaly Detector:** runs on the enhanced data, mainly via scikit-learn plus a couple of custom algorithms:
* Supervised (5): [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree), [SVM](https://en.wikipedia.org/wiki/Support_vector_machine), [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression), [Random Forest](https://en.wikipedia.org/wiki/Random_forest), [XGBoost](https://en.wikipedia.org/wiki/XGBoost)
* Unsupervised (4): [One-class SVM](https://en.wikipedia.org/wiki/Support_vector_machine#One-class_SVM), [Local Outlier Factor](https://en.wikipedia.org/wiki/Local_outlier_factor), [Isolation Forest](https://en.wikipedia.org/wiki/Isolation_forest), [K-Means](https://en.wikipedia.org/wiki/K-means_clustering)
* Custom unsupervised (2): [Out-of-Vocabulary Detector](https://github.com/EvoTestOps/LogLead/blob/main/loglead/OOV_detector.py) (novel words/n-grams vs. test set) and [Rarity Model](https://github.com/EvoTestOps/LogLead/blob/main/loglead/RarityModel.py) (rarity-based scoring) — see our [preprint](https://arxiv.org/abs/2312.01934).

**Comparing log folders (`loglead.delta`):** a second, unsupervised pipeline for when there's no labels but many comparable runs. Given a log root — a directory of log folders (a test run, a day, a release) — it judges one target against the others at four granularities (folder name, folder content, file content, line content), each posing a distance, anomaly, or visualization question. Returns Polars DataFrames and plotly figures, no module-level state, no files written. This is the layer the MCP server exposes.

## Reference
Mäntylä MV, Wang Y, Nyyssölä J. Loglead-fast and integrated log loader, enhancer, and anomaly detector. In2024 IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER) 2024 Mar 12 (pp. 395-399). IEEE.  [PDF](https://ieeexplore.ieee.org/abstract/document/10589612), [preprint](https://arxiv.org/abs/2311.11809)