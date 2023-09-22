# LogLEAD
LogLEAD stands for Log Loader, Enricher, and Anomaly Detector.

LogLEAD is composed of distinct modules: the Loader, Enricher, and Anomaly Detector.

Loader: This module deals with the specifics of log files and produces a dataframe with certain semi-mandatory fields. These fields guide the actions in the subsequent stages.

Enricher: This module extracts additional data from logs. The enrichment takes place directly within the dataframes, where each new field captures the results of a specific enrichment process. For example, log parsing, the creation of trigrams from log messages, and measuring sequence lengths are all considered forms of log enrichment. Enrichment can happen at the event level or be aggregated to the sequence level. For instance, the log sequence length either in terms of log events or duration needs to be aggregated at the sequence level.

Anomaly Detector: This module uses the enriched log data to calculate anomaly scores or assign binary labels to either individual log events or aggregated sequences. Anomaly detection can also be seen as a type of log enrichment, as it adds scores or labels to log events or sequences.

Polars is used for both loading and data processing. It's notably faster compared to alternatives like Pandas.