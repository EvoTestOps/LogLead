# LogLEAD
LogLEAD stands for Log Loader, Enhancer, and Anomaly Detector.

LogLEAD is composed of distinct modules: the Loader, Enhancer, and Anomaly Detector.

Loader: This module reads in the log files deals with the specifics features of each log file. It produces a dataframe with certain semi-mandatory fields. These fields enable actions in the subsequent stages.

Enhancer: This module extracts additional data from logs. The enhancement takes place directly within the dataframes, where each new field captures the results of a specific enhancement process. For example, log parsing, the creation of tokens from log messages, and measuring sequence lengths are all considered forms of log enhancement. Enhancement can happen at the event level or be aggregated to the sequence level. For instance, the log sequence length either in terms of log events or duration needs to be aggregated at the sequence level.

Anomaly Detector: This module uses the enhanced log data to calculate anomaly scores or assign binary labels to either individual log events or aggregated sequences. Anomaly detection can also be seen as a type of log enhancement, as it adds scores or labels to log events or sequences.

Polars is used for both loading and data processing. It's notably faster compared to alternatives like Pandas.
