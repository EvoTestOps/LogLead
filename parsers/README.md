# Log Parsing in LogLead
LogLead is integrated with many log parsers. 

## Integration
[Tipping](https://pypi.org/project/tipping/) and [Drain3](https://pypi.org/project/drain3/) are integrated via pip package management. 

Fast-Iplom is a new implementation of the Iplom algorithm that utilizes Polars dataframes to ensure state-of-the-art computational efficiency.

The others parser are imported with source code. For more details, please refer to the directories.

## Computational Efficiency Benchmark

Log files can be very large, making efficient log parsing crucial. Below are two benchmarks run on the well-known HDFS dataset on two different hardware setups.

Laptop
TODO

Virtual Machine
TODO

## Anomaly Detection Benchmark
Log parsing results can effect anomaly detection results. TODO