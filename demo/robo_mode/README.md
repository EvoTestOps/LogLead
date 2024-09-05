# Robo-Mode 
Robo-mode assumes your folders represent a collection of software logs of interest. Robo-mode performs a comparison between two or more folders using matching file names.  A **target run** represents a software run we are interested in analyzing. Robo-mode uses **comparison runs** as a baseline. For example, the "My_passing_logs" folder would be a comparison run, while "My_failing_logs" would be your target run that you want to analyze.

## Example
- Download [Hadoop data](https://zenodo.org/records/8196385/files/Hadoop.zip?download=1) from Zenodo
- Edit [config.yml](https://github.com/EvoTestOps/LogLead/blob/main/demo/robo_mode/config.yml) so that it points to wherever you unziped Hadoop
- Type: python main.py
- Observe excel-files in output folder. 

## Types of Analysis
In robo-mode, two types of analysis are available:

1. **Measure the distance between two logs or sets of logs** using:
   - Jaccard similarity
   - Cosine similarity
   - Containment similarity
   - Compression similarity

2. **Build an anomaly detection model** from a set of logs and use it to predict anomalies in a log using:
   - KMeans
   - IsolationForest
   - RarityModel

## Levels of Analysis
Both types of analysis can be done at four different levels:

1. **Run (folder) level**, investigating the names of files without looking at their contents.
2. **Run (folder) level**, investigating run contents (this is slower than what is done in 1).
3. **File level**, investigating file contents (matched with the same names between runs).
4. **Line level**, investigating line contents (matched with the same names between runs).
