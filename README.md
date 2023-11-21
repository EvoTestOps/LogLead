# LogLead
LogLead can be used for efficient benchmarking of log anomaly detection algorithms. 

For easy onboarding, take a look at our [5-minute screencast on YouTube](https://www.youtube.com/watch?v=8stdbtTfJVo) or see demo folder examples: [TB_samples.py](https://github.com/EvoTestOps/LogLead/blob/main/demo/TB_samples.py) and [HDFS_samples.py](https://github.com/EvoTestOps/LogLead/blob/main/demo/HDFS_samples.py).

## Architecutral overview
LogLead is composed of distinct modules: the Loader, Enhancer, and Anomaly Detector. We use [Polars] (https://www.pola.rs/) dataframes as its notably faster compared to alternatives like Pandas.

![Dataflow](images/LogLead_Dataflow_Diagram.png)

Loader: This module reads in the log files deals with the specifics features of each log file. It produces a dataframe with certain semi-mandatory fields. These fields enable actions in the subsequent stages.

Enhancer: This module extracts additional data from logs. The enhancement takes place directly within the dataframes, where new columns are added as a result of the enhancement process. For example, log parsing, the creation of tokens from log messages, and measuring log sequence lengths are all considered forms of log enhancement. Enhancement can happen at the event level or be aggregated to the sequence level. For instance, the log sequence length is aggregated at the sequence level.

Anomaly Detector: This module uses the enhanced log data to perform Anomaly Detection with mainly using SKlearn at the moment. Anomaly detection can also be seen as a type of log enhancement, as it adds scores or labels to log events or sequences.

## Example of Anomaly Detection results

Below you can see anomaly detection results (F1-Binary) trained on 0.5% subset of HDFS data. 
We use 5 different log message enhancement strategies: [Words](https://en.wikipedia.org/wiki/Bag-of-words_model), [Drain](https://github.com/logpai/Drain3), [LenMa](https://github.com/keiichishima/templateminer), [Spell](https://github.com/logpai/logparser/tree/main/logparser/Spell), and [BERT](https://github.com/google-research/bert) 

The enhancement strategies are tested with 5 different machine learning algorithms: DT (Decision Tree), SVM (Support Vector Machine), LR (Logistic Regression), RF (Random Forest), XGB (XGBoost).


|         | Words  | Drain  | Lenma  | Spell  | Bert   | Average |
|---------|--------|--------|--------|--------|--------|---------|
| DT      | 0.9719 | 0.9816 | 0.9803 | 0.9828 | 0.9301 | 0.9693  |
| SVM     | 0.9568 | 0.9591 | 0.9605 | 0.9559 | 0.8569 | 0.9378  |
| LR      | 0.9476 | 0.8879 | 0.8900 | 0.9233 | 0.5841 | 0.8466  |
| RF      | 0.9717 | 0.9749 | 0.9668 | 0.9809 | 0.9382 | 0.9665  |
| XGB     | 0.9721 | 0.9482 | 0.9492 | 0.9535 | 0.9408 | 0.9528  |
|---------|--------|--------|--------|--------|--------|---------|
| Average | 0.9640 | 0.9503 | 0.9494 | 0.9593 | 0.8500 |         |



## Paper 
More detailed description of the work is documented in [arxiv pre-print](https://arxiv.org/abs/2311.11809).

If you use this software in your research, please cite it as below:

```bibtex
@misc{mantyla2023loglead,
  author = {M\"{a}ntyl\"{a}, Mika and Wang, Yuqing and Nyyss\"{o}l\"{a}, Jesse},
  title = {LogLead -- Fast and Integrated Log Loader, Enhancer, and Anomaly Detector},
  year = {2023},
  publisher = {arXiv},
  journal = {arXiv preprint},
  howpublished = {\url{https://arxiv.org/abs/2311.11809}}
}
