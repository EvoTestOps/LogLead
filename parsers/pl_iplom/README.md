#Fast IPLoM

This parser implements a faster version of the IPLoM algorithm. Initial testing indicates a speedup of approximately 10x compared to the standard IPLoM implementation on the HDFS dataset. If you use the fast version of the IPLoM algorithm, please cite the LogLead paper.

```bibtex
@inproceedings{mantyla2023loglead,
  author = {M\"{a}ntyl\"{a}, Mika and Wang, Yuqing and Nyyss\"{o}l\"{a}, Jesse},
  title = {LogLead - Fast and Integrated Log Loader, Enhancer, and Anomaly Detector},
  booktitle = {Proceedings of the IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER)},
  year = {2024},
  publisher = {IEEE},
  address = {Rovaniemi, Finland},
  pages = {1-5},
  url  = {https://arxiv.org/abs/2311.11809}
}
```

Original IPLoM is described in the following papers. 

```bibtex
@inproceedings{makanju2009clustering,
  title={Clustering event logs using iterative partitioning},
  author={Makanju, Adetokunbo AO and Zincir-Heywood, A Nur and Milios, Evangelos E},
  booktitle={Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={1255--1264},
  year={2009}
}

@article{makanju2011lightweight,
  title={A lightweight algorithm for message type extraction in system application logs},
  author={Makanju, Adetokunbo and Zincir-Heywood, A Nur and Milios, Evangelos E},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={24},
  number={11},
  pages={1921--1936},
  year={2011},
  publisher={IEEE}
}
```