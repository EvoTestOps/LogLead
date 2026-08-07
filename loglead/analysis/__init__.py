"""Run-comparison analyses over a folder-of-runs log corpus.

This is the analysis layer that sits between LogLead's primitives (loaders,
enhancers, ``LogDistance``, ``AnomalyDetector``) and a caller that wants to ask
"which of these runs looks wrong, and where?". It was ported from the sibling
project `LogDelta <https://github.com/EvoTestOps/LogDelta>`_, which drives the
same analyses from a YAML file.

The grid is three question types across four granularities:

==============  ==========================  ==========================  ================
Level           Distance (pair)             Anomaly (one vs many)       Visualize (set)
==============  ==========================  ==========================  ================
L1 run / files  ``distance_run_file``       ``anomaly_run(file=True)``  ``plot_run(file=True)``
L2 run / text   ``distance_run_content``    ``anomaly_run()``           ``plot_run()``
L3 file         ``distance_file_content``   ``anomaly_file_content``    ``plot_file_content``
L4 line         ``distance_line_content``   ``anomaly_line_content``    --
==============  ==========================  ==========================  ================

Unlike the LogDelta originals, these functions hold no module-level state,
never change the process working directory, and never write files -- they
return Polars DataFrames (and plotly figures). Use :mod:`loglead.analysis.export`
if you want artifacts on disk.

Typical use::

    from loglead.analysis import corpus, anomaly

    df, n_runs = corpus.read_folders("/data/hadoop")
    df = EventLogEnhancer(df).normalize(regexs=masking.get_pattern("myllari_extended"))
    results, df = anomaly.anomaly_run(df, target_run="ALL", content_format="Words")
    print(results.sort("rank_sum", descending=True).head())

Keeping the returned ``df`` is what lets a long-lived session avoid re-parsing.
"""

from . import anomaly, corpus, distance, export, masking, scoring, visualize
from .anomaly import (
    anomaly_file_content,
    anomaly_line_content,
    anomaly_run,
    run_anomaly_detection,
)
from .corpus import prepare_content, prepare_files, prepare_runs, read_folders
from .distance import (
    distance_file_content,
    distance_line_content,
    distance_run_content,
    distance_run_file,
)
from .visualize import plot_file_content, plot_run

__all__ = [
    "anomaly",
    "corpus",
    "distance",
    "export",
    "masking",
    "scoring",
    "visualize",
    "read_folders",
    "prepare_runs",
    "prepare_files",
    "prepare_content",
    "distance_run_file",
    "distance_run_content",
    "distance_file_content",
    "distance_line_content",
    "anomaly_run",
    "anomaly_file_content",
    "anomaly_line_content",
    "run_anomaly_detection",
    "plot_run",
    "plot_file_content",
]
