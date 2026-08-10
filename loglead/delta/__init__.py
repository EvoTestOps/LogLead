"""Comparing log folders: what differs, what looks wrong, and where.

This is the layer between LogLead's primitives (loaders, enhancers,
``LogDistance``, ``AnomalyDetector``) and a caller who wants to ask "which of
these log folders looks wrong, and where?". Everything here is *comparative* --
a target is always judged against the other log folders, never in isolation --
which is what the name refers to.

.. note::
   ``loglead.delta`` is **not** the sibling project `LogDelta
   <https://github.com/EvoTestOps/LogDelta>`_, and does not depend on it.
   ``logdelta`` is never imported and is not a dependency of LogLead. The code
   here was *ported from* LogDelta, which drives the same analyses from a YAML
   file; the dependency runs the other way, LogDelta depends on LogLead.

The data shape is a **log root**: a directory whose immediate subdirectories are
**log folders** -- any set of logs that belong together, be it a test run, a
day, or a release. The levels nest: log folder -> log file -> log line.

Three question types across four granularities:

=================  ===========================  =============================  ==========================
Level              Distance (pair)              Anomaly (one vs many)          Visualize (set)
=================  ===========================  =============================  ==========================
L1 folder / files  ``distance_folder_filename`` ``anomaly_folder(file=True)``  ``plot_folder(file=True)``
L2 folder / text   ``distance_folder_content``  ``anomaly_folder()``           ``plot_folder()``
L3 file            ``distance_file_content``    ``anomaly_file_content``       ``plot_file_content``
L4 line            ``distance_line_content``    ``anomaly_line_content``       --
=================  ===========================  =============================  ==========================

Unlike the LogDelta originals, these functions hold no module-level state,
never change the process working directory, and never write files -- they
return Polars DataFrames (and plotly figures). Use :mod:`loglead.delta.export`
if you want artifacts on disk.

Typical use::

    from loglead.delta import log_root, anomaly

    df, n_folders = log_root.read_folders("/data/hadoop")
    df = EventLogEnhancer(df).normalize(regexs=masking.get_pattern("myllari_extended"))
    results, df = anomaly.anomaly_folder(df, target_folder="ALL", content_format="Words")
    print(results.sort("rank_sum", descending=True).head())

Keeping the returned ``df`` is what lets a long-lived session avoid re-parsing.
"""

from . import anomaly, distance, export, log_root, masking, scoring, visualize
from .anomaly import (
    anomaly_file_content,
    anomaly_line_content,
    anomaly_folder,
    run_anomaly_detection,
)
from .log_root import prepare_content, prepare_files, prepare_folders, read_folders
from .distance import (
    distance_file_content,
    distance_line_content,
    distance_folder_content,
    distance_folder_filename,
)
from .visualize import plot_file_content, plot_folder

__all__ = [
    "anomaly",
    "log_root",
    "distance",
    "export",
    "masking",
    "scoring",
    "visualize",
    "read_folders",
    "prepare_folders",
    "prepare_files",
    "prepare_content",
    "distance_folder_filename",
    "distance_folder_content",
    "distance_file_content",
    "distance_line_content",
    "anomaly_folder",
    "anomaly_file_content",
    "anomaly_line_content",
    "run_anomaly_detection",
    "plot_folder",
    "plot_file_content",
]
