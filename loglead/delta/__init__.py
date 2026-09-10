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
day, or a release. The levels nest: log folder -> log file -> log line. Its files
are read through whichever loader fits them: ``format="auto"`` detects per file,
or name a family to pin one (see :func:`log_root.available_formats`).

Three question types across four granularities:

=================  ===========================  =============================  ==========================
Granularity        Distance (pair)              Anomaly (one vs many)          Visualize (set)
=================  ===========================  =============================  ==========================
folder / files     ``distance_folder_filename`` ``anomaly_folder(file=True)``  ``plot_folder(file=True)``
folder / text      ``distance_folder_content``  ``anomaly_folder()``           ``plot_folder()``
file               ``distance_file_content``    ``anomaly_file_content``       ``plot_file_content``
line               ``distance_line_content``    ``anomaly_line_content``       --
=================  ===========================  =============================  ==========================

Unlike the LogDelta originals, these functions hold no module-level state,
never change the process working directory, and never write files -- they
return Polars DataFrames (and plotly figures). Use :mod:`loglead.delta.export`
if you want artifacts on disk.

Typical use::

    from loglead.delta import log_root, anomaly

    df, info = log_root.read_log_root("/data/hadoop")   # format="auto" by default
    df = EventLogEnhancer(df).normalize(regexs=masking.get_pattern("myllari_extended"))
    results, df = anomaly.anomaly_folder(df, target_folder="ALL", content_format="Words")
    print(results.sort("rank_sum", descending=True).head())

Keeping the returned ``df`` is what lets a long-lived session avoid re-parsing.
"""

import importlib

#: Public name -> the submodule that defines it, resolved on first use
#: (:pep:`562`) rather than at import. Importing every submodule eagerly meant
#: ``from loglead.delta import log_root`` pulled sklearn (via ``anomaly`` and
#: ``distance``) and plotly (via ``visualize``) whatever the caller was about to
#: do -- ~190MB before :func:`peek_log_root`, whose whole point is being cheap,
#: had stat'ed a single file. The submodules are named here as strings so that
#: only the one actually reached is imported; see ``_LAZY_MODULES``.
_LAZY_MODULES = ("anomaly", "distance", "export", "log_root", "masking", "scoring",
                 "split", "visualize")

_LAZY_NAMES = {
    "anomaly_file_content": "anomaly",
    "anomaly_line_content": "anomaly",
    "anomaly_folder": "anomaly",
    "run_anomaly_detection": "anomaly",
    "available_formats": "log_root",
    "peek_log_root": "log_root",
    "prepare_content": "log_root",
    "prepare_files": "log_root",
    "prepare_folders": "log_root",
    "read_folders": "log_root",
    "read_log_root": "log_root",
    "resolve_format": "log_root",
    "split_log_file": "split",
    "distance_file_content": "distance",
    "distance_line_content": "distance",
    "distance_folder_content": "distance",
    "distance_folder_filename": "distance",
    "DEFAULT_PLOTS": "visualize",
    "PLOTS": "visualize",
    "plot_file_content": "visualize",
    "plot_folder": "visualize",
}


def __getattr__(name):
    if name in _LAZY_MODULES:
        value = importlib.import_module(f".{name}", __name__)
    elif name in _LAZY_NAMES:
        value = getattr(importlib.import_module(f".{_LAZY_NAMES[name]}", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value  # imported once; every later access is a plain lookup
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_MODULES) | set(_LAZY_NAMES))

__all__ = [
    "anomaly",
    "log_root",
    "distance",
    "export",
    "masking",
    "scoring",
    "split",
    "visualize",
    "read_log_root",
    "read_folders",
    "peek_log_root",
    "split_log_file",
    "available_formats",
    "resolve_format",
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
    "PLOTS",
    "DEFAULT_PLOTS",
]
