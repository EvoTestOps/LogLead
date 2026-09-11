"""LogLead: log loading, enhancing and anomaly detection.

The public names below are resolved **lazily** (:pep:`562`). ``AnomalyDetector``
and friends live in modules that import sklearn and xgboost, which cost ~190MB
of resident memory and ~1.3s -- and importing *any* submodule of a package runs
its ``__init__``, so ``from loglead.delta import log_root`` used to pay all of
it before doing anything. That is the wrong bill for the cheap calls:
``peek_log_root`` stats files and reads a few hundred lines, and a process doing
only that now holds sklearn not at all.

Nothing changes for a caller: ``from loglead import AnomalyDetector``,
``import loglead; loglead.LogDistance`` and ``loglead.anomaly_detection`` all
work as before -- the import simply happens on first use.
"""

import importlib

#: Public name -> the submodule that defines it. The only thing an eager
#: ``__init__`` gave that this does not is an ImportError at import time rather
#: than at first use.
_LAZY = {
    "AnomalyDetector": ".anomaly_detection",
    "LogDistance": ".anomaly_detection",
    "OOV_detector": ".OOV_detector",
    "RarityModel": ".RarityModel",
    "NextEventPredictionNgram": ".next_event_prediction",
    "profile_columns": ".column_analyzer",
    "select_predictors": ".column_analyzer",
    "print_predictor_report": ".column_analyzer",
}

__all__ = list(_LAZY)


def __getattr__(name):
    try:
        module = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value  # imported once; every later access is a plain lookup
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
