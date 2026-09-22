"""Deprecated alias for :mod:`loglead.rarity_detector`.

Kept so existing callers (including sibling projects) keep working; new code should use
``loglead.rarity_detector`` instead. Will be removed in a future release.
"""

import warnings

from .rarity_detector import rarity_detector

__all__ = ['RarityModel']


class RarityModel(rarity_detector):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "RarityModel is deprecated, use rarity_detector instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
