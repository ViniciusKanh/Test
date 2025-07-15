"""Expose associative classifier implementations."""

from ._marca import ModularClassifier
from ._cba import CBA

# Backwards compatibility: some tests expect ``MARCA`` to be available.  It is
# just an alias to :class:`ModularClassifier` used throughout the code base.
MARCA = ModularClassifier

__all__ = ["ModularClassifier", "CBA", "MARCA"]
