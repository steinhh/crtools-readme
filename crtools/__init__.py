"""Top-level package for crtools

Exports the two C-implemented functions `fmedian` and `fsigma` at package top-level:

from .fmedian import fmedian
from .fsigma import fsigma

__all__ = ["fmedian", "fsigma"]
"""

from .fmedian import fmedian
from .fsigma import fsigma

__all__ = ["fmedian", "fsigma"]
