"""fmedian subpackage: exposes the C-implemented fmedian function."""
from ._fmedian import fmedian  # loaded from compiled extension

__all__ = ["fmedian"]
