"""fsigma subpackage: exposes the C-implemented fsigma function."""
from ._fsigma import fsigma  # loaded from compiled extension

__all__ = ["fsigma"]
