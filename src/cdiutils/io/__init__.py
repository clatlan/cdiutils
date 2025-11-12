"""
IO utilities for loading and saving BCDI data.

CXIExplorer is imported lazily to avoid requiring ipywidgets
for basic IO operations.
"""

from .cristal import CristalLoader
from .cxi import CXIFile, load_cxi, save_as_cxi
from .id01 import ID01Loader, SpecLoader
from .id27 import ID27Loader
from .loader import Loader, h5_safe_load
from .nanomax import NanoMAXLoader
from .p10 import P10Loader
from .sixs import SIXSLoader
from .vtk import save_as_vti

__all__ = [
    "Loader",
    "h5_safe_load",
    "ID01Loader",
    "ID27Loader",
    "P10Loader",
    "SpecLoader",
    "SIXSLoader",
    "CristalLoader",
    "NanoMAXLoader",
    "CXIFile",
    "CXIExplorer",
    "save_as_cxi",
    "load_cxi",
    "save_as_vti",
]


def __getattr__(name):
    """Lazy import for CXIExplorer to avoid ipywidgets dependency."""
    if name == "CXIExplorer":
        from .cxi_explorer import CXIExplorer

        return CXIExplorer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
