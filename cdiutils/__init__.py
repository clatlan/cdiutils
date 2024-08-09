"""
cdiutils - A Python package for (Bragg) Coherent X-ray Diffraction
Imaging processing, analysis and visualisation workflows.
"""

__version__ = "0.2.0"
__author__ = "Clément Atlan"
__email__ = "clement.atlan@esrf.fr"
__license__ = "MIT"


import importlib

from . import utils
from .utils import (
    energy_to_wavelength,
    make_support,
    get_centred_slices,
    CroppingHandler
)

from .pipeline import BcdiPipeline
from . import load, plot, process
from .geometry import Geometry
from .converter import SpaceConverter


__submodules__ = {
    "utils",
    "geometry",
    "converter",
    "load",
    "process",
    "pipeline",
    "plot"
}


__all__ = list(
    __submodules__ |
    {
        "Geometry", "SpaceConverter", "energy_to_wavelength", "make_support",
        "get_centred_slices", "CroppingHandler", "BcdiPipeline"
    }
)


def __getattr__(name):
    if name in __submodules__:
        return importlib.import_module(f'{__name__}.{name}')
    raise AttributeError(f"module {__name__} has no attribute {name}.")
