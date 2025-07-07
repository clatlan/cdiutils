"""
cdiutils - A Python package for (Bragg) Coherent X-ray Diffraction
Imaging processing, analysis and visualisation workflows.
"""

__version__ = "0.2.0"
__author__ = "Clément Atlan"
__email__ = "clement.atlan@esrf.fr"
__license__ = "MIT"


import importlib

from .utils import (
    energy_to_wavelength,
    wavelength_to_energy,
    make_support,
    get_centred_slices,
    hot_pixel_filter,
    CroppingHandler
)

__submodules__ = {
    "utils",
    "analysis",
    "geometry",
    "converter",
    "probe",
    "io",
    "process",
    "pipeline",
    "plot"
}

__class_submodules__ = {
    "Geometry": "geometry",
    "SpaceConverter": "converter",
    "BcdiPipeline": "pipeline",
    "Loader": "io",
    "CXIFile": "io"
}

__function_submodules__ = {
    "update_plot_params": "plot",
}
__all__ = [
    "energy_to_wavelength", "wavelength_to_energy", "make_support",
    "get_centred_slices", "CroppingHandler", "hot_pixel_filter"
]
__all__ += (
    list(__submodules__)
    + list(__class_submodules__)
    + list(__function_submodules__)
)


def __getattr__(name):
    # Lazy load submodules
    if name in __submodules__:
        return importlib.import_module(f"{__name__}.{name}")

    # Lazy load specific classes
    if name in __class_submodules__:
        submodule = importlib.import_module(
            f"{__name__}.{__class_submodules__[name]}"
        )
        return getattr(submodule, name)

    # Lazy load specific functions
    if name in __function_submodules__:
        submodule = importlib.import_module(
            f"{__name__}.{__function_submodules__[name]}"
        )
        return getattr(submodule, name)

    raise AttributeError(f"module {__name__} has no attribute {name}.")
