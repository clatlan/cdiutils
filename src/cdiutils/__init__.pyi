"""
Type stubs for cdiutils package.

This file is ONLY used by type checkers (Pylance, mypy, etc.) to understand
what names are available in the cdiutils namespace. It is NEVER executed at runtime.

At runtime, __init__.py uses __getattr__ for lazy loading.
At type-check time, this .pyi file tells the type checker what's available.
"""

# version info (type annotations only)
__version__: str
__author__: str
__email__: str
__license__: str

# re-exported utilities
# these are actually imported in __init__.py, not lazy loaded
from .utils import (
    energy_to_wavelength as energy_to_wavelength,
    wavelength_to_energy as wavelength_to_energy,
    make_support as make_support,
    get_centred_slices as get_centred_slices,
    hot_pixel_filter as hot_pixel_filter,
    CroppingHandler as CroppingHandler,
)

# submodules
# these imports tell the type checker they exist, but don't run at runtime
from . import utils as utils
from . import analysis as analysis
from . import geometry as geometry
from . import converter as converter
from . import wavefront as wavefront
from . import io as io
from . import process as process
from . import pipeline as pipeline
from . import plot as plot

# classes
from .geometry import Geometry as Geometry
from .converter import SpaceConverter as SpaceConverter
from .pipeline.bcdi import BcdiPipeline as BcdiPipeline
from .io.loader import Loader as Loader
from .io.cxi_protocol import CXIFile as CXIFile

# functions
from .plot import update_plot_params as update_plot_params
