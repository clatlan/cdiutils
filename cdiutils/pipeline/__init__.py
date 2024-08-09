"""
Pipeline module.

Implement the engines to run the pipelines.
"""

import importlib

from .bcdi_pipeline import BcdiPipeline
from .pipeline_plotter import PipelinePlotter
from .parameters import get_parameters_from_notebook_variables


__class_func_submodules__ = {
    "BcdiPipeline": "bcdi_pipeline",
    "PipelinePlotter": "pipeline_plotter",
    "get_parameters_from_notebook_variables": "parameters"
}

__all__ = list(__class_func_submodules__)

def __getattr__(name):
    if name in __class_func_submodules__:
        submodule = importlib.import_module(
            f"{__name__}.{__class_func_submodules__[name]}"
        )
        return getattr(submodule, name)
    raise AttributeError(f"module {__name__} has no attribute {name}.")
