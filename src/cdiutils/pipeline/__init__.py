"""
Pipeline module.

Implement the engines to run the pipelines.
"""

import importlib

from .base import Pipeline
from .bcdi import BcdiPipeline
from .parameters import get_params_from_variables
from .pipeline_plotter import PipelinePlotter

__class_func_submodules__ = {
    "Pipeline": "base",
    "BcdiPipeline": "bcdi",
    "PipelinePlotter": "pipeline_plotter",
    "get_params_from_variables": "parameters",
}

__all__ = [
    "Pipeline",
    "BcdiPipeline",
    "PipelinePlotter",
    "get_params_from_variables",
]


def __getattr__(name):
    if name in __class_func_submodules__:
        submodule = importlib.import_module(
            f"{__name__}.{__class_func_submodules__[name]}"
        )
        return getattr(submodule, name)
    raise AttributeError(f"module {__name__} has no attribute {name}.")
