import importlib

from .colormap import PARULA, RED_TO_TEAL, TURBO_FIRST_HALF, TURBO_SECOND_HALF
from .formatting import (
    add_colorbar,
    add_labels,
    get_extent,
    get_figure_size,
    get_plot_configs,
    get_x_y_limits_extents,
    save_fig,
    set_plot_configs,
    set_x_y_limits_extents,
    update_plot_params,
    white_interior_ticks_labels,
    x_y_lim_from_support,
)

__submodules__ = {"slice", "volumeinteractive", "stats"}

__class_func_submodules__ = {
    "Plotter": "interactive",
    "plot_volume_slices": "slice",
    "plot_multiple_volume_slices": "slice",
    "VolumeViewer": "volume",
    "plot_histogram": "stats",
    "strain_statistics": "stats",
}

__all__ = [
    "update_plot_params",
    "get_figure_size",
    "set_plot_configs",
    "get_plot_configs",
    "get_extent",
    "add_colorbar",
    "x_y_lim_from_support",
    "get_x_y_limits_extents",
    "set_x_y_limits_extents",
    "add_labels",
    "save_fig",
    "white_interior_ticks_labels",
    "PARULA",
    "RED_TO_TEAL",
    "TURBO_FIRST_HALF",
    "TURBO_SECOND_HALF",
]
__all__ += list(__submodules__) + list(__class_func_submodules__)


def __getattr__(name):
    if name in __submodules__:
        return importlib.import_module(f"{__name__}.{name}")

    if name in __class_func_submodules__:
        submodule = importlib.import_module(
            f"{__name__}.{__class_func_submodules__[name]}"
        )
        return getattr(submodule, name)
    raise AttributeError(f"module {__name__} has no attribute {name}.")
