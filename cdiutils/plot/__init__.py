import importlib

from .formatting import update_plot_params
from .formatting import get_figure_size
from .formatting import set_plot_configs
from .formatting import get_plot_configs
from .formatting import get_extent
from .formatting import add_colorbar
from .formatting import x_y_lim_from_support
from .formatting import get_x_y_limits_extents
from .formatting import set_x_y_limits_extents
from .formatting import add_labels
from .formatting import save_fig
from .formatting import white_interior_ticks_labels


__submodules__ = {
    "slice",
    "volume"
    "interactive",
}

__class_func_submodules__ = {
    "Plotter": "interactive",
    "plot_volume_slices": "slice",
    "VolumeViewer": "volume"
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
]
__all__ += list(__submodules__) + list(__class_func_submodules__)


def __getattr__(name):
    if name in __submodules__:
        return importlib.import_module(f'{__name__}.{name}')

    if name in __class_func_submodules__:
        submodule = importlib.import_module(
            f"{__name__}.{__class_func_submodules__[name]}"
        )
        return getattr(submodule, name)
    raise AttributeError(f"module {__name__} has no attribute {name}.")
