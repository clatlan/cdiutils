import matplotlib
import numpy as np
from typing import Union
import xrayutilities as xu

# from cdiutils.utils import nan_to_zero

PLOT_CONFIGS = {
    "amplitude":{
        "title": "Amplitude (a.u.)",
        "cmap": "turbo",
        "vmin": 0,
        "vmax": 1
    },
    "phase": {
        "title": "Phase (rad)",
        "cmap": "turbo",
        "vmin": -np.pi / 8,
        "vmax": np.pi / 8,
    },
    "displacement": {
        "title": r"Displacement ($\AA$)",
        "cmap": "cet_CET_D1A",
        "vmin": -0.1,
        "vmax": 0.1,
    },
    "strain": {
        "title": "Strain (%)",
        "cmap": "cet_CET_D13",
        "vmin": -0.05,
        "vmax": 0.05,
    },
    "dspacing": {
        "title": r"dspacing ($\AA$)",
        "cmap": "turbo",
        "vmin": None,
        "vmax": None,
    },
    "lattice_constant": {
        "title": r"lattice constant ($\AA$)",
        "cmap": "turbo",
        "vmin": None,
        "vmax": None,
    }
}
PLOT_CONFIGS["local_strain"] = PLOT_CONFIGS["strain"].copy()
PLOT_CONFIGS["numpy_local_strain"] = PLOT_CONFIGS["strain"].copy()
PLOT_CONFIGS["numpy_local_strain"]["title"] = "numpy strain (%)"


def update_plot_params(
        max_open_warning=100,
        dpi=200,
        lines_marker="",
        lines_linewidth=2.5,
        lines_linestyle="-",
        lines_markersize=7,
        figure_titlesize=22,
        font_size=16,
        axes_titlesize=16,
        axes_labelsize=16,
        xtick_labelsize=12,
        ytick_labelsize=12,
        legend_fontsize=10,
        # **kwargs
):
    matplotlib.pyplot.rcParams.update(
        {
            "figure.max_open_warning": max_open_warning,
            "figure.dpi": dpi,
            "lines.marker": lines_marker,
            "text.usetex": True,
            'text.latex.preamble':
                r'\usepackage{siunitx}'
                r'\sisetup{detect-all}'
                r'\usepackage{helvet}'
                r'\usepackage{sansmath}'
                r'\sansmath',
            "lines.linewidth": lines_linewidth,
            "lines.linestyle": lines_linestyle,
            "lines.markersize": lines_markersize,
            "figure.titlesize": figure_titlesize,
            "font.size": font_size,
            "axes.titlesize": axes_titlesize,
            "axes.labelsize": axes_labelsize,
            "xtick.labelsize": xtick_labelsize,
            "ytick.labelsize": ytick_labelsize,
            "legend.fontsize": legend_fontsize
        }
    )


def plot_background(
        ax: matplotlib.axes.Axes,
        grey_background_opacity=0
) -> matplotlib.axes.Axes:
    """Plot a grey background and a grid"""

    ax.grid(True, linestyle="--", linewidth=0.5, zorder=0)
    ax.patch.set_facecolor("lightgrey")
    ax.patch.set_alpha(grey_background_opacity)
    return ax
