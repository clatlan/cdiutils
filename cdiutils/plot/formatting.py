import matplotlib
import matplotlib.ticker as mticker
import numpy as np


def set_plot_configs():

    ANGSTROM_SYMBOL = None
    PERCENT_SYMBOL = None
    PLOT_CONFIGS = None
    
    
    if matplotlib.rcParams["text.usetex"]:
        ANGSTROM_SYMBOL = r"$\si{\angstrom}$"
        PERCENT_SYMBOL = r"\%"
    else:
        ANGSTROM_SYMBOL = r"$\AA$"
        PERCENT_SYMBOL = "%"

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
            "title": fr"Displacement ({ANGSTROM_SYMBOL})",
            "cmap": "cet_CET_D1A",
            "vmin": -0.1,
            "vmax": 0.1,
        },
        "strain": {
            "title": fr"Strain ({PERCENT_SYMBOL})",
            "cmap": "cet_CET_D13",
            "vmin": -0.05,
            "vmax": 0.05,
        },
        "dspacing": {
            "title": fr"dspacing ({ANGSTROM_SYMBOL})",
            "cmap": "turbo",
            "vmin": None,
            "vmax": None,
        },
        "lattice_parameter": {
            "title": fr"Lattice parameter ({ANGSTROM_SYMBOL})",
            "cmap": "turbo",
            "vmin": None,
            "vmax": None
        }
    }
    PLOT_CONFIGS["local_strain"] = PLOT_CONFIGS["strain"].copy()
    PLOT_CONFIGS["numpy_local_strain"] = PLOT_CONFIGS["strain"].copy()
    PLOT_CONFIGS["numpy_local_strain"]["title"] = (
        fr"Numpy strain ({PERCENT_SYMBOL})"
    )
    PLOT_CONFIGS["local_strain_from_dspacing"] = PLOT_CONFIGS["strain"].copy()
    PLOT_CONFIGS["local_strain_from_dspacing"]["title"] = (
        fr"Strain from dspacing ({PERCENT_SYMBOL})"
    )
    PLOT_CONFIGS["local_strain_with_ramp"] = PLOT_CONFIGS["strain"].copy()
    PLOT_CONFIGS["local_strain_with_ramp"]["title"] = (
        fr"Strain with ramp ({PERCENT_SYMBOL})"
    )
    return ANGSTROM_SYMBOL, PERCENT_SYMBOL, PLOT_CONFIGS

# ANGSTROM_SYMBOL, PERCENT_SYMBOL, PLOT_CONFIGS = set_plot_configs()



def update_plot_params(
        usetex: bool=True,
        max_open_warning: int=100,
        dpi: int=200,
        lines_marker: str="",
        lines_linewidth: str=2.5,
        lines_linestyle: str="-",
        lines_markersize: int=7,
        figure_titlesize: int=22,
        font_size: int=16,
        axes_titlesize: int=16,
        axes_labelsize: int=16,
        xtick_labelsize: int=12,
        ytick_labelsize: int=12,
        legend_fontsize: int=10,
        **kwargs
) -> None:
    """Update the matplotlib plot parameters to plublication style"""
    matplotlib.pyplot.rcParams.update(
        {
            "figure.max_open_warning": max_open_warning,
            "figure.dpi": dpi,
            "lines.marker": lines_marker,
            "mathtext.default": "regular",
            "text.usetex": usetex,
            'text.latex.preamble':
                r'\usepackage{siunitx}'
                r'\sisetup{detect-all}'
                r'\usepackage{helvet}'
                r'\usepackage{sansmath}'
                r'\sansmath' if usetex else "",
            "lines.linewidth": lines_linewidth,
            "lines.linestyle": lines_linestyle,
            "lines.markersize": lines_markersize,
            "figure.titlesize": figure_titlesize,
            "font.size": font_size,
            "axes.titlesize": axes_titlesize,
            "axes.labelsize": axes_labelsize,
            "xtick.labelsize": xtick_labelsize,
            "ytick.labelsize": ytick_labelsize,
            "legend.fontsize": legend_fontsize,
            "image.cmap": "turbo"
        }
    )
    matplotlib.pyplot.rcParams.update(kwargs)


def plot_background(
        ax: matplotlib.axes.Axes,
        grey_background_opacity=0
) -> matplotlib.axes.Axes:
    """Plot a grey background and a grid"""

    ax.grid(True, linestyle="--", linewidth=0.5, zorder=0)
    ax.patch.set_facecolor("lightgrey")
    ax.patch.set_alpha(grey_background_opacity)
    return ax


def white_interior_ticks_labels(
        ax: matplotlib.axes.Axes,
        xtick_pad: int=-15,
        ytick_pad: int=-25
) -> None:
    """Place the ticks and labels inside the provided axis."""
    ax.tick_params(axis="x", direction="in", pad=xtick_pad, colors="w")
    ax.tick_params(axis="y", direction="in", pad=ytick_pad, colors="w")
    ax.xaxis.set_ticks_position("bottom")

    xticks_loc, yticks_loc = ax.get_xticks(), ax.get_yticks()
    xticks_loc[1] = yticks_loc[1] = None
    
    xlabels, ylabels = ax.get_xticklabels(), ax.get_yticklabels()
    xlabels[1] = ylabels[1] = ""
    ax.xaxis.set_major_locator(mticker.FixedLocator(xticks_loc))
    ax.yaxis.set_major_locator(mticker.FixedLocator(yticks_loc))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)


class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = f"10^{sign, exponent}"
        if significand and exponent:
            s =  fr"{significand}\times{exponent}"
        else:
            s =  fr"{significand, exponent}"
        return f"${s}$"