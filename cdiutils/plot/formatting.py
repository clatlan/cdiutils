from typing import Union

import colorcet
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
            "cmap": "cet_CET_C9s",
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
        style: str="nature",
        usetex: bool=True,
        use_siunitx: bool=False,
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

    if style in ("nature", "NATURE"):
        parameters =  {
            "lines.linewidth": 1,
            "lines.markersize": 1,
            "figure.titlesize": 8,
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 7,
        }
    elif style == "thesis":
         parameters =  {
            "lines.linewidth": 1,
            "lines.markersize": 1,
            "figure.titlesize": 12,
            "font.size": 8,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }

    matplotlib.pyplot.rcParams.update(
        **parameters
    )
    if use_siunitx:
        usetex = True
        matplotlib.pyplot.rcParams.update(
            **{
                'text.latex.preamble': (
                    r'\usepackage{siunitx}'
                    r'\sisetup{detect-all}'
                    r'\usepackage{helvet}'
                    + (
                        r'\usepackage{sansmath} \sansmath'
                        if style == "nature" else r'\usepackage{amsmath}'
                    )
                ) 
            }

    )
        #         'text.latex.preamble':
        #         r'\usepackage{siunitx}'
        #         r'\sisetup{detect-all}'
        #         r'\usepackage{helvet}'
        #         r'\usepackage{sansmath} \sansmath' if style == "nature" else r'\usepackage{amsmath}'
        #         # r'\sansmath'
        #     }
        # )

    if usetex:
        matplotlib.pyplot.rcParams.update(
            **{
                "mathtext.default": "regular",
                "text.usetex": usetex,
                "font.family":"sans-serif",
                "font.sans-serif":["Liberation Sans"]
            }
        )
    
    # in any case
    matplotlib.pyplot.rcParams.update(
        {
        "image.cmap": "turbo",
        "figure.dpi": 200
        }
    )

    # matplotlib.pyplot.rcParams.update(
    #     {
    #         "figure.max_open_warning": max_open_warning,
    #         "figure.dpi": dpi,
    #         "lines.marker": lines_marker,
    #         "mathtext.default": "regular",
    #         "text.usetex": usetex,
    #         # 'text.latex.preamble':
    #         #     r'\usepackage{siunitx}'
    #         #     r'\sisetup{detect-all}'
    #         #     r'\usepackage{helvet}'
    #         #     r'\usepackage{sansmath}'
    #         #     r'\sansmath' if usetex else "",
    #         "font.family":"sans-serif",
    #         "font.sans-serif":["Liberation Sans"],
    #         "lines.linewidth": lines_linewidth,
    #         "lines.linestyle": lines_linestyle,
    #         "lines.markersize": lines_markersize,
    #         "figure.titlesize": figure_titlesize,
    #         "font.size": font_size,
    #         "axes.titlesize": axes_titlesize,
    #         "axes.labelsize": axes_labelsize,
    #         "xtick.labelsize": xtick_labelsize,
    #         "ytick.labelsize": ytick_labelsize,
    #         "legend.fontsize": legend_fontsize,
    #         "image.cmap": "turbo"
    #     }
    # )
    matplotlib.pyplot.rcParams.update(kwargs)


def get_figure_size(
        width: int or str,
        fraction: float=1,
        subplots: tuple=(1, 1)
) -> tuple:
    """
    Get the figure dimensions to avoid scaling in LaTex.
    
    This function was copied from
    https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    :param width: Document width in points, or string of predefined
    document type (float or string)
    :param fraction: fraction of the width which you wish the figure to
    occupy (float)
    :param subplots: the number of rows and columns of subplots

    :return: dimensions of the figure in inches (tuple)
    """
    if width == 'thesis':
        width_pt = 455.30101
    elif width == 'beamer':
        width_pt = 398.3386
    elif width == "nature":
        width_pt = 518.74
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def two_spine_frameless_ax(
        ax: matplotlib.axes.Axes,
        left_spine_pos: float,
        bottom_spine_pos: float
) -> None:
    ax.spines["left"].set_position(("data", left_spine_pos))
    ax.spines["bottom"].set_position(("data", bottom_spine_pos))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot(
        left_spine_pos,
        1,
        "^k", 
        transform=ax.get_xaxis_transform(), 
        clip_on=False
    )
    ax.plot(
        1,
        bottom_spine_pos,
        ">k", 
        transform=ax.get_yaxis_transform(),
        clip_on=False
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