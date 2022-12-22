import matplotlib
import matplotlib.ticker as mticker
import numpy as np

ANGSTROM_SYMBOL = None
PERCENT_SYMBOL = None
PLOT_CONFIGS = None

def _set_configs():
    global ANGSTROM_SYMBOL
    global PERCENT_SYMBOL
    global PLOT_CONFIGS
    
    if matplotlib.rcParams["text.usetex"]:
        ANGSTROM_SYMBOL = "$\si{\angstrom}$"
        PERCENT_SYMBOL = "\%"
    else:
        ANGSTROM_SYMBOL = "\AA"
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
        "lattice_constant": {
            "title": fr"lattice constant ({ANGSTROM_SYMBOL})",
            "cmap": "turbo",
            "vmin": None,
            "vmax": None
        }
    }
    PLOT_CONFIGS["local_strain"] = PLOT_CONFIGS["strain"].copy()
    PLOT_CONFIGS["numpy_local_strain"] = PLOT_CONFIGS["strain"].copy()
    PLOT_CONFIGS["numpy_local_strain"]["title"] = (
        fr"numpy strain ({PERCENT_SYMBOL})"
    )
    PLOT_CONFIGS["local_strain_from_dspacing"] = PLOT_CONFIGS["strain"].copy()
    PLOT_CONFIGS["local_strain_from_dspacing"]["title"] = (
        fr"strain from dspacing ({PERCENT_SYMBOL})"
    )
    PLOT_CONFIGS["local_strain_with_ramp"] = PLOT_CONFIGS["strain"].copy()
    PLOT_CONFIGS["local_strain_with_ramp"]["title"] = (
        fr"strain with ramp ({PERCENT_SYMBOL})"
    )

_set_configs()



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
        **kwargs
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
    _set_configs()


def plot_background(
        ax: matplotlib.axes.Axes,
        grey_background_opacity=0
) -> matplotlib.axes.Axes:
    """Plot a grey background and a grid"""

    ax.grid(True, linestyle="--", linewidth=0.5, zorder=0)
    ax.patch.set_facecolor("lightgrey")
    ax.patch.set_alpha(grey_background_opacity)
    return ax


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
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)