# flake8: noqa, E501
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import colorcet  # noqa, F401


# Planes are given with the indexing convention,
# i.e. [n, m] -> x-axis = m, y-axis = n
CXI_VIEW_PARAMETERS = {
   "z+": {
       "axis": 0, "plane": [1, 2], "xaxis_points_left": True,
       "xlabel": r"$x_{\mathrm{CXI}}$ or $y_{\mathrm{XU}}$",
       "ylabel": r"$y_{\mathrm{CXI}}$ or $z_{\mathrm{XU}}$",
       "qxlabel": r"q$_{x, \mathrm{CXI}}$ or q$_{y, \mathrm{XU}}$",
       "qylabel": r"q$_{y, \mathrm{CXI}}$ or q$_{z, \mathrm{XU}}$"
    },
   "z-": {
       "axis": 0, "plane": [1, 2], "xaxis_points_left": False,
       "xlabel": r"$x_{\mathrm{CXI}}$ or $y_{\mathrm{XU}}$",
       "ylabel": r"$y_{\mathrm{CXI}}$ or $z_{\mathrm{XU}}$",
       "qxlabel": r"q$_{x, \mathrm{CXI}}$ or q$_{y, \mathrm{XU}}$",
       "qylabel": r"q$_{y, \mathrm{CXI}}$ or q$_{z, \mathrm{XU}}$"
    },
   "y+": {
       "axis": 1, "plane": [0, 2], "xaxis_points_left": False,
       "xlabel": r"$x_{\mathrm{CXI}}$ or $y_{\mathrm{XU}}$",
       "ylabel": r"$z_{\mathrm{CXI}}$ or $x_{\mathrm{XU}}$",
       "qxlabel": r"q$_{x, \mathrm{CXI}}$ or q$_{y, \mathrm{XU}}$",
       "qylabel": r"q$_{z, \mathrm{CXI}}$ or q$_{x, \mathrm{XU}}$"
    },
   "y-": {
       "axis": 1, "plane": [0, 2], "xaxis_points_left": True,
       "xlabel": r"$x_{\mathrm{CXI}}$ or $y_{\mathrm{XU}}$",
       "ylabel": r"$z_{\mathrm{CXI}}$ or $x_{\mathrm{XU}}$",
       "qxlabel": r"q$_{x, \mathrm{CXI}}$ or q$_{y, \mathrm{XU}}$",
       "qylabel": r"q$_{z, \mathrm{CXI}}$ or q$_{x, \mathrm{XU}}$"
    },
   "x+": {
       "axis": 2, "plane": [1, 0], "xaxis_points_left": False,
       "xlabel": r"$z_{\mathrm{CXI}}$ or $x_{\mathrm{XU}}$",
       "ylabel": r"$y_{\mathrm{CXI}}$ or $z_{\mathrm{XU}}$",
       "qxlabel": r"q$_{z, \mathrm{CXI}}$ or q$_{x, \mathrm{XU}}$",
       "qylabel": r"q$_{y, \mathrm{CXI}}$ or q$_{z, \mathrm{XU}}$"
    },
   "x-": {
       "axis": 2, "plane": [1, 0], "xaxis_points_left": True,
       "xlabel": r"$z_{\mathrm{CXI}}$ or $x_{\mathrm{XU}}$",
       "ylabel": r"$y_{\mathrm{CXI}}$ or $z_{\mathrm{XU}}$",
       "qxlabel": r"q$_{z, \mathrm{CXI}}$ or q$_{x, \mathrm{XU}}$",
       "qylabel": r"q$_{y, \mathrm{CXI}}$ or q$_{z, \mathrm{XU}}$"
    },
}

# Planes are given with the indexing convention,
# i.e. [n, m] -> x-axis = m, y-axis = n
XU_VIEW_PARAMETERS = {
   "x+": {
       "axis": 0, "plane": [2, 1], "xaxis_points_left": True,
       "xlabel": r"$y_{\mathrm{XU}}$ or $x_{\mathrm{CXI}}$",
       "ylabel": r"$z_{\mathrm{XU}}$ or $y_{\mathrm{CXI}}$",
       "qxlabel": r"q$_{y, \mathrm{XU}}$ or q$_{x, \mathrm{CXI}}$",
       "qylabel": r"q$_{z, \mathrm{XU}}$ or q$_{y, \mathrm{CXI}}$"
    },
   "x-": {
       "axis": 0, "plane": [2, 1], "xaxis_points_left": False,
       "xlabel": r"$y_{\mathrm{XU}}$ or $x_{\mathrm{CXI}}$",
       "ylabel": r"$z_{\mathrm{XU}}$ or $y_{\mathrm{CXI}}$",
       "qxlabel": r"q$_{y, \mathrm{XU}}$ or q$_{x, \mathrm{CXI}}$",
       "qylabel": r"q$_{z, \mathrm{XU}}$ or q$_{y, \mathrm{CXI}}$"
    },
   "y+": {
       "axis": 1, "plane": [2, 0], "xaxis_points_left": True,
       "xlabel": r"$x{\mathrm{XU}}$ or $z_{\mathrm{CXI}}$",
       "ylabel": r"$z_{\mathrm{XU}}$ or $y_{\mathrm{CXI}}$",
       "qxlabel": r"q$_{x, \mathrm{XU}}$ or q$_{z, \mathrm{CXI}}$",
       "qylabel": r"q$_{z, \mathrm{XU}}$ or q$_{y, \mathrm{CXI}}$"
    },
   "y-": {
       "axis": 1, "plane": [2, 0], "xaxis_points_left": False,
       "xlabel": r"$x_{\mathrm{XU}}$ or $z_{\mathrm{CXI}}$",
       "ylabel": r"$z_{\mathrm{XU}}$ or $y_{\mathrm{CXI}}$",
       "qxlabel": r"q$_{x, \mathrm{XU}}$ or q$_{z, \mathrm{CXI}}$",
       "qylabel": r"q$_{z, \mathrm{XU}}$ or q$_{y, \mathrm{CXI}}$"
    },
   "z+": {
       "axis": 2, "plane": [1, 0], "xaxis_points_left": True,
       "xlabel": r"$x_{\mathrm{XU}}$ or $z_{\mathrm{CXI}}$",
       "ylabel": r"$y_{\mathrm{XU}}$ or $x_{\mathrm{CXI}}$",
       "qxlabel": r"q$_{x, \mathrm{XU}}$ or q$_{z, \mathrm{CXI}}$",
       "qylabel": r"q$_{y, \mathrm{XU}}$ or q$_{x, \mathrm{CXI}}$"
    },
   "z-": {
       "axis": 2, "plane": [1, 0], "xaxis_points_left": False,
       "xlabel": r"$x_{\mathrm{XU}}$ or $z_{\mathrm{CXI}}$",
       "ylabel": r"$y_{\mathrm{XU}}$ or $x_{\mathrm{CXI}}$",
       "qxlabel": r"q$_{x, \mathrm{XU}}$ or q$_{z, \mathrm{CXI}}$",
       "qylabel": r"q$_{y, \mathrm{XU}}$ or q$_{x, \mathrm{CXI}}$"
    },
}


# Planes are given with the indexing convention,
# i.e. [n, m] -> x-axis = m, y-axis = n
NATURAL_VIEW_PARAMETERS = {
    "dim0": {"axis": 0, "plane": [1, 2], "xaxis_points_left": False},
    "dim1": {"axis": 1, "plane": [0, 2], "xaxis_points_left": False},
    "dim2": {"axis": 2, "plane": [0, 1], "xaxis_points_left": False}
}


def save_fig(fig: plt.Figure, path: str, **kwargs) -> None:
    default_params = {
        "bbox_inches": "tight",
        "dpi": 200,
        "transparent": True
    }
    default_params.update(kwargs)
    fig.savefig(path, **default_params)


def add_labels(
        axes: plt.Axes,
        views: tuple[str] = None,
        space: str = "direct",
        convention: str = "cxi",
        unit: str = None
) -> None:
    if convention.lower() in ("xu", "lab"):
        view_params = XU_VIEW_PARAMETERS.copy()
        if views is None:
            views = ("x-", "y-", "z-")
    elif convention.lower() == "cxi":
        view_params = CXI_VIEW_PARAMETERS.copy()
        if views is None:
            views = ("z-", "y+", "x+")
    else:
        raise ValueError(f"Invalid convention ({convention}).")

    if len(axes) != len(views):
        raise ValueError(
            "axes and views must have the same length "
            f"(len(axes) = {len(axes)} != len(views) = {len(views)})"
        )

    if space.lower() in ("reciprocal", "rcp"):
        xlabel_key = "qxlabel"
        ylabel_key = "qylabel"
    elif space.lower() in ("direct", "dr", "drct", "drt"):
        xlabel_key = "xlabel"
        ylabel_key = "ylabel"
    else:
        raise ValueError(f"Invalid space name ({space}).")

    if unit is None:
        unit = " (nm)"
        if space.lower() in ("reciprocal", "rcp"):
            unit = r" ($\mathrm{\AA^{-1}}$)"
    elif not unit.startswith(" ("):
        unit = f" ({unit})"

    for ax, v in zip(axes.flat, views):
        ax.set_xlabel(view_params[v][xlabel_key] + unit)
        ax.set_ylabel(view_params[v][ylabel_key] + unit)


def get_x_y_limits_extents(
        shape,
        voxel_size,
        data_centre=None,
        equal_limits: bool = False
):
    shape = np.array(shape)
    voxel_size = np.array(voxel_size)

    extents = np.array(shape) * np.array(voxel_size)

    if equal_limits:
        # Must be used only for limits !
        extents = np.repeat(np.max(extents), len(shape))

    if data_centre is None:
        return [(0, e) for e in extents]
    return [(c - e/2, c + e/2) for c, e in zip(data_centre, extents)]


def set_x_y_limits_extents(
        ax: plt.Axes,
        extents: list | tuple,
        limits: list | tuple,
        plane: list | tuple,
        xaxis_points_left: bool = False
) -> None:
    image = ax.images[0]
    image.origin = "lower"
    extent = extents[plane[1]] + extents[plane[0]]
    if xaxis_points_left:
        extent = (extent[1], extent[0], *extent[2:])
    image.set_extent(extent)
    if xaxis_points_left:
        ax.set_xlim(limits[plane[1]][1], limits[plane[1]][0])
    else:
        ax.set_xlim(limits[plane[1]])
    ax.set_ylim(limits[plane[0]])


def x_y_lim_from_support(
        support: np.ndarray,
        pixel_size: tuple = (1, 1),
        central_pixel: tuple = None,
        pad: tuple = (-10, 10)
) -> list:
    """
    Return the x and y limits of the a plot using support constraints.
    The plot will be limited to the support dimension + the pad.

    Args:
        support (np.ndarray): the support to get the limits from.
        pixel_size (tuple, optional): the pixel size. Defaults to (1, 1).
        central_pixel (tuple, optional): the position of the central
            pixel. This matters only if extent/aspect/pixel size are
            specific. In this case, the user might want to specify where
            to centre the plotting at.
        pad (tuple, optional): the space between the limits found from
            the support limits and the ax frame. Defaults to (-5, 5).

    Returns:
        list: the x_limits and y_limits.
    """
    if support.sum() > 0:
        pad = np.array(pad) * np.array(pixel_size)
        lims = []
        for i in range(2):
            lim = np.nonzero(support.sum(axis=i))[0][[0, -1]] * pixel_size[0]
            lim += np.array(pad)
            if central_pixel:
                lim -= (lim.mean() - central_pixel[i])
            lims.append(lim)
        return lims
    return None


def get_extent(
        shape: tuple,
        voxel_size: tuple | list | np.ndarray,
        plane: list,
        zero_centred: bool = True,
        indexing: str = "ij"
) -> tuple:
    """Find the extents for matshow/imshow plotting, for a given plane.
    Note that in matlotlib convention, the extent must be provided in
    the order x, y, but the imshow function plot axis0 along y and axis1
    along x. Therefore, plane and indexing must be chosen appropriately. 

    Args:
        shape (tuple): the shape of the data to plot.
            voxel_size (tuple | list | np.ndarray): the voxel size of
            the data to plot.
        voxel_size (tuple | list | np.ndarray): the voxel size of
            the data to plot.
        plane (list): what plane to get the extents from. Should be a
            list of 2 axis integers.
        zero_centred (bool, optional): whether the plot must be
            zero_centred at zero. Defaults to True.
        indexing (str): the indexing convention. If 'xy', plane[0] and
            plane[1] must correspond to x and y respectively. If 'ij',
            plane[0] and plane[1] must correspond to y and x,
            respectively (numpy/matrix convent).

    Returns:
        tuple: first two values correspond to x-axis extent, last two
            to the y-axis extent in the matshow/imshow plot.
    """
    if isinstance(voxel_size, (int, float)):
        voxel_size = np.repeat(voxel_size, len(shape))
    absolute_extents = np.array(voxel_size) * shape / (2 if zero_centred else 1)
    if indexing == "xy":
        return (
            -absolute_extents[plane[0]] if zero_centred else 0,
            absolute_extents[plane[0]],
            -absolute_extents[plane[1]] if zero_centred else 0,
            absolute_extents[plane[1]],
        )
    if indexing == "ij":
        return (
            -absolute_extents[plane[1]] if zero_centred else 0,
            absolute_extents[plane[1]],
            -absolute_extents[plane[0]] if zero_centred else 0,
            absolute_extents[plane[0]],
        )


def get_plot_configs(key: str) -> dict:
    """
    Get the plotting configurations according to the provided key. If
    the key matches the generic PLOT_CONFIGS, the configurations are
    returned, otherwise error is raised.

    Args:
        key (str): the key word used for accessing the configurations.

    Raises:
        ValueError: if the key does not match any generic keys.

    Returns:
        dict: the plotting configurations.
    """
    _, _, PLOT_CONFIGS = set_plot_configs()
    for k in PLOT_CONFIGS.keys():
        if k in key:
            return PLOT_CONFIGS[k].copy()
    raise ValueError(f"Invalid key ({key}).")


def set_plot_configs():

    ANGSTROM_SYMBOL = None
    PERCENT_SYMBOL = None
    PLOT_CONFIGS = None

    if plt.rcParams["text.usetex"]:
        ANGSTROM_SYMBOL = r"$\si{\angstrom}$"
        PERCENT_SYMBOL = r"\%"
    else:
        ANGSTROM_SYMBOL = r"$\AA$"
        PERCENT_SYMBOL = "%"

    PLOT_CONFIGS = {
        "amplitude": {
            "title": "Amplitude (a.u.)",
            "cmap": "turbo",
            "vmin": 0,
            "vmax": None
        },
        "support": {
            "title": "Support (a.u.)",
            "cmap": "turbo",
            "vmin": 0,
            "vmax": 1
        },
        "intensity": {
            "title": "Intensity (a.u.)",
            "cmap": "turbo",
            "vmin": 0,
            "vmax": None
        },
        "phase": {
            "title": "Phase (rad)",
            "cmap": "cet_CET_C9s_r",
            "vmin": -np.pi,
            "vmax": np.pi,
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
        "displacement_gradient": {
            "title": r"Displacement gradient",
            "cmap": "Spectral_r",
            "vmin": -5 * 1e-4,
            "vmax": 5 * 1e-4
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
    PLOT_CONFIGS["het_strain"] = PLOT_CONFIGS["strain"].copy()
    PLOT_CONFIGS["numpy_het_strain"] = PLOT_CONFIGS["strain"].copy()
    PLOT_CONFIGS["numpy_het_strain"]["title"] = (
        fr"Numpy strain ({PERCENT_SYMBOL})"
    )
    PLOT_CONFIGS["het_strain_from_dspacing"] = PLOT_CONFIGS["strain"].copy()
    PLOT_CONFIGS["het_strain_from_dspacing"]["title"] = (
        fr"Strain from dspacing ({PERCENT_SYMBOL})"
    )
    PLOT_CONFIGS["het_strain_with_ramp"] = PLOT_CONFIGS["strain"].copy()
    PLOT_CONFIGS["het_strain_with_ramp"]["title"] = (
        fr"Strain with ramp ({PERCENT_SYMBOL})"
    )
    return ANGSTROM_SYMBOL, PERCENT_SYMBOL, PLOT_CONFIGS

# ANGSTROM_SYMBOL, PERCENT_SYMBOL, PLOT_CONFIGS = set_plot_configs()


def update_plot_params(
        style: str = "default",
        usetex: bool = False,
        use_siunitx: bool = True,
        **kwargs
) -> None:
    """Update the matplotlib plot parameters to plublication style"""

    if style in ("default", "nature"):
        parameters = {
            "lines.linewidth": 1,
            "lines.markersize": 1,
            "figure.titlesize": 8,
            "font.size": 7,
            "svg.fonttype": "none",
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 7,
            "image.interpolation": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
            # "font.sans-serif": "DejaVu Sans",
            "figure.figsize": (4.5, 3.0)
        }
    elif style == "thesis":
        parameters = {
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
    plt.rcParams.update(parameters)
    if usetex:
        if use_siunitx:
            plt.rcParams.update(
                {
                    'text.latex.preamble': (
                        r'\usepackage{siunitx}'
                        r'\sisetup{detect-all}'
                        r'\usepackage{helvet}'
                        + (
                            r'\usepackage{sansmath} \sansmath'
                            r'\usepackage{textgreek}'
                            if style in ("default", "nature")
                            else r'\usepackage{amsmath}'
                        )
                    ),
                    "text.usetex": True
                }
            )
        else:
            plt.rcParams.update(
                {
                    "text.usetex": usetex,
                    "text.latex.preamble": "",
                    "mathtext.default": "regular",
                    "font.family": "sans-serif",
                    "font.sans-serif": ["Liberation Sans"]
                }
            )

    # in any case
    plt.rcParams.update(
        {
            "image.cmap": "turbo",
            "figure.dpi": 200,
        }
    )
    plt.rcParams.update(**kwargs)


def get_figure_size(
        width: int | str = "default",
        scale: float = 1,
        subplots: tuple = (1, 1)
) -> tuple:
    """
    Get the figure dimensions to avoid scaling in LaTex.

    This function was taken from
    https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    :param width: Document width in points, or string of predefined
    document type (float or string)
    :param fraction: fraction of the width which you wish the figure to
    occupy (float)
    :param subplots: the number of rows and columns of subplots

    :return: dimensions of the figure in inches (tuple)
    """
    if width == 'default':
        width_pt = 420
    elif width == 'thesis':
        width_pt = 455.30101
    elif width == 'beamer':
        width_pt = 398.3386
    elif width == "nature":
        width_pt = 518.74
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * scale

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def add_colorbar(
    ax: plt.Axes,
    mappable: matplotlib.cm.ScalarMappable = None,
    loc: str = "right",
    size: str = "5%",
    pad: float = 0.05,
    label_size: int = 6,
    scientific_notation: bool = False,
    **kwargs,
) -> matplotlib.colorbar.Colorbar:
    """
    Add a colorbar to the given axes. Stolen from Edoardo Zatterin sxdm
    package (https://gitlab.esrf.fr/id01-science/id01-sxdm-utils/).

    Args:
        ax (plt.Axes): the axes to which the colorbar will
            be added.
        mappable (matplotlib.cm.ScalarMappable, optional): the mappable
            object that the colorbar will be based on. If None, will
            take ax.images[0]. Defaults to None.
        loc (str, optional): the location where the colorbar will be
            placed. Defaults to "right".
        size (str, optional): the size of the colorbar. Defaults to
            "5%".
        pad (float, optional): the padding between the colorbar and the
            axes. Defaults to 0.05.
        label_size (int, optional): the size of the colorbar labels.
            Defaults to 6.
        scientific_notation (bool, optional): whether to use scientific
            notation for colorbar labels. Defaults to False.

    Returns:
        matplotlib.colorbar.Colorbar: the colorbar object.
    """
    if mappable is None:
        if ax.images == []:
            raise ValueError(
                "mappable is None and ax.images is empty! "
                "Provide ax on which an image has beem drawn or provide a "
                "mappable."
            )
        mappable = ax.images[0]
    
    # check if vmin and vmax from the normalisation object are valid
    norm = mappable.norm
    vmin, vmax = norm.vmin, norm.vmax
     # Handle LogNorm-specific issues
    if isinstance(norm, matplotlib.colors.LogNorm):
        if vmin is None or vmax is None or vmin <= 0 or vmax <= 0:
            warnings.warn(
                "Invalid vmin or vmax detected for LogNorm. "
                "LogNorm requires vmin and vmax to be strictly positive. "
                "Skipping colorbar creation.",
                UserWarning
            )
            return None  # skip colorbar if LogNorm is invalid
        
    
    fig = ax.get_figure()
    cax = make_axes_locatable(ax).append_axes(loc, size=size, pad=pad)
    cax.tick_params(labelsize=label_size)
    cbar = fig.colorbar(mappable, cax=cax, **kwargs)
    if scientific_notation:
        cax.ticklabel_format(
            axis="y", style="scientific", scilimits=(0, 0), useMathText=True
        )

    return cbar


def two_spine_frameless_ax(
        ax: plt.Axes,
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
        ax: plt.Axes,
        grey_background_opacity=0
) -> plt.Axes:
    """Plot a grey background and a grid"""

    ax.grid(True, linestyle="--", linewidth=0.5, zorder=0)
    ax.patch.set_facecolor("lightgrey")
    ax.patch.set_alpha(grey_background_opacity)
    return ax


def white_interior_ticks_labels(
        ax: plt.Axes,
        xtick_pad: int = -15,
        ytick_pad: int = -25
) -> None:
    """Place the ticks and labels inside the provided axis."""
    ax.tick_params(axis="x", direction="in", pad=xtick_pad, colors="w")
    ax.tick_params(axis="y", direction="in", pad=ytick_pad, colors="w")
    ax.xaxis.set_ticks_position("bottom")

    xticks_loc, yticks_loc = ax.get_xticks(), ax.get_yticks()
    xticks_loc[1] = yticks_loc[1] = None

    xlabels, ylabels = ax.get_xticklabels(), ax.get_yticklabels()
    xlabels[1] = ylabels[1] = ""
    for t in ax.yaxis.get_majorticklabels():
        t.set_ha("left")
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