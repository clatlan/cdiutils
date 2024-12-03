import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import xrayutilities as xu
import warnings

from cdiutils.utils import (
    nan_to_zero,
    extract_reduced_shape,
    get_centred_slices
)
from cdiutils.plot.formatting import (
    get_figure_size,
    add_colorbar,
    get_x_y_limits_extents,
    set_x_y_limits_extents,
    XU_VIEW_PARAMETERS,
    CXI_VIEW_PARAMETERS,
    NATURAL_VIEW_PARAMETERS
)


def plot_volume_slices(
        data: np.ndarray,
        support: np.ndarray = None,
        voxel_size: tuple | list = None,
        data_centre: tuple | list = None,
        views: tuple[str] = None,
        convention: str = None,
        title: str = None,
        equal_limits: bool = True,
        slice_shift: tuple | list = None,
        integrate: bool = False,
        show: bool = True,
        **plot_params
) -> tuple[plt.Figure, plt.Axes]:
    """
    Generic function for plotting 2D slices (cross section or sum, with
    option 'integrate') of 3D volumes. The slices are plotted according
    to the specified views and conventions. If not specified, natural
    views are plotted in matrix convention (x-axis: 2nd dim, y-axis:
    1st dim), i.e:
    * first slice: taken at the centre of axis0
    * second slice: taken at the centre of axis1
    * third slice: taken at the centre of axis2

    Args:
        data (np.ndarray): the data to plot.
        support (np.ndarray, optional): a support for the data. Defaults
            to None.
        voxel_size (tuple | list, optional): the voxel size to modify
            the aspect ratio accordingly. Defaults to None.
        data_centre (tuple | list, optional): the centre to take the
            data at. Defaults to None.
        views (tuple[str], optional): the views for each plot according
            to the provided convention. If None default views of the
            specified convention are plotted. Defaults to None.
        convention (str, optional): the convention employed to plot the
            multiple slices, if views not specified, will set the
            default views for the specified convention, i.e.:
            ("x-", "y-", "z-") for XU convention and ("z+", "y-", "x+")
            for the CXI convention. If None, natural views are plotted.
            Defaults to None.
        title (str, optional): the title of the plot. Defaults to None.
        equal_limits (bool, optional): whether to have the same limit
            extend for all axes. Defaults to True.
        slice_shift (tuple | list, optional): the shift in the slice
            selection, by default will use the centre for each dim.
            Defaults to None.
        integrate (bool, optional): whether to sum the data instead of
            taking the slice. Defaults to False.
        show (bool, optional): whether to show the plot. Defaults to
            True. False might be useful if the function is only used for
            generating the axes that are then redrawn afterwards.
        **plot_params: additional plot params that will be parsed into
            the matplotlib imshow() function.

    Returns:
        tuple[plt.Figure, plt.Axes]: the generated figure and axes.
    """
    _plot_params = {"cmap": "turbo"}

    if plot_params:
        _plot_params.update(plot_params)

    view_params = CXI_VIEW_PARAMETERS.copy()
    if convention is None:
        if views is None:  # Simplest case, no swapping, no flipping etc.
            # For the default behaviour we use the 'natural views'
            view_params = NATURAL_VIEW_PARAMETERS.copy()
            views = ("dim0", "dim1", "dim2")
    elif convention.lower() in ("xu", "lab"):
        view_params = XU_VIEW_PARAMETERS.copy()  # overwrite the params
        if views is None:
            views = ("x-", "y-", "z-")
    elif convention.lower() == "cxi":
        if views is None:
            views = ("z+", "y-", "x+")

    slices = get_centred_slices(data.shape, shift=slice_shift)

    shape = data.shape
    if support is not None:
        shape = extract_reduced_shape(support)

    if voxel_size is not None:
        extents = get_x_y_limits_extents(data.shape, voxel_size, data_centre)
        limits = get_x_y_limits_extents(
            shape, voxel_size, data_centre, equal_limits=equal_limits
        )

    figure, axes = plt.subplots(1, 3, layout="tight", figsize=(6, 2))
    for i, v in enumerate(views):
        plane = view_params[v]["plane"]
        to_plot = data.sum(axis=i) if integrate else data[slices[i]]
        if plane[0] > plane[1]:
            to_plot = np.swapaxes(to_plot, 1, 0)

        if view_params[v]["xaxis_points_left"]:
            to_plot = to_plot[np.s_[:, ::-1]]

        axes[i].imshow(to_plot, **_plot_params)
        add_colorbar(axes[i], axes[i].images[0])
        if voxel_size is not None:
            set_x_y_limits_extents(
                axes[i], extents, limits,
                plane, view_params[v]["xaxis_points_left"]
            )

    figure.suptitle(title)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure, axes


def plot_slices(
        *data: list[np.ndarray],
        slice_labels: list = None,
        figsize: tuple[float] = None,
        data_stacking: str = "vertical",
        nan_supports: list = None,
        vmin: float = None,
        vmax: float = None,
        alphas: list = None,
        origin: str = "lower",
        cmap: str = "turbo",
        show_cbar: bool = True,
        cbar_title: str = None,
        cbar_location: str = "top",
        cbar_extend: str = "both",
        norm: matplotlib.colors.Normalize = None,
        cbar_ticks: list = None,
        slice_name: str = None,
        suptitle: str = None,
        show: bool = True,
) -> matplotlib.figure.Figure:
    """Plot 2D slices of the provided data."""

    if figsize is None:
        if data_stacking in ("vertical", "v"):
            figsize = (6, 4 * len(data))
        else:
            figsize = (6 * len(data), 4)
    if data_stacking in ("vertical", "v"):
        nrows_ncols = (len(data), 1)
    elif data_stacking in ("horizontal", "h"):
        nrows_ncols = (1, len(data))
    else:
        raise ValueError(
            "data_stacking should be 'vertical' or 'horizontal'.")
    if slice_labels is None:
        slice_labels = [None for i in range(len(data))]
    elif len(slice_labels) != len(data):
        print(
            "Number of slice_labels should be identical to number of *data.\n"
            "slice_labels won't be displayed."
        )
        slice_labels = ["" for i in range(len(data))]

    if figsize is None:
        figsize = get_figure_size()

    figure = plt.figure(figsize=figsize)
    grid = AxesGrid(
        figure,
        111,
        nrows_ncols=nrows_ncols,
        axes_pad=0.05,
        cbar_mode="single" if show_cbar else None,
        cbar_location=cbar_location,
        cbar_pad=0.25 if show_cbar else None,
        cbar_size=0.2 if show_cbar else None
    )

    for i, to_plot in enumerate(data):
        if nan_supports is not None:
            if isinstance(nan_supports, list):
                to_plot = to_plot * nan_supports[i]
            else:
                to_plot = to_plot * nan_supports
        # params = {
        #     "vmin": vmin, "vmax": vmax, "cmap": cmap, "origin": origin, "norm": 
        # }
        im = grid[i].matshow(
            to_plot,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            origin=origin,
            norm=norm,
            alpha=None if alphas is None else alphas[i]
        )

        if data_stacking in ("vertical", "v"):
            grid[i].annotate(
                slice_labels[i] if slice_labels is not None else "",
                xy=(0.2, 0.5),
                xytext=(-grid[i].yaxis.labelpad - 2, 0),
                xycoords=grid[i].yaxis.label,
                textcoords='offset points',
                ha='right',
                va='center'
            )
        else:
            grid[i].annotate(
                slice_labels[i] if slice_labels is not None else "",
                xy=(0.5, 0.9),
                xytext=(0, -grid[i].xaxis.labelpad - 2),
                xycoords=grid[i].xaxis.label,
                textcoords='offset points',
                ha='center',
                va='top'
            )

    if data_stacking in ("vertical", "v"):
        grid[len(data)-1].annotate(
            slice_name,
            xy=(0.5, 0.2),
            xytext=(0, -grid[len(data)-1].xaxis.labelpad - 2),
            xycoords=grid[len(data)-1].xaxis.label,
            textcoords='offset points',
            ha='center',
            va='top'
        )
    else:
        grid[0].annotate(
            slice_name,
            xy=(0.2, 0.5),
            xytext=(-grid[0].yaxis.labelpad - 2, 0),
            xycoords=grid[0].yaxis.label,
            textcoords='offset points',
            ha='right',
            va='center'
        )
    for i, ax in enumerate(grid):
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
    if show_cbar:
        ticklocation = (
            "bottom" if cbar_location in ("top", "bottom") else "auto"
        )
        cbar = grid.cbar_axes[0].colorbar(
            im,
            extend=cbar_extend,
            ticklocation=ticklocation
        )
        grid.cbar_axes[0].set_title(cbar_title)
        if cbar_ticks:
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_ticks)
    figure.suptitle(suptitle)
    figure.tight_layout()
    if show:
        plt.show()
    return figure


def plot_3d_volume_slices(
        *data: list[np.ndarray],
        slice_labels: list[str] = None,
        shapes: list[tuple] = None,
        nan_supports: list[np.ndarray] = None,
        figsize: tuple[float] = None,
        cmap: str | matplotlib.colors.Colormap = "turbo",
        vmin: float = None,
        vmax: float = None,
        alphas: list[np.ndarray] = None,
        log_scale: bool = False,
        do_sum: bool = False,
        suptitle: str = None,
        show: bool = True,
        return_fig: bool = False,
        show_cbar: bool = True,
        cbar_title: str = None,
        cbar_location: str = "top",
        cbar_extend: str = "both",
        cbar_ticks: list = None,
        aspect_ratios: dict = None,
        norm: matplotlib.colors.Normalize = None,
        data_stacking="vertical",
        slice_names=[
            r"(xy)$_{cxi}$ slice",
            r"(xz)$_{cxi}$ slice",
            r"(yz)$_{cxi}$ slice"
        ],
        **plot_params
):
    """
    Plot 2D slices of a 3D volume data in three directions.

    :param *data: the 3D data to plot (np.array). Several 3D matrices
    may be given. For each matrice, three slices are plotted.
    :param slice_labels: list of slice_labels corresponding to the given 3D
    matrices (list). Must be the same length as the number of provided
    *data. Otherwise, no slice_labels will be displayed. Default: None.
    :param figsize: figure size (tuple). Default: (6, 4).
    :param cmap: the matplotlib colormap (str) used for the colorbar
    (default: "turbo").
    :param vmin: the minimum value (float) for the color scale
    (default: None).
    :param vmax: the maximum value (float) for the color scale
    (default: None).
    :param log_scale: whether or not the scale is logaritmhic (bool).
    Default: False.
    :param suptitle: global title of the figure (str). Default: None.
    :param show: whether or not to show the figure (bool). If False, the
    figure is not displayed but returned.
    :param data_stacking: stacking direction for the slice plot (str).
    Can only be "vertical" or "horizontal", default: "vertical".
    :param slice_names: the name of the slices (list of str). For each
    *data, three slices are plotted, this str are the name of each
    slice.
    :return: figure if show is false.
    """

    if figsize is None:
        if data_stacking in ("vertical", "v"):
            figsize = (18, 4 * len(data))
        else:
            figsize = (6 * len(data), 12)
    fig = plt.figure(figsize=figsize)

    if log_scale:
        if norm is not None:
            print("norm provided, will not use log_scale.")
            log_scale = False            
    if norm is None:
        if vmin is None:
            vmin = None if do_sum or len(data) > 1 else np.nanmin(data)
        if vmax is None:
            vmax = None if do_sum or len(data) > 1 else np.nanmax(data)
    else:
        vmin = vmax = None

    if data_stacking in ("vertical", "v"):
        nrows_ncols = (len(data), 3)
    elif data_stacking in ("horizontal", "h"):
        nrows_ncols = (3, len(data))
    else:
        raise ValueError(
            "data_stacking should be 'vertical', 'v', 'horizontal' or 'h'.")
    if slice_labels is None:
        slice_labels = ["" for i in range(len(data))]
    elif len(slice_labels) != len(data):
        print(
            "Number of slice_labels should be identical to number of *data.\n"
            "slice_labels won't be displayed.")
        slice_labels = ["" for i in range(len(data))]
    
    grid = AxesGrid(
        fig,
        111,
        nrows_ncols=nrows_ncols,
        axes_pad=0.05,
        cbar_mode="single" if show_cbar else None,
        cbar_location=cbar_location,
        cbar_pad=0.25 if show_cbar else None,
        cbar_size=0.2 if show_cbar else None
    )

    for i, plot in enumerate(data):
        if log_scale:
            norm = matplotlib.colors.LogNorm(plot.min(), plot.max())
        if nan_supports is not None:
            if isinstance(nan_supports, list):
                plot = plot * nan_supports[i]
            else:
                plot = plot * nan_supports
        if not shapes:
            shape = plot.shape
        else:
            shape = shapes[i]

        if data_stacking in ("vertical", "v"):
            ind1 = 3 * i
            ind2 = 3 * i + 1
            ind3 = 3 * i + 2
        else:
            ind1 = i
            ind2 = i + len(data)
            ind3 = i + 2 * len(data)
        im = grid[ind1].matshow(
            np.sum(plot, axis=0) if do_sum else plot[shape[0]//2,],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect=aspect_ratios["yz"] if aspect_ratios else "auto",
            norm=norm,
            alpha=None if alphas is None else alphas[i][shape[0]//2,],
            **plot_params
        )
        grid[ind2].matshow(
            np.sum(plot, axis=1) if do_sum else plot[:, shape[1]//2, :],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect=aspect_ratios["xz"] if aspect_ratios else "auto",
            norm=norm,
            alpha=None if alphas is None else alphas[i][:, shape[1]//2, :],
            **plot_params
        )
        grid[ind3].matshow(
            np.sum(plot, axis=2) if do_sum else plot[:, :, shape[2]//2],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect=aspect_ratios["xy"] if aspect_ratios else "auto",
            norm=norm,
            alpha=None if alphas is None else alphas[i][:, :, shape[2]//2],
            **plot_params
        )

        if data_stacking in ("vertical", "v"):
            grid[ind1].annotate(
                slice_labels[i] if slice_labels is not None else "",
                xy=(0.2, 0.5),
                xytext=(-grid[ind1].yaxis.labelpad - 2, 0),
                xycoords=grid[ind1].yaxis.label,
                textcoords='offset points',
                ha='right',
                va='center'
            )
        else:
            grid[ind3].annotate(
                slice_labels[i] if slice_labels is not None else "",
                xy=(0.5, 0.9),
                xytext=(0, -grid[ind3].xaxis.labelpad - 2),
                xycoords=grid[ind3].xaxis.label,
                textcoords='offset points',
                ha='center',
                va='top'
            )

    for i, slice_name in enumerate(slice_names):
        if data_stacking == "vertical":
            ind = 3*(len(data)-1) + i
            grid[ind].annotate(
                slice_name,
                xy=(0.5, 0.2),
                xytext=(0, -grid[ind].xaxis.labelpad - 2),
                xycoords=grid[ind].xaxis.label,
                textcoords='offset points',
                ha='center',
                va='top'
            )

        else:
            ind = i * len(data)
            grid[ind].annotate(
                slice_name,
                xy=(0.2, 0.5),
                xytext=(-grid[ind].yaxis.labelpad - 2, 0),
                xycoords=grid[ind].yaxis.label,
                textcoords='offset points',
                ha='right',
                va='center'
            )

    for i, ax in enumerate(grid):
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
    if show_cbar:
        ticklocation = (
            "bottom" if cbar_location in ("top", "bottom") else "auto"
        )
        cbar = grid.cbar_axes[0].colorbar(
            im,
            extend=cbar_extend,
            ticklocation=ticklocation
        )
        grid.cbar_axes[0].set_title(cbar_title)
        if cbar_ticks:
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_ticks)
    fig.suptitle(suptitle)
    if show:
        plt.show()
    return fig if return_fig else None


def plot_support_contour(
        amplitudes,
        supports,
        isosurfaces,
        conditions,
        scan_ref,
        threshold,
        contour_linewidths=2.5,
        contour_colors=("azure", "deepskyblue"),
        **kwargs
    ):
    scan_digits = list(amplitudes.keys())
    
    filtered_amplitudes = {
        scan: np.where(amplitudes[scan] < isosurfaces[scan], np.nan, amplitudes[scan])
    for scan in scan_digits
    }

    filtered_amp_fig = plot_3D_volume_slices(
        *filtered_amplitudes.values(),
        titles=list(conditions.values()),
        vmin=threshold,
        vmax=1,
        **kwargs
    )

    support_ref = supports[scan_ref]
    shape = support_ref.shape

    for i, ax in enumerate(filtered_amp_fig.axes):
        if i < len(scan_digits):
            X, Y = np.meshgrid(
                np.arange(0, shape[2]), (np.arange(0, shape[1])))
            ax.contour(
                X,
                Y,
                support_ref[shape[0] // 2],
                levels=[0, 1],
                linewidths=contour_linewidths,
                colors=contour_colors[0] ,

            )
            if i % len(scan_digits) != 0:
                ax.contour(
                    X,
                    Y,
                    supports[scan_digits[i]][shape[0] // 2],
                    levels=[0, 1],
                    linewidths=contour_linewidths,
                    colors=contour_colors[1],

                )
        elif i < 2*len(scan_digits):
            X, Y = np.meshgrid(np.arange(0, shape[2]), (np.arange(0, shape[0])))
            ax.contour(
                X,
                Y,
                support_ref[:, shape[1] // 2, :],
                levels=[0, 1],
                linewidths=contour_linewidths,
                colors=contour_colors[0],

            )
            if i % len(scan_digits) != 0:
                ax.contour(
                    X,
                    Y,
                    supports[scan_digits[i%len(scan_digits)]][:, shape[1] // 2, :],
                    levels=[0, 1],
                    linewidths=contour_linewidths,
                    colors=contour_colors[1],
                )
        elif i < 3*len(scan_digits):
            X, Y = np.meshgrid(np.arange(0, shape[1]), (np.arange(0, shape[0])))
            ax.contour(
                X,
                Y,
                support_ref[..., shape[2] // 2],
                levels=[0, 0.1],
                linewidths=contour_linewidths,
                colors=contour_colors[0],

            )
            if i % len(scan_digits) != 0:
                ax.contour(
                    X,
                    Y,
                    supports[scan_digits[i%len(scan_digits)]][..., shape[2] // 2],
                    levels=[0, 1],
                    linewidths=contour_linewidths,
                    colors=contour_colors[1],
                )
    return filtered_amp_fig



def plot_diffraction_patterns(
        intensities,
        gridders,
        titles=None,
        data_stacking="vertical",
        figsize=(8, 8),
        aspect_ratio="equal",
        maplog_min=3,
        levels=100,
        xlim=None,
        ylim=None,
        zlim=None,
        show=True,
        cmap="turbo",
        angstrom_symbol=r"\si{\angstrom}"
):
    if len(intensities) != len(gridders):
        print("lists intensities and gridders must have the same length")
        return None

    no_title = True
    if titles is not None and len(titles) != len(intensities):
        print("lists intensities and titles must have the same length")
    elif titles is not None:
        no_title = False

    if data_stacking not in ["vertical", "horizontal"]:
        print("data_stacking should be 'vertical' or 'horizontal'")
        return None

    fig, axes = plt.subplots(
        len(intensities) if data_stacking == "vertical" else 3,
        3 if data_stacking == "vertical" else len(intensities),
        figsize=figsize,
        squeeze=False
    )

    for i, (intensity, (qx, qy, qz)) in enumerate(zip(intensities, gridders)):
        log_intensity = xu.maplog(intensity, maplog_min, 0)

        if data_stacking == "vertical":
            ax_coord = (i, 0)
            increment = (0, 1)
        else:
            ax_coord = (0, i)
            increment = (1, 0)

        summed_intensity = log_intensity.sum(axis=2).T
        normalized_intensity = (
            (summed_intensity - np.min(summed_intensity))
            / np.ptp(summed_intensity)
        )
        cnt = axes[ax_coord].contourf(
            qx, qy, summed_intensity, levels=levels, cmap=cmap
        )
        try:
            axes[ax_coord].set_xlabel(r"$Q_X (" + angstrom_symbol + r"^{-1})$")
        except ValueError:
            angstrom_symbol = r"\AA"
            axes[ax_coord].set_xlabel(r"$Q_X (" + angstrom_symbol + r"^{-1})$")
        axes[ax_coord].set_ylabel(r"$Q_Y (" + angstrom_symbol + r"^{-1})$")
        for c in cnt.collections:
            c.set_edgecolor("face")
        if xlim is not None:
            axes[ax_coord].set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            axes[ax_coord].set_ylim(ylim[0], ylim[1])
        if not no_title and data_stacking == "horizontal":
            axes[ax_coord].set_title(titles[i])

        ax_coord = tuple([sum(t) for t in zip(ax_coord, increment)])
        summed_intensity = log_intensity.sum(axis=1).T
        normalized_intensity = (
            (summed_intensity - np.min(summed_intensity))
            / np.ptp(summed_intensity)
        )
        cnt = axes[ax_coord].contourf(
            qx, qz, summed_intensity, levels=levels, cmap=cmap
        )
        axes[ax_coord].set_xlabel(r"$Q_X (" + angstrom_symbol + r"^{-1})$")
        axes[ax_coord].set_ylabel(r"$Q_Z (" + angstrom_symbol + r"^{-1})$")
        for c in cnt.collections:
            c.set_edgecolor("face")

        if xlim is not None:
            axes[ax_coord].set_xlim(xlim[0], xlim[1])
        if zlim is not None:
            axes[ax_coord].set_ylim(zlim[0], zlim[1])
        if not no_title and data_stacking == "vertical":
            axes[ax_coord].set_title(titles[i])

        ax_coord = tuple([sum(t) for t in zip(ax_coord, increment)])
        summed_intensity = log_intensity.sum(axis=0).T
        normalized_intensity = (
            (summed_intensity - np.min(summed_intensity))
            / np.ptp(summed_intensity)
        )
        cnt = axes[ax_coord].contourf(
            qy, qz, normalized_intensity, levels=levels, cmap=cmap
        )
        axes[ax_coord].set_xlabel(r"$Q_Y (" + angstrom_symbol + r"^{-1})$")
        axes[ax_coord].set_ylabel(r"$Q_Z (" + angstrom_symbol + r"^{-1})$")
        for c in cnt.collections:
            c.set_edgecolor("face")
        if ylim is not None:
            axes[ax_coord].set_xlim(ylim[0], ylim[1])
        if zlim is not None:
            axes[ax_coord].set_ylim(zlim[0], zlim[1])

        if aspect_ratio:
            for ax in axes.ravel():
                ax.set_aspect(aspect_ratio)

    fig.tight_layout()
    if show:
        plt.show()
        return fig, cnt
    else:
        return fig, cnt


def plot_contour(
        ax,
        support_2d,
        linewidth=1,
        color="k",
        pixel_size=None,
        data_centre=None
):
    shape = support_2d.shape
    x_range = np.arange(0, shape[1])
    y_range = np.arange(0, shape[0])
    if pixel_size is not None:
        x_range = x_range * pixel_size[1]
        y_range = y_range * pixel_size[0]
    if data_centre is not None:
        x_range = x_range - x_range.mean() + data_centre[1]
        y_range = y_range - y_range.mean() + data_centre[0]

    X, Y = np.meshgrid(x_range, y_range)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        ax.contour(
            X,
            Y,
            nan_to_zero(support_2d),
            levels=[0, 1],
            linewidths=linewidth,
            colors=color,
        )