import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import xrayutilities as xu
import warnings

from cdiutils.utils import nan_to_zero


def plot_slices(
            *data: list[np.ndarray],
            slice_labels: list=None,
            figsize: tuple[float]=None,
            data_stacking="vertical",
            nan_supports: list=None,
            vmin: float=None,
            vmax: float=None,
            origin: str="lower",
            cmap: str="turbo",
            show_cbar: bool=True,
            slice_name: str=None,
            cbar_title: str="",
            suptitle: str="",
            show: bool=True,
) -> matplotlib.figure.Figure:
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
            "Number of titles should be identical to number of *data.\n"
            "Titles won't be displayed.")
        slice_labels = ["" for i in range(len(data))]

    figure = plt.figure(figsize=figsize)
    grid = AxesGrid(
        figure,
        111,
        nrows_ncols=nrows_ncols,
        axes_pad=0.05,
        cbar_mode="single" if show_cbar else None,
        cbar_location="top" if show_cbar else None,
        cbar_pad=0.25 if show_cbar else None,
        cbar_size=0.2 if show_cbar else None
    )

    for i, to_plot in enumerate(data):
        if nan_supports is not None:
            if type(nan_supports) is list:
                to_plot = to_plot * nan_supports[i]
            else:
                to_plot = to_plot * nan_supports

        im = grid[i].matshow(
            to_plot,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            origin=origin,
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
        grid.cbar_axes[0].colorbar(im)
        grid.cbar_axes[0].set_title(cbar_title)
    figure.suptitle(suptitle)
    figure.tight_layout()
    if show:
        plt.show()
    return figure


def plot_3D_volume_slices(
            *data,
            titles=None,
            shapes=None,
            nan_support=None,
            figsize=(6, 4),
            cmap="turbo",
            vmin=None,
            vmax=None,
            log_scale=False,
            do_sum=False,
            suptitle=None,
            show=True,
            return_fig=False,
            cbar_title=None,
            show_cbar=False,
            cbar_location="top",
            aspect_ratios=None,
            data_stacking="vertical",
            slice_names=[
                r"(xy)$_{cxi}$ slice",
                r"(xz)$_{cxi}$ slice",
                r"(yz)$_{cxi}$ slice"
            ]
):
    """
    Plot 2D slices of a 3D volume data in three directions.

    :param *data: the 3D data to plot (np.array). Several 3D matrices
    may be given. For each matrice, three slices are plotted.
    :param titles: list of titles corresponding to the given 3D
    matrices (list). Must be the same length as the number of provided
    *data. Otherwise, no titles will be displayed. Default: None.
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

    fig = plt.figure(figsize=figsize)

    if log_scale:
        data = np.log(data)
    if vmin is None:
        vmin = None if do_sum or len(data) > 1 else np.nanmin(data)
    if vmax is None:
        vmax = None if do_sum or len(data) > 1 else np.nanmax(data)

    if data_stacking == "vertical":
        nrows_ncols = (len(data), 3)
    elif data_stacking == "horizontal":
        nrows_ncols = (3, len(data))
    else:
        raise ValueError(
            "data_stacking should be 'vertical' or 'horizontal'.")
    if titles is None:
        titles = ["" for i in range(len(data))]
    elif len(titles) != len(data):
        print(
            "Number of titles should be identical to number of *data.\n"
            "Titles won't be displayed.")
        titles = ["" for i in range(len(data))]
    
    grid = AxesGrid(fig, 111,
                    nrows_ncols=nrows_ncols,
                    axes_pad=0.05,
                    cbar_mode='single' if show_cbar else None,
                    cbar_location=cbar_location if show_cbar else None,
                    cbar_pad=0.25 if show_cbar else None)

    for i, plot in enumerate(data):
        if nan_support is not None:
            if type(nan_support) is list:
                plot = plot * nan_support[i]
            else:
                plot = plot * nan_support
        if not shapes:
            shape = plot.shape
        else:
            shape = shapes[i]
        if data_stacking == "vertical":
            ind1 = 3 * i
            ind2 = 3 * i + 1
            ind3 = 3 * i + 2
        else:
            ind1 = i
            ind2 = i + len(data)
            ind3 = i + 2 * len(data)
        im = grid[ind1].matshow(
            np.sum(plot, axis=0) if do_sum else plot[shape[0]//2, ...],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect=aspect_ratios["yz"] if aspect_ratios else "auto"
        )
        grid[ind2].matshow(
            np.sum(plot, axis=1) if do_sum else plot[:, shape[1]//2, :],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect=aspect_ratios["xz"] if aspect_ratios else "auto"
        )
        grid[ind3].matshow(
            np.sum(plot, axis=2) if do_sum else plot[..., shape[2]//2],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect=aspect_ratios["xy"] if aspect_ratios else "auto"
        )

        if data_stacking == "vertical":
            grid[ind1].annotate(
                titles[i] if titles is not None else "",
                xy=(0.2, 0.5),
                xytext=(-grid[ind1].yaxis.labelpad - 2, 0),
                xycoords=grid[ind1].yaxis.label,
                textcoords='offset points',
                ha='right',
                va='center'
            )
        else:
            grid[ind3].annotate(
                titles[i] if titles is not None else "",
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
        grid.cbar_axes[0].colorbar(im)
        grid.cbar_axes[0].set_title(cbar_title)
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


def plot_contour(ax, support_2D, linewidth=2, color="k"):
    shape = support_2D.shape
    X, Y = np.meshgrid(np.arange(0, shape[0]), np.arange(0, shape[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        ax.contour(
            X,
            Y,
            nan_to_zero(support_2D),
            levels=[0, 1],
            linewidths=linewidth,
            colors=color
        )