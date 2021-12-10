import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import xrayutilities as xu

# mpl.rcParams["mpl_toolkits.legacy_colorbar"] = False


def plot_slices(
        *data,
        titles=None,
        figsize=(6, 4),
        cmap="viridis",
        vmin=None,
        vmax=None,
        log_scale=False,
        suptitle=None,
        show=True,
        data_stacking="vertical",
        slice_names=["YZ slice", "XZ slice", "XY slice"]):
    """
    Plot 2D slices of a 3D volume data in threde directions.

    :param *data: the 3D data to plot (np.array). Several 3D matrices
    may be given. For each matrice, three slices are plotted.
    :param titles: list of titles corresponding to the given 3D
    matrices (list). Must be the same length as the number of provided
    *data. Otherwise, no titles will be displayed. Default: None.
    :param figsize: figure size (tuple). Default: (6, 4).
    :param cmap: the matplotlib colormap (str) used for the colorbar
    (default: "viridis").
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
    :return: figure is show is false.
    """

    fig = plt.figure(figsize=figsize)

    if log_scale:
        data = np.log(data)
    vmin = np.min(data) if vmin is None else vmin
    vmax = np.max(data) if vmax is None else vmax

    if data_stacking == "vertical":
        nrows_ncols = (len(data), 3)
    elif data_stacking == "horizontal":
        nrows_ncols = (3, len(data))
    else:
        print("data_stacking should be 'vertical' or 'horizontal'.")
        return None
    if titles is None:
        print("No titles given.")
        titles = ["" for i in range(len(data))]
    elif len(titles) != len(data):
        print(
            "Number of titles should be identical to number of *data.\n"
            "Titles won't be displayed.")
        titles = ["" for i in range(len(data))]

    grid = AxesGrid(fig, 111,
                    nrows_ncols=nrows_ncols,
                    axes_pad=0.05,
                    cbar_mode='single',
                    cbar_location='top',
                    cbar_pad=0.2)

    for i, plot in enumerate(data):
        shape = plot.shape
        if data_stacking == "vertical":
            ind1 = 3 * i
            ind2 = 3 * i + 1
            ind3 = 3 * i + 2
        else:
            ind1 = i
            ind2 = i + len(data)
            ind3 = i + 2 * len(data)
        im = grid[ind1].matshow(
            plot[shape[0]//2, ...],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
            )
        grid[ind2].matshow(
            plot[:, shape[1]//2, :],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
            )
        grid[ind3].matshow(
            plot[..., shape[2]//2],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
            )

        if data_stacking == "vertical":
            grid[ind1].annotate(
                titles[i] if titles is not None else "",
                xy=(0.2, 0.5),
                xytext=(-grid[ind1].yaxis.labelpad - 2, 0),
                xycoords=grid[ind1].yaxis.label,
                textcoords='offset points',
                size="medium",
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
                size="medium",
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
                size="medium",
                ha='right',
                va='center'
                )

        else:
            ind = i * len(data)
            grid[ind].annotate(
                slice_name,
                xy=(0.2, 0.5),
                xytext=(-grid[ind].yaxis.labelpad - 2, 0),
                xycoords=grid[ind].yaxis.label,
                textcoords='offset points',
                size="medium",
                ha='right',
                va='center'
                )

    for i, ax in enumerate(grid):
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    # cbar = grid.cbar_axes[0].colorbar(im)
    fig.suptitle(suptitle)
    # fig.tight_layout()
    if show:
        plt.show()
        return None
    else:
        return fig


def fancy_plot(
        data,
        title=None,
        log_scale=False,
        figsize=None,
        cmap='jet'):

    fig = plt.figure(figsize=figsize)

    if log_scale:
        data = np.where(data <= 1, 1, data)
        data = np.log10(data)
    vmin = np.min(data)
    vmax = np.max(data)

    shape = data.shape
    plt.imshow(data[:, shape[1] // 2, :], vmin=vmin, vmax=vmax)
    fig.suptitle(title)
    plt.show()


def plot_diffraction_patterns(
        intensities,
        gridders,
        titles=None,
        maplog_min=3,
        levels=100,
        xlim=None,
        ylim=None,
        zlim=None,
        show=True
):
    if len(intensities) != len(gridders):
        print("lists intensities and gridders must have the same length")
        return
    no_title = True
    if titles is not None and len(titles) != len(intensities):
        print("lists intensities and titles must have the same length")
    elif titles is not None:
        no_title = False
    fig, axes = plt.subplots(
        len(intensities),
        3,
        figsize=(12, 3 * len(intensities))
    )
    for i, (intensity, (qx, qy, qz)) in enumerate(zip(intensities, gridders)):
        log_intensity = xu.maplog(intensity, maplog_min, 0)

        axes[i, 0].contourf(
            qx, qy, log_intensity.sum(axis=2).T, levels=levels, cmap="turbo"
        )
        axes[i, 0].set_xlabel(r"$Q_X (\AA^{-1})$")
        axes[i, 0].set_ylabel(r"$Q_Y (\AA^{-1})$")   
    
        axes[i, 1].contourf(
            qx, qz, log_intensity.sum(axis=1).T, levels=levels, cmap="turbo"
        )
        axes[i, 1].set_xlabel(r"$Q_X (\AA^{-1})$")
        axes[i, 1].set_ylabel(r"$Q_Z (\AA^{-1})$")
        if not no_title:
              axes[i, 1].set_title(titles[i])

        axes[i, 2].contourf(
            qy, qz, log_intensity.sum(axis=0).T, levels=levels, cmap="turbo"
        )
        axes[i, 2].set_xlabel(r"$Q_Y (\AA^{-1})$")
        axes[i, 2].set_ylabel(r"$Q_Z (\AA^{-1})$")
        
        if xlim is not None:
            axes[i, 0].set_xlim(xlim[0], xlim[1])
            axes[i, 1].set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            axes[i, 0].set_ylim(ylim[0], ylim[1])
            axes[i, 2].set_xlim(ylim[0], ylim[1])
        if zlim is not None:
            axes[i, 1].set_ylim(zlim[0], zlim[1])
            axes[i, 2].set_ylim(zlim[0], zlim[1])
    
    fig.tight_layout()
    if show:
        plt.show()
        return None
    else:
        return fig
