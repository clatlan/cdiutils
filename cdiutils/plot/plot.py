import colorcet
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import xrayutilities as xu

# mpl.rcParams["mpl_toolkits.legacy_colorbar"] = False

def update_plot_params(
        max_open_warning=100,
        dpi=200,
        lines_marker="",
        lines_linewidth=2.5,
        lines_linestyle="-",
        lines_markersize=7,
        figure_titlesize=22,
        axes_titesize=16,
        axes_labelsize=16,
        xtick_labelsize=12,
        ytick_labelsize=12,
        legend_fontsize=10

):
    plt.rcParams.update(
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
            "axes.titlesize": axes_titesize,
            "axes.labelsize": axes_labelsize,
            "xtick.labelsize": xtick_labelsize,
            "ytick.labelsize": ytick_labelsize,
            "legend.fontsize": legend_fontsize
        }
    )

def plot_slices(
            *data,
            titles=None,
            figsize=(6, 4),
            cmap="jet",
            vmin=None,
            vmax=None,
            log_scale=False,
            suptitle=None,
            show=True,
            cbar_title=None,
            show_cbar=False,
            data_stacking="vertical",
            slice_names=["YZ slice", "XZ slice", "XY slice"]
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
    vmin = np.nanmin(data) if vmin is None else vmin
    vmax = np.nanmax(data) if vmax is None else vmax

    if data_stacking == "vertical":
        nrows_ncols = (len(data), 3)
    elif data_stacking == "horizontal":
        nrows_ncols = (3, len(data))
    else:
        print("data_stacking should be 'vertical' or 'horizontal'.")
        return
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
                    cbar_mode='single' if show_cbar else None,
                    cbar_location='top'if show_cbar else None,
                    cbar_pad=0.25 if show_cbar else None)

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
            plot[shape[0]//2],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )
        grid[ind2].matshow(
            plot[:, shape[1]//2, :],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower"
        )
        grid[ind3].matshow(
            plot[..., shape[2]//2],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower"
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
    if show_cbar:
        grid.cbar_axes[0].colorbar(im)
        grid.cbar_axes[0].set_title(cbar_title)
    fig.suptitle(suptitle)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_support_contour(
        amplitudes,
        supports,
        isosurfaces,
        potentials,
        scan_ref,
        threshold
    ):
    scan_digits = list(amplitudes.keys())
    
    filtered_amplitudes = {
        scan: np.where(amplitudes[scan] < isosurfaces[scan], np.nan, amplitudes[scan])
    for scan in scan_digits
    }

    filtered_amp_fig = plot_slices(
        *filtered_amplitudes.values(),
        titles=list(potentials.values()),
        figsize=(12, 8),
        cmap="turbo",
        data_stacking="horizontal",
        show_cbar=True,
        cbar_title="Filtered amplitude (a. u.)",
        vmin=threshold,
        vmax=1
    )

    support_ref = supports[scan_ref]
    shape = support_ref.shape

    for i, ax in enumerate(filtered_amp_fig.axes):
        if i < len(scan_digits):
            X, Y = np.meshgrid(np.arange(0, shape[2]), (np.arange(0, shape[1])))
            ax.contour(
                X,
                Y,
                support_ref[shape[0] // 2],
                levels=[0, 1],
                linewidths=1.5,
                colors="blue",

            )
            ax.contour(
                X,
                Y,
                supports[scan_digits[i]][shape[0] // 2],
                levels=[0, 1],
                linewidths=1.5,
                colors="green",

            )
        elif i < 2*len(scan_digits):
            X, Y = np.meshgrid(np.arange(0, shape[2]), (np.arange(0, shape[0])))
            ax.contour(
                X,
                Y,
                support_ref[:, shape[1] // 2, :],
                levels=[0, 1],
                linewidths=1.5,
                colors="blue",

            )
            ax.contour(
                X,
                Y,
                supports[scan_digits[i%len(scan_digits)]][:, shape[1] // 2, :],
                levels=[0, 1],
                linewidths=1.5,
                colors="green",

            )
        elif i < 3*len(scan_digits):
            X, Y = np.meshgrid(np.arange(0, shape[1]), (np.arange(0, shape[0])))
            ax.contour(
                X,
                Y,
                support_ref[..., shape[2] // 2],
                levels=[0, 0.1],
                linewidths=1.5,
                colors="blue",

            )
            ax.contour(
                X,
                Y,
                supports[scan_digits[i%len(scan_digits)]][..., shape[2] // 2],
                levels=[0, 1],
                linewidths=1.5,
                colors="green",
            )
        
    return filtered_amp_fig


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
        show=True,
        cmap="turbo"
):
    if len(intensities) != len(gridders):
        print("lists intensities and gridders must have the same length")
        return None
    no_title = True
    if titles is not None and len(titles) != len(intensities):
        print("lists intensities and titles must have the same length")
    elif titles is not None:
        no_title = False
    fig, axes = plt.subplots(
        len(intensities),
        3,
        figsize=(12, 3 * len(intensities)),
        squeeze=False
    )
    for i, (intensity, (qx, qy, qz)) in enumerate(zip(intensities, gridders)):
        log_intensity = xu.maplog(intensity, maplog_min, 0)

        cnt = axes[i, 0].contourf(
            qx, qy, log_intensity.sum(axis=2).T, levels=levels, cmap=cmap
        )
        axes[i, 0].set_xlabel(r"$Q_X (\si{\angstrom}^{-1})$")
        axes[i, 0].set_ylabel(r"$Q_Y (\si{\angstrom}^{-1})$")
        for c in cnt.collections:
            c.set_edgecolor("face")   
    
        cnt = axes[i, 1].contourf(
            qx, qz, log_intensity.sum(axis=1).T, levels=levels, cmap=cmap
        )
        axes[i, 1].set_xlabel(r"$Q_X (\si{\angstrom}^{-1})$")
        axes[i, 1].set_ylabel(r"$Q_Z (\si{\angstrom}^{-1})$")
        for c in cnt.collections:
            c.set_edgecolor("face")
        if not no_title:
              axes[i, 1].set_title(titles[i])

        cnt = axes[i, 2].contourf(
            qy, qz, log_intensity.sum(axis=0).T, levels=levels, cmap=cmap
        )
        axes[i, 2].set_xlabel(r"$Q_Y (\si{\angstrom}^{-1})$")
        axes[i, 2].set_ylabel(r"$Q_Z (\si{\angstrom}^{-1})$")
        for c in cnt.collections:
            c.set_edgecolor("face")
        
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
        return fig
    else:
        return fig
