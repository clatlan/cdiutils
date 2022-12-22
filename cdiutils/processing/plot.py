from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from cdiutils.utils import zero_to_nan
from cdiutils.plot.formatting import PLOT_CONFIGS, ANGSTROM_SYMBOL, PERCENT_SYMBOL
from cdiutils.plot.slice import plot_contour


def preprocessing_detector_data_plot(
        detector_data: np.array,
        cropped_data: np.array,
        det_pixel_reference: Union[np.array, list, tuple],
        det_max_pixel: Union[np.array, list, tuple],
        det_com_pixel: Union[np.array, list, tuple],
        cropped_max_pixel: Union[np.array, list, tuple],
        cropped_com_pixel: Union[np.array, list, tuple],
        title: str=""

) -> matplotlib.figure.Figure:
    """
    Plot the detector data in the full detector data frame and in the 
    cropped/centered frame.fits

    :param detector_data: the raw detector data (np.array)
    :param cropped_data: the cropped/centered data (np.array)
    :det_pixel_reference: the pixel reference in the full detector frame
    (np.array, list or tuple)
    :det_max_pixel: the max pixel in the full detector frame
    (np.array, list or tuple)
    :det_com_pixel: the com pixel in the full detector fame (np.array,
    list, tuple)
    :cropped_max_pixel: the max pixel in the centered/cropped detector
    frame (np.array, list or tuple)
    :cropped_com_pixel: the com pixel in the centered/cropped detector
    frame (np.array, list or tuple)
    :title: the tile of the figure (string)

    :return: the matplotlib figure object
    """

    figure, axes = plt.subplots(2, 3, figsize=(14, 10))

    log_data = np.log10(detector_data +1)
    log_cropped_data = np.log10(cropped_data +1)
    vmin = 0
    vmax = np.max(log_data)

    initial_shape = detector_data.shape
    final_shape = cropped_data.shape


    axes[0, 0].matshow(
        log_data[det_max_pixel[0]],
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",
        origin="upper"
    )
    axes[0, 0].plot(
        np.repeat(det_pixel_reference[2], 2),
        det_pixel_reference[1] + np.array(
            [-0.1*initial_shape[1], 0.1*initial_shape[1]]),
        color="w", 
        lw=0.5
    )
    axes[0, 0].plot(
        det_pixel_reference[2] + np.array(
            [-0.1*initial_shape[2], 0.1*initial_shape[2]]),
        np.repeat(det_pixel_reference[1], 2),
        color="w", 
        lw=0.5
    )
    axes[0, 0].plot(
        det_com_pixel[2], 
        det_com_pixel[1],
        marker="x",
        markersize=10,
        linestyle="None",
        color="green",
        label="com",
    )
    axes[0, 0].plot(
        det_max_pixel[2], 
        det_max_pixel[1],
        marker="x",
        markersize=10,
        linestyle="None",
        color="red",
        label="max"
    )

    axes[0, 1].matshow(
        log_data[:, det_max_pixel[1], :],
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",
        origin="lower"
    )
    axes[0, 1].plot(
        np.repeat(det_pixel_reference[2], 2),
        det_pixel_reference[0] + np.array(
            [-0.1*initial_shape[0], 0.1*initial_shape[0]]),
        color="w", 
        lw=0.5
    )
    axes[0, 1].plot(
        det_pixel_reference[2] + np.array(
            [-0.1*initial_shape[2], 0.1*initial_shape[2]]),
        np.repeat(det_pixel_reference[0], 2),
        color="w", 
        lw=0.5
    )
    axes[0, 1].plot(
        det_com_pixel[2], 
        det_com_pixel[0],
        marker="x",
        markersize=10,
        linestyle="None",
        color="green",
        label="com",
    )
    axes[0, 1].plot(
        det_max_pixel[2], 
        det_max_pixel[0],
        marker="x",
        markersize=10,
        linestyle="None",
        color="red",
        label="max"
    )

    mappable = axes[0, 2].matshow(
        np.swapaxes(log_data[..., det_max_pixel[2]], axis1=0, axis2=1),
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",
        origin="upper"
    )
    axes[0, 2].plot(
        np.repeat(det_pixel_reference[0], 2),
        det_pixel_reference[1] + np.array(
            [- 0.1 * initial_shape[1],  + 0.1 * initial_shape[1]]),
        color="w", 
        lw=0.5
    )
    axes[0, 2].plot(
        det_pixel_reference[0] + np.array(
            [- 0.1 * initial_shape[0],  + 0.1 * initial_shape[0]]),
        np.repeat(det_pixel_reference[1], 2),
        color="w", 
        lw=0.5
    )
    axes[0, 2].plot(
        det_com_pixel[0], 
        det_com_pixel[1],
        marker="x",
        markersize=10,
        color="green",
        label="com"
    )
    axes[0, 2].plot(
        det_max_pixel[0], 
        det_max_pixel[1],
        marker="x",
        markersize=10,
        color="red",
        label="max",
    )

    axes[1, 0].matshow(
        log_cropped_data[cropped_max_pixel[0]],
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",
        origin="upper"
    )

    axes[1, 0].plot(
        np.repeat(final_shape[2]//2, 2),
        np.array([0.4*final_shape[1], 0.6*final_shape[1]]),
        color="w", 
        lw=0.5
    )
    axes[1, 0].plot(
        np.array([0.4*final_shape[2], 0.6*final_shape[2]]),
        np.repeat(final_shape[1]//2, 2),
        color="w", 
        lw=0.5
    )
    axes[1, 0].plot(
        cropped_com_pixel[2], 
        cropped_com_pixel[1],
        marker="x",
        markersize=10,
        color="green",
        label="com",
    )
    axes[1, 0].plot(
        cropped_max_pixel[2], 
        cropped_max_pixel[1],
        marker="x",
        markersize=10,
        color="red",
        label="max",
    )

    axes[1, 1].matshow(
        log_cropped_data[:, cropped_max_pixel[1], :],
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",
        origin="lower"
    )
    axes[1, 1].plot(
        np.repeat(final_shape[2]//2, 2),
        np.array([0.4*final_shape[0], 0.6*final_shape[0]]),
        color="w", 
        lw=0.5
    )
    axes[1, 1].plot(
        np.array([0.4*final_shape[2], 0.6*final_shape[2]]),
        np.repeat(final_shape[0]//2, 2),
        color="w", 
        lw=0.5
    )
    axes[1, 1].plot(
        cropped_com_pixel[2], 
        cropped_com_pixel[0],
        marker="x",
        markersize=10,
        linestyle="None",
        color="green",
        label="com",
    )
    axes[1, 1].plot(
        cropped_max_pixel[2], 
        cropped_max_pixel[0],
        marker="x",
        markersize=10,
        linestyle="None",
        color="red",
        label="max"
    )

    mappable = axes[1, 2].matshow(
        np.swapaxes(
            log_cropped_data[..., cropped_max_pixel[2]], axis1=0, axis2=1),
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",
        origin="upper"
    )
    axes[1, 2].plot(
        np.repeat(final_shape[0]//2, 2),
        np.array([0.4*final_shape[1], 0.6*final_shape[1]]),
        color="w", 
        lw=0.5
    )
    axes[1, 2].plot(
        np.array([0.4*final_shape[0], 0.6*final_shape[0]]),
        np.repeat(final_shape[1]//2, 2),
        color="w", 
        lw=0.5
    )
    axes[1, 2].plot(
        cropped_com_pixel[0], 
        cropped_com_pixel[1],
        marker="x",
        markersize=10,
        color="green",
        label="com"
    )
    axes[1, 2].plot(
        cropped_max_pixel[0], 
        cropped_max_pixel[1],
        marker="x",
        markersize=10,
        color="red",
        label="max",
    )

    # handle the labels
    axes[0, 0].set_xlabel("detector dim 2 axis")
    axes[0, 0].set_ylabel("detector dim 1 axis")

    axes[0, 1].set_xlabel("detector dim 2 axis")
    axes[0, 1].set_ylabel("rocking angle axis")

    axes[0, 2].set_xlabel("rocking angle axis")
    axes[0, 2].set_ylabel("detector dim 1 axis")

    axes[1, 0].set_xlabel("cropped dim 2 axis")
    axes[1, 0].set_ylabel("cropped dim 1 axis")

    axes[1, 1].set_xlabel("cropped dim 2 axis")
    axes[1, 1].set_ylabel("cropped rocking angle axis")

    axes[1, 2].set_xlabel("cropped rocking angle axis")
    axes[1, 2].set_ylabel("cropped dim 1 axis")

    axes[0, 1].set_title("raw detector data", size=18, y=1.8)
    axes[1, 1].set_title("cropped detector data", size=18, y=1.05)

    figure.canvas.draw()
    for ax in axes.ravel():
        ax.tick_params(axis="x",direction="in", pad=-15, colors="w")
        ax.tick_params(axis="y",direction="in", pad=-25, colors="w")
        ax.xaxis.set_ticks_position("bottom")

        xticks_loc, yticks_loc = ax.get_xticks(), ax.get_yticks()
        xticks_loc[1] = yticks_loc[1] = None
        
        xlabels, ylabels = ax.get_xticklabels(), ax.get_yticklabels()
        xlabels[1] = ylabels[1] = ""
        ax.xaxis.set_major_locator(mticker.FixedLocator(xticks_loc))
        ax.yaxis.set_major_locator(mticker.FixedLocator(yticks_loc))
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)

    # handle the colorbar
    l0, b0, w0, _ = axes[0, 1].get_position().bounds
    _, b1, _, h1 = axes[1, 1].get_position().bounds
    center_y = (b0 + (b1+h1)) / 2
    # cax = figure.add_axes([l0, center_y, w0, 0.025])
    cax = figure.add_axes([l0, 0.52, w0, 0.020])
    cax.set_title("Log(Int.) (a.u.)")
    figure.colorbar(mappable, cax=cax, orientation="horizontal")

    # handle the legend
    axes[0, 1].legend(
        loc="center",
        ncol=2,
        bbox_to_anchor=((l0 + l0 + w0)/2, (b0 + center_y)/2),
        bbox_transform=figure.transFigure
    )

    figure.suptitle(title, y=0.95)

    return figure


def summary_slice_plot(
        save: str=None,
        title: str="",
        dpi: int=300,
        show: bool=True,
        voxel_size: Union[np.array, list, tuple]=None,
        isosurface: float=None,
        averaged_dspacing: float=None,
        averaged_lattice_constant: float=None,
        det_pixel_reference: Union[np.array, list, tuple]=None,
        respect_aspect=False,
        support: np.array=None,
        vmin: float=None,
        vmax: float=None,
        **kwargs
) -> matplotlib.figure.Figure:

    # take care of the aspect ratios:
    if voxel_size is not None and respect_aspect:
        aspect_ratios = {
            "xy": voxel_size[0]/voxel_size[1],
            "xz": voxel_size[0]/voxel_size[2],
            "yz":  voxel_size[1]/voxel_size[2]

        }
    else:
        aspect_ratios = {"xy": "auto", "xz": "auto","yz": "auto"}

    array_nb = len(kwargs)
    figure, axes = plt.subplots(3, array_nb, figsize=(18, 9))
    
    axes[0, 0].annotate(
                "ZY slice",
                xy=(0.2, 0.5),
                xytext=(-axes[0, 0].yaxis.labelpad - 2, 0),
                xycoords=axes[0, 0].yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                size=18
    )

    axes[1, 0].annotate(
                "ZX slice",
                xy=(0.2, 0.5),
                xytext=(-axes[1, 0].yaxis.labelpad - 2, 0),
                xycoords=axes[1, 0].yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                size=18
    )

    axes[2, 0].annotate(
                "YX slice",
                xy=(0.2, 0.5),
                xytext=(-axes[2, 0].yaxis.labelpad - 2, 0),
                xycoords=axes[2, 0].yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                size=18
    )

    mappables = {}
    if support is not None:
        support = zero_to_nan(support)
    for i, (key, array) in enumerate(kwargs.items()):
        if support is not None and key != "amplitude":
            array = support * array

        if key in PLOT_CONFIGS.keys():
            cmap = PLOT_CONFIGS[key]["cmap"]
            # check if vmin and vmax are given or not
            if vmin is None or vmax is None:
                if support is not None:
                    if key == "dspacing" or key == "lattice_constant":
                        vmin = np.nanmin(array)
                        vmax = np.nanmax(array)
                    elif key == "amplitude":
                        vmin = 0
                        vmax = np.nanmax(array)
                    else:
                        vmax = np.nanmax(np.abs(array))
                        vmin = -vmax
            else:
                vmin = PLOT_CONFIGS[key]["vmin"]
                vmax = PLOT_CONFIGS[key]["vmax"]    

        shape = array.shape

        axes[0, i].matshow(
            array[shape[0] // 2],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            origin="lower",
            aspect=aspect_ratios["yz"]
        )
        axes[1, i].matshow(
            array[:, shape[1] // 2, :],
            vmin=vmin, 
            vmax=vmax,
            cmap=cmap,
            origin="lower",
            aspect=aspect_ratios["xz"]
        )
        mappables[key] = axes[2, i].matshow(
            array[..., shape[2] // 2],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            origin="lower",
            aspect=aspect_ratios["xy"]
        )

        if key == "amplitude":
            plot_contour(axes[0, i], support[shape[0] // 2], color="k")
            plot_contour(axes[1, i], support[:, shape[1] // 2, :], color="k")
            plot_contour(axes[2, i], support[..., shape[2] // 2], color="k")
    

    table_ax = figure.add_axes([0.25, -0.05, 0.5, 0.2])
    table_ax.axis("tight")
    table_ax.axis("off")

    # format the data
    isosurface = round(isosurface, 3)
    averaged_dspacing = round(averaged_dspacing, 4)
    averaged_lattice_constant = round(averaged_lattice_constant, 4)

    # voxel_s
    table = table_ax.table(
        cellText=np.transpose([
            [np.array2string(
                voxel_size,
                formatter={"float_kind":lambda x: "%.2f" % x}
            )],
            [np.array2string(np.array(det_pixel_reference))],
            [isosurface],
            [averaged_dspacing],
            [averaged_lattice_constant]
        ]),
        colLabels=(
            "Voxel size (nm)",
            "Detetector pixel reference",
            "Isosurface",
            f"Averaged dspacing ({ANGSTROM_SYMBOL})",
            f"Averaged lattice ({ANGSTROM_SYMBOL})"
        ),
        loc="center",
        cellLoc="center"
    )
    table.scale(1.5, 2)
    table.set_fontsize(18)

    figure.subplots_adjust(hspace=0.04, wspace=0.04)
    
    for i, key in enumerate(kwargs.keys()):
        l, _, w, _ = axes[0, i].get_position().bounds
        cax = figure.add_axes([l+0.01, 0.905, w-0.02, .02])
        cax.set_title(PLOT_CONFIGS[key]["title"], size=18)
        figure.colorbar(mappables[key], cax=cax, orientation="horizontal")
    
    figure.canvas.draw()
    for i, ax in enumerate(axes.ravel()):
        if i % array_nb == 0 and list(kwargs.keys())[i//len(kwargs.keys())] == "amplitude":
            ax.tick_params(axis="x",direction="in", pad=-22, colors="w")
            ax.tick_params(axis="y",direction="in", pad=-15, colors="w")
            ax.xaxis.set_ticks_position("bottom")

            # remove the first ticks and labels
            xticks_loc, yticks_loc = ax.get_xticks(), ax.get_yticks()
            xticks_loc[1] = yticks_loc[1] = None
            
            xlabels, ylabels = ax.get_xticklabels(), ax.get_yticklabels()
            xlabels[1] = ylabels[1] = ""
            ax.xaxis.set_major_locator(mticker.FixedLocator(xticks_loc))
            ax.yaxis.set_major_locator(mticker.FixedLocator(yticks_loc))
            ax.set_xticklabels(xlabels)
            ax.set_yticklabels(ylabels)

        else:
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])

    figure.suptitle(title, size=22, y=1.03)
    # figure.subplots_adjust(hspace=0.03, wspace=0.03)

    if show:
        plt.show()
    # save the figure
    if save:
        figure.savefig(save, dpi=dpi, bbox_inches="tight")
    
    return figure