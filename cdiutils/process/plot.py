from typing import Union, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftn, fftshift, ifftshift
import os
import silx.io
import h5py

from cdiutils.utils import (
    zero_to_nan, find_suitable_array_shape, nan_center_of_mass, center,
    crop_at_center
)
from cdiutils.plot.formatting import (
    set_plot_configs, white_interior_ticks_labels, get_figure_size
)
from cdiutils.plot.slice import plot_contour


def plot_phasing_result(file_path: str) -> None:
    """
    Plot the reconstructed object in reciprocal and direct spaces.
    """
    if os.path.isfile(file_path):
        with silx.io.h5py_utils.File(file_path, "r") as file:
            data = file["entry_1/data_1/data"][()]
            support = file["entry_1/image_1/support"][()]

        shape = find_suitable_array_shape(support)
        reciprocal_space_data = np.abs(ifftshift(fftn(fftshift(data))))**2

        subplots = (2, 3)
        figure, axes = plt.subplots(
            subplots[0],
            subplots[1],
            figsize=(6, 4)
        )

        com = nan_center_of_mass(support)
        data = center(data, where=com)
        data = crop_at_center(data, final_shape=shape)
        shape = data.shape

        direct_space_amplitude = np.abs(data)
        direct_space_amplitude = (
            (direct_space_amplitude - np.min(direct_space_amplitude))
            / np.ptp(direct_space_amplitude)
        )
        direct_space_phase = np.angle(data)

        for i in range(3):
            s = [slice(None) for k in range(3)]
            s[i] = shape[i] // 2
            s = tuple(s)

            rcp_im = axes[0, i].matshow(
                np.log10(np.sum(reciprocal_space_data, axis=i)),
                cmap="turbo"
            )
            direct_space_im = axes[1, i].matshow(
                direct_space_phase[s],
                vmin=-np.pi,
                vmax=np.pi,
                alpha=direct_space_amplitude[s],
                 cmap="cet_CET_C9s"
            )
        figure.colorbar(rcp_im, ax=axes[0, 2], extend="both")
        figure.colorbar(direct_space_im, ax=axes[1, 2], extend="both")

        axes[0, 1].set_title("Log(Int.) (a.u.)")
        axes[1, 1].set_title("Phase (rad)")

        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        
        run_number = int(file_path.split("Run")[1][:4])
        scan_number = file_path.split("/")[-2][1:]
        figure.suptitle(
            f"Phasing results, scan {scan_number}, run {run_number}"
        )


def preprocessing_detector_data_plot(
        detector_data: np.array,
        cropped_data: np.array,
        det_reference_voxel: Union[np.array, list, tuple],
        det_max_voxel: Union[np.array, list, tuple],
        det_com_voxel: Union[np.array, list, tuple],
        cropped_max_voxel: Union[np.array, list, tuple],
        cropped_com_voxel: Union[np.array, list, tuple],
        title: str=""
) -> matplotlib.figure.Figure:
    """
    Plot the detector data in the full detector data frame and in the 
    cropped/centered frame.

    :param detector_data: the raw detector data (np.array)
    :param cropped_data: the cropped/centered data (np.array)
    :det_reference_voxel: the voxel reference in the full detector frame
    (np.array, list or tuple)
    :det_max_voxel: the max voxel in the full detector frame
    (np.array, list or tuple)
    :det_com_voxel: the com voxel in the full detector fame (np.array,
    list, tuple)
    :cropped_max_voxel: the max voxel in the centered/cropped detector
    frame (np.array, list or tuple)
    :cropped_com_voxel: the com voxel in the centered/cropped detector
    frame (np.array, list or tuple)
    :title: the tile of the figure (string)

    :return: the matplotlib figure object
    """
    
    subplots = (2+1, 3)
    figsize = get_figure_size("nature", subplots=subplots)
    figure, axes = plt.subplots(subplots[0]-1, subplots[1], figsize=figsize)

    log_data = np.log10(detector_data +1)
    log_cropped_data = np.log10(cropped_data +1)
    vmin = 0
    vmax = np.max(log_data)

    initial_shape = detector_data.shape
    final_shape = cropped_data.shape

    markersize = 4


    axes[0, 0].matshow(
        log_data[det_reference_voxel[0]],
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",
        origin="upper"
    )
    axes[0, 0].plot(
        np.repeat(det_reference_voxel[2], 2),
        det_reference_voxel[1] + np.array(
            [-0.1*initial_shape[1], 0.1*initial_shape[1]]),
        color="w",
        lw=0.5
    )
    axes[0, 0].plot(
        det_reference_voxel[2] + np.array(
            [-0.1*initial_shape[2], 0.1*initial_shape[2]]),
        np.repeat(det_reference_voxel[1], 2),
        color="w",
        lw=0.5
    )
    axes[0, 0].plot(
        det_com_voxel[2],
        det_com_voxel[1],
        marker="x",
        markersize=markersize,
        linestyle="None",
        color="green",
        label="com",
    )
    axes[0, 0].plot(
        det_max_voxel[2], 
        det_max_voxel[1],
        marker="x",
        markersize=markersize,
        linestyle="None",
        color="red",
        label="max"
    )

    axes[0, 1].matshow(
        log_data[:, det_reference_voxel[1], :],
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",
        origin="lower"
    )
    axes[0, 1].plot(
        np.repeat(det_reference_voxel[2], 2),
        det_reference_voxel[0] + np.array(
            [-0.1*initial_shape[0], 0.1*initial_shape[0]]),
        color="w", 
        lw=0.5
    )
    axes[0, 1].plot(
        det_reference_voxel[2] + np.array(
            [-0.1*initial_shape[2], 0.1*initial_shape[2]]),
        np.repeat(det_reference_voxel[0], 2),
        color="w", 
        lw=0.5
    )
    axes[0, 1].plot(
        det_com_voxel[2], 
        det_com_voxel[0],
        marker="x",
        markersize=markersize,
        linestyle="None",
        color="green",
        label="com",
    )
    axes[0, 1].plot(
        det_max_voxel[2],
        det_max_voxel[0],
        marker="x",
        markersize=markersize,
        linestyle="None",
        color="red",
        label="max"
    )

    mappable = axes[0, 2].matshow(
        np.swapaxes(log_data[..., det_reference_voxel[2]], axis1=0, axis2=1),
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",
        origin="upper"
    )
    axes[0, 2].plot(
        np.repeat(det_reference_voxel[0], 2),
        det_reference_voxel[1] + np.array(
            [- 0.1 * initial_shape[1],  + 0.1 * initial_shape[1]]),
        color="w",
        lw=0.5
    )
    axes[0, 2].plot(
        det_reference_voxel[0] + np.array(
            [- 0.1 * initial_shape[0],  + 0.1 * initial_shape[0]]),
        np.repeat(det_reference_voxel[1], 2),
        color="w",
        lw=0.5
    )
    axes[0, 2].plot(
        det_com_voxel[0],
        det_com_voxel[1],
        marker="x",
        markersize=markersize,
        color="green",
        label="com"
    )
    axes[0, 2].plot(
        det_max_voxel[0],
        det_max_voxel[1],
        marker="x",
        markersize=markersize,
        color="red",
        label="max",
    )

    axes[1, 0].matshow(
        log_cropped_data[final_shape[0]//2],
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
        cropped_com_voxel[2],
        cropped_com_voxel[1],
        marker="x",
        markersize=markersize,
        color="green",
        label="com",
    )
    axes[1, 0].plot(
        cropped_max_voxel[2],
        cropped_max_voxel[1],
        marker="x",
        markersize=markersize,
        color="red",
        label="max",
    )

    axes[1, 1].matshow(
        log_cropped_data[:, final_shape[1]//2, :],
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
        cropped_com_voxel[2],
        cropped_com_voxel[0],
        marker="x",
        markersize=markersize,
        linestyle="None",
        color="green",
        label="com",
    )
    axes[1, 1].plot(
        cropped_max_voxel[2],
        cropped_max_voxel[0],
        marker="x",
        markersize=markersize,
        linestyle="None",
        color="red",
        label="max"
    )

    mappable = axes[1, 2].matshow(
        np.swapaxes(
            log_cropped_data[..., final_shape[2]//2], axis1=0, axis2=1),
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
        cropped_com_voxel[0],
        cropped_com_voxel[1],
        marker="x",
        markersize=markersize,
        color="green",
        label="com"
    )
    axes[1, 2].plot(
        cropped_max_voxel[0],
        cropped_max_voxel[1],
        marker="x",
        markersize=markersize,
        color="red",
        label="max",
    )

    # handle the labels
    axes[0, 0].set_xlabel(r"detector axis$_2$")
    axes[0, 0].set_ylabel(r"detector axis$_1$")

    axes[0, 1].set_xlabel(r"detector axis$_2$")
    axes[0, 1].set_ylabel("rocking angle axis")

    axes[0, 2].set_xlabel("rocking angle axis")
    axes[0, 2].set_ylabel(r"detector axis$_1$")

    axes[1, 0].set_xlabel(r"cropped axis$_2$")
    axes[1, 0].set_ylabel(r"cropped axis$_1$")

    axes[1, 1].set_xlabel(r"cropped axis$_2$")
    axes[1, 1].set_ylabel("cropped rocking angle axis")

    axes[1, 2].set_xlabel("cropped rocking angle axis")
    axes[1, 2].set_ylabel(r"cropped axis$_1$")

    axes[0, 1].set_title("raw detector data", y=1.8)
    axes[1, 1].set_title("cropped detector data", y=1.05)

    figure.canvas.draw()
    for ax in axes.ravel():
        white_interior_ticks_labels(ax, -15, -15)

    # handle the colorbar
    l0, b0, w0, _ = axes[0, 1].get_position().bounds
    _, b1, _, h1 = axes[1, 1].get_position().bounds
    center_y = (b0 + (b1+h1)) / 2
    cax = figure.add_axes([l0, center_y, w0, 0.020])
    # cax = figure.add_axes([l0, 0.52, w0, 0.020])
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
        dpi: int=200,
        show: bool=False,
        voxel_size: Union[np.array, list, tuple]=None,
        isosurface: float=None,
        averaged_dspacing: float=None,
        averaged_lattice_parameter: float=None,
        det_reference_voxel: Union[np.array, list, tuple]=None,
        respect_aspect=False,
        support: np.array=None,
        single_vmin: float=None,
        single_vmax: float=None,
        **kwargs
) -> matplotlib.figure.Figure:

    ANGSTROM_SYMBOL, _, PLOT_CONFIGS = set_plot_configs()

    # take care of the aspect ratios:
    if voxel_size is not None and respect_aspect:
        aspect_ratios = {
            "xy": voxel_size[0]/voxel_size[1],
            "xz": voxel_size[0]/voxel_size[2],
            "yz":  voxel_size[1]/voxel_size[2]

        }
    else:
        aspect_ratios = {"xy": "auto", "xz": "auto","yz": "auto"}

    subplots = (4, len(kwargs))
    figsize = get_figure_size("nature", fraction=1, subplots=subplots)
    figure, axes = plt.subplots(
        subplots[0]-1, subplots[1], figsize=figsize
    )

    axes[0, 0].annotate(
                r"(xy)$_{cxi}$ slice",
                xy=(0.2, 0.5),
                xytext=(-axes[0, 0].yaxis.labelpad - 2, 0),
                xycoords=axes[0, 0].yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
    )

    axes[1, 0].annotate(
                r"(xz)$_{cxi}$ slice",
                xy=(0.2, 0.5),
                xytext=(-axes[1, 0].yaxis.labelpad - 2, 0),
                xycoords=axes[1, 0].yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
    )

    axes[2, 0].annotate(
                r"(zy)$_{cxi}$ slice",
                xy=(0.2, 0.5),
                xytext=(-axes[2, 0].yaxis.labelpad - 2, 0),
                xycoords=axes[2, 0].yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
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
            if single_vmin is None or single_vmax is None:
                if support is not None:
                    if key in ("dspacing", "lattice_parameter"):
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
            else:
                vmin = single_vmin
                vmax = single_vmax

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
            np.swapaxes(array[..., shape[2] // 2], axis1=0, axis2=1),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            origin="lower",
            aspect=aspect_ratios["xy"]
        )

        if key == "amplitude":
            plot_contour(axes[0, i], support[shape[0] // 2], color="k")
            plot_contour(axes[1, i], support[:, shape[1] // 2, :], color="k")
            plot_contour(
                axes[2, i],
                np.swapaxes(support[..., shape[2] // 2], axis1=0, axis2=1),
                color="k"
            )


    table_ax = figure.add_axes([0.25, -0.05, 0.5, 0.15])
    table_ax.axis("tight")
    table_ax.axis("off")

    # format the data
    isosurface = round(isosurface, 3)
    averaged_dspacing = round(averaged_dspacing, 4)
    averaged_lattice_parameter = round(averaged_lattice_parameter, 4)

    table = table_ax.table(
        cellText=np.transpose([
            [np.array2string(
                voxel_size,
                formatter={"float_kind":lambda x: "%.2f" % x}
            )],
            [np.array2string(np.array(det_reference_voxel))],
            [isosurface],
            [averaged_dspacing],
            [averaged_lattice_parameter]
        ]),
        colLabels=(
            "Voxel size (nm)",
            "Detetector voxel reference",
            "Isosurface",
            f"Averaged dspacing ({ANGSTROM_SYMBOL})",
            f"Averaged lattice ({ANGSTROM_SYMBOL})"
        ),
        loc="center",
        cellLoc="center"
    )
    table.scale(1.5, 1.5)
    table.set_fontsize(8)

    figure.subplots_adjust(hspace=0.04, wspace=0.04)
    
    for i, key in enumerate(kwargs.keys()):
        l, _, w, _ = axes[0, i].get_position().bounds
        cax = figure.add_axes([l+0.01, 0.93, w-0.02, .02])
        cax.set_title(PLOT_CONFIGS[key]["title"])
        figure.colorbar(mappables[key], cax=cax, orientation="horizontal")
    
    figure.canvas.draw()
    for i, ax in enumerate(axes.ravel()):
        ax.set_aspect("equal")
        if (
                i % len(kwargs) == 0
                and list(kwargs.keys())[i%len(kwargs.keys())] == "amplitude"
        ):
            white_interior_ticks_labels(ax, -10, -12)

        else:
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])

    figure.suptitle(title, y=1.035)
    # figure.subplots_adjust(hspace=0.03, wspace=0.03)

    if show:
        plt.show()
    # save the figure
    if save:
        figure.savefig(save, dpi=dpi)
    
    return figure


def plot_q_lab_orthogonalization_process(
        detector_data: np.ndarray,
        orthogonalized_data: np.ndarray,
        q_lab_regular_grid: np.ndarray,
        where_in_det_space: Optional[tuple]=None,
        where_in_ortho_space: Optional[tuple]=None,
        title: str=""
) -> matplotlib.figure.Figure:
    """
    Plot the intensity in the detector frame, index-of-q lab frame
    and q lab frame.
    """

    if where_in_det_space is None:
        print(
            "where_in_det_space parameter not provided, will plot the data"
            " at the center"
        )
        where_in_det_space = tuple(e // 2 for e in detector_data.shape)

    figure, axes = plt.subplots(3, 3, figsize=(12, 8))

    axes[0, 0].matshow(np.log(detector_data[where_in_det_space[0]]+1))
    axes[0, 0].plot(
        where_in_det_space[2], where_in_det_space[1], color="w", marker="x")

    axes[0, 1].matshow(
        np.log(detector_data[:, where_in_det_space[1]]+1))
    axes[0, 1].plot(
        where_in_det_space[2], where_in_det_space[0], color="w", marker="x")

    axes[0, 2].matshow(
        np.log(
            np.swapaxes(
                detector_data[:, :, where_in_det_space[2]],
                axis1=0,
                axis2=1
            ) + 1
        ),
    )
    axes[0, 2].plot(
        where_in_det_space[0], where_in_det_space[1], color="w", marker="x")

    axes[0, 0].set_xlabel(r"detector axis$_2$")
    axes[0, 0].set_ylabel(r"detector axis$_1$")
    axes[0, 1].set_xlabel(r"detector axis$_2$")
    axes[0, 1].set_ylabel(r"detector axis$_0$")
    axes[0, 2].set_xlabel(r"detector axis$_0$")
    axes[0, 2].set_ylabel(r"detector axis$_1$")

    if where_in_ortho_space is None:
        print(
            "where_in_ortho_space parameter not provided, will plot the "
            "data at the center"
        )
        where_in_ortho_space = tuple(
            e // 2 for e in orthogonalized_data.shape)

    axes[1, 0].matshow(
        np.log(
            np.swapaxes(orthogonalized_data[where_in_ortho_space[0]],
            axis1=0,
            axis2=1
            )+1 # add 1 to avoid log(0)
        ),
        origin="lower"
    )

    axes[1, 1].matshow(
        np.log(
            np.swapaxes(
                orthogonalized_data[:, where_in_ortho_space[1]],
                axis1=0,
                axis2=1
            )+1 # add 1 to avoid log(0)
        ),
        origin="lower"
    )

    axes[1, 2].matshow(
        np.log(orthogonalized_data[:, :, where_in_ortho_space[2]]+1),
        origin="lower"
    )

    axes[1, 0].set_xlabel(r"y$_{lab}/$x$_{cxi}$")
    axes[1, 0].set_ylabel(r"z$_{lab}/$y$_{cxi}$")
    axes[1, 1].set_xlabel(r"x$_{lab}/$z$_{cxi}$")
    axes[1, 1].set_ylabel(r"z$_{lab}/$y$_{cxi}$")
    axes[1, 2].set_xlabel(r"y$_{lab}/$x$_{cxi}$")
    axes[1, 2].set_ylabel(r"x$_{lab}/$z$_{cxi}$")

    # load the orthogonalized grid values
    x_array, y_array, z_array = q_lab_regular_grid

    # careful here, in contourf it is not the matrix convention !
    axes[2, 0].contourf(
        y_array, # must be the matplotlib xaxis array / numpy axis1
        z_array, # must be the matplotlib yaxis array / numpy axis0
        np.log(
            np.swapaxes(
                orthogonalized_data[where_in_ortho_space[0]]+1,
                axis1=0,
                axis2=1
            )
        ),
        levels=100,
    )

    axes[2, 1].contourf(
        x_array, # must be the matplotlib xaxis array / numpy axis1
        z_array, # must be the matplotlib yaxis array / numpy axis0
        np.log(
            np.swapaxes(
                orthogonalized_data[:, where_in_ortho_space[1]]+1,
                axis1=0,
                axis2=1
            )
        ),
        levels=100,
    )

    axes[2, 2].contourf(
        y_array, # must be the matplotlib xaxis array / numpy axis1
        x_array, # must be the matplotlib yaxis array / numpy axis0
        np.log(
            orthogonalized_data[:, :, where_in_ortho_space[2]]
            +1 # add 1 to avoid log(0)
        ),
        levels=100,
    )
    ANGSTROM_SYMBOL, _, _ = set_plot_configs()
    # axes[2, 0].set_xlabel(
    #     r"Q$_{\text{y}_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
    # axes[2, 0].set_ylabel(
    #     r"Q$_{\text{z}_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
    # axes[2, 1].set_xlabel(
    #     r"Q$_{\text{x}_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
    # axes[2, 1].set_ylabel(
    #     r"Q$_{\text{z}_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
    # axes[2, 2].set_xlabel(
    #     r"Q$_{\text{y}_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
    # axes[2, 2].set_ylabel(
    #     r"Q$_{\text{x}_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
    
    axes[2, 0].set_xlabel(
        r"Q$_{y_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
    axes[2, 0].set_ylabel(
        r"Q$_{z_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
    axes[2, 1].set_xlabel(
        r"Q$_{x_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
    axes[2, 1].set_ylabel(
        r"Q$_{z_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
    axes[2, 2].set_xlabel(
        r"Q$_{y_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
    axes[2, 2].set_ylabel(
        r"Q$_{x_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")


    axes[0, 1].set_title(r"Raw data in \textbf{detector frame}")
    axes[1, 1].set_title(r"Orthogonalized data in \textbf{index-of-q lab frame}")
    axes[2, 1].set_title(r"Orthogonalized data in \textbf{q lab frame}")

    figure.canvas.draw()
    for ax in axes.ravel():
        # ax.tick_params(axis="x", bottom=True, top=False, labeltop=False, labelbottom=True)
        white_interior_ticks_labels(ax)
    for ax in axes[2].ravel():
        ax.set_aspect("equal")

    figure.suptitle(title)
    text = (
        "The white X marker shows the\nreference voxel used for the"
        "\ntransformation"
    )
    figure.text(0.05, 0.92, text, fontsize=12, transform=figure.transFigure)
    figure.tight_layout()

    return figure


def plot_direct_lab_orthogonalization_process(
        detector_direct_space_data: np.ndarray,
        direct_lab_data: np.ndarray,
        direct_lab_regular_grid: list[np.ndarray],
        title: str=""
) -> matplotlib.figure.Figure:
    """
    Plot the intensity in the detector frame, index-of-direct lab frame
    and direct lab frame.
    """

    plot_at = tuple(e // 2 for e in detector_direct_space_data.shape)

    figure, axes = plt.subplots(3, 3, figsize=(12, 8))

    axes[0, 0].matshow(detector_direct_space_data[plot_at[0]])
    axes[0, 1].matshow(detector_direct_space_data[:, plot_at[1]])
    axes[0, 2].matshow(
        np.swapaxes(
            detector_direct_space_data[:, :, plot_at[2]],
            axis1=0,
            axis2=1
        )
    )

    axes[0, 0].set_xlabel(r"detector axis$_2$")
    axes[0, 0].set_ylabel(r"detector axis$_1$")
    axes[0, 1].set_xlabel(r"detector axis$_2$")
    axes[0, 1].set_ylabel(r"detector axis$_0$")
    axes[0, 2].set_xlabel(r"detector axis$_0$")
    axes[0, 2].set_ylabel(r"detector axis$_1$")

    plot_at = tuple(e // 2 for e in direct_lab_data.shape)

    axes[1, 0].matshow(
            np.swapaxes(direct_lab_data[plot_at[0]],
            axis1=0,
            axis2=1
        ),
        origin="lower"
    )

    axes[1, 1].matshow(
        np.swapaxes(
            direct_lab_data[:, plot_at[1]],
            axis1=0,
            axis2=1
        ),
        origin="lower"
    )

    axes[1, 2].matshow(
        direct_lab_data[:, :, plot_at[2]],
        origin="lower"
    )

    x_array, y_array, z_array = direct_lab_regular_grid
    axes[2, 0].contourf(
        y_array,
        z_array,
        np.swapaxes(
            direct_lab_data[plot_at[0]],
            axis1=0,
            axis2=1
        ),
        levels=100
    )
    axes[2, 1].contourf(
        x_array,
        z_array,
        np.swapaxes(
            direct_lab_data[:, plot_at[1]],
            axis1=0,
            axis2=1
        ),
        levels=100,
    )

    axes[2, 2].contourf(
        y_array,
        x_array,
        direct_lab_data[:, :, plot_at[2]],
        levels=100
    )
    # for ax in axes[2, :].ravel():
    #     ax.minorticks_on()
    for ax in axes.ravel():
        ax.set_aspect("equal")

    axes[1, 0].set_xlabel(r"y$_{lab}/$x$_{cxi}$")
    axes[1, 0].set_ylabel(r"z$_{lab}/$y$_{cxi}$")
    axes[1, 1].set_xlabel(r"x$_{lab}/$z$_{cxi}$")
    axes[1, 1].set_ylabel(r"z$_{lab}/$y$_{cxi}$")
    axes[1, 2].set_xlabel(r"y$_{lab}/$x$_{cxi}$")
    axes[1, 2].set_ylabel(r"x$_{lab}/$z$_{cxi}$")

    axes[2, 0].set_xlabel(r"y$_{lab}$ (nm)")
    axes[2, 0].set_ylabel(r"z$_{lab}$ (nm)")
    axes[2, 1].set_xlabel(r"x$_{lab}$ (nm)")
    axes[2, 1].set_ylabel(r"z$_{lab}$ (nm)")
    axes[2, 2].set_xlabel(r"y$_{lab}$ (nm)")
    axes[2, 2].set_ylabel(r"x$_{lab}$ (nm)")

    axes[0, 1].set_title(r"Raw data in \textbf{detector frame}")
    axes[1, 1].set_title(r"Orthogonalized data in \textbf{index-of-direct lab frame}")
    axes[2, 1].set_title(r"Orthogonalized data in \textbf{direct lab frame}")

    figure.canvas.draw()
    for ax in axes.ravel():
        white_interior_ticks_labels(ax)

    figure.suptitle(title)
    figure.tight_layout()

    return figure


def plot_final_object_fft(
        final_object_fft: np.ndarray,
        experimental_ortho_data: np.ndarray,
        final_object_q_lab_regular_grid: np.ndarray,
        exp_data_q_lab_regular_grid: np.ndarray,
        where_in_ortho_space: Optional[tuple]=None,
        title: str=""
) -> matplotlib.figure.Figure:
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))

    plot_at = tuple(e // 2 for e in final_object_fft.shape)

    # load the orthogonalized grid values
    x_array, y_array, z_array = final_object_q_lab_regular_grid

     # careful here, in contourf it is not the matrix convention !
    axes[0, 0].contourf(
        y_array, # must be the matplotlib xaxis array / numpy axis1
        z_array, # must be the matplotlib yaxis array / numpy axis0
        np.log(
            np.swapaxes(
                final_object_fft[plot_at[0]]+1,
                axis1=0,
                axis2=1
            )
        ),
        levels=100,
    )

    axes[0, 1].contourf(
        x_array, # must be the matplotlib xaxis array / numpy axis1
        z_array, # must be the matplotlib yaxis array / numpy axis0
        np.log(
            np.swapaxes(
                final_object_fft[:, plot_at[1]]+1,
                axis1=0,
                axis2=1
            )
        ),
        levels=100,
    )

    axes[0, 2].contourf(
        y_array, # must be the matplotlib xaxis array / numpy axis1
        x_array, # must be the matplotlib yaxis array / numpy axis0
        np.log(
            final_object_fft[:, :, plot_at[2]]
            +1 # add 1 to avoid log(0)
        ),
        levels=100,
    )

    # load the orthogonalized grid values
    x_array, y_array, z_array = exp_data_q_lab_regular_grid

    # careful here, in contourf it is not the matrix convention !
    axes[1, 0].contourf(
        y_array, # must be the matplotlib xaxis array / numpy axis1
        z_array, # must be the matplotlib yaxis array / numpy axis0
        np.log(
            np.swapaxes(
                experimental_ortho_data[where_in_ortho_space[0]]+1,
                axis1=0,
                axis2=1
            )
        ),
        levels=100,
    )

    axes[1, 1].contourf(
        x_array, # must be the matplotlib xaxis array / numpy axis1
        z_array, # must be the matplotlib yaxis array / numpy axis0
        np.log(
            np.swapaxes(
                experimental_ortho_data[:, where_in_ortho_space[1]]+1,
                axis1=0,
                axis2=1
            )
        ),
        levels=100,
    )

    axes[1, 2].contourf(
        y_array, # must be the matplotlib xaxis array / numpy axis1
        x_array, # must be the matplotlib yaxis array / numpy axis0
        np.log(
            experimental_ortho_data[:, :, where_in_ortho_space[2]]
            +1 # add 1 to avoid log(0)
        ),
        levels=100,
    )

    ANGSTROM_SYMBOL, _, _ = set_plot_configs()
    for i in range(2):
        # axes[i, 0].set_xlabel(
        #     r"Q$_{\text{y}_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
        # axes[i, 0].set_ylabel(
        #     r"Q$_{\text{z}_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
        # axes[i, 1].set_xlabel(
        #     r"Q$_{\text{x}_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
        # axes[i, 1].set_ylabel(
        #     r"Q$_{\text{z}_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
        # axes[i, 2].set_xlabel(
        #     r"Q$_{\text{y}_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
        # axes[i, 2].set_ylabel(
        #     r"Q$_{\text{x}_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
        
        axes[i, 0].set_xlabel(
            r"Q$_{y_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
        axes[i, 0].set_ylabel(
            r"Q$_{z_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
        axes[i, 1].set_xlabel(
            r"Q$_{x_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
        axes[i, 1].set_ylabel(
            r"Q$_{z_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
        axes[i, 2].set_xlabel(
            r"Q$_{y_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")
        axes[i, 2].set_ylabel(
            r"Q$_{x_{lab}}$ " + f"({ANGSTROM_SYMBOL}" + r"$^{-1})$")

    axes[0, 1].set_title(
        r"FFT of final object in \textbf{centered q lab frame}")
    axes[1, 1].set_title(
        r"Orthogonalized experimental data in \textbf{q lab frame}")

    figure.canvas.draw()
    for ax in axes.ravel():
        ax.set_aspect("equal")
        white_interior_ticks_labels(ax)

    figure.suptitle(title)
    figure.tight_layout()

    return figure
