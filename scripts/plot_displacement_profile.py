import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import splev, splrep
from scipy.ndimage import rotate
import sys
import os
import colorcet as cc
import imageio

sys.path.append('/data/id01/inhouse/clatlan/pythonies/cdiutils')
from cdiutils.plot.plot import plot_slices
from cdiutils.utils import crop_at_center


def remove_useless(data, support, nan_value=4.2e-6):
    B = np.where(support == 0, nan_value, 0)
    return data + B


def normalize(data, zero_centered=True):
    if zero_centered:
        abs_max = np.max([np.abs(np.min(data)), np.abs(np.max(data))])
        vmin, vmax = -abs_max, abs_max
        ptp = vmax - vmin
    else:
        ptp = np.ptp(data)
    return (data - vmin) / ptp


if __name__ == '__main__':
    import argparse

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--files", required=False, type=str, nargs="+",
                    help="files to read")
    args = vars(ap.parse_args())

    plt.rcParams.update({
        # "figure.facecolor": "#51576e",
        # "axes.facecolor": "#51576e",
        "font.size": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "figure.titlesize": 22,
        "text.usetex": True,
        "figure.dpi": 300,
        "axes.prop_cycle": mpl.cycler(
            color=mpl.cm.gnuplot(np.linspace(0, 1, 30)))})

    scan_digits = [182, 183, 184, 185]
    # scan_digits = [185]

    if args["files"] is None:
        file_template = "/data/id01/inhouse/clatlan/experiments/ihhc3567/"\
                        "analysis/results/S{}/pynxraw/S{}_amp-disp-strain_"\
                        "0.65_mode_avg3_apodize_blackman_crystal-frame.npz"
        files = [file_template.format(i, i) for i in scan_digits]
    else:
        files = args["files"]
    # dhlk = 2pi / qhkl, qhlk = 2pi / dhkl
    # a = dhkl / sqrt(h**2 + k**2 + l**2)
    # ||uhkl|| = phihkl / ||qhkl||
    lattice_parameter = 3.92
    q_002 = 4 * np.pi / lattice_parameter
    # lattice_parameter = 3.06
    # q_002 = 2 * np.pi / lattice_parameter

    # y_shift = 0.062  # voxels
    # # y_shift = 2  # Angstroms

    crop_fit = [1, -4]
    crop_fit = [0, -3]
    isosurface = 0.65

    # sweep_range = np.arange(64, 87)
    target_shape = (50, 23, 50)
    # target_shape = (52, 30, 52)

    scale = 18
    scale = 22

    background_cmap = cc.cm.CET_D13
    #background_cmap = "seismic"
    foreground_cmap = cc.cm.CET_D8

    make_gif = False
    images = []

    comment = "scale_up"

    for file, scan in zip(files, scan_digits):
        print("[INFO] Working on scan # {}".format(scan))
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))

        data = np.load(file, allow_pickle=False)
        support = data["bulk"]
        modulus = data["amp"]
        modulus = (modulus - np.min(modulus)) / np.ptp(modulus)
        support = np.where(modulus >= isosurface, 1, 0)
        if scan == 182:
            print("The support of reference is that of scan", scan)
            modulus = data["amp"]
            modulus = (modulus - np.min(modulus)) / np.ptp(modulus)
            support_ref = np.where(modulus >= isosurface, 1, 0)
        #support = support_ref

        disp = data["displacement"] * support / q_002
        strain = data["strain"] * support * 100

        strain = rotate(strain, +45, axes=(0, 2))
        disp = rotate(disp, +45, axes=(0, 2))
        support = rotate(support, +45, axes=(0, 2))

        disp = crop_at_center(
            disp,
            final_shape=target_shape
        )
        strain = crop_at_center(
            strain,
            final_shape=target_shape
        )
        support = crop_at_center(
            support,
            final_shape=target_shape
        )

        # Flip the slice if needed
        disp = np.flip(disp, axis=2)
        strain = np.flip(strain, axis=2)
        support = np.flip(disp, axis=2)

        # disp = disp * support / q_002
        # strain = strain * np.where(support == 0, np.nan, 1) * 100

        # Make the last minor slice adjustement
        disp = disp[..., crop_fit[0]: crop_fit[-1]]
        strain = strain[..., crop_fit[0]: crop_fit[-1]]
        support = support[..., crop_fit[0]: crop_fit[-1]]

        # disp = remove_useless(disp, support)
        # strain = remove_useless(strain, support)

        vmin_disp, vmax_disp = -0.1, 0.1
        # vmin, vmax = -0.03, 0.03
        shape = disp.shape
        x_axis = np.arange(0, shape[2])
        interpolate = 2
        x_linescan_pos = shape[0] // 2

        background = ax1.matshow(
            support[x_linescan_pos, ...],
            origin="lower",
            cmap="Greys",
            alpha=1
        )
        background = ax1.matshow(
            strain[x_linescan_pos, ...],
            origin="lower",
            cmap=background_cmap,
            vmin=-4.6e-2,
            vmax=4.6e-2,
            # vmin=-0.1,
            # vmax=0.1,
            alpha=0.6
        )

        for z in np.arange(0, disp.shape[1]):
            print("[INFO] Processing line scan #{}".format(z))

            ax1.axhline(
                y=z,
                xmin=0.1,
                color="grey",
                ls=":",
                linewidth=.6)

            ax1, sm = plot_deviation(
                ax1,
                x_axis,
                deviation=disp[x_linescan_pos, z, :],
                y_pos=z,
                scale=scale,
                # attribute=strain[x_linescan_pos, z, :],
                vmin=-0.1,
                vmax=0.1,
                centered=True,
                # cmap="BrBG_r",
                # cmap="PRGn",
                # cmap=cc.cm.CET_D4,
                cmap=foreground_cmap,
                arrow=True,
                interpolate=interpolate)

        ax1.spines["left"].set_position(("data", -2))
        ax1.spines["bottom"].set_position(("data", -1))
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        ax1.plot(
            1,
            -1,
            ">k",
            transform=ax1.get_yaxis_transform(),
            clip_on=False
        )
        ax1.plot(
            -2,
            1,
            "^k",
            transform=ax1.get_xaxis_transform(),
            clip_on=False
        )
        ax1.set_xlabel(r"Y [$\overline{11}0$]")
        ax1.set_ylabel("Z [001]")
        ax1.tick_params(
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,)

        divider = make_axes_locatable(ax1)

        cax = divider.append_axes("right", size="4%", pad=0.2)
        bgd_cbar = fig1.colorbar(
            background,
            cax=cax,
        )
        bgd_cbar.ax.yaxis.set_ticks_position("right")
        bgd_cbar.ax.yaxis.set_label_position("left")
        bgd_cbar.ax.set_ylabel(
            r"$\epsilon_{002}$ (\%)",
            # r"$u_{002}$ ($\AA$)",
            fontsize=10,
            labelpad=1)
        bgd_cbar.solids.set_edgecolor("face")

        cax = divider.append_axes("right", size="4%", pad=0.4)
        cbar = fig1.colorbar(
            sm,
            cax=cax,
        )
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.set_ylabel(
            # r"$\epsilon_{002}$ (\%)",
            r"$u_{002}$ ($\AA$)",
            fontsize=10,
            # rotation=270,
            labelpad=1)

        [t.set_visible(False) for t in ax1.get_xticklines()]
        [t.set_visible(False) for t in ax1.get_yticklines()]
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

        fig1.tight_layout()
        output_file = (
            "/data/id01/inhouse/clatlan/exchange/facet-dependent-images/"
            "cross_section_quivers/strain-disp/cut_111_plane/"
            + comment
            + "_{}-{}_strain-disp_S{}.pdf".format(
                background_cmap if isinstance(background_cmap, str)
                else background_cmap.name,
                foreground_cmap if isinstance(foreground_cmap, str)
                else foreground_cmap.name,
                scan
            )
        )
        fig1.savefig(output_file, dpi=300)
        if make_gif:
            images.append(imageio.imread(output_file))

    # plt.show()
    plt.close("all")

    if make_gif:
        imageio.mimsave(
            "/data/id01/inhouse/clatlan/exchange/facet-dependent-images/"
            "cross_section_quivers/strain-disp/gifs/"
            "cut_111_plane-{}-{}strain-disp.gif".format(
                background_cmap if isinstance(background_cmap, str)
                else backbackground_cmap.name,
                foreground_cmap if isinstance(foreground_cmap, str)
                else foreground_cmap.name,
            ),
            images)
