import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os


if __name__ == "__main__":
    import argparse

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--files", required=False, type=str, nargs="+",
                    help="files to read")
    args = vars(ap.parse_args())

    plt.rcParams.update({
        # "text.color": "b",
        # "xtick.color": "b",
        "font.size": 12,
        "xtick.labelsize": 12,
        "figure.titlesize": 18,
        "text.usetex": True,
        "figure.dpi": 220,
        "axes.prop_cycle": mpl.cycler(
            # color=mpl.cm.gist_ncar(np.linspace(0.02, 0.95, 6)))})
            color=mpl.cm.gnuplot(np.linspace(0.02, 0.95, 4)))})

    # scan_digits = [179, 181, 182, 183, 184, 185]
    scan_digits = [182, 183, 184, 185]

    if args["files"] is None:
        file_template = "/data/id01/inhouse/clatlan/experiments/ihhc3567/"\
                        "analysis/results/S{}/pynxraw/S{}_amp-disp-strain_"\
                        "0.65_mode_avg3_apodize_blackman_crystal-frame.npz"
        files = [file_template.format(i, i) for i in scan_digits]
    else:
        files = args["files"]

    isosurface = 0.66
    crop = 63
    labels = ["0.255 V",
              "0.355 V",
              "0.455 V",
              "0.555 V"]

    xticks_labels = ["bottom",
                     "top"]

    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 4))

    for file, label in zip(files, labels):
        print(file, label)
        data = np.load(file, allow_pickle=False)
        modulus = data["amp"]
        bulk = data["bulk"]
        modulus = (modulus - modulus.min()) / modulus.ptp()
        support = np.where(modulus > isosurface, 1, 0)
        # support = bulk
        disp = data["displacement"] * support
        strain = data["strain"] * support * 100
        strain = strain[crop:-crop,
                        crop:-crop,
                        crop:-crop]

        x_linescan_pos = strain.shape[0] // 2
        z_linescan_pos = strain.shape[2] // 2

        linescan = strain[x_linescan_pos, :, z_linescan_pos]
        linescan_sum = np.cumsum(linescan, dtype=float)
        x_axis = np.arange(0, strain.shape[1])
        line = ax1.plot(x_axis, linescan, label=label)
        line2 = ax2.plot(x_axis, linescan_sum, label=label)

        if label == "0.255 V":
            ax3.matshow(
                strain[strain.shape[0] // 2, ...],
                origin="lower",
                cmap="seismic",
                vmin=-3e-2,
                vmax=3e-2)
            ax3.axvline(x=x_linescan_pos, color="black")
            # ax2.plot(
            #     x_axis,
            #     np.ones(x_axis.shape[0]) * x_linescan_pos,
            #     color="black")

    fig1.suptitle("Line scan of projected strain")
    fig2.suptitle("Cumulative line scan of projected strain")
    for ax in [ax1, ax2]:
        ax.set_xticks([1, 22])
        ax.set_xticklabels(xticks_labels, fontsize=12)
        ax.locator_params(tight=True, nbins=2)
        ax.set_xlabel(
            "Position along the [001] direction, from bottom to top",
            fontsize=14)
        ax.set_ylabel(r"$\epsilon_{002}$ (\%)", fontsize=15)
        ax.legend()
    plt.show()
