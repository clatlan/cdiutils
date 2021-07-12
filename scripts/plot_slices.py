#!/data/id01/inhouse/clatlan/.envs/cdiutils/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os

sys.path.append('/data/id01/inhouse/clatlan/pythonies/cdiutils')
from cdiutils.plot.plot import plot_slices


if __name__ == "__main__":
    import argparse

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--files", required=False, type=str, nargs="+",
                    help="files to read")
    args = vars(ap.parse_args())

    plt.rcParams.update({
    # "figure.facecolor": "#51576e",
    # "axes.facecolor": "#51576e",
    "text.color": "b",
    "xtick.color": "b",
    "font.size": 12,
    "xtick.labelsize": 14,
    "figure.titlesize": 22,
    "figure.dpi": 140,
    "text.usetex": True,
    "figure.dpi": 210,
    "axes.prop_cycle": mpl.cycler(
        color=mpl.cm.gist_ncar(np.linspace(0, 1, 9)))})

    scan_digits=[179, 181, 182, 183, 184, 185]

    if args["files"] is None:
        file_template = "/data/id01/inhouse/clatlan/experiments/ihhc3567/"\
                        "analysis/results/S{}/pynxraw/S{}_amp-disp-strain_"\
                        "0.65_mode_avg3_apodize_blackman_crystal-frame.npz"
        files = [file_template.format(i, i) for i in scan_digits]
    else:
        files = args["files"]

    disps = []
    strains = []
    titles = []
    for file in files:
        scan = os.path.splitext(os.path.basename(file))[0][:4]
        if scan == "S178" or scan =="S180" or scan =="S179":
            continue
        data = np.load(file, allow_pickle=False)
        support = data["bulk"]
        modulus = data["amp"] * support
        disp = data["displacement"] * support
        crop = 50
        shape = disp.shape
        disp = disp[crop: shape[0] - crop,
                    crop: shape[1] - crop,
                    crop: shape[2] - crop]
        strain = data["strain"] * support * 100
        strain[support == 0] = np.nan
        strain = strain[crop: shape[0] - crop,
                        crop: shape[1] - crop,
                        crop: shape[2] - crop]



        disps.append(disp)
        strains.append(strain)
        titles.append(scan)


    titles = ["1\nNo electrolyte",
              "2\nNo electrolyte",
              "3\nNo potential",
              "4\nNo potential",
              "5\n0.282 V/ RHE",
              "6\n0.382 V/ RHE",
              "7\n0.482 V/ RHE",
              "8\n0.582 V/ RHE"]
    titles = ["1\nNo electrolyte",
              "2\nNo potential",
              "3\n0.282 V/ RHE",
              "4\n0.382 V/ RHE",
              "5\n0.482 V/ RHE",
              "6\n0.582 V/ RHE"]
    titles = ["Open Circuit Potential",
              "0.282 V/ RHE",
              "0.382 V/ RHE",
              "0.482 V/ RHE",
              "0.582 V/ RHE"]

    disp_fig = plot_slices(*disps, titles=titles,
                           show=False, cmap="seismic",
                           vmin=-3.2e-1, vmax=3.2e-1,
                           suptitle="2D slides of displacement field ($\AA$)",
                           data_stacking="horizontal")

    strain_fig = plot_slices(*strains, titles=titles,
                           show=False, cmap="seismic",
                           vmin=-4.0e-2, vmax=4.0e-2,
                           suptitle=r"$\epsilon_{002}$ (\%)",
                           data_stacking="horizontal",
                           slice_names=["XY slice", "XZ slice", "YZ slice"])
    # strain_fig = plot_slices(*strain, titles=titles, show=False, cmap="coolwarm",
    #                          suptitle="Strain Slice")


        # axes = disp_fig.get_axes()
        # shape = support.shape
        # zeros = np.where(support[shape[0]//2, ...] == 0)
        # print(zeros)
        # # proj1 = support[zeros][shape[0]//2, ...]
        # # print(proj1.shape)
        # axes[0].matshow(zeros, =np.zeros((zeros[0].shape[0], zeros[0].shape[0])), cmap="gray_r")



    # sup_fig = plot_slices(supports[0] - supports[1])
    plt.show()
