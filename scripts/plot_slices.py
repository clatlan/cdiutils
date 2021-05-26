#!/data/id01/inhouse/clatlan/.envs/cdiutils/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os

sys.path.append("/data/id01/inhouse/clatlan/pythonies/cdiutils")
from cdiutils.plot.plot import plot_slices


if __name__ == "__main__":
    import argparse

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--files", required=True, type=str, nargs="+",
                    help="files to read")
    args = vars(ap.parse_args())

    plt.rcParams.update({
    "font.size": 12,
    "figure.dpi": 140,
    "text.usetex": True,
    "axes.prop_cycle": mpl.cycler(
        color=mpl.cm.gist_ncar(np.linspace(0, 1, 9)))})

    disps = []
    strains = []
    titles = []
    for file in args["files"]:
        scan = os.path.splitext(os.path.basename(file))[0][:4]
        # if scan != "S178":
        #     continue
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
        strain = strain[crop: shape[0] - crop,
                        crop: shape[1] - crop,
                        crop: shape[2] - crop]


        disps.append(disp)
        strains.append(strain)
        titles.append(scan)


    titles = ["S178\nNo electrolyte\nNo potential",
              "S179\nNo electrolyte\nNo potential",
              "S180\nElectrolyte\nNo potential",
              "S181\nElectrolyte\nNo potential",
              "S182\n0.282 V/ RHE",
              "S183\n0.382 V/ RHE",
              "S184\n0.482 V/ RHE",
              "S185\n0.582 V/ RHE"]

    disp_fig = plot_slices(*disps, titles=titles,
                           show=False, cmap="bwr",
                           vmin=-3.2e-1, vmax=3.2e-1,
                           suptitle="2D slides of displacement field ($\AA$)",
                           data_stacking="horizontal")

    disp_fig = plot_slices(*strains, titles=titles,
                           show=False, cmap="seismic",
                           vmin=-4.6e-2, vmax=4.6e-2,
                           suptitle="2D slides of strain field (\%)",
                           data_stacking="horizontal")
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
