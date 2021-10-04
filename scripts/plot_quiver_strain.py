import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np

if __name__ == '__main__':

    import argparse

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--files", required=False, type=str, nargs="+",
                    help="files to read")
    args = vars(ap.parse_args())

    scan_digits = [182]

    if args["files"] is None:
        file_template = "/data/id01/inhouse/clatlan/experiments/ihhc3567/"\
                        "analysis/results/S{}/pynxraw/S{}_amp-disp-strain_"\
                        "0.65_mode_avg3_apodize_blackman_crystal-frame.npz"
        files = [file_template.format(i, i) for i in scan_digits]
    else:
        files = args["files"]

    for file in files:
        isosurface = 0.65
        data = np.load(file, allow_pickle=False)
        modulus = data["amp"]
        support = np.where(
            modulus > isosurface * np.max(modulus),
            1,
            0)

        strain = data["strain"]

        nonzero_coordinates = np.where(support > 0)
        strain_OI = strain[nonzero_coordinates[0],
                           nonzero_coordinates[1],
                           nonzero_coordinates[2]]

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(projection="3d")

        vmin = np.min(strain_OI)
        vmax = np.max(strain_OI)

        colormap = mpl.cm.get_cmap('seismic')
        colors = (strain_OI.ravel() - strain_OI.min()) / strain_OI.ptp()
        colors = np.concatenate((colors, np.repeat(colors, 2)))
        colors = colormap(colors)

        sm = mpl.cm.ScalarMappable(cmap=colormap, norm=None)

        q = ax.quiver(
            nonzero_coordinates[0],
            nonzero_coordinates[1],
            nonzero_coordinates[2],
            0,
            0,
            strain_OI[...],
            # arrow_length_ratio=0.2,
            # normalize=False,
            # length=5,
            colors=colors
            )

        # sm.set_array(np.linspace(vmin, vmax))
        fig.colorbar(sm, ax=ax, orientation='vertical')
        q.set_edgecolor(colors)
        q.set_facecolor(colors)

        plt.show()
