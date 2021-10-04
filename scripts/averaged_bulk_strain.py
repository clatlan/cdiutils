import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import argparse
    import matplotlib as mpl

    plt.rcParams.update({
        "mathtext.fallback": "cm",
        "font.size": 14,
        "figure.dpi": 220,
        "text.usetex": True,
        "axes.prop_cycle": mpl.cycler(
            color=mpl.cm.gnuplot(np.linspace(0, 1, 18)))})

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-f", "--files", required=False, type=str, nargs="+",
                    help="files to read")
    args = vars(ap.parse_args())

    scan_digits = [181, 182, 183, 184, 185]

    if args["files"] is None:
        file_template = (
            "/data/id01/inhouse/clatlan/experiments/ihhc3567/analysis/results/"
            "S{}/pynxraw/S{}_amp-disp-strain_0.65_mode_avg3_apodize_blackman_"
            "crystal-frame.npz")
        files = [file_template.format(i, i) for i in scan_digits]
    else:
        files = args["files"]

    isosurfaces = [0.6, 0.65, 0.66, 0.7, 0.73, 0.75, 0.78, 0.8, 0.81, 0.82,
        0.83, 0.84, 0.85, 0.86, 0.88, 0.89, 0.9, 0.95]
    strains = {key: [] for key in isosurfaces}
    stds = {key: [] for key in isosurfaces}


    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 4))

    for iso in isosurfaces:

        for file, scan in zip(files, scan_digits):
            data = np.load(file, allow_pickle=False)
            # support = data["bulk"]
            modulus = data["amp"]
            support = np.where(
                modulus > iso * np.max(modulus),
                1,
                0)
            strain = data["strain"] * support

            averaged_bulk_strain = np.mean(strain) * 100
            std = np.std(strain) * 100

            # print("Averaged bulk strain of scan {}: {}".format(
            #     scan, averaged_bulk_strain))
            strains[iso].append(averaged_bulk_strain)
            stds[iso].append(std)
        ax.plot(
            scan_digits,
            strains[iso],
            marker="o",
            label="isosurf = {}".format(iso))
        ax2.plot(
            scan_digits,
            stds[iso],
            marker="o",
            label="isosurf = {}".format(iso))

    xticks_labels = ["",
                     "OCP",
                     "0.282 V",
                     "0.382 V",
                     "0.482 V",
                     "0.582 V"]
    ax.set_xticklabels(xticks_labels, fontsize=12)
    ax2.set_xticklabels(xticks_labels, fontsize=12)
    ax.locator_params(tight=True, nbins=5)
    ax2.locator_params(tight=True, nbins=5)
    ax.set_xlabel("Course of experiment", fontsize=12)
    ax2.set_xlabel("Course of experiment", fontsize=12)
    ax.set_ylabel(
        r"bulk averaged $\epsilon_{002}$ (\%)", fontsize=12)
    ax2.set_ylabel(
        r"bulk $\epsilon_{002}$ std (\%)", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax2.tick_params(axis="both", labelsize=12)
    ax2.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    pos = ax.get_position()
    ax.set_position([pos.x0 -0.03, pos.y0, pos.width, pos.height])
    pos = ax2.get_position()
    ax2.set_position([pos.x0 -0.03, pos.y0, pos.width, pos.height])
    ax.legend(
        ncol=1,
        fontsize=5,
        loc="upper right",
        bbox_to_anchor=(1.14, 1))
    ax2.legend(
        ncol=1,
        fontsize=5,
        loc="upper right",
        bbox_to_anchor=(1.14, 1))
    fig.suptitle("Bulk averaged strain evolution")
    fig2.suptitle("Bulk strain std evolution")
    plt.show()
