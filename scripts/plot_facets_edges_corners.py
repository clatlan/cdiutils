import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    import argparse
    import os
    import matplotlib as mpl

    plt.rcParams.update({
    # "axes.facecolor": "gainsboro",
    "font.size": 16,
    "figure.dpi": 140,
    "axes.labelsize": 19,
    "text.usetex": True,
    "axes.prop_cycle": mpl.cycler(
        color=mpl.cm.gist_ncar(np.linspace(0, 1, 8)))})

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--files", required=False, type=str, nargs="+",
                    help="files to read")

    args = vars(ap.parse_args())

    voltage_shift = 0.282
    potentials = [voltage_shift + i \
        for i in [0.6, 0.6, 0.6, 0.6, 0, 0.1, 0.2, 0.3]]

    scan_digits = [178, 179, 180, 181, 182, 183, 184, 185]
    scan_digits=[181, 182, 183, 184, 185]

    xticks_labels = ["",
                     "1\nNo electrolyte",
                     "2\nNo electrolyte",
                     "3\nOCP",
                     "4\nOCP",
                     "5\n0.282 V/ RHE",
                     "6\n0.382 V/ RHE",
                     "7\n0.482 V/ RHE",
                     "8\n0.582 V/ RHE"]
    xticks_labels = ["",
                     "Open Circuit Potential",
                     "0.282 V/ RHE",
                     "0.382 V/ RHE",
                     "0.482 V/ RHE",
                     "0.582 V/ RHE"]

    if args["files"] is None:
        file_template = "/data/id01/inhouse/clatlan/experiments/ihhc3567/"\
                        "analysis/results/facet_analysis/"\
                        "S{}/S{}_planes_iso0.65.dat"
        files = [file_template.format(i, i) for i in scan_digits]
    else:
        files = args["files"]

    surface_strain = []
    edge_strain = []
    corner_strain = []

    surface_strain_std = []
    edge_strain_std = []
    corner_strain_std = []

    for i, file in enumerate(files):
        scan = os.path.splitext(os.path.basename(file))[0][:4]
        print("[INFO] working on scan {} (potential={})".format(scan,
                                                                potentials[i]))

        df = pd.read_csv(file, delimiter="\t", index_col=0)

        surface_strain.append(df.loc["surface   ", "<strain>  "] * 100)
        edge_strain.append(df.loc["edges     ", "<strain>  "] * 100)
        corner_strain.append(df.loc["corners   ", "<strain>  "] * 100)

        surface_strain_std.append(df.loc["surface   ", "std dev   "] * 100)
        edge_strain_std.append(df.loc["edges     ", "std dev   "] * 100)
        corner_strain_std.append(df.loc["corners   ", "std dev   "] * 100)

    fig, ax = plt.subplots(figsize=(9, 6))


    # overwrite the scan digits so that spacing between scan is constant
    # scan_digits[0] = 180
    ax.errorbar(scan_digits, surface_strain, surface_strain_std,
                label="Facet strain", marker="o", capsize=5.0, capthick=1.5)
    ax.errorbar(scan_digits, edge_strain, edge_strain_std,
                label="Edge strain", marker="s", capsize=5.0, capthick=1.5)
    ax.errorbar(scan_digits, corner_strain, corner_strain_std,
                label="Corner strain", marker="^", capsize=5.0, capthick=1.5)

    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.plot(1, -.025, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    # ax.spines["bottom"].set_position(("data", -0.025))
    # ax.spines["left"].set_position(("data", 177))
    # ax.plot(177, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    ax.set_xlim(180.7, 185.3)

    ax.set_xticklabels(xticks_labels, fontsize=12)
    ax.set_xlabel("Course of experiment", fontsize=16)
    ax.set_ylabel(r"$\overline{\epsilon_{002}}$  (\%)", fontsize=16)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.94), fontsize=12,
               ncol=len(ax.lines))
    fig.suptitle("Facet, edge and corner strain evolutions", fontsize=20)

    plt.show()
