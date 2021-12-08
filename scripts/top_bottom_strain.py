import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import sys
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict

sys.path.append('/data/id01/inhouse/clatlan/pythonies/cdiutils')
from cdiutils.load.load_data import load_vtk
from cdiutils.facetanalysis.get_facet_data import facet_data_from_vtk
from cdiutils.facetanalysis.facet_utils import (
    get_rotation_matrix,
    get_miller_indices,
    planes_111_110_100,
    format_plane_name)



if __name__ == '__main__':
    import os
    import argparse
    import matplotlib as mpl

    plt.rcParams.update({
        "mathtext.fallback": "cm",
        "font.size": 14,
        "figure.dpi": 300,
        "text.usetex": True,
        "axes.prop_cycle": mpl.cycler(
            # color=mpl.cm.gist_ncar(np.linspace(0, 1, 10)))})
            color=mpl.cm.terrain(np.linspace(0, 0.95, 11)))})

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-f", "--files", required=False, type=str, nargs="+",
                    help="files to read")
    args = vars(ap.parse_args())

    # get rotation matrix
    u0 = [0, 0, 1]
    v0 = [1, 0, 0]
    u1 = [0, 1, 0]
    v1 = [0, 0, 1]
    rotation_matrix = get_rotation_matrix(u0, v0, u1, v1)
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    data = {}

    scan_digits = [182, 183, 184, 185]

    if args["files"] is None:
        file_template = "/data/id01/inhouse/clatlan/experiments/ihhc3567/" \
                        "analysis/results/facet_analysis/"\
                        "13-facets-4WS/S{}.vtk"
        files = [file_template.format(i) for i in scan_digits]
    else:
        files = args["files"]

    for file in files:
        scan = os.path.splitext(os.path.basename(file))[0]
        scan_digit = int(scan[1:])
        print("[INFO] working on scan {}".format(scan))

        vtk_data = load_vtk(file)

        scan_data = facet_data_from_vtk(vtk_data, rotation_matrix)
        scan_data["scan_digit"] = scan_digit

        data[scan] = scan_data

    # Get the 111, 110, 100 plane families
    planes111, planes110, planes100 = planes_111_110_100()

    xticks_labels = ["",
                     "0.255",
                     "0.355",
                     "0.455",
                     "0.555"]

    fig1, axes1 = plt.subplots(1, 1, figsize=(6, 4))
    lines = []

    for plane_family in [planes111, planes110, planes100]:
        for plane in plane_family:

            facet_disp = []
            facet_strain = []
            disp_std = []
            strain_std = []
            potentials = []
            scan_digits = []
            for scan, dat in data.items():
                # potentials.append(dat["potential"])
                scan_digits.append(dat["scan_digit"])
                if plane not in dat["miller_indices"].values():
                    facet_strain.append(np.nan)
                    strain_std.append(np.nan)
                    facet_disp.append(np.nan)
                    disp_std.append(np.nan)
                    continue

                for facet_id, miller_indices in dat["miller_indices"].items():
                    if plane == miller_indices:
                        ref_strain = data["S182"]["strain_avg"][facet_id]
                        ref_disp = data["S182"]["disp_avg"][facet_id]

                        facet_disp.append(dat["disp_avg"][facet_id])
                        # facet_disp.append(
                        #     dat["disp_avg"][facet_id]
                        #     - ocp_disp)
                        # facet_strain.append(dat["strain_avg"][facet_id]* 100)
                        facet_strain.append(
                            (dat["strain_avg"][facet_id]
                             - ref_strain)* 100)
                        disp_std.append(dat["disp_std"][facet_id])
                        strain_std.append(dat["strain_std"][facet_id] * 100)

            # make the label for legends
            label = "(" + str(plane).strip("[]") + ")"
            label = format_plane_name(plane)

            # SMALL CHANGE TO REMOVE 178 AND 180 scans
            # scan_digits[0] = 180

            df = pd.DataFrame({ #"potentials": potentials,
                               "scan_digits": scan_digits,
                               "strain": facet_strain,
                               "strain_std": strain_std,
                               "disp": facet_disp,
                               "disp_std": disp_std})
            if df.isnull().values.any():
                continue


            # sort the entire dataframe by scan digits
            df.sort_values(by="scan_digits", inplace=True)

            xaxis = df["scan_digits"]

            quantity = "strain"

            if plane == [0, 0, 1]:
                line_001 = axes1.plot(
                    xaxis,
                    df[quantity],
                    color="orangered",
                    # color="sandybrown",
                    marker="o",
                    label="top (001) facet"
                )[0]
            elif plane == [0, 0, -1]:
                line_00m1 = axes1.plot(
                    xaxis,
                    df[quantity],
                    color="darkorange",
                    # color="olivedrab",
                    marker="o",
                    label=r"bottom (00$\overline{1}$) facet"
                )[0]
            else:
                line = axes1.plot(
                    xaxis,
                    df[quantity],
                    # color="royalblue",
                    # color="steelblue",
                    marker="o",
                    # label="other facets from \{001\} and \{111\} families"
                    label=label
                )[0]
                lines.append(line)

    axes1.set_xlabel("E/V vs. RHE", fontsize=12)
    axes1.set_ylabel(
        r"$\delta\overline{\epsilon_{002}}$ (\%)",
        fontsize=12)
    axes1.locator_params(tight=True, nbins=4)
    # axes1.set_ylabel(
    #     r"$\overline{u_{002}}$ ($\AA$)",
    #     fontsize=12)
    all_lines = [line_001, line_00m1] + lines
    labels = [line.get_label() for line in all_lines]
    by_label = OrderedDict(zip(labels, all_lines))
    axes1.legend(
        by_label.values(),
        by_label.keys(),
        fontsize=8,
        ncol=3)

    axes1.set_xticklabels(xticks_labels, fontsize=12)
    axes1.tick_params(axis="y", labelsize=12)
    axes1.set_ylim(-0.032, 0.032)

    # fig1.suptitle("Top and bottom facet strain evolutions w.r.t OCP strain",
    #               fontsize=16)
    # fig1.suptitle(
    #     "Top and bottom facet displacement evolutions", #w.r.t OCP displacement",
    #     fontsize=15)
    plt.tight_layout()
    plt.show()
