#!/data/id01/inhouse/clatlan/.envs/cdiutils/bin/python

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import sys
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('/data/id01/inhouse/clatlan/pythonies/cdiutils')
from cdiutils.load.load_data import load_vtk
from cdiutils.facetanalysis.get_facet_data import facet_data_from_vtk
from cdiutils.facetanalysis.facet_utils import get_rotation_matrix, \
    get_miller_indices, planes_111_110_100


def plot_facets_from_vtk(vtk, points, cells):

    data = get_facet_data(vtk_data)
    facet_ids = data["disp"].keys().tolist()

    fig1 = plt.figure()
    ax = fig1.add_subplot(projection="3d")
    x = np.concatenate([data["point_coord"][facet][..., 0]
                        for facet in facet_ids])
    y = np.concatenate([data["point_coord"][facet][..., 1]
                        for facet in facet_ids])
    z = np.concatenate([data["point_coord"][facet][..., 2]
                        for facet in facet_ids])
    c = np.concatenate([np.repeat(data["disp_avg"][facet],
                                  data["point_coord"][facet].shape[0])
                        for facet in facet_ids])
    ax.scatter(x, y, z, c=c, marker="o")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    for facet in facet_ids:
        ax2.errorbar(facet, disp_avg[facet], disp_std[facet],
                     marker="o", label=facet)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import os
    import argparse
    import matplotlib as mpl

    plt.rcParams.update({
        # # "figure.facecolor": "#51576e",
        # # "axes.facecolor": "#51576e",
        # "text.color": "white",
        # # "xtick.color": "w",
        "mathtext.fallback": "cm",
        "font.size": 14,
        "figure.dpi": 250,
        "text.usetex": True,
        "axes.prop_cycle": mpl.cycler(
            color=mpl.cm.gist_ncar(np.linspace(0, 1, 12)))})

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
    data = {}

    scan_digits = [181, 182, 183, 184, 185]

    if args["files"] is None:
        file_template = "/data/id01/inhouse/clatlan/experiments/ihhc3567/" \
                        "analysis/results/facet_analysis/"\
                        "13-bigger-facets/S{}.vtk"
        files = [file_template.format(i) for i in scan_digits]
    else:
        files = args["files"]

    for file in files:
        scan = os.path.splitext(os.path.basename(file))[0]
        scan_digit = int(scan[1:])
        print("[INFO] working on scan {}".format(scan))
        if scan == "S178" or scan == "S180" or scan == "S179":
            continue
        vtk_data = load_vtk(file)

        scan_data = facet_data_from_vtk(vtk_data, rotation_matrix)
        scan_data["scan_digit"] = scan_digit

        # for facet in scan_data["facet_normals"].keys():
        #     print("facet id: {}, miller indices: {}, rotated norm: {}".format(
        #         facet,
        #         scan_data["miller_indices"][facet],
        #         scan_data["facet_normals"][facet]))

        data[scan] = scan_data

    # Get the 111, 110, 100 plane families
    planes111, planes110, planes100 = planes_111_110_100()

    voltage_shift = 0.282
    # data["S178"]["potential"] = 0.6 + voltage_shift
    # data["S179"]["potential"] = 0.6 + voltage_shift
    # data["S180"]["potential"] = 0.6 + voltage_shift
    data["S181"]["potential"] = 0.6 + voltage_shift
    data["S182"]["potential"] = 0 + voltage_shift
    data["S183"]["potential"] = 0.1 + voltage_shift
    data["S184"]["potential"] = 0.2 + voltage_shift
    data["S185"]["potential"] = 0.3 + voltage_shift

    potential_axis = [i + voltage_shift for i in [0, 0.1, 0.2, 0.3]]
    scan_axis = [data[key]["scan_digit"] for key in data.keys()]
    globals = {key: np.empty(shape=(len(scan_axis),)) for key in
               ["disp", "disp_std", "strain", "strain_std"]}

    edge_corner_strain = [data[scan]["strain_avg"][0] * 100 for scan in
                          ["S181", "S182", "S183", "S184", "S185"]]
    edge_corner_strain_std = [data[scan]["strain_std"][0] * 100 for scan in
                              ["S181", "S182", "S183", "S184", "S185"]]

    xticks_labels = ["",
                     "OCP",
                     "0.282 V/ RHE",
                     "0.382 V/ RHE",
                     "0.482 V/ RHE",
                     "0.582 V/ RHE"]

    for plane_family in [planes111, planes110, planes100]:
        print("[INFO] Working on the following planes: {}".format(
            plane_family))

        # fig, axes = plt.subplots(2, 1, figsize=(7, 5))
        # fig2, axes2 = plt.subplots(2, 1, figsize=(7, 5))
        fig4, axes4 = plt.subplots(2, 1, figsize=(7, 5))

        for plane in plane_family:

            # print("[INFO] Current plane is: {}".format(plane))
            facet_disp = []
            facet_strain = []
            disp_std = []
            strain_std = []
            potentials = []
            scan_digits = []
            for scan, dat in data.items():
                potentials.append(dat["potential"])
                scan_digits.append(dat["scan_digit"])
                if plane not in dat["miller_indices"].values():
                    facet_strain.append(np.nan)
                    strain_std.append(np.nan)
                    facet_disp.append(np.nan)
                    disp_std.append(np.nan)
                    continue

                for facet_id, miller_indices in dat["miller_indices"].items():
                    if plane == miller_indices:
                        facet_disp.append(dat["disp_avg"][facet_id])
                        facet_strain.append(dat["strain_avg"][facet_id] * 100)
                        disp_std.append(dat["disp_std"][facet_id])
                        strain_std.append(dat["strain_std"][facet_id] * 100)

            # make the label for legends
            label = "(" + str(plane).strip("[]") + ")"

            # SMALL CHANGE TO REMOVE 178 AND 180 scans
            # scan_digits[0] = 180

            df = pd.DataFrame({"potentials": potentials,
                               "scan_digits": scan_digits,
                               "strain": facet_strain,
                               "strain_std": strain_std,
                               "disp": facet_disp,
                               "disp_std": disp_std})
            if df.isnull().values.any():
                continue

            for key in globals.keys():
                if not df[key].isnull().values.any():
                    globals[key] = np.concatenate(
                        [globals[key], df[key]])

            # sort the entire dataframe by potentials
            # df.sort_values(by="potentials", inplace=True)
            # df = df[df["potentials"] <=0.3 + voltage_shift]

            # sort the entire dataframe by scan digits
            df.sort_values(by="scan_digits", inplace=True)

            xaxis = df["scan_digits"]

            # # plot displacement average per facet
            # line, = axes[0].plot(xaxis.fillna(method="ffill"),
			# 			 	 	 df["disp"].fillna(method="ffill"), ls="--")
            # axes[0].plot(xaxis, df["disp"], color=line.get_color(),
			# 		 	 marker="o", label=label)
            #
            # # plot strain average per facet
            # line, = axes[1].plot(xaxis.fillna(method="ffill"),
			# 			 	     df["strain"].fillna(method="ffill"), ls="--")
            # axes[1].plot(xaxis, df["strain"], color=line.get_color(),
			# 		 	 marker="o", label=label)
            #
            # # plot displacement std per facet
            # line, = axes2[0].plot(xaxis.fillna(method="ffill"),
			# 			 	 	  df["disp_std"].fillna(method="ffill"),
            #                       ls="--")get_data_from_vtk
            # axes2[0].plot(xaxis, df["disp_std"],
            #               color=line.get_color(), marker="o", label=label)
            #
            # # plot strain std per facet
            # line, = axes2[1].plot(xaxis.fillna(method="ffill"),
			# 			 	      df["strain_std"].fillna(method="ffill"),
            #                       ls="--")
            # axes2[1].plot(xaxis, df["strain_std"],
            #               color=line.get_color(), marker="o", label=label)

            # plot displacement std per facet
            line, = axes4[0].plot(
                xaxis.fillna(method="ffill"),
                df["strain"].fillna(method="ffill"),
                ls="--"
                )
            axes4[0].plot(
                xaxis,
                df["strain"],
                color=line.get_color(),
                marker="o",
                label=label
                )

            # plot strain std per facet
            line, = axes4[1].plot(
                xaxis.fillna(method="ffill"),
                df["strain_std"].fillna(method="ffill"),
                ls="--"
                )
            axes4[1].plot(
                xaxis,
                df["strain_std"],
                color=line.get_color(),
                marker="o",
                label=label
                )

        # axes[0].set_ylabel("Displacement average ($\AA$)", fontsize=12)
        # axes[1].set_xlabel("Potentials (V) / RHE", fontsize=12)
        # axes[1].set_xlabel("Course of experiment (scan \#)", fontsize=12)
        # axes[1].set_ylabel(r"$\overline{\epsilon_{002}}$ (\%)", fontsize=12)
        # axes[0].tick_params(axis="x", direction="out", pad=0) # pad=-15
        # axes[0].set_xticklabels(xticks_labels, fontsize=9)
        # axes[1].set_xticklabels(xticks_labels, fontsize=9)
        # fig.legend(handles=axes[0].get_legend_handles_labels()[0],
        #           loc="upper center", bbox_to_anchor=(0.5, 0.92), fontsize=8,
        #           ncol=len(axes[0].lines))
        # fig.suptitle("Displacement and strain averages", fontsize=18)
        #
        # # Print the standard deviations
        #
        # axes2[0].set_ylabel("Displacement standard deviation ($\AA$)",
        #                     fontsize=14)
        # axes2[1].set_xlabel("Potentials (V) / RHE", fontsize=12)
        # axes2[1].set_xlabel("Course of experiment (scan \#)", fontsize=14)
        # axes2[1].set_ylabel(r"$\epsilon_{002}$ standard deviation (\%)", fontsize=14)
        # axes2[0].tick_params(axis="x", direction="out", pad=0)
        # axes2[0].set_xticklabels(xticks_labels, fontsize=9)
        # axes2[1].set_xticklabels(xticks_labels, fontsize=9)
        # fig2.legend(handles=axes2[0].get_legend_handles_labels()[0],
        #             loc="upper center", bbox_to_anchor=(0.5, 0.94), fontsize=8,
        #             ncol=len(axes[0].lines))facet_edges_strain
        # fig2.suptitle("Displacement and strain standard deviations",
        #               fontsize=20)


        # Print strain average and standard deviations

        axes4[0].set_ylabel(r"$\overline{\epsilon_{002}}$  (\%)",
                            fontsize=14)
        axes4[1].set_xlabel("Course of experiment", fontsize=14)
        axes4[1].set_ylabel(r"$\epsilon_{002}$ standard deviation (\%)",
                            fontsize=14)
        axes4[0].tick_params(axis="x", direction="out")
        axes4[0].set_xticklabels(xticks_labels, fontsize=12)
        axes4[1].set_xticklabels(xticks_labels, fontsize=12)
        fig4.legend(handles=axes4[0].get_legend_handles_labels()[0],
                    loc="upper center", bbox_to_anchor=(0.5, 0.94), fontsize=8,
                    ncol=len(axes4[0].lines))
        fig4.suptitle("Strain average and associated standard deviation",
                      fontsize=20)
        axes4[0].locator_params(tight=True, nbins=5)
        axes4[1].locator_params(tight=True, nbins=5)
        axes4[0].set_ylim(-0.027, 0.027)
        axes4[1].set_ylim(0.0018, 0.0154)

    for key in globals.keys():
        glb = globals[key]
        glb = glb.reshape([glb.shape[0]//5, 5]) # CAREFULL make it 8, 8 if 8 scans
        glb = np.delete(glb, 0, 0)
        globals[key] = np.mean(glb, axis=0)

    # plt.close("all")
    fig3, axes3 = plt.subplots(figsize=(9, 6))

    # labels = {"disp": "disp avg", "strain": "strain avg",
    #           "disp_std": "disp std", "strain_std": "strain std"}
    # for key in ["disp_std"]:
    #     axes3[0].plot(scan_axis, globals[key], label=labels[key],
    #                   marker="o")
    # for key in ["strain_std"]:
    #     axes3[1].plot(scan_axis, globals[key], label=labels[key],
    #                   marker="o")

    # axes3.errorbar(scan_axis, globals["disp"], globals["disp_std"],
    #                   marker="^", capsize=5.0, capthick=1.5, label="average")
    axes3.errorbar(
        scan_axis,
        globals["strain"],
        globals["strain_std"],
        marker="^",
        capsize=5.0,
        capthick=1.5,
        label="facet strain avg")
    # axes3[0].plot(scan_axis, globals["disp_std"], label="standard deviation",
    #               marker="o")
    axes3.plot(
        scan_axis,
        globals["strain_std"],
        label="facet strain std",
        marker="o")

    axes3.errorbar(
        scan_axis,
        edge_corner_strain,
        edge_corner_strain_std,
        marker="s",
        capsize=5.0,
        capthick=1.5,
        label="edges/corners strain avg")

    axes3.plot(
        scan_axis,
        edge_corner_strain_std,
        label="edge/corner strain std",
        marker="o")

    # for i, scan in enumerate([182, 183, 184, 185]):
    #     axes3[0].text(scan, globals["disp"][i+4], data["S{}".format(scan)]["potential"])

    # axes3[0].tick_params(axis="x", direction="out", pad=0) # pad=-15
    axes3.tick_params(axis="x", direction="out", pad=0) # pad=-15
    # axes3[1].set_xticks([182], minor=True)
    # axes3[0].set_xticklabels(xticks_labels, fontsize=9)
    axes3.set_xticklabels(xticks_labels, fontsize=12)
    axes3.locator_params(tight=True, nbins=5)
    axes3.tick_params(axis='x', which='minor', direction='out', length=30)

    # axes3[0].set_ylabel("Displacement ($\AA$)", fontsize=16)
    # axes3[0].legend(loc="upper center", bbox_to_anchor=(0.5, 0.92), fontsize=8,
    #     ncol=len(axes3[0].lines))
    axes3.set_xlabel("Course of experiment", fontsize=14)
    axes3.set_ylabel(
        r"$\overline{\epsilon_{002}}(\%)$ and $\epsilon_{002}$ std",# and \epsilon_{002}} std$",
        fontsize=14)
    axes3.legend()
    fig3.suptitle("Facet, edge corner strain evolutions",
                  fontsize=18)

    plt.show()
