#!/data/id01/inhouse/clatlan/.envs/cdiutils/bin/python

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import sys
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("/data/id01/inhouse/clatlan/pythonies/cdiutils")
from cdiutils.load.load_data import get_data_from_vtk
from cdiutils.facetanalysis.get_facet_data import get_facet_data
from cdiutils.facetanalysis.facet_utils import get_rotation_matrix, \
    get_miller_indices, planes_111_110_100


def plot_facets_from_vtk(vtk, points, cells):

    data = get_facet_data(vtk_data)
    facet_ids = data["disp"].keys().tolist()

    fig1 = plt.figure()
    ax = fig1.add_subplot(projection="3d")
    x = np.concatenate([data["point_coord"][facet][..., 0] \
        for facet in facet_ids])
    y = np.concatenate([data["point_coord"][facet][..., 1] \
        for facet in facet_ids])
    z = np.concatenate([data["point_coord"][facet][..., 2] \
        for facet in facet_ids])
    c = np.concatenate([np.repeat(data["disp_avg"][facet],
                                  data["point_coord"][facet].shape[0]) \
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


    font = {'size'   : 12}
    mpl.rc('font', **font)

    plt.rcParams.update({
    "text.usetex": True,
    "axes.prop_cycle": mpl.cycler(
        color=mpl.cm.gist_ncar(np.linspace(0, 1, 9)))})

    dpi=120

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-f", "--files", required=True, type=str, nargs="+",
                    help="files to read")
    args = vars(ap.parse_args())

    # get rotation matrix
    u0 = [0, 0, 1]
    v0 = [1, 0, 0]
    u1 = [0, 1, 0]
    v1 = [0, 0, 1]
    rotation_matrix = get_rotation_matrix(u0, v0, u1, v1)
    data = {}

    for file in args["files"]:
        scan = os.path.splitext(os.path.basename(file))[0]
        scan_digit = int(scan[1:])
        print("[INFO] working on scan {}".format(scan))
        # if scan != "S178":
        #     continue
        vtk_data = get_data_from_vtk(file)

        scan_data = get_facet_data(vtk_data, rotation_matrix)
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
    data["S178"]["potential"] = 0.6 + voltage_shift
    data["S179"]["potential"] = 0.6 + voltage_shift
    data["S180"]["potential"] = 0.6 + voltage_shift
    data["S181"]["potential"] = 0.6 + voltage_shift
    data["S182"]["potential"] = 0 + voltage_shift
    data["S183"]["potential"] = 0.1 + voltage_shift
    data["S184"]["potential"] = 0.2 + voltage_shift
    data["S185"]["potential"] = 0.3 + voltage_shift

    potential_axis = [i + voltage_shift for i in [0, 0.1, 0.2, 0.3]]
    scan_axis = [data[key]["scan_digit"] for key in data.keys()]
    globals = {key: np.empty(shape=(len(scan_axis),)) for key in \
        ["disp", "disp_std", "strain", "strain_std"]}

    xticks_labels = [""] + [str(i) for i in scan_axis]
    xticks_labels[1] = "{}\n\n{}".format(178, "No electrolyte")
    xticks_labels[2] = "{}\n\n{}".format(179, "No electrolyte")
    xticks_labels[3] = "{}\n\n{}".format(180, "OCP")
    xticks_labels[4] = "{}\n\n{}".format(181, "OCP")
    xticks_labels[5:] = ["{}\n\n{} V\n/ RHE".format(i,
        data["S{}".format(i)]["potential"]) for i in scan_axis[4:]]

    for plane_family in [planes111, planes110, planes100]:
        print("[INFO] Working on the following planes: {}".format(plane_family))

        fig, axes = plt.subplots(2, 1, figsize=(9, 8), dpi=dpi)
        fig2, axes2 = plt.subplots(2, 1, figsize=(9, 8), dpi=dpi)

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

            # plot displacement average per facet
            line, = axes[0].plot(xaxis.fillna(method="ffill"),
						 	 	 df["disp"].fillna(method="ffill"), ls="--")
            axes[0].plot(xaxis, df["disp"], color=line.get_color(),
					 	 marker="o", label=label)

            # plot strain average per facet
            line, = axes[1].plot(xaxis.fillna(method="ffill"),
						 	     df["strain"].fillna(method="ffill"), ls="--")
            axes[1].plot(xaxis, df["strain"], color=line.get_color(),
					 	 marker="o", label=label)

            # plot displacement std per facet
            line, = axes2[0].plot(xaxis.fillna(method="ffill"),
						 	 	  df["disp_std"].fillna(method="ffill"),
                                  ls="--")
            axes2[0].plot(xaxis, df["disp_std"],
                          color=line.get_color(), marker="o", label=label)

            # plot strain std per facet
            line, = axes2[1].plot(xaxis.fillna(method="ffill"),
						 	      df["strain_std"].fillna(method="ffill"),
                                  ls="--")
            axes2[1].plot(xaxis, df["strain_std"],
                          color=line.get_color(), marker="o", label=label)

        axes[0].set_ylabel("Displacement average ($\AA$)", fontsize=16)
        axes[1].set_xlabel("Potentials (V) / RHE", fontsize=16)
        axes[1].set_xlabel("Course of experiment (scan \#)", fontsize=16)
        axes[1].set_ylabel("Strain average (\%)", fontsize=16)
        axes[0].tick_params(axis="x", direction="in", pad=-15)
        axes[0].set_xticklabels(xticks_labels, fontsize=9)
        axes[1].set_xticklabels(xticks_labels, fontsize=9)
        fig.legend(handles=axes[0].get_legend_handles_labels()[0],
                  loc="upper center", bbox_to_anchor=(0.5, 0.92), fontsize=8,
                  ncol=len(axes[0].lines))
        fig.suptitle("Displacement and strain averages", fontsize=20)

        axes2[0].set_ylabel("Displacement standard deviation ($\AA$)",
                            fontsize=16)
        axes2[1].set_xlabel("Potentials (V) / RHE", fontsize=16)
        axes2[1].set_xlabel("Course of experiment (scan \#)", fontsize=16)
        axes2[1].set_ylabel("Strain standard deviation (\%)", fontsize=16)
        axes2[0].tick_params(axis="x", direction="in", pad=-15)
        axes2[0].set_xticklabels(xticks_labels, fontsize=9)
        axes2[1].set_xticklabels(xticks_labels, fontsize=9)
        fig2.legend(handles=axes2[0].get_legend_handles_labels()[0],
                    loc="upper center", bbox_to_anchor=(0.5, 0.92), fontsize=8,
                    ncol=len(axes[0].lines))
        fig2.suptitle("Displacement and strain standard deviations",
                      fontsize=20)


    for key in globals.keys():
        glb = globals[key]
        glb = glb.reshape([glb.shape[0]//8, 8])
        glb = np.delete(glb, 0, 0)
        globals[key] = np.mean(glb, axis=0)

    # plt.close("all")
    fig3, axes3 = plt.subplots(2, 1, figsize=(9, 8), dpi=dpi, sharex=False)

    # labels = {"disp": "disp avg", "strain": "strain avg",
    #           "disp_std": "disp std", "strain_std": "strain std"}
    # for key in ["disp_std"]:
    #     axes3[0].plot(scan_axis, globals[key], label=labels[key],
    #                   marker="o")
    # for key in ["strain_std"]:
    #     axes3[1].plot(scan_axis, globals[key], label=labels[key],
    #                   marker="o")

    axes3[0].errorbar(scan_axis, globals["disp"], globals["disp_std"],
                      marker="^", capsize=5.0, capthick=1.5, label="average")
    axes3[1].errorbar(scan_axis, globals["strain"], globals["strain_std"],
                      marker="^", capsize=5.0, capthick=1.5, label="strain")
    axes3[0].plot(scan_axis, globals["disp_std"], label="standard deviation",
                  marker="o")
    axes3[1].plot(scan_axis, globals["strain_std"], label="standard deviation",
                  marker="o")

    # for i, scan in enumerate([182, 183, 184, 185]):
    #     axes3[0].text(scan, globals["disp"][i+4], data["S{}".format(scan)]["potential"])


    axes3[0].tick_params(axis="x", direction="in", pad=-15)
    axes3[1].tick_params(axis="x", direction="in", pad=-15)
    # axes3[1].set_xticks([182], minor=True)
    axes3[0].set_xticklabels(xticks_labels, fontsize=9)
    axes3[1].set_xticklabels(xticks_labels, fontsize=9)
    axes3[1].tick_params(axis='x', which='minor', direction='out', length=30)

    axes3[0].set_ylabel("Displacement ($\AA$)", fontsize=18)
    axes3[0].legend(loc="upper center", bbox_to_anchor=(0.5, 0.92), fontsize=8,
        ncol=len(axes3[0].lines))
    axes3[1].set_xlabel("Potentials (V) / RHE", fontsize=18)
    axes3[1].set_xlabel("Course of experiment (scan \#)", fontsize=18)
    axes3[1].set_ylabel("Strain (\%)", fontsize=18)
    # axes3[1].legend()
    fig3.suptitle("Surface displacement and strain evolutions",
                  fontsize=20)

    plt.show()
