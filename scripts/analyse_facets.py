import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import sys
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('/data/id01/inhouse/clatlan/pythonies/cdiutils')
from cdiutils.load.load_data import get_data_from_vtk
from cdiutils.facetanalysis.get_facet_data import get_facet_data
from cdiutils.facetanalysis.facet_utils import get_rotation_matrix, \
    get_miller_indices, planes_111_110_100


def analyse_facets(vtk, points, cells):
    cell_facet_ids = vtk_to_numpy(cells.GetArray('FacetIds'))
    facet_ids = np.unique(cell_facet_ids)

    facet_data = {key: {} for key in facet_ids}

    disp = {}
    strain = {}
    disp_avg = {}
    strain_avg = {}
    disp_std = {}
	# planes111, planes110, planes100 = planes_111_110_100()
    strain_std = {}
    point_coord = {}

    for facet in facet_ids:
        # get the indices of interest
        indices_oi = np.where(cell_facet_ids == facet)[0]
        point_oi_id = []

        for id in indices_oi:
            cell = vtk.GetCell(id)
            point_oi_id.append(cell.GetPointId(0))
            point_oi_id.append(cell.GetPointId(1))
            point_oi_id.append(cell.GetPointId(2))

        point_oi_id = np.unique(point_oi_id)

        # finally get the the disp and strain of the point of interest
        disp[facet] = vtk_to_numpy(points.GetArray("disp"))[point_oi_id]
        strain[facet] = vtk_to_numpy(points.GetArray("strain"))[point_oi_id]
        point_coord[facet] = np.array([vtk.GetPoint(i) for i in point_oi_id])
        disp_avg[facet] = np.mean(disp[facet])
        strain_avg[facet] = np.mean(strain[facet])
        disp_std[facet] = np.std(disp[facet])
        strain_std[facet] = np.std(strain[facet])

    fig1 = plt.figure()
    ax = fig1.add_subplot(projection="3d")
    x = np.concatenate([point_coord[facet][..., 0] for facet in facet_ids])
    y = np.concatenate([point_coord[facet][..., 1] for facet in facet_ids])
    z = np.concatenate([point_coord[facet][..., 2] for facet in facet_ids])
    c = np.concatenate([np.repeat(disp_avg[facet], point_coord[facet].shape[0]) \
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
    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(
    #     color=mpl.cm.jet(np.linspace(0,1,10)))

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

    data = {}

    # get rotation matrix
    u0 = [0, 0, 1]
    v0 = [1, 0, 0]
    u1 = [0, 1, 0]
    v1 = [0, 0, 1]
    rotation_matrix = get_rotation_matrix(u0, v0, u1, v1)

    for file in args["files"]:
        scan = os.path.splitext(os.path.basename(file))[0]
        print("[INFO] working on scan {}".format(scan))
        # if scan != "S183":
        #     continue
        vtk_data = get_data_from_vtk(file)

        scan_data = {key: value for (key, value) in zip(
                ["disp", "strain", "disp_avg", "strain_avg", "disp_std", \
                 "strain_std", "point_coord", "facet_normals",
                 "miller_indices"],
                get_facet_data(vtk_data, rotation_matrix))}

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
    globals = {key: np.empty(shape=(4,)) for key in \
        ["disp", "disp_std", "strain", "strain_std"]}

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
            for scan, dat in data.items():
                potentials.append(dat["potential"])
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
							   "strain": facet_strain,
							   "strain_std": strain_std,
							   "disp": facet_disp,
							   "disp_std": disp_std})
            if df.isnull().values.any():
                continue

            # sort the entire dataframe by potentials
            df.sort_values(by="potentials", inplace=True)
            df = df[df["potentials"] <=0.3 + voltage_shift]

            # plot displacement average per facet
            line, = axes[0].plot(df["potentials"].fillna(method="ffill"),
						 	 	 df["disp"].fillna(method="ffill"), ls="--")
            axes[0].plot(df["potentials"], df["disp"], color=line.get_color(),
					 	 marker="o", label=label)

            # plot strain average per facet
            line, = axes[1].plot(df["potentials"].fillna(method="ffill"),
						 	     df["strain"].fillna(method="ffill"), ls="--")
            axes[1].plot(df["potentials"], df["strain"], color=line.get_color(),
					 	 marker="o", label=label)

            # plot displacement std per facet
            line, = axes2[0].plot(df["potentials"].fillna(method="ffill"),
						 	 	  df["disp_std"].fillna(method="ffill"),
                                  ls="--")
            axes2[0].plot(df["potentials"], df["disp_std"],
                          color=line.get_color(), marker="o", label=label)

            # plot strain std per facet
            line, = axes2[1].plot(df["potentials"].fillna(method="ffill"),
						 	      df["strain_std"].fillna(method="ffill"),
                                  ls="--")
            axes2[1].plot(df["potentials"], df["strain_std"],
                          color=line.get_color(), marker="o", label=label)
            #
            for key in globals.keys():
                if not df[key].isnull().values.any():
                    globals[key] = np.concatenate(
                        [globals[key], df[key]])

        axes[0].set_ylabel("Displacement average ($\AA$)", fontsize=16)
        axes[1].set_xlabel("Potentials (V) / RHE", fontsize=16)
        axes[1].set_ylabel("Strain average (\%)", fontsize=16)
        fig.legend(handles=axes[0].get_legend_handles_labels()[0],
                  loc="upper center", bbox_to_anchor=(0.5, 0.92), fontsize=8,
                  ncol=len(axes[0].lines))
        fig.suptitle("Displacement and strain averages", fontsize=20)

        axes2[0].set_ylabel("Displacement standard deviation ($\AA$)",
                            fontsize=16)
        axes2[1].set_xlabel("Potentials (V) / RHE", fontsize=16)
        axes2[1].set_ylabel("Strain standard deviation (\%)", fontsize=16)
        fig2.legend(handles=axes2[0].get_legend_handles_labels()[0],
                    loc="upper center", bbox_to_anchor=(0.5, 0.92), fontsize=8,
                    ncol=len(axes[0].lines))
        fig2.suptitle("Displacement and strain standard deviations",
                      fontsize=20)


    for key in globals.keys():
        glb = globals[key]
        glb = glb.reshape([glb.shape[0]//4, 4])
        glb = np.delete(glb, 0, 0)
        globals[key] = np.mean(glb, axis=0)

    fig3, axes3 = plt.subplots(2, 1, figsize=(9, 8), dpi=dpi, sharex=False)

    labels = {"disp": "disp avg", "strain": "strain avg",
              "disp_std": "disp std", "strain_std": "strain std"}
    for key in ["disp_std"]:
        axes3[0].plot(potential_axis, globals[key], label=labels[key],
                      marker="o")
    for key in ["strain_std"]:
        axes3[1].plot(potential_axis, globals[key], label=labels[key],
                      marker="o")

    axes3[0].errorbar(potential_axis, globals["disp"], globals["disp_std"],
                      marker="^", capsize=5.0, capthick=1.5, label="disp avg")
    axes3[1].errorbar(potential_axis, globals["strain"], globals["strain_std"],
                      marker="^", capsize=5.0, capthick=1.5, label="strain avg")
    axes3[0].set_ylabel("Displacement ($\AA$)", fontsize=18)
    axes3[0].legend()
    axes3[1].set_xlabel("Potentials (V) / RHE", fontsize=18)
    axes3[1].set_ylabel("Strain (\%)", fontsize=18)
    axes3[1].legend()
    fig3.suptitle("Surface displacement and strain evolutions",
                  fontsize=20)

    plt.show()
