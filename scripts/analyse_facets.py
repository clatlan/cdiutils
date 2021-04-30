import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import sys
import matplotlib.pyplot as plt

sys.path.append('/data/id01/inhouse/clatlan/pythonies/cdiutils')
from cdiutils.load.load_data import get_data_from_vtk
from cdiutils.facetanalysis.get_facet_data import get_facet_data


def analyse_facets(vtk, points, cells):
    cell_facet_ids = vtk_to_numpy(cells.GetArray('FacetIds'))
    facet_ids = np.unique(cell_facet_ids)

    facet_data = {key: {} for key in facet_ids}

    disp = {}
    strain = {}
    disp_avg = {}
    strain_avg = {}
    disp_std = {}
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
    ap = argparse.ArgumentParser()

    ap.add_argument("-f", "--files", required=True, type=str, nargs="+",
                    help="files to read")
    args = vars(ap.parse_args())

    for file in args["files"]:
        scan = os.path.splitext(os.path.basename(file))[0]
        if scan != "S181":
            continue
        vtk_data = get_data_from_vtk(file)
        scan_data = {key: value for (key, value) in zip(
                ["disp", "strain", "disp_avg", "strain_avg", \
                "disp_std", "strain_std", "point_coord", "facet_normals"],
                get_facet_data(vtk_data))}
        # print(scan_data["disp"].keys())
