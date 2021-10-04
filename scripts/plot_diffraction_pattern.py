import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
from scipy.ndimage.measurements import center_of_mass

sys.path.append('/data/id01/inhouse/clatlan/pythonies/cdiutils')

from cdiutils.plot.plot3D import plot_3D_object
from cdiutils.utils import find_hull


def compute_distance_from_com(data, com):
    nonzero_coordinates = np.nonzero(data)
    distance_matrix = np.zeros(shape=data.shape)

    for x, y, z in zip(nonzero_coordinates[0],
                       nonzero_coordinates[1],
                       nonzero_coordinates[2]):
        distance = np.sqrt((x-com[0])**2 +(y-com[1])**2 +(z-com[2])**2)
        distance_matrix[x, y, z] = distance

    return distance_matrix


if __name__ == '__main__':
    import argparse
	
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=False, type=str,
                    help="file to read")
    args = vars(ap.parse_args())

    if args["file"] is None:
        file = "/data/id01/inhouse/clatlan/experiments/ihhc3644/analysis/" \
	       "results/S322/pynxraw/S322_pynx_norm_60_288_294_1_1_1.npz"
    else:
        file = args["file"]
    
    data = np.load(file)["data"]

    isosurface_threshold = 0.001

    diffraction_pattern = np.where(data > isosurface_threshold * np.max(data),
    							   data, 0)
    support = np.where(data > isosurface_threshold * np.max(data), 1, 0)

    com = center_of_mass(diffraction_pattern)

    distance_matrix = compute_distance_from_com(diffraction_pattern, com)

    # fig1 = plot_3D_object(np.log10(np.log10(data)), support, cmap="jet",
    #                      marker="o", show=False, vmax=0.8)

    plt.rcParams.update({
    "lines.markersize": 80})
    fig2 = plot_3D_object(distance_matrix, support, cmap="CMRmap",
                         marker="o", show=False)

    plt.grid(False)
    # plt.axis('off')

    # Get rid of the ticks
    # for fig in (fig1, fig2):
    #     fig.axes[0].set_xticks([])
    #     fig.axes[0].set_yticks([])
    #     fig.axes[0].set_zticks([])
    fig2.axes[0].set_xticks([])
    fig2.axes[0].set_yticks([])
    fig2.axes[0].set_zticks([])

    plt.show()


    # from matplotlib.colors import ListedColormap, LinearSegmentedColormap

    # cmap = cm.get_cmap('seismic')

    # x = np.arange(0,1+0.03125,0.03125)

    # for e in x:
        # print(e,',')
        # print(cmap(e)[0],',')
        # print(cmap(e)[1],',')
        # print(cmap(e)[2],',')
