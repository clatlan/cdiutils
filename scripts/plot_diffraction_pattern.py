import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
from scipy.ndimage.measurements import center_of_mass

sys.path.append('/data/id01/inhouse/clatlan/pythonies/cdiutils')

from cdiutils.plot.plot import plot_3D_object
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

    file = "/users/atlan/Desktop/ihhc3567/S185/pynxraw/" \
           "S185_pynx_norm_64_252_300_1_1_1.npz"
    data = np.load(file)["data"]

    isosurface_threshold = 0.001

    diffraction_pattern = np.where(data > isosurface_threshold * np.max(data), data, 0)
    support = np.where(data > isosurface_threshold * np.max(data), 1, 0)

    com = center_of_mass(diffraction_pattern)

    distance_matrix = compute_distance_from_com(diffraction_pattern, com)

    # fig1 = plot_3D_object(np.log10(np.log10(data)), support, cmap="jet",
    #                      marker="o", show=False, vmax=0.8)

    plt.rcParams.update({
    "lines.markersize": 50})
    fig2 = plot_3D_object(distance_matrix, support, cmap="CMRmap",
                         marker="o", show=False, vmin=-20, vmax=130)

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

    # plt.show()


    from matplotlib.colors import ListedColormap, LinearSegmentedColormap

    cmap = cm.get_cmap('seismic')

    x = np.arange(0,1+0.03125,0.03125)

    for e in x:
        print(e,',')
        print(cmap(e)[0],',')
        print(cmap(e)[1],',')
        print(cmap(e)[2],',')
