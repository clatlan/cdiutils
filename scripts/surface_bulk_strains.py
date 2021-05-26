import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from mayavi import mlab
from scipy.ndimage import convolve

sys.path.append('/data/id01/inhouse/clatlan/pythonies/cdiutils')
from cdiutils.utils import find_hull
from cdiutils.plot.plot import plot_3D_object

def find_surface_planes(data):
    kernel = np.ones(shape=(3, 3, 3))
    convolved_data = convolve(data, kernel, mode="constant", cval=0.0)
    plane_value, count = np.unique(convolved_data, return_counts=True)
    return plane_value, count



if __name__ == '__main__':
    import argparse

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-f", "--files", required=True, type=str, nargs="+",
                    help="files to read")
    args = vars(ap.parse_args())

    support_threshold = 0.7

    for file in args["files"]:
        data = np.load(file)
        modulus = data["amp"] / np.max(data["amp"])
        support = np.where(modulus > 0.7, 1, 0)
        hull = find_hull(support, threshold=15)
        # plot_3D_object(modulus, hull)
        print("Number of points plotted using hull: {}".format(
            hull.nonzero()[0].shape[0]))

        plane_value, count = find_surface_planes(support)
        fig, ax = plt.subplots()
        ax.plot(plane_value[1:], count[1:])

        # nonzero = hull.nonzero()
        # X, Y, Z = np.meshgrid(nonzero[0], nonzero[1], nonzero[2])
        # new_hull = hull[nonzero]
        # print(nonzero[0].shape, Y.shape, Z.shape, new_hull.shape, hull.shape)
        obj = mlab.contour3d(hull, contours=8, transparent=True)

        # surf = ax.plot_surface(X, Y, Z, cmap=mpl.cm.coolwarm,
        #                linewidth=0, antialiased=False)

        # fft = np.fft.fftn(modulus.astype(complex))
        # intensity = abs(np.fft.fftshift(fft))**2
        # itensity_support = np.where(np.log10(intensity) > 2, 1, 0)
        # plot_3D_object(np.log10(intensity), itensity_support)
        mlab.show()
        # plt.show()
