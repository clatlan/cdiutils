import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from mayavi import mlab
from scipy.ndimage import convolve
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2

from bcdi.facet_recognition.facet_utils import taubin_smooth

sys.path.append('/data/id01/inhouse/clatlan/pythonies/cdiutils')
from cdiutils.utils import find_hull
from cdiutils.plot.plot import plot_3D_object

def find_surface_planes(data):
    kernel = np.ones(shape=(3, 3, 3))
    convolved_data = convolve(data, kernel, mode="constant", cval=0.0)
    plane_value, count = np.unique(convolved_data, return_counts=True)
    return plane_value, count

def generate_2D_slice(axis, shape):
    if axis == 0:
        def f(indx):
            return (indx, slice(0, shape[1]), slice(0, shape[2]))
    elif axis == 1:
        def f(indx):
            return (slice(0, shape[0]), indx, slice(0, shape[2]))
    elif axis == 2:
        def f(indx):
            return (slice(0, shape[0]), slice(0, shape[1]), indx)
    else:
        f = None
    return f

def find_edges_corners(data):
    shape = data.shape
    data = data.astype("uint8")
    corner_coordinates = []
    corners_edges = np.zeros(shape=shape)
    corners = np.zeros(shape=shape)
    edges = np.zeros(shape=shape)

    for axis in range(3):
        s = generate_2D_slice(axis, shape)
        for k in range(shape[1]):
            data_slice = data[s(k)]
            corners_edges_slice = cv2.cornerHarris(data_slice, 5, 19, 0.01)
            # slice_corners = cv2.dilate(slice_corners, None)
            # if not np.all((slice_corners == 0)):
            #     slice_corners = (slice_corners - np.min(slice_corners)) / np.ptp(slice_corners)

            corner_slice = np.where(
                corners_edges_slice > 0.4 * np.max(corners_edges_slice),
                1,
                0)
            edge_slice = np.where(
                corners_edges_slice < 0,
                1,
                0)

            if not np.all((corners_edges_slice == 0)):
                corner_slice = (corner_slice - np.min(corner_slice)) \
                    / np.ptp(corner_slice)
                edge_slice = (edge_slice - np.min(edge_slice)) \
                    / np.ptp(edge_slice)

            corners_edges[s(k)] += corners_edges_slice
            corners[s(k)] += corner_slice
            edges[s(k)] += edge_slice


    # corners = np.where(corners_edges>0.4*np.max(corners_edges), corners_edges, 0)
    # edges = np.where(corners_edges < 0, corners_edges, 0)

    s = (slice(0, shape[0]), shape[1] // 2, slice(0, shape[2]))
    data_slice = data[s]
    corner_edge_slice = corners_edges[s]
    corner_slice = corners[s]
    edges_slice = edges[s]
    fig, axes = plt.subplots(1, 4)
    # cv2.imshow('dst',middle_slice)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()
    for plot, ax in zip([data_slice, corner_edge_slice,
                         corner_slice, edges_slice], axes.ravel()):
        im = ax.imshow(plot)
        fig.colorbar(im, ax=ax)
    plot_3D_object(data)
    plot_3D_object(corners)
    plt.show()



if __name__ == '__main__':
    import argparse

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-f", "--files", required=True, type=str, nargs="+",
                    help="files to read")
    args = vars(ap.parse_args())

    support_threshold = 0.65

    for file in args["files"]:
        data = np.load(file)
        modulus = data["amp"] / np.max(data["amp"])
        support = np.where(modulus > support_threshold, 1, 0)
        hull = find_hull(support, threshold=25)
        # # hull = find_hull(hull, threshold=15)
        # plot_3D_object(modulus, hull)
        print("Number of points plotted using hull: {}".format(
            hull.nonzero()[0].shape[0]))

        # plane_value, count = find_surface_planes(modulus)
        # fig, ax = plt.subplots()
        # ax.plot(plane_value[1:], count[1:])

        # support_plot = mlab.contour3d(support, contours=8, transparent=True)
        # hull_plot = mlab.contour3d(hull, contours=8, transparent=True)



        # fft = np.fft.fftn(modulus.astype(complex))
        # intensity = abs(np.fft.fftshift(fft))**2
        # itensity_support = np.where(np.log10(intensity) > 2, 1, 0)
        # plot_3D_object(np.log10(intensity), itensity_support)
        # mlab.show()
        # plt.show()


        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(211, projection='3d')
        ax2 = fig.add_subplot(212, projection='3d')
        verts, faces, normals, values = measure.marching_cubes(
            hull, support_threshold)

        mesh1 = Poly3DCollection(verts[faces])
        mesh1.set_edgecolor('k')
        ax1.add_collection3d(mesh1)
        ax1.set_xlim(50, 100)  # a = 6 (times two for 2nd ellipsoid)
        ax1.set_ylim(50, 100)  # b = 10
        ax1.set_zlim(50, 100)

        newverts, normals, areas, intensity, newfaces, _ = taubin_smooth(faces, verts,
                                                                   iterations=5)
        mesh2 = Poly3DCollection(newverts[newfaces])
        mesh2.set_edgecolor('r')
        ax2.add_collection3d(mesh2)


        ax2.set_xlim(50, 100)  # a = 6 (times two for 2nd ellipsoid)
        ax2.set_ylim(50, 100)  # b = 10
        ax2.set_zlim(50, 100)

        print(verts.shape, faces.shape, normals.shape, values.shape)
        plt.show()

        # find_edges_corners(support)
