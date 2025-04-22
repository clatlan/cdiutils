import numpy as np
import mcubes
import sys
import matplotlib.pyplot as plt

sys.path.append("/data/id01/inhouse/clatlan/pythonies/cdiutils")
from cdiutils.plot.plot import plot_3D_object


def compute_correlation(data):
    data_shape = data[0].shape
    concatenated_data = np.zeros(
        shape=((len(data), )
               + (data_shape[0] * data_shape[1] * data_shape[2],)
               )
            )
    for i, d in enumerate(data):
        concatenated_data[i] = np.ravel(d)

    return np.corrcoef(concatenated_data, rowvar=True)


def find_support_reference(supports, show=False):

    support_correlation = compute_correlation(supports)
    print("Correlation coefficient between data support: \n",
          support_correlation)

    support_sum = np.zeros(supports[0].shape)
    for support in supports:
        support_sum += support

    support_reference = np.where(
        support_sum > (len(supports)//2 + len(supports) % 2),
        1,
        0)

    if show:
        for support in supports:
            plot_3D_object(support, show=False)
        plot_3D_object(support_reference, show=False, title="Reference")
        plt.show()

    return support_reference



    # support = data[0]['bulk']
    # smoothed_support = smoothed_sphere = mcubes.smooth(support)
    # vertices, triangles = mcubes.marching_cubes(smoothed_support, 0)
    # print(smoothed_support.shape, vertices.shape, triangles.shape)
    # # mcubes.export_mesh(vertices, triangles, "/users/atlan/Desktop/smoothed_support.dae", "smoothed_support")
    #
    # plot_3D_object(support)
    # plot_3D_object(smoothed_support, np.where(smoothed_support>0,smoothed_support, 0))


if __name__ == '__main__':
    import argparse

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-f", "--files", required=False, type=str, nargs="+",
                    help="files to read")
    args = vars(ap.parse_args())

    scan_digits = [181, 182, 183, 184, 185]
    # scan_digits=[181, 182]

    if args["files"] is None:
        file_template = "/data/id01/inhouse/clatlan/experiments/ihhc3567"\
                        "/analysis/results/S{}/pynxraw/S{}_amp-disp-strain_"\
                        "0.65_mode_avg3_apodize_blackman_crystal-frame.npz"
        files = [file_template.format(i, i) for i in scan_digits]
    else:
        files = args["files"]
    data = [np.load(f) for f in files]
    supports = [d["bulk"] for d in data]
    support_reference = find_support_reference(supports)
