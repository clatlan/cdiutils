import numpy as np
import sys
from bcdi.graph.graph_utils import save_to_vti

sys.path.append('/data/id01/inhouse/clatlan/pythonies/cdiutils')

from cdiutils.facetanalysis.facet_correlation import find_support_reference
from cdiutils.load.load_data import get_data_from_vtk


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

    # data = [np.load(f) for f in files]
    support_threshold = 0.65
    supports = [np.where(np.load(f)["amp"]
                >= support_threshold * np.load(f)["amp"].max(), 1, 0)
                for f in files]
    # supports = [np.load(f)["bulk"]for f in files]
    # support_reference = find_support_reference(supports)
    support_reference = supports[1]
    modulus_reference = np.load(files[1])["amp"]

    voxel_size = (5, 5, 5)

    for scan, file in zip(scan_digits, files):
        d = np.load(file)
        shape = d["amp"].shape
        # modulus = d["amp"] * support_reference
        modulus = modulus_reference
        displacement = d["displacement"]
        strain = d["strain"]
        save_path = "/data/id01/inhouse/clatlan/experiments/ihhc3567/" \
                    "analysis/results/new_vti/S{}.vti".format(scan)
        save_to_vti(
            save_path,
            voxel_size,
            (support_reference, modulus, displacement, strain),
            ("support", "amp", "disp", "strain"),
            amplitude_threshold=0.01)
