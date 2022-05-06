import h5py
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import json
from matplotlib.colors import LinearSegmentedColormap
import xrayutilities as xu


def get_cmap_dict_from_json(file_path):
    """Make a matplotlib cmap from json file."""

    f = open(file_path)
    my_cmap = LinearSegmentedColormap("my_cmap", json.load(f))
    f.close()

    return my_cmap


def get_data_from_cxi(file, *items):

    """
    Get data from .cxi file.

    :param file_path: file path. The string path to the file.
    :param *items: items needed. The items needed that the file must
    contain.
    :returns: data_dic. A dictionary whose keys are the parsed *items,
    values are the data retrieved from the file.
    """

    data_dic = {}
    print("[INFO] Opening file:", file)

    try:
        data = h5py.File(file, "r")

        if "support" in items:
            data_dic["support"] = data["entry_1/image_1/support"][...]

        if "electronic_density" in items:
            data_dic["electronic_density"] = data["entry_1/data_1/data"][...]

        if "llkf" in items:
            data_dic["llkf"] = float(data["entry_1/image_1/process_1/results/"
                                          "free_llk_poisson"][...])

        if "llk" in items:
            data_dic["llk"] = float(data["entry_1/image_1/process_1/results/"
                                         "llk_poisson"][...])

        data.close()
        return data_dic

    except Exception as e:
        print("[ERROR] An error occured while opening the file:", f,
              "\n", e.__str__())
        return None


def load_vtk(file):
    """Get raw data from .vtk file."""

    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(file)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllTensorsOn()
    reader.Update()

    return reader.GetOutput()


def load_amp_phase_strain(
        file,
        strain_in_percent=False,
        normalised_amp=False):
    data = np.load(file, allow_pickle=False)
    amp = data["amp"]
    phase = data["phase"]
    strain = data["strain"] * (100 if strain_in_percent else 1)
    if normalised_amp:
        amp = (amp - np.min(amp)) / np.ptp(amp)

    return amp, phase, strain


def load_raw_scan(
        specfile,
        edf_file_template: str,
        scan: str,
        hxrd,
        nav=[1, 1],
        roi=[0, 516, 0, 516],
):

    frames_id = specfile[scan + ".1/measurement/mpx4inr"][...]
    frames_nb = len(frames_id)
    data = np.empty((frames_nb, roi[1], roi[3]))

    positioners = specfile[scan + ".1/instrument/positioners"]
    eta = positioners["eta"][...]
    delta = positioners["del"][...]
    phi = positioners["phi"][...]
    nu = positioners["nu"][...]

    for i, frame_id in enumerate(frames_id):
        edf_data = xu.io.EDFFile(
            edf_file_template.format(id=int(frame_id))
        ).data
        ccdraw = xu.blockAverage2D(edf_data, nav[0], nav[1], roi=roi)
        data[i, ...] = ccdraw

    area = hxrd.Ang2Q.area(eta, phi, nu, delta, delta=(0, 0, 0, 0))

    nx, ny, nz = data.shape
    gridder = xu.Gridder3D(nx, ny, nz)
    gridder(area[0], area[1], area[2], data)
    qx, qy, qz = gridder.xaxis, gridder.yaxis, gridder.zaxis
    intensity = gridder.data

    return intensity, (qx, qy, qz)