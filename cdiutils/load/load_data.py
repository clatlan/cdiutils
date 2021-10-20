import h5py
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import json
from matplotlib.colors import LinearSegmentedColormap


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
    phase = data["displacement"]
    strain = data["strain"] * (100 if strain_in_percent else 1)
    if normalised_amp:
        amp = (amp - np.min(amp)) / np.ptp(amp)

    return amp, phase, strain
