import h5py
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import json
from matplotlib.colors import LinearSegmentedColormap
import silx.io
import xrayutilities as xu

from cdiutils.utils import crop_at_center, make_support


def load_specfile(path: str):
    """Load the specfile from the given path"""
    #  return silx.io.specfile.SpecFile(path)
    with silx.io.open(path) as specfile:
        data = specfile
    return data


def get_cmap_dict_from_json(file_path):
    """Make a matplotlib cmap from json file."""

    f = open(file_path, encoding="utf8")
    my_cmap = LinearSegmentedColormap("my_cmap", json.load(f))
    f.close()

    return my_cmap


def load_data_from_cxi(file, *items):

    """
    Get data from .cxi file.

    :param file_path: file path. The string path to the file.
    :param *items: items needed. The items needed that the file must
    contain.
    :returns: data_dic. A dictionary whose keys are the parsed *items,
    values are the data retrieved from the file.
    """

    data_dic = {}

    try:
        data = h5py.File(file, "r")

        if "support" in items:
            data_dic["support"] = data["entry_1/image_1/support"][...]

        if "reconstructed_data" in items:
            data_dic["reconstructed_data"] = data["entry_1/data_1/data"][...]

        if "llkf" in items:
            data_dic["llkf"] = float(data["entry_1/image_1/process_1/results/"
                                          "free_llk_poisson"][...])

        if "llk" in items:
            data_dic["llk"] = float(data["entry_1/image_1/process_1/results/"
                                         "llk_poisson"][...])

        data.close()
        return data_dic

    except Exception as exc:
        print("[ERROR] An error occured while opening the file:", exc,
              "\n", exc.__str__())
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
        file_path,
        strain_in_percent=False,
        normalised_amp=False):
    with np.load(file_path) as data:
        amp = data["amp"]
        try:
            phase = data["phase"]
        except KeyError:
            try:
                phase = data["displacement"]
            except KeyError:
                phase = data["disp"]
        strain = data["strain"] * (100 if strain_in_percent else 1)
    if normalised_amp:
        amp = (amp - np.min(amp)) / np.ptp(amp)

    return amp, phase, strain


def load_raw_scan(
        specfile,
        edf_file_template: str,
        scan: int,
        hxrd,
        nav=[1, 1],
        roi=[0, 516, 0, 516],
        start_end_frames=None
):

    if start_end_frames:
        print(
            "[INFO] start_end_frames parameter provided, will consider only "
            f"the frames between {start_end_frames[0]} and "
            f"{start_end_frames[1]}"
        )
        frames_nb = start_end_frames[1] - start_end_frames[0]
        frame_ids = specfile[f"{scan}.1/measurement/mpx4inr"][...][
            start_end_frames[0]: start_end_frames[1]
        ]
    else:
        frame_ids = specfile[f"{scan}.1/measurement/mpx4inr"][...]
        frames_nb = len(frame_ids)
        start_end_frames = [0, frames_nb]
    
    data = np.empty((frames_nb, roi[1]-roi[0], roi[3]-roi[2]))

    positioners = specfile[f"{scan}.1/instrument/positioners"]
    try:
        eta = positioners["eta"][start_end_frames[0]: start_end_frames[1]]
    except:
        eta = positioners["eta"][...]
    try:
        delta = positioners["del"][start_end_frames[0]: start_end_frames[1]]
    except:
        delta = positioners["del"][...]
    try:
        phi = positioners["phi"][start_end_frames[0]: start_end_frames[1]]
    except:
        phi = positioners["phi"][...]
    try:
        nu = positioners["nu"][start_end_frames[0]: start_end_frames[1]]
    except:
        nu = positioners["nu"][...]
    try:
        mu = positioners["mu"][start_end_frames[0]: start_end_frames[1]]
    except:
        mu = positioners["mu"][...]
    phi += mu
    
    for i, frame_id in enumerate(frame_ids):
        edf_data = xu.io.EDFFile(
            edf_file_template.format(id=int(frame_id))
        ).data
        ccdraw = xu.blockAverage2D(edf_data, nav[0], nav[1], roi=roi)
        data[i, ...] = ccdraw

    detector_to_Q_space = hxrd.Ang2Q.area(eta, phi, nu, delta, delta=(0, 0, 0, 0))

    nx, ny, nz = data.shape
    gridder = xu.Gridder3D(nx, ny, nz)
    gridder(
        detector_to_Q_space[0],
        detector_to_Q_space[1],
        detector_to_Q_space[2],
        data
    )
    qx, qy, qz = gridder.xaxis, gridder.yaxis, gridder.zaxis
    intensity = gridder.data

    return intensity, (qx, qy, qz), detector_to_Q_space, data


def load_post_bcdi_data(
        file_path: str,
        isosurface: float,
        shape: tuple=(100, 100, 100),
        reference_voxel: tuple=None,
        qnorm: float=None,
        hkl: tuple=(1, 1, 1)
) -> dict:

    """
    Load data from the post bcdi processing.

    :param file_path: the file path of the file to load (str)
    :param isosurface: the isosurface threshold to specify what is part
    the reconstructed object or not (float).
    :param shape: the shape of the voulme to consider. The data will be
    cropped so the center of the orignal data remains the center of the 
    cropped data (tuple). Default: (100, 100, 100])
    :param reference_voxel: the voxel of reference to define the origin
    of the phase (tuple). Default: None
    :param qnorm: The norma of the q vector measured. This allows to 
    compute the dpsacing and lattice parameter maps (float). 
    Default: None
    :param hkl: the Bragg peak measured (tuple). Default: (1, 1, 1)

    :return dict: a dictionary containing all the necessary data.
    """

    amp, phase, strain = load_amp_phase_strain(
        file_path,
        strain_in_percent=True,
        normalised_amp=True
    )

    amp = crop_at_center(amp, final_shape=shape)
    phase = crop_at_center(phase, final_shape=shape)
    strain = crop_at_center(strain, final_shape=shape)

    support = make_support(amp, isosurface=isosurface, nan_values=False)
    # nan_support = zero_to_nan(support)
    # phase *= nan_support
    # strain *= nan_support

    if not qnorm:
        print(
            "[INFO] qnorm not provided, only amp, support, phase and "
            "local_strain will be returned"
        )
        return {
            "amp": amp,
            "support": support,
            "phase": phase, 
            "local_strain": strain
        }
    else:

        # define the origin of the pase
        if reference_voxel is None:
            phase_reference = np.nanmean(phase)
        else:
            phase_reference = phase[reference_voxel]
        centred_phase = phase - phase_reference

        # Not the most efficient but emphasizes the clarity
        displacement = centred_phase / qnorm
        d_bragg =  (2*np.pi / qnorm)

        dspacing = d_bragg * (1 + strain/100)
        lattice_constant = (
            np.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2)
            * dspacing
        )

        return {
            "amplitude": amp,
            "support": support,
            "phase": phase, 
            "local_strain": strain,
            "centred_phase": centred_phase,
            "displacement": displacement,
            "dspacing": dspacing,
            "lattice_constant": lattice_constant,
        }

def get_data_from_npyz(file_path, *keys):
    output = {}
    with np.load(file_path) as data:
        for key in keys:
            output[key] = data[key]
    return output