from typing import Any, Dict
import warnings

import numpy as np

AUTHORIZED_KEYS = {
    "cdiutils": {
        "metadata": "REQUIRED",
        "preprocessing_output_shape": "REQUIRED",
        "energy": "REQUIRED",
        "hkl": "REQUIRED",
        "det_reference_voxel_method": "REQUIRED",
        "light_loading": False,
        "det_reference_voxel": None,
        "binning_along_axis0": None,
        "q_lab_reference": None,
        "q_lab_max": None,
        "q_lab_com": None,
        "det_calib_parameters": "REQUIRED",
        "voxel_size": None,
        "apodize": True,
        "flip": False,
        "isosurface": None,
        "usetex": False,
        "show": False,
        "verbose": True,
        "debug": True,
        "show_phasing_results": False,
        "unwrap_before_orthogonalization": False
    },
    "pynx": {
        "data": None,
        "mask": None,
        "data2cxi": True,
        "auto_center_resize": False,
        "support_type": None,
        "support_size": None,
        "support": "auto",
        "support_threshold": "0.25, 0.40",
        "support_threshold_method": "rms",
        "support_only_shrink": False,
        "support_update_period": 20,
        "support_smooth_width_begin": 2,
        "support_smooth_width_end": 1,
        "support_post_expand": "1,-2,1",
        "psf": "pseudo-voigt,0.5,0.1,10",
        "nb_raar": 1000,
        "nb_hio": 150,
        "nb_er": 150,
        "nb_ml": 10,
        "nb_run": 20,
        "nb_run_keep": 5,
        "zero_mask": False,
        "crop_output": 0,
        "positivity": False,
        "beta": 0.9,
        "detwin": False,
        "rebin": "1, 1, 1",
        "detector_distance": "REQUIRED",
        "pixel_size_detector": "REQUIRED",
        "wavelength": "REQUIRED",
        "verbose": 100,
        "output_format": "cxi",
        "live_plot": False,
        "save_plot": True,
        "mpi": "run"
    }
}


def convert_np_arrays(dictionary) -> None:
    """
    Recursively converts np.ndarray values in a dictionary to tuple or
    a single value.

    Args:
        dictionary (Dict[str, Any]): The dictionary to be processed.

    Returns:
        None: This function modifies the dictionary in-place.

    """
    for key, value in dictionary.items():
        if isinstance(value, np.ndarray):
            if value.size == 1:
                dictionary[key] = value[0]
            else:
                if value.dtype == int:
                    dictionary[key] = tuple(value.astype(int))

        elif isinstance(value, list):
            for i, v in enumerate(value):
                if isinstance(v, int):
                    dictionary[key][i] = int(v)

        elif isinstance(value, (tuple, list)):
            if isinstance(value[0], (int, int, np.int64, np.int32)):
                dictionary[key] = tuple(int(v) for v in value)

        elif isinstance(value, dict):
            convert_np_arrays(value)

def check_parameters(parameters: dict) -> None:
    """
    Check parameters given by user, handle when parameters are
    required or not provided.
    """
    for e in ["cdiutils", "pynx"]:
        for name, value in AUTHORIZED_KEYS[e].items():
            if not name in parameters[e].keys() or parameters[e][name] is None:
                if value == "REQUIRED":
                    raise ValueError(f"Arguement '{name}' is required")
                else:
                    parameters[e].update({name: value})
        for name in parameters[e].keys():
            if not isparameter(name):
                warnings.warn(
                    f"Parameter '{name}' is unknown, will not be used")
    for name in parameters.keys():
        if not isparameter(name):
            warnings.warn(
                f"Parameter '{name}' is unknown, will not be used."
            )


def isparameter(string: str):
    """Return whether or not the given string is in AUTHORIZED_KEYS."""
    return (
        string in list(AUTHORIZED_KEYS["cdiutils"].keys())
        +  list(AUTHORIZED_KEYS["pynx"].keys())
        + ["cdiutils", "pynx"]
    )

def get_parameters_from_notebook_variables(
            dir_list: list,
            globals_dict: dict
) -> dict:
    """
    Return a dictionary of parameters whose keys are authorized by the 
    AUTHORIZED_KEYS list.
    """
    parameters = {
        "cdiutils": {},
        "pynx": {}
    }
    for e in dir_list:
        if e in AUTHORIZED_KEYS["cdiutils"]:
            parameters["cdiutils"][e] = globals_dict[e]
        elif e in AUTHORIZED_KEYS["pynx"]:
            parameters["pynx"][e] = globals_dict[e]
    return parameters
