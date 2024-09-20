import warnings

import numpy as np

from cdiutils.utils import energy_to_wavelength

AUTHORIZED_KEYS = {
    # Formerly the "metadata"
    "beamline_setup": "REQUIRED",
    "scan": "REQUIRED",
    "experiment_file_path": "REQUIRED",
    "dump_dir": "REQUIRED",
    "sample_name": None,
    "experiment_data_dir_path": None,
    "detector_data_path": None,
    "edf_file_template": None,
    "detector_name": None,
    "flat_field": None,
    "alien_mask": None,

    "preprocessing_output_shape": "REQUIRED",
    "energy": None,
    "hkl": "REQUIRED",
    "det_reference_voxel_method": "REQUIRED",
    "light_loading": False,
    "det_reference_voxel": None,
    "binning_along_axis0": None,
    "q_lab_reference": None,
    "q_lab_max": None,
    "q_lab_com": None,
    "dspacing_reference": None,
    "dspacing_max": None,
    "dspacing_com": None,
    "lattice_parameter_reference": None,
    "lattice_parameter_max": None,
    "lattice_parameter_com": None,
    "det_calib_params": None,
    "voxel_size": None,
    "apodize": "blackman",
    "flip": False,
    "isosurface": None,
    "show": False,
    "verbose": True,
    "debug": True,
    "binning_factors": (1, 1, 1),
    "handle_defects": False,
    "orthogonalize_before_phasing": False,
    "orientation_convention": "cxi",
    "method_det_support": None,
    "raw_process": True,
    "support_path": None,
    "remove_edges": True,
    "nb_facets": None,
    "order_of_derivative": None,
    "derivative_threshold": None,
    "amplitude_threshold": None,
    "top_facet_reference_index": [1, 1, 1],
    "authorized_index": 1,
    "nb_nghbs_min": 0,
    "index_to_display": None,
    "display_f_e_c": 'facet',
    "size": 10,
    "pynx": {
        "data": None,
        "mask": None,
        "data2cxi": True,
        "auto_center_resize": False,
        "support": "auto",
        "support_size": None,
        "support_threshold": "0.15, 0.40",
        "support_threshold_method": "rms",
        "support_only_shrink": False,
        "support_update_period": 20,
        "support_smooth_width_begin": 2,
        "support_smooth_width_end": 1,
        "support_post_expand": None,  # (-1, 1)
        "psf": "pseudo-voigt,0.5,0.1,10",
        "nb_raar": 500,
        "nb_hio": 300,
        "nb_er": 200,
        "nb_ml": 0,
        "nb_run": 20,
        "nb_run_keep": 5,
        "zero_mask": False,
        "crop_output": 0,
        "positivity": False,
        "beta": 0.9,
        "detwin": True,
        "rebin": "1, 1, 1",
        "detector_distance": None,
        "pixel_size_detector": None,
        "wavelength": None,
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


def check_params(params: dict) -> None:
    """
    Check parameters given by user, handle when parameters are
    required or not provided.
    """
    for name, value in AUTHORIZED_KEYS.items():
        if name not in params or params[name] is None:
            if value == "REQUIRED":
                raise ValueError(f"Parameter '{name}' is required.")
            params.update({name: value})
    for name, value in AUTHORIZED_KEYS["pynx"].items():
        if name not in params["pynx"] or params["pynx"][name] is None:
            if value == "REQUIRED":
                raise ValueError(f"Parameter '{name}' is required.")
            params["pynx"].update({name: value})
    for name in params["pynx"]:
        if not isparameter(name):
            warnings.warn(
                f"Parameter '{name}' is unknown, will not be used")
    for name in params.keys():
        if not isparameter(name):
            warnings.warn(
                f"Parameter '{name}' is unknown, will not be used."
            )


def fill_pynx_params(params: dict) -> None:
    params["pynx"]["pixel_size_detector"] = (
        params["det_calib_params"]["pwidth1"]
    )
    params["pynx"]["detector_distance"] = (
        params["det_calib_params"]["distance"]
    )
    params["pynx"]["wavelength"] = energy_to_wavelength(
        params["energy"]
    )


def isparameter(string: str):
    """Return whether or not the given string is in AUTHORIZED_KEYS."""
    return (
        string in list(AUTHORIZED_KEYS.keys())
        + list(AUTHORIZED_KEYS["pynx"].keys())
        + ["pynx"]
    )


def get_params_from_notebook_variables(
            dir_list: list,
            globals_dict: dict
) -> dict:
    """
    Return a dictionary of parameters whose keys are authorized by the 
    AUTHORIZED_KEYS list.
    """
    params = {
        "pynx": {}
    }
    for e in dir_list:
        if e in AUTHORIZED_KEYS:
            params[e] = globals_dict[e]
        elif e in AUTHORIZED_KEYS["pynx"]:
            params["pynx"][e] = globals_dict[e]

    return params
