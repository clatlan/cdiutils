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

    "background_level": None,
    "preprocess_shape": (150, 150),
    "energy": None,
    "hkl": [1, 1, 1],
    "hot_pixel_filter": False,
    "voxel_reference_methods": ["max", "com", "com"],
    "q_lab_ref": None,
    "light_loading": False,
    "det_reference_voxel": None,
    "rocking_angle_binning": None,
    "det_calib_params": None,
    "voxel_size": None,
    "apodize": "blackman",
    "flip": False,
    "isosurface": None,
    "show": False,
    "verbose": True,
    "debug": True,
    "handle_defects": False,
    "orthogonalise_before_phasing": False,
    "orientation_convention": "cxi",
    "pynx": {
        "data": None,
        "mask": None,
        "data2cxi": False,
        "auto_center_resize": False,
        "support": "auto",
        "support_size": None,
        "support_threshold": "0.15, 0.40",
        "support_threshold_method": "rms",
        "support_only_shrink": False,
        "support_update_period": 20,
        "support_smooth_width_begin": 2,
        "support_smooth_width_end": 0.5,
        "support_post_expand": None,  # (-1, 1)
        "support_update_border_n": None,
        "psf": "pseudo-voigt,0.5,0.1,10",
        "nb_raar": 500,
        "nb_hio": 300,
        "nb_er": 200,
        "nb_ml": 0,
        "nb_run": 20,
        "nb_run_keep": 10,
        "zero_mask": False,
        "crop_output": 0,
        "roi": "full",
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
    },
    "support": {
        "support_method": None,
        "raw_process": True,
        "support_path": None,
    },
    "facets": {
        "nb_facets": None,
        "remove_edges": True,
        "order_of_derivative": None,
        "derivative_threshold": None,
        "amplitude_threshold": None,
        "top_facet_reference_index": [1, 1, 1],
        "authorised_index": 1,
        "nb_nghbs_min": 0,
        "index_to_display": None,
        "display_f_e_c": 'facet',
        "size": 10,
    }
}


def convert_np_arrays(**data) -> dict:
    """
    Recursively converts numpy types and arrays in a dictionary to
    standard Python types for YAML serialization.

    Args:
        **data: arbitrary keyword arguments representing a dictionary
            with potential numpy types.

    Returns:
        dict: A dictionary with all numpy types converted to standard
            Python types.
    """
    def convert_value(value):
        # Handle numpy arrays
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return convert_value(value.item())
            return tuple(convert_value(v) for v in value)

        # Handle numpy scalar types.
        if isinstance(value, (np.integer, np.int32, np.int64)):
            return int(value)
        if isinstance(value, (np.floating, np.float32, np.float64)):
            return float(value)
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, (np.str_, str)):
            return str(value)

        # Handle nested lists or tuples.
        if isinstance(value, (list, tuple)):
            return type(value)(convert_value(v) for v in value)

        # If value is a dictionary, convert its contents recursively.
        if isinstance(value, dict):
            return convert_np_arrays(**value)

        # Return the value as is if no conversion is needed.
        return value

    # Apply the conversion function to each dictionary entry
    return {
        key: convert_value(value) for key, value in data.items()
    }


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

    for subdict in ("pynx", "support", "facets"):
        for name, value in AUTHORIZED_KEYS[subdict].items():
            if name not in params[subdict] or params[subdict][name] is None:
                # None of these parameters are required.
                params[subdict].update({name: value})
        for name in params[subdict]:
            if not isparameter(name):
                warnings.warn(
                    f"Parameter '{name}' is unknown, will not be used")
    for name in params:
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
    authorised = list(AUTHORIZED_KEYS.keys())
    for key in ("pynx", "support", "facets"):
        authorised += list(AUTHORIZED_KEYS[key].keys())
    return string in authorised


def get_params_from_variables(
            dir_list: list,
            globals_dict: dict
) -> dict:
    """
    Return a dictionary of parameters whose keys are authorized by the
    AUTHORIZED_KEYS list.
    """
    params = {"pynx": {}, "facets": {}, "support": {}}
    for e in dir_list:
        if e in AUTHORIZED_KEYS:
            params[e] = globals_dict[e]
        elif e in AUTHORIZED_KEYS["pynx"]:
            params["pynx"][e] = globals_dict[e]
        elif e in AUTHORIZED_KEYS["facets"]:
            params["facets"][e] = globals_dict[e]
        elif e in AUTHORIZED_KEYS["support"]:
            params["support"][e] = globals_dict[e]
    return params
