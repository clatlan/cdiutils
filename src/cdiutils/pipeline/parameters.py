from collections.abc import Mapping  # more flexible than dict
import numpy as np
import warnings

DEFAULT_PIPELINE_PARAMS = {
    # Formerly the "metadata"
    "beamline_setup": "REQUIRED",
    "scan": "REQUIRED",
    "experiment_file_path": None,
    "dump_dir": "REQUIRED",
    "sample_name": None,
    "experiment_data_dir_path": None,
    "detector_data_path": None,
    "edf_file_template": None,
    "detector_name": None,
    "flat_field": None,
    "alien_mask": None,
    "sample_orientation": None,
    "sample_surface_normal": None,

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
        "algorithm": None,
        "psf": "pseudo-voigt,1,0.05,20",  # "pseudo-voigt,1,0.05,20",
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


# cache the valid keys once, instead of recomputing every time
_VALID_KEYS_CACHE = None  # global variable to store keys


def validate_and_fill_params(
        user_params: dict,
        defaults: dict = DEFAULT_PIPELINE_PARAMS
) -> dict:
    """
    Validate user parameters against DEFAULT_PIPELINE_PARAMS. Ensures
    required parameters are present and fills in missing optional ones.

    Args:
        user_params (dict): dict of user-provided parameters.
        defaults (dict, optional): default pipeline parameters (can be
            nested). Defaults to DEFAULT_PIPELINE_PARAMS.

    Raises:
        ValueError: if a required parameter is missing.

    Returns:
        dict: new dictionary with defaults filled in
    """
    filled_params = {}

    for key, default in defaults.items():
        user_value = user_params.get(key, None)

        # handle nested dictionaries recursively
        if isinstance(default, Mapping):
            if user_value is None:
                user_value = {}  # create an empty dict if missing
            filled_params[key] = validate_and_fill_params(user_value, default)

        # check for required parameters
        elif default == "REQUIRED" and user_value is None:
            raise ValueError(f"Missing required parameter: '{key}'")

        # use user value or fallback to default
        else:
            filled_params[key] = (
                user_value if user_value is not None else default
            )

    # warn for unexpected parameters
    known_keys = set(defaults.keys())  # optimised for membership tests
    for key in user_params:
        if key not in known_keys:
            warnings.warn(
                f"Parameter '{key}' is unknown and will not be used.",
                UserWarning
            )

    return filled_params


def collect_keys(d: dict) -> set:
    """Recursively collect all keys from a nested dictionary."""
    keys = set(d.keys())
    for value in d.values():
        if isinstance(value, Mapping):
            keys |= collect_keys(value)
    return keys


def isparameter(string: str) -> bool:
    """
    Check if a string is a valid parameter name in
    DEFAULT_PIPELINE_PARAMS.
    """
    global _VALID_KEYS_CACHE  # use the global cache

    if _VALID_KEYS_CACHE is None:  # compute only once
        _VALID_KEYS_CACHE = collect_keys(DEFAULT_PIPELINE_PARAMS)

    return string in _VALID_KEYS_CACHE


def get_params_from_variables(
            dir_list: list,
            globals_dict: dict
) -> dict:
    """
    Return a dictionary of parameters whose keys are authorized by the
    DEFAULT_PIPELINE_PARAMS list.
    """
    params = {"pynx": {}, "facets": {}, "support": {}}
    for e in dir_list:
        if e in DEFAULT_PIPELINE_PARAMS:
            params[e] = globals_dict[e]
        elif e in DEFAULT_PIPELINE_PARAMS["pynx"]:
            params["pynx"][e] = globals_dict[e]
        elif e in DEFAULT_PIPELINE_PARAMS["facets"]:
            params["facets"][e] = globals_dict[e]
        elif e in DEFAULT_PIPELINE_PARAMS["support"]:
            params["support"][e] = globals_dict[e]
    return params


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