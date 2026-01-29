"""Parameter validation and management for BCDI pipeline.

This module provides utilities for managing pipeline parameters,
including validation, default filling, and type conversion. It defines
the DEFAULT_PIPELINE_PARAMS structure that serves as the schema for
all pipeline configurations.
"""

import warnings
from collections.abc import Mapping  # more flexible than dict
from typing import Any

import numpy as np

# default parameter schema for BCDI data processing pipeline
DEFAULT_PIPELINE_PARAMS = {
    # === experiment metadata (REQUIRED fields must be provided) ===
    "beamline_setup": "REQUIRED",  # beamline configuration name
    "scan": "REQUIRED",  # scan number identifier
    "experiment_file_path": None,  # path to experiment spec/log file
    "dump_dir": "REQUIRED",  # output directory for results
    "sample_name": None,  # sample identifier for organisation
    # === data paths and detector configuration ===
    "experiment_data_dir_path": None,  # root dir for raw data
    "detector_data_path": None,  # specific path to detector files
    "edf_file_template": None,  # template for EDF file names
    "detector_name": None,  # detector type (e.g., 'Maxipix')
    "flat_field": None,  # flat field correction array path
    "alien_mask": None,  # mask for defective pixels
    # === sample geometry and orientation ===
    "sample_orientation": None,  # sample rotation angles
    "sample_surface_normal": None,  # surface normal vector
    "background_level": None,  # background intensity level
    # === preprocessing parameters ===
    "preprocess_shape": (150, 150),  # cropped data shape
    "energy": None,  # x-ray energy in eV
    "hkl": [1, 1, 1],  # Miller indices of reflection
    "hot_pixel_filter": False,  # enable hot pixel removal
    "voxel_reference_methods": ["max", "com", "com"],  # centering
    "q_lab_ref": None,  # reference q-vector in lab frame
    "light_loading": False,  # load minimal data to save memory
    "det_reference_voxel": None,  # reference voxel on detector
    "rocking_angle_binning": None,  # binning for rocking curve
    "det_calib_params": None,  # detector calibration params
    # === postprocessing and visualisation ===
    "voxel_size": None,  # real space voxel size in nm
    "apodize": "blackman",  # apodisation window type
    "flip": False,  # flip reconstruction geometry
    "isosurface": None,  # isosurface threshold for 3D plot
    "show": False,  # display interactive plots
    "verbose": True,  # enable verbose output
    "debug": True,  # enable debug mode
    "handle_defects": False,  # handle detector defects
    "orthogonalise_before_phasing": False,  # orthogonalise early
    "convention": "cxi",  # coordinate convention ('cxi' or 'nexus')
    # === PyNX phase retrieval parameters ===
    "pynx": {
        "data": None,  # input data array path
        "mask": None,  # input mask array path
        "data2cxi": False,  # convert data to CXI format
        "auto_center_resize": False,  # auto-center and resize
        # support constraints
        "support": "auto",  # support determination method
        "support_size": None,  # initial support size
        "support_threshold": "0.15, 0.40",  # min/max thresholds
        "support_threshold_method": "rms",  # threshold method
        "support_only_shrink": False,  # only allow shrinking
        "support_update_period": 20,  # update every N cycles
        "support_smooth_width_begin": 2,  # initial smoothing
        "support_smooth_width_end": 0.5,  # final smoothing
        "support_post_expand": None,  # post-expand support (x/- N)
        "support_update_border_n": 0,  # border update width
        # algorithm configuration
        "algorithm": None,  # phasing algorithm sequence
        "psf": "pseudo-voigt,1,0.05,20",  # point spread function
        "nb_raar": 500,  # number of RAAR iterations
        "nb_hio": 300,  # number of HIO iterations
        "nb_er": 200,  # number of ER iterations
        "nb_ml": 0,  # number of ML iterations
        "nb_run": 20,  # number of independent runs
        "nb_run_keep": 10,  # number of runs to keep
        # constraints and processing
        "zero_mask": False,  # force masked values to zero
        "crop_output": 0,  # crop output by N pixels
        "roi": "full",  # region of interest
        "positivity": False,  # enforce positivity constraint
        "beta": 0.9,  # feedback parameter for HIO/RAAR
        "detwin": True,  # enable detwinning
        "rebin": "1, 1, 1",  # rebinning factors (z, y, x)
        # output and monitoring
        "verbose": 100,  # verbosity level (print every N iter)
        "output_format": "cxi",  # output file format
        "live_plot": False,  # enable live plotting
        "save_plot": True,  # save final plots
        "mpi": "run",  # MPI mode for parallel processing
    },
    # === support generation parameters ===
    "support": {
        "support_method": None,  # method for support generation
        "raw_process": True,  # process from raw data
        "support_path": None,  # path to precomputed support
    },
    # === facet analysis parameters ===
    "facets": {
        "nb_facets": None,  # expected number of facets
        "remove_edges": True,  # remove edge facets
        "order_of_derivative": None,  # derivative order for edge
        "derivative_threshold": None,  # threshold for derivatives
        "amplitude_threshold": None,  # min amplitude for facets
        "top_facet_reference_index": [1, 1, 1],  # reference facet
        "authorised_index": 1,  # max allowed facet index
        "nb_nghbs_min": 0,  # min neighbours for valid facet
        "index_to_display": None,  # specific facet to display
        "display_f_e_c": "facet",  # display mode (facet/edge/corner)
        "size": 10,  # facet marker size for plots
    },
}


# cache the valid keys once, instead of recomputing every time
_VALID_KEYS_CACHE = None  # global variable to store keys


def validate_and_fill_params(
    user_params: dict[str, Any],
    defaults: dict[str, Any] = DEFAULT_PIPELINE_PARAMS,
) -> dict[str, Any]:
    """Validate user parameters and fill missing values with defaults.

    Recursively validates a user-provided parameter dictionary against
    a schema of defaults. Ensures all required parameters (marked as
    'REQUIRED') are present, fills in missing optional parameters with
    their default values, and warns about unknown parameters.

    This function handles nested dictionaries (e.g., 'pynx', 'facets')
    by recursing into them and validating each level independently.

    Args:
        user_params: dictionary of user-provided pipeline parameters.
            Can be nested (e.g., {'pynx': {'nb_run': 10}}).
        defaults: schema dictionary defining allowed parameters and
            their default values. Parameters with value 'REQUIRED'
            must be provided by the user. Defaults to
            DEFAULT_PIPELINE_PARAMS.

    Returns:
        A new dictionary containing all parameters from the schema,
        with user values where provided and defaults elsewhere.

    Raises:
        ValueError: if a required parameter (value='REQUIRED' in
            defaults) is missing from user_params.

    Examples:
        >>> user = {'scan': 42, 'dump_dir': '/tmp',
        ...         'beamline_setup': 'ID01'}
        >>> params = validate_and_fill_params(user)
        >>> params['scan']
        42
        >>> params['energy']  # filled with default
        None
        >>> params['pynx']['nb_run']  # nested default
        20

        >>> # missing required parameter raises error
        >>> validate_and_fill_params({'scan': 42})
        ValueError: Missing required parameter: 'beamline_setup'
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
                UserWarning,
            )

    return filled_params


def collect_keys(d: dict[str, Any]) -> set[str]:
    """Recursively collect all keys from a nested dictionary.

    Traverses a potentially nested dictionary structure and extracts
    all keys at all levels, returning them as a flat set. This is
    useful for building a complete list of valid parameter names from
    the hierarchical DEFAULT_PIPELINE_PARAMS structure.

    Args:
        d: dictionary to extract keys from. Can contain nested dicts.

    Returns:
        Set containing all keys found at all nesting levels.

    Examples:
        >>> params = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        >>> collect_keys(params)
        {'a', 'b', 'c', 'd', 'e'}
    """
    keys = set(d.keys())
    for value in d.values():
        if isinstance(value, Mapping):
            keys |= collect_keys(value)
    return keys


def isparameter(string: str) -> bool:
    """Check if a string is a valid pipeline parameter name.

    Determines whether the given string corresponds to any key in the
    DEFAULT_PIPELINE_PARAMS schema, at any nesting level. Uses lazy
    caching to avoid recomputing the full key set on every call.

    The valid keys are computed once and stored in a global cache for
    subsequent calls, making this function efficient for repeated
    validation checks.

    Args:
        string: candidate parameter name to check.

    Returns:
        True if string is a valid parameter name in the schema, False
        otherwise.
    """
    global _VALID_KEYS_CACHE  # use the global cache

    if _VALID_KEYS_CACHE is None:  # compute only once
        _VALID_KEYS_CACHE = collect_keys(DEFAULT_PIPELINE_PARAMS)

    return string in _VALID_KEYS_CACHE


def get_params_from_variables(dir_list: list, globals_dict: dict) -> dict:
    """
    Extract pipeline parameters from global variables.

    Filters global variables by matching names against
    DEFAULT_PIPELINE_PARAMS keys. Organises parameters into
    top-level and nested sub-dicts ('pynx', 'facets', 'support').

    Args:
        dir_list (list): list of variable names (e.g., from dir()).
        globals_dict (dict): global namespace dict (e.g., globals()).

    Returns:
        dict: filtered parameter dictionary with nested structure.

    Example:
        >>> scan = 42
        >>> nb_raar = 500
        >>> params = get_params_from_variables(
        ...     dir(), globals()
        ... ) numpy types and arrays to Python types for
    YAML serialisation.
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


def convert_np_arrays(**data: Any) -> dict[str, Any]:
    """
    Handles numpy arrays, scalars (int, float, bool, str), nested
    structures (lists, tuples, dicts), and converts them to YAML-
    compatible types

    Args:
        **data: arbitrary keyword arguments representing a dictionary
            with potential numpy types.

    Returns:
        dict: A dictionary with all numpy types converted to standard
            Python types.
    """

    def convert_value(value):
        # handle numpy arrays
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return convert_value(value.item())
            return tuple(convert_value(v) for v in value)

        # handle numpy scalar types.
        if isinstance(value, (np.integer, np.int32, np.int64)):
            return int(value)
        if isinstance(value, (np.floating, np.float32, np.float64)):
            return float(value)
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, (np.str_, str)):
            return str(value)

        # handle nested lists or tuples.
        if isinstance(value, (list, tuple)):
            return type(value)(convert_value(v) for v in value)

        # if value is a dictionary, convert its contents recursively.
        if isinstance(value, dict):
            return convert_np_arrays(**value)

        # return the value as is if no conversion is needed.
        return value

    # apply the conversion function to each dictionary entry
    return {key: convert_value(value) for key, value in data.items()}
