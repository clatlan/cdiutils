
AUTHORIZED_KEYS = {
    "cdiutils": [
        "metadata",
        "preprocessing_output_shape",
        "energy",
        "roi",
        "hkl",
        "det_reference_voxel_method",
        "det_reference_voxel",
        "q_lab_reference",
        "q_lab_max",
        "q_lab_com",
        "det_calib_parameters",
        "voxel_size",
        "apodize",
        "flip",
        "isosurface",
        "usetex",
        "show",
        "verbose"
    ],
    "pynx": [
        "data",
        "mask",
        "data2cxi",
        "auto_center_resize",
        "support_type",
        "support_size",
        "support",
        "support_threshold",
        "support_threshold_method",
        "support_only_shrink",
        "support_update_period",
        "support_smooth_width_begin",
        "support_smooth_width_end",
        "support_post_expand",
        "psf",
        "nb_raar",
        "nb_hio",
        "nb_er",
        "nb_ml",
        "nb_run",
        "nb_run_keep",
        "zero_mask",
        "crop_output",
        "positivity",
        "beta",
        "detwin",
        "rebin",
        "detector_distance",
        "pixel_size_detector",
        "wavelength",
        "verbose",
        "output_format",
        "live_plot",
        "save_plot",
        "mpi"
    ]
}

def isparameter(string: str):
    """Return whether or not the given string is in AUTHORIZED_KEYS."""
    return (string in AUTHORIZED_KEYS["cdiutils"] +  AUTHORIZED_KEYS["pynx"])

def get_parameters_from_notebook_variables(
            dir_list: list,
            globals_dict: dict
) -> dict:
    """
    Return a dictionary of parameters whose keys are authorized by the 
    AUTORIZD_KEYS list.
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