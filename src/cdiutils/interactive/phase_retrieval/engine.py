"""
Helper functions for phase retrieval operations.

This module provides utility functions for initializing CDI operators,
saving results, and managing files in phase retrieval workflows.
"""

import glob
import os
from datetime import datetime

import numpy as np
from scipy.fft import fftshift

try:
    from pynx.cdi import CDI
    from pynx.utils.array import rebin

    IS_PYNX_AVAILABLE = True
except ImportError:
    IS_PYNX_AVAILABLE = False


def initialise_cdi_operator(
    iobs: str,
    mask: str | None,
    support: str | None,
    obj: str | None,
    binning: tuple[int, int, int] = (1, 1, 1),
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
    """
    Initialise the CDI operator by processing the possible inputs:
        - iobs
        - mask
        - support
        - obj
    Will also crop and centre the data if specified.

    Args:
        iobs: path to npz or npy that stores the intensity observations
            data
        mask: path to npz or npy that stores the mask data
        support: path to npz or npy that stores the support data
        obj: path to npz or npy that stores the object data
        binning: tuple, applied to all the arrays, e.g. (1, 1, 1)

    Return:
        cdi operator or None if initialisation fails.
    """
    if not IS_PYNX_AVAILABLE:
        raise ImportError("PyNX is required for CDI operator initialization")

    if os.path.isfile(str(iobs)):
        if iobs.endswith(".npy"):
            iobs = np.load(iobs)
            print("\tCXI input: loading data")
        elif iobs.endswith(".npz"):
            try:
                iobs = np.load(iobs)["data"]
                print("\tCXI input: loading data")
            except KeyError:
                print('\t"data" key does not exist.')
                return None
        if binning != (1, 1, 1):
            iobs = rebin(iobs, binning)
            print("\tBinned data.")

        iobs = fftshift(iobs)
    else:
        iobs = None
        print("At least iobs must exist.")
        return None

    if os.path.isfile(str(mask)):
        if mask.endswith(".npy"):
            mask = np.load(mask).astype(np.int8)
            nb = mask.sum()
            mask_percentage = nb * 100 / mask.size
            print(
                f"\tCXI input: loading mask, "
                f"with {nb} pixels masked ({mask_percentage:0.3f}%)"
            )
        elif mask.endswith(".npz"):
            for key in ["mask", "data"]:
                try:
                    mask = np.load(mask)[key].astype(np.int8)
                    nb = mask.sum()
                    mask_percentage = nb * 100 / mask.size
                    print(
                        f"\tCXI input: loading mask, "
                        f"with {nb} pixels masked ({mask_percentage:0.3f}%)"
                    )
                    break
                except KeyError:
                    print(f'\t"{key}" key does not exist.')
            else:
                print("\t--> Could not load mask array.")

        if binning != (1, 1, 1):
            mask = rebin(mask, binning)
            print("\tBinned mask.")

        mask = fftshift(mask)

    else:
        mask = None

    if os.path.isfile(str(support)):
        if support.endswith(".npy"):
            support = np.load(support)
            print("\tCXI input: loading support")
        elif support.endswith(".npz"):
            for key in ["data", "support", "obj"]:
                try:
                    support = np.load(support)[key]
                    print("\tCXI input: loading support")
                    break
                except KeyError:
                    print(f'\t"{key}" key does not exist.')
            else:
                print("\t--> Could not load support array.")

        if binning != (1, 1, 1):
            support = rebin(support, binning)
            print("\tBinned support.")

        support = fftshift(support)

    else:
        support = None

    if os.path.isfile(str(obj)):
        if obj.endswith(".npy"):
            obj = np.load(obj)
            print("\tCXI input: loading object")
        elif obj.endswith(".npz"):
            try:
                obj = np.load(obj)["data"]
                print("\tCXI input: loading object")
            except KeyError:
                print('\t"data" key does not exist.')

        if binning != (1, 1, 1):
            obj = rebin(obj, binning)
            print("\tBinned obj.")

        obj = fftshift(obj)

    else:
        obj = None

    # Create cdi object with data and mask, load the main parameters
    cdi = CDI(
        iobs,
        support=support,
        obj=obj,
        mask=mask,
    )

    return cdi


def save_cdi_operator_as_cxi(
    cdi_operator,
    path_to_cxi: str,
    params: dict | None,
):
    """
    We need to create a dictionary with the parameters to save in the
    cxi file.

    Args:
        cdi_operator: cdi object created with PyNX
        path_to_cxi: path to future cxi data
            Below are parameters that can be saved in the cxi file
            - filename: the file name to save the data to
            - iobs: the observed intensity
            - mask: the mask indicating valid (=0) and bad pixels (>0)
            - params: a dictionary of parameters which will be saved as a
              NXcollection
        params: dictionary of additional parameters to save.
    """
    print(
        "\nSaving phase retrieval parameters selected "
        "in the PyNX tab in the cxi file."
    )
    cdi_operator.save_data_cxi(
        filename=path_to_cxi,
        process_parameters=params,
    )


def list_files(
    folder: str, glob_pattern: str, verbose: bool = False
) -> list[str]:
    """
    List all files in a specified folder that match a specified
     glob pattern, and sort by creation time.

    Args:
        folder: The path to the folder where the files are located.
        glob_pattern: A string that specifies the pattern
            of the filenames to match.
        verbose: If set to True, the function will print
            the filenames and their creation timestamps to the console.
            Default is False.

    Return:
        A list of file paths that match the specified pattern and
        are sorted by creation time (most recent first).

    Example:
        file_list = list_files("/path/to/folder", verbose=True)
    """
    file_list = sorted(
        glob.glob(folder + "/" + glob_pattern),
        key=os.path.getmtime,
        reverse=True,
    )

    if verbose:
        print(80 * "#")
        for f in file_list:
            file_timestamp = datetime.fromtimestamp(
                os.path.getmtime(f)
            ).strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"\nFile: {os.path.basename(f)}\n\tCreated: {file_timestamp}"
            )
        print(80 * "#")

    return file_list
