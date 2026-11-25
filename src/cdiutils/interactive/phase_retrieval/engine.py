"""
Helper functions for phase retrieval operations.

This module provides utility functions for initializing CDI operators,
saving results, and managing files in phase retrieval workflows.
"""

import glob
import os
from datetime import datetime

import h5py
import numpy as np
from scipy.fft import fftshift

try:
    from pynx.cdi import CDI
    from pynx.utils.array import rebin

    IS_PYNX_AVAILABLE = True
except ImportError:
    IS_PYNX_AVAILABLE = False


def load_object(path: str) -> np.ndarray:
    """
    Load a complex object from a cxi, npz or npy file.

    The array must be centred, as stored in the PyNX cxi files. For a
    cxi file, the object is read from /entry_1/image_1/data. For a npz
    file, the keys "object", "obj" and "data" are tried, in this order.

    Args:
        path: path to the file that stores the object data.

    Raises:
        ValueError: if the file format is not supported or the object
            is not found in the file.

    Returns:
        the object array.
    """
    if path.endswith(".npy"):
        return np.load(path)

    if path.endswith(".npz"):
        with np.load(path) as data:
            for key in ("object", "obj", "data"):
                if key in data:
                    return data[key]
        raise ValueError(
            f"None of the keys 'object', 'obj' or 'data' found in {path}."
        )

    if path.endswith(".cxi"):
        with h5py.File(path, "r") as file:
            if "entry_1/image_1/data" not in file:
                raise ValueError(
                    f"No object at /entry_1/image_1/data in {path}."
                )
            return file["entry_1/image_1/data"][()]

    raise ValueError(
        f"Unsupported object file {path}, expected cxi, npz or npy."
    )


def initialise_cdi_operator(
    iobs: str,
    mask: str | None,
    support: str | None,
    obj: str | None,
    binning: tuple[int, int, int] = (1, 1, 1),
    iobs_saturation: float | None = None,
):
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
        obj: path to cxi, npz or npy that stores the object data
        binning: tuple, applied to all the arrays, e.g. (1, 1, 1)
        iobs_saturation: if not None and > 0, pixels with an observed
            intensity (after binning) above this value are masked, as
            the PyNX --iobs_saturation option.

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
        obj = load_object(obj)
        print("\tCXI input: loading object")

        if binning != (1, 1, 1):
            obj = rebin(obj, binning)
            print("\tBinned obj.")

        if obj.shape != iobs.shape:
            raise ValueError(
                f"Object shape {obj.shape} does not match the data shape "
                f"{iobs.shape}."
            )

        obj = fftshift(obj)

    else:
        obj = None

    # Mask saturated pixels (iobs and mask are both fft-shifted here)
    if iobs_saturation is not None and iobs_saturation > 0:
        saturated = iobs > iobs_saturation
        if mask is None:
            mask = np.zeros(iobs.shape, dtype=np.int8)
        mask = (mask.astype(bool) | saturated).astype(np.int8)
        print(
            f"\tMasked {saturated.sum()} pixels with iobs > {iobs_saturation}"
        )

    # Create cdi object with data and mask, load the main parameters
    cdi = CDI(
        iobs,
        support=support,
        obj=obj,
        mask=mask,
    )

    return cdi


def build_operator_chain_steps(
    nb_hio: int,
    nb_raar: int,
    nb_er: int,
    nb_ml: int = 0,
    support_update_period: int = 0,
    psf: bool = False,
    update_psf: int = 0,
    zero_mask: bool | str = False,
    detwin: bool = False,
) -> list[dict]:
    """
    Describe the operator chain of one phase retrieval run.

    The chain is the cdiutils one: HIO (nb_hio), then RAAR (nb_raar),
    then ER (nb_er), then ML (nb_ml). This is also the order used by the
    PyNX runner (pynx/cdi/runner/runner.py, CDIRunnerScan.run_algorithm).
    If support_update_period > 0, each algorithm is applied in blocks of
    support_update_period cycles, each block being followed by a
    SupportUpdate (HIO, RAAR and ER only). A last, shorter block holds
    the remaining cycles, without support update.

    Events are inserted between two blocks, before the block that
    contains the event cycle:
        - PSF initialisation after nb_hio + nb_raar // 2 cycles (cdiutils
          convention; the PyNX runner uses 66 % of nb_hio + nb_raar).
        - zero_mask="auto": masked pixels are forced to zero (1) until
          60 % of nb_hio + nb_raar cycles, then left free (0), as in the
          PyNX runner.
        - detwin: 10 cycles of DetwinHIO or DetwinRAAR (support cut in
          half) after 25 % of nb_hio + nb_raar cycles, as in the PyNX
          runner.
    ML is skipped when the PSF is used, as in the PyNX runner.

    This function does not need PyNX: the steps are converted into PyNX
    operators by PhaseRetrievalGUI, and printed by
    format_operator_chain.

    Args:
        nb_hio: number of HIO cycles.
        nb_raar: number of RAAR cycles.
        nb_er: number of ER cycles.
        nb_ml: number of maximum-likelihood cycles, run after ER.
        support_update_period: number of cycles between two support
            updates, 0 to keep the support fixed.
        psf: whether the point-spread function is used.
        update_psf: the PSF is updated every update_psf cycles once
            initialised.
        zero_mask: True, False or "auto".
        detwin: whether to run the 10 detwinning cycles.

    Returns:
        The list of steps, in execution order. Each step is a dict with
        a "kind" key, one of:
            - "algo": keys "algo" ("HIO", "RAAR", "ER" or "ML"),
              "cycles" (per block), "repeat" (number of blocks),
              "support_update", "zero_mask", "update_psf", "start" and
              "end" (cycle numbers).
            - "init_psf": key "start".
            - "detwin": keys "algo" ("DetwinHIO" or "DetwinRAAR"),
              "cycles", "zero_mask" and "start".
            - "zero_mask_switch": keys "start" and "zero_mask" (new
              value).
            - "skip": key "message", for a step that is not run.
    """
    nb_hio_raar = nb_hio + nb_raar

    # Events, as (cycle, kind) tuples, fired before the block holding them
    events = []
    if psf:
        events.append((nb_hio + nb_raar // 2, "init_psf"))
    auto_zero_mask = isinstance(zero_mask, str) and zero_mask.lower() == "auto"
    if auto_zero_mask:
        events.append((int(0.6 * nb_hio_raar), "zero_mask_switch"))
        current_zero_mask = 1
    else:
        current_zero_mask = zero_mask
    if detwin:
        events.append((nb_hio_raar // 4, "detwin"))
    events.sort()

    current_update_psf = 0
    steps = []
    cycle = 0
    for algo, nb_cycles in (("HIO", nb_hio), ("RAAR", nb_raar), ("ER", nb_er)):
        if nb_cycles <= 0:
            continue

        # Block sizes: full support update blocks, or one block split at
        # the events that fall inside this algorithm
        if support_update_period > 0:
            nb_full_blocks = nb_cycles // support_update_period
            block_sizes = [support_update_period] * nb_full_blocks
            remainder = nb_cycles % support_update_period
            if remainder:
                block_sizes.append(remainder)
        else:
            cuts = sorted(
                {
                    event_cycle - cycle
                    for event_cycle, _ in events
                    if cycle < event_cycle < cycle + nb_cycles
                }
            )
            bounds = [0] + cuts + [nb_cycles]
            block_sizes = [
                stop - start for start, stop in zip(bounds[:-1], bounds[1:])
            ]

        for size in block_sizes:
            # Fire the events that fall in the coming block
            while events and events[0][0] < cycle + size:
                _, event = events.pop(0)
                if event == "init_psf":
                    steps.append({"kind": "init_psf", "start": cycle})
                    current_update_psf = update_psf
                elif event == "zero_mask_switch":
                    current_zero_mask = 0
                    steps.append(
                        {
                            "kind": "zero_mask_switch",
                            "start": cycle,
                            "zero_mask": 0,
                        }
                    )
                elif event == "detwin":
                    steps.append(
                        {
                            "kind": "detwin",
                            "algo": (
                                "DetwinHIO" if algo == "HIO" else "DetwinRAAR"
                            ),
                            "cycles": 10,
                            "zero_mask": current_zero_mask,
                            "start": cycle,
                        }
                    )

            steps.append(
                {
                    "kind": "algo",
                    "algo": algo,
                    "cycles": size,
                    "repeat": 1,
                    "support_update": (
                        support_update_period > 0
                        and size == support_update_period
                    ),
                    "zero_mask": current_zero_mask,
                    "update_psf": current_update_psf,
                    "start": cycle,
                    "end": cycle + size,
                }
            )
            cycle += size

    if nb_ml > 0:
        if psf:
            steps.append(
                {
                    "kind": "skip",
                    "message": (
                        f"ML**{nb_ml} skipped: ML cannot be used with the "
                        "PSF (as in the PyNX runner)"
                    ),
                }
            )
        else:
            steps.append(
                {
                    "kind": "algo",
                    "algo": "ML",
                    "cycles": nb_ml,
                    "repeat": 1,
                    "support_update": False,
                    "zero_mask": current_zero_mask,
                    "update_psf": 0,
                    "start": cycle,
                    "end": cycle + nb_ml,
                }
            )

    # Merge identical consecutive blocks into a single repeated block
    merged_steps = []
    merge_keys = (
        "algo",
        "cycles",
        "support_update",
        "zero_mask",
        "update_psf",
    )
    for step in steps:
        previous = merged_steps[-1] if merged_steps else None
        if (
            previous is not None
            and step["kind"] == "algo"
            and previous["kind"] == "algo"
            and all(step[k] == previous[k] for k in merge_keys)
        ):
            previous["repeat"] += 1
            previous["end"] = step["end"]
        else:
            merged_steps.append(dict(step))
    return merged_steps


def format_operator_chain(steps: list[dict], params: dict) -> str:
    """
    Describe, as text, the operator chain applied in each run.

    Args:
        steps: the algorithm steps, from build_operator_chain_steps.
        params: the phase retrieval parameters, as built by
            PhaseRetrievalGUI.init_phase_retrieval_tab.

    Returns:
        A multi-line string listing the operators in execution order
        (PyNX applies a product of operators from right to left).
    """
    lines = [
        "Operator chain of each run, in execution order "
        "(PyNX applies 'A * B * cdi' as B first, then A):",
    ]

    # Initialisation operators
    init_lines = ["InitFreePixels()"]
    if params.get("mask_interp"):
        interp_d, interp_n = params["mask_interp"]
        init_lines.append(f"InterpIobsMask(d={interp_d}, n={interp_n})")
    if params.get("support"):
        init_lines.append("support loaded from file")
    else:
        init_lines.append(
            "AutoCorrelationSupport(threshold="
            f"{params.get('support_autocorrelation_threshold')})"
        )
    if params.get("obj"):
        init_lines.append("object loaded from file")
    else:
        phirange = "0" if params.get("positivity") else "2pi"
        init_lines.append(
            f"InitObjRandom(src='support', amin=0, amax=1, "
            f"phirange={phirange})"
        )
    init_lines.append("ScaleObj()")

    # Algorithm steps
    algo_lines = []
    totals = {"HIO": 0, "RAAR": 0, "ER": 0, "ML": 0}
    nb_support_updates = 0
    for step in steps:
        kind = step["kind"]
        if kind == "algo":
            block = f"{step['algo']}**{step['cycles']}"
            if step["support_update"]:
                block = f"SupportUpdate * {block}"
                nb_support_updates += step["repeat"]
            if step["repeat"] > 1:
                block = f"({block})**{step['repeat']}"
            if step["algo"] == "ML":
                block = f"{block} * FourierApplyAmplitude()"
            details = f"zero_mask={step['zero_mask']}"
            if step["update_psf"]:
                details += f", update_psf={step['update_psf']}"
            algo_lines.append(
                f"{block:<40s} cycles {step['start']:>5d}-"
                f"{step['end']:<5d} ({details})"
            )
            totals[step["algo"]] += step["cycles"] * step["repeat"]
        elif kind == "init_psf":
            psf_text = (
                f"InitPSF(model={params.get('psf_model')}, "
                f"fwhm={params.get('fwhm')}"
            )
            if params.get("psf_model") == "pseudo-voigt":
                psf_text += f", eta={params.get('eta')}"
            psf_text += f", filter={params.get('psf_filter')})"
            algo_lines.append(f"{psf_text:<40s} at cycle {step['start']}")
        elif kind == "detwin":
            detwin_text = f"{step['algo']}(nb_cycle={step['cycles']})"
            algo_lines.append(
                f"{detwin_text:<40s} at cycle {step['start']} "
                "(support cut in half)"
            )
        elif kind == "zero_mask_switch":
            algo_lines.append(
                f"{'zero_mask: 1 -> 0':<40s} at cycle {step['start']}"
            )
        elif kind == "skip":
            algo_lines.append(f"[{step['message']}]")

    step_number = 1
    for line in init_lines + algo_lines:
        lines.append(f"  {step_number:2d}. {line}")
        step_number += 1

    lines.append(
        "Total cycles: "
        + ", ".join(f"{k} {v}" for k, v in totals.items())
        + f"; {nb_support_updates} support updates."
    )
    if nb_support_updates:
        lines.append(
            "SupportUpdate(threshold_relative=<drawn per run in "
            f"{params.get('support_threshold')}>, "
            f"method={params.get('support_method')}, "
            f"smooth_width={params.get('support_smooth_width')}, "
            f"post_expand={params.get('support_post_expand')}, "
            f"force_shrink={params.get('support_only_shrink')}, "
            f"update_border_n={params.get('support_update_border_n')}, "
            f"min_fraction={params.get('support_fraction_min')}, "
            f"max_fraction={params.get('support_fraction_max')})"
        )
    lines.append(
        f"HIO/RAAR beta={params.get('beta')}, "
        f"positivity={params.get('positivity')}, "
        f"tv_hio={params.get('tv_hio')}, tv_raar={params.get('tv_raar')}, "
        f"tv_er={params.get('tv_er')}, confidence_interval_factor_mask="
        f"({params.get('confidence_interval_factor_mask_min')}, "
        f"{params.get('confidence_interval_factor_mask_max')})"
    )
    return "\n".join(lines)


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
