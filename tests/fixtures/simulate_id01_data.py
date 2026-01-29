"""
Simulate ID01 beamline HDF5 data structure for testing purposes.

This module creates a minimal but complete representation of the nested
HDF5 file structure used at the ID01 beamline (ESRF), including:
- Master experiment file with external links
- Sample folder with dataset files and external links
- Dataset folder with scan files and internal links
- Scan folder with detector data files

The simulated data includes minimal metadata required by ID01Loader.
"""

import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


def create_detector_file(
    filepath: str,
    detector_name: str,
    num_frames: int,
    detector_shape: tuple[int, int],
    detector_data: np.ndarray = None,
) -> None:
    """
    Create the lowest-level detector file containing actual measurements.

    This file contains the detector images and acquisition metadata at:
    `entry_0000/instrument/detector_name/data`

    Args:
        filepath: path where the detector file will be created.
        detector_name: name of the detector (e.g., 'mpx1x4').
        num_frames: number of frames in the rocking curve.
        detector_shape: detector dimensions (rows, cols).
        detector_data: actual detector data. If None, random Poisson
            data is generated. Defaults to None.

    Raises:
        ValueError: if detector_data shape doesn't match expected
            dimensions.
    """
    if detector_data is None:
        # generate realistic-looking diffraction data
        detector_data = np.random.poisson(
            lam=10, size=(num_frames, *detector_shape)
        ).astype(np.int32)

    if detector_data.shape != (num_frames, *detector_shape):
        raise ValueError(
            f"detector_data shape {detector_data.shape} does not "
            f"match expected shape ({num_frames}, {detector_shape})"
        )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # get current timestamp
    timestamp = datetime.now().isoformat(timespec="seconds")

    with h5py.File(filepath, "w") as f:
        # create entry_0000 group
        entry = f.create_group("entry_0000")
        entry.create_dataset("start_time", data=timestamp)
        entry.create_dataset("end_time", data=timestamp)
        entry.create_dataset("title", data="test_scan")

        # create instrument group
        instrument = entry.create_group("instrument")
        detector = instrument.create_group(detector_name)

        # add detector data - this is what the loader reads
        detector.create_dataset("data", data=detector_data)

        # add acquisition parameters
        acq = detector.create_group("acquisition")
        acq.create_dataset("exposure_time", data=0.1)
        acq.create_dataset("latency_time", data=0.01)
        acq.create_dataset("nb_frames", data=num_frames)
        acq.create_dataset("mode", data="ACCUMULATION")
        acq.create_dataset("trigger_mode", data="EXTERNAL_GATE")

        # create empty measurement group (required by structure)
        entry.create_group("measurement")


def create_scan_metadata_file(
    filepath: str,
    scan_number: int,
    detector_name: str,
    detector_shape: tuple[int, int],
    num_frames: int,
    motor_positions: dict[str, float | np.ndarray] = None,
    det_calib_params: dict[str, float] = None,
    energy: float = 9000.0,
    scan_folder: str = None,
) -> None:
    """
    Create scan-level file with metadata and links to detector file.

    This file contains calibration parameters, motor positions, and
    soft links to the actual detector data.

    Args:
        filepath: path where the scan metadata file will be created.
        scan_number: scan number (e.g., 54 for scan0054).
        detector_name: name of the detector.
        detector_shape: detector dimensions.
        num_frames: number of frames in the rocking curve.
        motor_positions: diffractometer angles. If None, default
            values are used. Defaults to None.
        det_calib_params: detector calibration parameters (distance,
            beam centre, pixel size). If None, default values are
            used. Defaults to None.
        energy: beam energy in eV. Defaults to 9000.0.
        scan_folder: name of scan folder for external link. Defaults
            to None.

    Raises:
        ValueError: if motor_positions contain invalid array shapes.
    """
    if motor_positions is None:
        # default diffractometer angles (ID01 names)
        motor_positions = {
            "eta": 30.0,  # sample out-of-plane
            "phi": 0.0,  # sample in-plane
            "nu": 35.0,  # detector in-plane
            "delta": 30.0,  # detector out-of-plane
        }

    if det_calib_params is not None:
        detector_distance = det_calib_params.get("distance", 1.0)
        beam_center = (
            det_calib_params.get("cch1", detector_shape[0] / 2),
            det_calib_params.get("cch2", detector_shape[1] / 2),
        )
        pixel_size = (
            det_calib_params.get("pwidth1", 55e-6),
            det_calib_params.get("pwidth2", 55e-6),
        )
    else:
        detector_distance = 1.0
        beam_center = (detector_shape[0] / 2, detector_shape[1] / 2)
        pixel_size = (55e-6, 55e-6)  # 55 microns (typical for mpx1x4)

    if scan_folder is None:
        scan_folder = f"scan{scan_number:04d}"

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # get current timestamp
    timestamp = datetime.now().isoformat(timespec="seconds")

    with h5py.File(filepath, "w") as f:
        # create scan entry (format: scan_number.1)
        entry_name = f"{scan_number}.1"
        entry = f.create_group(entry_name)

        entry.create_dataset("start_time", data=timestamp)
        entry.create_dataset("end_time", data=timestamp)
        entry.create_dataset("title", data=f"scan {scan_number}")

        # create instrument group with detector calibration
        instrument = entry.create_group("instrument")
        detector = instrument.create_group(detector_name)

        # detector calibration parameters (read by
        # load_det_calib_params)
        detector.create_dataset("beam_center_y", data=beam_center[0])
        detector.create_dataset("beam_center_x", data=beam_center[1])
        detector.create_dataset("y_pixel_size", data=pixel_size[0])
        detector.create_dataset("x_pixel_size", data=pixel_size[1])
        detector.create_dataset("distance", data=detector_distance)
        detector.create_dataset("dim_j", data=detector_shape[0])
        detector.create_dataset("dim_i", data=detector_shape[1])

        # create positioners group (read by load_motor_positions)
        positioners = instrument.create_group("positioners")
        positioners.create_dataset("mononrj", data=energy / 1e3)  # keV

        for motor_name, position in motor_positions.items():
            if isinstance(position, np.ndarray):
                if position.shape != (num_frames,):
                    raise ValueError(
                        f"Motor array for '{motor_name}' must have "
                        f"shape ({num_frames},), got {position.shape}"
                    )
                positioners.create_dataset(motor_name, data=position)
            else:
                positioners.create_dataset(motor_name, data=position)

        # create measurement group with soft links to detector data
        measurement = entry.create_group("measurement")

        # soft link to detector data in scan folder
        detector_file = f"{scan_folder}/{detector_name}_0000.h5"
        link_target = f"/entry_0000/instrument/{detector_name}/data"
        measurement[detector_name] = h5py.ExternalLink(
            detector_file, link_target
        )

        # create sample group
        sample = entry.create_group("sample")
        sample.create_dataset("name", data="test_sample")
        sample.create_dataset("description", data="simulated sample")

        # create empty plotselect group (required by structure)
        entry.create_group("plotselect")


def create_dataset_file(
    filepath: str,
    dataset_name: str,
    scan_numbers: list[int],
) -> None:
    """
    Create dataset-level file with external links to scan files.

    This file contains external links to individual scan metadata
    files.

    Args:
        filepath: path where the dataset file will be created.
        dataset_name: name of the dataset.
        scan_numbers: list of scan numbers to link to.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with h5py.File(filepath, "w") as f:
        for scan_num in scan_numbers:
            # create external link to scan file
            scan_file = f"{dataset_name}.h5"
            entry_name = f"{dataset_name}_{scan_num}.1"
            link_path = f"/{scan_num}.1"

            f[entry_name] = h5py.ExternalLink(scan_file, link_path)


def create_sample_file(
    filepath: str,
    sample_name: str,
    dataset_names: list[str],
    scan_numbers: list[int],
) -> None:
    """
    Create sample-level file with external links to dataset files.

    Args:
        filepath: path where the sample file will be created.
        sample_name: name of the sample.
        dataset_names: list of dataset names.
        scan_numbers: list of scan numbers for each dataset.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with h5py.File(filepath, "w") as f:
        for dataset_name in dataset_names:
            for scan_num in scan_numbers:
                # external link to dataset file
                dataset_file = f"{dataset_name}/{dataset_name}.h5"
                entry_name = f"{dataset_name}_{scan_num}.1"
                link_path = f"/{scan_num}.1"

                f[entry_name] = h5py.ExternalLink(dataset_file, link_path)


def create_experiment_file(
    filepath: str,
    experiment_name: str,
    sample_name: str,
    dataset_names: list[str],
    scan_numbers: list[int],
) -> None:
    """
    Create master experiment file with external links to sample files.

    Args:
        filepath: path where the experiment file will be created.
        experiment_name: name of the experiment.
        sample_name: name of the sample.
        dataset_names: list of dataset names per sample.
        scan_numbers: list of scan numbers.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with h5py.File(filepath, "w") as f:
        for dataset_name in dataset_names:
            for scan_num in scan_numbers:
                # external link to sample file
                sample_file = (
                    f"{sample_name}/{experiment_name}_{sample_name}.h5"
                )
                entry_name = f"{dataset_name}_{scan_num}.1"
                link_path = f"/{dataset_name}_{scan_num}.1"

                f[entry_name] = h5py.ExternalLink(sample_file, link_path)


def create_id01_experiment_file(
    output_dir: str,
    experiment_name: str,
    sample_name: str,
    dataset_name: str,
    scan_number: int,
    detector_name: str = "mpx1x4",
    num_frames: int = 41,
    detector_shape: tuple[int, int] = (516, 516),
    detector_data: np.ndarray = None,
    motor_positions: dict[str, float | np.ndarray] = None,
    det_calib_params: dict[str, float] = None,
    energy: float = 9000.0,
) -> Path:
    """
    Create a complete simulated ID01 experiment HDF5 file structure.

    This function creates all necessary files and folders to simulate
    a minimal but complete ID01 beamline experiment, compatible with
    ID01Loader.

    The structure created is:
    ```
    output_dir/
        experiment_name.h5              # master file
        sample_name/
            experiment_name_sample_name.h5  # sample file
            dataset_name/
                sample_name_dataset_name.h5  # dataset file
                scan{scan_number:04d}/
                    detector_name_0000.h5  # detector data
    ```

    Args:
        output_dir: directory where structure will be created.
        experiment_name: name of the experiment.
        sample_name: name of the sample.
        dataset_name: name of the dataset.
        scan_number: scan number.
        detector_name: detector name. Defaults to 'mpx1x4'.
        num_frames: number of frames. Defaults to 41.
        detector_shape: detector dimensions. Defaults to (516, 516).
        detector_data: actual detector data. Defaults to None.
        motor_positions: motor positions. Defaults to None.
        det_calib_params: detector calibration parameters. Defaults to
            None.
        energy: beam energy in eV. Defaults to 9000.0.

    Returns:
        path to the master experiment file.
    """
    output_path = Path(output_dir)

    # create scan folder and detector file
    scan_folder = f"scan{scan_number:04d}"
    detector_filepath = (
        output_path
        / sample_name
        / dataset_name
        / scan_folder
        / f"{detector_name}_0000.h5"
    )
    create_detector_file(
        str(detector_filepath),
        detector_name,
        num_frames,
        detector_shape,
        detector_data,
    )

    # create dataset metadata file
    dataset_filepath = (
        output_path / sample_name / dataset_name / f"{dataset_name}.h5"
    )
    create_scan_metadata_file(
        str(dataset_filepath),
        scan_number,
        detector_name,
        detector_shape,
        num_frames,
        motor_positions,
        det_calib_params,
        energy,
        scan_folder,
    )

    # create sample file with links to dataset
    sample_filepath = (
        output_path / sample_name / f"{experiment_name}_{sample_name}.h5"
    )
    create_sample_file(
        str(sample_filepath),
        sample_name,
        [dataset_name],
        [scan_number],
    )

    # create experiment master file
    experiment_filepath = output_path / f"{experiment_name}.h5"
    create_experiment_file(
        str(experiment_filepath),
        experiment_name,
        sample_name,
        [dataset_name],
        [scan_number],
    )

    return experiment_filepath
