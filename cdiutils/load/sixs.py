import numpy as np
import silx.io.h5py_utils

from cdiutils.load import Loader


class SIXS2022Loader(Loader):
    """
    A class for loading data from SIXS beamline experiments.
    """

    angle_names = {
        "sample_outofplane_angle": "mu",
        "sample_inplane_angle": "omega",
        "detector_outofplane_angle": "gamma",
        "detector_inplane_angle": "delta"
    }

    def __init__(
            self,
            experiment_data_dir_path: str,
            detector_name: str,
            sample_name: str = None,
            flat_field: np.ndarray | str = None,
            alien_mask: np.ndarray | str = None,
            **kwargs
    ) -> None:
        """
        Initialise SIXSLoader with experiment data directory path and
        detector information.

        Args:
            experiment_data_dir_path (str): path to the experiment data
                directory.
            detector_name (str): name of the detector.
            sample_name (str, optional): name of the sample. Defaults to
                None.
            flat_field (np.ndarray | str, optional): flat field to
                account for the non homogeneous counting of the
                detector. Defaults to None.
            alien_mask (np.ndarray | str, optional): array to mask the
                aliens. Defaults to None.
        """
        super(SIXS2022Loader, self).__init__(flat_field, alien_mask)
        self.experiment_data_dir_path = experiment_data_dir_path
        self.detector_name = detector_name
        self.sample_name = sample_name

    def _get_file_path(
            self,
            scan: int,
            sample_name: str,
            data_type: str = "detector_motor_data"
    ) -> str:
        """
        Get the file path based on scan number, sample name, and data
        type. Only works for mu scans (out-of-plane RC).

        Args:
            scan (int): Scan number.
            sample_name (str): Name of the sample.
            data_type (str, optional): Type of data. Defaults to
                                       "detector_data".

        Returns:
            str: File path.
        """
        if data_type == "detector_motor_data":
            return (
                self.experiment_data_dir_path
                + f"/{sample_name}_ascan_mu_{scan:05d}.nxs"
            )

        raise ValueError(
            f"data_type {data_type} is not valid. Must be detector_motor_data."
        )

    def load_detector_data(
            self,
            scan: int,
            sample_name: str = None,
            roi: tuple[slice] = None,
            rocking_angle_binning: int = None,
            binning_method: str = "sum"
    ) -> np.ndarray:
        """
        Load detector data for a given scan and sample.

        Args:
            scan (int): Scan number.
            sample_name (str, optional): Name of the sample. Defaults to
                None.
            roi (tuple, optional): Region of interest. Defaults to None.
            rocking_angle_binning (int, optional): Binning factor along
                axis 0. Defaults to None.
            binning_method (str, optional): Binning method. Defaults to
                "sum".

        Returns:
            np.ndarray: Loaded detector data.
        """
        if sample_name is None:
            sample_name = self.sample_name

        path = self._get_file_path(scan, sample_name)
        key_path = "com/scan_data/test_image"

        roi = self._check_roi(roi)

        with silx.io.h5py_utils.File(path) as h5file:
            if rocking_angle_binning:
                # we first apply the roi for axis1 and axis2
                data = h5file[key_path][(slice(None), roi[1], roi[2])]
                # But then we'll keep only the roi for axis0
                roi = (roi[0], slice(None), slice(None))
            else:
                data = h5file[key_path][roi]

        return self.bin_flat_mask(
            data,
            roi,
            self.flat_field,
            self.alien_mask,
            rocking_angle_binning,
            binning_method
        )

    def load_motor_positions(
            self,
            scan: int,
            sample_name: str = None,
            roi: tuple[slice] = None,
            rocking_angle_binning: int = None,
    ) -> dict:
        """
        Load the motor positions, i.e diffractometer angles associated
        with a scan.

        Args:
            scan (int): the scan number
            sample_name (str, optional): the sample name.
                Defaults to None.
            roi (tuple[slice], optional): the region of interest.
                Defaults to None.
            rocking_angle_binning (int, optional): the factor for the
                binning along the rocking curve axis. Defaults to None.

        Returns:
            dict: the four diffractometer angles.
        """

        if sample_name is None:
            sample_name = self.sample_name

        path = self._get_file_path(scan, sample_name)
        key_path = "com/scan_data/"

        if roi is None or len(roi) == 2:
            roi = slice(None)
        elif len(roi) == 3:
            roi = roi[0]

        angles = {key: None for key in SIXS2022Loader.angle_names.keys()}

        with silx.io.h5py_utils.File(path) as h5file:
            for angle, name in SIXS2022Loader.angle_names.items():
                if rocking_angle_binning:
                    angles[angle] = h5file[key_path + name][()]
                else:
                    try:
                        angles[angle] = h5file[key_path + name][roi]
                    except ValueError:
                        angles[angle] = h5file[key_path + name][()]

        self.rocking_angle_name = self.get_rocking_angle_axis(angles)
        angles[self.rocking_angle] = self.bin_rocking_angle_values(
            angles[self.rocking_angle]
        )
        if roi and rocking_angle_binning:
            angles[self.rocking_angle] = angles[self.rocking_angle][roi]
        return angles
