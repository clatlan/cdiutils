import numpy as np
import h5py

from cdiutils.load.loader import H5TypeLoader, h5_safe_load


class SIXSLoader(H5TypeLoader):
    """
    A class for loading data from SIXS beamline experiments.
    """

    angle_names = {
        "sample_outofplane_angle": "mu",
        "sample_inplane_angle": "omega",
        "detector_outofplane_angle": "gamma",
        "detector_inplane_angle": "delta"
    }
    authorised_detector_names = ("maxipix", )

    def __init__(
            self,
            experiment_file_path: str,
            scan: int = None,
            sample_name: str = None,
            detector_name: str = None,
            flat_field: np.ndarray | str = None,
            alien_mask: np.ndarray | str = None,
            version: str = None,
            **kwargs
    ) -> None:
        """
        Initialise SIXSLoader with experiment data directory path and
        detector information.

        Args:
            experiment_file_path (str): path to the experiment file.
            detector_name (str): name of the detector.
            sample_name (str, optional): name of the sample. Defaults to
                None.
            flat_field (np.ndarray | str, optional): flat field to
                account for the non homogeneous counting of the
                detector. Defaults to None.
            alien_mask (np.ndarray | str, optional): array to mask the
                aliens. Defaults to None.
            version (str, optional): the version of the loader. Defaults
                to None.
        """
        self.version = version
        if version is None:
            self.version = "2022"
        super().__init__(
            experiment_file_path,
            scan,
            sample_name,
            detector_name,
            flat_field,
            alien_mask
        )

    @h5_safe_load
    def load_detector_data(
            self,
            scan: int = None,
            sample_name: str = None,
            roi: tuple[slice] = None,
            rocking_angle_binning: int = None,
            binning_method: str = "sum"
    ) -> np.ndarray:
        """
        Load detector data for a given scan and sample.

        Args:
            scan (int, optional): Scan number.
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
        scan, sample_name = self._check_scan_sample(scan, sample_name)

        # path = self._get_file_path(scan, sample_name)
        # key_path = "com/scan_data/test_image"
        key_path = self._get_detector_key_path(self.h5file)

        roi = self._check_roi(roi)

        # with h5py.File(path) as h5file:
        if rocking_angle_binning:
            # we first apply the roi for axis1 and axis2
            data = self.h5file[key_path][(slice(None), roi[1], roi[2])]
        else:
            data = self.h5file[key_path][roi]

        return self.bin_flat_mask(
            data,
            roi,
            self.flat_field,
            self.alien_mask,
            rocking_angle_binning,
            binning_method
        )

    @h5_safe_load
    def load_motor_positions(
            self,
            scan: int = None,
            sample_name: str = None,
            roi: tuple[slice] = None,
            rocking_angle_binning: int = None,
    ) -> dict:
        """
        Load the motor positions, i.e diffractometer angles associated
        with a scan.

        Args:
            scan (int, optional): the scan number. Defaults to None.
            sample_name (str, optional): the sample name.
                Defaults to None.
            roi (tuple[slice], optional): the region of interest.
                Defaults to None.
            rocking_angle_binning (int, optional): the factor for the
                binning along the rocking curve axis. Defaults to None.

        Returns:
            dict: the four diffractometer angles.
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)

        if roi is None or len(roi) == 2:
            roi = slice(None)
        elif len(roi) == 3:
            roi = roi[0]

        angles = {key: None for key in self.angle_names}

        for angle, name in self.angle_names.items():
            motor_key_path = self._get_motor_key_path(self.h5file, name)
            angles[angle] = self.h5file[motor_key_path][()]

        # take care of the rocking angle
        self.rocking_angle = "sample_outofplane_angle"
        if self.version == "2019":
            node_name = "data_07"
        elif self.version == "2022":
            node_name = "actuator_1_1"
        else:
            raise ValueError(f"Version {self.version} not supported yet.")
        angles[self.rocking_angle] = self.h5file[
            f"com/scan_data/{node_name}"
        ][()]

        if rocking_angle_binning:
            angles[self.rocking_angle] = self.bin_rocking_angle_values(
                angles[self.rocking_angle], rocking_angle_binning
            )

        # take care of the roi
        if isinstance(roi, (tuple, list)):
            if len(roi) == 2:
                roi = slice(None)
            else:
                roi = roi[0]
        elif roi is None:
            roi = slice(None)
        elif not isinstance(roi, slice):
            raise ValueError(
                f"roi should be tuple of slices, or a slice, not {type(roi)}"
            )
        angles[self.rocking_angle] = angles[self.rocking_angle][roi]

        return angles

    def _get_detector_key_path(self, h5file: h5py.File) -> str:
        """
        Get the key path for the detector data.

        Args:
            h5file (h5py.File): the h5 file to search in.

        Returns:
            str: the key path.
        """
        key_path = "com/scan_data/"
        for key in h5file[key_path]:
            data = h5file[key_path + key][()]
            if isinstance(data, np.ndarray) and data.ndim == 3:
                return key_path + key
        raise ValueError("No detector data found in the file.")

    def _get_motor_key_path(self, h5file: h5py.File, name: str) -> str:
        """
        Get the key path for the motor data.

        Args:
            h5file (h5py.File): the h5 file to search in.
            name (str): the angle name to search for.

        Returns:
            str: the key path.
        """
        if self.version == "2022":
            key_path = "com/SIXS/i14-c-cx1-ex-med-v-dif-group.1"
            for key in h5file[key_path]:
                if name in key:
                    return key_path + f"/{key}/position"
        if self.version == "2019":
            key_path = "com/SIXS/"
            for key in h5file[key_path]:
                if name in key:
                    return key_path + key + "/position_pre"
        raise ValueError("No motor data found in the file.")

    def load_det_calib_params(self) -> dict:
        return None

    @h5_safe_load
    def load_energy(self, scan: int = None) -> tuple:
        """
        Load the energy of the beamline.

        Args:
            scan (int, optional): the scan number. Defaults to None.

        Returns:
            tuple: the photon energy used during beamtime.
        """
        scan, _ = self._check_scan_sample(scan, None)
        key_path = "com/SIXS/i14-c-c02-op-mono/energy"

        return self.h5file[key_path][()].item() * 1e3
