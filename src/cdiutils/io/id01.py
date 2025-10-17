import dateutil.parser
import numpy as np
import warnings

import fabio
import silx.io

from cdiutils.io.loader import H5TypeLoader, h5_safe_load, Loader


class ID01Loader(H5TypeLoader):
    """
    A class to handle loading/reading .h5 files that were created using
    Bliss at the ID01 beamline.
    """

    angle_names = {
        "sample_outofplane_angle": "eta",
        "sample_inplane_angle": "phi",
        "detector_outofplane_angle": "delta",
        "detector_inplane_angle": "nu",
    }
    authorised_detector_names = ("mpxgaas", "mpx1x4", "eiger2M")

    def __init__(
        self,
        experiment_file_path: str,
        scan: int = None,
        sample_name: str = None,
        detector_name: str = None,
        flat_field: np.ndarray | str = None,
        alien_mask: np.ndarray | str = None,
        **kwargs,
    ) -> None:
        """
        Initialise ID01Loader with experiment data file path and
        detector information.

        Args:
            experiment_file_path (str): path to the bliss master file
                used for the experiment.
            detector_name (str): name of the detector.
            scan (int, optional): the scan number. Defaults to None.
            sample_name (str, optional): name of the sample. Defaults
                to None.
            flat_field (np.ndarray | str, optional): flat field to
                account for the non homogeneous counting of the
                detector. Defaults to None.
            alien_mask (np.ndarray | str, optional): array to mask the
                aliens. Defaults to None.
        """
        super().__init__(
            experiment_file_path,
            scan,
            sample_name,
            detector_name,
            flat_field,
            alien_mask,
        )

    @h5_safe_load
    def get_detector_name(
        self, start_scan: int = 1, max_attempts: int = 5
    ) -> str:
        """
        Get the detector name from the HDF5 file by searching through
        available scans.

        Args:
            start_scan (int): The scan number to start searching from.
            Defaults to 1.
            max_attempts (int): Maximum number of scan numbers to try.
            Defaults to 5.

        Returns:
            str: The detector name found in the file.

        Raises:
            ValueError: If no detector is found after max_attempts, or
            if multiple detectors are found.
            KeyError: If the key path structure is invalid.
        """

        msg = "Please provide a detector_name (str)."

        # Try to find the detector name in the current scan number
        key_path = f"{self.sample_name}_{start_scan}.1/measurement/"

        # If we've exceeded max attempts, raise an error
        if start_scan > max_attempts:
            raise ValueError(
                f"No detector found after checking {max_attempts} scans.\n"
                f"{msg}"
            )

        # Check if the key path exists
        if key_path not in self.h5file:
            # Try the next scan number recursively
            return self.get_detector_name(start_scan + 1, max_attempts)

        # Look for detector names in the current scan
        detector_names = []
        for key in self.authorised_detector_names:
            if key in self.h5file[key_path]:
                detector_names.append(key)

        if len(detector_names) == 0:
            # Try the next scan number recursively
            return self.get_detector_name(start_scan + 1, max_attempts)

        if len(detector_names) > 1:
            raise ValueError(
                f"Several detector names found ({detector_names}).\n"
                f"Not handled yet.\n{msg}"
            )

        return detector_names[0]

    @h5_safe_load
    def load_det_calib_params(
        self, scan: int = None, sample_name: str = None
    ) -> dict:
        """
        Load the detector calibration parameters from the scan directly.
        Note that this will only provide the direct beam position, the
        sample-to-detector distance, and the pixel size. To get the
        tilt angles of the detector run the detector calibration
        notebook.
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        key_path = (
            "_".join((sample_name, str(scan)))
            + f".1/instrument/{self.detector_name}"
        )
        try:
            return {
                "cch1": float(self.h5file[key_path + "/beam_center_y"][()]),
                "cch2": float(self.h5file[key_path + "/beam_center_x"][()]),
                "pwidth1": float(self.h5file[key_path + "/y_pixel_size"][()]),
                "pwidth2": float(self.h5file[key_path + "/x_pixel_size"][()]),
                "distance": float(self.h5file[key_path + "/distance"][()]),
                "tiltazimuth": 0.0,
                "tilt": 0.0,
                "detrot": 0.0,
            }
        except KeyError as exc:
            raise KeyError(
                f"key_path is wrong (key_path='{key_path}'). "
                "Are sample_name, scan number or detector name correct?"
            ) from exc

    @h5_safe_load
    def load_detector_shape(
        self,
        scan: int = None,
        sample_name: str = None,
    ) -> tuple:
        scan, sample_name = self._check_scan_sample(scan, sample_name)

        key_path = (
            "_".join((sample_name, str(scan)))
            + f".1/instrument/{self.detector_name}"
        )
        return (
            self.h5file[f"{key_path}/dim_j"][()],
            self.h5file[f"{key_path}/dim_i"][()],
        )

    @h5_safe_load
    def load_detector_data(
        self,
        scan: int = None,
        sample_name: str = None,
        roi: tuple[slice] = None,
        rocking_angle_binning: int = None,
        binning_method: str = "sum",
    ) -> np.ndarray:
        """
        Load the detector data.

        Args:
            scan (int, optional): the scan number. Defaults to None.
            sample_name (str, optional): the sample name.
                Defaults to None.
            roi (tuple[slice], optional): the region of interest to
                light load the data. Defaults to None.
            rocking_angle_binning (int, optional): the factor for the
                binning along the rocking curve axis. Defaults to None.
            binning_method (str, optional): the method for the binning
                along the rocking curve axis. Defaults to "sum".

        Raises:
            KeyError: if the key path is incorrect.

        Returns:
            np.ndarray: the detector data.
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        key_path = (
            "_".join((sample_name, str(scan)))
            + f".1/measurement/{self.detector_name}"
        )
        roi = self._check_roi(roi)
        try:
            if rocking_angle_binning:
                # we first apply the roi for axis1 and axis2
                data = self.h5file[key_path][(slice(None), roi[1], roi[2])]
            else:
                data = self.h5file[key_path][roi]
        except KeyError as exc:
            raise KeyError(
                f"key_path is wrong (key_path='{key_path}'). "
                "Are sample_name, scan number or detector name correct?"
            ) from exc

        return self.bin_flat_mask(
            data,
            roi,
            self.flat_field,
            self.alien_mask,
            rocking_angle_binning,
            binning_method,
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
            sample_name (str, optional): the sample name. Defaults to
                None.
            roi (tuple[slice], optional): the region of interest.
                Defaults to None.
            rocking_angle_binning (int, optional): binning factor along
                the rocking curve axis. Defaults to None.

        Returns:
            dict: the formatted diffractometer angles.
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        angles = self.load_angles(
            key_path=f"{sample_name}_{scan}.1/instrument/positioners/"
        )

        # ensure angles dictionary has correct keys and defaults to 0.0
        # if missing
        formatted_angles = {
            key: angles.get(name, 0.0)
            for key, name in ID01Loader.angle_names.items()
        }
        self.rocking_angle = self.get_rocking_angle(formatted_angles)

        scan_axis_roi = self._check_roi(roi)[0]

        # format the angles and map them back to their corresponding keys
        formatted_values = self.format_scanned_counters(
            *formatted_angles.values(),
            scan_axis_roi=scan_axis_roi,
            rocking_angle_binning=rocking_angle_binning,
        )

        # return a dictionary mapping original angle keys to their
        # formatted values. This is possible because Python maintains
        # order !
        return dict(zip(formatted_angles.keys(), formatted_values))

    @h5_safe_load
    def load_energy(self, scan: int = None, sample_name: str = None) -> float:
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        key_path = f"{sample_name}_{scan}.1/instrument/positioners/"
        try:
            energy = self.h5file[key_path + "mononrj"][()] * 1e3
            if isinstance(energy, np.ndarray):
                return energy
            return float(energy)
        except KeyError:
            warnings.warn(f"Energy not found at {key_path + 'mononrj'}. ")
            return None

    @h5_safe_load
    def show_scan_attributes(
        self,
        scan: int = None,
        sample_name: str = None,
    ) -> None:
        """Print the attributes (keys) of a given scan number"""
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        key_path = "_".join((sample_name, str(scan))) + ".1"
        print(self.h5file[key_path].keys())

    @h5_safe_load
    def load_measurement_parameters(
        self, parameter_name: str, scan: int = None, sample_name: str = None
    ) -> tuple:
        """Load the measurement parameters of the specified scan."""
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        key_path = "_".join((sample_name, str(scan))) + ".1/measurement"
        requested_mes_parameters = self.h5file[f"{key_path}/{parameter_name}"][
            ()
        ]
        return requested_mes_parameters

    @h5_safe_load
    def load_instrument_parameters(
        self,
        instrument_parameter: str,
        scan: int = None,
        sample_name: str = None,
    ) -> tuple:
        """Load the instrument parameters of the specified scan."""
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        key_path = "_".join((sample_name, str(scan))) + ".1/instrument"

        return self.h5file[key_path + "/" + instrument_parameter][()]

    @h5_safe_load
    def load_sample_parameters(
        self,
        sam_parameter: str,
        scan: int = None,
        sample_name: str = None,
    ) -> tuple:
        """Load the sample parameters of the specified scan."""
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        key_path = "_".join((sample_name, str(scan))) + ".1/sample"
        requested_parameters = self.h5file[key_path + "/" + sam_parameter][()]
        return requested_parameters

    @h5_safe_load
    def load_plotselect_parameter(
        self,
        plot_parameter,
        scan: int = None,
        sample_name: str = None,
    ) -> tuple:
        """Load the plotselect parameters of the specified scan."""
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        key_path = "_".join((sample_name, str(scan))) + ".1/plotselect"
        requested_parameter = self.h5file[key_path + "/" + plot_parameter][()]

        return requested_parameter

    @h5_safe_load
    def get_start_time(self, scan: int = None, sample_name: str = None) -> str:
        """
        This functions will return the start time of the given scan.
        the returned object is of type datetime.datetime and can
        be easily manipulated arithmetically.
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        key_path = "_".join((sample_name, str(scan))) + ".1/start_time"

        return dateutil.parser.isoparse(self.h5file[key_path][()])


def safe(func):
    def wrap(self, *args, **kwargs):
        with silx.io.open(self.experiment_file_path) as self.specfile:
            return func(self, *args, **kwargs)

    return wrap


# TODO: Implement roi parameter for detector, motors and mask methods
class SpecLoader(Loader):
    """A loader for loading .spec files."""

    angle_names = {
        "sample_outofplane_angle": "eta",
        "sample_inplane_angle": "phi",
        "detector_outofplane_angle": "del",
        "detector_inplane_angle": "nu",
    }

    def __init__(
        self,
        experiment_file_path: str,
        detector_data_path: str,
        edf_file_template: str,
        detector_name: str,
        scan: int = None,
        flat_field: str | np.ndarray = None,
        alien_mask: np.ndarray | str = None,
        **kwargs,
    ) -> None:
        """
        Initialise SpecLoader with experiment data and detector
        information.

        Args:
            experiment_file_path (str): path to the spec master file
                used for the experiment.
            detector_data_path (str): the path to the directory
                containing the detector data.
            edf_file_template (str): the file name template of the
                detector data frame.
            detector_name (str): name of the detector.
            scan (int, optional): the scan number. Defaults to None.
            flat_field (str | np.ndarray, optional): flat field to
                account for the non homogeneous counting of the
                detector. Defaults to None.
            alien_mask (np.ndarray | str, optional): array to mask the
                aliens. Defaults to None.
        """
        super().__init__(scan, None, flat_field, alien_mask)
        self.experiment_file_path = experiment_file_path
        self.detector_data_path = detector_data_path
        self.edf_file_template = edf_file_template
        self.detector_name = detector_name

    @safe
    def load_detector_data(
        self,
        scan: int = None,
        roi: tuple[slice] = None,
        rocking_angle_binning: int = None,
        binning_method: str = "sum",
    ):
        scan, _ = self._check_scan_sample(scan, None)
        roi = self._check_roi(roi)

        frame_ids = self.specfile[
            f"{scan}.1/measurement/{self.detector_name}"
        ][()]

        data = []

        template = self.detector_data_path + self.edf_file_template

        for frame_id in frame_ids:
            with fabio.open(template % frame_id) as edf_data:
                data.append(edf_data.data[roi[1:]])
        data = np.asarray(data)

        if rocking_angle_binning is None:
            data = data[roi[0]]

        return self.bin_flat_mask(
            data,
            roi,
            self.flat_field,
            self.alien_mask,
            rocking_angle_binning,
            binning_method,
        )

    @safe
    def load_motor_positions(
        self,
        scan: int = None,
        roi: tuple[slice] = None,
        rocking_angle_binning: int = None,
    ):
        scan, _ = self._check_scan_sample(scan, None)
        roi = self._check_roi(roi)
        roi = roi[0]

        positioners = self.specfile[f"{scan}.1/instrument/positioners"]

        angles = {key: None for key in SpecLoader.angle_names}
        for angle, name in SpecLoader.angle_names.items():
            try:
                angles[angle] = positioners[name][roi]
            except ValueError:
                angles[angle] = angles[angle] = positioners[name][()]

        self.rocking_angle = self.get_rocking_angle(angles)
        if self.rocking_angle is None:
            raise ValueError("No rocking angle found.")

        angles[self.rocking_angle] = self.bin_rocking_angle_values(
            angles[self.rocking_angle], rocking_angle_binning
        )
        if roi and rocking_angle_binning:
            angles[self.rocking_angle] = angles[self.rocking_angle][roi]
        return angles

    def load_det_calib_params(self) -> dict:
        return None

    def load_energy(self) -> float:
        return None

    def load_detector_shape(self) -> tuple:
        return None
