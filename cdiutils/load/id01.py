import dateutil.parser
import numpy as np
import warnings

import fabio
import silx.io
import silx.io.specfile

from cdiutils.load.loader import H5TypeLoader, h5_safe_load, Loader


class ID01Loader(H5TypeLoader):
    """
    A class to handle loading/reading .h5 files that were created using
    Bliss at the ID01 beamline.
    """

    angle_names = {
        "sample_outofplane_angle": "eta",
        "sample_inplane_angle": "phi",
        "detector_outofplane_angle": "delta",
        "detector_inplane_angle": "nu"
    }
    authorised_detector_names = ("mpxgaas", "mpx1x4", "eiger2M")

    def __init__(
            self,
            experiment_file_path: str,
            sample_name: str = None,
            detector_name: str = None,
            flat_field: np.ndarray | str = None,
            alien_mask: np.ndarray | str = None,
            **kwargs
    ) -> None:
        """
        Initialise ID01Loader with experiment data file path and
        detector information.

        Args:
            experiment_file_path (str): path to the bliss master file
                used for the experiment.
            detector_name (str): name of the detector.
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
            sample_name,
            detector_name,
            flat_field,
            alien_mask
        )

    @h5_safe_load
    def get_detector_name(self) -> str:
        h5file = self.h5file
        key_path = ("_".join((self.sample_name, "1")) + ".1/measurement/")
        detector_names = []
        for key in h5file[key_path]:
            if key in self.authorised_detector_names:
                detector_names.append(key)
        if len(detector_names) == 0:
            raise ValueError(
                f"No detector name found in {self.authorised_detector_names}"
            )
        elif len(detector_names) > 1:
            raise ValueError(
                f"Several detector names found ({detector_names}).\n"
                "Not handled yet."
            )
        return detector_names[0]

    @h5_safe_load
    def load_det_calib_params(
            self,
            scan: int,
            sample_name: str = None
    ) -> dict:
        """
        Load the detector calibration parameters from the scan directly.
        Note that this will only provide the direct beam position, the
        sample-to-detector distance, and the pixel size. To get the
        tilt angles of the detector run the detector calibration
        notebook.
        """
        h5file = self.h5file
        if sample_name is None:
            sample_name = self.sample_name
        key_path = (
            "_".join((sample_name, str(scan)))
            + f".1/instrument/{self.detector_name}"
        )
        try:
            return {
                "cch1": float(h5file[key_path + "/beam_center_y"][()]),
                "cch2": float(h5file[key_path + "/beam_center_x"][()]),
                "pwidth1": float(h5file[key_path + "/y_pixel_size"][()]),
                "pwidth2": float(h5file[key_path + "/x_pixel_size"][()]),
                "distance": float(h5file[key_path + "/distance"][()]),
                "tiltazimuth": 0.,
                "tilt": 0.,
                "detrot": 0.,
            }
        except KeyError as exc:
            raise KeyError(
                f"key_path is wrong (key_path='{key_path}'). "
                "Are sample_name, scan number or detector name correct?"
            ) from exc

    @h5_safe_load
    def load_detector_shape(
            self,
            scan: int,
            sample_name: str = None,
    ) -> tuple:
        h5file = self.h5file
        if sample_name is None:
            sample_name = self.sample_name

        key_path = (
            "_".join((sample_name, str(scan)))
            + f".1/instrument/{self.detector_name}"
        )
        return h5file[f"{key_path}/dim_j"][()], h5file[f"{key_path}/dim_i"][()]

    @h5_safe_load
    def load_detector_data(
            self,
            scan: int,
            sample_name: str = None,
            roi: tuple[slice] = None,
            rocking_angle_binning: int = None,
            binning_method: str = "sum"
    ) -> np.ndarray:
        """
        Load the detector data.

        Args:
            scan (int): the scan number
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
        h5file = self.h5file
        if sample_name is None:
            sample_name = self.sample_name

        key_path = (
            "_".join((sample_name, str(scan)))
            + f".1/measurement/{self.detector_name}"
        )
        roi = self._check_roi(roi)
        try:
            if rocking_angle_binning:
                # we first apply the roi for axis1 and axis2
                data = h5file[key_path][(slice(None), roi[1], roi[2])]
                # But then we'll keep only the roi for axis0
                roi = (roi[0], slice(None), slice(None))
            else:
                data = h5file[key_path][roi]
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
            binning_method
        )

    @h5_safe_load
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

        angles = self.load_angles(
            key_path=f"{sample_name}_{scan}.1/instrument/positioners/"
        )

        formatted_angles = {
            key: angles[name] if angles.get(name) is not None else 0.
            for key, name in ID01Loader.angle_names.items()
        }

        if rocking_angle_binning:
            self.rocking_angle = self.get_rocking_angle(formatted_angles)

            formatted_angles[
                self.rocking_angle
            ] = self.bin_rocking_angle_values(
                formatted_angles[self.rocking_angle], rocking_angle_binning
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
        formatted_angles[
            self.rocking_angle
        ] = formatted_angles[self.rocking_angle][roi]

        return formatted_angles

    @h5_safe_load
    def load_energy(
            self,
            scan: int,
            sample_name: str = None
    ) -> float:
        if sample_name is None:
            sample_name = self.sample_name
        key_path = f"{sample_name}_{scan}.1/instrument/positioners/"
        try:
            return float(self.h5file[key_path + "mononrj"][()] * 1e3)
        except KeyError:
            warnings.warn(f"Energy not found at {key_path + 'mononrj'}. ")
            return None

    @h5_safe_load
    def get_array_shape(self, scan: int, sample_name: str = None) -> tuple:
        h5file = self.h5file
        if sample_name is None:
            sample_name = self.sample_name

        key_path = (
            "_".join((sample_name, str(scan)))
            + f".1/measurement/{self.detector_name}"
        )
        return h5file[key_path].shape

    @h5_safe_load
    def show_scan_attributes(
            self,
            scan: int,
            sample_name: str = None,
    ) -> None:
        """Print the attributes (keys) of a given scan number"""
        h5file = self.h5file
        if sample_name is None:
            sample_name = self.sample_name
        key_path = "_".join((sample_name, str(scan))) + ".1"
        print(h5file[key_path].keys())

    @h5_safe_load
    def load_measurement_parameters(
            self,
            scan: int,
            parameter_name: str,
            sample_name: str = None
    ) -> tuple:
        """Load the measurement parameters of the specified scan."""
        if sample_name is None:
            sample_name = self.sample_name
        key_path = "_".join(
             (sample_name, str(scan))
        ) + ".1/measurement"
        requested_mes_parameters = self.h5file[
            f"{key_path}/{parameter_name}"
        ][()]
        return requested_mes_parameters

    @h5_safe_load
    def load_instrument_parameters(
            self,
            scan: int,
            instrument_parameter: str,
            sample_name: str = None
    ) -> tuple:
        """Load the instrument parameters of the specified scan."""
        if sample_name is None:
            sample_name = self.sample_name
        key_path = "_".join(
             (sample_name, str(scan))
             ) + ".1/instrument"

        return self.h5file[key_path + "/" + instrument_parameter][()]

    @h5_safe_load
    def load_sample_parameters(
            self,
            scan: int,
            sample_name: str,
            sam_parameter: str
    ) -> tuple:
        """Load the sample parameters of the specified scan."""
        key_path = "_".join(
             (sample_name, str(scan))
             ) + ".1/sample"
        requested_parameters = self.h5file[
            key_path + "/" + sam_parameter
        ][()]
        return requested_parameters

    @h5_safe_load
    def load_plotselect_parameter(
            self,
            sample_name,
            scan,
            plot_parameter
    ) -> tuple:
        """Load the plotselect parameters of the specified scan."""

        key_path = "_".join((sample_name, str(scan))) + ".1/plotselect"
        requested_parameter = self.h5file[key_path + "/" + plot_parameter][()]

        return requested_parameter

    @h5_safe_load
    def get_start_time(self, scan: int, sample_name: str = None) -> str:
        """
        This functions will return the start time of the given scan.
        the returned object is of type datetime.datetime and can
        be easily manipulated arithmetically.
        """

        if sample_name is None:
            sample_name = self.sample_name

        key_path = "_".join((sample_name, str(scan))) + ".1/start_time"

        return dateutil.parser.isoparse(self.h5file[key_path][()])


def safe(func):
    def wrap(self, *args, **kwargs):
        with silx.io.open(self.experiment_file_path) as specfile:
            return func(self, specfile, *args, **kwargs)
    return wrap


# TODO: Implement roi parameter for detector, motors and mask methods
class SpecLoader(Loader):

    angle_names = {
        "sample_outofplane_angle": "eta",
        "sample_inplane_angle": "phi",
        "detector_outofplane_angle": "del",
        "detector_inplane_angle": "nu"
    }

    def __init__(
            self,
            experiment_file_path: str,
            detector_data_path: str,
            edf_file_template: str,
            detector_name: str,
            flat_field: str | np.ndarray = None,
            alien_mask: np.ndarray | str = None,
            **kwargs
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
            flat_field (str | np.ndarray, optional): flat field to
                account for the non homogeneous counting of the
                detector. Defaults to None.
            alien_mask (np.ndarray | str, optional): array to mask the
                aliens. Defaults to None.
        """
        super(SpecLoader, self).__init__(flat_field, alien_mask)
        self.experiment_file_path = experiment_file_path
        self.detector_data_path = detector_data_path
        self.edf_file_template = edf_file_template
        self.detector_name = detector_name

    @safe
    def load_detector_data(
            self,
            specfile: silx.io.specfile.SpecFile,
            scan: int,
            roi: tuple[slice] = None,
            rocking_angle_binning: int = None,
            binning_method: str = "sum"
    ):
        roi = self._check_roi(roi)

        frame_ids = specfile[f"{scan}.1/measurement/{self.detector_name}"][...]

        data = []

        template = self.detector_data_path + self.edf_file_template

        for frame_id in frame_ids:
            with fabio.open(template % frame_id) as edf_data:
                data.append(edf_data.data)

        return self.bin_flat_mask(
            np.asarray(data),
            roi,
            self.flat_field,
            self.alien_mask,
            rocking_angle_binning,
            binning_method
        )

    @safe
    def load_motor_positions(
            self,
            specfile: silx.io.specfile.SpecFile,
            scan: int,
            roi: tuple[slice] = None,
            rocking_angle_binning: int = None,
    ):

        if roi is None or len(roi) == 2:
            roi = slice(None)
        elif len(roi) == 3:
            roi = roi[0]

        positioners = specfile[f"{scan}.1/instrument/positioners"]

        angles = {key: None for key in SpecLoader.angle_names.keys()}
        for angle, name in SpecLoader.angle_names.items():
            try:
                angles[angle] = positioners[name][roi]
            except ValueError:
                angles[angle] = angles[angle] = positioners[name][()]

        self.rocking_angle = self.get_rocking_angle(angles)

        angles[self.rocking_angle] = self.bin_rocking_angle_values(
            angles[self.rocking_angle], rocking_angle_binning
        )
        if roi and rocking_angle_binning:
            angles[self.rocking_angle] = angles[self.rocking_angle][roi]
        return angles
