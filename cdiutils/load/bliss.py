import dateutil.parser
import numpy as np
import hdf5plugin

from cdiutils.load import Loader, h5_safe_load


class BlissLoader(Loader):
    """
    A class to handle loading/reading .h5 files that were created using
    Bliss on the ID01 beamline.
    """

    angle_names = {
        "sample_outofplane_angle": "eta",
        "sample_inplane_angle": "phi",
        "detector_outofplane_angle": "delta",
        "detector_inplane_angle": "nu"
    }

    def __init__(
            self,
            experiment_file_path: str,
            detector_name: str,
            sample_name: str = None,
            flat_field: np.ndarray | str = None,
            alien_mask: np.ndarray | str = None,
            **kwargs
    ) -> None:
        """
        Initialise BlissLoader with experiment data file path and
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
        super(BlissLoader, self).__init__(flat_field, alien_mask)
        self.experiment_file_path = experiment_file_path
        self.detector_name = detector_name
        self.sample_name = sample_name

    @h5_safe_load
    def load_detector_data(
            self,
            scan: int,
            sample_name: str = None,
            roi: tuple[slice] = None,
            binning_along_axis0: int = None,
            binning_method: str = "sum"
    ) -> np.ndarray:
        """Load the detector data of a given scan number."""

        h5file = self.h5file
        if sample_name is None:
            sample_name = self.sample_name

        key_path = (
            "_".join((sample_name, str(scan)))
            + f".1/measurement/{self.detector_name}"
        )

        roi = self._check_roi(roi)

        try:
            if binning_along_axis0:
                data = h5file[key_path][()]
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
            binning_along_axis0,
            binning_method
        )

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

    # @silx.io.h5py_utils.retry()
    @h5_safe_load
    def load_motor_positions(
            self,
            scan: int,
            sample_name: str = None,
            roi: tuple[slice] = None,
            binning_along_axis0: int = None,
            binning_method: str = "mean"
    ) -> dict:
        """
        Load the motor positions and return it as a dict of:
        - sample out of plane angle
        - sample in plane angle
        - detector out of plane angle
        - detector in plane angle
        """

        if sample_name is None:
            sample_name = self.sample_name

        key_path = "_".join(
                (sample_name, str(scan))
        ) + ".1/instrument/positioners/"

        if roi is None or len(roi) == 2:
            roi = slice(None)
        elif len(roi) == 3:
            roi = roi[0]

        angles = {key: None for key in BlissLoader.angle_names.keys()}

        for angle, name in BlissLoader.angle_names.items():
            if binning_along_axis0:
                angles[angle] = self.h5file[key_path + name][()]
            else:
                try:
                    angles[angle] = self.h5file[key_path + name][roi]
                except ValueError:
                    angles[angle] = self.h5file[key_path + name][()]

        if binning_along_axis0:
            original_dim0 = angles["sample_outofplane_angle"].shape[0]
            nb_of_bins = original_dim0 // binning_along_axis0
            first_slices = nb_of_bins * binning_along_axis0
            last_slices = first_slices + original_dim0 % binning_along_axis0
            if binning_method == "mean":
                if original_dim0 % binning_along_axis0 != 0:
                    binned_sample_outofplane_angle = [
                        np.mean(e, axis=0)
                        for e in np.split(
                                angles["sample_outofplane_angle"][
                                    :first_slices
                                ],
                                nb_of_bins
                            )
                    ]
                    binned_sample_outofplane_angle.append(
                        np.mean(
                            angles["sample_outofplane_angle"][last_slices-1:],
                            axis=0
                        )
                    )
                else:
                    binned_sample_outofplane_angle = [
                        np.mean(e, axis=0)
                        for e in np.split(
                                angles["sample_outofplane_angle"],
                                nb_of_bins
                            )
                    ]
                angles["sample_outofplane_angle"] = np.asarray(
                    binned_sample_outofplane_angle
                )
        if binning_along_axis0 and roi:
            for name, value in angles.items():
                try:
                    angles[name] = value[roi]
                except IndexError:
                    # note that it is not the same error as above
                    continue
        return angles

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
