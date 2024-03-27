from typing import Callable
import dateutil.parser
import numpy as np
import hdf5plugin
import silx.io.h5py_utils

from cdiutils.utils import CroppingHandler
from cdiutils.load import Loader


def safe(func: Callable) -> Callable:
    """A wrapper to safely load data in h5 file"""
    def wrap(self, *args, **kwargs):
        with silx.io.h5py_utils.File(self.experiment_file_path) as self.h5file:
            return func(self, *args, **kwargs)
    return wrap


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

    @safe
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

        if roi is None:
            roi = tuple(slice(None) for i in range(3))
        elif len(roi) == 2:
            roi = tuple([slice(None), roi[0], roi[1]])
        elif all(isinstance(e, int) for e in roi):
            roi = CroppingHandler.roi_list_to_slices(roi)

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

        if binning_along_axis0:
            original_dim0 = data.shape[0]
            nb_of_bins = original_dim0 // binning_along_axis0
            first_slices = nb_of_bins * binning_along_axis0
            last_slices = first_slices + original_dim0 % binning_along_axis0
            if binning_method == "sum":
                binned_data = [
                    np.sum(e, axis=0)
                    for e in np.array_split(data[:first_slices], nb_of_bins)
                ]
                if original_dim0 % binning_along_axis0 != 0:
                    binned_data.append(np.sum(data[last_slices:], axis=0))
                data = np.asarray(binned_data)

        if binning_along_axis0 and roi:
            data = data[roi]

        if self.flat_field is not None:
            data = data * self.flat_field[roi[1:]]

        if self.alien_mask is not None:
            data = data * self.alien_mask[roi[1:]]

        return data

    @safe
    def get_array_shape(self, scan: int, sample_name: str = None) -> tuple:
        h5file = self.h5file
        if sample_name is None:
            sample_name = self.sample_name

        key_path = (
            "_".join((sample_name, str(scan)))
            + f".1/measurement/{self.detector_name}"
        )
        return h5file[key_path].shape

    @safe
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
    @safe
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

    @safe
    def load_measurement_parameters(
            self,
            sample_name: str,
            scan: int,
            parameter_name: str
    ) -> tuple:
        """Load the measurement parameters of the specified scan."""
        key_path = "_".join(
             (sample_name, str(scan))
        ) + ".1/measurement"
        requested_mes_parameters = self.h5file[
            f"{key_path}/{parameter_name}"
        ][()]
        return requested_mes_parameters

    @safe
    def load_instrument_parameters(
            self,
            scan: int,
            sample_name: str,
            instrument_parameter: str
    ) -> tuple:
        """Load the instrument parameters of the specified scan."""
        key_path = "_".join(
             (sample_name, str(scan))
             ) + ".1/instrument"

        return self.h5file[key_path + "/" + instrument_parameter][()]

    @safe
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

    @safe
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

    @safe
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
