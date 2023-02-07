from typing import Optional, Union, Callable
import numpy as np
import os
import silx.io.h5py_utils


def safe(func: Callable) -> Callable:
    """A wrapper to safely load data in h5 file"""
    def wrap(self, *args, **kwargs):
        with silx.io.h5py_utils.File(self.experiment_file_path) as self.h5file:
            return func(self, *args, **kwargs)
    return wrap


class BlissLoader():
    """
    A class to handle loading/reading .h5 files that were created using
    Bliss on the ID01 beamline.
    """
    def __init__(
            self,
            experiment_file_path: str,
            detector_name: str="flexible",
            sample_name: Optional[str]=None,
            flatfield: Union[np.ndarray, str]=None
        ):
        self.experiment_file_path = experiment_file_path
        self.detector_name = detector_name
        self.sample_name = sample_name
        self.h5file = None

        self.experiment_root_directory = os.path.dirname(experiment_file_path)

        if isinstance(flatfield, str) and flatfield.endswith(".npz"):
            self.flatfield = np.load(flatfield)["arr_0"]
        elif isinstance(flatfield, np.ndarray):
            self.flatfield=flatfield
        elif flatfield is None:
            self.flatfield = None
        else:
            raise ValueError(
                "[ERROR] wrong value for flatfield parameter, provide a path, "
                "np.ndarray or leave it to None"
            )

    @safe
    def load_detector_data(
            self,
            scan: int,
            sample_name: Optional[str]=None,
            binning_along_axis0: Optional[int]=None,
            binning_method: Optional[str]="sum"
    ) -> np.ndarray:
        """Load the detector data of a given scan number."""

        h5file = self.h5file
        if sample_name is None:
            sample_name = self.sample_name

        key_path = "_".join((sample_name, str(scan))) + ".1"
        if self.detector_name == "flexible":
            try:
                data = h5file[key_path + "/measurement/mpx1x4"][()]
            except KeyError:
                data = h5file[key_path + "/measurement/mpxgaas"][()]
        else:
            data = h5file[key_path + f"/measurement/{self.detector_name}"][()]
        if not self.flatfield is None:
            data = data * self.flatfield
        if binning_along_axis0:
            original_dim0 = data.shape[0]
            nb_of_bins = original_dim0 // binning_along_axis0
            last_slices = original_dim0 % binning_along_axis0
            if binning_method == "sum":
                binned_data = [
                    np.sum(e, axis=0)
                    for e in np.array_split(data[:-last_slices], nb_of_bins)
                ]
                if last_slices != 0:
                    binned_data.append(np.sum(data[last_slices:], axis=0))
                data = np.asarray(binned_data)
        return data

    @safe
    def show_scan_attributes(
            self,
            scan: int,
            sample_name: Optional[str]=None,
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
            sample_name: Optional[str]=None,
            binning_along_axis0: Optional[int]=None,
            binning_method: Optional[str]="mean"
    ) -> tuple:
        """
        Load the motor positions and return it in the following order:
        - sample out of plane angle
        - sample in plane angle
        - detector in plane angle
        - detector out of plane angle
        """

        if sample_name is None:
            sample_name = self.sample_name

        key_path = "_".join(
                (sample_name, str(scan))
        ) + ".1/instrument/positioners"

        nu = self.h5file[key_path + "/nu"][()]
        delta = self.h5file[key_path + "/delta"][()]
        eta = self.h5file[key_path + "/eta"][()]
        phi = self.h5file[key_path + "/phi"][()]

        if binning_along_axis0:
            original_dim0 = eta.shape[0]
            nb_of_bins = original_dim0 // binning_along_axis0
            last_slices = original_dim0 % binning_along_axis0
            if binning_method == "mean":
                if last_slices != 0:
                    binned_eta = [
                        e
                        for e in np.split(eta[:-last_slices], nb_of_bins)
                    ]
                    binned_eta.append(np.mean(eta[last_slices:], axis=0))
                else:
                    binned_eta = [
                        e
                        for e in np.split(eta, nb_of_bins)
                    ]
                eta = np.asarray(binned_eta)

        return eta, phi, nu, delta

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
            ins_parameter: str
    ) -> tuple:
        """Load the instrument parameters of the specified scan."""
        key_path = "_".join(
             (sample_name, str(scan))
             ) + ".1/instrument"
        requested_parameters = self.h5file[
            key_path + "/" + ins_parameter
        ][()]
        return requested_parameters

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
        h5file = self.h5file
        key_path = "_".join(
             (sample_name, str(scan))
             ) + ".1/plotselect"
        requested_parameter = h5file[key_path + "/" + plot_parameter][()]
        return requested_parameter

    @staticmethod
    def get_mask(
            channel: Optional[int],
            detector_name: str="Maxipix"
    ) -> np.ndarray:
        """Load the mask of the given detector_name."""

        if detector_name in ("maxipix", "Maxipix", "mpxgaas"):
            mask = np.zeros(shape=(516, 516))
            mask[:, 255:261] = 1
            mask[255:261, :] = 1

        elif detector_name in ("Eiger2M", "eiger2m", "eiger2M", "Eiger2m"):
            mask = np.zeros(shape=(2164, 1030))
            mask[:, 255:259] = 1
            mask[:, 513:517] = 1
            mask[:, 771:775] = 1
            mask[0:257, 72:80] = 1
            mask[255:259, :] = 1
            mask[511:552, :] = 1
            mask[804:809, :] = 1
            mask[1061:1102, :] = 1
            mask[1355:1359, :] = 1
            mask[1611:1652, :] = 1
            mask[1905:1909, :] = 1
            mask[1248:1290, 478] = 1
            mask[1214:1298, 481] = 1
            mask[1649:1910, 620:628] = 1
        else:
            raise ValueError("Unknown detector_name")
        if channel:
            return np.repeat(mask[np.newaxis, :, :,], channel, axis=0)
        return mask
