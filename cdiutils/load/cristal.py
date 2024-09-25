"""
Loader for the Crisal beamline at SOLEIL.
"""


import numpy as np

from cdiutils.load import Loader, h5_safe_load


class CristalLoader(Loader):
    """
    A class to handle loading/reading .h5 files that were created at the
    Cristal beamline.

    Args:
        experiment_file_path (str): path to the master file
            used for the experiment.
        flat_field (np.ndarray | str, optional): flat field to
            account for the non homogeneous counting of the
            detector. Defaults to None.
        alien_mask (np.ndarray | str, optional): array to mask the
            aliens. Defaults to None.
    """

    angle_names = {
        "sample_outofplane_angle": "i06-c-c07-ex-mg_omega",
        "sample_inplane_angle": "i06-c-c07-ex-mg_phi",
        "detector_outofplane_angle": "Diffractometer/i06-c-c07-ex-dif-delta",
        "detector_inplane_angle": "Diffractometer/i06-c-c07-ex-dif-gamma"
    }

    def __init__(
            self,
            experiment_file_path: str,
            flat_field: np.ndarray | str = None,
            alien_mask: np.ndarray | str = None,
            **kwargs
    ) -> None:
        """
        Initialise NanoMaxLoader with experiment data file path and
        detector information.

        Args:
            experiment_file_path (str): path to the master file
                used for the experiment.
            sample_name (str, optional): name of the sample. Defaults
                to None.
            flat_field (np.ndarray | str, optional): flat field to
                account for the non homogeneous counting of the
                detector. Defaults to None.
            alien_mask (np.ndarray | str, optional): array to mask the
                aliens. Defaults to None.
        """
        super(CristalLoader, self).__init__(flat_field, alien_mask)
        self.experiment_file_path = experiment_file_path

    @h5_safe_load
    def load_detector_data(
            self,
            scan: int,
            roi: tuple[slice] = None,
            binning_along_axis0: int = None,
            binning_method: str = "sum"
    ) -> np.ndarray:
        """Load the detector data of a given scan number."""

        h5file = self.h5file

        # First, find the key that corresponds to the detector data
        for k in h5file[f"exp_{scan:04d}/scan_data"]:
            if h5file[f"exp_{scan:04d}/scan_data"][k].ndim == 3:
                data_key = k

        key_path = f"exp_{scan:04d}/scan_data/{data_key}"

        roi = self._check_roi(roi)

        try:
            if binning_along_axis0:
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
            binning_along_axis0,
            binning_method
        )

    @h5_safe_load
    def load_motor_positions(
            self,
            scan: int,
            roi: tuple[slice] = None,
            binning_along_axis0: int = None,
            binning_method: str = "mean"
    ) -> dict:

        h5file = self.h5file

        key_path = f"exp_{scan:04d}"
        key_path_template = key_path + "/CRISTAL/{}/position"

        angles = {}
        for angle, name in CristalLoader.angle_names.items():
            angles[angle] = h5file[key_path_template.format(name)][()]

        # Get the motor name used for the rocking curve
        rocking_motor = h5file[
            key_path + "/scan_config/name"
        ][()].decode("utf-8")[-4:]  # here, only the last 3 char are needed
        # Get the associated motor values
        rocking_values = h5file[key_path + "/scan_data/actuator_1_1"][()]

        # replace the value for the rocking angle by the array of values
        for angle, name in CristalLoader.angle_names.items():
            if name.endswith(rocking_motor):
                angles[angle] = rocking_values
        if binning_along_axis0:
            raise ValueError("Not implemented yet.")

        return angles

    @h5_safe_load
    def load_energy(self, scan: int) -> float:
        h5file = self.h5file
        key_path = f"exp_{scan:04d}/CRISTAL/Monochromator/energy"
        return h5file[key_path][0] * 1e3
