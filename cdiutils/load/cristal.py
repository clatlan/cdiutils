"""
Loader for the Crisal beamline at SOLEIL.
"""


import numpy as np

from cdiutils.load import Loader, h5_safe_load


class Cristal(Loader):
    """
    A class to handle loading/reading .h5 files that were created at the
    Cristal beamline.

    Args:
        experiment_file_path (str): path to the master file
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

    angle_names = {
        "sample_outofplane_angle": "gontheta",
        "sample_inplane_angle": "gonphi",
        "detector_outofplane_angle": "delta",
        "detector_inplane_angle": "nu"
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
        super(Cristal, self).__init__(flat_field, alien_mask)
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
