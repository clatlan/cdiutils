"""
    Loader for the Nanomax beamlien at MAXIV.
    See:
    https://www.maxiv.lu.se/beamlines-accelerators/beamlines/nanomax/
"""

import numpy as np

from cdiutils.load import Loader
from cdiutils.load.bliss import safe


class NanoMaxLoader(Loader):
    """
    A class to handle loading/reading .h5 files that were created using
    on the NanoMax beamline.

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
            detector_name: str = "eiger500k",
            sample_name: str = None,
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
            detector_name (str): name of the detector.
            sample_name (str, optional): name of the sample. Defaults
                to None.
            flat_field (np.ndarray | str, optional): flat field to
                account for the non homogeneous counting of the
                detector. Defaults to None.
            alien_mask (np.ndarray | str, optional): array to mask the
                aliens. Defaults to None.
        """
        super(NanoMaxLoader, self).__init__(flat_field, alien_mask)
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
            binnig_method: str = "sum"
    ) -> np.ndarray:
        """
        Main method to load the detector data (collected intensity).

        Args:
            scan (int): the scan number you want to load the data from.
            sample_name (str, optional): the sample name for this scan.
                Only used if self.sample_name is None. Defaults to None.
            roi (tuple[slice], optional): the region of interest of the
                detector to load. Defaults to None.
            binning_along_axis0 (int, optional): whether to bin the data
                along the rocking curve axis. Defaults to None.
            binnig_method (str, optional): the method employed for the
                binning. It can be sum or "mean". Defaults to "sum".

        Returns:
            np.ndarray: the detector data.
        """
        # The self.h5file is initialised by the @safe decorator.
        h5file = self.h5file
        if sample_name is None:
            sample_name = self.sample_name

        # Where to find the data.
        key_path = (
            "_".join((sample_name, str(scan)))
            + f"/entry/measurement/{self.detector_name}"
        )

        roi = self._check_roi(roi)

