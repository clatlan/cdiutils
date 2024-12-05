"""
Loader for the Nanomax beamline at MAXIV.
See:
https://www.maxiv.lu.se/beamlines-accelerators/beamlines/nanomax/
"""

import numpy as np

from cdiutils.load.loader import H5TypeLoader, h5_safe_load


class NanoMAXLoader(H5TypeLoader):
    """
    A class to handle loading/reading .h5 files that were created at the
    NanoMax beamline.
    This loader class does not need any of the 'sample_name' or
    'experiment_file_path' because NanoMAX data layering is rather
    simple.

    Args:
        experiment_file_path (str): path to the scan file.
        detector_name (str): name of the detector.
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
        "detector_inplane_angle": "gamma"
    }
    authorised_detector_names = ("eiger500k", )

    def __init__(
            self,
            experiment_file_path: str,
            detector_name: str = "eiger500k",
            flat_field: np.ndarray | str = None,
            alien_mask: np.ndarray | str = None,
            **kwargs
    ) -> None:
        """
        Initialise NanoMaxLoader with experiment data file path and
        detector information.

        Args:
            experiment_file_path (str): path to the scan file.
            detector_name (str): name of the detector.
            flat_field (np.ndarray | str, optional): flat field to
                account for the non homogeneous counting of the
                detector. Defaults to None.
            alien_mask (np.ndarray | str, optional): array to mask the
                aliens. Defaults to None.
        """
        super().__init__(
            experiment_file_path,
            detector_name=detector_name,
            flat_field=flat_field,
            alien_mask=alien_mask
        )

    @h5_safe_load
    def load_detector_data(
            self,
            roi: tuple[slice] = None,
            rocking_angle_binning: int = None,
            binning_method: str = "sum"
    ) -> np.ndarray:
        """
        Main method to load the detector data (collected intensity).

        Args:
            roi (tuple[slice], optional): the region of interest of the
                detector to load. Defaults to None.
            rocking_angle_binning (int, optional): whether to bin the data
                along the rocking curve axis. Defaults to None.
            binning_method (str, optional): the method employed for the
                binning. It can be sum or "mean". Defaults to "sum".

        Returns:
            np.ndarray: the detector data.
        """
        # The self.h5file is initialised by the @safe decorator.
        h5file = self.h5file

        # Where to find the data.
        key_path = (
            # "_".join((sample_name, str(scan)))
            f"/entry/measurement/{self.detector_name}/frames"
        )
        roi = self._check_roi(roi)
        try:
            if rocking_angle_binning:
                # we first apply the roi for axis1 and axis2
                data = h5file[key_path][(slice(None), roi[1], roi[2])]
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
            roi: tuple[slice] = None,
            rocking_angle_binning: int = None,
    ) -> dict:
        h5file = self.h5file

        if roi is None or len(roi) == 2:
            roi = slice(None)
        elif len(roi) == 3:
            roi = roi[0]

        key_path = "entry/snapshots/post_scan/"
        angles = {key: None for key in NanoMAXLoader.angle_names}

        for angle, name in NanoMAXLoader.angle_names.items():
            angles[angle] = h5file[key_path + name][()]

        # Take care of the rocking curve angle
        for angle in ("gonphi", "gontheta"):
            if angle in h5file["entry/measurement"].keys():
                rocking_angle_name = angle
                if rocking_angle_binning:
                    rocking_angle_values = h5file["entry/measurement"][angle][
                        ()
                    ]
                else:
                    rocking_angle_values = h5file["entry/measurement"][angle][
                        roi
                    ]
                # Find what generic angle (in-plane or out-of-plane) it
                # corresponds to.
                for angle, name in NanoMAXLoader.angle_names.items():
                    if name == rocking_angle_name:
                        rocking_angle = angle

        self.rocking_angle = rocking_angle
        angles[rocking_angle] = rocking_angle_values

        angles[self.rocking_angle] = self.bin_rocking_angle_values(
            angles[self.rocking_angle]
        )
        if roi and rocking_angle_binning:
            angles[self.rocking_angle] = angles[self.rocking_angle][roi]
        return angles

    @h5_safe_load
    def load_energy(self) -> float:
        """
        Load and return the energy used during the experiment.

        Args:
            scan (int): the scan number of the file to load the energy
                from.

        Returns:
            float: the energy value in keV.
        """
        h5file = self.h5file
        return h5file["entry/snapshots/post_scan/energy"][0]

    def load_det_calib_params(self) -> dict:
        return None
