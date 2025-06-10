"""Loader for the Cristal beamline at SOLEIL."""


import numpy as np

from cdiutils.io.loader import H5TypeLoader, h5_safe_load


class CristalLoader(H5TypeLoader):
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
    authorised_detector_names = ("maxipix", )

    def __init__(
            self,
            experiment_file_path: str,
            scan: int = None,
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
            scan (int, optional): scan number. Defaults to None.
            flat_field (np.ndarray | str, optional): flat field to
                account for the non homogeneous counting of the
                detector. Defaults to None.
            alien_mask (np.ndarray | str, optional): array to mask the
                aliens. Defaults to None.
        """
        super().__init__(
            experiment_file_path,
            scan=scan,
            flat_field=flat_field,
            alien_mask=alien_mask
        )

    @h5_safe_load
    def load_detector_data(
            self,
            scan: int = None,
            roi: tuple[slice] = None,
            rocking_angle_binning: int = None,
            binning_method: str = "sum"
    ) -> np.ndarray:
        scan, _ = self._check_scan_sample(scan)

        # First, find the key that corresponds to the detector data
        for k in self.h5file[f"exp_{scan:04d}/scan_data"]:
            if self.h5file[f"exp_{scan:04d}/scan_data"][k].ndim == 3:
                data_key = k

        key_path = f"exp_{scan:04d}/scan_data/{data_key}"

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
            binning_method
        )

    @h5_safe_load
    def load_motor_positions(
            self,
            scan: int = None,
            roi: tuple[slice] = None,
            rocking_angle_binning: int = None,
    ) -> dict:
        scan, _ = self._check_scan_sample(scan)

        key_path = f"exp_{scan:04d}"
        key_path_template = key_path + "/CRISTAL/{}/position"

        angles = {}
        for angle, name in CristalLoader.angle_names.items():
            angles[angle] = float(
                self.h5file[key_path_template.format(name)][()]
            )

        # get the full motor name used for the rocking curve
        full_rocking_motor_name = self.h5file[
            key_path + "/scan_config/name"
        ][()].decode("utf-8")

        # extract the important part (omega or phi)
        if "omega" in full_rocking_motor_name.lower():
            rocking_motor_name = "omega"
        elif "phi" in full_rocking_motor_name.lower():
            rocking_motor_name = "phi"
        else:
            # If we can't determine, use the original approach
            rocking_motor_name = full_rocking_motor_name[-3:]

        # Get the associated motor values
        rocking_values = self.h5file[key_path + "/scan_data/actuator_1_1"][()]

        if rocking_angle_binning:
            rocking_values = self.bin_rocking_angle_values(
                rocking_values, rocking_angle_binning
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
        rocking_values = rocking_values[roi]

        # replace the value for the rocking angle by the array of values
        rocking_angle_found = False
        for angle, name in CristalLoader.angle_names.items():
            # check if the standardized name contains the key part (omega/phi)
            if rocking_motor_name in name.lower():
                angles[angle] = rocking_values
                rocking_angle_found = True
                break

        if not rocking_angle_found:
            raise ValueError(
                "Could not identify rocking angle from name: "
                f"{full_rocking_motor_name}"
            )
        return angles

    @h5_safe_load
    def load_energy(self, scan: int = None) -> float:
        scan, _ = self._check_scan_sample(scan)
        key_path = f"exp_{scan:04d}/CRISTAL/Monochromator/energy"
        return self.h5file[key_path][0] * 1e3

    @h5_safe_load
    def load_det_calib_params(self, scan: int = None) -> dict:
        scan, _ = self._check_scan_sample(scan)
        return None

    @h5_safe_load
    def load_detector_shape(self, scan: int) -> tuple:
        return None
