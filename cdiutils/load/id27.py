import numpy as np

from cdiutils.load import Loader, h5_safe_load
from cdiutils.utils import wavelength_to_energy
import warnings


class ID27Loader(Loader):
    """
    A class to handle loading/reading .h5 files that were created using
    Bliss on the ID27 beamline.
    """

    angle_names = {
        "sample_outofplane_angle": None,
        "sample_inplane_angle": "nath",
        "detector_outofplane_angle": None,
        "detector_inplane_angle": None
    }

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
        Initialise ID27Loader with experiment data file path and
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
        super().__init__(flat_field, alien_mask)
        self.experiment_file_path = experiment_file_path
        self.sample_name = sample_name
        if detector_name is None:
            if sample_name is not None:
                self.detector_name = self.get_detector_name()
                print(
                    "Detector name automatically found "
                    f"('{self.detector_name}')."
                )
            else:
                print(
                    "detector_name is not provided, cannot automatically find "
                    "it since sample_name is not provided either.\n"
                    "Will set detector_name to 'eiger'."
                )
                self.detector_name = "eiger"
        else:
            self.detector_name = detector_name

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

        if self.detector_name in ("eiger", "eiger9m", "e9m"):
            # Must apply mask on ID21 Eiger data
            mask = self.get_mask(
                channel=data.shape[0],
                detector_name="e9m",
                roi=(slice(None), roi[1], roi[2])
            )
            data = data * np.where(mask, 0, 1)

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

        key_path = f"{sample_name}_{scan}.1/instrument/positioners/"

        if roi is None or len(roi) == 2:
            roi = slice(None)
        elif len(roi) == 3:
            roi = roi[0]

        angles = {key: 0 for key in ID27Loader.angle_names}

        if rocking_angle_binning:
            angles["sample_inplane_angle"] = self.h5file[key_path + "nath"][()]
        else:
            try:
                angles["sample_inplane_angle"] = self.h5file[
                    key_path + "nath"
                ][roi]
            except ValueError:
                angles["sample_inplane_angle"] = self.h5file[
                    key_path + "nath"
                ][()]

        self.rocking_angle = "sample_inplane_angle"
        angles[self.rocking_angle] = self.bin_rocking_angle_values(
            angles[self.rocking_angle], rocking_angle_binning
        )
        if roi and rocking_angle_binning:
            angles[self.rocking_angle] = angles[self.rocking_angle][roi]
        return angles

    @h5_safe_load
    def get_detector_name(self) -> str:
        h5file = self.h5file
        key_path = ("_".join((self.sample_name, "1")) + ".1/measurement/")
        detector_names = []
        authorised_names = ("eiger")
        for key in h5file[key_path]:
            if key in authorised_names:
                detector_names.append(key)
        if len(detector_names) == 0:
            raise ValueError(f"No detector name found in {authorised_names}")
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

        key_path = f"{sample_name}_{scan}.1/instrument/calibration/"

        params = {}
        # First retrieve the data stored in the metadata
        try:
            params["distance"] = float(h5file[key_path + "distance"][()])

            if self.detector_name in ("eiger", "eiger9m", "e9m"):
                params["pwidth1"], params["pwidth2"] = 75e-6, 75e-6

            # Take care of the direct beam position
            center = h5file[key_path + "center"][()]
            center = tuple(
                float(c)
                for c in center.decode("utf-8").strip("()").split(", ")
            )
            params["cch1"], params["cch2"] = -center[0], center[1]

            # Now correct the distance, cch1, cch2 with eigx, eigy, eigz
            key_path = f"{sample_name}_{scan}.1/instrument/positioners/"
            params["cch1"] += (
                (float(self.h5file[key_path + "eigz"][()]) -5.0647) * 1e-3  # to m
                // 75e-6  # pixel size in metres
            )
            params["cch2"] += (
                float(self.h5file[key_path + "eigy"][()]) * 1e-3  # to m
                // 75e-6  # pixel size in metres
            )
            params["distance"] += float(self.h5file[key_path + "eigx"][()]) - 231.92
            params["distance"] *= 1e-3  # Convert from mm to metres

        except KeyError as e:
            raise KeyError(
                f"key_path is wrong (key_path='{key_path}'). "
                "Are sample_name, scan number or detector name correct?"
            ) from e

        return params

    def load_detector_shape(
            self,
            scan: int,
            sample_name: str = None,
    ) -> tuple:
        if sample_name is None:
            sample_name = self.sample_name
        h5file = self.h5file
        if self.detector_name in ("eiger", "eiger9m", "e9m"):
            shape = (3262, 3108)
            key_path = f"{sample_name}_{scan}.1/instrument/eiger/acq_nb_frames"
            try:
                return (int(h5file[key_path][()]), ) + shape
            except KeyError:
                print("Could not load original detector data shape.")
        return None

    @h5_safe_load
    def load_energy(
            self,
            scan: int,
            sample_name: str = None
    ) -> float:
        h5file = self.h5file
        if sample_name is None:
            sample_name = self.sample_name

        key_path = f"{sample_name}_{scan}.1/instrument/calibration/"
        try:
            # Convert from angstrom to m
            return wavelength_to_energy(
                float(h5file[key_path + "wavelength"][()]) * 1e-10
            )
        except KeyError:
            warnings.warn(f"Energy not found at {key_path + 'wavelength'}. ")
            return None