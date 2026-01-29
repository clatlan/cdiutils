"""
Loader for the Nanomax beamline at MAXIV.
See:
https://www.maxiv.lu.se/beamlines-accelerators/beamlines/nanomax/
"""

import numpy as np

from cdiutils.io.loader import H5TypeLoader, h5_safe_load


class NanoMAXLoader(H5TypeLoader):
    """
    Data loader for MAX IV NanoMAX beamline.

    Handles HDF5 files from NanoMAX beamline at MAX IV, supporting
    Eiger500k detector. NanoMAX has simpler file structure than
    other beamlines - no separate sample_name or scan number
    parameters needed.

    Attributes:
        angle_names: Mapping from canonical names to NanoMAX motor
            names:

            - ``sample_outofplane_angle`` -> ``"gontheta"``
            - ``sample_inplane_angle`` -> ``"gonphi"``
            - ``detector_outofplane_angle`` -> ``"delta"``
            - ``detector_inplane_angle`` -> ``"gamma"``

        authorised_detector_names: Tuple of supported detectors:
            ``("eiger500k",)``.

    Notes:
        Unlike other beamlines, NanoMAX stores all data in a single
        HDF5 file per measurement, eliminating need for sample_name
        or scan parameters in most methods.

    See Also:
        :class:`Loader` for factory method and base class
        documentation.
    """

    angle_names = {
        "sample_outofplane_angle": "gontheta",
        "sample_inplane_angle": "gonphi",
        "detector_outofplane_angle": "delta",
        "detector_inplane_angle": "gamma",
    }
    authorised_detector_names = ("eiger500k",)

    def __init__(
        self,
        experiment_file_path: str,
        detector_name: str = "eiger500k",
        flat_field: np.ndarray | str = None,
        alien_mask: np.ndarray | str = None,
        **kwargs,
    ) -> None:
        """
        Initialise NanoMAX data loader.

        Args:
            experiment_file_path: Path to HDF5 scan file. Contains
                all data and metadata for the measurement.
            detector_name: Detector identifier. Defaults to
                ``"eiger500k"``.
            flat_field: Flat-field correction array or path to
                .npy/.npz file.
            alien_mask: Bad pixel mask array or path.
            **kwargs: Additional parameters (reserved for future use).
        """
        super().__init__(
            experiment_file_path,
            detector_name=detector_name,
            flat_field=flat_field,
            alien_mask=alien_mask,
        )

    @h5_safe_load
    def load_detector_data(
        self,
        roi: tuple[slice] = None,
        rocking_angle_binning: int = None,
        binning_method: str = "sum",
    ) -> np.ndarray:
        """
        Load raw detector frames from NanoMAX HDF5 file.

        Retrieves 3D detector data array with optional ROI, binning,
        flat-field correction, and masking.

        Args:
            roi: Region of interest as tuple of slices or integers.
            rocking_angle_binning: Binning factor along rocking
                curve axis.
            binning_method: Binning operation (``"sum"``,
                ``"mean"``, or ``"max"``). Default ``"sum"``.

        Returns:
            Preprocessed detector data with shape
            ``(n_frames//binning, n_y, n_x)``.

        Raises:
            KeyError: If detector data path
                ``/entry/measurement/{detector}/frames`` not found in
                HDF5 file.
        """
        # Where to find the data.
        key_path = f"/entry/measurement/{self.detector_name}/frames"
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
            binning_method,
        )

    @h5_safe_load
    def load_motor_positions(
        self,
        roi: tuple[slice] = None,
        rocking_angle_binning: int = None,
    ) -> dict:
        """
        Load diffractometer motor angles from HDF5 file.

        Retrieves motor positions from post-scan snapshots and
        scanned rocking curve values, applying same ROI and binning
        as detector data.

        Args:
            roi: ROI tuple. Only first element (rocking curve axis)
                is used.
            rocking_angle_binning: Binning factor matching detector
                binning. Angles are averaged when binned.

        Returns:
            dict: Motor angles with canonical keys (see
            :attr:`angle_names` for NanoMAX-specific mapping). Scanned
            angle (gonphi or gontheta) is 1D array, others are
            scalars.
        """
        roi = self._check_roi(roi)
        roi = roi[0]

        key_path = "entry/snapshots/post_scan/"
        angles = {key: None for key in NanoMAXLoader.angle_names}

        for angle, name in NanoMAXLoader.angle_names.items():
            angles[angle] = self.h5file[key_path + name][()]

        # Take care of the rocking curve angle
        self.rocking_angle = rocking_angle = None
        for angle in ("gonphi", "gontheta"):
            if angle in self.h5file["entry/measurement"].keys():
                rocking_angle = angle
                if rocking_angle_binning:
                    rocking_angle_values = self.h5file["entry/measurement"][
                        angle
                    ][()]
                else:
                    rocking_angle_values = self.h5file["entry/measurement"][
                        angle
                    ][roi]
                # Find what generic angle (in-plane or out-of-plane) it
                # corresponds to.
                for angle, name in NanoMAXLoader.angle_names.items():
                    if name == rocking_angle:
                        self.rocking_angle = angle

        if self.rocking_angle is not None:
            angles[self.rocking_angle] = rocking_angle_values

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
        return self.h5file["entry/snapshots/post_scan/energy"][0]

    def load_det_calib_params(self) -> dict:
        return None

    @h5_safe_load
    def load_detector_shape(self, scan: int = None) -> tuple:
        return None
