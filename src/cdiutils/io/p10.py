import numpy as np
import silx.io.h5py_utils

from cdiutils.io import Loader


class P10Loader(Loader):
    """
    Data loader for PETRA III P10 beamline.

    Handles HDF5 master files and .fio motor position files from P10
    beamline at PETRA III, supporting Eiger4M and Eiger500k
    detectors. Data is organised in separate directories per scan.

    Attributes:
        angle_names: Mapping from canonical names to P10 motor names:

            - ``sample_outofplane_angle`` -> ``"om"`` (EH1) or
              ``"samth"`` (EH2)
            - ``sample_inplane_angle`` -> ``"phi"`` (EH1 only)
            - ``detector_outofplane_angle`` -> ``"del"`` (EH1) or
              ``"e2_t02"`` (EH2)
            - ``detector_inplane_angle`` -> ``"gam"`` (EH1 only)

        authorised_detector_names: Tuple of supported detectors:
            ``("eiger4m", "e2500")``.

    Notes:
        EH2 (experimental hutch 2) has different motor names and lacks
        in-plane rotation stages. Specify ``hutch="EH2"`` during
        initialisation for EH2 experiments.

    See Also:
        :class:`Loader` for factory method and base class
        documentation.
    """

    angle_names = {
        "sample_outofplane_angle": "om",
        "sample_inplane_angle": "phi",
        "detector_outofplane_angle": "del",
        "detector_inplane_angle": "gam",
    }
    authorised_detector_names = ("eiger4m", "e2500")

    def __init__(
        self,
        experiment_data_dir_path: str,
        scan: int = None,
        sample_name: str = None,
        detector_name: str = None,
        flat_field: np.ndarray | str = None,
        alien_mask: np.ndarray | str = None,
        hutch: str = "EH1",
        **kwargs,
    ) -> None:
        """
        Initialise P10 data loader.

        Args:
            experiment_data_dir_path: Root data directory containing
                scan subdirectories. Expected structure:
                ``{root}/{sample}_{scan:05d}/{detector}/``.
            scan: Scan number (5-digit zero-padded in file paths).
            sample_name: Sample identifier matching directory names.
            detector_name: Detector identifier (``"eiger4m"`` or
                ``"e2500"``). If None, defaults to ``"e4m"``.
            flat_field: Flat-field correction array or path to
                .npy/.npz file.
            alien_mask: Bad pixel mask array or path.
            hutch: Experimental hutch (``"EH1"`` or ``"EH2"``).
                Affects motor name mappings. Defaults to ``"EH1"``.
            **kwargs: Additional parameters (reserved for future use).

        Raises:
            ValueError: If ``hutch`` is not ``"EH1"`` or ``"EH2"``.
        """
        self.experiment_data_dir_path = experiment_data_dir_path
        super().__init__(scan, sample_name, flat_field, alien_mask)
        self.detector_name = detector_name
        if self.detector_name is None:
            self.detector_name = "e4m"

        if hutch.lower() == "eh2":
            self.angle_names["sample_outofplane_angle"] = "samth"
            self.angle_names["detector_outofplane_angle"] = "e2_t02"
            self.angle_names["sample_inplane_angle"] = None
            self.angle_names["detector_inplane_angle"] = None
        elif hutch.lower() != "eh1":
            raise ValueError(
                f"Hutch name (hutch={hutch}) is not valid. Can be 'EH1' or "
                "'EH2'."
            )

    def _get_file_path(
        self, scan: int, sample_name: str, data_type: str = "detector_data"
    ) -> str:
        """
        Construct file path for detector data or motor positions.

        Args:
            scan: Scan number (zero-padded to 5 digits).
            sample_name: Sample name.
            data_type: Either ``"detector_data"`` (HDF5 master file)
                or ``"motor_positions"`` (.fio file).

        Returns:
            Absolute path to requested file.

        Raises:
            ValueError: If ``data_type`` is not recognised.
        """
        if data_type == "detector_data":
            return (
                self.experiment_data_dir_path
                + f"/{sample_name}_{scan:05d}"
                + f"/{self.detector_name}"
                + f"/{sample_name}_{scan:05d}_master.h5"
            )
        if data_type == "motor_positions":
            return (
                self.experiment_data_dir_path
                + f"/{sample_name}_{scan:05d}"
                + f"/{sample_name}_{scan:05d}.fio"
            )
        raise ValueError(
            f"data_type {data_type} is not valid. Must be either detector_data"
            " or motor_positions."
        )

    def load_detector_data(
        self,
        scan: int = None,
        sample_name: str = None,
        roi: tuple[slice] = None,
        rocking_angle_binning: int = None,
        binning_method: str = "sum",
    ) -> None:
        """
        Load raw detector frames from P10 HDF5 file.

        Retrieves 3D detector data array with optional ROI, binning,
        flat-field correction, and masking. Automatically applies
        detector chip gap mask for Eiger detectors.

        Args:
            scan: Scan number. If None, uses ``self.scan``.
            sample_name: Sample name. If None, uses
                ``self.sample_name``.
            roi: Region of interest as tuple of slices or integers.
            rocking_angle_binning: Binning factor along rocking
                curve axis.
            binning_method: Binning operation (``"sum"``,
                ``"mean"``, or ``"max"``). Default ``"sum"``.

        Returns:
            Preprocessed detector data with shape
            ``(n_frames//binning, n_y, n_x)``.
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)

        path = self._get_file_path(scan, sample_name)
        key_path = "entry/data/data_000001"

        roi = self._check_roi(roi)

        with silx.io.h5py_utils.File(path) as h5file:
            if rocking_angle_binning:
                data = h5file[key_path][()]
            else:
                data = h5file[key_path][roi]

        data = self.bin_flat_mask(
            data,
            roi,
            self.flat_field,
            self.alien_mask,
            rocking_angle_binning,
            binning_method,
        )

        # Must apply mask on P10 Eiger data
        mask = self.get_mask(
            channel=data.shape[0],
            detector_name=self.detector_name,
            roi=(slice(None), roi[1], roi[2]),
        )
        data = data * np.where(mask, 0, 1)

        return data

    def load_motor_positions(
        self,
        scan: int = None,
        sample_name: str = None,
        roi: tuple[slice] = None,
        rocking_angle_binning: int = None,
    ) -> None:
        """
        Load diffractometer motor angles from .fio file.

        Parses P10's text-based .fio files to extract motor
        positions, applying same ROI and binning as detector data.

        Args:
            scan: Scan number. If None, uses ``self.scan``.
            sample_name: Sample name. If None, uses
                ``self.sample_name``.
            roi: ROI tuple. Only first element (rocking curve axis)
                is used.
            rocking_angle_binning: Binning factor matching detector
                binning. Angles are averaged when binned.

        Returns:
            dict: Motor angles with canonical keys (see
            :attr:`angle_names` for P10-specific mapping). Values
            are scalars (fixed motor) or 1D arrays (scanned motor).
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)

        path = self._get_file_path(
            scan, sample_name, data_type="motor_positions"
        )
        if roi is None or len(roi) == 2:
            roi = slice(None)
        elif len(roi) == 3:
            roi = roi[0]

        angles = {name: None for name in self.angle_names.values()}

        rocking_angle_values = []

        with open(path, encoding="utf8") as fio_file:
            lines = fio_file.readlines()
            rocking_angle_column = None
            for line in lines:
                line = line.strip()
                words = line.split()

                for name in self.angle_names.values():
                    if name in words:
                        if "=" in words:
                            angles[name] = float(words[-1])
                        if "Col" in words and rocking_angle_column is None:
                            rocking_angle_column = int(words[1]) - 1
                            rocking_angle = words[2]

            for line in lines:
                line = line.strip()
                words = line.split()

                # check if the first word is numeric, if True the line
                # contains motor position values
                # if words[0].replace(".", "", 1).isdigit():
                if words[0].replace(".", "").replace("-", "").isnumeric():
                    rocking_angle_values.append(
                        float(words[rocking_angle_column])
                    )
                    if "e2_t02" in angles:
                        # This means that 'e2_t02' must be used as the
                        # detector out-of-plane angle.
                        angles["e2_t02"] = float(words[1])
        self.rocking_angle = rocking_angle
        angles[rocking_angle] = np.array(rocking_angle_values)
        for name in angles:
            if angles[name] is None:
                angles[name] = 0

        angles[self.rocking_angle] = self.bin_rocking_angle_values(
            angles[self.rocking_angle], rocking_angle_binning
        )
        if roi:
            angles[rocking_angle] = angles[rocking_angle][roi]

        return {
            angle: angles[name] for angle, name in self.angle_names.items()
        }

    def load_energy(self, scan: int = None, sample_name: str = None) -> float:
        """
        Load X-ray beam energy from .fio file.

        Args:
            scan: Scan number. If None, uses ``self.scan``.
            sample_name: Sample name. If None, uses
                ``self.sample_name``.

        Returns:
            Beam energy in eV from ``fmbenergy`` motor, or None if
            not found.
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        path = self._get_file_path(
            scan, sample_name, data_type="motor_positions"
        )
        with open(path, encoding="utf8") as fio_file:
            lines = fio_file.readlines()
            for line in lines:
                line = line.strip()
                words = line.split()
                if "fmbenergy" in words:
                    return float(words[-1])
        return None

    def load_det_calib_params(self) -> dict:
        """
        Load detector calibration parameters.

        Returns:
            None. P10 does not store calibration in scan files.
            Calibration must be determined separately.
        """
        return None

    def load_detector_shape(self, scan: int = None) -> tuple:
        """
        Load detector shape.

        Returns:
            None. Shape is determined from data files directly.
        """
        return None
