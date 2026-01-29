import warnings

import dateutil.parser
import fabio
import numpy as np
import silx.io

from cdiutils.io.loader import H5TypeLoader, Loader, h5_safe_load


class ID01Loader(H5TypeLoader):
    """
    Data loader for ESRF ID01 beamline (BLISS acquisition).

    Handles HDF5 master files produced by BLISS control software at
    ID01, supporting Maxipix and Eiger2M detectors. Provides detector
    calibration, motor angles, and beam energy from scan metadata.

    This loader is for modern BLISS-based experiments. For legacy SPEC
    format data (pre-2020), use :class:`SpecLoader` instead.

    Attributes:
        angle_names: Mapping from canonical names to ID01 motor names:

            - ``sample_outofplane_angle`` -> ``"eta"``
            - ``sample_inplane_angle`` -> ``"phi"``
            - ``detector_outofplane_angle`` -> ``"delta"``
            - ``detector_inplane_angle`` -> ``"nu"``

        authorised_detector_names: Tuple of supported detectors:
            ``("mpxgaas", "mpx1x4", "eiger2M")``.

    Examples:
        Basic usage with factory pattern:

        >>> from cdiutils.io import Loader
        >>> loader = Loader.from_setup(
        ...     beamline_setup="id01",
        ...     scan=42,
        ...     sample_name="PtNP",
        ...     experiment_file_path="/data/id01/sample.h5"
        ... )

        Direct instantiation:

        >>> from cdiutils.io.id01 import ID01Loader
        >>> loader = ID01Loader(
        ...     experiment_file_path="/data/id01/sample.h5",
        ...     scan=42,
        ...     sample_name="PtNP",
        ...     detector_name="eiger2M"
        ... )

        Load data with preprocessing:

        >>> data, angles = loader.load_data(
        ...     roi=(100, 400, 150, 450),
        ...     rocking_angle_binning=2
        ... )

    See Also:
        :class:`SpecLoader` for legacy SPEC format data.
        :class:`Loader` for factory method and base class documentation.
    """

    angle_names = {
        "sample_outofplane_angle": "eta",
        "sample_inplane_angle": "phi",
        "detector_outofplane_angle": "delta",
        "detector_inplane_angle": "nu",
    }
    authorised_detector_names = ("mpxgaas", "mpx1x4", "eiger2M")

    def __init__(
        self,
        experiment_file_path: str,
        scan: int = None,
        sample_name: str = None,
        detector_name: str = None,
        flat_field: np.ndarray | str = None,
        alien_mask: np.ndarray | str = None,
        **kwargs,
    ) -> None:
        """
        Initialise ID01 data loader with experiment file and metadata.

        Args:
            experiment_file_path: Path to BLISS HDF5 master file
                (typically ``sample_name.h5``). File must contain scan
                groups in format ``{sample}_{scan}.1/``.
            scan: Scan number to load. If None, must be specified in
                subsequent :meth:`load_data` calls.
            sample_name: Sample identifier matching HDF5 group names.
                Required to construct scan paths.
            detector_name: Detector identifier (``"mpxgaas"``,
                ``"mpx1x4"``, or ``"eiger2M"``). If None, automatically
                detected from first available scan.
            flat_field: Flat-field correction array or path to .npy/.npz
                file. Shape must match detector's 2D frame. Applied
                multiplicatively to raw data.
            alien_mask: Bad pixel mask array or path. Binary mask with
                1 = bad pixel, 0 = good pixel. Combined with detector's
                chip gap mask.
            **kwargs: Additional parameters (currently unused, reserved
                for future extensions).

        Raises:
            FileNotFoundError: If ``experiment_file_path`` does not
                exist.
            ValueError: If ``detector_name`` is not in
                :attr:`authorised_detector_names`.
            KeyError: If ``scan`` or ``sample_name`` do not match HDF5
                structure.

        Examples:
            Minimal setup (auto-detect detector):

            >>> loader = ID01Loader(
            ...     experiment_file_path="/data/id01/PtNP.h5",
            ...     scan=42,
            ...     sample_name="PtNP"
            ... )

            With flat-field and detector specification:

            >>> loader = ID01Loader(
            ...     experiment_file_path="/data/id01/sample.h5",
            ...     scan=100,
            ...     sample_name="sample",
            ...     detector_name="eiger2M",
            ...     flat_field="/path/to/flatfield.npy"
            ... )
        """
        super().__init__(
            experiment_file_path,
            scan,
            sample_name,
            detector_name,
            flat_field,
            alien_mask,
        )

    @h5_safe_load
    def get_detector_name(
        self, start_scan: int = 1, max_attempts: int = 5
    ) -> str:
        """
        Auto-detect detector from HDF5 file scan metadata.

        Searches through scan groups to find which authorised detector
        is present in the measurement data. Used when detector is not
        explicitly specified during initialisation.

        Args:
            start_scan: Scan number to begin search. Recursively
                increments if scan not found or contains no detector.
            max_attempts: Maximum number of scans to check before
                giving up.

        Returns:
            First matching detector name from
            :attr:`authorised_detector_names` found in file.

        Raises:
            ValueError: If no detector found after ``max_attempts``
                scans, or if multiple detectors found in same scan
                (ambiguous configuration).
            KeyError: If HDF5 structure does not match expected
                ``{sample}_{scan}.1/measurement/`` format.

        Notes:
            Recursion avoids issues with missing or incomplete
            scans. For files with both Eiger and Maxipix data,
            explicitly specify ``detector_name`` to avoid ambiguity.
        """

        msg = "Please provide a detector_name (str)."

        # Try to find the detector name in the current scan number
        key_path = f"{self.sample_name}_{start_scan}.1/measurement/"

        # If we've exceeded max attempts, raise an error
        if start_scan > max_attempts:
            raise ValueError(
                f"No detector found after checking {max_attempts} scans.\n"
                f"{msg}"
            )

        # Check if the key path exists
        if key_path not in self.h5file:
            # Try the next scan number recursively
            return self.get_detector_name(start_scan + 1, max_attempts)

        # Look for detector names in the current scan
        detector_names = []
        for key in self.authorised_detector_names:
            if key in self.h5file[key_path]:
                detector_names.append(key)

        if len(detector_names) == 0:
            # Try the next scan number recursively
            return self.get_detector_name(start_scan + 1, max_attempts)

        if len(detector_names) > 1:
            raise ValueError(
                f"Several detector names found ({detector_names}).\n"
                f"Not handled yet.\n{msg}"
            )

        return detector_names[0]

    @h5_safe_load
    def load_det_calib_params(
        self, scan: int = None, sample_name: str = None
    ) -> dict:
        """
        Load detector calibration from scan metadata.

        Retrieves calibration parameters stored in BLISS HDF5 file
        during detector alignment. Returns parameters compatible with
        xrayutilities conventions.

        Args:
            scan: Scan number to load calibration from. If None, uses
                ``self.scan``.
            sample_name: Sample name for HDF5 path construction. If
                None, uses ``self.sample_name``.

        Returns:
            dict: Calibration parameters with keys:

                - ``"cch1"``: Direct beam row (y) position in pixels
                - ``"cch2"``: Direct beam column (x) position in pixels
                - ``"pwidth1"``: Pixel height in metres
                - ``"pwidth2"``: Pixel width in metres
                - ``"distance"``: Sample-to-detector distance in metres
                - ``"tiltazimuth"``: Detector azimuthal tilt (0.0, not
                  calibrated by BLISS)
                - ``"tilt"``: Detector polar tilt (0.0, not calibrated)
                - ``"detrot"``: Detector rotation (0.0, not calibrated)

        Raises:
            KeyError: If scan/sample combination does not exist in HDF5
                file or if detector name is incorrect.

        Examples:
            Load calibration for current scan:

            >>> loader = ID01Loader(
            ...     experiment_file_path="/data/id01/sample.h5",
            ...     scan=42,
            ...     sample_name="sample"
            ... )
            >>> calib = loader.load_det_calib_params()
            >>> print(f"Direct beam at ({calib['cch1']}, {calib['cch2']})")

            Load from different scan:

            >>> calib = loader.load_det_calib_params(scan=15)

        Notes:
            Tilt angles (``tiltazimuth``, ``tilt``, ``detrot``) are set
            to 0.0 as BLISS does not calibrate these. For accurate tilt
            values, run detector calibration notebook or use PyNX's
            ``cdi_findcenter`` utility.

        See Also:
            :doc:`/user_guide/detector_calibration` for calibration
            procedures and angle definitions.
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        key_path = (
            "_".join((sample_name, str(scan)))
            + f".1/instrument/{self.detector_name}"
        )
        try:
            return {
                "cch1": float(self.h5file[key_path + "/beam_center_y"][()]),
                "cch2": float(self.h5file[key_path + "/beam_center_x"][()]),
                "pwidth1": float(self.h5file[key_path + "/y_pixel_size"][()]),
                "pwidth2": float(self.h5file[key_path + "/x_pixel_size"][()]),
                "distance": float(self.h5file[key_path + "/distance"][()]),
                "tiltazimuth": 0.0,
                "tilt": 0.0,
                "detrot": 0.0,
            }
        except KeyError as exc:
            raise KeyError(
                f"key_path is wrong (key_path='{key_path}'). "
                "Are sample_name, scan number or detector name correct?"
            ) from exc

    @h5_safe_load
    def load_detector_shape(
        self,
        scan: int = None,
        sample_name: str = None,
    ) -> tuple:
        """
        Load detector's native pixel array dimensions from scan.

        Args:
            scan: Scan number. If None, uses ``self.scan``.
            sample_name: Sample name. If None, uses ``self.sample_name``.

        Returns:
            Two-element tuple ``(n_rows, n_columns)`` with detector's
            full frame shape (e.g., ``(2164, 1030)`` for Eiger2M).

        Raises:
            KeyError: If scan/sample combination or detector not
                found in HDF5 file.
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)

        key_path = (
            "_".join((sample_name, str(scan)))
            + f".1/instrument/{self.detector_name}"
        )
        return (
            self.h5file[f"{key_path}/dim_j"][()],
            self.h5file[f"{key_path}/dim_i"][()],
        )

    @h5_safe_load
    def load_detector_data(
        self,
        scan: int = None,
        sample_name: str = None,
        roi: tuple[slice] = None,
        rocking_angle_binning: int = None,
        binning_method: str = "sum",
    ) -> np.ndarray:
        """
        Load raw detector frames from BLISS HDF5 file.

        Retrieves 3D detector data array with optional ROI selection,
        binning, flat-field correction, and masking applied via
        :meth:`Loader.bin_flat_mask`.

        Args:
            scan: Scan number. If None, uses ``self.scan``.
            sample_name: Sample name for HDF5 path. If None, uses
                ``self.sample_name``.
            roi: Region of interest as tuple of slices or integers. See
                :meth:`Loader._check_roi` for format. Applied before
                binning to reduce memory usage.
            rocking_angle_binning: Binning factor along rocking curve
                (frame) axis. If None or 1, no binning performed.
            binning_method: Binning operation (``"sum"``, ``"mean"``, or
                ``"max"``). Default ``"sum"`` preserves total counts.

        Returns:
            Preprocessed detector data with shape
            ``(n_frames//binning, n_y, n_x)``. Data type is uint16
            (Maxipix) or uint32 (Eiger).

        Raises:
            KeyError: If scan/sample/detector combination does not exist
                in HDF5 file.

        Examples:
            Full detector, no preprocessing:

            >>> data = loader.load_detector_data(scan=42)
            >>> data.shape
            (51, 2164, 1030)

            With ROI and binning:

            >>> data = loader.load_detector_data(
            ...     scan=42,
            ...     roi=(100, 400, 150, 450),
            ...     rocking_angle_binning=2,
            ...     binning_method="sum"
            ... )
            >>> # Returns (25, 300, 300) array

        See Also:
            :meth:`load_data` for combined data + motor positions.
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        key_path = (
            "_".join((sample_name, str(scan)))
            + f".1/measurement/{self.detector_name}"
        )
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
        scan: int = None,
        sample_name: str = None,
        roi: tuple[slice] = None,
        rocking_angle_binning: int = None,
    ) -> dict:
        """
        Load diffractometer motor angles for scan.

        Retrieves sample and detector motor positions, applying same ROI
        and binning as detector data to maintain synchronisation.

        Args:
            scan: Scan number. If None, uses ``self.scan``.
            sample_name: Sample name. If None, uses ``self.sample_name``.
            roi: ROI tuple matching detector data ROI. Only first
                element (rocking curve axis) is used. If None, full scan
                loaded.
            rocking_angle_binning: Binning factor matching detector
                binning. Angles are averaged (mean) when binned.

        Returns:
            dict: Motor angles with canonical keys (see
            :attr:`angle_names` for ID01-specific mapping):

                - ``"sample_outofplane_angle"``: eta values (degrees)
                - ``"sample_inplane_angle"``: phi values (degrees)
                - ``"detector_outofplane_angle"``: delta values
                  (degrees)
                - ``"detector_inplane_angle"``: nu values (degrees)

            Values are scalars (if motor fixed) or 1D arrays (if
            scanned). Array lengths match binned detector's first
            dimension.

        Raises:
            KeyError: If scan/sample combination not found in HDF5 file.

        Examples:
            Load angles matching data:

            >>> data = loader.load_detector_data(
            ...     scan=42,
            ...     roi=(10, 40, 100, 400),
            ...     rocking_angle_binning=2
            ... )
            >>> angles = loader.load_motor_positions(
            ...     scan=42,
            ...     roi=(slice(10, 40),),
            ...     rocking_angle_binning=2
            ... )
            >>> angles["sample_outofplane_angle"].shape
            (15,)  # (40-10)//2 = 15

        See Also:
            :meth:`load_data` for combined data + angles loading.
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        angles = self.load_angles(
            key_path=f"{sample_name}_{scan}.1/instrument/positioners/"
        )

        # ensure angles dictionary has correct keys and defaults to 0.0
        # if missing
        formatted_angles = {
            key: angles.get(name, 0.0)
            for key, name in ID01Loader.angle_names.items()
        }
        self.rocking_angle = self.get_rocking_angle(formatted_angles)

        scan_axis_roi = self._check_roi(roi)[0]

        # format the angles and map them back to their corresponding keys
        formatted_values = self.format_scanned_counters(
            *formatted_angles.values(),
            scan_axis_roi=scan_axis_roi,
            rocking_angle_binning=rocking_angle_binning,
        )

        # return a dictionary mapping original angle keys to their
        # formatted values. This is possible because Python maintains
        # order !
        return dict(zip(formatted_angles.keys(), formatted_values))

    @h5_safe_load
    def load_energy(self, scan: int = None, sample_name: str = None) -> float:
        """
        Load X-ray beam energy for scan.

        Args:
            scan: Scan number. If None, uses ``self.scan``.
            sample_name: Sample name. If None, uses ``self.sample_name``.

        Returns:
            Beam energy in eV (converted from monochromator energy in
            keV). Returns scalar or array depending on whether energy
            was scanned.

        Warns:
            UserWarning: If energy key (``"mononrj"``) not found in HDF5
            file, returns None.

        Examples:
            >>> energy = loader.load_energy(scan=42)
            >>> print(f"Energy: {energy/1e3:.2f} keV")
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        key_path = f"{sample_name}_{scan}.1/instrument/positioners/"
        try:
            energy = self.h5file[key_path + "mononrj"][()] * 1e3
            if isinstance(energy, np.ndarray):
                return energy
            return float(energy)
        except KeyError:
            warnings.warn(f"Energy not found at {key_path + 'mononrj'}. ")
            return None

    @h5_safe_load
    def show_scan_attributes(
        self,
        scan: int = None,
        sample_name: str = None,
    ) -> None:
        """
        Print HDF5 keys available for scan (debugging utility).

        Displays top-level group structure for specified scan, useful
        for inspecting file organisation and finding custom metadata.

        Args:
            scan: Scan number. If None, uses ``self.scan``.
            sample_name: Sample name. If None, uses
                ``self.sample_name``.
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        key_path = "_".join((sample_name, str(scan))) + ".1"
        print(self.h5file[key_path].keys())

    @h5_safe_load
    def load_measurement_parameters(
        self, parameter_name: str, scan: int = None, sample_name: str = None
    ) -> tuple:
        """
        Load custom measurement data from scan.

        Retrieves arbitrary datasets stored under
        ``{scan}/measurement/`` HDF5 group. Useful for accessing
        non-standard counters or experimental metadata.

        Args:
            parameter_name: Dataset name under measurement group (e.g.,
                ``"mu"``, ``"chi"``, custom IOC counters).
            scan: Scan number. If None, uses ``self.scan``.
            sample_name: Sample name. If None, uses ``self.sample_name``.

        Returns:
            Dataset contents (type depends on stored data: array,
            scalar, or string).
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        key_path = "_".join((sample_name, str(scan))) + ".1/measurement"
        requested_mes_parameters = self.h5file[f"{key_path}/{parameter_name}"][
            ()
        ]
        return requested_mes_parameters

    @h5_safe_load
    def load_instrument_parameters(
        self,
        instrument_parameter: str,
        scan: int = None,
        sample_name: str = None,
    ) -> tuple:
        """
        Load instrument metadata from scan.

        Retrieves datasets under ``{scan}/instrument/`` group, including
        positioners, detectors, and beamline equipment metadata.

        Args:
            instrument_parameter: Dataset path under instrument group
                (e.g., ``"positioners/delta"``, ``"eiger2M/roi_mode"``).
            scan: Scan number. If None, uses ``self.scan``.
            sample_name: Sample name. If None, uses ``self.sample_name``.

        Returns:
            Dataset contents (type depends on stored data).
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        key_path = "_".join((sample_name, str(scan))) + ".1/instrument"

        return self.h5file[key_path + "/" + instrument_parameter][()]

    @h5_safe_load
    def load_sample_parameters(
        self,
        sam_parameter: str,
        scan: int = None,
        sample_name: str = None,
    ) -> tuple:
        """
        Load sample metadata from scan.

        Retrieves sample-specific information stored under
        ``{scan}/sample/`` group (e.g., temperature, pressure, notes).

        Args:
            sam_parameter: Dataset name under sample group.
            scan: Scan number. If None, uses ``self.scan``.
            sample_name: Sample name. If None, uses ``self.sample_name``.

        Returns:
            Dataset contents (type depends on stored data).
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        key_path = "_".join((sample_name, str(scan))) + ".1/sample"
        requested_parameters = self.h5file[key_path + "/" + sam_parameter][()]
        return requested_parameters

    @h5_safe_load
    def load_plotselect_parameter(
        self,
        plot_parameter,
        scan: int = None,
        sample_name: str = None,
    ) -> tuple:
        """
        Load BLISS plotselect metadata from scan.

        Retrieves counters selected for real-time plotting during data
        acquisition. Rarely needed for data analysis.

        Args:
            plot_parameter: Counter name in plotselect group.
            scan: Scan number. If None, uses ``self.scan``.
            sample_name: Sample name. If None, uses ``self.sample_name``.

        Returns:
            Counter values as stored in HDF5.
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        key_path = "_".join((sample_name, str(scan))) + ".1/plotselect"
        requested_parameter = self.h5file[key_path + "/" + plot_parameter][()]

        return requested_parameter

    @h5_safe_load
    def get_start_time(self, scan: int = None, sample_name: str = None) -> str:
        """
        Get scan acquisition start timestamp.

        Parses ISO 8601 timestamp stored by BLISS into datetime object
        for temporal analysis or logging.

        Args:
            scan: Scan number. If None, uses ``self.scan``.
            sample_name: Sample name. If None, uses ``self.sample_name``.

        Returns:
            ISO-formatted timestamp string parsable by
            :func:`dateutil.parser.isoparse`.
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        key_path = "_".join((sample_name, str(scan))) + ".1/start_time"

        return dateutil.parser.isoparse(self.h5file[key_path][()])


def safe(func):
    def wrap(self, *args, **kwargs):
        with silx.io.open(self.experiment_file_path) as self.specfile:
            return func(self, *args, **kwargs)

    return wrap


# TODO: Implement roi parameter for detector, motors and mask methods
class SpecLoader(Loader):
    """A loader for loading .spec files."""

    angle_names = {
        "sample_outofplane_angle": "eta",
        "sample_inplane_angle": "phi",
        "detector_outofplane_angle": "del",
        "detector_inplane_angle": "nu",
    }

    def __init__(
        self,
        experiment_file_path: str,
        detector_data_path: str,
        edf_file_template: str,
        detector_name: str,
        scan: int = None,
        flat_field: str | np.ndarray = None,
        alien_mask: np.ndarray | str = None,
        **kwargs,
    ) -> None:
        """
        Initialise SpecLoader with experiment data and detector
        information.

        Args:
            experiment_file_path (str): path to the spec master file
                used for the experiment.
            detector_data_path (str): the path to the directory
                containing the detector data.
            edf_file_template (str): the file name template of the
                detector data frame.
            detector_name (str): name of the detector.
            scan (int, optional): the scan number. Defaults to None.
            flat_field (str | np.ndarray, optional): flat field to
                account for the non homogeneous counting of the
                detector. Defaults to None.
            alien_mask (np.ndarray | str, optional): array to mask the
                aliens. Defaults to None.
        """
        super().__init__(scan, None, flat_field, alien_mask)
        self.experiment_file_path = experiment_file_path
        self.detector_data_path = detector_data_path
        self.edf_file_template = edf_file_template
        self.detector_name = detector_name

    @safe
    def load_detector_data(
        self,
        scan: int = None,
        roi: tuple[slice] = None,
        rocking_angle_binning: int = None,
        binning_method: str = "sum",
    ):
        scan, _ = self._check_scan_sample(scan, None)
        roi = self._check_roi(roi)

        frame_ids = self.specfile[
            f"{scan}.1/measurement/{self.detector_name}"
        ][()]

        data = []

        template = self.detector_data_path + self.edf_file_template

        for frame_id in frame_ids:
            with fabio.open(template % frame_id) as edf_data:
                data.append(edf_data.data[roi[1:]])
        data = np.asarray(data)

        if rocking_angle_binning is None:
            data = data[roi[0]]

        return self.bin_flat_mask(
            data,
            roi,
            self.flat_field,
            self.alien_mask,
            rocking_angle_binning,
            binning_method,
        )

    @safe
    def load_motor_positions(
        self,
        scan: int = None,
        roi: tuple[slice] = None,
        rocking_angle_binning: int = None,
    ):
        scan, _ = self._check_scan_sample(scan, None)
        roi = self._check_roi(roi)
        roi = roi[0]

        positioners = self.specfile[f"{scan}.1/instrument/positioners"]

        angles = {key: None for key in SpecLoader.angle_names}
        for angle, name in SpecLoader.angle_names.items():
            try:
                angles[angle] = positioners[name][roi]
            except ValueError:
                angles[angle] = angles[angle] = positioners[name][()]

        self.rocking_angle = self.get_rocking_angle(angles)
        if self.rocking_angle is None:
            raise ValueError("No rocking angle found.")

        angles[self.rocking_angle] = self.bin_rocking_angle_values(
            angles[self.rocking_angle], rocking_angle_binning
        )
        if roi and rocking_angle_binning:
            angles[self.rocking_angle] = angles[self.rocking_angle][roi]
        return angles

    def load_det_calib_params(self) -> dict:
        return None

    def load_energy(self) -> float:
        return None

    def load_detector_shape(self) -> tuple:
        return None
