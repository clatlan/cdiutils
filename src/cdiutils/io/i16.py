import warnings

import dateutil.parser
import fabio
import numpy as np
import silx.io

from cdiutils.io.loader import H5TypeLoader, Loader, h5_safe_load


class I16Loader(H5TypeLoader):
    """
    Data loader for Diamond Light Source I16 beamline.

    Loads data from NeXus files.

    Attributes:
        angle_names: Mapping from canonical names to ID01 motor names:

            - ``sample_outofplane_angle`` -> ``"eta"``
            - ``sample_inplane_angle`` -> ``"chi"``
            - ``detector_outofplane_angle`` -> ``"delta"``
            - ``detector_inplane_angle`` -> ``"gamma"``

        authorised_detector_names: Tuple of supported detectors:
            ``("merlin")``.

    Examples:
        Basic usage with factory pattern:

        >>> from cdiutils.io import Loader
        >>> loader = Loader.from_setup(
        ...     beamline_setup="i16",
        ...     sample_name="PtNP",
        ...     experiment_file_path="/dls/i16/data/2026/mm12345-1/12345.nxs"
        ... )

        Direct instantiation:

        >>> from cdiutils.io.i16 import I16Loader
        >>> loader = I16Loader(
        ...     experiment_file_path=="/dls/i16/data/2026/mm12345-1/12345.nxs",
        ...     sample_name="PtNP",
        ...     detector_name="merlin"
        ... )

        Load data with preprocessing:

        >>> data, angles = loader.load_data(
        ...     roi=(100, 400, 150, 450),
        ...     rocking_angle_binning=2
        ... )

    See Also:
        :class:`Loader` for factory method and base class documentation.
    """

    angle_names = {
        "sample_outofplane_angle": "eta",
        "sample_inplane_angle": "mu",
        "detector_outofplane_angle": "delta",
        "detector_inplane_angle": "gam",
    }
    authorised_detector_names = ("merlin", )

    def __init__(
        self,
        experiment_file_path: str,
        detector_name: str = None,
        flat_field: np.ndarray | str = None,
        alien_mask: np.ndarray | str = None,
        **kwargs,
    ):
        """
        Initialise I16 data loader with experiment file and metadata.

        Args:
            experiment_file_path: Path to Nexus scan file
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

            >>> loader = I16Loader(
            ...     experiment_file_path="/data/id01/PtNP.h5"
            ... )

            With flat-field and detector specification:

            >>> loader = I16Loader(
            ...     experiment_file_path="/data/id01/sample.h5",
            ...     detector_name="merlin",
            ...     flat_field="/path/to/flatfield.npy"
            ... )
        """
        super().__init__(
            experiment_file_path,
            None,
            None,
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
        instrument = self.h5file['entry/instrument']
        detector = instrument[self.detector_name]
        module = detector["module"]
        try:
            return {
                "cch1": float(instrument['merlin_centre_i'][()]) if 'merlin_centre_i' in instrument else 147,
                "cch2": float(instrument['merlin_centre_j'][()]) if 'merlin_centre_j' in instrument else 335,
                "pwidth1": float(module['fast_pixel_direction'][()]),
                "pwidth2": float(module['slow_pixel_direction'][()]),
                "distance": float(detector['transformations/origin_offset'][()]),
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
    ) -> tuple:
        """
        Load detector's native pixel array dimensions from scan.

        Returns:
            Two-element tuple ``(n_rows, n_columns)`` with detector's
            full frame shape (e.g., ``(2164, 1030)`` for Eiger2M).

        Raises:
            KeyError: If detector not found in HDF5 file.
        """
        # /entry/instrument/merlin/module/data_size
        instrument = self.h5file['entry/instrument']
        detector = instrument[self.detector_name]
        module = detector["module"]
        return module['data_size'][()]

    @h5_safe_load
    def load_detector_data(
        self,
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
        key_path = f"entry/instrument/{self.detector_name}/data"
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
        Load diffractometer motor angles for scan.

        Retrieves sample and detector motor positions, applying same ROI
        and binning as detector data to maintain synchronisation.

        Args:
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
        angles = self.load_angles(
            key_path=f"entry/instrument/diffractometer_sample/"
        )

        # ensure angles dictionary has correct keys and defaults to 0.0
        # if missing
        formatted_angles = {
            key: angles.get(name, 0.0)
            for key, name in I16Loader.angle_names.items()
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
    def load_energy(self) -> float:
        """
        Load X-ray beam energy for scan.

        Returns:
            Beam energy in eV (converted from monochromator energy in
            keV). Returns scalar or array depending on whether energy
            was scanned.

        Warns:
            UserWarning: If energy key (``"mononrj"``) not found in HDF5
            file, returns None.

        Examples:
            >>> energy = loader.load_energy()
            >>> print(f"Energy: {energy/1e3:.2f} keV")
        """
        energy = self.h5file["entry/sample/beam/incident_energy"][()] * 1e3
        return float(energy)

    @h5_safe_load
    def show_scan_attributes(
        self,
    ) -> None:
        """
        Print HDF5 keys available for scan (debugging utility).

        Displays top-level group structure for specified scan, useful
        for inspecting file organisation and finding custom metadata.
        """
        print(self.h5file['entry'].keys())

    @h5_safe_load
    def load_measurement_parameters(
        self, parameter_name: str
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
        key_path = "entry/measurement"
        return self.h5file[f"{key_path}/{parameter_name}"][()]

    @h5_safe_load
    def load_instrument_parameters(
        self,
        instrument_parameter: str,
    ) -> tuple:
        """
        Load instrument metadata from scan.

        Retrieves datasets under ``{scan}/instrument/`` group, including
        positioners, detectors, and beamline equipment metadata.

        Args:
            instrument_parameter: Dataset path under instrument group
                (e.g., ``"positioners/delta"``, ``"eiger2M/roi_mode"``).

        Returns:
            Dataset contents (type depends on stored data).
        """
        key_path = "entry/instrument"
        return self.h5file[f"{key_path}/{instrument_parameter}"][()]

    @h5_safe_load
    def load_sample_parameters(
        self,
        sam_parameter: str,
    ) -> tuple:
        """
        Load sample metadata from scan.

        Retrieves sample-specific information stored under
        ``{scan}/sample/`` group (e.g., temperature, pressure, notes).

        Args:
            sam_parameter: Dataset name under sample group.

        Returns:
            Dataset contents (type depends on stored data).
        """
        key_path = "entry/sample"
        return self.h5file[f"{key_path}/{sam_parameter}"][()]

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
        key_path = "entry/start_time"
        return dateutil.parser.isoparse(self.h5file[key_path][()])


def safe(func):
    def wrap(self, *args, **kwargs):
        with silx.io.open(self.experiment_file_path) as self.specfile:
            return func(self, *args, **kwargs)

    return wrap

