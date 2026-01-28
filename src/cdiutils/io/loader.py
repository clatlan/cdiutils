"""
Beamline-specific data loaders for BCDI experiments.

This module provides the abstract base Loader class and beamline-specific
implementations for loading detector data, motor positions, and metadata
from synchrotron BCDI experiments.
"""

from abc import ABC, abstractmethod
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import silx.io.h5py_utils

from cdiutils.plot import add_colorbar
from cdiutils.utils import CroppingHandler, bin_along_axis, get_centred_slices


class Loader(ABC):
    """
    Abstract base class for beamline-specific data loaders.

    Loaders handle experiment-specific data I/O operations including:
    - HDF5/NeXus/SPEC file parsing
    - Detector data extraction with ROI support
    - Motor angle retrieval
    - Energy and detector calibration parameter loading
    - Flat-field correction and bad pixel masking

    Use the factory method :meth:`from_setup` to instantiate the appropriate
    subclass for your beamline, or directly instantiate beamline-specific
    loaders (ID01Loader, P10Loader, etc.) for advanced configuration.

    Supported beamlines:
        - ID01 (ESRF): :class:`ID01Loader`
        - P10 (PETRA III): :class:`P10Loader`
        - SIXS (SOLEIL): :class:`SIXSLoader`
        - NanoMAX (MAX IV): :class:`NanoMAXLoader`
        - CRISTAL (SOLEIL): :class:`CristalLoader`
        - ID27 (ESRF): :class:`ID27Loader`

    Attributes:
        scan (int): Scan number identifier.
        sample_name (str): Sample identifier for file organisation.
        flat_field (np.ndarray): Flat-field correction array for detector
            non-uniformity.
        alien_mask (np.ndarray): Mask for defective detector pixels.
        detector_name (str): Detector type (set by subclass).
        rocking_angle (str): Name of rocking curve motor (beamline-specific).

    See Also:
        :class:`~cdiutils.pipeline.BcdiPipeline`: Uses loaders automatically
        :class:`ID01Loader`: ESRF ID01 beamline implementation
        :class:`P10Loader`: PETRA III P10 beamline implementation

    Examples:
        Using factory pattern (recommended):

        >>> loader = Loader.from_setup(
        ...     beamline_setup="id01",
        ...     sample_name="PtNP",
        ...     scan=42,
        ...     data_dir="/data/id01/sample"
        ... )
        >>> data, angles = loader.load_data()

        Direct instantiation:

        >>> from cdiutils.io import ID01Loader
        >>> loader = ID01Loader(
        ...     sample_name="PtNP",
        ...     scan=42,
        ...     experiment_file_path="/data/sample.h5"
        ... )
    """

    def __init__(
        self,
        scan: int = None,
        sample_name: str = None,
        flat_field: np.ndarray | str = None,
        alien_mask: np.ndarray | str = None,
    ) -> None:
        """
        Initialise the base Loader.

        Typically called by subclass constructors. Users should prefer
        :meth:`from_setup` factory method or direct subclass instantiation.

        Args:
            scan: Scan number identifier. Required for data loading.
            sample_name: Sample identifier used in file paths and logging.
            flat_field: Flat-field correction array or path to `.npy`/`.npz`
                file. Applied as multiplicative correction to detector data.
                Shape must match detector dimensions. Defaults to None (no
                correction).
            alien_mask: Bad pixel mask array or path to `.npy`/`.npz` file.
                Pixels with value 1 are masked (invalid), 0 are kept. Shape
                must match detector. Defaults to None (no masking).

        Raises:
            ValueError: If flat_field or alien_mask path is invalid or file
                format is unsupported.
        """
        self.scan = scan
        self.sample_name = sample_name
        self.flat_field = self._check_load(flat_field)
        self.alien_mask = self._check_load(alien_mask)
        self.detector_name = None
        self.rocking_angle = "sample_outofplane_angle"

    def get_alien_mask(
        self, roi: tuple[slice, slice, slice] = None
    ) -> np.ndarray:
        if self.alien_mask is None:
            return None

        if roi is None:
            return self.alien_mask

        return self.alien_mask[roi]

    @classmethod
    def from_setup(cls, beamline_setup: str, **metadata) -> "Loader":
        """
        Factory method to instantiate beamline-specific loader.

        Automatically selects and returns the appropriate Loader
        subclass based on beamline name. This is the recommended way
        to create loaders as it handles beamline-specific
        initialisation automatically.

        Args:
            beamline_setup: Beamline identifier (case-insensitive).
                Supported values:

                - ``"id01"`` or ``"id01bliss"``: ESRF ID01 (BLISS
                  format)
                - ``"id01spec"``: ESRF ID01 (legacy SPEC format)
                - ``"sixs2019"`` or ``"sixs2022"``: SOLEIL SIXS
                  (specify year)
                - ``"p10"`` or ``"p10eh2"``: PETRA III P10 (specify
                  hutch)
                - ``"cristal"``: SOLEIL CRISTAL
                - ``"nanomax"``: MAX IV NanoMAX
                - ``"id27"``: ESRF ID27

            **metadata: Beamline-specific keyword arguments passed to loader
                constructor. Common parameters include:

                - ``scan`` (int): Scan number
                - ``sample_name`` (str): Sample identifier
                - ``experiment_file_path`` (str): Path to experiment HDF5/SPEC
                - ``data_dir`` (str): Root data directory
                - ``flat_field`` (np.ndarray | str): flat-field
              correction
                - ``alien_mask`` (np.ndarray | str): bad pixel mask

        Returns:
            Beamline-specific Loader subclass instance.

        Raises:
            ValueError: If ``beamline_setup`` is not recognised.
            NotImplementedError: If beamline version (e.g., SIXS
                year) is not specified or unsupported.

        Examples:
            Basic usage:

            >>> loader = Loader.from_setup(
            ...     beamline_setup="id01",
            ...     scan=42,
            ...     sample_name="PtNP",
            ...     experiment_file_path="/data/id01/beamtile_id01.h5"
            ... )

            With version specification:

            >>> loader = Loader.from_setup(
            ...     beamline_setup="sixs2022",
            ...     scan=100,
            ...     sample_name="SrTiO3"
            ... )

            With flat-field and mask:

            >>> loader = Loader.from_setup(
            ...     beamline_setup="p10",
            ...     scan=15,
            ...     flat_field="/path/to/flatfield.npy",
            ...     alien_mask="/path/to/badpixels.npy"
            ... )
        """
        if "id01" in beamline_setup.lower():
            if beamline_setup.lower() == "id01spec":
                from . import SpecLoader

                return SpecLoader(**metadata)
            from . import ID01Loader

            return ID01Loader(**metadata)

        if "sixs" in beamline_setup.lower():
            from . import SIXSLoader

            if "2022" in beamline_setup.lower():
                return SIXSLoader(version="2022", **metadata)
            if "2019" in beamline_setup.lower():
                return SIXSLoader(version="2019", **metadata)
            raise NotImplementedError(
                "Only 2019 and 2022 versions are available for now. Specify "
                "the version in the beamline_setup."
            )

        if "p10" in beamline_setup.lower():
            from . import P10Loader

            if beamline_setup.lower() == "p10eh2":
                return P10Loader(hutch="EH2", **metadata)
            return P10Loader(**metadata)

        if beamline_setup.lower() == "cristal":
            from . import CristalLoader

            return CristalLoader(**metadata)

        if beamline_setup.lower() == "nanomax":
            from . import NanoMAXLoader

            return NanoMAXLoader(**metadata)

        if beamline_setup.lower() == "id27":
            from . import ID27Loader

            return ID27Loader(**metadata)
        raise ValueError(f"Invalid beamline setup: {beamline_setup = }")  # noqa: E251

    @staticmethod
    def _check_load(data_or_path: np.ndarray | str) -> np.ndarray:
        """
        Load flat-field or alien mask from file or validate array.

        Handles both direct array input and file paths (.npy/.npz).
        For .npz archives, searches for data under common key names.
        Returns:
            Loaded or validated numpy array, or None if input was None.

        Raises:
            KeyError: If .npz file does not contain expected keys
                ('arr_0', 'data', 'mask', 'flatfield',
                'flat_field').
            ValueError: If input is neither array, path, nor None.
        """
        if isinstance(data_or_path, str):
            if data_or_path.endswith(".npy"):
                return np.load(data_or_path)
            if data_or_path.endswith(".npz"):
                with np.load(data_or_path, "r") as file:
                    for possible_key in (
                        "arr_0",
                        "data",
                        "mask",
                        "flatfield",
                        "flat_field",
                    ):
                        if possible_key in dict(file):
                            return file[possible_key]
                    raise KeyError(
                        f"Invalid file provided containing {file.keys()}."
                    )
        elif data_or_path is None or isinstance(data_or_path, np.ndarray):
            return data_or_path
        raise ValueError(
            "[ERROR] wrong value for flat_field and/or alien_mask "
            "parameter provide a path, np.ndarray or leave it to None"
        )

    @staticmethod
    def _check_roi(roi: tuple = None) -> tuple[slice]:
        """
        Validate and normalise region of interest (ROI) specification.

        Accepts ROI as either slice objects or integer boundaries, and
        normalises to 3D tuple of slices. Handles both 2D and 3D ROIs.

        Args:
            roi: ROI specification in one of three formats:

                - **Tuple of slices**: ``(slice1, slice2)`` for 2D or
                  ``(slice0, slice1, slice2)`` for 3D.
                - **Tuple of integers**: ``(y_min, y_max, x_min, x_max)``
                  for 2D or ``(z_min, z_max, y_min, y_max, x_min, x_max)``
                  for 3D.
                - **None**: No ROI (entire array).

        Returns:
            Normalised 3D tuple of slices. For 2D input, first dimension
            is ``slice(None)`` (no cropping in detector's radial direction).

        Raises:
            ValueError: If ``roi`` format is invalid (wrong length, mixed
                types, or neither slices nor integers).

        Examples:
            No ROI (full array):

            >>> roi = Loader._check_roi(None)
            >>> # Returns (slice(None), slice(None), slice(None))

            2D slice specification:

            >>> roi = Loader._check_roi((slice(10, 50), slice(20, 60)))
            >>> # Returns (slice(None), slice(10, 50), slice(20, 60))

            Integer boundaries (2D):

            >>> roi = Loader._check_roi((100, 200, 150, 250))
            >>> # Returns (slice(None), slice(100, 200), slice(150, 250))

            Full 3D specification:

            >>> roi = Loader._check_roi((5, 45, 100, 200, 150, 250))
            >>> # Returns (slice(5, 45), slice(100, 200), slice(150, 250))
        """
        usage_text = (
            f"Wrong value for roi ({roi = }), roi should be:\n"  # noqa: E251
            "\t - either a tuple of slices with len = 2 or len = 3"
            "\t - either a tuple of int with len = 4 or len = 6"
        )
        if roi is None:
            return tuple(slice(None) for _ in range(3))
        if all(isinstance(e, slice) for e in roi):
            if len(roi) == 2:
                return (slice(None), roi[0], roi[1])
            if len(roi) == 3:
                return roi
        if (len(roi) == 4 or len(roi) == 6) and all(
            isinstance(e, (int, np.integer)) for e in roi
        ):
            return CroppingHandler.roi_list_to_slices(roi)
        raise ValueError(usage_text)

    def _check_scan_sample(
        self, scan: str = None, sample_name: str = None
    ) -> tuple:
        """
        Validate scan number and sample name, using instance
        defaults.

        Utility method that falls back to instance attributes if
        parameters are not explicitly provided. Used by data
        loading methods to ensure scan/sample context is always
        available.
                ``self.sample_name``.

        Returns:
            Two-element tuple: ``(scan, sample_name)`` with
            validated values.
        """
        if scan is None:
            scan = self.scan
        if sample_name is None:
            sample_name = self.sample_name
        return scan, sample_name

    @staticmethod
    def bin_flat_mask(
        data: np.ndarray,
        roi: list = None,
        flat_field: np.ndarray = None,
        alien_mask: np.ndarray = None,
        rocking_angle_binning: int = None,
        binning_method: str = "sum",
    ) -> np.ndarray:
        """
        Apply preprocessing: binning, flat-field, and masking.

        Combines three common preprocessing steps in correct order:

        1. Bin along rocking curve (if requested)
        2. Apply flat-field correction (if provided)
        3. Apply alien mask (if provided)

        Args:
            data: 3D detector data with shape (n_frames, n_y, n_x).
            roi: Region of interest as tuple of slices or integers. If
                None, uses full array. See :meth:`_check_roi` for
                format details.
            flat_field: 2D array with detector efficiency correction.
                Shape must match ``data.shape[1:]``. If None, no
                correction applied.
            alien_mask: Binary mask of bad pixels (1 = bad, 0 = good).
                Shape must match ``data.shape`` (3D) or
                ``data.shape[1:]`` (2D). If None, no masking applied.
            rocking_angle_binning: Binning factor along rocking curve
                axis (frames). If None or 1, no binning performed.
            binning_method: Binning operation. Options:

                - ``"sum"``: Sum frames (default, preserves total counts)
                - ``"mean"``: Average frames (reduces noise)
                - ``"max"``: Maximum projection (peak intensity)

        Returns:
            Preprocessed 3D array with same dtype as input. Shape is
            ``(n_frames//binning, n_y, n_x)`` if binned.

        Examples:
            ROI + flat-field + mask:

            >>> roi = (slice(None), slice(100, 400), slice(150, 450))
            >>> processed = Loader.bin_flat_mask(
            ...     data=raw_data,
            ...     roi=roi,
            ...     flat_field=flat,
            ...     alien_mask=mask
            ... )

            Binning only:

            >>> binned = Loader.bin_flat_mask(
            ...     data=raw_data,
            ...     rocking_angle_binning=2,
            ...     binning_method="sum"
            ... )
        """
        if roi is None:
            roi = (slice(None), slice(None), slice(None))

        if rocking_angle_binning:
            data = bin_along_axis(
                data, rocking_angle_binning, binning_method, axis=0
            )
            # If binning, roi[1] and roi[2] have been applied already.
            data = data[roi[0]]

        if flat_field is not None:
            data = data * flat_field[roi[1:]]

        if alien_mask is not None:
            data = data * (1 - alien_mask[roi])
        return data

    @staticmethod
    def bin_rocking_angle_values(
        values: list | np.ndarray, binning_factor: int = None
    ) -> np.ndarray:
        """
        Bin rocking angle values to match binned detector frames.

        Averages angle values when frames are binned together. Used to
        maintain synchronisation between data and motor positions.

        Args:
            values: Rocking angle values for each frame (e.g., delta,
                omega motor positions). Length must match original number
                of frames.
            binning_factor: Number of consecutive frames to average. If
                None or 1, returns input unchanged.

        Returns:
            Binned angle values with length
            ``len(values)//binning_factor``. Uses mean binning to
            get average angle per binned frame.
        """
        return bin_along_axis(values, binning_factor, binning_method="mean")

    @abstractmethod
    def load_energy(self):
        """
        Load X-ray beam energy for the scan.

        Must be implemented by beamline-specific subclass.

        Returns:
            Beam energy in keV.
        """
        pass

    @abstractmethod
    def load_det_calib_params(self):
        """
        Load detector calibration parameters from experiment file.

        Must be implemented by beamline-specific subclass. Typically
        reads values stored during detector alignment procedure.

        Returns:
            dict: Calibration parameters with keys:

                - ``"direct_beam"``: (y, x) pixel coordinates of direct
                  beam position
                - ``"detector_distance"``: sample-to-detector distance
                  in metres
                - ``"outofplane_angle"``: detector rotation $\delta$
                  or $\gamma$ in degrees
                - ``"inplane_angle"``: detector rotation $\nu$ in
                  degrees

        See Also:
            :doc:`/user_guide/detector_calibration` for calibration
            procedures and parameter definitions.
        """
        pass

    @abstractmethod
    def load_detector_shape(self):
        """
        Load detector's native pixel array shape.

        Must be implemented by beamline-specific subclass if detector
        shape cannot be determined from data files.

        Returns:
            Detector shape as (n_rows, n_columns) tuple, or None if
            shape is determined from data.
        """
        return None

    def get_detector_name(self) -> str:
        """
        Get canonical detector identifier for this beamline.

        Returns the first name from :attr:`authorised_detector_names`,
        which is the standard identifier for detector geometry
        calculations.

        Returns:
            Detector name string (e.g., ``"Eiger2M"``,
            ``"Maxipix"``, ``"Lambda750k"``).
        """
        return self.authorised_detector_names[0]

    @staticmethod
    def get_rocking_angle(angles: dict) -> str | None:
        """
        Identify which motor was scanned during rocking curve.

        Determines whether out-of-plane or in-plane angle was varied
        based on which array has more than one unique value. Used to
        automatically detect scan geometry.

        Args:
            angles: Dictionary with keys:

                - ``"sample_outofplane_angle"``: $\omega$ or $\eta$
                  values (scalar or array)
                - ``"sample_inplane_angle"``: $\chi$ or $\phi$ values
                  (scalar or array)

        Returns:
            Name of scanned angle key, or None if neither angle was
            scanned (single-frame measurement).

        Examples:
            Out-of-plane scan (typical):

            >>> angles = {
            ...     "sample_outofplane_angle": np.linspace(30.0, 30.5, 51),
            ...     "sample_inplane_angle": 0.0
            ... }
            >>> Loader.get_rocking_angle(angles)
            'sample_outofplane_angle'

            In-plane scan (grazing incidence):

            >>> angles = {
            ...     "sample_outofplane_angle": 2.0,
            ...     "sample_inplane_angle": np.linspace(-10, 10, 41)
            ... }
            >>> Loader.get_rocking_angle(angles)
            'sample_inplane_angle'
        """

    @staticmethod
    def format_scanned_counters(
        *counters: float | np.ndarray | list,
        scan_axis_roi: tuple[slice] = None,
        rocking_angle_binning: int = None,
    ):
        """
        Preprocess motor positions to match ROI and binning of data.

        Applies same binning and ROI selection to motor counter arrays
        as applied to detector data, maintaining synchronisation between
        intensity and position information.

        Args:
            *counters: One or more motor position values. Each can be:

                - **Scalar**: Fixed motor position (e.g., 30.0 degrees)
                - **Array**: Scanned motor positions (one per frame)

            scan_axis_roi: ROI slice along rocking curve axis (first
                dimension). Applied after binning. Typically
                ``(slice(start, stop),)``.
            rocking_angle_binning: Binning factor for scanned arrays.
                Scalar values are unaffected.

        Returns:
            Formatted counter(s) with same type as input. If multiple
            counters provided, returns tuple in same order. If single
            counter, returns that value directly.

        Examples:
            Single scanned angle with binning:

            >>> omega = np.linspace(30.0, 30.5, 100)
            >>> formatted = Loader.format_scanned_counters(
            ...     omega,
            ...     rocking_angle_binning=2
            ... )
            >>> # Returns array of length 50

            Multiple counters with ROI:

            >>> omega = np.linspace(30.0, 30.5, 100)
            >>> energy = 8.5  # fixed
            >>> omega_fmt, energy_fmt = Loader.format_scanned_counters(
            ...     omega, energy,
            ...     scan_axis_roi=(slice(10, 90),)
            ... )
            >>> # omega_fmt has 80 values, energy_fmt is 8.5
        """

        formatted_counters = []
        for counter in counters:
            if isinstance(counter, (list, np.ndarray)):
                formatted_counter = np.array(counter)  # ensure it's an ndarray

                # apply binning if required
                if rocking_angle_binning:
                    formatted_counter = bin_along_axis(
                        formatted_counter, rocking_angle_binning, "mean"
                    )

                # apply ROI slicing if available
                if scan_axis_roi is not None:
                    formatted_counter = formatted_counter[scan_axis_roi]

            else:  # handle scalar values (floats)
                formatted_counter = float(counter)  # ensure it's a float

            formatted_counters.append(formatted_counter)

        if len(formatted_counters) == 1:
            return formatted_counters[0]

        return tuple(formatted_counters)

    @classmethod
    def get_mask(
        cls,
        detector_name: str = None,
        channel: int = None,
        roi: tuple[slice] = None,
    ) -> np.ndarray:
        """
        Generate detector-specific bad pixel mask.

        Returns hardcoded masks for common BCDI detectors, marking chip
        gaps and known bad pixel regions. Masks are detector-specific
        due to different chip layouts and geometries.

        Args:
            detector_name: Detector identifier (case-insensitive).
                Supported detectors:

                - **Maxipix**: ``"maxipix"``, ``"mpxgaas"``, ``"mpx4inr"``
                - **Eiger2M**: ``"Eiger2M"``, ``"eiger2m"``
                - **Eiger4M**: ``"Eiger4M"``, ``"eiger4m"``, ``"e4m"``
                - **Eiger9M**: ``"eiger9m"``, ``"e9m"``
                - **Eiger500k**: ``"eiger500k"``, ``"e2500"``
                - **Merlin**: ``"merlin"``

                If None and called as instance method, uses
                ``self.detector_name``.

            channel: If provided, extends 2D mask to 3D by repeating
                along first axis (for 3D data). Specifies number of
                frames.
            roi: ROI applied after mask generation. See
                :meth:`_check_roi` for format. Typically
                ``(slice(y1,y2), slice(x1,x2))`` for 2D.

        Returns:
            Binary mask array (1 = bad pixel, 0 = good pixel). Shape is:

            - 2D: ``detector_shape`` if no ROI
            - 2D: cropped to ROI if provided
            - 3D: ``(channel, n_y, n_x)`` if channel specified

        Raises:
            ValueError: If ``detector_name`` is not recognized or if
                called as class method without providing ``detector_name``.

        Examples:
            Instance method (uses loader's detector):

            >>> loader = ID01Loader(scan=42, ...)
            >>> mask = loader.get_mask(channel=100)
            >>> # Returns (100, 2164, 1030) Eiger2M mask

            Class method with explicit detector:

            >>> mask = Loader.get_mask(detector_name="Maxipix")
            >>> # Returns (516, 516) Maxipix mask

            With ROI:

            >>> roi = (slice(100, 400), slice(200, 800))
            >>> mask = Loader.get_mask(
            ...     detector_name="Eiger2M",
            ...     roi=roi
            ... )
            >>> # Returns (300, 600) cropped mask

        Notes:
            Eiger masks include:

            - Chip gaps (horizontal and vertical)
            - Module boundaries
            - Known bad pixel clusters

            Maxipix masks include central cross gaps (256±3 pixels).
        """
        if detector_name is None:
            # Handling the case whenever the method is called as
            # a static method.
            local_params = locals()
            if isinstance(local_params[0], cls):
                detector_name = cls.detector_name
            else:
                raise ValueError(
                    "When called as a static method, detector_name must be "
                    "provided."
                )
        roi = cls._check_roi(roi)
        if len(roi) == 3:
            roi = roi[1:]
        if channel:
            roi = (slice(None),) + roi[-2:]

        if detector_name in (
            "maxipix",
            "Maxipix",
            "mpxgaas",
            "mpx4inr",
            "mpx1x4",
        ):
            mask = np.zeros(shape=(516, 516))
            mask[:, 255:261] = 1
            mask[255:261, :] = 1

        elif detector_name in ("Eiger2M", "eiger2m", "eiger2M", "Eiger2m"):
            mask = np.zeros(shape=(2164, 1030))
            mask[:, 255:259] = 1
            mask[:, 513:517] = 1
            mask[:, 771:775] = 1
            mask[0:257, 72:80] = 1
            mask[255:259, :] = 1
            mask[511:552, :] = 1
            mask[804:809, :] = 1
            mask[1061:1102, :] = 1
            mask[1355:1359, :] = 1
            mask[1611:1652, :] = 1
            mask[1905:1909, :] = 1
            mask[1248:1290, 478] = 1
            mask[1214:1298, 481] = 1
            mask[1649:1910, 620:628] = 1

        elif detector_name in ("Eiger4M", "eiger4m", "e4m"):
            mask = np.zeros(shape=(2167, 2070))
            mask[:, 0:1] = 1
            mask[:, -1:] = 1
            mask[0:1, :] = 1
            mask[-1:, :] = 1
            mask[:, 1029:1041] = 1
            mask[513:552, :] = 1
            mask[1064:1103, :] = 1
            mask[1615:1654, :] = 1

        # Having a name such as "eiger" is super bad, it's id27...
        elif detector_name.lower() in ("eiger9m", "e9m", "eiger"):
            mask = np.zeros(shape=(3262, 3108))
            mask[:, 0:1] = 1
            mask[:, -1:] = 1
            mask[0:1, :] = 1
            mask[-1:, :] = 1
            mask[:, 513:515] = 1
            mask[:, 1028:1040] = 1
            mask[:, 1553:1555] = 1
            mask[:, 2068:2080] = 1
            mask[:, 2593:2595] = 1
            mask[512:550, :] = 1
            mask[1062:1100, :] = 1
            mask[1612:1650, :] = 1
            mask[2162:2200, :] = 1
            mask[2712:2750, :] = 1

        elif detector_name.lower() in ("eiger500k", "e2500"):
            mask = np.zeros(shape=(512, 1028))
        elif detector_name.lower() == "merlin":
            mask = np.zeros(shape=(512, 512))
        else:
            raise ValueError(f"Invalid detector name: {detector_name}")
        if channel:
            mask = np.repeat(
                mask[
                    np.newaxis,
                    :,
                    :,
                ],
                channel,
                axis=0,
            )
        return mask[roi]

    @staticmethod
    def plot_detector_data(
        data: np.ndarray,
        title: str = None,
        return_fig: bool = False,
        equal_limits: bool = False,
        **plot_params,
    ) -> plt.Figure:
        """
        Quick visualisation of 2D or 3D detector data.

        Creates diagnostic plots showing orthogonal slices (for 3D data)
        or single 2D image. Uses log-scale colouring by default for
        dynamic range typical of BCDI diffraction patterns.

        Args:
            data: Detector data array. If 3D, shape is (n_frames, n_y,
                n_x). If 2D, shape is (n_y, n_x).
            title: Plot title displayed above figure. If None, no title
                shown.
            return_fig: If True, returns Figure object for further
                customisation. If False (default), displays figure
                interactively.
            equal_limits: If True, uses same axis limits for all
                subplots (helpful for comparing slice scales). If False,
                each plot uses its own optimal limits.
            **plot_params: Additional arguments passed to
                :func:`matplotlib.pyplot.imshow`. Defaults are:

                - ``norm="log"``: Logarithmic colour scale
                - ``origin="upper"``: [0,0] at top-left
                - ``cmap="turbo"``: Rainbow-like colourmap

        Returns:
            If ``return_fig=True``, returns matplotlib Figure object.
            Otherwise, displays interactively and returns None.

        Examples:
            Quick 3D data check:

            >>> data = loader.load_data(scan=42)
            >>> Loader.plot_detector_data(data, title="Scan 42")

            Custom colouring:

            >>> Loader.plot_detector_data(
            ...     data,
            ...     cmap="viridis",
            ...     norm="linear",
            ...     vmin=0,
            ...     vmax=1e5
            ... )

            Save figure for publication:

            >>> fig = Loader.plot_detector_data(data, return_fig=True)
            >>> fig.savefig("detector_scan42.png", dpi=300)

        Notes:
            For 3D data, creates 2×3 subplot grid:

            - Top row: Central slices along each axis
            - Bottom row: Sum projections along each axis

            This quickly reveals Bragg peak position and rocking curve
            quality.
        """
        _plot_params = {
            "norm": "log",
            "origin": "upper",
            "cmap": "turbo",  # "PuBu_r"
        }
        if plot_params:
            _plot_params.update(plot_params)

        if data.ndim == 3:
            limits = [
                (
                    s / 2 - np.max(data.shape) / 2,
                    s / 2 + np.max(data.shape) / 2,
                )
                for s in data.shape
            ]
            slices = get_centred_slices(data.shape)
            planes = ((1, 2), (0, 2), (1, 0))  # indexing convention

            fig, axes = plt.subplots(2, 3, layout="tight", figsize=(6, 4))
            for i, p in enumerate(planes):
                axes[0, i].imshow(
                    (
                        np.swapaxes(data[slices[i]], 0, 1)
                        if p[0] > p[1]
                        else data[slices[i]]
                    ),
                    **_plot_params,
                )
                axes[1, i].imshow(
                    (
                        np.swapaxes(data.sum(axis=i), 0, 1)
                        if p[0] > p[1]
                        else data.sum(axis=i)
                    ),
                    **_plot_params,
                )
                for ax in (axes[0, i], axes[1, i]):
                    add_colorbar(ax, ax.images[0])
                    if equal_limits:
                        ax.set_xlim(limits[p[1]])
                        if _plot_params["origin"] == "upper":
                            ax.set_ylim(limits[p[0]][1], limits[p[0]][0])
                        ax.set_ylim(limits[p[0]])

            for i in range(2):
                axes[i, 0].set_xlabel(r"axis$_{2}$, det. horiz.")
                axes[i, 0].set_ylabel(r"axis$_{1}$, det. vert.")

                axes[i, 1].set_xlabel(r"axis$_{2}$, det. horiz.")
                axes[i, 1].set_ylabel(r"axis$_{0}$, rocking curve")

                axes[i, 2].set_xlabel(r"axis$_{0}$, rocking curve")
                axes[i, 2].set_ylabel(r"axis$_{1}$, det. vert.")

            axes[0, 1].set_title("Intensity slice")
            axes[1, 1].set_title("Intensity sum")
            fig.suptitle(title)
        elif data.ndim == 2:
            pass
        else:
            raise ValueError(
                f"Invalid data shape (detector_data.shape={data.shape})."
                "Should be 2D or 3D."
            )
        if return_fig:
            return fig
        return None


def h5_safe_load(func: Callable) -> Callable:
    """A wrapper to safely load data in h5 file"""

    def wrap(self, *args, **kwargs):
        with silx.io.h5py_utils.File(self.experiment_file_path) as self.h5file:
            return func(self, *args, **kwargs)

    return wrap


class H5TypeLoader(Loader):
    """A child class of Loader for H5-type loaders."""

    def __init__(
        self,
        experiment_file_path: str,
        scan: int = None,
        sample_name: str = None,
        detector_name: str = None,
        flat_field: np.ndarray | str = None,
        alien_mask: np.ndarray | str = None,
    ) -> None:
        super().__init__(scan, sample_name, flat_field, alien_mask)
        self.experiment_file_path = experiment_file_path
        if detector_name is None:
            self.detector_name = self.get_detector_name()
        else:
            self.detector_name = detector_name

    @h5_safe_load
    def load_angles(self, key_path: str) -> dict:
        angles = {}
        for name in self.angle_names.values():
            if name is not None:
                angles[name] = self.h5file[key_path + name][()]

        return angles
