"""
BCDI measurement simulation with detector geometry.

This module provides a comprehensive simulator for Bragg Coherent
Diffraction Imaging (BCDI) measurements, including diffractometer
geometry, detector frame transformations, and realistic detector
effects.
"""

import numpy as np
from scipy.fft import fftfreq, fftn, fftshift, ifftshift

from cdiutils.converter import SpaceConverter
from cdiutils.geometry import Geometry
from cdiutils.io.loader import Loader
from cdiutils.plot.slice import plot_volume_slices
from cdiutils.utils import (
    energy_to_wavelength,
    get_reciprocal_voxel_size,
    symmetric_pad,
    transform_volume,
    wavelength_to_energy,
)

# import from simulation sub-package
from .noise import add_noise
from .objects import (
    add_linear_phase,
    add_quadratic_phase,
    add_random_phase,
    make_box,
    make_cylinder,
    make_ellipsoid,
)


def get_phase_factor(
    measurement_frame_shape: tuple[int, int, int],
    bragg_angle: float,
    measurement_frame_voxel_size: tuple[float, float, float],
    direct_lab_frame_voxel_size: tuple[float, float, float],
    shear_plane_axes: tuple[int, int] | list[int] = (0, 2),
    do_fftshift: bool = False,
) -> np.ndarray:
    """
    Compute phase factor for shear FFT transformation.

    This function calculates the phase modulation needed to perform
    a shear transformation in Fourier space, which relates the
    reciprocal space grid to the detector frame in BCDI geometry.

    Args:
        measurement_frame_shape: Shape of the 3D measurement frame
            (nframes, ny, nx).
        bragg_angle: Bragg angle in degrees.
        measurement_frame_voxel_size: Voxel sizes in reciprocal
            space (qz, qy, qx) in inverse metres.
        direct_lab_frame_voxel_size: Voxel sizes in direct space
            (z, y, x) in metres.
        shear_plane_axes: Axes defining the shear plane, as
            (propagation_axis, sheared_axis). Default is (0, 2).
        do_fftshift: Whether to apply fftshift to the phase factor
            along the propagation axis. Default is False.

    Returns:
        Phase factor array with same shape as measurement frame.

    Example:
        >>> phase = get_phase_factor(
        ...     measurement_frame_shape=(100, 100, 100),
        ...     bragg_angle=30.5,
        ...     measurement_frame_voxel_size=(1e7, 1e7, 1e7),
        ...     direct_lab_frame_voxel_size=(10e-9, 10e-9, 10e-9),
        ... )
    """
    propagation_axis = shear_plane_axes[0]
    sheared_axis = shear_plane_axes[1]

    # create frequency grids for each dimension
    grid = np.meshgrid(
        *[fftshift(fftfreq(s)) * s for s in measurement_frame_shape],
        indexing="ij",
    )

    # compute shear slope from Bragg angle
    slope = -measurement_frame_voxel_size[propagation_axis] * np.tan(
        np.radians(bragg_angle)
    )

    # compute phase modulation
    phase_factor = np.exp(
        1j
        * slope
        * grid[sheared_axis]
        * direct_lab_frame_voxel_size[sheared_axis]
        * grid[propagation_axis]
    )

    if do_fftshift:
        phase_factor = fftshift(phase_factor, axes=propagation_axis)

    return phase_factor


def shear_fft(
    wavefront: np.ndarray,
    phase_factor: np.ndarray,
    propagation_axis: int = 0,
    handle_fftshift: bool = True,
) -> np.ndarray:
    """
    Apply shear transformation using FFT.

    This function performs a shear transformation in Fourier space
    by applying a phase factor, which is used to transform from
    reciprocal space to detector frame in BCDI geometry.

    Args:
        wavefront: Complex wavefront to transform.
        phase_factor: Phase modulation factor from
            :func:`get_phase_factor`.
        propagation_axis: Axis along which X-rays propagate
            (rocking axis). Default is 0.
        handle_fftshift: Whether to apply fftshift before and
            ifftshift after transformation. Default is True.

    Returns:
        Transformed wavefront in detector frame.

    Notes:
        This is an alternative to matrix-based transformation and
        can be faster for large datasets. However, it may be less
        accurate for non-orthogonal geometries.
    """
    if handle_fftshift:
        wavefront = fftshift(wavefront)
        phase_factor = fftshift(phase_factor)

    # FFT along propagation axis
    wavefront = fftn(wavefront, axes=(propagation_axis,))
    wavefront *= phase_factor

    # FFT along remaining axes
    fft_axes = tuple(i for i in range(wavefront.ndim) if i != propagation_axis)
    wavefront = fftn(wavefront, axes=fft_axes)

    if handle_fftshift:
        wavefront = ifftshift(wavefront)

    return wavefront


def shift_no_wrap(
    data: np.ndarray,
    shift: tuple[float, float],
) -> np.ndarray:
    """
    Shift 3D data without circular wrapping.

    This function shifts detector data in the spatial dimensions
    (y, x) without wrapping around the edges. Regions that would
    wrap are left as zeros.

    Args:
        data: 3D array with shape (frames, height, width).
        shift: Shift amounts as (shift_y, shift_x) in pixels.
            Positive values shift down/right, negative values shift
            up/left. Non-integer values are rounded to nearest
            integer.

    Returns:
        Shifted data with same shape as input. Wrapped regions are
        filled with zeros.

    Raises:
        TypeError: If data is not a numpy array.
        ValueError: If data is not 3D.

    Example:
        >>> data = np.ones((10, 512, 512))
        >>> shifted = shift_no_wrap(data, shift=(5, -3))
        >>> shifted.shape
        (10, 512, 512)
    """
    # validate inputs
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be np.ndarray, got {type(data)}")

    if data.ndim != 3:
        raise ValueError(
            f"data must be 3D (frames, height, width), got shape {data.shape}"
        )

    # round shifts to integers
    shift_y = int(round(shift[0]))
    shift_x = int(round(shift[1]))

    # trivial case: no shift
    if shift_y == 0 and shift_x == 0:
        return data.copy()

    # extract spatial dimensions (ignore frames dimension)
    spatial_shape = data.shape[1:]

    # initialise output with zeros
    shifted = np.zeros_like(data)

    # initialise source and destination slice lists
    source_slices = [slice(None) for _ in range(2)]
    dest_slices = [slice(None) for _ in range(2)]

    # calculate slices for each spatial dimension
    for i, shift_val in enumerate([shift_y, shift_x]):
        if shift_val >= 0:
            # positive shift: move data forward
            source_slices[i] = slice(0, spatial_shape[i] - shift_val)
            dest_slices[i] = slice(shift_val, spatial_shape[i])
        else:
            # negative shift: move data backward
            source_slices[i] = slice(-shift_val, spatial_shape[i])
            dest_slices[i] = slice(0, spatial_shape[i] + shift_val)

    # apply the shift (keep all frames)
    shifted[:, dest_slices[0], dest_slices[1]] = data[
        :, source_slices[0], source_slices[1]
    ]

    return shifted


class BCDISimulator:
    """
    Simulator for BCDI measurement with realistic detector geometry.

    This class provides end-to-end simulation of a BCDI measurement,
    including:
    - Sample object creation with customisable geometry and phase
    - Diffraction pattern computation
    - Diffractometer angle calculations
    - Coordinate transformations (Q-space ↔ detector frame)
    - Realistic detector effects (noise, masking, binning)

    The simulator handles the full measurement geometry including
    Bragg angle, detector angles, and rocking curves, making it
    suitable for testing reconstruction algorithms and planning
    experiments.

    Attributes:
        geometry: Diffractometer geometry configuration.
        energy: X-ray energy in eV.
        wavelength: X-ray wavelength in metres.
        structure: Crystal structure ('cubic', etc.).
        hkl: Miller indices of the Bragg reflection.
        lattice_parameter: Lattice parameter in metres.
        det_calib_params: Detector calibration parameters (distance,
            pixel size, centre position).
        detector_name: Name of the detector ('maxipix', etc.).
        mask: Detector mask (dead pixels, gaps).
        detector_shape: Shape of the detector (height, width).
        num_frames: Number of frames in rocking curve.
        target_peak_position: Target pixel position for Bragg peak
            (row, col).
        obj: Simulated 3D object (complex array).
        intensity: Diffraction intensity in reciprocal space.
        voxel_size: Real-space voxel sizes (z, y, x) in metres.
        q_voxel_size: Reciprocal-space voxel sizes (qz, qy, qx) in
            inverse metres.
        all_angles: Dictionary of computed diffractometer angles.
        diffractometer_angles: Dictionary of measurement angles
            (sample + detector).
        detector_to_q_matrix: Transformation matrix from detector
            to Q-space.
        q_to_detector_matrix: Transformation matrix from Q-space to
            detector.
        phase_factor: Phase modulation for shear FFT.

    Example:
        >>> # create simulator for ID01 beamline
        >>> sim = BCDISimulator(
        ...     energy=9000,  # 9 keV
        ...     structure='cubic',
        ...     hkl=[1, 1, 1],
        ...     lattice_parameter=4.08e-10,  # gold
        ...     detector_name='maxipix',
        ...     num_frames=200,
        ... )
        >>>
        >>> # simulate a box-shaped particle with random phase
        >>> sim.simulate_object(
        ...     shape=(100, 100, 100),
        ...     voxel_size=5e-9,
        ...     geometric_shape='box',
        ...     geometric_shape_params={'dimensions': 30},
        ...     phase_type='random',
        ...     phase_params={'amplitude': 0.2},
        ... )
        >>>
        >>> # set up measurement geometry
        >>> bragg = sim.lattice_parameter_to_bragg_angle()
        >>> detector_angles = sim.get_detector_angles(
        ...     scattering_angle=2 * bragg
        ... )
        >>> sim.set_measurement_params(
        ...     bragg_angle=bragg,
        ...     rocking_range=0.5,
        ...     detector_angles=detector_angles,
        ... )
        >>>
        >>> # transform to detector frame and add noise
        >>> detector_data = sim.to_detector_frame()
        >>> realistic_data = sim.get_realistic_detector_data(
        ...     detector_data,
        ...     photon_budget=1e10,
        ... )
    """

    def __init__(
        self,
        geometry: Geometry | None = None,
        energy: float | None = None,
        wavelength: float | None = None,
        structure: str | None = None,
        hkl: list[int] | None = None,
        lattice_parameter: float | None = None,
        det_calib_params: dict | None = None,
        detector_name: str | None = None,
        num_frames: int = 256,
        target_peak_position: tuple[int, int] = (258, 258),
    ):
        """
        Initialise BCDI measurement simulator.

        Args:
            geometry: Diffractometer geometry. If None, uses ID01
                default geometry.
            energy: X-ray energy in eV. Mutually exclusive with
                wavelength.
            wavelength: X-ray wavelength in metres. Mutually
                exclusive with energy.
            structure: Crystal structure type. Default is 'cubic'.
            hkl: Miller indices [h, k, l]. Default is [1, 1, 1].
            lattice_parameter: Lattice parameter in metres.
            det_calib_params: Dictionary with detector calibration
                parameters: 'distance' (m), 'pwidth1', 'pwidth2'
                (m), 'cch1', 'cch2' (pixels).
            detector_name: Detector name for loading mask. Default
                is 'maxipix'.
            num_frames: Number of frames in rocking curve. Default
                is 256.
            target_peak_position: Target pixel position (row, col)
                for Bragg peak. Default is (258, 258).

        Raises:
            ValueError: If both energy and wavelength are provided
                and don't match, or if neither is provided.
        """
        # set up geometry
        if geometry is None:
            self.geometry = Geometry.from_setup("id01")
        else:
            self.geometry = geometry

        # set up energy/wavelength
        if (
            energy is not None
            and wavelength is not None
            and not np.isclose(
                wavelength,
                energy_to_wavelength(energy),
                rtol=1e-6,
            )
        ):
            raise ValueError(
                "Provided energy and wavelength do not match. "
                "You can provide only one of them."
            )

        if energy is not None:
            self.energy = energy  # energy in eV
            self.wavelength = energy_to_wavelength(energy)  # metres
        elif wavelength is not None:
            self.wavelength = wavelength  # wavelength in metres
            self.energy = wavelength_to_energy(wavelength)  # eV
        else:
            raise ValueError("Either energy or wavelength must be provided.")

        # set up crystal structure
        self.structure = structure if structure is not None else "cubic"
        self.hkl = hkl if hkl is not None else [1, 1, 1]
        self.lattice_parameter = lattice_parameter

        # set up detector parameters
        self.det_calib_params = det_calib_params
        self.detector_name = (
            detector_name if detector_name is not None else "maxipix"
        )
        self.mask = Loader.get_mask(self.detector_name)
        self.detector_shape = self.mask.shape
        self.num_frames = num_frames
        self.target_peak_position = target_peak_position

        # initialise angle-related attributes
        self.all_angles = None

        # initialise object-related attributes
        self.obj = None
        self.intensity = None
        self.voxel_size = None
        self.q_voxel_size = None

        # initialise measurement-related attributes
        self.diffractometer_angles = {}
        self.detector_to_q_matrix = None
        self.q_to_detector_matrix = None
        self.phase_factor = None

        # ROI size for transformation matrix computation
        self.roi_length_for_matrix_computation = (150, 150)

    def lattice_parameter_to_bragg_angle(
        self,
        lattice_parameter: float | None = None,
    ) -> float:
        """
        Calculate Bragg angle from lattice parameter.

        Uses Bragg's law (2d sin(θ) = λ) to compute the Bragg angle
        for the specified reflection in specular geometry.

        Args:
            lattice_parameter: Lattice parameter in metres. If None,
                uses the instance attribute.

        Returns:
            Bragg angle in degrees.

        Raises:
            NotImplementedError: If structure is not 'cubic'.

        Example:
            >>> sim = BCDISimulator(
            ...     energy=9000,
            ...     structure='cubic',
            ...     hkl=[1, 1, 1],
            ...     lattice_parameter=4.08e-10,
            ... )
            >>> bragg = sim.lattice_parameter_to_bragg_angle()
            >>> print(f"{bragg:.2f}°")
        """
        if lattice_parameter is None:
            lattice_parameter = self.lattice_parameter

        if self.structure == "cubic":
            d_spacing = lattice_parameter / np.sqrt(
                np.sum(np.array(self.hkl) ** 2)
            )
        else:
            raise NotImplementedError("Only cubic structure is implemented.")

        # Bragg's law: 2d sin(θ) = λ
        theta = np.arcsin(self.wavelength / (2 * d_spacing))

        return np.degrees(theta)

    def bragg_angle_to_lattice_parameter(
        self,
        bragg_angle: float,
    ) -> float:
        """
        Calculate lattice parameter from Bragg angle.

        Inverts Bragg's law to compute lattice parameter from the
        measured Bragg angle.

        Args:
            bragg_angle: Bragg angle in degrees.

        Returns:
            Lattice parameter in metres.

        Raises:
            NotImplementedError: If structure is not 'cubic'.

        Example:
            >>> sim = BCDISimulator(
            ...     energy=9000,
            ...     structure='cubic',
            ...     hkl=[1, 1, 1],
            ... )
            >>> a = sim.bragg_angle_to_lattice_parameter(20.5)
            >>> print(f"{a*1e10:.3f} Å")
        """
        bragg_angle_rad = np.radians(bragg_angle)

        # Bragg's law: 2d sin(θ) = λ
        d_spacing = self.wavelength / (2 * np.sin(bragg_angle_rad))

        if self.structure == "cubic":
            lattice_parameter = d_spacing * np.sqrt(
                np.sum(np.array(self.hkl) ** 2)
            )
        else:
            raise NotImplementedError("Only cubic structure is implemented.")

        return lattice_parameter

    def get_angular_offsets(
        self,
        target_peak_position: tuple[int, int] | None = None,
    ) -> tuple[float, float]:
        """
        Compute angular offsets to centre Bragg peak at target pixel.

        Calculates the detector angle corrections needed to place
        the Bragg peak at the specified pixel position, accounting
        for the difference from the calibrated detector centre.

        Note that the sign convention here is specific to ID01 geometry,
        the pixel count increases opposite to delta and nu angles.

        Args:
            target_peak_position: Target pixel position (row, col).
                If None, uses instance attribute.

        Returns:
            Angular offsets (delta_offset, nu_offset) in degrees.

        Example:
            >>> sim = BCDISimulator(
            ...     energy=9000,
            ...     det_calib_params={
            ...         'distance': 1.0,
            ...         'pwidth1': 55e-6,
            ...         'pwidth2': 55e-6,
            ...         'cch1': 256,
            ...         'cch2': 256,
            ...     },
            ... )
            >>> offsets = sim.get_angular_offsets((280, 240))
            >>> print(f"Delta: {offsets[0]:.3f}°, Nu: {offsets[1]:.3f}°")

        Notes:
            This is specific to ID01 geometry where pixel count
            increases opposite to delta angle.
        """
        if target_peak_position is None:
            target_peak_position = self.target_peak_position

        # compute pixel offset from calibrated centre
        pixel_offset = tuple(
            target_peak_position[i] - self.det_calib_params[f"cch{i + 1}"]
            for i in range(2)
        )

        # convert pixel offset to angular offset
        # (ID01-specific: pixel count increases opposite to delta)
        delta_offset = +np.arctan(
            pixel_offset[0]
            * self.det_calib_params["pwidth1"]
            / self.det_calib_params["distance"]
        )

        nu_offset = +np.arctan(
            pixel_offset[1]
            * self.det_calib_params["pwidth2"]
            / self.det_calib_params["distance"]
        )

        return np.degrees(delta_offset), np.degrees(nu_offset)

    @staticmethod
    def _convert_to_unit(
        *angles: float | None,
        unit: str,
        **dict_angles: float,
    ) -> float | tuple[float, ...] | dict[str, float] | None:
        """
        Convert angles to specified unit (degrees or radians).

        Args:
            *angles: Variable number of angle values to convert.
            unit: Target unit ('degrees', 'deg', 'd' or 'radians',
                'rad', 'r').
            **dict_angles: Dictionary of named angles to convert.

        Returns:
            Converted angles in same format as input (single value,
            tuple, or dict).

        Raises:
            ValueError: If unit is not recognised.

        Example:
            >>> # convert single angle
            >>> rad = BCDISimulator._convert_to_unit(
            ...     30.0, unit='radians'
            ... )
            >>>
            >>> # convert multiple angles
            >>> rad_tuple = BCDISimulator._convert_to_unit(
            ...     30.0, 45.0, unit='radians'
            ... )
            >>>
            >>> # convert dictionary
            >>> rad_dict = BCDISimulator._convert_to_unit(
            ...     unit='radians',
            ...     delta=30.0,
            ...     nu=45.0,
            ... )
        """
        if unit.lower() in ("d", "deg", "degrees"):
            func = np.degrees
        elif unit.lower() in ("radians", "r", "rad"):
            func = np.radians
        else:
            raise ValueError(
                f"unit must be 'radians' or 'degrees', got '{unit}'"
            )

        # convert positional arguments
        if len(angles) > 0:
            converted = tuple(
                func(a) if a is not None else None for a in angles
            )
            return converted if len(converted) > 1 else converted[0]

        # convert dictionary arguments
        if dict_angles:
            return {k: func(v) for k, v in dict_angles.items()}

        return None

    def compute_angles(
        self,
        target_peak_position: tuple[int, int] | None = None,
        scattering_angle: float | None = None,
        detector_outofplane_angle: float | None = None,
        detector_inplane_angle: float | None = None,
    ) -> None:
        """
        Compute consistent diffractometer angles.

        Given partial angle information, compute all related angles
        in a self-consistent way, accounting for detector offset
        from calibrated centre. Results are stored in
        ``self.all_angles``.

        Here we consider the effective angles to be the physical
        angles at the desired self.peak_position. They are the actual
        angles corresponding to that position. They are defined as:
        **effective angles = detector angles - angular offsets**
        The detector calibration gives the direct beam position on the
        detector, which corresponds to where the diffractometer angles
        are effective (at this position the effective angles = detector
        angles). Refer to :func:`get_angular_offsets` for more details.

        Args:
            target_peak_position: Target pixel position (row, col).
                If None, uses instance attribute.
            scattering_angle: Total scattering angle (2θ) in
                degrees.
            detector_outofplane_angle: Detector out-of-plane angle
                (delta at ID01) in degrees.
            detector_inplane_angle: Detector in-plane angle (nu at
                ID01) in degrees.

        Raises:
            ValueError: If angles are inconsistent or insufficient
                information is provided.

        Example:
            >>> sim = BCDISimulator(energy=9000)
            >>> # compute detector angles from scattering angle
            >>> sim.compute_angles(scattering_angle=41.0)
            >>> print(sim.all_angles)

        Notes:
            At least one angle must be provided. The function will
            compute missing angles from the provided information.
        """
        # convert angles to radians for computation
        (
            scattering_angle,
            detector_outofplane_angle,
            detector_inplane_angle,
        ) = self._convert_to_unit(
            scattering_angle,
            detector_outofplane_angle,
            detector_inplane_angle,
            unit="rad",
        )

        # get angular offsets (in degrees from method)
        angular_offsets = self.get_angular_offsets(target_peak_position)
        # convert to radians for computation
        angular_offsets = self._convert_to_unit(
            angular_offsets[0], angular_offsets[1], unit="rad"
        )

        effective_angles = [0, 0]
        detector_angles = [
            detector_outofplane_angle,
            detector_inplane_angle,
        ]

        # compute missing angles based on provided information
        if (
            detector_outofplane_angle is None
            and detector_inplane_angle is None
        ):
            # both detector angles missing: use scattering angle
            effective_angles = [scattering_angle, 0.0]  # delta, nu
            # effective_angles = [
            #     d - o for d, o in zip(detector_angles, angular_offsets)
            # ]
        elif scattering_angle is None:
            # scattering angle missing: compute from detector angles
            if (
                detector_outofplane_angle is None
                or detector_inplane_angle is None
            ):
                raise ValueError(
                    "If scattering angle is not provided, both "
                    "detector angles must be given."
                )
            effective_angles = [
                d - o for d, o in zip(detector_angles, angular_offsets)
            ]
            scattering_angle = np.arccos(
                np.cos(effective_angles[0]) * np.cos(effective_angles[1])
            )
        elif detector_outofplane_angle is None:
            # out-of-plane angle missing
            effective_angles[1] = detector_angles[1] - angular_offsets[1]
            effective_angles[0] = np.arccos(
                np.cos(scattering_angle) / np.cos(effective_angles[1])
            )
        elif detector_inplane_angle is None:
            # in-plane angle missing
            effective_angles[0] = detector_angles[0] - angular_offsets[0]
            effective_angles[1] = np.arccos(
                np.cos(scattering_angle) / np.cos(effective_angles[0])
            )
        else:
            # all angles provided: check consistency
            effective_angles = [
                d - o for d, o in zip(detector_angles, angular_offsets)
            ]
            computed_scattering_angle = np.arccos(
                np.cos(effective_angles[0]) * np.cos(effective_angles[1])
            )
            if not np.isclose(
                computed_scattering_angle,
                scattering_angle,
                rtol=1e-6,
            ):
                raise ValueError(
                    "Provided angles are inconsistent: "
                    f"outofplane_angle={np.degrees(detector_outofplane_angle):.3f}° "
                    f"and inplane_angle={np.degrees(detector_inplane_angle):.3f}° "
                    f"implies scattering_angle="
                    f"{np.degrees(computed_scattering_angle):.3f}°, "
                    f"but scattering_angle={np.degrees(scattering_angle):.3f}° "
                    "was provided."
                )

        # recompute detector angles with offsets
        detector_angles = [
            e + o for e, o in zip(effective_angles, angular_offsets)
        ]

        # store all computed angles
        self.all_angles = dict(
            scattering_angle=scattering_angle,
            effective_outofplane_angle=effective_angles[0],
            effective_inplane_angle=effective_angles[1],
            detector_outofplane_angle=detector_angles[0],
            detector_inplane_angle=detector_angles[1],
        )

        # convert all angles back to degrees for storage
        self.all_angles = self._convert_to_unit(unit="deg", **self.all_angles)

    def get_detector_angles(
        self,
        target_peak_position: tuple[int, int] | None = None,
        scattering_angle: float | None = None,
        detector_outofplane_angle: float | None = None,
        detector_inplane_angle: float | None = None,
    ) -> dict[str, float]:
        """
        Get detector angles for measurement.

        Convenience method that computes angles and returns only the
        detector-specific angles (out-of-plane and in-plane).

        Args:
            target_peak_position: Target pixel position (row, col).
                If None, uses instance attribute.
            scattering_angle: Total scattering angle (2θ) in
                degrees.
            detector_outofplane_angle: Detector out-of-plane angle
                (delta at ID01) in degrees.
            detector_inplane_angle: Detector in-plane angle (nu at
                ID01) in degrees.

        Returns:
            Dictionary with keys 'detector_outofplane_angle' and
            'detector_inplane_angle' in degrees.

        Example:
            >>> sim = BCDISimulator(energy=9000)
            >>> angles = sim.get_detector_angles(
            ...     scattering_angle=41.0
            ... )
            >>> print(f"Delta: {angles['detector_outofplane_angle']:.3f}°")
            >>> print(f"Nu: {angles['detector_inplane_angle']:.3f}°")
        """
        self.compute_angles(
            target_peak_position,
            scattering_angle,
            detector_outofplane_angle,
            detector_inplane_angle,
        )
        return {
            k: self.all_angles[k]
            for k in (
                "detector_outofplane_angle",
                "detector_inplane_angle",
            )
        }

    def simulate_object(
        self,
        shape: tuple | list | np.ndarray = (100, 100, 100),
        voxel_size: float | tuple[float, float, float] = 10e-9,
        geometric_shape: str | None = None,
        geometric_shape_params: dict | None = None,
        phase_type: str | None = None,
        phase_params: dict | None = None,
        swap_convention: bool = False,
        plot: bool = True,
    ) -> None:
        """
        Create simulated object and compute its diffraction pattern.

        Generates a 3D object with specified geometry and phase,
        computes its diffraction pattern, and optionally plots the
        results. The object and intensity are stored as instance
        attributes.

        Args:
            shape: Shape of 3D array (nz, ny, nx).
            voxel_size: Real-space voxel size in metres. Can be a
                single value (isotropic) or tuple (z, y, x).
            geometric_shape: Shape type: 'box', 'cylinder', or
                'ellipsoid'. Default is 'box'.
            geometric_shape_params: Parameters for shape function
                (e.g., {'dimensions': 30, 'rotation': (0, 0, 45)}).
            phase_type: Phase type: 'linear', 'quadratic', or
                'random'. Default is 'random'.
            phase_params: Parameters for phase function (e.g.,
                {'amplitude': 0.2}).
            swap_convention: If True, swap axes to match detector
                convention. Default is False.
            plot: If True, plot object and diffraction pattern.
                Default is True.

        Raises:
            ValueError: If geometric_shape is not recognised.

        Example:
            >>> sim = BCDISimulator(energy=9000)
            >>> sim.simulate_object(
            ...     shape=(80, 80, 80),
            ...     voxel_size=8e-9,
            ...     geometric_shape='ellipsoid',
            ...     geometric_shape_params={
            ...         'semi_axes': (25, 20, 20)
            ...     },
            ...     phase_type='random',
            ...     phase_params={'amplitude': 0.15},
            ... )

        Notes:
            The diffraction is computed with conjugate of the object
            to match crystallographic convention (Bragg CDI).
        """
        # handle voxel size
        if isinstance(voxel_size, (float, int)):
            voxel_size = [voxel_size] * 3

        # create geometric shape
        if geometric_shape is None:
            geometric_shape = "box"

        if geometric_shape.lower() == "box":
            make_function = make_box
        elif geometric_shape.lower() == "cylinder":
            make_function = make_cylinder
        elif geometric_shape.lower() == "ellipsoid":
            make_function = make_ellipsoid
        else:
            raise ValueError(
                f"Unknown geometric shape '{geometric_shape}', select "
                "one among: 'box', 'cylinder', 'ellipsoid'"
            )

        self.obj = make_function(shape=shape, **(geometric_shape_params or {}))
        self.voxel_size = voxel_size

        # add phase
        if phase_type is None:
            phase_type = "random"

        if phase_type.lower() == "quadratic":
            add_phase_function = add_quadratic_phase
        elif phase_type.lower() == "linear":
            add_phase_function = add_linear_phase
        elif phase_type.lower() == "random":
            add_phase_function = add_random_phase
        else:
            raise ValueError(
                f"Unknown phase type '{phase_type}', select one "
                "among: 'linear', 'quadratic', 'random'"
            )

        self.obj = add_phase_function(self.obj, **(phase_params or {}))

        # optionally swap convention
        if swap_convention:
            self.obj = Geometry.swap_convention(self.obj)
            self.voxel_size = Geometry.swap_convention(voxel_size)

        # compute diffraction (Bragg CDI convention: use conjugate)
        self.intensity = np.abs(fftshift(fftn(np.conj(self.obj)))) ** 2
        self.q_voxel_size = get_reciprocal_voxel_size(voxel_size, shape)

        # optional plotting
        if plot:
            alphas = np.abs(self.obj) / np.max(np.abs(self.obj))
            plot_volume_slices(
                np.abs(self.obj),
                title="Simulated object amplitude",
                voxel_size=self.voxel_size,
                data_centre=(0, 0, 0),
            )
            plot_volume_slices(
                np.angle(self.obj),
                title="Simulated object phase",
                alpha=alphas,
                cmap="cet_CET_C9s_r",
                voxel_size=self.voxel_size,
                data_centre=(0, 0, 0),
            )
            plot_volume_slices(
                self.intensity,
                title="Simulated diffraction intensity",
                norm="log",
                voxel_size=self.q_voxel_size,
                data_centre=(0, 0, 0),
            )

    def set_object(
        self,
        obj: np.ndarray,
        voxel_size: float | tuple[float, float, float],
    ) -> None:
        """
        Set object directly and compute diffraction pattern.

        Use this method to provide a custom object instead of using
        :meth:`simulate_object`. The diffraction pattern is
        computed automatically.

        Args:
            obj: Complex 3D object array.
            voxel_size: Real-space voxel size in metres. Can be a
                single value (isotropic) or tuple (z, y, x).

        Example:
            >>> # create custom object
            >>> obj = np.ones((64, 64, 64), dtype=complex)
            >>> obj *= np.exp(1j * np.random.rand(64, 64, 64))
            >>>
            >>> sim = BCDISimulator(energy=9000)
            >>> sim.set_object(obj, voxel_size=10e-9)
        """
        self.obj = obj

        if isinstance(voxel_size, (int, float)):
            voxel_size = [voxel_size] * 3

        self.voxel_size = voxel_size

        # compute diffraction (Bragg CDI convention: use conjugate)
        self.intensity = np.abs(fftshift(fftn(np.conj(self.obj)))) ** 2
        self.q_voxel_size = get_reciprocal_voxel_size(voxel_size, obj.shape)

    @staticmethod
    def get_rocking_angles(
        bragg_angle: float,
        rocking_range: float,
        num_frames: int,
    ) -> np.ndarray:
        """
        Generate linearly-spaced rocking angles.

        Creates an array of angles for a rocking curve centred on
        the Bragg angle.

        Args:
            bragg_angle: Centre angle (Bragg angle) in degrees.
            rocking_range: Total angular range to cover in degrees.
            num_frames: Number of angular steps.

        Returns:
            Array of rocking angles in degrees.

        Example:
            >>> angles = BCDISimulator.get_rocking_angles(
            ...     bragg_angle=20.5,
            ...     rocking_range=0.5,
            ...     num_frames=200,
            ... )
            >>> angles.shape
            (200,)
            >>> print(f"Range: {angles[0]:.3f}° to {angles[-1]:.3f}°")
        """
        return np.linspace(
            bragg_angle - rocking_range / 2,
            bragg_angle + rocking_range / 2,
            num_frames,
        )

    def _get_roi(self) -> tuple[int, int, int, int]:
        """
        Get ROI for transformation matrix computation.

        Returns:
            ROI as (row_start, row_end, col_start, col_end).
        """
        return (
            self.target_peak_position[0]
            - self.roi_length_for_matrix_computation[0] // 2,
            self.target_peak_position[0]
            + self.roi_length_for_matrix_computation[0] // 2,
            self.target_peak_position[1]
            - self.roi_length_for_matrix_computation[1] // 2,
            self.target_peak_position[1]
            + self.roi_length_for_matrix_computation[1] // 2,
        )

    def set_measurement_params(
        self,
        bragg_angle: float,
        rocking_range: float,
        detector_angles: dict[str, float],
        rocking_angle: str | None = None,
    ) -> None:
        """
        Set measurement parameters and compute transformation matrices.

        Configures the rocking curve angles and detector position,
        then computes the transformation matrices needed to convert
        between Q-space and detector frame.

        Args:
            bragg_angle: Bragg angle in degrees.
            rocking_range: Total rocking range in degrees.
            detector_angles: Dictionary with
                'detector_outofplane_angle' and
                'detector_inplane_angle' in degrees.
            rocking_angle: Rocking axis ('outofplane' or 'inplane').
                Default is 'outofplane'.

        Raises:
            NotImplementedError: If rocking_angle is not
                'outofplane'.

        Example:
            >>> sim = BCDISimulator(energy=9000)
            >>> # ... simulate object ...
            >>> detector_angles = sim.get_detector_angles(
            ...     scattering_angle=41.0
            ... )
            >>> sim.set_measurement_params(
            ...     bragg_angle=20.5,
            ...     rocking_range=0.5,
            ...     detector_angles=detector_angles,
            ... )

        Notes:
            This method must be called after :meth:`simulate_object`
            or :meth:`set_object` as it requires the object's
            reciprocal space voxel sizes.
        """
        if rocking_angle is None:
            rocking_angle = "outofplane"

        if rocking_angle.lower() == "outofplane":
            self.diffractometer_angles["sample_outofplane_angle"] = (
                self.get_rocking_angles(
                    bragg_angle,
                    rocking_range,
                    num_frames=self.num_frames,
                )
            )
            self.diffractometer_angles["sample_inplane_angle"] = 0
        else:
            raise NotImplementedError(
                f"The option '{rocking_angle}' for rocking_angle is "
                "not implemented yet"
            )

        self.diffractometer_angles.update(detector_angles)

        # compute transformation matrices
        roi = self._get_roi()

        converter = SpaceConverter(
            self.geometry,
            det_calib_params=self.det_calib_params,
            energy=self.energy,
            roi=roi,
        )

        converter.init_q_space(**self.diffractometer_angles)
        self.detector_to_q_matrix = converter.get_transformation_matrix()
        self.q_to_detector_matrix = (
            np.linalg.inv(self.detector_to_q_matrix)
            * np.array(self.q_voxel_size)
            * 1e-10  # convert from 1/m to 1/Å
        )

        # compute phase factor for shear FFT
        self.phase_factor = get_phase_factor(
            measurement_frame_shape=self.intensity.shape,
            bragg_angle=bragg_angle,
            measurement_frame_voxel_size=self.q_voxel_size,
            direct_lab_frame_voxel_size=self.voxel_size,
            shear_plane_axes=(0, 2),
        )

    def to_detector_frame(
        self,
        method: str | None = None,
        output_shape: tuple[int, int, int] | None = None,
        plot: bool = True,
    ) -> np.ndarray:
        """
        Transform reciprocal space intensity to detector frame.

        Applies coordinate transformation to map the simulated
        diffraction pattern from reciprocal space coordinates to
        detector pixel coordinates.

        Args:
            method: Transformation method: 'matrix_transform' or
                'shear_fft'. Default is 'matrix_transform'. For now,
                only 'matrix_transform' is implemented.
            output_shape: Desired output shape. Default is same as
                detector frame shape.
            plot: If True, plot the transformed intensity. Default
                is True.

        Returns:
            Intensity in detector frame coordinates.

        Raises:
            ValueError: If method is not recognised or if
                'shear_fft' is requested (not yet implemented).

        Example:
            >>> sim = BCDISimulator(energy=9000)
            >>> # ... simulate object and set measurement params ...
            >>> detector_data = sim.to_detector_frame()

        Notes:
            The 'shear_fft' method is faster but currently has
            implementation issues. Use 'matrix_transform' for now.
        """
        if method is None:
            method = "matrix_transform"

        output_shape = (self.num_frames,) + self.detector_shape

        if method.lower() == "matrix_transform":
            detector_frame_intensity = transform_volume(
                self.intensity,
                self.q_to_detector_matrix,
                output_shape=output_shape,
            )
        elif method.lower() == "shear_fft":
            raise ValueError(
                "shear_fft method not implemented yet (still buggy)"
            )
            # TODO: fix shear_fft implementation
        else:
            raise ValueError(
                f"Unknown method '{method}' for transforming to "
                "detector frame. Available methods are: "
                "'matrix_transform', 'shear_fft'"
            )

        if plot:
            plot_volume_slices(
                detector_frame_intensity,
                title="Simulated diffraction intensity in detector frame",
                norm="log",
            )

        return detector_frame_intensity

    def shift_to_target_pixel(
        self,
        intensity: np.ndarray,
    ) -> np.ndarray:
        """
        Shift and pad intensity to place peak at target pixel.

        Adjusts the intensity array to match the detector shape and
        shifts it so the Bragg peak appears at the target pixel
        position.

        Args:
            intensity: Intensity array to shift.

        Returns:
            Shifted and padded intensity with shape
            (num_frames, detector_height, detector_width).

        Example:
            >>> # ... compute detector_frame_intensity ...
            >>> shifted = sim.shift_to_target_pixel(
            ...     detector_frame_intensity
            ... )
        """
        # compute shift to centre peak at target position
        shift = tuple(
            self.target_peak_position[i] - self.detector_shape[i] // 2
            for i in range(2)
        )

        # pad to detector shape if needed
        current_shape = intensity.shape
        desired_shape = (self.num_frames,) + self.detector_shape

        if current_shape != desired_shape:
            intensity = symmetric_pad(intensity, desired_shape, values=0.0)

        # apply non-wrapping shift
        shifted_intensity = shift_no_wrap(intensity, shift=shift)

        return shifted_intensity

    def get_realistic_detector_data(
        self,
        intensity: np.ndarray,
        photon_budget: float | None = None,
        max_intensity: float | None = None,
        shift: bool = True,
        noise_params: list[dict] | dict | None = None,
    ) -> np.ndarray:
        """
        Apply realistic detector effects to simulated data.

        Adds scaling, noise, masking, and other detector effects to
        create realistic measurement data.

        Args:
            intensity: Simulated intensity array.
            photon_budget: Total photon count for scaling. Default
                is 1e8 if neither photon_budget nor max_intensity
                is provided.
            max_intensity: Maximum intensity value for scaling.
                Ignored if photon_budget is also provided.
            shift: If True, shift peak to target pixel position.
                Default is True.
            noise_params: Noise model parameters. Can be a single
                dict or list of dicts for multiple noise sources.
                Default is [{'gaussian_mean': 0.5,
                'gaussian_std': 1.0}, {'poisson_statistics': True}].

        Returns:
            Realistic detector data as integer array with dead
            pixels masked.

        Example:
            >>> detector_data = sim.to_detector_frame()
            >>> realistic_data = sim.get_realistic_detector_data(
            ...     detector_data,
            ...     photon_budget=1e10,
            ...     noise_params=[
            ...         {'gaussian_mean': 0.5, 'gaussian_std': 1.0},
            ...         {'poisson_background': 2.0},
            ...         {'poisson_statistics': True},
            ...     ],
            ... )

        Notes:
            Noise is applied sequentially if multiple noise_params
            dicts are provided. The final result is clipped to
            non-negative integers and masked.
        """
        # shift to target pixel position if requested
        if shift:
            intensity = self.shift_to_target_pixel(intensity)

        # determine scaling
        if photon_budget is None and max_intensity is None:
            photon_budget = 1e8
        elif photon_budget is not None and max_intensity is not None:
            print(
                "Both photon_budget and max_intensity provided, "
                "will only use photon_budget"
            )
            max_intensity = None

        if photon_budget is not None:
            scale = photon_budget / np.sum(intensity)
        else:  # max_intensity is not None
            scale = max_intensity / np.max(intensity)

        intensity = intensity * scale
        # add noise
        if noise_params is not None:
            if isinstance(noise_params, dict):
                noise_params = [noise_params]
        else:
            # default: air scattering + Poisson statistics, mind the order!
            noise_params = [
                {"gaussian_mean": 0.0, "gaussian_std": 0.1},
                {"poisson_statistics": True},
            ]

        for params in noise_params:
            intensity = add_noise(intensity, **params)

        # ensure no negative counts and convert to integer
        intensity = np.maximum(intensity, 0)
        intensity = intensity.astype(np.int32)

        # apply detector mask (dead pixels, gaps)
        intensity *= 1 - self.mask[np.newaxis, :, :].astype(np.int32)

        return intensity
