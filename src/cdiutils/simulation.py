"""
Simulation tools for BCDI experiments.

This module provides functions to create synthetic 3D objects, add
various phase patterns, and simulate diffraction patterns. Useful for
testing reconstruction algorithms and understanding experimental
artefacts.

Functions
---------
Geometry creation:
    make_box : Create a 3D box (parallelepiped/cube).
    make_ellipsoid : Create a 3D ellipsoid or sphere.
    make_cylinder : Create a 3D cylinder.

Phase manipulation:
    add_linear_phase : Add linear phase gradient.
    add_quadratic_phase : Add quadratic phase (defocus, strain).
    add_displacement_field : Add phase from displacement field.
    add_random_phase : Add random phase noise.

Diffraction simulation:
    simulate_diffraction : Compute diffraction pattern via FFT.
    add_noise_and_scale : Add Poisson noise and scale intensity.
"""

import numpy as np


def make_box(
    shape: tuple[int, int, int],
    dimensions: int | tuple[int, int, int],
    center: tuple[int, int, int] | None = None,
    rotation: np.ndarray | tuple[float, float, float] | None = None,
    value: float = 1.0,
) -> np.ndarray:
    """
    Create a 3D parallelepiped (rectangular cuboid) binary array.

    A cube is a special case where all dimensions are equal.

    Args:
        shape: 3D array shape (nz, ny, nx).
        dimensions: Side lengths in pixels. If scalar, creates a
            cube. If tuple, order is (length_z, length_y, length_x).
        center: Centre position (z, y, x). If None, uses array
            centre.
        rotation: Rotation to apply. Can be:
            - None: no rotation
            - (3,3) array: rotation matrix
            - tuple of 3 floats: Euler angles (deg) as
              (alpha, beta, gamma) combined as
              Rz(alpha) @ Ry(beta) @ Rx(gamma)
        value: Value to fill inside the parallelepiped.

    Returns:
        Binary array with `value` inside parallelepiped and 0
        outside.

    Raises:
        ValueError: If shape is not 3D, dimensions invalid, or
            rotation has invalid shape.
    """
    if len(shape) != 3:
        raise ValueError(f"shape must be 3D, got {len(shape)}D")

    # parse dimensions
    dims_array = np.asarray(dimensions)
    if dims_array.size == 1:
        dim_z = dim_y = dim_x = int(dims_array)
    elif dims_array.size == 3:
        dim_z, dim_y, dim_x = [int(d) for d in dims_array]
    else:
        raise ValueError("dimensions must be scalar or sequence of 3 elements")

    if any(d <= 0 for d in [dim_z, dim_y, dim_x]):
        raise ValueError("all dimensions must be positive")

    if center is None:
        center = tuple(np.array(shape) // 2)

    center_z, center_y, center_x = center

    if rotation is None:
        # simple case: axis-aligned parallelepiped
        parallelepiped = np.zeros(shape, dtype=float)
        half_z, half_y, half_x = dim_z // 2, dim_y // 2, dim_x // 2
        parallelepiped[
            center_z - half_z : center_z + half_z,
            center_y - half_y : center_y + half_y,
            center_x - half_x : center_x + half_x,
        ] = value
        return parallelepiped

    # rotated case: use coordinate transformation
    coord_z, coord_y, coord_x = np.ogrid[: shape[0], : shape[1], : shape[2]]

    # coordinates relative to centre
    coords_z = coord_z - center_z
    coords_y = coord_y - center_y
    coords_x = coord_x - center_x

    # parse rotation matrix
    rotation_matrix = np.asarray(rotation)
    if rotation_matrix.shape == (3,):
        alpha, beta, gamma = np.deg2rad(rotation_matrix)
        rot_z = np.array(
            [
                [np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 1],
            ]
        )
        rot_y = np.array(
            [
                [np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)],
            ]
        )
        rot_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(gamma), -np.sin(gamma)],
                [0, np.sin(gamma), np.cos(gamma)],
            ]
        )
        rotation_matrix = rot_z @ rot_y @ rot_x
    elif rotation_matrix.shape != (3, 3):
        raise ValueError(
            "rotation must be None, (3,3) matrix or "
            "length-3 sequence of Euler angles (deg)"
        )

    # apply inverse rotation (to transform world coords to local)
    rotation_inv = rotation_matrix.T
    rotated_z = (
        rotation_inv[0, 0] * coords_z
        + rotation_inv[0, 1] * coords_y
        + rotation_inv[0, 2] * coords_x
    )
    rotated_y = (
        rotation_inv[1, 0] * coords_z
        + rotation_inv[1, 1] * coords_y
        + rotation_inv[1, 2] * coords_x
    )
    rotated_x = (
        rotation_inv[2, 0] * coords_z
        + rotation_inv[2, 1] * coords_y
        + rotation_inv[2, 2] * coords_x
    )

    # check if inside box
    half_z, half_y, half_x = dim_z / 2, dim_y / 2, dim_x / 2
    inside = (
        (np.abs(rotated_z) <= half_z)
        & (np.abs(rotated_y) <= half_y)
        & (np.abs(rotated_x) <= half_x)
    )

    parallelepiped = np.zeros(shape, dtype=float)
    parallelepiped[inside] = value
    return parallelepiped


def make_ellipsoid(
    shape: tuple[int, int, int],
    radii: float | tuple[float, float, float],
    center: tuple[float, float, float] | None = None,
    rotation: np.ndarray | tuple[float, float, float] | None = None,
    value: float = 1.0,
) -> np.ndarray:
    """
    Create a 3D ellipsoid binary array.

    A sphere is a special case where all radii are equal.

    Args:
        shape: 3D array shape (nz, ny, nx).
        radii: Radii in pixels. If scalar, creates a sphere.
            If tuple, order is (rz, ry, rx).
        center: Centre position (z, y, x). If None, uses array
            centre.
        rotation: Rotation to apply. Can be:
            - None: no rotation
            - (3,3) array: rotation matrix
            - tuple of 3 floats: Euler angles (deg) as
              (alpha, beta, gamma) combined as
              Rz(alpha) @ Ry(beta) @ Rx(gamma)
        value: Value to fill inside the ellipsoid.

    Returns:
        3D array with `value` inside ellipsoid and 0 outside.

    Raises:
        ValueError: If radii or rotation have invalid shape.
    """
    radii_array = np.asarray(radii)
    if radii_array.size == 1:
        radius_z = radius_y = radius_x = float(radii_array)
    elif radii_array.size == 3:
        radius_z, radius_y, radius_x = [float(r) for r in radii_array]
    else:
        raise ValueError(
            "radii must be scalar or sequence of 3 elements (rz, ry, rx)"
        )

    if center is None:
        center_array = np.array(shape, dtype=float) // 2
    else:
        center_array = np.asarray(center, dtype=float)

    # prepare coordinate grids (broadcast-friendly)
    coord_z, coord_y, coord_x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    center_z, center_y, center_x = center_array

    # coordinates relative to centre
    coords_z = coord_z - center_z
    coords_y = coord_y - center_y
    coords_x = coord_x - center_x

    # apply rotation if requested
    if rotation is not None:
        rotation_matrix = np.asarray(rotation)
        if rotation_matrix.shape == (3,):
            # treat as Euler angles in degrees
            alpha, beta, gamma = np.deg2rad(rotation_matrix)
            rot_z = np.array(
                [
                    [np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha), np.cos(alpha), 0],
                    [0, 0, 1],
                ]
            )
            rot_y = np.array(
                [
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)],
                ]
            )
            rot_x = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(gamma), -np.sin(gamma)],
                    [0, np.sin(gamma), np.cos(gamma)],
                ]
            )
            rotation_matrix = rot_z @ rot_y @ rot_x
        elif rotation_matrix.shape != (3, 3):
            raise ValueError(
                "rotation must be None, (3,3) matrix or "
                "length-3 sequence of Euler angles (deg)"
            )

        # compute rotated coordinates (vectorised broadcasting)
        rotated_z = (
            rotation_matrix[0, 0] * coords_z
            + rotation_matrix[0, 1] * coords_y
            + rotation_matrix[0, 2] * coords_x
        )
        rotated_y = (
            rotation_matrix[1, 0] * coords_z
            + rotation_matrix[1, 1] * coords_y
            + rotation_matrix[1, 2] * coords_x
        )
        rotated_x = (
            rotation_matrix[2, 0] * coords_z
            + rotation_matrix[2, 1] * coords_y
            + rotation_matrix[2, 2] * coords_x
        )
    else:
        rotated_z = coords_z
        rotated_y = coords_y
        rotated_x = coords_x

    # ellipsoid equation: (z/rz)^2 + (y/ry)^2 + (x/rx)^2 <= 1
    with np.errstate(divide="ignore", invalid="ignore"):
        inside = (rotated_z / radius_z) ** 2 + (rotated_y / radius_y) ** 2 + (
            rotated_x / radius_x
        ) ** 2 <= 1.0

    array = np.zeros(shape, dtype=float)
    array[inside] = value
    return array


def make_cylinder(
    shape: tuple[int, int, int],
    radius: float,
    height: float,
    center: tuple[float, float, float] | None = None,
    axis: str = "z",
    rotation: np.ndarray | tuple[float, float, float] | None = None,
    value: float = 1.0,
) -> np.ndarray:
    """
    Create a 3D cylinder binary array.

    Args:
        shape: 3D array shape (nz, ny, nx).
        radius: Cylinder radius in pixels.
        height: Cylinder height in pixels.
        center: Centre position (z, y, x). If None, uses array
            centre.
        axis: Cylinder axis direction ('x', 'y', or 'z').
        rotation: Additional rotation to apply after axis
            alignment. Same format as other shape functions.
        value: Value to fill inside the cylinder.

    Returns:
        3D array with `value` inside cylinder and 0 outside.

    Raises:
        ValueError: If axis is invalid or parameters negative.
    """
    if radius <= 0 or height <= 0:
        raise ValueError("radius and height must be positive")
    if axis not in ["x", "y", "z"]:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    if center is None:
        center_array = np.array(shape, dtype=float) // 2
    else:
        center_array = np.asarray(center, dtype=float)

    coord_z, coord_y, coord_x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    center_z, center_y, center_x = center_array

    coords_z = coord_z - center_z
    coords_y = coord_y - center_y
    coords_x = coord_x - center_x

    # apply rotation if requested
    if rotation is not None:
        rotation_matrix = np.asarray(rotation)
        if rotation_matrix.shape == (3,):
            alpha, beta, gamma = np.deg2rad(rotation_matrix)
            rot_z = np.array(
                [
                    [np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha), np.cos(alpha), 0],
                    [0, 0, 1],
                ]
            )
            rot_y = np.array(
                [
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)],
                ]
            )
            rot_x = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(gamma), -np.sin(gamma)],
                    [0, np.sin(gamma), np.cos(gamma)],
                ]
            )
            rotation_matrix = rot_z @ rot_y @ rot_x
        elif rotation_matrix.shape != (3, 3):
            raise ValueError(
                "rotation must be None, (3,3) matrix or "
                "length-3 sequence of Euler angles (deg)"
            )

        coords_z = (
            rotation_matrix[0, 0] * coords_z
            + rotation_matrix[0, 1] * coords_y
            + rotation_matrix[0, 2] * coords_x
        )
        coords_y = (
            rotation_matrix[1, 0] * coords_z
            + rotation_matrix[1, 1] * coords_y
            + rotation_matrix[1, 2] * coords_x
        )
        coords_x = (
            rotation_matrix[2, 0] * coords_z
            + rotation_matrix[2, 1] * coords_y
            + rotation_matrix[2, 2] * coords_x
        )

    # determine radial and axial coordinates based on axis
    if axis == "z":
        radial_dist_sq = coords_y**2 + coords_x**2
        axial_coord = coords_z
    elif axis == "y":
        radial_dist_sq = coords_z**2 + coords_x**2
        axial_coord = coords_y
    else:  # axis == 'x'
        radial_dist_sq = coords_z**2 + coords_y**2
        axial_coord = coords_x

    # cylinder equation
    inside = (radial_dist_sq <= radius**2) & (
        np.abs(axial_coord) <= height / 2
    )

    array = np.zeros(shape, dtype=float)
    array[inside] = value
    return array


def add_linear_phase(
    obj: np.ndarray,
    phase_gradient: tuple[float, float, float],
    apply_to_support: bool = True,
) -> np.ndarray:
    """
    Add a linear phase gradient to an object.

    Args:
        obj: Real-valued object (amplitude).
        phase_gradient: Phase gradient (radians/pixel) along
            (z, y, x) directions.
        apply_to_support: If True, apply phase only where obj > 0.

    Returns:
        Complex object with linear phase applied.
    """
    shape = obj.shape
    grad_z, grad_y, grad_x = phase_gradient

    # create coordinate grids
    coord_z, coord_y, coord_x = np.ogrid[: shape[0], : shape[1], : shape[2]]

    # centre coordinates
    center_z, center_y, center_x = np.array(shape) // 2
    coord_z = coord_z - center_z
    coord_y = coord_y - center_y
    coord_x = coord_x - center_x

    # compute linear phase
    phase = grad_z * coord_z + grad_y * coord_y + grad_x * coord_x

    # apply to support if requested
    if apply_to_support:
        phase = phase * (obj > 0)

    return obj * np.exp(1j * phase)


def add_quadratic_phase(
    obj: np.ndarray,
    curvature: tuple[float, float, float],
    apply_to_support: bool = True,
) -> np.ndarray:
    """
    Add quadratic phase (e.g., defocus, strain) to an object.

    Args:
        obj: Real-valued object (amplitude).
        curvature: Phase curvature coefficients (radians/pixel²)
            along (z, y, x) directions.
        apply_to_support: If True, apply phase only where obj > 0.

    Returns:
        Complex object with quadratic phase applied.
    """
    shape = obj.shape
    curv_z, curv_y, curv_x = curvature

    coord_z, coord_y, coord_x = np.ogrid[: shape[0], : shape[1], : shape[2]]

    center_z, center_y, center_x = np.array(shape) // 2
    coord_z = coord_z - center_z
    coord_y = coord_y - center_y
    coord_x = coord_x - center_x

    # quadratic phase
    phase = curv_z * coord_z**2 + curv_y * coord_y**2 + curv_x * coord_x**2

    if apply_to_support:
        phase = phase * (obj > 0)

    return obj * np.exp(1j * phase)


def add_displacement_field(
    obj: np.ndarray,
    displacement_field: np.ndarray,
    q_bragg: tuple[float, float, float],
) -> np.ndarray:
    """
    Add phase from a 3D displacement field.

    The phase is computed as φ = 2π * Q · u(r), where Q is the
    Bragg vector and u(r) is the displacement field.

    Args:
        obj: Real-valued object (amplitude).
        displacement_field: 3D displacement vector field with
            shape (3, nz, ny, nx) where first axis is (uz, uy, ux).
        q_bragg: Bragg vector (qz, qy, qx) in reciprocal
            space units.

    Returns:
        Complex object with displacement-induced phase.

    Raises:
        ValueError: If displacement_field shape is incompatible.
    """
    if displacement_field.shape[0] != 3:
        raise ValueError("displacement_field must have shape (3, nz, ny, nx)")
    if displacement_field.shape[1:] != obj.shape:
        raise ValueError(
            "displacement_field spatial shape must match obj shape"
        )

    qz, qy, qx = q_bragg

    # compute phase: φ = 2π * Q · u
    phase = (
        2
        * np.pi
        * (
            qz * displacement_field[0]
            + qy * displacement_field[1]
            + qx * displacement_field[2]
        )
    )

    # apply only to object support
    phase = phase * (obj > 0)

    return obj * np.exp(1j * phase)


def add_random_phase(
    obj: np.ndarray,
    phase_std: float,
    correlation_length: float | None = None,
    apply_to_support: bool = True,
) -> np.ndarray:
    """
    Add random phase noise to an object.

    Args:
        obj: Real-valued object (amplitude).
        phase_std: Standard deviation of phase noise (radians).
        correlation_length: Correlation length for spatially
            correlated noise (pixels). If None, uses uncorrelated
            noise.
        apply_to_support: If True, apply phase only where obj > 0.

    Returns:
        Complex object with random phase noise.
    """
    shape = obj.shape

    # generate random phase
    phase = np.random.normal(0, phase_std, shape)

    # apply spatial correlation if requested
    if correlation_length is not None:
        # Gaussian smoothing in Fourier space
        kz = np.fft.fftfreq(shape[0])
        ky = np.fft.fftfreq(shape[1])
        kx = np.fft.fftfreq(shape[2])
        kz, ky, kx = np.meshgrid(kz, ky, kx, indexing="ij")
        k_sq = kz**2 + ky**2 + kx**2

        # Gaussian filter
        filter_func = np.exp(-2 * (np.pi * correlation_length) ** 2 * k_sq)

        phase_fft = np.fft.fftn(phase)
        phase = np.real(np.fft.ifftn(phase_fft * filter_func))

        # renormalise to maintain std
        phase = phase / phase.std() * phase_std

    if apply_to_support:
        phase = phase * (obj > 0)

    return obj * np.exp(1j * phase)


def simulate_diffraction(
    obj: np.ndarray,
    max_intensity: float | None = None,
    noise_level: float | None = None,
) -> np.ndarray:
    """
    Simulate diffraction pattern from real-space object.

    The simulation performs FFT to reciprocal space and computes
    intensity. Optionally adds Poisson noise then scales to
    max_intensity (noise first, then scaling).

    Note:
        For realistic rocking curve simulations, consider leaving
        both parameters as None and applying noise/scaling after
        interpolation to detector frame.

    Args:
        obj: Real-space complex object to simulate.
        max_intensity: Maximum intensity value for final scaling.
            If None, no scaling is applied.
        noise_level: Photon count scaling factor for Poisson
            noise. Higher values give better SNR. If None, no
            noise is added. Applied before max_intensity scaling.

    Returns:
        Simulated diffraction pattern (intensity).

    Raises:
        ValueError: If max_intensity or noise_level are negative.
    """
    # FFT to reciprocal space (shift before and after)
    reciprocal = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))

    # compute intensity
    intensity = np.abs(reciprocal) ** 2

    # apply noise and scaling using the dedicated function
    if noise_level is not None or max_intensity is not None:
        intensity = add_noise_and_scale(
            intensity, noise_level=noise_level, max_intensity=max_intensity
        )

    return intensity


def add_noise_and_scale(
    intensity: np.ndarray,
    noise_level: float | None = None,
    max_intensity: float | None = None,
) -> np.ndarray:
    """
    Add Poisson noise and scale intensity data.

    This function applies noise before scaling, which ensures the
    noise characteristics are set by noise_level independently of
    the final intensity scale. Useful for processing interpolated
    detector data.

    Args:
        intensity: Input intensity array.
        noise_level: Photon count scaling factor for Poisson
            noise. Higher values give better SNR. If None, no
            noise is added.
        max_intensity: Maximum intensity value for final scaling.
            If None, no scaling is applied.

    Returns:
        Processed intensity array.

    Raises:
        ValueError: If max_intensity or noise_level are negative.
    """
    if noise_level is not None and noise_level < 0:
        raise ValueError(
            f"noise_level must be non-negative, got {noise_level}"
        )
    if max_intensity is not None and max_intensity <= 0:
        raise ValueError(
            f"max_intensity must be positive, got {max_intensity}"
        )

    result = intensity.copy()

    # add noise first
    if noise_level is not None:
        # scale to photon counts
        photons = result * noise_level
        # sample from Poisson distribution
        photons = np.random.poisson(photons)
        # scale back to intensity units
        result = photons.astype(float) / noise_level

    # then scale to max intensity
    if max_intensity is not None:
        result = result / result.max() * max_intensity

    return result
