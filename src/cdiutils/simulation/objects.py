import numpy as np
from scipy.fft import fftn, fftshift


def make_box(
    shape: tuple[int, int, int],
    dimensions: int | tuple[int, int, int] = 15,
    centre: tuple[int, int, int] | None = None,
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
        centre: Centre position (z, y, x). If None, uses array
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

    # parse centre
    if centre is None:
        centre = tuple(np.array(shape) // 2)

    # optimised case: axis-aligned parallelepiped (no rotation)
    rotation_matrix = _parse_rotation_matrix(rotation)
    if rotation_matrix is None:
        parallelepiped = np.zeros(shape, dtype=float)
        half_dims = np.array([dim_z, dim_y, dim_x]) // 2

        # compute bounds using vectorised operations
        starts = np.maximum(0, np.array(centre) - half_dims)
        ends = np.minimum(shape, np.array(centre) + half_dims)

        parallelepiped[
            starts[0] : ends[0],
            starts[1] : ends[1],
            starts[2] : ends[2],
        ] = value
        return parallelepiped

    # rotated case: use coordinate transformation
    coords_z, coords_y, coords_x = _get_centred_coordinates(shape, centre)

    # apply inverse rotation (to transform world coords to local)
    rotation_inv = rotation_matrix.T
    rotated_z, rotated_y, rotated_x = _apply_rotation_to_coordinates(
        coords_z, coords_y, coords_x, rotation_inv
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
    radii: float | tuple[float, float, float] = 15,
    centre: tuple[float, float, float] | None = None,
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
        centre: Centre position (z, y, x). If None, uses array
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
    # parse radii
    radii_array = np.asarray(radii)
    if radii_array.size == 1:
        radius_z = radius_y = radius_x = float(radii_array)
    elif radii_array.size == 3:
        radius_z, radius_y, radius_x = [float(r) for r in radii_array]
    else:
        raise ValueError(
            "radii must be scalar or sequence of 3 elements (rz, ry, rx)"
        )

    # get centred coordinates
    coords_z, coords_y, coords_x = _get_centred_coordinates(shape, centre)

    # apply rotation if requested
    rotation_matrix = _parse_rotation_matrix(rotation)
    if rotation_matrix is not None:
        rotated_z, rotated_y, rotated_x = _apply_rotation_to_coordinates(
            coords_z, coords_y, coords_x, rotation_matrix
        )
    else:
        rotated_z, rotated_y, rotated_x = coords_z, coords_y, coords_x

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
    radius: float = 10.0,
    height: float = 25.0,
    centre: tuple[float, float, float] | None = None,
    axis: int = 0,
    rotation: np.ndarray | tuple[float, float, float] | None = None,
    value: float = 1.0,
) -> np.ndarray:
    """
    Create a 3D cylinder binary array.

    Args:
        shape: 3D array shape (nz, ny, nx).
        radius: Cylinder radius in pixels.
        height: Cylinder height in pixels.
        centre: Centre position (z, y, x). If None, uses array
            centre.
        axis: Cylinder axis direction as integer (0=z, 1=y, 2=x).
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
    if axis not in [0, 1, 2]:
        raise ValueError("axis must be 0 (z), 1 (y), or 2 (x)")

    # get centred coordinates as list for easier indexing
    coords = list(_get_centred_coordinates(shape, centre))

    # apply rotation if requested
    rotation_matrix = _parse_rotation_matrix(rotation)
    if rotation_matrix is not None:
        coords = list(
            _apply_rotation_to_coordinates(
                coords[0], coords[1], coords[2], rotation_matrix
            )
        )

    # determine radial and axial coordinates using modular arithmetic
    # for axis=0 (z): radial uses indices [1,2] (y,x)
    # for axis=1 (y): radial uses indices [0,2] (z,x)
    # for axis=2 (x): radial uses indices [0,1] (z,y)
    radial_indices = [(axis + 1) % 3, (axis + 2) % 3]
    radial_dist_sq = coords[radial_indices[0]] ** 2 + (
        coords[radial_indices[1]] ** 2
    )
    axial_coord = coords[axis]

    # cylinder equation
    inside = (radial_dist_sq <= radius**2) & (
        np.abs(axial_coord) <= height / 2
    )

    array = np.zeros(shape, dtype=float)
    array[inside] = value
    return array


def add_linear_phase(
    obj: np.ndarray,
    phase_gradient: tuple[float, float, float] = (1.0, 1.0, 1.0),
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

    # create centred coordinate grids
    centre = tuple(np.array(shape) // 2)
    grids = np.ogrid[: shape[0], : shape[1], : shape[2]]
    coords = [grids[i] - centre[i] for i in range(3)]

    # compute linear phase using vectorised dot product
    phase = sum(phase_gradient[i] * coords[i] for i in range(3))

    # apply to support if requested
    if apply_to_support:
        phase = phase * (obj > 0)

    return obj * np.exp(1j * phase)


def add_quadratic_phase(
    obj: np.ndarray,
    curvature: tuple[float, float, float] = (2, 2, 2),
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

    # create centred coordinate grids
    centre = tuple(np.array(shape) // 2)
    grids = np.ogrid[: shape[0], : shape[1], : shape[2]]
    coords = [grids[i] - centre[i] for i in range(3)]

    # compute quadratic phase
    phase = sum(curvature[i] * coords[i] ** 2 for i in range(3))

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

    # compute phase: φ = 2π * Q · u using vectorised dot product
    phase = (
        2 * np.pi * sum(q_bragg[i] * displacement_field[i] for i in range(3))
    )

    # apply only to object support
    phase = phase * (obj > 0)

    return obj * np.exp(1j * phase)


def add_random_phase(
    obj: np.ndarray,
    phase_std: float = 2.0,
    correlation_length: float | None = 10,
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


def _parse_rotation_matrix(
    rotation: np.ndarray | tuple[float, float, float] | None,
) -> np.ndarray | None:
    """
    Parse rotation input and return a rotation matrix.

    Helper function to standardise rotation matrix creation across
    shape generation functions.

    Args:
        rotation: Rotation specification. Can be:
            - None: returns None (no rotation)
            - (3,3) array: returns as-is (assumed rotation matrix)
            - tuple of 3 floats: Euler angles (deg) as
              (alpha, beta, gamma) combined as
              Rz(alpha) @ Ry(beta) @ Rx(gamma)

    Returns:
        3x3 rotation matrix or None if rotation is None.

    Raises:
        ValueError: If rotation has invalid shape.
    """
    if rotation is None:
        return None

    rotation_matrix = np.asarray(rotation)

    if rotation_matrix.shape == (3,):
        # convert Euler angles (degrees) to rotation matrix
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
        return rot_z @ rot_y @ rot_x

    elif rotation_matrix.shape == (3, 3):
        return rotation_matrix

    else:
        raise ValueError(
            "rotation must be None, (3,3) matrix or "
            "length-3 sequence of Euler angles (deg)"
        )


def _get_centred_coordinates(
    shape: tuple[int, int, int],
    centre: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create coordinate grids centred at specified position.

    Helper function to standardise coordinate grid creation across
    shape generation functions.

    Args:
        shape: 3D array shape (nz, ny, nx).
        centre: Centre position (z, y, x). If None, uses array
            centre.

    Returns:
        Tuple of (coords_z, coords_y, coords_x) as centred
        coordinate grids suitable for broadcasting.
    """
    if centre is None:
        centre = tuple(np.array(shape) // 2)

    # create coordinate grids (broadcast-friendly)
    grids = np.ogrid[: shape[0], : shape[1], : shape[2]]

    # centre coordinates and return them
    return tuple(grids[i] - centre[i] for i in range(3))


def _apply_rotation_to_coordinates(
    coords_z: np.ndarray,
    coords_y: np.ndarray,
    coords_x: np.ndarray,
    rotation_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply rotation matrix to coordinate grids.

    Helper function to apply rotation transformation to centred
    coordinates in a vectorised manner.

    Args:
        coords_z: Z-coordinates relative to centre.
        coords_y: Y-coordinates relative to centre.
        coords_x: X-coordinates relative to centre.
        rotation_matrix: 3x3 rotation matrix to apply.

    Returns:
        Tuple of (rotated_z, rotated_y, rotated_x) coordinate grids.
    """
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
    return rotated_z, rotated_y, rotated_x


def simulate_diffraction(
    obj: np.ndarray,
    photon_budget: float | None = None,
    max_intensity: float | None = None,
    scale: float = 1.0,
    poisson_statistics: bool = True,
    convention: str | None = None,
) -> np.ndarray:
    """
    Simulate diffraction pattern from a real-space object.

    This uses a single, consistent numerical convention: a forward
    n-dimensional FFT followed by fftshift to place the Bragg peak
    at the centre of the array::

        reciprocal_obj = fftshift(fftn(obj))

    The ``convention`` argument is kept for future extension (e.g.
    axis re-ordering or additional phase factors) but does not
    change the underlying FFT operator. This keeps the
    implementation simple and avoids mixing ``fftn``/``ifftn``
    conventions. If you need to match a sign convention used in
    analytical Bragg-CDI derivations or in external tools, do so
    via the definition of the q-grid (e.g. flipping an axis or
    using ``-phase`` consistently), not by swapping FFT directions
    here.

    Intensity is computed as ``|reciprocal_obj|**2``. Optionally,
    the result can be scaled to ``photon_budget`` and/or
    ``max_intensity``. Note that first scaling to
    ``max_intensity`` will discard photon budget scaling if both
    are provided. Scale is applied in any case as a multiplicative
    factor.

    For noise modelling refer to :func:`add_noise`.

    Args:
        obj: Real-space complex object to simulate.
        photon_budget: Total photon budget for the exposure. If
            provided, intensity is scaled so that the sum equals
            this value (before Poisson sampling).
        max_intensity: Maximum intensity value for final scaling.
            If None, no scaling is applied.
        scale: Multiplicative scale factor applied to intensity.
            Defaults to 1.0.
        poisson_statistics: If True, apply Poisson statistics to
            the final intensity (photon counting). Default is True.
        convention: Placeholder for future FFT/q-space
            conventions. Currently ignored except for being accepted
            for API compatibility.

    Returns:
        Simulated diffraction pattern (intensity).

    Raises:
        ValueError: If max_intensity or photon_budget are negative.

    Example:
        >>> # simulate diffraction from a 3D object
        >>> obj = make_box((64, 64, 64), dimensions=20)
        >>> obj = add_random_phase(obj, amplitude=0.1)
        >>> intensity = simulate_diffraction(obj, photon_budget=1e9)
        >>> intensity.shape
        (64, 64, 64)

    Notes:
        The FFT convention is fixed to avoid confusion. The forward
        FFT is always used. For crystallographic sign conventions,
        adjust the phase of your object (e.g., use ``np.conj(obj)``
        if needed) rather than changing the FFT direction.
    """
    # validate inputs
    if photon_budget is not None and photon_budget < 0:
        raise ValueError(
            f"photon_budget must be non-negative, got {photon_budget}"
        )

    if max_intensity is not None and max_intensity < 0:
        raise ValueError(
            f"max_intensity must be non-negative, got {max_intensity}"
        )

    # compute diffraction pattern
    reciprocal_obj = fftshift(fftn(obj))
    intensity = np.abs(reciprocal_obj) ** 2

    # apply scaling
    if photon_budget is not None:
        scale *= photon_budget / intensity.sum()

    if max_intensity is not None:
        scale *= max_intensity / intensity.max()

    intensity = intensity * scale

    # apply Poisson statistics (photon counting)
    if poisson_statistics:
        intensity = np.random.poisson(intensity)

    return intensity
