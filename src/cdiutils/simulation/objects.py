import numpy as np
from scipy.fft import fftn, fftshift, ifftn


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
    if correlation_length is not None and phase_std > 0:
        # Gaussian smoothing in Fourier space
        kz = np.fft.fftfreq(shape[0])
        ky = np.fft.fftfreq(shape[1])
        kx = np.fft.fftfreq(shape[2])
        kz, ky, kx = np.meshgrid(kz, ky, kx, indexing="ij")
        k_sq = kz**2 + ky**2 + kx**2

        # Gaussian filter
        filter_func = np.exp(-2 * (np.pi * correlation_length) ** 2 * k_sq)

        phase_fft = fftn(phase)
        phase = np.real(ifftn(phase_fft * filter_func))

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
    poisson_statistics: bool = False,
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
            the final intensity (photon counting). Default is False.
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


def fill_domain_gaps(seeds: np.ndarray, width: float = 3.0) -> np.ndarray:
    """
    Fill only the gaps between domains, leaving their outer surface
    unchanged: a voxel outside every seed is added when two different
    seeds are both within ``width`` voxels of it, and given to the
    nearest one.

    Args:
        seeds: Integer domain labels (0 = outside), e.g. the ``seeds``
            output of :func:`segment_domains`.
        width: Largest distance (voxels) to each of the two seeds.

    Returns:
        Integer labels: the seeds plus their filled gaps.
    """
    from scipy import ndimage

    count = int(seeds.max())
    distances = np.stack([
        ndimage.distance_transform_edt(seeds != label)
        for label in range(1, count + 1)
    ])
    ordered = np.sort(distances, axis=0)
    gap = (seeds == 0) & (ordered[0] <= width)
    if count > 1:
        gap &= ordered[1] <= width
    # inside the closing of the union only: fills the planes between
    # domains, not the rim just outside their surface
    structure = ndimage.generate_binary_structure(seeds.ndim, seeds.ndim)
    gap &= ndimage.binary_closing(
        seeds > 0, structure=structure, iterations=int(np.ceil(width)))
    filled = seeds.copy()
    filled[gap] = np.argmin(distances, axis=0)[gap] + 1
    return filled.astype(np.int32)


def segment_domains(
    modulus: np.ndarray,
    seed_threshold: float = 0.25,
    outline_threshold: float = 0.2,
    closing_iterations: int = 3,
    min_size: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a reconstructed object into domains separated by
    low-modulus planes, and share its filled outline between them.

    A phase step between two parts of a crystal makes a band-limited
    reconstruction drop in modulus at their boundary (the modulus goes
    as |cos(step / 2)|), so the domains appear as blobs separated by
    "empty planes". The blobs above ``seed_threshold`` are the domain
    seeds. The outline is the modulus above ``outline_threshold``,
    closed and with its holes filled, so that it bridges the low-modulus
    planes. Every outline voxel is given to the nearest seed.

    Args:
        modulus: Modulus of the reconstructed object (any scale).
        seed_threshold: Relative modulus (fraction of the maximum)
            above which connected voxels form a domain seed.
        outline_threshold: Relative modulus of the particle outline.
        closing_iterations: Binary closing iterations of the outline
            (bridges the low-modulus planes between domains).
        min_size: Seeds with fewer voxels are ignored.

    Returns:
        (domains, seeds): integer arrays of the modulus shape. 0 is
        outside, 1..n the domains, numbered by decreasing seed size.
        ``domains`` covers the filled outline, ``seeds`` only the
        seed blobs (the "gaps as voids" geometry).
    """
    from scipy import ndimage

    relative = modulus / modulus.max()
    labels, count = ndimage.label(relative > seed_threshold)
    sizes = np.bincount(labels.ravel())[1:]
    order = [k + 1 for k in np.argsort(sizes)[::-1] if sizes[k] >= min_size]
    seeds = np.zeros(modulus.shape, dtype=np.int32)
    for number, label in enumerate(order, start=1):
        seeds[labels == label] = number
    outline = ndimage.binary_closing(
        relative > outline_threshold, iterations=closing_iterations
    )
    outline = ndimage.binary_fill_holes(outline | (seeds > 0))
    # index of the nearest seed voxel for every voxel
    _, nearest = ndimage.distance_transform_edt(
        seeds == 0, return_indices=True
    )
    domains = seeds[tuple(nearest)] * outline
    return domains.astype(np.int32), seeds


def _centred_coordinates(shape: tuple) -> list:
    """
    Voxel coordinates relative to the array centre (n // 2), the
    convention of :func:`add_linear_phase`.

    Args:
        shape: Array shape.

    Returns:
        List of broadcastable coordinate arrays, one per axis.
    """
    grids = np.ogrid[tuple(slice(0, n) for n in shape)]
    return [grids[i] - shape[i] // 2 for i in range(len(shape))]


def measure_domain_phases(
    obj: np.ndarray, domains: np.ndarray
) -> list[dict]:
    """
    Linear phase ramp and constant phase of every domain of a complex
    object.

    The ramp of a domain is the mean phase difference between
    neighbouring voxels along each axis (modulus-weighted, insensitive
    to phase wrapping). Its offset is the phase left after removing the
    ramp, with coordinates centred on the array (:func:`add_linear_phase`
    convention), so offsets of domains that share a ramp are directly
    their phase steps. The coherence |sum(z)| / sum(|z|) is 1 when the
    domain is exactly a ramp plus a constant.

    Args:
        obj: Complex object.
        domains: Integer domain labels (0 = outside).

    Returns:
        One dict per domain: label, size, ramp (rad / voxel per axis),
        offset (rad), coherence, mean_modulus (relative to the maximum).
    """
    coordinates = _centred_coordinates(obj.shape)
    modulus = np.abs(obj) / np.abs(obj).max()
    results = []
    for label in range(1, int(domains.max()) + 1):
        inside = domains == label
        ramp = []
        for axis in range(obj.ndim):
            product = obj * np.conj(np.roll(obj, 1, axis=axis))
            pairs = inside & np.roll(inside, 1, axis=axis)
            ramp.append(float(np.angle(product[pairs].sum())))
        phase = sum(ramp[i] * coordinates[i] for i in range(obj.ndim))
        flattened = (obj * np.exp(-1j * phase))[inside]
        results.append({
            "label": label,
            "size": int(inside.sum()),
            "ramp": ramp,
            "offset": float(np.angle(flattened.sum())),
            "coherence": float(
                np.abs(flattened.sum()) / np.abs(flattened).sum()
            ),
            "mean_modulus": float(modulus[inside].mean()),
        })
    return results


# order of the quadratic phase coefficients (rad / voxel^2):
# phase = sum_i g_i x_i + sum_(i<=j) h_ij x_i x_j
QUADRATIC_TERMS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _phase_field(coordinates, ramp, quadratic=None):
    """
    Linear (+ quadratic) phase on centred coordinates.

    Args:
        coordinates: Output of :func:`_centred_coordinates`.
        ramp: Three linear coefficients (rad / voxel).
        quadratic: Six quadratic coefficients in the order of
            ``QUADRATIC_TERMS`` (rad / voxel^2), or None.

    Returns:
        Phase array (broadcast to the full shape).
    """
    phase = sum(ramp[i] * coordinates[i] for i in range(3))
    if quadratic is not None:
        for value, (i, j) in zip(quadratic, QUADRATIC_TERMS):
            phase = phase + value * coordinates[i] * coordinates[j]
    return phase


def fit_quadratic_phase(obj: np.ndarray, region: np.ndarray) -> tuple:
    """
    Linear + quadratic phase that best describes a complex object inside
    a region (a bent or graded crystal), by least squares on the local
    phase differences between neighbouring voxels (insensitive to phase
    wrapping): d(phase)/dx_a = g_a + sum_b dh/dx_a terms.

    Args:
        obj: Complex object.
        region: Boolean region of the fit.

    Returns:
        (ramp, quadratic): three linear and six quadratic coefficients
        (``QUADRATIC_TERMS`` order), with coordinates centred on the
        array (:func:`_centred_coordinates`).
    """
    coordinates = _centred_coordinates(obj.shape)
    rows, values = [], []
    for axis in range(3):
        product = obj * np.conj(np.roll(obj, 1, axis=axis))
        pairs = region & np.roll(region, 1, axis=axis)
        weights = np.abs(product)[pairs]
        # midpoint between the voxel and its neighbour along this axis
        midpoint = [np.broadcast_to(c, obj.shape)[pairs].astype(float)
                    for c in coordinates]
        midpoint[axis] = midpoint[axis] - 0.5
        design = np.zeros((weights.size, 9))
        design[:, axis] = 1.0
        for column, (i, j) in enumerate(QUADRATIC_TERMS):
            # derivative of h_ij x_i x_j along this axis
            if i == j == axis:
                design[:, 3 + column] = 2 * midpoint[axis]
            elif i == axis:
                design[:, 3 + column] = midpoint[j]
            elif j == axis:
                design[:, 3 + column] = midpoint[i]
        rows.append(design * np.sqrt(weights)[:, None])
        values.append(np.angle(product[pairs]) * np.sqrt(weights))
    solution, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(values),
                                   rcond=None)
    return solution[:3], solution[3:]


class DomainModelFit:
    """
    Fit the measured Bragg intensity with a crystal made of domains of
    uniform density, each with its own complex amplitude and linear
    phase ramp, in the detector frame of the phasing (PyNX convention:
    intensity = |fftshift(fftn(ifftshift(object)))|^2).

    Model intensity: |sum_k c_k FFT(rho_k exp(i g_k . r))|^2 + b, with
    rho_k the domain shapes, c_k = a_k exp(i phi_k) and b a constant
    background. Models:

    * "C": one common ramp, no phase steps (phi_k = 0), free a_k.
    * "A": one common ramp, free phase steps phi_k (phi_1 = 0).
    * "B": no phase steps, free ramp per domain (tilted domains).
    * "full": free phase steps and free ramps.
    * "bend": no phase steps, free ramp and free quadratic phase per
      domain (smooth bending or d-spacing gradient; with one domain, a
      continuous crystal).

    ``equal_amplitudes=True`` forces one a for all domains (density
    equal in every domain). The fit minimises the Poisson deviance on
    the unmasked voxels of ``region``.

    Args:
        data: Measured intensity (photons), detector frame.
        mask: Non-zero for masked voxels.
        domains: Integer domain labels, same shape as data.
        region: Tuple of slices where the deviance is computed
            (default: the whole array).
    """

    def __init__(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        domains: np.ndarray,
        region: tuple = None,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.domains = domains
        self.count = int(domains.max())
        self.region = region or tuple(slice(None) for _ in data.shape)
        self.valid = (np.asarray(mask) == 0)[self.region]
        self.observed = self.data[self.region][self.valid]
        self.coordinates = _centred_coordinates(data.shape)

    def domain_fields(
        self, ramps: np.ndarray, quadratics: np.ndarray = None
    ) -> np.ndarray:
        """
        Diffracted field of every domain (unit density), on the valid
        voxels of the region.

        Args:
            ramps: (n_domains, 3) phase ramps in rad / voxel.
            quadratics: (n_domains, 6) quadratic phase coefficients
                (``QUADRATIC_TERMS`` order, rad / voxel^2), or None.

        Returns:
            (n_domains, n_valid) complex array.
        """
        fields = []
        for label in range(1, self.count + 1):
            phase = _phase_field(
                self.coordinates, ramps[label - 1],
                None if quadratics is None else quadratics[label - 1])
            density = (self.domains == label) * np.exp(1j * phase)
            field = fftshift(fftn(np.fft.ifftshift(density), workers=-1))
            fields.append(field[self.region][self.valid])
        return np.array(fields)

    def deviance(self, model: np.ndarray) -> float:
        """
        Poisson deviance between the model and the observed intensity.

        Args:
            model: Model intensity on the valid voxels of the region.

        Returns:
            2 * sum(m - d + d ln(d / m)).
        """
        model = np.maximum(model, 1e-12)
        observed = self.observed
        positive = observed > 0
        log_term = np.zeros_like(observed)
        log_term[positive] = observed[positive] * np.log(
            observed[positive] / model[positive]
        )
        return float(2 * np.sum(model - observed + log_term))

    def intensity(
        self,
        fields: np.ndarray,
        amplitudes: np.ndarray,
        phases: np.ndarray,
        background: float,
    ) -> np.ndarray:
        """
        Model intensity on the valid voxels of the region.

        Args:
            fields: Output of :meth:`domain_fields`.
            amplitudes: a_k, one per domain.
            phases: phi_k, one per domain (rad).
            background: Constant background b (photons / voxel).

        Returns:
            Model intensity, same length as the observed vector.
        """
        coefficients = amplitudes * np.exp(1j * phases)
        return np.abs(coefficients @ fields) ** 2 + background

    def fit(
        self,
        model: str = "A",
        ramps: np.ndarray = None,
        equal_amplitudes: bool = False,
        n_starts: int = 8,
        seed: int = 0,
        maxiter: int = 400,
        quadratics: np.ndarray = None,
    ) -> dict:
        """
        Fit one model.

        Args:
            model: "A", "B", "C", "full" or "bend" (see the class
                docstring).
            ramps: (n_domains, 3) starting ramps in rad / voxel. Models
                A and C use their mean (weighted by domain size) for
                every domain; B and full start from them. Default 0.
            equal_amplitudes: One amplitude for all domains.
            n_starts: Random starts of the phase steps (A, full); B and
                C use one start.
            seed: Seed of the random starts.
            maxiter: Iterations of each L-BFGS-B / Powell run.
            quadratics: (n_domains, 6) starting quadratic coefficients
                for "bend" (e.g. from :func:`fit_quadratic_phase`).
                Default 0.

        Returns:
            dict: model, deviance, n_parameters, amplitudes, phases
            (rad, phases[0] = 0), ramps, background, start deviances.
        """
        from scipy.optimize import minimize

        count = self.count
        sizes = np.bincount(self.domains.ravel())[1:count + 1]
        ramps = np.zeros((count, 3)) if ramps is None else np.array(ramps)
        if model in ("A", "C"):
            common = (sizes[:, None] * ramps).sum(0) / sizes.sum()
            ramps = np.repeat(common[None], count, axis=0)
        free_phases = model in ("A", "full")
        free_ramps = model in ("B", "full", "bend")
        free_quadratics = model == "bend"
        quadratics = (np.zeros((count, 6)) if quadratics is None
                      else np.array(quadratics, dtype=float))
        # quadratic coefficients are optimised in units of 1e-3 rad/voxel^2
        # so that all parameters have comparable steps
        quadratic_unit = 1e-3
        n_amplitudes = 1 if equal_amplitudes else count
        fields = self.domain_fields(ramps)
        # starting amplitude: total measured / total unit-density model
        scale = np.sqrt(
            max(self.observed.sum(), 1.0)
            / max((np.abs(fields.sum(0)) ** 2).sum(), 1e-12)
        )

        def unpack(parameters):
            """Split the parameter vector into the model quantities."""
            position = 0
            # clipped: exp() stays finite on wild steps (Powell has no
            # bounds here, they stall it)
            log_amplitudes = np.clip(
                parameters[position:position + n_amplitudes],
                np.log(scale) - 10, np.log(scale) + 10)
            position += n_amplitudes
            amplitudes = np.exp(log_amplitudes) * np.ones(count)
            phases = np.zeros(count)
            if free_phases:
                phases[1:] = parameters[position:position + count - 1]
                position += count - 1
            background = np.logaddexp(0, np.clip(parameters[position],
                                                 -30, 30))
            position += 1
            fitted_ramps = ramps
            if free_ramps:
                fitted_ramps = parameters[position:position + 3 * count]
                fitted_ramps = fitted_ramps.reshape(count, 3)
                position += 3 * count
            fitted_quadratics = None
            if free_quadratics:
                fitted_quadratics = quadratic_unit * parameters[
                    position:position + 6 * count].reshape(count, 6)
            return (amplitudes, phases, background, fitted_ramps,
                    fitted_quadratics)

        cache = {}

        def objective(parameters):
            """Poisson deviance of one parameter vector."""
            (amplitudes, phases, background, fitted_ramps,
             fitted_quadratics) = unpack(parameters)
            current = fields
            if free_ramps:
                key = fitted_ramps.tobytes() + (
                    b"" if fitted_quadratics is None
                    else fitted_quadratics.tobytes())
                if key not in cache:
                    cache.clear()
                    cache[key] = self.domain_fields(fitted_ramps,
                                                    fitted_quadratics)
                current = cache[key]
            return self.deviance(
                self.intensity(current, amplitudes, phases, background)
            )

        rng = np.random.default_rng(seed)
        starts = n_starts if free_phases else 1
        best, start_deviances = None, []
        for _ in range(starts):
            initial = [np.log(scale)] * n_amplitudes
            if free_phases:
                initial += list(rng.uniform(-np.pi, np.pi, count - 1))
            initial += [np.log(np.expm1(max(np.median(self.observed),
                                            1e-3)))]
            if free_ramps:
                initial += list(ramps.ravel())
            if free_quadratics:
                initial += list((quadratics / quadratic_unit).ravel())
            # bounds keep exp(log-amplitude) and the softplus background
            # finite on wild steps (overflow seen on real data)
            bounds = [(np.log(scale) - 10, np.log(scale) + 10)]
            bounds *= n_amplitudes
            if free_phases:
                bounds += [(None, None)] * (count - 1)
            bounds += [(-30, 30)]
            if free_ramps:
                bounds += [(-np.pi, np.pi)] * (3 * count)
            if free_quadratics:
                bounds += [(None, None)] * (6 * count)
            method = "Powell" if free_ramps else "L-BFGS-B"
            result = minimize(objective, np.array(initial, dtype=float),
                              method=method,
                              bounds=None if free_ramps else bounds,
                              options={"maxiter": maxiter})
            start_deviances.append(float(result.fun))
            if best is None or result.fun < best.fun:
                best = result
        (amplitudes, phases, background, fitted_ramps,
         fitted_quadratics) = unpack(best.x)
        return {
            "model": model,
            "equal_amplitudes": equal_amplitudes,
            "deviance": float(best.fun),
            "n_parameters": int(best.x.size),
            "amplitudes": amplitudes.tolist(),
            "phases": list(np.angle(np.exp(1j * phases))),
            "ramps": np.asarray(fitted_ramps).tolist(),
            "quadratics": (None if fitted_quadratics is None
                           else fitted_quadratics.tolist()),
            "background": float(background),
            "start_deviances": start_deviances,
        }

    def model_intensity(self, result: dict) -> np.ndarray:
        """
        Full-array model intensity of a fit result (for plots).

        Args:
            result: Output of :meth:`fit`.

        Returns:
            Model intensity with the shape of the data.
        """
        field = np.zeros(self.data.shape, dtype=complex)
        for label in range(1, self.count + 1):
            quadratics = result.get("quadratics")
            phase = _phase_field(
                self.coordinates, result["ramps"][label - 1],
                None if quadratics is None else quadratics[label - 1])
            density = (self.domains == label) * np.exp(1j * phase)
            coefficient = result["amplitudes"][label - 1] * np.exp(
                1j * result["phases"][label - 1]
            )
            field += coefficient * fftshift(
                fftn(np.fft.ifftshift(density), workers=-1)
            )
        return np.abs(field) ** 2 + result["background"]
