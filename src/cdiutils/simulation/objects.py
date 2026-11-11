import itertools

import numpy as np
from scipy.fft import fftn, fftshift, ifftn
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.distance import pdist


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


def hexagonal_family_normals(
    hkl: tuple[int, ...],
    c_over_a: float,
) -> np.ndarray:
    """
    Unit normals of all planes of an hexagonal {hkil} family.

    The family is expanded with the 24 operations of the 6/mmm Laue
    group: the 6 permutations of (h, k, i), the inversion of
    (h, k, i) and l -> -l. The Cartesian frame has x along a1 and z
    along c. Normals are returned in the (z, y, x) order used by the
    shape functions of this module, so that (0001) points along +z
    (array axis 0).

    Args:
        hkl: Miller indices, either 3-index (h, k, l) or 4-index
            Miller-Bravais (h, k, i, l) with i = -(h + k).
        c_over_a: Ratio c/a of the hexagonal lattice.

    Returns:
        (n, 3) array of unique unit normals in (z, y, x) order.

    Raises:
        ValueError: If hkl does not have 3 or 4 indices, or if
            i != -(h + k) for a 4-index input.

    Example:
        >>> normals = hexagonal_family_normals((1, 0, -1, 1), 1.583)
        >>> normals.shape
        (12, 3)
    """
    indices = [int(index) for index in hkl]
    if len(indices) == 4:
        h, k, i, l_index = indices
        if i != -(h + k):
            raise ValueError("4-index hkl must satisfy i = -(h + k)")
    elif len(indices) == 3:
        h, k, l_index = indices
        i = -(h + k)
    else:
        raise ValueError("hkl must have 3 or 4 indices")

    # real-space basis with a = 1 (rows are a1, a2, c), x along a1
    real_basis = np.array(
        [
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3) / 2, 0.0],
            [0.0, 0.0, c_over_a],
        ]
    )
    # rows of the reciprocal basis are a1*, a2*, c* (no 2 pi factor)
    reciprocal_basis = np.linalg.inv(real_basis).T

    normals = []
    for permutation in itertools.permutations((h, k, i)):
        for sign_hki in (1, -1):
            for sign_l in (1, -1):
                g_vector = (
                    sign_hki * permutation[0] * reciprocal_basis[0]
                    + sign_hki * permutation[1] * reciprocal_basis[1]
                    + sign_l * l_index * reciprocal_basis[2]
                )
                normals.append(g_vector / np.linalg.norm(g_vector))

    # adding 0.0 turns -0.0 into 0.0 so that np.unique sees duplicates
    unique_normals = np.unique(np.round(normals, 10) + 0.0, axis=0)
    return unique_normals[:, ::-1]


def polyhedron_from_planes(
    normals: np.ndarray,
    distances: np.ndarray,
    tolerance: float = 1e-7,
) -> dict:
    """
    Convex polyhedron bounded by the half-spaces n_i . r <= d_i.

    The interior point needed by the half-space intersection is the
    Chebyshev centre (centre of the largest inscribed sphere), so the
    origin does not have to lie inside the polyhedron (e.g. a
    Winterbottom shape with a negative interface distance).

    Args:
        normals: (n, 3) outward plane normals, any length.
        distances: (n,) plane distances, in the same units as the
            returned vertices.
        tolerance: Relative tolerance for deciding that a vertex lies
            on a plane.

    Returns:
        Dictionary with:
            - "vertices": (m, 3) vertex coordinates.
            - "faces": list of (plane index, ordered vertex indices)
              for the planes that form a face.
            - "areas": (n,) face area of each input plane (0 for
              planes that do not form a face).
            - "volume": polyhedron volume.
            - "interior_point": Chebyshev centre used.

    Raises:
        ValueError: If the half-spaces do not bound a polyhedron with
            a non-empty interior.
    """
    normals = np.asarray(normals, dtype=float)
    distances = np.asarray(distances, dtype=float)
    lengths = np.linalg.norm(normals, axis=1)
    unit_normals = normals / lengths[:, None]
    unit_distances = distances / lengths

    # Chebyshev centre: maximise r subject to n_i . x + r <= d_i
    cost = np.array([0.0, 0.0, 0.0, -1.0])
    constraints = np.hstack([unit_normals, np.ones((len(lengths), 1))])
    solution = linprog(
        cost,
        A_ub=constraints,
        b_ub=unit_distances,
        bounds=[(None, None)] * 3 + [(0, None)],
    )
    if not solution.success or solution.x[3] <= 0:
        raise ValueError(
            "The planes do not bound a polyhedron with a non-empty "
            "interior (unbounded or empty)."
        )
    interior_point = solution.x[:3]

    # scipy convention for half-spaces: A x + b <= 0
    halfspaces = np.hstack([unit_normals, -unit_distances[:, None]])
    intersection = HalfspaceIntersection(halfspaces, interior_point)
    scale = np.max(np.abs(intersection.intersections))
    vertices = np.unique(
        np.round(intersection.intersections / scale, 9), axis=0
    ) * scale

    faces = []
    areas = np.zeros(len(lengths))
    for plane_index, normal in enumerate(unit_normals):
        offsets = vertices @ normal - unit_distances[plane_index]
        on_plane = np.where(np.abs(offsets) < tolerance * scale)[0]
        if on_plane.size < 3:
            continue

        # order the face vertices by angle around their centroid
        face_points = vertices[on_plane]
        centroid = face_points.mean(axis=0)
        axis_u = face_points[0] - centroid
        axis_u /= np.linalg.norm(axis_u)
        axis_v = np.cross(normal, axis_u)
        relative = face_points - centroid
        angles = np.arctan2(relative @ axis_v, relative @ axis_u)
        ordered = on_plane[np.argsort(angles)]

        # polygon area from the sum of cross products
        polygon = vertices[ordered]
        cross_sum = np.sum(
            np.cross(polygon, np.roll(polygon, -1, axis=0)), axis=0
        )
        area = 0.5 * abs(cross_sum @ normal)
        if area > (tolerance * scale) ** 2:
            faces.append((plane_index, ordered))
            areas[plane_index] = area

    return {
        "vertices": vertices,
        "faces": faces,
        "areas": areas,
        "volume": ConvexHull(vertices).volume,
        "interior_point": interior_point,
    }


def winterbottom_planes(
    normals: np.ndarray,
    surface_energies: np.ndarray,
    adhesion_energy: float,
    substrate_normal: tuple[float, float, float] = (-1.0, 0.0, 0.0),
    interface_facet_energy: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Planes of the Winterbottom construction of a supported crystal.

    Each facet sits at a distance proportional to its surface energy
    (Wulff). The facet parallel to the substrate is replaced by the
    interface plane, whose distance is proportional to
    gamma_interface - gamma_substrate = gamma_parallel - W_adh
    (Dupre relation), with gamma_parallel the surface energy of the
    metal facet parallel to the interface. W_adh = 0 gives the free
    Wulff shape, W_adh = gamma_parallel a half-crystal truncated at
    the Wulff point, and larger values flatter shapes.

    Args:
        normals: (n, 3) outward facet normals in (z, y, x) order.
        surface_energies: (n,) surface energies (any unit).
        adhesion_energy: Work of adhesion W_adh, same unit.
        substrate_normal: Outward normal of the interface plane
            (pointing into the substrate), (z, y, x). Default is -z.
        interface_facet_energy: Surface energy of the metal facet
            parallel to the interface. If None, it is taken from the
            facet in `normals` parallel to `substrate_normal`.

    Returns:
        Tuple (plane_normals, plane_distances) in units of the
        surface energies (i.e. for a Wulff scale factor of 1).

    Raises:
        ValueError: If interface_facet_energy is None and no facet is
            parallel to the substrate normal.
    """
    normals = np.asarray(normals, dtype=float)
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    energies = np.asarray(surface_energies, dtype=float)
    substrate = np.asarray(substrate_normal, dtype=float)
    substrate = substrate / np.linalg.norm(substrate)

    # facets parallel to the interface are replaced by the interface
    parallel = np.isclose(normals @ substrate, 1.0, atol=1e-9)
    if interface_facet_energy is None:
        if not np.any(parallel):
            raise ValueError(
                "No facet is parallel to the substrate normal; give "
                "interface_facet_energy explicitly."
            )
        interface_facet_energy = energies[parallel][0]

    interface_distance = interface_facet_energy - adhesion_energy
    plane_normals = np.vstack([normals[~parallel], substrate])
    plane_distances = np.append(energies[~parallel], interface_distance)
    return plane_normals, plane_distances


def winterbottom_geometry(
    normals: np.ndarray,
    surface_energies: np.ndarray,
    adhesion_energy: float,
    diameter: float | None = None,
    substrate_normal: tuple[float, float, float] = (-1.0, 0.0, 0.0),
    interface_facet_energy: float | None = None,
) -> dict:
    """
    Geometry of a Winterbottom crystal, optionally scaled to a size.

    Args:
        normals: (n, 3) outward facet normals in (z, y, x) order.
        surface_energies: (n,) surface energies.
        adhesion_energy: Work of adhesion W_adh (same unit).
        diameter: Largest in-plane extent (parallel to the
            interface) wanted for the crystal. If None, the Wulff
            scale factor is 1 (lengths in surface-energy units).
        substrate_normal: Outward normal of the interface plane,
            (z, y, x). Default is -z.
        interface_facet_energy: See :func:`winterbottom_planes`.

    Returns:
        Dictionary of :func:`polyhedron_from_planes` with the vertices
        shifted so that the interface is at height 0 and the in-plane
        centre at the origin, plus "plane_normals",
        "plane_distances" (same frame), "height" (extent along the
        interface normal), "diameter" (largest in-plane extent) and
        "aspect_ratio" (height / diameter).
    """
    plane_normals, plane_distances = winterbottom_planes(
        normals,
        surface_energies,
        adhesion_energy,
        substrate_normal=substrate_normal,
        interface_facet_energy=interface_facet_energy,
    )
    geometry = polyhedron_from_planes(plane_normals, plane_distances)
    substrate = plane_normals[-1]
    vertices = geometry["vertices"]

    # in-plane extent: largest distance between projected vertices
    heights = -(vertices @ substrate)
    projected = vertices + heights[:, None] * substrate
    width = np.max(pdist(projected))
    scale = 1.0 if diameter is None else diameter / width

    # shift: interface at height 0, in-plane centre at the origin
    projected_centre = 0.5 * (projected.min(axis=0) + projected.max(axis=0))
    shift = projected_centre + heights.min() * (-substrate)
    shifted_vertices = (vertices - shift) * scale
    shifted_distances = (plane_distances - plane_normals @ shift) * scale

    geometry.update(
        {
            "vertices": shifted_vertices,
            "areas": geometry["areas"] * scale**2,
            "volume": geometry["volume"] * scale**3,
            "interior_point": (geometry["interior_point"] - shift) * scale,
            "plane_normals": plane_normals,
            "plane_distances": shifted_distances,
            "height": (heights.max() - heights.min()) * scale,
            "diameter": width * scale,
        }
    )
    geometry["aspect_ratio"] = geometry["height"] / geometry["diameter"]
    return geometry


def make_polyhedron(
    shape: tuple[int, int, int],
    normals: np.ndarray,
    distances: np.ndarray,
    centre: tuple[float, float, float] | None = None,
    rotation: np.ndarray | tuple[float, float, float] | None = None,
    value: float = 1.0,
) -> np.ndarray:
    """
    Create a 3D convex polyhedron from bounding planes.

    A voxel at position r (relative to `centre`) is inside when
    n_i . r <= d_i for every plane i.

    Args:
        shape: 3D array shape (nz, ny, nx).
        normals: (n, 3) outward plane normals in (z, y, x) order.
        distances: (n,) plane distances in pixels (for unit
            normals).
        centre: Origin of the plane equations (z, y, x). If None,
            uses array centre.
        rotation: Rotation to apply to the polyhedron (same forms as
            :func:`make_box`).
        value: Value to fill inside the polyhedron.

    Returns:
        3D array with `value` inside the polyhedron and 0 outside.
    """
    normals = np.asarray(normals, dtype=float)
    lengths = np.linalg.norm(normals, axis=1)
    unit_normals = normals / lengths[:, None]
    unit_distances = np.asarray(distances, dtype=float) / lengths

    coords_z, coords_y, coords_x = _get_centred_coordinates(shape, centre)
    rotation_matrix = _parse_rotation_matrix(rotation)
    if rotation_matrix is not None:
        coords_z, coords_y, coords_x = _apply_rotation_to_coordinates(
            coords_z, coords_y, coords_x, rotation_matrix.T
        )

    inside = np.ones(shape, dtype=bool)
    for normal, distance in zip(unit_normals, unit_distances):
        inside &= (
            normal[0] * coords_z + normal[1] * coords_y + normal[2] * coords_x
            <= distance
        )

    array = np.zeros(shape, dtype=float)
    array[inside] = value
    return array


def make_winterbottom(
    shape: tuple[int, int, int],
    normals: np.ndarray,
    surface_energies: np.ndarray,
    adhesion_energy: float,
    diameter: float,
    substrate_normal: tuple[float, float, float] = (-1.0, 0.0, 0.0),
    interface_facet_energy: float | None = None,
    centre: tuple[float, float, float] | None = None,
    value: float = 1.0,
) -> np.ndarray:
    """
    Create a 3D supported crystal with its Winterbottom shape.

    The crystal is scaled to the requested in-plane diameter and
    placed so that the middle of its height and its in-plane centre
    are at `centre`.

    Args:
        shape: 3D array shape (nz, ny, nx).
        normals: (n, 3) outward facet normals in (z, y, x) order,
            e.g. from :func:`hexagonal_family_normals`.
        surface_energies: (n,) surface energies.
        adhesion_energy: Work of adhesion W_adh (same unit).
        diameter: Largest in-plane extent in pixels.
        substrate_normal: Outward normal of the interface plane,
            (z, y, x). Default is -z (substrate below).
        interface_facet_energy: See :func:`winterbottom_planes`.
        centre: Array position (z, y, x) of the crystal centre. If
            None, uses array centre.
        value: Value to fill inside the crystal.

    Returns:
        3D array with `value` inside the crystal and 0 outside.
    """
    geometry = winterbottom_geometry(
        normals,
        surface_energies,
        adhesion_energy,
        diameter=diameter,
        substrate_normal=substrate_normal,
        interface_facet_energy=interface_facet_energy,
    )
    # move the origin from the interface to half the crystal height
    substrate = geometry["plane_normals"][-1]
    half_height = -0.5 * geometry["height"] * substrate
    distances = (
        geometry["plane_distances"] - geometry["plane_normals"] @ half_height
    )
    return make_polyhedron(
        shape,
        geometry["plane_normals"],
        distances,
        centre=centre,
        value=value,
    )


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
