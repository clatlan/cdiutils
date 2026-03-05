import math

import numpy as np


## Compute the theoretical phase due to a dislocation.
def dislo_phase_model(
    theta,
    t,
    G,
    b,
    nu=0.3,
    fact=-1,
    r=1.0,
    print_debug=False,
    only_theta_dep=True,
    print_debug_u=False,
    align_theta=None,
) -> np.ndarray:
    """
    Compute the theoretical phase shift due to a dislocation.

    Parameters:
    - theta: np.ndarray or float, polar angle(s) in radians
    - t: (3,) array, dislocation line direction
    - G: (3,) array, reciprocal lattice vector
    - b: (3,) array, Burgers vector
    - nu: float, Poisson's ratio (default = 0.3)
    - d_hkl: float, Interplanar spacing (default = 0.39239)
    - r: np.ndarray or float, radial distance(s) from dislocation core
    - print_debug: bool, whether to print debug information

    Returns:
    - u_final: np.ndarray, theoretical phase shift
    """

    # Convert inputs to NumPy arrays and ensure correct shape
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    G = np.asarray(G, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    theta = np.asarray(theta, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)

    # Compute perpendicular component of Burgers vector
    b_perp = project_vector(b, t)
    b_paral = b - b_perp
    b_par_normalized = normalize_vector(b_paral)
    b_perp_norm = np.linalg.norm(b_perp)
    if align_theta is not None:
        # here we need the reference of the exp  : means the vector from the center to the point in the ring at theta 0 (in another word the x axis of the experiment of the ring in crystallographic basis)
        theta_shift = signed_angle_3d(b_perp, align_theta, b_par_normalized)
        theta_shift_rad = np.deg2rad(theta_shift)
        if print_debug:
            print(
                f" the bper is off by {theta_shift} ° from the experimental reference"
            )
        theta += theta_shift_rad
    b_screw = np.dot(b, t / np.linalg.norm(t))

    if print_debug:
        print(f"b_perp: {b_perp}, b_perp_norm: {b_perp_norm}")
        print(f"b_screw: {b_paral}  b_screw_norm: {b_screw}")

    if np.isclose(b_perp_norm, 0) and print_debug:
        print("Warning: b_perp is zero, phase shift will be zero.")

    # Compute displacement fields
    if only_theta_dep:
        u_x_theo = (b_perp_norm / (2 * np.pi)) * (
            theta + np.sin(2 * theta) / (4 * (1 - nu))
        )
        u_y_theo = -(b_perp_norm / (8 * np.pi * (1 - nu))) * (
            np.cos(2 * theta)
        )
        u_z_theo = (b_screw / (2 * np.pi)) * theta

    else:
        u_x_theo = (b_perp_norm / (2 * np.pi)) * (
            theta + np.sin(2 * theta) / (4 * (1 - nu))
        )
        u_y_theo = -(b_perp_norm / (8 * np.pi * (1 - nu))) * (
            2 * (1 - 2 * nu) * np.log(r) + np.cos(2 * theta)
        )
        u_z_theo = (b_screw / (2 * np.pi)) * theta

    if print_debug_u:
        print(
            f"u_x_theo: {u_x_theo}, u_y_theo: {u_y_theo}, u_z_theo: {u_z_theo}"
        )

    # Compute rotation matrix from real space to dislocation frame
    R = dislo_rotation_matrix_real_to_theo(t, b)

    if print_debug:
        print(f"Rotation matrix R:\n{R}")

    # Rotate G vector
    G_theo = np.dot(R, G)

    if print_debug:
        print(f"G_theo: {G_theo}")

    # Compute phase shift
    u_final = fact * (
        G_theo[0] * u_x_theo + G_theo[1] * u_y_theo + G_theo[2] * u_z_theo
    )

    if print_debug_u:
        print(f"Final Phase Shift: {u_final}")

    return u_final


## utils for dislo_phase_model
def dislo_rotation_matrix_real_to_theo(t, b):
    """
    Compute the rotation matrix from the real (laboratory or crystal) frame
    to the dislocation (theoretical) frame.

    The dislocation frame is defined as:
    - ẑ aligned with the dislocation line direction `t`,
    - x̂ aligned with the edge component of the Burgers vector, i.e. the
      component of `b` perpendicular to `t`,
    - ŷ completing a right-handed orthonormal basis (ŷ = ẑ × x̂).

    Parameters
    ----------
    t : array_like, shape (3,)
        Dislocation line direction vector in real space. Must be non-zero.
    b : array_like, shape (3,)
        Burgers vector in real space.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Rotation matrix `R` whose rows correspond to the unit vectors
        (x̂, ŷ, ẑ) of the dislocation frame expressed in the real-space
        coordinate system. A vector `v_real` can be transformed to the
        dislocation frame via:
            v_theo = R @ v_real

    Notes
    -----
    - If the Burgers vector is parallel to the dislocation line
      (pure screw dislocation), the perpendicular component vanishes.
      In this case, an arbitrary direction perpendicular to `t` is chosen
      to define x̂.
    - The resulting basis is orthonormal and right-handed.
    - The accuracy of the rotation depends on the numerical stability of
      the normalization and projection operations.

    Examples
    --------
    >>> t = [0, 0, 1]
    >>> b = [1, 0, 0]
    >>> R = dislo_rotation_matrix_real_to_theo(t, b)
    >>> R.shape
    (3, 3)
    """
    # 1) ẑ = t̂ = t / ||t||
    t_hat = normalize_vector(t)  # new z-axis

    # 2) b_perp = b - (b·t̂) t̂  (the component of b perpendicular to t)
    b_perp = project_vector(b, t)
    b_perp_norm = np.linalg.norm(b_perp)

    # 3) x̂ = b_perp / ||b_perp||  (edge direction) unless b_perp=0 => pick any perpendicular
    if b_perp_norm < 1e-10:
        # Choose an arbitrary x-axis perpendicular to t
        temp = np.array([1.0, 0.0, 0.0])
        x_prime = temp - np.dot(temp, t_hat) * t_hat
        x_prime = normalize_vector(x_prime)
    else:
        x_prime = b_perp / b_perp_norm

    # 4) ŷ = ẑ × x̂  (right-hand rule)
    y_prime = normalize_vector(np.cross(t_hat, x_prime))

    # 5) R has rows = [x̂, ŷ, ẑ]
    R = np.array([x_prime, y_prime, t_hat])
    return R


def normalize_vector(v):
    """
    Normalize a vector to unit length.

    Parameters
    ----------
    v : array_like
        Input vector. Must have non-zero magnitude.

    Returns
    -------
    numpy.ndarray
        Unit vector in the direction of `v`.

    Raises
    ------
    ValueError
        If the input vector has zero magnitude.

    Notes
    -----
    This function performs an ℓ2 (Euclidean) normalization using
    ``np.linalg.norm``. The direction of the vector is preserved.

    Examples
    --------
    >>> normalize_vector([3, 0, 4])
    array([0.6, 0. , 0.8])
    """
    return v / np.linalg.norm(v)


def project_vector(v, t):
    """
    Compute the component of vector `v` perpendicular to vector `t`.

    This function removes the projection of `v` along `t`:
        v_perp = v - (v · t / ||t||²) t

    Parameters
    ----------
    v : array_like
        Input vector to be projected.
    t : array_like
        Reference vector defining the direction to be removed.
        Must be non-zero.

    Returns
    -------
    numpy.ndarray
        Component of `v` perpendicular to `t`.

    Raises
    ------
    ValueError
        If `t` has zero magnitude.

    Notes
    -----
    The function does not normalize the output. If a unit vector is required,
    apply `normalize_vector` to the result.

    Examples
    --------
    >>> project_vector([1, 1, 0], [1, 0, 0])
    array([0., 1., 0.])
    """
    v = np.array(v, dtype=np.float64)  # Ensure `v` is a NumPy array
    t = np.array(t, dtype=np.float64)  # Ensure `t` is a NumPy array
    return v - (np.dot(v, t) / np.linalg.norm(t) ** 2) * t


def signed_angle_3d(u, v, normal):
    """
    Compute the signed angle (in degrees) between two 3D vectors `u` and `v`,
    measured around a specified `normal` axis direction.

    The sign of the angle is determined by the direction of the cross product
    of `u` and `v` relative to `normal`.
    - Positive if the rotation from `u` to `v` is counterclockwise around `normal`.
    - Negative if the rotation is clockwise.

    Args:
        u (array-like): First 3D vector (starting vector).
        v (array-like): Second 3D vector (ending vector).
        normal (array-like): 3D vector defining the rotation axis (normal to the rotation plane).

    Returns:
        float: Signed angle in degrees.

    Example:
        >>> u = np.array([1, 0, 0])
        >>> v = np.array([0, 1, 0])
        >>> normal = np.array([0, 0, 1])
        >>> signed_angle_3d(u, v, normal)
        90.0

        >>> signed_angle_3d(v, u, normal)
        -90.0
    """
    u = np.array(u)
    v = np.array(v)
    normal = np.array(normal)

    angle = angle_between_vectors(u, v)
    cross = np.cross(u, v)
    sign = np.sign(np.dot(cross, normal))
    return angle * sign


def angle_between_vectors(u, v):
    """
    Compute the angle between two vectors in Euclidean space.

    The angle is calculated using the dot product formula:
        cos(θ) = (u · v) / (||u|| ||v||)
    and returned in degrees.

    Parameters
    ----------
    u : sequence of float
        First input vector. Must be a non-zero vector.
    v : sequence of float
        Second input vector. Must be a non-zero vector.

    Returns
    -------
    float
        Angle between vectors `u` and `v` in degrees, in the range [0, 180].

    Raises
    ------
    ValueError
        If either vector has zero magnitude.

    Notes
    -----
    The function assumes that `u` and `v` have the same dimensionality.
    Numerical errors may occur if the dot product divided by the product
    of magnitudes is slightly outside the interval [-1, 1].

    Examples
    --------
    >>> angle_between_vectors([1, 0, 0], [0, 1, 0])
    90.0
    >>> angle_between_vectors([1, 0], [1, 0])
    0.0
    """

    # Calculate dot product
    dot_product = sum(u_i * v_i for u_i, v_i in zip(u, v))

    # Calculate magnitudes
    magnitude_u = math.sqrt(sum(u_i**2 for u_i in u))
    magnitude_v = math.sqrt(sum(v_i**2 for v_i in v))

    # Calculate angle in radians and then convert to degrees
    angle_radians = math.acos(dot_product / (magnitude_u * magnitude_v))
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


def transform_known_vector_to_crystallographic(vx, vy, vz, R):
    """
    Transforms a given vector (vx, vy, vz) from the original frame to the crystallographic basis.

    Args:
        vx: X-component of the vector in the original frame (can be scalar or array)
        vy: Y-component of the vector in the original frame (can be scalar or array)
        vz: Z-component of the vector in the original frame (can be scalar or array)
        R: 3x3 rotation matrix that maps the original frame to the crystallographic basis.

    Returns:
        - Transformed vector components (vx_cryst, vy_cryst, vz_cryst) in the crystallographic basis.
    """
    # Stack vector components into a matrix form
    original_vector = np.array([vx, vy, vz]).reshape(3, -1)

    # Apply the rotation matrix (no translation)
    transformed_vector = R @ original_vector

    # Extract transformed components
    vx_cryst = transformed_vector[0].squeeze()
    vy_cryst = transformed_vector[1].squeeze()
    vz_cryst = transformed_vector[2].squeeze()

    return vx_cryst, vy_cryst, vz_cryst


def normalize_vectors_3d(vx, vy, vz):
    """
    Normalizes a set of vectors given their X, Y, and Z components.

    Args:
        vx: X-component of vectors (array or scalar)
        vy: Y-component of vectors (array or scalar)
        vz: Z-component of vectors (array or scalar)

    Returns:
        - Normalized vector components (vx_norm, vy_norm, vz_norm)
    """
    # Convert to numpy arrays if inputs are scalars
    vx, vy, vz = np.asarray(vx), np.asarray(vy), np.asarray(vz)

    # Compute vector magnitudes
    magnitudes = np.sqrt(vx**2 + vy**2 + vz**2)

    # Avoid division by zero (if magnitude is 0, set to 1 to prevent NaN)
    magnitudes = np.where(magnitudes == 0, 1, magnitudes)

    # Normalize each component
    vx_norm = vx / magnitudes
    vy_norm = vy / magnitudes
    vz_norm = vz / magnitudes

    return vx_norm, vy_norm, vz_norm


def closest_to_zero_in_array(vec):
    vec = np.asarray(vec)  # Ensure it's a NumPy array
    idx = np.argmin(np.abs(vec))  # Index of the value closest to zero
    return vec[idx], idx
