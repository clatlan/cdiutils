import numpy as np

from cdiutils.io.vtk import save_as_vti
from cdiutils.utils import nan_to_zero


def extract_structure(volume, threshold=0.5):
    """Extract points from the volume where the intensity exceeds a threshold."""
    indices = np.argwhere(volume > threshold)
    return indices


def fit_line_3d(points):
    """Fit a 3D line to the given points using SVD."""
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    _, _, vh = np.linalg.svd(centered_points)
    direction = -vh[0]
    return centroid, direction


def generate_filled_cylinder(
    shape, centroid, direction, radius, height, step=1
):
    """Generate a 3D volume with a filled cylinder using disks along the fitted line."""
    direction = direction / np.linalg.norm(direction)
    volume = np.zeros(shape)

    # Generate points along the line within the specified height
    t_values = np.arange(-height / 2, height / 2, step)
    for t in t_values:
        # Compute the center of the current disk
        disk_center = centroid + t * direction

        # Create grid coordinates for the volume
        x, y, z = np.indices(shape)

        # Compute the distance of each grid point to the disk center
        distances = np.sqrt(
            (x - disk_center[0]) ** 2
            + (y - disk_center[1]) ** 2
            + (z - disk_center[2]) ** 2
        )

        # Set points within the disk radius to 1
        volume[distances <= radius] = 1

    return volume


def create_circular_mask(
    data_shape,
    centroid,
    direction,
    selected_point_index,
    r,
    dr,
    slice_thickness=2,
):
    """Create a circular mask and compute polar angles and displacement vectors from the disk center.

    Args:
        data_shape (tuple): Shape of the 3D data (e.g., (100, 100, 100)).
        centroid (np.array): Central point of the fitted line (e.g., np.array([50, 50, 50])).
        direction (np.array): Direction vector of the line (must be normalized).
        selected_point_index (float): Scalar to move along the direction vector from the centroid.
        r (float): Inner radius of the circular mask.
        dr (float): Thickness of the circular mask.
        slice_thickness (float): Thickness of the slice along the direction vector.

    Returns:
        circular_mask (np.ndarray): 3D mask with the circular region marked (1s for the mask, 0s elsewhere).
        polar_angles_masked (np.ndarray): 3D array with polar angles where the mask is applied.
        displacement_vectors (np.ndarray): 3D array storing vectors from disk center to each masked point.
    """
    selected_point_index = selected_point_index / 2  # Adjust the index scaling

    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)

    # Compute the disk center based on the selected point index along the direction
    disk_center = centroid + selected_point_index * direction

    # Define the local Z-axis (parallel to the direction vector)
    z_axis = direction

    # Define a random perpendicular vector to the Z-axis as the X-axis
    random_vector = (
        np.array([1, 0, 0]) if np.abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    )
    x_axis = np.cross(z_axis, random_vector)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Define the Y-axis as orthogonal to both Z and X
    y_axis = np.cross(z_axis, x_axis)

    # Generate a grid of all voxel indices
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(data_shape[0]),
        np.arange(data_shape[1]),
        np.arange(data_shape[2]),
        indexing="ij",
    )
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    # Shift grid points relative to the disk center
    shifted_points = grid_points - disk_center

    # Convert the shifted points to the local cylindrical coordinate system
    local_x = np.dot(shifted_points, x_axis)
    local_y = np.dot(shifted_points, y_axis)
    local_z = np.dot(shifted_points, z_axis)

    # Compute the radial distances and polar angles
    radial_distances = np.sqrt(local_x**2 + local_y**2)
    polar_angles = np.arctan2(local_y, local_x)

    # Create the circular mask within the specified radius range and slice thickness
    circular_mask = np.zeros(data_shape, dtype=np.uint8)
    circular_mask_flat = (
        (radial_distances >= r)
        & (radial_distances <= r + dr)
        & (np.abs(local_z) <= slice_thickness)
    )
    circular_mask.flat[circular_mask_flat] = 1

    # Polar angles within the mask
    polar_angles_masked = np.zeros(data_shape, dtype=np.float32)
    polar_angles_masked.flat[circular_mask_flat] = polar_angles[
        circular_mask_flat
    ]

    # Compute displacement vectors from disk center to masked points
    displacement_vectors = np.zeros(
        (*data_shape, 3), dtype=np.float32
    )  # 3D vector field
    displacement_vectors_flat = grid_points[
        circular_mask_flat
    ]  # Select only masked points
    displacement_vectors.reshape(-1, 3)[circular_mask_flat] = (
        displacement_vectors_flat  # Assign vectors
    )

    return circular_mask, polar_angles_masked, displacement_vectors, direction


def plot_phase_around_dislo(
    amp,
    phase,
    selected_dislocation_data,
    r,
    dr,
    centroid,
    direction,
    slice_thickness=1,
    selected_point_index=0,
    save_vti=False,
    fig_title=None,
    plot_debug=True,
    save_path=None,
    voxel_sizes=(1, 1, 1),
):
    """
    Plot the phase around a dislocation.

    Args:
        amp: The amplitude data.
        phase: The phase data.
        selected_dislocation_data: The selected dislocation data.
        r: The radius of the circular mask.
        dr: The thickness of the circular mask.
    Plot the phase around a dislocation.
    Args:
        amp: The amplitude data.
        phase: The phase data.
        selected_dislocation_data: The selected dislocation data.
        r: The radius of the circular mask.
        dr: The thickness of the circular mask.
        centroid: The centroid of the dislocation.
        direction: The direction of the dislocation.
    """
    # create the circular mask and polar angle map
    (
        circular_mask,
        polar_angles,
        displacement_vectors,
        direction,
    ) = create_circular_mask(
        selected_dislocation_data.shape,
        centroid,
        direction,
        selected_point_index,
        r,
        dr,
        slice_thickness=slice_thickness,
    )
    masked_region_phase = phase * circular_mask

    if save_vti:
        vect_x = displacement_vectors[..., 0]
        vect_y = displacement_vectors[..., 1]
        vect_z = displacement_vectors[..., 2]

        # Save or visualize the circular mask and polar angles#
        dict_to_vti = {
            "density": nan_to_zero(amp),
            "phase": nan_to_zero(phase),
            "dislo": selected_dislocation_data,
            "circular_mask": circular_mask,
            "polar_angles": polar_angles,
            "vect_x": vect_x,
            "vect_y": vect_y,
            "vect_z": vect_z,
        }
        save_as_vti(
            output_path=save_path, voxel_size=tuple(voxel_sizes), **dict_to_vti
        )
    return (
        masked_region_phase,
        polar_angles,
        circular_mask,
        displacement_vectors,
        direction,
    )
