import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from cdiutils.plot import plot_volume_slices
from cdiutils.utils import hybrid_gradient, nan_to_zero, zero_to_nan

# from numbers import Number, Real
# from typing import Optional, Tuple, Union
# from cdiutils.utils import fill_up_support


def save_to_vti(
    filename,
    voxel_size,
    tuple_array,
    tuple_fieldnames,
    origin=(0, 0, 0),
    amplitude_threshold=0.01,
    **kwargs,
):
    """
    Save arrays defined by their name in a single vti file.

    Paraview expects data in an orthonormal basis (x,y,z). For BCDI data in the .cxi
    convention (hence: z, y,x) it is necessary to flip the last axis. The data sent
    to Paraview will be in the orthonormal frame (z,y,-x), therefore Paraview_x is z
    (downstream), Paraview_y is y (vertical up), Paraview_z is -x (inboard) of the
    .cxi convention.

    :param filename: the file name of the vti file
    :param voxel_size: tuple (voxel_size_axis0, voxel_size_axis1, voxel_size_axis2)
    :param tuple_array: tuple of arrays of the same dimension
    :param tuple_fieldnames: tuple of strings for the field names, same number of
     elements as tuple_array
    :param origin: tuple of points for vtk SetOrigin()
    :param amplitude_threshold: lower threshold for saving the reconstruction
     modulus (save memory space)
    :param kwargs:

    :return: nothing
    """
    import vtk
    from vtk.util import numpy_support

    if isinstance(tuple_array, np.ndarray):
        tuple_array = (tuple_array,)
    nb_arrays = len(tuple_array)
    nbz, nby, nbx = tuple_array[0].shape

    if isinstance(tuple_fieldnames, str):
        tuple_fieldnames = (tuple_fieldnames,)

    #############################
    # initialize the VTK object #
    #############################
    image_data = vtk.vtkImageData()
    image_data.SetOrigin(origin[0], origin[1], origin[2])
    image_data.SetSpacing(voxel_size[0], voxel_size[1], voxel_size[2])
    image_data.SetExtent(0, nbz - 1, 0, nby - 1, 0, nbx - 1)

    #######################################
    # check if one of the fields in 'amp' #
    #######################################
    # it will use the thresholded normalized 'amp' as support
    # when saving other fields, in order to save disk space
    try:
        index_first = tuple_fieldnames.index(
            "amp" if "amp" in tuple_fieldnames else "density"
        )
        first_array = tuple_array[index_first]
        first_array = first_array / first_array.max()
        first_array[first_array < amplitude_threshold] = (
            0  # theshold low amplitude values in order to save disk space
        )
        is_amp = True
    except ValueError:
        print(
            '"amp"/"density" not in fieldnames, will save arrays without thresholding'
        )
        index_first = 0
        first_array = tuple_array[0]
        is_amp = False

    first_arr = np.flip(np.transpose(np.flip(first_array, 2)), axis=0).reshape(
        first_array.size
    )
    first_arr = numpy_support.numpy_to_vtk(first_arr)
    pd = image_data.GetPointData()
    pd.SetScalars(first_arr)
    pd.GetArray(0).SetName(tuple_fieldnames[index_first])
    counter = 1
    for idx in range(nb_arrays):
        if idx == index_first:
            continue
        temp_array = tuple_array[idx]
        if is_amp:
            temp_array[first_array == 0] = (
                0  # use the thresholded amplitude as a support
            )
            # in order to save disk space
        temp_array = np.flip(
            np.transpose(np.flip(temp_array, 2)), axis=0
        ).reshape(temp_array.size)
        temp_array = numpy_support.numpy_to_vtk(temp_array)
        pd.AddArray(temp_array)
        pd.GetArray(counter).SetName(tuple_fieldnames[idx])
        pd.Update()
        counter = counter + 1

    # export data to file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image_data)
    writer.Write()


def map_min_gradient(
    path: str = None,
    obj: np.ndarray = None,
    voxel_size: tuple | list = [1.0, 1.0, 1.0],
    nb_of_phase_to_test: int = 10,
    font_size: int = 12,
    path_to_save: str = "",
    save_filename_vti: str = "",
    plot_debug: bool = False,
    save_plot: bool = False,
    verbose: bool = False,
):
    def calculate_displacement_gradient(phase, voxel_size):
        return hybrid_gradient(phase, *voxel_size)

    def find_closest_to_zero(gradients):
        closest_to_zero_indices = np.argmin(
            np.abs(nan_to_zero(gradients)), axis=0
        )
        shape_data = gradients.shape
        i, j, k = np.meshgrid(
            np.arange(shape_data[1]),
            np.arange(shape_data[2]),
            np.arange(shape_data[3]),
            indexing="ij",
        )
        return gradients[closest_to_zero_indices, i, j, k]

    def calculate_strain(displacement_gradient_min):
        strain_amp = (
            displacement_gradient_min[0] ** 2
            + displacement_gradient_min[1] ** 2
            + displacement_gradient_min[2] ** 2
        ) ** 0.5
        strain_amp = strain_amp / np.nanmax(strain_amp)
        strain_mask = (
            (nan_to_zero(displacement_gradient_min) != 0.0)
            .astype(float)
            .sum(axis=0)
            != 0.0
        ).astype(float)
        return strain_amp, strain_mask

    if path is not None and path != "":
        obj_list = np.array(h5py.File(path)["entry_1/data_1/data"])[0]
    elif obj is not None:
        obj_list = obj
    else:
        print("no obj or path to mode are provided")
        obj_list = None
        return None, None
    if str(np.abs(obj_list).max()) == "nan":
        obj_list = nan_to_zero(np.abs(obj_list)) * np.exp(
            1j * nan_to_zero(np.angle(obj_list))
        )

    modulus = zero_to_nan(np.abs(obj_list))
    phase_0 = np.angle(np.exp(1j * zero_to_nan(np.angle(obj_list))))

    displacement_gradient_0 = calculate_displacement_gradient(
        phase_0, voxel_size
    )
    displacement_gradient_0 = np.asarray(displacement_gradient_0)

    all_gradients = [displacement_gradient_0]
    phase_futures = []
    for i_phase in np.linspace(-2 * np.pi, 2 * np.pi, nb_of_phase_to_test):
        phase_1 = np.angle(np.exp((phase_0 + i_phase) * 1j))
        phase_futures.append(
            calculate_displacement_gradient(phase_1, voxel_size)
        )

    all_gradients.extend(phase_futures)

    all_gradients_x = [grad[0] for grad in all_gradients]
    all_gradients_y = [grad[1] for grad in all_gradients]
    all_gradients_z = [grad[2] for grad in all_gradients]

    closest_futures = [
        find_closest_to_zero(np.array(all_gradients_x)),
        find_closest_to_zero(np.array(all_gradients_y)),
        find_closest_to_zero(np.array(all_gradients_z)),
    ]
    displacement_gradient_min = np.stack(closest_futures, axis=0)

    strain_amp, strain_mask = calculate_strain(displacement_gradient_min)

    shd_0, shd_x, sh_y, sh_z = displacement_gradient_min.shape
    if verbose:
        print(
            "Displacement Gradient Min shape:", displacement_gradient_min.shape
        )

        print(
            f"\nOriginal values at position ({shd_x // 2, sh_y // 2, sh_z // 2}) for x direction:"
        )
        print(
            f"Gradient 0: {displacement_gradient_0[0][shd_x // 2, sh_y // 2, sh_z // 2]}"
        )
        print(
            f"Min value:  {displacement_gradient_min[0][shd_x // 2, sh_y // 2, sh_z // 2]}"
        )

        print(
            f"\nOriginal values at position ({shd_x // 2, sh_y // 2, sh_z // 2}) for y direction:"
        )
        print(
            f"Gradient 0: {displacement_gradient_0[1][shd_x // 2, sh_y // 2, sh_z // 2]}"
        )
        print(
            f"Min value:  {displacement_gradient_min[1][shd_x // 2, sh_y // 2, sh_z // 2]}"
        )

        print(
            f"\nOriginal values at position ({shd_x // 2, sh_y // 2, sh_z // 2}) for z direction:"
        )
        print(
            f"Gradient 0: {displacement_gradient_0[2][shd_x // 2, sh_y // 2, sh_z // 2]}"
        )
        print(
            f"Min value:  {displacement_gradient_min[2][shd_x // 2, sh_y // 2, sh_z // 2]}"
        )

    if save_filename_vti:
        save_to_vti(
            filename=save_filename_vti,
            voxel_size=list(voxel_size),
            tuple_array=(
                nan_to_zero(modulus),
                nan_to_zero(phase_0),
                zero_to_nan(phase_1),
                nan_to_zero(displacement_gradient_min[0]),
                nan_to_zero(displacement_gradient_0[0]),
                nan_to_zero(displacement_gradient_min[1]),
                nan_to_zero(displacement_gradient_0[1]),
                nan_to_zero(displacement_gradient_min[2]),
                nan_to_zero(displacement_gradient_0[2]),
                strain_mask,
                strain_amp,
            ),
            tuple_fieldnames=(
                "modulus",
                "phase_0",
                "phase_1",
                "displacement_gradient_min_x",
                "displacement_gradient_0_x",
                "displacement_gradient_min_y",
                "displacement_gradient_0_y",
                "displacement_gradient_min_z",
                "displacement_gradient_0_z",
                "strain_mask",
                "strain_amp",
            ),
            amplitude_threshold=0.1,
        )
    if plot_debug:
        rcParams["font.size"] = font_size
        rcParams.update(
            {
                "font.weight": "bold",
                "axes.titleweight": "bold",
                "axes.labelweight": "bold",
                "savefig.bbox": "tight",
            }
        )
        figure, _ = plot_volume_slices(
            zero_to_nan(phase_0),
            plot_type="contourf",
            figsize=(10, 4),
            label_size=12,
            cmap="jet",
            vmin=-np.pi,
            vmax=np.pi,
            title="original phase midlle slice",
        )
        if save_plot:
            figure.savefig(path_to_save + "original_phase.png")
        figure, _ = plot_volume_slices(
            zero_to_nan(phase_1),
            plot_type="contourf",
            figsize=(10, 4),
            label_size=12,
            cmap="jet",
            vmin=-np.pi,
            vmax=np.pi,
            title=f"phase + {np.round(i_phase, 4)} midlle slice",
        )
        if save_plot:
            figure.savefig(path_to_save + "plus_phase.png")
        figure, _ = plot_volume_slices(
            displacement_gradient_0[1],
            plot_type="contourf",
            figsize=(10, 4),
            label_size=12,
            cmap="jet",
            vmin=-0.3,
            vmax=0.3,
            title="original phase gradient midlle slice",
        )
        if save_plot:
            figure.savefig(path_to_save + "original_gradientphase.png")
        figure, _ = plot_volume_slices(
            displacement_gradient_min[1],
            plot_type="contourf",
            figsize=(10, 4),
            label_size=12,
            cmap="jet",
            vmin=-0.3,
            vmax=0.3,
            title=f"phase  + {np.round(i_phase, 4)} gradient midlle slice",
        )
        if save_plot:
            figure.savefig(path_to_save + "plus_gradientphase.png")
        rcParams["font.size"] = 12

    return nan_to_zero(strain_mask), nan_to_zero(strain_amp)


def clusters_dislo_strain_map(
    data,
    amp,
    phase,
    save_path,
    voxel_sizes,
    threshold=0.35,
    min_cluster_size=10,
    distance_threshold=10.0,
    cylinder_radius=3.0,
    num_spline_points=1000,
    smoothing_param=2,
    eps=2.0,
    min_samples=5,
    save_output=True,
    debug_plot=True,
    font_size=12,
):
    import imageio
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import label
    from scipy.spatial import cKDTree

    def create_cylinder_stencil(radius):
        r = np.arange(-radius, radius + 1)
        xx, yy, zz = np.meshgrid(r, r, r, indexing="ij")
        return (xx**2 + yy**2 + zz**2) <= radius**2

    # Placeholder for user-defined functions
    def refine_cluster_with_dbscan(points, eps, min_samples):
        from sklearn.cluster import DBSCAN

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        return clustering.labels_

    def fit_splines_to_dbscan_components(
        points, labels, smoothing_param, num_spline_points
    ):
        # Placeholder: return one spline per unique label (mocked as a straight line for now)
        splines = []
        for label_id in np.unique(labels):
            if label_id == -1:
                continue
            component_points = points[labels == label_id]
            if len(component_points) < 2:
                continue
            sorted_points = component_points[
                np.argsort(component_points[:, 0])
            ]
            splines.append(sorted_points)
        return splines

    binary_data = (data > threshold).astype(np.uint8)
    labeled_data, num_clusters = label(binary_data)
    print(f"Number of clusters identified: {num_clusters}")

    filtered_clusters = np.zeros_like(labeled_data)
    cluster_points = {}

    for cluster_id in range(1, num_clusters + 1):
        cluster_indices = np.argwhere(labeled_data == cluster_id)
        if len(cluster_indices) >= min_cluster_size:
            filtered_clusters[labeled_data == cluster_id] = cluster_id
            cluster_points[cluster_id] = cluster_indices

    print(f"Filtered clusters: {np.unique(filtered_clusters)[1:]}")

    merge_mapping = {}
    cluster_ids = list(cluster_points.keys())

    for i, cluster_id_a in enumerate(cluster_ids):
        for j in range(i + 1, len(cluster_ids)):
            cluster_id_b = cluster_ids[j]
            points_a = cluster_points[cluster_id_a]
            points_b = cluster_points[cluster_id_b]
            tree_a = cKDTree(points_a)
            tree_b = cKDTree(points_b)
            dists = tree_a.sparse_distance_matrix(
                tree_b, distance_threshold, output_type="ndarray"
            )
            if dists.size > 0:
                merge_mapping[cluster_id_b] = cluster_id_a

    merged_clusters = np.zeros_like(filtered_clusters)
    for cluster_id in np.unique(filtered_clusters):
        if cluster_id == 0:
            continue
        current_label = cluster_id
        while current_label in merge_mapping:
            current_label = merge_mapping[current_label]
        merged_clusters[filtered_clusters == cluster_id] = current_label

    cylindrical_mask = np.zeros_like(merged_clusters)
    stencil = create_cylinder_stencil(cylinder_radius)

    for cluster_id in range(1, np.max(merged_clusters) + 1):
        if cluster_id not in cluster_points:
            continue
        cluster_indices = np.vstack(cluster_points[cluster_id])
        dbscan_labels = refine_cluster_with_dbscan(
            cluster_indices, eps=eps, min_samples=min_samples
        )
        splines = fit_splines_to_dbscan_components(
            cluster_indices, dbscan_labels, smoothing_param, num_spline_points
        )

        for spline_points in splines:
            for point in spline_points:
                x_center, y_center, z_center = point.astype(int)
                r = int(cylinder_radius)
                x_min, x_max = x_center - r, x_center + r + 1
                y_min, y_max = y_center - r, y_center + r + 1
                z_min, z_max = z_center - r, z_center + r + 1
                if (
                    x_min < 0
                    or y_min < 0
                    or z_min < 0
                    or x_max > cylindrical_mask.shape[0]
                    or y_max > cylindrical_mask.shape[1]
                    or z_max > cylindrical_mask.shape[2]
                ):
                    continue
                cylindrical_mask[x_min:x_max, y_min:y_max, z_min:z_max] |= (
                    stencil
                )

    print("Cylindrical mask constructed.")
    final_labeled_clusters, num_final_clusters = label(cylindrical_mask > 0)

    if debug_plot:
        rcParams["font.size"] = font_size
        rcParams.update(
            {
                "font.weight": "bold",
                "axes.titleweight": "bold",
                "axes.labelweight": "bold",
                "savefig.bbox": "tight",
            }
        )
        frames = []
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1, 1, 1])

        cluster_indices = np.argwhere(final_labeled_clusters > 0)
        scatter = ax.scatter(
            cluster_indices[:, 0],
            cluster_indices[:, 1],
            cluster_indices[:, 2],
            s=1,
            c=final_labeled_clusters[final_labeled_clusters > 0],
            cmap="jet",
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label("Cluster Labels")
        ax.set_title("Refined Clustering")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        for angle in range(0, 180, 4):  # Fewer frames
            ax.view_init(30, angle)
            plt.draw()
            buf, (w, h) = fig.canvas.print_to_buffer()
            rgba = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
            frame = rgba[..., :3].copy()
            # frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8").reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)

        gif_path = (
            save_path
            + "_Step1_refined_dislocation_clustering_and_processing.gif"
        )
        imageio.mimsave(gif_path, frames, fps=10)
        print(f"Saved debug GIF to {gif_path}")
        rcParams["font.size"] = 12

    return final_labeled_clusters, num_final_clusters


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
    # Create the circular mask and polar angle map
    circular_mask, polar_angles, displacement_vectors, direction = (
        create_circular_mask(
            selected_dislocation_data.shape,
            centroid,
            direction,
            selected_point_index,
            r,
            dr,
            slice_thickness=slice_thickness,
        )
    )
    masked_region_phase = phase * circular_mask

    if save_vti:
        vect_x = displacement_vectors[..., 0]
        vect_y = displacement_vectors[..., 1]
        vect_z = displacement_vectors[..., 2]

        # Save or visualize the circular mask and polar angles
        save_to_vti(
            filename=save_path + ".vti",
            voxel_size=tuple(voxel_sizes),
            tuple_array=(
                nan_to_zero(amp),
                nan_to_zero(phase),
                selected_dislocation_data,
                circular_mask,
                polar_angles,
                vect_x,
                vect_y,
                vect_z,
            ),
            tuple_fieldnames=(
                "density",
                "phase",
                "dislo",
                "circular_mask",
                "polar_angles",
                "vect_x",
                "vect_y",
                "vect_z",
            ),
            amplitude_threshold=0.01,
        )
    return (
        masked_region_phase,
        polar_angles,
        circular_mask,
        displacement_vectors,
        direction,
    )


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


def remove_large_jumps(x, y, threshold_factor=1.5):
    """
    Removes points with large jumps in the y-data based on a threshold.

    Args:
        x (np.ndarray): The x-values of the data.
        y (np.ndarray): The y-values of the data.
        threshold_factor (float): The factor for the threshold to detect large jumps.

    Returns:
        x_clean (np.ndarray): The x-values with large jumps removed.
        y_clean (np.ndarray): The y-values with large jumps removed.
        dy (np.ndarray): The computed differences for each point.
    """
    # Compute differences index-by-index, including the first and last points
    dy = np.zeros(len(y))

    # For the first point, difference with the next point

    # For the intermediate points, take the max difference with neighbors
    for i in range(1, len(y) - 1):
        dy[i] = max(np.abs(y[i + 1] - y[i]), np.abs(y[i - 1] - y[i]))

    # For the last point, difference with the previous point
    dy[-1] = np.max([np.abs(y[-1] - y[i]) for i in range(-4, -1)])
    dy[0] = np.max([np.abs(y[0] - y[i]) for i in range(1, 3)])

    # Define a threshold for identifying large jumps
    threshold = threshold_factor * np.std(dy)

    # Create a mask for valid points (where the jump is below the threshold)
    valid_mask = dy < threshold

    # Filter the data to remove points with large jumps
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]

    return x_clean, y_clean


def center_angles(angles):
    """
    Centers a list of angles between -max_angle and max_angle.

    Parameters:
        angles (list or np.ndarray): List of angles in degrees or radians.
        max_angle (float): Maximum angle for centering.

    Returns:
        np.ndarray: Angles centered between -max_angle and max_angle.
    """
    import numpy as np

    min_angle = np.nanmin(angles)
    # Convert angles to a numpy array for vectorized operations
    angles = np.array(angles)
    shift_tozero = angles - min_angle
    # Normalize angles to [-max_angle, max_angle]
    max_angle_new = (np.nanmax(shift_tozero)) / 2
    centered_angles = shift_tozero - max_angle_new

    return centered_angles


def dislo_process_phase_ring(
    angle,
    phase,
    displacement_vectors,
    factor_phase=1,
    poly_order=1,
    jump_filter_ML=False,
    jump_filter_gradient_only=False,
    filter_by_slope=False,
    plot_debug=False,
    save_path=None,
    period_jump=360,
    font_size=12,
    figsize=(12, 18),
    markersize=10,
    linewidth=1,
):
    """
    Processes the phase and angle data to analyze dislocation properties in a phase ring.

    This function:
    1. Extracts and filters nonzero phase values.
    2. Sorts the phase and angle data.
    3. Removes phase jumps and outliers using an adaptive filtering method.
    4. Unwraps and centers the phase data to ensure phase continuity.
    5. Applies a Savitzky-Golay filter to smooth phase variations.
    6. Removes polynomial trends dynamically from the phase data.
    7. Tracks filtered data indices and visualizes the selection.
    8. Visualizes displacement vectors alongside phase data.

    Args:
        angle (np.ndarray): The angle data.
        phase (np.ndarray): The phase data.
        displacement_vectors (np.ndarray): The displacement vectors associated with the phase data.
        factor_phase (float, optional): Scaling factor applied to phase data. Defaults to 1.
        poly_order (int, optional): Order of polynomial fit for trend removal. Defaults to 1 (linear).
        jump_filter (bool, optional): If True, applies phase jump removal and outlier filtering.
        plot_debug (bool, optional): If True, generates detailed debugging plots.
        save_path (str, optional): Path to save the debug plots.

    Returns:
        tuple: A tuple containing:
            - angle_raw (np.ndarray): Original angle data (before processing).
            - phase_raw (np.ndarray): Original phase data (before processing).
            - angle_final (np.ndarray): Processed angle data after jump removal.
            - phase_final (np.ndarray): Processed phase data after unwrapping and centering.
            - phase_ring_1_smooth (np.ndarray): Smoothed phase data.
            - phase_sinu (np.ndarray): Sinusoidal phase deviation after polynomial trend removal.
            - displacement_vectors_ring_sorted (np.ndarray): Sorted displacement vectors before filtering.
            - displacement_vectors_final (np.ndarray): Displacement vectors after filtering.
            - sel___ (np.ndarray): Boolean mask indicating selected (kept) data points.
    """

    def filter_phase_data(
        angle_ring,
        phase_ring,
        adaptive_threshold_factor=2.0,
        median_filter_sizes=(3, 7),
        zscore_threshold=2.8,
    ):
        """
        Filters phase data by removing large phase jumps, applying an adaptive median filter,
        and filtering out statistical outliers. Also tracks the selected indices.

        Parameters:
        - angle_ring (numpy array): Angle values in degrees.
        - phase_ring (numpy array): Phase values in degrees.
        - adaptive_threshold_factor (float): Factor for detecting large jumps based on standard deviation.
        - median_filter_sizes (tuple): (small, large) filter sizes for adaptive filtering.
        - zscore_threshold (float): Threshold for filtering out extreme outliers.

        Returns:
        - angle_filtered (numpy array): Filtered angle values.
        - phase_filtered (numpy array): Filtered phase values.
        - selected_indices (numpy array): Indices of the selected data points in the original array.
        """
        import numpy as np
        from scipy.ndimage import median_filter
        from scipy.stats import zscore

        original_indices = np.arange(len(angle_ring))  # Track original indices

        # Step 1: Identify Large Phase Jumps
        diff_phi = np.abs(np.diff(phase_ring, append=phase_ring[-1]))
        threshold_jump = np.median(
            diff_phi
        ) + adaptive_threshold_factor * np.std(diff_phi)

        # Identify large jumps
        large_jump_indices = np.where(diff_phi > threshold_jump)[0]

        if len(large_jump_indices) > 0:
            # Correct only the largest discontinuity
            diff_phi_positionmax = np.argmax(diff_phi)
            phase_shift = (
                phase_ring[diff_phi_positionmax]
                - phase_ring[diff_phi_positionmax - 1]
            )
            phase_ring[diff_phi_positionmax:] -= (
                phase_shift  # Adjust phase after the jump
            )

        # Step 2: Apply Adaptive Median Filter
        phase_ring_smoothed = median_filter(
            phase_ring, size=median_filter_sizes[0]
        )

        # Apply larger filtering only where large jumps occur
        for idx in large_jump_indices:
            if idx > 2 and idx < len(phase_ring) - 2:
                phase_ring_smoothed[idx] = np.median(
                    phase_ring[idx - 2 : idx + 3]
                )

        # Step 3: Use an Adaptive Threshold for Filtering
        diff_phi = np.abs(
            np.diff(phase_ring_smoothed, append=phase_ring_smoothed[-1])
        )
        adaptive_threshold = np.median(
            diff_phi
        ) + adaptive_threshold_factor * np.std(diff_phi)
        FILTER_DIFF_ = diff_phi < adaptive_threshold

        # Apply filtering
        angle_filtered, phase_filtered, selected_indices = (
            angle_ring[FILTER_DIFF_],
            phase_ring_smoothed[FILTER_DIFF_],
            original_indices[FILTER_DIFF_],
        )

        # Step 4: Final Cleanup with Z-Score Filtering
        z_scores = np.abs(zscore(phase_filtered))  # type: ignore
        final_selection = (
            z_scores < zscore_threshold
        )  # Final mask after Z-score filtering

        return (
            angle_filtered[final_selection],
            phase_filtered[final_selection],
            selected_indices[final_selection],
        )

    def filter_by_slope_deviation(
        x, phase, slope_target=1.0, slope_tol=0.3, min_cluster=5, pad=3
    ):
        """
        Remove regions where unwrapped phase vs x deviates from the expected slope.

        Parameters:
        - x: 1D array (angle or position)
        - phase: 1D array (raw phase)
        - slope_target: expected slope (usually 1)
        - slope_tol: allowed deviation (±)
        - min_cluster: minimum length of abnormal region
        - pad: how many extra points to mask on each side of a bad region

        Returns:
        - x_filtered, phase_filtered: filtered data arrays
        - bad_indices: indices of removed points
        """
        x = np.array(x)
        phase = np.unwrap(phase)

        dx = np.diff(x)
        dphase = np.diff(phase)
        local_slope = dphase / dx
        local_slope = np.concatenate(
            [[local_slope[0]], local_slope]
        )  # same size as input

        # Define bad slope mask
        bad_slope = np.abs(local_slope - slope_target) > slope_tol

        # Group and mask extended regions
        bad_mask = np.zeros_like(phase, dtype=bool)
        i = 0
        while i < len(bad_slope):
            if bad_slope[i]:
                start = i
                while i < len(bad_slope) and bad_slope[i]:
                    i += 1
                end = i
                if end - start >= min_cluster:
                    bad_mask[
                        max(0, start - pad) : min(len(phase), end + pad)
                    ] = True
            else:
                i += 1

        # Filter good data
        x_filtered = x[~bad_mask]
        phase_filtered = phase[~bad_mask]
        bad_indices = np.where(bad_mask)[0]
        good_indices = np.where(~bad_mask)[0]
        return x_filtered, phase_filtered, good_indices, bad_indices

    def remove_large_jumps_alter_unwrap(y, threshold=10):
        """
        Detects and removes large jumps in y based on a given threshold.

        Parameters:
            y (numpy array): Dependent variable (e.g., phase or measured value).
            threshold (float): Threshold for detecting large jumps.

        Returns:
            numpy array: Corrected y values.
        """
        y_fixed = y.copy()
        y_diff = np.diff(y)
        if np.abs(y_diff).max() < threshold:
            return y
        else:
            jumps = np.where(np.abs(y_diff) > threshold)[0]
            for j in jumps:
                y_fixed[j + 1 :] -= y_diff[
                    j
                ]  # Shift the remaining data to remove jump

            return y_fixed

    from scipy.signal import savgol_filter

    # Extract indices where phase is nonzero
    nonzero_indices = np.nonzero(phase)
    displacement_vectors_ring = displacement_vectors[nonzero_indices]
    angle_ring = angle[nonzero_indices].flatten()
    phase_ring = phase[nonzero_indices].flatten()

    # Sort by angle
    sort_indices = np.argsort(angle_ring)
    angle_ring = angle_ring[sort_indices]
    phase_ring = phase_ring[sort_indices]
    displacement_vectors_ring_sorted = displacement_vectors_ring[sort_indices]

    # Convert phase to degrees
    phase_ring = phase_ring * (180 / np.pi)
    angle_ring *= 180 / np.pi

    # Store raw data
    phase_raw, angle_raw = phase_ring.copy(), angle_ring.copy()
    if jump_filter_ML:
        # Select displacement vectors corresponding to filtered indices
        sel___ = np.zeros_like(angle_ring, dtype=bool)
        angle_ring, phase_ring, filtered_indices = filter_phase_data(
            angle_ring, phase_ring
        )
        displacement_vectors_final = displacement_vectors_ring_sorted[
            filtered_indices
        ]
        # Create a mask for selected (kept) points
        sel___[filtered_indices] = True  # Mark selected indices as True
    elif jump_filter_gradient_only:
        phase_ring = remove_large_jumps_alter_unwrap(phase_ring)
        displacement_vectors_final = displacement_vectors_ring_sorted.copy()
    elif filter_by_slope:
        sel___ = np.zeros_like(angle_ring, dtype=bool)
        angle_ring, phase_ring, filtered_indices, bad_indices = (
            filter_by_slope_deviation(
                angle_ring,
                phase_ring,
                slope_target=1.0,
                slope_tol=0.5,
                min_cluster=5,
                pad=3,
            )
        )
        displacement_vectors_final = displacement_vectors_ring_sorted[
            filtered_indices
        ]
        # Create a mask for selected (kept) points
        sel___[filtered_indices] = True  # Mark selected indices as True
    else:
        displacement_vectors_final = displacement_vectors_ring_sorted.copy()

    phase_final = np.unwrap(phase_ring, period=period_jump)
    phase_final = np.unwrap(phase_final, period=period_jump)
    print("Raw angle :", np.min(angle_raw), np.max(angle_raw))
    print("Raw phase :", np.min(phase_raw), np.max(phase_raw))

    print("unwrapped phase :", np.min(phase_final), np.max(phase_final))
    phase_final = center_angles(phase_final + angle_ring) - angle_ring

    # Remove polynomial trend
    poly_coeffs = np.polyfit(angle_ring, phase_final, poly_order)
    slope, intercept = poly_coeffs
    # if (slope >1.2) or ((slope <0.9)):
    # slope = factor_phase * 1.0
    poly_coeffs = slope, intercept
    poly_fit = np.polyval(poly_coeffs, angle_ring)
    phase_sinu = center_angles(phase_final - poly_fit)

    # Apply Savitzky-Golay filter
    window_length = min(100, len(phase_final) - 1)
    if window_length % 2 == 0:
        window_length -= 1
    phase_ring_1_smooth_sinu = center_angles(
        savgol_filter(
            phase_sinu,
            window_length=window_length,
            polyorder=min(poly_order, window_length - 1),
        )
    )
    phase_ring_1_smooth = phase_ring_1_smooth_sinu + slope * angle_ring

    ### --- Debug Plotting --- ###
    if plot_debug:
        rcParams["font.size"] = font_size
        rcParams.update(
            {
                "font.weight": "bold",
                "axes.titleweight": "bold",
                "axes.labelweight": "bold",
                "savefig.bbox": "tight",
            }
        )
        fig, axes = plt.subplots(6, 1, figsize=figsize, sharex=True)

        axes[0].plot(
            angle_raw,
            phase_raw,
            ">",
            label="Raw Phase",
            color="black",
            alpha=0.7,
            linewidth=linewidth,
        )
        if jump_filter_ML:
            axes[0].plot(
                angle_raw[~sel___],
                phase_raw[~sel___],
                ".",
                label="Filtered Out",
                color="red",
                alpha=0.7,
                linewidth=linewidth,
                markersize=markersize,
            )  # type: ignore

        axes[0].set_title("Raw Phase Data")
        axes[0].legend()

        axes[1].plot(
            angle_ring,
            phase_final,
            ">",
            label="Processed Phase",
            color="blue",
            alpha=0.7,
            linewidth=linewidth,
            markersize=markersize,
        )
        axes[1].set_title("Processed Phase (Unwrapped & Centered)")
        axes[1].legend()

        axes[2].plot(
            angle_ring,
            phase_ring_1_smooth,
            ">",
            label="Smoothed Phase",
            color="red",
            alpha=0.7,
            linewidth=linewidth,
            markersize=markersize,
        )
        axes[2].set_title("Smoothed Phase (Savitzky-Golay)")
        axes[2].legend()

        axes[3].plot(
            angle_ring,
            phase_sinu,
            ">",
            label="Phase Sinusoidal Deviation",
            color="green",
            alpha=0.7,
            linewidth=linewidth,
            markersize=markersize,
        )
        poly_eq_str = " + ".join(
            f"{coef:.2f} θ^{i}" for i, coef in enumerate(poly_coeffs[::-1])
        )
        axes[3].set_title(
            f"Phase Sinusoidal Deviation (Trend Removed: {poly_eq_str})"
        )
        axes[3].legend()

        # Overlay all plots
        axes[4].plot(
            angle_raw,
            phase_raw,
            ">-",
            label="Raw Phase",
            color="black",
            alpha=0.5,
            linewidth=2,
            markersize=markersize,
        )
        axes[4].plot(
            angle_ring,
            phase_final,
            ">-",
            label="Processed Phase",
            color="blue",
            alpha=0.6,
            linewidth=linewidth,
            markersize=markersize,
        )
        axes[4].plot(
            angle_ring,
            phase_ring_1_smooth,
            ">-",
            label="Smoothed Phase",
            color="red",
            alpha=0.7,
            linewidth=linewidth,
            markersize=markersize,
        )
        axes[4].set_title("All Phase Data Overlaid")
        axes[4].legend()

        # **NEW PLOT: Displacement Vectors as Quiver**
        # displacement_magnitudes = np.linalg.norm(displacement_vectors_final, axis=1)
        axes[5].plot(
            angle_ring,
            displacement_vectors_final[..., 0],
            ">",
            label="Displacement Vector X",
            alpha=0.7,
            linewidth=linewidth,
            markersize=markersize,
        )
        axes[5].plot(
            angle_ring,
            displacement_vectors_final[..., 1],
            "<",
            label="Displacement Vector Y",
            alpha=0.7,
            linewidth=linewidth,
            markersize=markersize,
        )
        axes[5].plot(
            angle_ring,
            displacement_vectors_final[..., 2],
            "^",
            label="Displacement Vector Z",
            alpha=0.7,
            linewidth=linewidth,
            markersize=markersize,
        )

        axes[5].set_title("Displacement Vector Magnitudes vs. Angle")
        axes[5].set_ylabel("Vector Magnitude")
        axes[5].set_xlabel("Angle (Degrees)")
        axes[5].legend()

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        rcParams["font.size"] = 12

    return (
        angle_raw,
        phase_raw,
        angle_ring,
        phase_final,
        phase_ring_1_smooth,
        phase_sinu,
        displacement_vectors_ring_sorted,
        displacement_vectors_final,
    )


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
    import numpy as np

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
    import numpy as np

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
):
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
    import math

    # Calculate dot product
    dot_product = sum(u_i * v_i for u_i, v_i in zip(u, v))

    # Calculate magnitudes
    magnitude_u = math.sqrt(sum(u_i**2 for u_i in u))
    magnitude_v = math.sqrt(sum(v_i**2 for v_i in v))

    # Calculate angle in radians and then convert to degrees
    angle_radians = math.acos(dot_product / (magnitude_u * magnitude_v))
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


## utils phase decomposition
def decompose_experimental_phase(theta, phi_exp):
    """
    Decompose an experimental phase signal into linear, low-frequency,
    and second-harmonic (2θ) oscillatory components.

    The procedure consists of:
    1) Removing a global linear trend from the experimental phase.
    2) Fitting and subtracting low-frequency angular components
       (cos θ, sin θ, and constant offset).
    3) Isolating and fitting the second-harmonic oscillation
       (cos 2θ, sin 2θ).
    4) Reconstructing filtered phase components with proper angular
       centering.

    Parameters
    ----------
    theta : array_like
        Angular coordinate (in radians) at which the phase is sampled.
    phi_exp : array_like
        Experimental phase values corresponding to `theta`.

    Returns
    -------
    f_oscillation_final : numpy.ndarray
        Experimental phase after removal of low-frequency components,
        retaining the dominant oscillatory content and linear trend.
    f_fitoscillation_final : numpy.ndarray
        Fitted second-harmonic (cos 2θ, sin 2θ) oscillatory component,
        centered in angular space.
    coeffs : numpy.ndarray
        Least-squares coefficients for the second-harmonic fit
        [A_cos2θ, A_sin2θ].
    f_linear : numpy.ndarray
        Linear trend reconstructed from the filtered phase.
    coeffs_linear : numpy.ndarray
        Coefficients of the final linear fit [slope, intercept].

    Notes
    -----
    - Least-squares fitting is performed using ``np.linalg.lstsq``.
    - Low-frequency contributions (cos θ, sin θ, constant) are explicitly
      removed to avoid contamination of the 2θ harmonic.
    - The function assumes that `theta` and `phi_exp` have the same shape.
    - The function `center_angles` is expected to wrap or center angular
      phase values consistently within a chosen interval (e.g. [-π, π]).

    Examples
    --------
    >>> f_phase, f_fit2theta, coeffs2, f_lin, lin_coeffs = \
    ...     decompose_experimental_phase(theta, phi_exp)
    """
    coeffs_linear = np.polyfit(theta, phi_exp, 1)
    f_linear = np.polyval(coeffs_linear, theta)
    f_oscillation_0 = phi_exp - f_linear

    X_full = np.column_stack(
        [
            np.cos(theta),
            np.sin(theta),
            np.cos(2 * theta),
            np.sin(2 * theta),
            np.ones_like(theta),
        ]
    )
    coeffs, *_ = np.linalg.lstsq(X_full, f_oscillation_0, rcond=None)
    low_freq_fit = X_full[:, [0, 1, 4]] @ coeffs[[0, 1, 4]]
    f_oscillation_final = center_angles(f_oscillation_0 - low_freq_fit)
    f_filterlowfreq_final = f_oscillation_final + f_linear
    coeffs_linear = np.polyfit(theta, f_filterlowfreq_final, 1)
    f_linear = np.polyval(coeffs_linear, theta)
    f_oscillation_final = f_filterlowfreq_final

    X_full = np.column_stack(
        [
            np.cos(2 * theta),
            np.sin(2 * theta),
        ]
    )
    coeffs, *_ = np.linalg.lstsq(X_full, f_oscillation_0, rcond=None)
    high_freq_fit = X_full @ coeffs
    f_fitoscillation_final = center_angles(high_freq_fit)
    return (
        f_oscillation_final,
        f_fitoscillation_final,
        coeffs,
        f_linear,
        coeffs_linear,
    )


# plotting
def plot_phase_data_comparison_exp_to_theo(
    exp_angle,
    exp_phase,
    theo_phases,
    labels,
    save_path=None,
    marker_size=5,
    line_width=5,
    alpha=0.7,
    ref_band=0.1,
    show_band=True,
    figsize=(9, 5),
    band_theo=False,
    show_slope=False,
    filter_low_freq=True,
    fix_exp_slope=None,
    font_size=12,
    offset_theta=None,
    ncol=3,
):
    rcParams["font.size"] = font_size
    rcParams.update(
        {
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "savefig.bbox": "tight",
        }
    )
    # Preprocess experimental phase
    filtred_phase, filter_fit, _, f_linear, coeffs_linear = (
        decompose_experimental_phase(exp_angle, exp_phase)
    )
    slope_exp, intercept_exp = coeffs_linear
    if fix_exp_slope is not None:
        slope_exp = fix_exp_slope
        f_linear = slope_exp * exp_angle + intercept_exp

    y_exp = (
        filtred_phase - f_linear if filter_low_freq else exp_phase - f_linear
    )

    if offset_theta is None:
        offset_theta = 0
    # Setup 2-row subplot (main + residual)
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    # === MAIN PLOT ===
    if show_slope:
        exp_legend = f"Exp (slope={slope_exp:.2f})"
    else:
        exp_legend = "Exp"

    h_exp = ax1.plot(
        exp_angle - offset_theta,
        y_exp,
        "^",
        markersize=marker_size,
        label=exp_legend,
        color="black",
        alpha=alpha,
    )[0]
    if show_band:
        ax1.fill_between(
            exp_angle - offset_theta,
            y_exp - ref_band,
            y_exp + ref_band,
            color="gray",
            alpha=0.2,
        )
    handles = [h_exp]
    labels_all = [exp_legend]

    for i, theo_phase in enumerate(theo_phases):
        predicted_phase, pred_fit, _, f_linear_theo, coeffs_linear = (
            decompose_experimental_phase(exp_angle, theo_phase)
        )
        y_theo = (
            predicted_phase - f_linear_theo
            if filter_low_freq
            else theo_phase - f_linear_theo
        )
        slope_theo, intercept_theo = coeffs_linear
        if show_slope:
            theo_legend = labels[i] + f" (slope={slope_theo:.2f})"
        else:
            theo_legend = labels[i]

        (line,) = ax1.plot(
            exp_angle - offset_theta,
            y_theo,
            "-",
            linewidth=line_width,
            label=theo_legend,
            alpha=alpha,
        )
        handles.append(line)
        labels_all.append(theo_legend)

        if show_band and band_theo:
            ax1.fill_between(
                exp_angle - offset_theta,
                y_theo - ref_band,
                y_theo + ref_band,
                color=line.get_color(),
                alpha=0.2,
            )

        # === RESIDUAL SUBPLOT ===
        y_diff = center_angles(y_exp - y_theo)
        ax2.plot(
            exp_angle - offset_theta,
            y_diff,
            "-",
            linewidth=line_width,
            color=line.get_color(),
            alpha=alpha,
        )

    # === Styling for Main Plot ===
    ax1.set_ylabel("Phase Residual (rad)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.tick_params(labelsize=font_size - 2)

    # === Styling for Residual Subplot ===
    ax2.set_xlabel("Polar Angle (rad)")
    ax2.set_ylabel("Diff.")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.tick_params(labelsize=font_size - 2)

    # === Unified Legend Above Plots ===
    fig.legend(
        handles,
        labels_all,
        loc="upper center",
        frameon=False,
        ncol=ncol,
        bbox_to_anchor=(0.5, 1.12),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    rcParams["font.size"] = 12

    plt.show()
