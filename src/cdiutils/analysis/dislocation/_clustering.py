import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.ndimage import label
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


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
    """
    Cluster the voxels that have a phase jump.
    """
    def create_cylinder_stencil(radius):
        """
        Create a cylinder stencil.
        """
        r = np.arange(-radius, radius + 1)
        xx, yy, zz = np.meshgrid(r, r, r, indexing="ij")
        return (xx**2 + yy**2 + zz**2) <= radius**2

    # Placeholder for user-defined functions
    def refine_cluster_with_dbscan(points, eps, min_samples):
        """
        Refine the cluster with DBSCAN.
        """

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


