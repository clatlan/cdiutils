import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.stats import zscore


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

    min_angle = np.nanmin(angles)
    # Convert angles to a numpy array for vectorized operations
    angles = np.array(angles)
    shift_tozero = angles - min_angle
    # Normalize angles to [-max_angle, max_angle]
    max_angle_new = (np.nanmax(shift_tozero)) / 2
    centered_angles = shift_tozero - max_angle_new

    return centered_angles

