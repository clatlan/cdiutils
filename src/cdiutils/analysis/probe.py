import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass

from matplotlib.patches import Ellipse
import matplotlib.gridspec as gridspec

from cdiutils.utils import CroppingHandler
from cdiutils.plot.formatting import add_colorbar


def get_width_metrics(
        profile: np.ndarray, axis_values: np.ndarray, verbose: bool = False
) -> dict:
    """
    Compute the FWHM and other width metrics of a given profile.
    The function uses the `find_peaks` and `peak_widths` functions from
    `scipy.signal` to find the peaks and calculate the full width at half
    maximum (FWHM) and full width at 10% maximum (FW10%M) of the profile.

    Args:
        profile (np.ndarray): the profile data to be analysed.
        axis_values (np.ndarray): the axis values corresponding to the
            profile data.
        verbose (bool, optional): whether to print out info. Defaults to
            False.

    Returns:
        dict: a dictionary containing the FWHM and FW10%M values, their
            indices, heights, and boundaries.
    """

    # Find peaks
    peaks, properties = find_peaks(profile, height=0.5 * np.max(profile))

    # Use the highest peak or the middle if no peak is found
    if len(peaks) > 0:
        max_index = np.argmax(properties["peak_heights"])
        highest_peak_idx = peaks[max_index]
    else:
        highest_peak_idx = len(profile) // 2

    # for absolute FWHM - using original peak_widths method
    fwhm_indices, fwhm_height, left_idx, right_idx = peak_widths(
        profile, [highest_peak_idx], rel_height=0.5
    )

    # for absolute FW10%M - using peak_widths with 0.9 relative height
    # (10% from max)
    fw10m_indices, fw10m_height, left10_idx, right10_idx = peak_widths(
        profile, [highest_peak_idx], rel_height=0.9
    )

    # Convert width indices to physical units using axis values
    # for FWHM
    left_pos = np.interp(left_idx[0], np.arange(len(axis_values)), axis_values)
    right_pos = np.interp(
        right_idx[0], np.arange(len(axis_values)), axis_values
    )
    fwhm = abs(right_pos - left_pos)

    # for FW10%M
    left10_pos = np.interp(
        left10_idx[0], np.arange(len(axis_values)), axis_values
    )
    right10_pos = np.interp(
        right10_idx[0], np.arange(len(axis_values)), axis_values
    )
    fw10m = abs(right10_pos - left10_pos)

    # Statistical FWHM using Gaussian fitting
    def gaussian(x, amp, mean, sigma, offset):
        return amp * np.exp(-((x - mean) ** 2) / (2 * sigma**2)) + offset

    try:
        # Fit Gaussian to profile
        p0 = [
            np.max(profile),
            axis_values[highest_peak_idx],
            fwhm / 2.355,
            0,
        ]  # Initial guess
        params, _ = curve_fit(gaussian, axis_values, profile, p0=p0)
        gauss_fwhm = 2.355 * params[2]  # FWHM = 2.355 * sigma for Gaussian

        # generate fitted curves for plotting
        gauss_fit = gaussian(axis_values, *params)

        fit_success = True
    except Exception as e:
        if verbose:
            print(f"Gaussian fitting failed: {e}")
        gauss_fwhm = float("nan")
        gauss_fit, params = None, None
        fit_success = False

    if verbose:
        print(
            f"Highest peak at index {highest_peak_idx}, "
            f"pos = {axis_values[highest_peak_idx]:.2e}, "
            f"value = {profile[highest_peak_idx]:.2e}\n"
            f"FWHM of the probe: {fwhm:.2e} m "
            f"({fwhm_indices[0]:.1f} pixels)\n"
        )
    return {
        "highest_peak_idx": highest_peak_idx,
        "fwhm": {
            "value": fwhm,
            "indices": fwhm_indices[0],
            "height": fwhm_height[0],
            "boundaries": (left_pos, right_pos),
        },
        "fw10m": {
            "value": fw10m,
            "indices": fw10m_indices[0],
            "height": fw10m_height[0],
            "boundaries": (left10_pos, right10_pos),
        },
        "gaussian": {
            "value": gauss_fwhm,
            "fit": gauss_fit,
            "success": fit_success,
            "params": params,
        },
    }


def probe_metrics(
        probe: np.ndarray,
        pixel_size: tuple,
        zoom_factor: int | str = "auto",
        probe_convention: str = "pynx",
        centre_at_max: bool = False,
        verbose: bool = False,
) -> tuple:
    """
    Plot the probe along with the line profile and its FWHM estimate.
    The probe is displayed in a 2D image, and the line profile is shown
    as a 1D plot. The FWHM is calculated and displayed on the line
    profile. If modes is True, all modes are plotted, otherwise
    only the first mode is plotted.
    > Notes: In the PyNX imshow plot of the probe, the origin is set to
    "lower" and exent is set to (-x_min, x_max, y_min, y_max). This
    means that in the PyNX convention, the probe is stored as a 2D
    (y = y_cxi, x = -x_cxi) array.

    Args:
        probe (np.ndarray): the probe data to be plotted in matrix
            (y, x) convention.
        pixel_size (tuple): the pixel size of the probe data (y, x)
        zoom_factor (int | str, optional): the zoom factor for the probe
            plot. If "auto", the window is set to be 3 times the FWHM.
            Defaults to "auto".
        probe_convention (str, optional): the convention used for the
            probe data. If "pynx", the probe is stored as a 2D (y =
            y_cxi, x = -x_cxi) array. Defaults to "pynx".
        centre_at_max (bool, optional): if True, the probe is centred
            before the analysis. Defaults to False.
        verbose (bool, optional): if True, prints additional
            information. Defaults to False.

    Returns:
        tuple: the figure and axes objects for the probe and line
        profile plots.
    """
    if probe.ndim != 2:
        raise ValueError("Probe must be 2D array (2D single-mode probe)")
    if probe_convention != "pynx":
        raise ValueError(
            "Only PyNX convention is supported for now. "
            "Use probe_convention='pynx'."
        )

    # probe amplitude or intensity
    probe_intensity = np.abs(probe) ** 2

    # Check if the probe is well centred
    com = center_of_mass(probe_intensity)
    if verbose:
        print(
            f"Probe intensity centre of mass: {com[0]:.2f} (y), "
            f"{com[1]:.2f} (x) pixels\n"
            f"Probe shape: {probe.shape}."
        )
    if centre_at_max:
        if verbose:
            print("Centring the probe at the maximum intensity.")
        probe = CroppingHandler.force_centred_cropping(
            probe, where="max"
        )
        # recompute the probe intensity in the new cropped frame
        probe_intensity = np.abs(probe) ** 2

    # Initialise the main dictionary containing all the line profile
    # metrics
    metrics = {"x": {"color": "dodgerblue"}, "y": {"color": "lightcoral"}}

    for i, axis in enumerate(["y", "x"]):
        metrics[axis]["axis_values"] = np.linspace(
            -pixel_size[i] * probe_intensity.shape[i] / 2,
            pixel_size[i] * probe_intensity.shape[i] / 2,
            probe_intensity.shape[i],
        )

    # the probe is stored as a (y = y_cxi, x = -x_cxi) array, so we need
    # to flip the x-axis extent.
    metrics["x"]["axis_values"] = np.flip(metrics["x"]["axis_values"])

    metrics["x"]["profile"] = probe_intensity[probe_intensity.shape[0] // 2, :]
    metrics["x"]["centre"] = metrics["x"]["axis_values"][
        probe_intensity.shape[1]
        // 2  # the pos that serves to get the y profile
    ]
    metrics["y"]["profile"] = probe_intensity[:, probe_intensity.shape[1] // 2]
    metrics["y"]["centre"] = metrics["y"]["axis_values"][
        probe_intensity.shape[0]
        // 2  # the pos that serves to get the x profile
    ]

    extent = (
        metrics["x"]["axis_values"][0],
        metrics["x"]["axis_values"][-1],
        metrics["y"]["axis_values"][0],
        metrics["y"]["axis_values"][-1],
    )

    # Compute FWHM and other metrics for x and y profiles
    for axis in ["x", "y"]:
        metrics[axis].update(
            get_width_metrics(
                metrics[axis]["profile"],
                metrics[axis]["axis_values"],
                verbose=verbose,
            )
        )
        metrics[axis]["peak_pos"] = metrics[axis]["axis_values"][
            metrics[axis]["highest_peak_idx"]
        ]

    # Calculate ROI boundaries (in physical units)
    if zoom_factor == "auto":  # 8x FWHM. /2 means each side
        zoom_extent = (
            8
            / 2
            * np.array(
                [
                    metrics["x"]["fwhm"]["value"],
                    -metrics["x"]["fwhm"]["value"],
                    -metrics["y"]["fwhm"]["value"],
                    metrics["y"]["fwhm"]["value"],
                ]
            )
        )
    elif zoom_factor == 1:
        zoom_extent = extent
    else:
        zoom_extent = np.array(extent) * 1 / zoom_factor

    metrics["x"]["min"] = metrics["x"]["centre"] + zoom_extent[1]  # inverted!
    metrics["x"]["max"] = metrics["x"]["centre"] + zoom_extent[0]  # inverted!
    metrics["y"]["min"] = metrics["y"]["centre"] + zoom_extent[2]
    metrics["y"]["max"] = metrics["y"]["centre"] + zoom_extent[3]

    fig = plt.figure(figsize=(6, 6), layout="tight")
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.4, 0.4, 0.2])
    axes = np.array(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ]
    )
    table_ax = fig.add_subplot(gs[2, :])

    # Show the probe intensity
    imshow_kwargs = {
        "extent": extent,
        "origin": "lower",
    }

    X, Y = np.meshgrid(
        metrics["x"]["axis_values"], metrics["y"]["axis_values"]
    )

    axes[0, 0].imshow(probe_intensity, cmap="viridis", **imshow_kwargs)
    add_colorbar(axes[0, 0])
    axes[0, 0].set_title(r"Probe Intensity ($|\mathcal{P}|^2$, a. u.)")

    opacity = probe_intensity / np.max(probe_intensity)
    axes[0, 1].imshow(
        np.angle(probe),
        alpha=opacity,
        cmap="cet_CET_C9s_r",
        **imshow_kwargs
    )
    add_colorbar(axes[0, 1], extend="both")

    axes[0, 1].set_title(r"Probe Phase ($\text{arg}(\mathcal{P})$, rad)")

    # # Add FWHM indicators as ellipse
    indicator_params = {"alpha": 0.5, "lw": 0.5, "linestyle": "--"}
    ellipse = Ellipse(
        (metrics["x"]["centre"], metrics["y"]["centre"]),
        width=metrics["x"]["fwhm"]["value"],
        height=metrics["y"]["fwhm"]["value"],
        edgecolor="w",
        facecolor="none",
        **indicator_params,
    )
    axes[0, 0].add_patch(ellipse)

    # Add crosshair at peak
    axes[0, 0].axvline(x=metrics["x"]["centre"], color="w", **indicator_params)
    axes[0, 0].axhline(y=metrics["y"]["centre"], color="w", **indicator_params)

    # Set limits to zoomed region
    for ax in axes[0, :]:
        ax.set_xlim(metrics["x"]["max"], metrics["x"]["min"])  # inverted!
        ax.set_ylim(metrics["y"]["min"], metrics["y"]["max"])
        # ax.set_ylim(xy_min, xy_max)
        ax.set_xlabel(r"$x_{\text{CXI}}$ (m)")
        ax.set_ylabel(r"$y_{\text{CXI}}$ (m)")

    for i, (subplot, axis) in enumerate(zip(axes[1, :], ["x", "y"])):
        subplot.plot(
            metrics[axis]["axis_values"],
            metrics[axis]["profile"],
            color=metrics[axis]["color"],
            label="line profile",
            lw=1,
        )
        subplot.axvspan(
            *metrics[axis]["fwhm"]["boundaries"],
            color=metrics[axis]["color"],
            alpha=0.4,
            label="FWHM width",
            lw=0,
        )
        subplot.axvspan(
            *metrics[axis]["fw10m"]["boundaries"],
            color=metrics[axis]["color"],
            alpha=0.1,
            label="FW10M width",
            lw=0,
        )

        # Gaussian fit
        if metrics[axis]["gaussian"]["success"]:
            subplot.plot(
                metrics[axis]["axis_values"],
                metrics[axis]["gaussian"]["fit"],
                color="k",
                label="gaussian fit",
                marker="o",
                markersize=2,
                markerfacecolor=metrics[axis]["color"],
                markeredgewidth=0.4,
                markeredgecolor="k",
                lw=0.25,
            )

        # other markers
        line_params = {
            "lw": 0.5,
            "linestyle": "--",
            "color": "k",
            "alpha": 0.5,
            "label": "peak position",
        }
        subplot.axvline(
            x=metrics[axis]["axis_values"][metrics[axis]["highest_peak_idx"]],
            **line_params,
        )

        # titles and labels
        formatted_axis = r"$" + axis + r"_{\text{CXI}}$"
        subplot.set_title(f"{formatted_axis} profile")
        subplot.set_xlabel(f"{formatted_axis} (m)")
        subplot.set_ylabel(r"$|\mathcal{P}|^2$ (a. u.)")
        subplot.legend(frameon=False, fontsize=6)

    # x plot, min and max are inverted
    axes[1, 0].set_xlim(metrics["x"]["max"], metrics["x"]["min"])
    axes[1, 1].set_xlim(metrics["y"]["min"], metrics["y"]["max"])

    # Add a table with the metrics
    table_ax.axis("off")
    metrics_table = [
        [
            "Metric",
            r"$x_{\text{CXI}}$ direction",
            r"$y_{\text{CXI}}$ direction",
        ],
        [
            "Absolute FWHM",
            f"{metrics['x']['fwhm']['value']:.2e} m",
            f"{metrics['y']['fwhm']['value']:.2e} m",
        ],
        [
            "FW10%M",
            f"{metrics['x']['fw10m']['value']:.2e} m",
            f"{metrics['y']['fw10m']['value']:.2e} m",
        ],
        [
            "Gaussian FWHM",
            f"{metrics['x']['gaussian']['value']:.2e} m",
            f"{metrics['y']['gaussian']['value']:.2e} m",
        ],
        [
            "Probe peak",
            f"{metrics['x']['peak_pos']:.2e} m",
            f"{metrics['y']['peak_pos']:.2e} m",
        ],
    ]
    table = table_ax.table(
        cellText=metrics_table, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for (i, _), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        if i == 0:
            cell.set_text_props(fontweight="bold")

    return fig, axes
