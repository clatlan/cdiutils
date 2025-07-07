import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass

from matplotlib.patches import Ellipse
import matplotlib.gridspec as gridspec

from cdiutils.utils import CroppingHandler, angular_spectrum_propagation
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

    # find peaks
    peaks, properties = find_peaks(profile, height=0.5 * np.max(profile))

    # use the highest peak or the middle if no peak is found
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

    # convert width indices to physical units using axis values
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
        # fit Gaussian to profile
        p0 = [
            np.max(profile),
            axis_values[highest_peak_idx],
            fwhm / 2.355,
            0,
        ]  # initial guess
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
    probe_amplitude = np.abs(probe)

    # check if the probe is well centred
    com = center_of_mass(probe_amplitude)
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
            probe, where="max", verbose=verbose
        )
        # recompute the probe intensity in the new cropped frame
        probe_amplitude = np.abs(probe)

    # initialise the main dictionary containing all the line profile
    # metrics
    metrics = {"x": {"color": "dodgerblue"}, "y": {"color": "lightcoral"}}

    for i, axis in enumerate(["y", "x"]):
        metrics[axis]["axis_values"] = np.linspace(
            -pixel_size[i] * probe.shape[i] / 2,
            pixel_size[i] * probe.shape[i] / 2,
            probe.shape[i],
        )

    # the probe is stored as a (y = y_cxi, x = -x_cxi) array, so we need
    # to flip the x-axis extent.
    metrics["x"]["axis_values"] = np.flip(metrics["x"]["axis_values"])

    metrics["x"]["profile"] = probe_amplitude[probe.shape[0] // 2, :]
    metrics["x"]["centre"] = metrics["x"]["axis_values"][
        probe.shape[1]
        // 2  # the pos that serves to get the y profile
    ]
    metrics["y"]["profile"] = probe_amplitude[:, probe.shape[1] // 2]
    metrics["y"]["centre"] = metrics["y"]["axis_values"][
        probe.shape[0]
        // 2  # the pos that serves to get the x profile
    ]

    extent = (
        metrics["x"]["axis_values"][0],
        metrics["x"]["axis_values"][-1],
        metrics["y"]["axis_values"][0],
        metrics["y"]["axis_values"][-1],
    )

    # compute FWHM and other metrics for x and y profiles
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

    # calculate ROI boundaries (in physical units)
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

    axes[0, 0].imshow(probe_amplitude, cmap="viridis", **imshow_kwargs)
    add_colorbar(axes[0, 0])
    axes[0, 0].set_title(r"Probe amplitude ($|\mathcal{P}|$, a. u.)")

    opacity = np.abs(probe) / np.max(np.abs(probe))
    axes[0, 1].imshow(
        np.angle(probe),
        alpha=opacity,
        cmap="cet_CET_C9s_r",
        **imshow_kwargs
    )
    axes[0, 1].set_facecolor("black")
    add_colorbar(axes[0, 1], extend="both")

    axes[0, 1].set_title(r"Probe phase ($\text{arg}(\mathcal{P})$, rad)")

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
        subplot.set_ylabel(r"$|\mathcal{P}|$ (a. u.)")
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


def probe_focus_sweep(
        probe: np.ndarray,
        pixel_size: tuple,
        wavelength: float,
        step_nb: int = 100,
        step_size: float = 10e-6
) -> np.ndarray:
    """
    Propagate the probe through a range of distances using the angular
    spectrum method. The function computes the propagated probe at each
    distance and returns the propagated probe and the corresponding
    propagation positions.

    Args:
        probe (np.ndarray): the probe to be propagated, in the shape
            (height, width) or (modes, height, width).
        pixel_size (tuple): the pixel size of the probe in meters.
        wavelength (float): the wavelength of the X-ray beam in meters.
        step_nb (int, optional): the number of steps for the propagation
            sweep. This defines the number of propagation positions.
            Defaults to 100.
        step_size (_type_, optional): the step size for the propagation
            sweep in meters. This defines the distance between each
            propagation position. Defaults to 10e-6 m.

    Returns:
        np.ndarray: the propagated probe at each propagation position,
            in the shape (step_nb, height, width) or (modes, step_nb,
            height, width).
        np.ndarray: the propagation positions in meters.
    """
    propagation_positions = step_size * np.arange(-step_nb // 2, step_nb // 2)
    progpated_probe = np.empty(
        propagation_positions.shape + probe.shape, dtype=np.complex64
    )
    for i, distance in enumerate(propagation_positions):
        progpated_probe[i] = angular_spectrum_propagation(
            probe,
            propagation_distance=distance,  # in meters
            wavelength=wavelength,
            pixel_size=pixel_size[0],
            do_fftshift=True,  # If true fftshift before and ifftshit after
            verbose=False
        )

    if probe.ndim == 3:  # multiple modes
        progpated_probe = progpated_probe.transpose(1, 0, 2, 3)

    return progpated_probe, propagation_positions


def plot_propagated_probe(
        propagated_probe: np.ndarray,
        pixel_size: tuple | float,
        propagation_step_size: float,
        convert_to_microns: bool = True,
        focal_distances: tuple | None = None,
        plot_phase: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the propagated probe as a 2D image with the phase or amplitude
    information. The function displays the probe at different
    propagation positions, with the option to plot the phase or
    amplitude. It also allows for the conversion of pixel size and
    propagation step size to microns. If focal distances are provided,
    vertical lines are drawn at those positions.

    Args:
        propagated_probe (np.ndarray): the propagated probe data, in the
            shape (step_nb, height, width) or (modes, step_nb, height,
            width). If 3D, the first dimension is considered as the
            propagation axis, and the rest are height and width. If 4D,
            will take the first mode only.
        pixel_size (tuple | float): the pixel size of the probe in
            meters.
        propagation_step_size (float, optional): the step size for the
            propagation sweep in meters. This defines the distance
            between each propagation position. If convert_to_microns is
            True, this value is converted to microns.
        convert_to_microns (bool, optional): if True, converts the pixel
            size and propagation step size to microns. This is useful for
            displaying the probe in microns instead of meters.
            Defaults to True.
        focal_distances (tuple | None, optional): the focal distances
            where vertical lines are drawn on the plot. If provided, it
            should be a tuple of two values representing the focal
            distances in meters for both directions. If
            convert_to_microns is True, these values are converted to
            microns. If None, no vertical lines are drawn. Defaults to
            None.
        plot_phase (bool, optional): if True, plots the phase of the
            propagated probe. If False, plots the amplitude. The opacity
            of the phase plot is set to the sum of the absolute values
            along the propagation axis, normalised to the maximum value.
            Defaults to True.

    Raises:
        ValueError: if the propagated_probe is not 3D or 4D, or if the
        pixel_size is not a float or a tuple of floats.

    Returns:
        tuple[plt.Figure, plt.Axes]: the figure and axes objects for the
        propagated probe plot.
    """
    if propagated_probe.ndim == 4:
        propagated_probe = propagated_probe[0]
    elif propagated_probe.ndim != 3:
        raise ValueError(
            "Expected propagated_probe to be 3 or 4D (propagation axis, "
            "height, width)."
        )

    if isinstance(pixel_size, float):
        pixel_size = (pixel_size, pixel_size, pixel_size)

    if convert_to_microns:
        pixel_size = tuple(p * 1e6 for p in pixel_size)
        propagation_step_size *= 1e6
    unit_label = "μm" if convert_to_microns else "m"

    if focal_distances is not None:
        if convert_to_microns:
            focal_distances = tuple(f * 1e6 for f in focal_distances)

    fig, axes = plt.subplots(2, 1, figsize=(5, 3), layout="tight", sharex=True)

    plot_params = {
        "cmap": "cet_CET_C9s_r" if plot_phase else "turbo",
        "alpha": None, "origin": "lower", "aspect": "auto",
    }

    for i, ax in enumerate(np.flip(axes).flat):
        if plot_phase:
            opacity = np.abs(propagated_probe).sum(axis=2-i).T
            opacity /= opacity.max()
            slices = [slice(None)] * propagated_probe.ndim
            slices[2-i] = propagated_probe.shape[2-i]//2
            to_plot = np.angle(propagated_probe[tuple(slices)]).T

            ax.set_facecolor("black")
            plot_params["alpha"] = opacity

        else:
            to_plot = np.abs(propagated_probe).sum(axis=2-i).T
        ax.imshow(
            to_plot,
            extent=(
                propagation_step_size * -propagated_probe.shape[0] / 2,
                propagation_step_size * propagated_probe.shape[0] / 2,
                -pixel_size[i] * propagated_probe.shape[i+1] / 2,
                pixel_size[i] * propagated_probe.shape[i+1] / 2
            ),
            **plot_params
        )

        ax.axvline(
            x=0,
            color="white",
            linestyle="-",
            linewidth=0.125,
        )

        if focal_distances is not None:
            label = (
                f"focal distance = {int(focal_distances[1-i])} {unit_label}"
            )
            ax.axvline(
                x=focal_distances[1-i],
                color="white",
                linestyle="--",
                label=label
            )

    axes[0].set_ylabel(r"$y_{\text{CXI}}$" + f", height ({unit_label})")
    axes[1].set_ylabel(r"$x_{\text{CXI}}$" + f", width ({unit_label})")
    axes[1].set_xlabel(
        r"$z_{\text{CXI}}$" + f", propagation distance ({unit_label})"
    )
    quantity = "phase" if plot_phase else "amplitude"
    fig.suptitle(f"Propagated probe ({quantity})")
    for ax in axes.flat:
        add_colorbar(ax, extend="both", size="2%")
        legend = ax.legend(frameon=False, fontsize=6)
        for text in legend.get_texts():
            text.set_color("white")

    return fig, axes


def get_focal_distances(
        propagated_probe: np.ndarray,
        propagation_positions: np.ndarray,
        method: str = "max"
) -> tuple[tuple[float, float], tuple[int, int]]:
    """
    Get the focal distances from the propagated probe data.
    The function computes the focal distances by reducing the probe
    data along the propagation axis using the specified method (sum or
    max). It returns the focal distances and the corresponding indexes
    in the propagation positions array.

    Args:
        propagated_probe (np.ndarray): the propagated probe data, in the
            shape (step_nb, height, width) or (modes, step_nb, height,
            width). If 3D, the first dimension is considered as the
            propagation axis, and the rest are height and width. If 4D,
            will take the first mode only.
        propagation_positions (np.ndarray): the propagation positions in
            meters.
        method (str, optional): the method to use for reducing the probe
            data. Can be "sum" or "max". If "sum", the function
            computes the sum of the absolute values along the
            propagation axis and then finds the maximum. Defaults to
            "max".

    Raises:
        ValueError: if the propagated_probe is not 3D or 4D.
        ValueError: if the method is not "sum" or "max".

    Returns:
        tuple[tuple[float, float], tuple[int, int]]: the focal distances
        as a tuple of two floats (focal_distance_1, focal_distance_2)
        and the corresponding indexes in the propagation positions array
        as a tuple of two integers (index_1, index_2).
    """
    if propagated_probe.ndim == 4:
        propagated_probe = propagated_probe[0]
    elif propagated_probe.ndim != 3:
        raise ValueError(
            "Expected propagated_probe to be 3 or 4D (propagation axis, "
            "height, width)."
        )
    if method == "sum":
        reducing_function = np.sum
    elif method == "max":
        reducing_function = np.max
    else:
        raise ValueError(f"Unknown method: {method}")

    focal_distances, indexes = [], []
    for i in range(2):
        indexes.append(
            np.argmax(
                reducing_function(
                    np.sum(np.abs(propagated_probe), axis=i+1),
                    axis=1
                ),
            )
        )
        focal_distances.append(propagation_positions[indexes[-1]])

    return tuple(focal_distances), indexes


def focus_probe(
        probe: np.ndarray,
        pixel_size: tuple,
        wavelength: float,
        step_nb: int = 100,
        step_size: float = 10e-6,
        plot: bool = True,
        **plot_kwargs
) -> tuple:
    """
    Complete analysis of probe focus characteristics by propagating the
    probe through a range of distances and computing the focal
    distances. The function performs a probe focus sweep, computes the
    focal distances using the specified method (max), and plots the
    propagated probe with the focal distances. It returns the focused
    probe and the focal distances.

    Args:
        probe (np.ndarray): the probe to be propagated, in the shape
            (height, width) or (modes, height, width). If 3D, the first
            dimension is considered as the propagation axis, and the rest
            are height and width. If 4D, will take the first mode only.
        pixel_size (tuple): the pixel size of the probe in meters, as a
            tuple (height, width) or a single float value for both
            dimensions.
        wavelength (float): the wavelength of the X-ray beam in meters.
        step_nb (int, optional): the number of steps for the propagation
            sweep. This defines the number of propagation positions.
            It determines how many times the probe is propagated through
            the range of distances. Defaults to 100.
        step_size (float, optional): the step size for the propagation
            sweep in meters. This defines the distance between each
            propagation position. Defaults to 10e-6.
        plot (bool, optional): whether to plot the propagated probe and
            focal distances. If True, the function will plot the
            propagated probe with the focal distances. If False, no plot
            is generated.Defaults to True.

    Returns:
        tuple: (focused_probe, focal_distances)
    """
    propagated_probe, propagation_positions = probe_focus_sweep(
        probe, pixel_size, wavelength, step_nb, step_size
    )

    focal_distances, indexes = get_focal_distances(
        propagated_probe, propagation_positions, method="max"
    )
    if propagated_probe.ndim == 4:
        focused_probe = propagated_probe[:, indexes[1], ...]
    elif propagated_probe.ndim == 3:
        focused_probe = propagated_probe[indexes[0]]

    if plot:
        plot_propagated_probe(
            propagated_probe,
            pixel_size,
            propagation_step_size=step_size,
            focal_distances=focal_distances
        )

    return focused_probe, focal_distances[1]
