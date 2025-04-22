import matplotlib.pyplot as plt
from matplotlib.typing import ColorType

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.stats import gaussian_kde
from seaborn import kdeplot

from cdiutils.utils import normalise
from cdiutils.plot.formatting import save_fig


def kde_from_histogram(
        counts: np.ndarray,
        bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Kernel Density Estimate (KDE) from histogram counts and
    bin edges provided by numpy.histogram function.

    Args:
        counts (np.ndarray): the number of elements in each bin.
        bin_edges (np.ndarray): the limits of each bin.

    Returns:
        tuple[np.ndarray, np.ndarray]: x values used to compute the KDE
            estimate, the y value (KDE estimate)
    """
    # Check if the histogram is density or not by checking the sum of
    # the counts
    bin_widths = np.diff(bin_edges)
    is_density = np.isclose(np.sum(counts * bin_widths), 1.0)

    if is_density:
        # When density=True, use the bin edges to reconstruct the data
        # for KDE
        data = []
        for count, left_edge, right_edge in zip(
                counts, bin_edges[:-1], bin_edges[1:]
        ):
            data.extend(
                np.random.uniform(
                    left_edge,
                    right_edge,
                    int(count * len(counts) * (right_edge - left_edge))
                )
            )
        data = np.array(data)

        kde = gaussian_kde(data)
        x = np.linspace(min(bin_edges), max(bin_edges))
        y = kde(x)

    else:
        # Reconstruct the data from histogram counts and bin edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        reconstructed_data = np.repeat(bin_centers, counts)

        # Calculate KDE using the reconstructed data
        kde = gaussian_kde(reconstructed_data)
        # Evaluate KDE
        x = np.linspace(bin_edges.min(), bin_edges.max())
        y = kde.pdf(x)

        # Scale the KDE values to match the original counts
        y *= len(reconstructed_data) * bin_width

    return x, y


def find_isosurface(
        amplitude: np.ndarray,
        nbins: int = 100,
        sigma_criterion: float = 3,
        plot: bool = False,
        show: bool = False,
        save: str = None
) -> tuple[float, plt.Axes] | float:
    """
    Estimate the isosurface value from the amplitude distribution.

    This function computes the isosurface value based on the amplitude
    distribution of a 3D volume. The isosurface is calculated as:
    `mu - sigma_criterion * sigma`, where `mu` is the mean and `sigma`
    is the standard deviation of the distribution.

    Args:
        amplitude (np.ndarray): The 3D amplitude volume.
        nbins (int, optional): The number of bins to use for the histogram.
            Defaults to 100.
        sigma_criterion (float, optional): The factor used to compute
            the isosurface. Defaults to 3.
        plot (bool, optional): Whether to generate a plot of the
            histogram and density estimate. Defaults to False.
        show (bool, optional): Whether to display the plot. Defaults to
            False.
        save (str, optional): File path to save the plot if generated.
            Defaults to None.

    Returns:
        tuple[float, plt.Axes] | float: The isosurface value. If `plot`
        or `show` is True, also returns the matplotlib figure object.
    """
    # normalise and flatten the amplitude
    flattened_amplitude = normalise(amplitude).ravel()

    counts, bins = np.histogram(flattened_amplitude, bins=nbins)

    # remove the background
    background_value = bins[
        np.where(counts == counts.max())[0] + 1 + nbins//20
    ]
    filtered_amplitude = flattened_amplitude[
        flattened_amplitude > background_value
    ]

    # redo the histogram with the filtered amplitude
    counts, bins = np.histogram(filtered_amplitude, bins=nbins, density=True)
    bin_centres = (bins[:-1] + bins[1:]) / 2
    bin_size = bin_centres[1] - bin_centres[0]

    # fit the amplitude distribution
    kernel = gaussian_kde(filtered_amplitude)
    x = np.linspace(0, 1, 1000)
    fitted_counts = kernel(x)

    max_index = np.argmax(fitted_counts)
    right_gaussian_part = np.where(x >= x[max_index], fitted_counts, 0)

    # find the closest indexes
    right_HM_index = np.argmin(
        np.abs(right_gaussian_part - fitted_counts.max() / 2)
    )
    left_HM_index = max_index - (right_HM_index - max_index)

    fwhm = x[right_HM_index] - x[left_HM_index]
    sigma_estimate = fwhm / 2*np.sqrt(2*np.log(2))
    isosurface = x[max_index] - sigma_criterion * sigma_estimate

    if plot or show:
        figsize = (6, 4)  # (5.812, 3.592)  # golden ratio
        fig, ax = plt.subplots(1, 1, layout="tight", figsize=figsize)
        ax.bar(
            bin_centres,
            counts,
            width=bin_size,
            color="dodgerblue",
            alpha=0.9,
            edgecolor=(0, 0, 0, 0.25),
            label=r"amplitude distribution"
        )
        kdeplot(
            filtered_amplitude,
            ax=ax,
            alpha=0.3,
            fill=True,
            color="navy",
            label=r"density estimate"
        )
        ax.axvspan(
            x[left_HM_index],
            x[right_HM_index],
            edgecolor="k",
            facecolor="green",
            alpha=0.2,
            label="FWHM"
        )
        ax.plot(
            [isosurface, isosurface],
            [0, fitted_counts[(np.abs(x - isosurface)).argmin()]],
            solid_capstyle="round",
            color="lightcoral",
            lw=5,
            label=fr"isosurface estimated at {isosurface:0.3f}"
        )

        ax.set_xlabel(r"normalised amplitude")
        ax.set_ylabel("counts")
        ax.legend(frameon=False)
        fig.suptitle(r"Reconstructed amplitude distribution")
        fig.tight_layout()
        if save is not None:
            save_fig(fig, save, transparent=False)
        if show:
            plt.show()
        return float(isosurface), fig
    return float(isosurface)


def get_histogram(
        data: np.ndarray,
        support: np.ndarray = None,
        bins: int = 50,
        density: bool = False,
        region: str = "overall"
) -> dict:
    """
    Calculate histogram and optionally kernel density estimate (KDE) of
    the data.
    Optionally applies a support mask to the data before and calculates
    the surface and bulk histograms.
    Args:
        data (np.ndarray): the data to be analysed
        support (np.ndarray): the support mask to be applied to the data
            before histogram calculation. If None, no mask is applied.
            Defaults to None.
        bins (int, optional): number of bins for the histogram.
            Defaults to 50.
        density (bool, optional): whether to normalise the histogram
            to form a probability density function. Defaults to False.
        region (str, optional): region of the data to be analysed. Can
            be "overall", "surface", "bulk" or "all". Defaults to
            "overall".

    Returns:
        dict: a dictionary containing the histograms for the specified
            region(s). If kde is True, also includes the KDEs.
    """
    if support is None and region != "overall":
        raise ValueError(
            "Support mask is required for surface or bulk region analysis."
        )
    if region not in ["overall", "surface", "bulk", "all"]:
        raise ValueError(
            "Invalid region specified. Choose from 'overall', "
            "'surface', 'bulk', or 'all'."
        )
    histograms = {}
    means = {}
    stds = {}

    if support is not None:
        overall_data = data[support > 0]

    # to handle any remaining NaN values, we need to specify the range
    histograms["overall"] = np.histogram(
        overall_data, bins=bins, density=density,
        range=(np.nanmin(overall_data), np.nanmax(overall_data))
    )
    means["overall"] = np.nanmean(overall_data)
    stds["overall"] = np.nanstd(overall_data)

    if region != "overall":
        bulk = binary_erosion(support)
        bulk_data = data[bulk > 0]

        if region == "bulk" or region == "all":
            histograms["bulk"] = np.histogram(
                bulk_data, bins=bins, density=density,
                range=(np.nanmin(bulk_data), np.nanmax(bulk_data))
            )
            means["bulk"] = np.nanmean(bulk_data)
            stds["bulk"] = np.nanstd(bulk_data)

        if region == "surface" or region == "all":
            surface = support - bulk
            surface_data = data[surface > 0]
            histograms["surface"] = np.histogram(
                surface_data, bins=bins, density=density,
                range=(np.nanmin(surface_data), np.nanmax(surface_data))
            )
            means["surface"] = np.nanmean(surface_data)
            stds["surface"] = np.nanstd(surface_data)

    kdes = {k: kde_from_histogram(*v) for k, v in histograms.items()}

    return histograms, kdes, means, stds


def plot_histogram(
        ax: plt.Axes,
        counts: np.ndarray,
        bin_edges: np.ndarray,
        kde_x: np.ndarray = None,
        kde_y: np.ndarray = None,
        color: ColorType = "lightcoral",
        fwhm: bool = True,
        bar_args: dict = None,
        kde_args: dict = None
) -> float:
    """
    Plot the bars of a histogram as well as the kernel density
    estimate.

    Args:
        ax (plt.Axes): the matplotlib ax to plot the histogram on.
        counts (np.ndarray): the count in each bin from
            np.histogram().
        bin_edges (np.ndarray): the bin edge values from
            np.histogram().
        kde_x (np.ndarray, optional): the x values used to
            calculate the kernel density estimate values.
        kde_y (np.ndarray, optional): the (y) values of the kernel
            density estimate.
        color (ColorType, optional): the colour of the bar and line.
            Defaults to "lightcoral".
        fwhm (bool, optional): whether to calculate and plot the full
            width at half maximum (FWHM) of the kernel density estimate.
            Defaults to True.
        bar_args (dict, optional): additional arguments for the
            matptlotlib bar function.
        kde_args (dict, optional): additional arguments for the
            matplotlib fill_between function. Can include boolean "fill"
            and float "fill_alpha" to control whether to fill the kde
            area and the alpha value of the fill. Defaults to None.

    Returns:
        float: the fwhm if required else None.
    """
    _bar_args = {
        "color": color,
        "alpha": 0.4,
        "edgecolor": color,
        "linewidth": 0.5,
        "label": ""
    }
    _bar_args.update(bar_args or {})

    _kde_args = {
        "color": color,
        "label": "Kernel density estimate"
    }
    fill_kde, fill_alpha = False, False
    if kde_args is not None:
        if "fill" in kde_args:
            fill_kde = kde_args.pop("fill")
        if "fill_alpha" in kde_args:
            fill_alpha = kde_args.pop("fill_alpha")
        _kde_args.update(kde_args)

    # Resample the histogram to calculate the kernel density estimate
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Plot the histogram bars
    ax.bar(bin_centres, counts, bin_width, **_bar_args)

    # Find the x-axis limits
    xmax = np.max(np.abs(bin_centres))
    xmin = -xmax
    ax.set_xlim(xmin, xmax)

    if kde_x is not None and kde_y is not None:
        # Plot the kernel density estimate
        ax.plot(kde_x, kde_y, **_kde_args)

        if fill_kde:
            ax.fill_between(
                kde_x, kde_y, 0,
                color=color, alpha=fill_alpha
            )

        # Calculate the FWHM
        if fwhm:
            halfmax = kde_y.max() / 2
            maxpos = kde_y.argmax()
            leftpos = (np.abs(kde_y[:maxpos] - halfmax)).argmin()
            rightpos = (np.abs(kde_y[maxpos:] - halfmax)).argmin() + maxpos
            fwhm_value = kde_x[rightpos] - kde_x[leftpos]

            fwhm_line, = ax.plot(
                [], [],
                label=f"FWHM = {fwhm_value:.4f}%",
                color=color, ls="--", linewidth=1
            )

            def update_fwhm_line(event_ax):
                xmin, xmax = event_ax.get_xlim()
                fwhm_line.set_data(
                    [kde_x[leftpos], kde_x[rightpos]], [halfmax, halfmax]
                )
                fwhm_line.set_transform(event_ax.transData)

            update_fwhm_line(ax)
            ax.callbacks.connect('xlim_changed', update_fwhm_line)
            ax.callbacks.connect('ylim_changed', update_fwhm_line)

            return fwhm_value
    return None
