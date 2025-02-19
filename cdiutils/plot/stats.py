import matplotlib.pyplot as plt
from matplotlib.typing import ColorType

import numpy as np
from scipy.ndimage import binary_erosion

from cdiutils.utils import kde_from_histogram
from cdiutils.plot.formatting import save_fig


def plot_histogram(
        ax: plt.Axes,
        counts: np.ndarray,
        bin_edges: np.ndarray,
        kde_x: np.ndarray = None,
        kde_y: np.ndarray = None,
        color: ColorType = "lightcoral",
        fwhm: bool = True
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

    Returns:
        float: the fwhm is required else None.
    """
    # Resample the histogram to calculate the kernel density estimate
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Plot the histogram bars
    ax.bar(
        bin_centres, counts, bin_width,
        color=color,
        alpha=0.4,
        edgecolor=color,
        linewidth=0.5,
        label="data histogram"
    )

    # Find the x-axis limits
    xmax = np.max(np.abs(bin_centres))
    xmin = -xmax
    ax.set_xlim(xmin, xmax)

    if kde_x is not None and kde_y is not None:
        # Plot the kernel density estimate
        ax.plot(kde_x, kde_y, color=color, label="Kernel density estimate")

        # Calculate the FWHM
        if fwhm:
            halfmax = kde_y.max() / 2
            maxpos = kde_y.argmax()
            leftpos = (np.abs(kde_y[:maxpos] - halfmax)).argmin()
            rightpos = (np.abs(kde_y[maxpos:] - halfmax)).argmin() + maxpos
            fwhm = kde_x[rightpos] - kde_x[leftpos]

            # Plot the FWHM line
            ax.axhline(
                y=halfmax,
                xmin=(kde_x[leftpos] - xmin) / (-2 * xmin),
                xmax=(kde_x[rightpos] + xmax) / (2 * xmax),
                label=f"FWHM = {fwhm:.4f}%",
                color=color, ls="--", linewidth=1
            )
        return fwhm
    return None


def strain_statistics(
        strain: np.ndarray,
        support: np.ndarray,
        bins: np.ndarray | int = 50,
        colors: dict = None,
        title: str = "",
        save: str = None
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a strain statistics graph displaying distribution of strain
    for the overall object, the bulk or the surface of the object.

    Args:
        strain (np.ndarray): the strain data.
        support (np.ndarray): the associated support.
        bins (np.ndarray | int, optional): the bins as accepted in
            numpy.histogram function. Defaults to 50.
        colors (dict, optional): the dictionary of colours.
            Defaults to None.
        title (str, optional): the title of the figure.
        save: (str, optional): the path where to save the figure.

    Returns:
        tuple[plt.Figure, plt.Axes]: the figure and axes.
    """
    support = np.nan_to_num(support)
    bulk = binary_erosion(support)
    surface = support - bulk

    sub_strains = {
        "overall": strain[support == 1].ravel(),
        "bulk": strain[bulk == 1].ravel(),
        "surface": strain[surface == 1].ravel(),
    }
    histograms = {
        k: np.histogram(v, bins=bins) for k, v in sub_strains.items()
    }
    histograms["bulk_density"] = np.histogram(
        sub_strains["bulk"], bins=bins, density=True
    )
    histograms["surface_density"] = np.histogram(
        sub_strains["surface"], bins=bins, density=True
    )
    means = {k: np.nanmean(v) for k, v in sub_strains.items()}
    means["bulk_density"] = means["bulk"]
    means["surface_density"] = means["surface"]

    kdes = {k: kde_from_histogram(*v) for k, v in histograms.items()}

    if colors is None:
        colors = {
            "overall": "lightcoral",
            "bulk": "orange",
            "bulk_density": "orange",
            "surface": "dodgerblue",
            "surface_density": "dodgerblue",
        }
    mosaic = """ABC
    DDD"""
    figure, axes = plt.subplot_mosaic(
        mosaic, layout="tight", figsize=(6, 4)
    )
    fwhms = {}
    # First plot the three histograms separately
    for e, key in zip(("overall", "bulk", "surface"), ("A", "B", "C")):
        fwhms[e] = plot_histogram(
            axes[key], *histograms[e], *kdes[e], color=colors[e]
        )

        # Plot the mean
        axes[key].plot(
            means[e], 0, color=colors[e], ms=4,
            markeredgecolor="k", marker="o", mew=0.5,
            label=f"Mean = {means[e]:.4f} %"
        )

    # Plot the density histograms for bulk and surface on the same subplot
    for e in ("bulk_density", "surface_density"):
        fwhms[e] = plot_histogram(
            axes["D"], *histograms[e], *kdes[e], color=colors[e]
        )

        axes["D"].plot(
            means[e], 0, color=colors[e], ms=4,
            markeredgecolor="k", marker="o", mew=0.5,
            label=f"Mean = {means[e]:.4f} %"
        )

    for key in (("A", "B", "C")):
        axes[key].set_ylabel(r"Counts")
        handles, labels = axes[key].get_legend_handles_labels()
        handles = handles[1:-1]
        labels = labels[1:-1]
        axes[key].legend(
            handles, labels,
            frameon=False,
            loc="upper center",  bbox_to_anchor=(0.25, 0.8, 0.5, 0.5),
            fontsize=6, markerscale=0.7,
            ncols=1
        )

    axes["D"].set_ylabel(r"Normalised counts")
    handles, labels = axes["D"].get_legend_handles_labels()
    handles = [handles[i] for i in [1, 2, 4, 5]]
    labels = [labels[i] for i in [1, 2, 4, 5]]
    # Shrink current axis by 20%
    box = axes["D"].get_position()
    axes["D"].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axes["D"].legend(
        handles, labels,
        frameon=False,
        loc="center left", bbox_to_anchor=(1, 0.5),
        # loc="right", bbox_to_anchor=(1.5, 0.25, 0.5, 0.5),
        fontsize=6, markerscale=0.7
    )
    axes["D"].set_title("Density distributions")

    axes["A"].set_xlabel(
        r"Overall strain, $\varepsilon$ (%)", color=colors["overall"]
    )
    axes["B"].set_xlabel(
        r"Bulk strain, $\varepsilon_{\text{bulk}}$ (%)",
        color=colors["bulk"]
    )
    axes["C"].set_xlabel(
        r"Surface strain, $\varepsilon_{\text{surface}}$ (%)",
        color=colors["surface"]
    )
    axes["D"].set_xlabel(
        r"$\varepsilon_{\text{bulk}}$, $\varepsilon_{\text{surface}}$ (%)"
    )

    figure.suptitle(title)
    if save:
        save_fig(figure, path=save, transparent=False)

    return figure, axes, means, fwhms
