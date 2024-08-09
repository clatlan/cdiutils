import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
import numpy as np
from scipy.ndimage import binary_erosion

from cdiutils.utils import kde_from_histogram


class PipelinePlotter:
    @staticmethod
    def plot_histogram(
            ax: plt.Axes,
            counts: np.ndarray,
            bin_edges: np.ndarray,
            kde_x: np.ndarray = None,
            kde_y: np.ndarray = None,
            color: ColorType = "lightcoral",
            fwhm: bool = True
    ) -> None:
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

    @classmethod
    def strain_statistics(
            cls,
            strain: np.ndarray,
            support: np.ndarray,
            bins: np.ndarray | int = 50,
            colors: dict = None
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
        figure, axes = plt.subplots(1, 4, layout="tight", figsize=(8, 2))

        # First plot the three histograms separately
        for k, ax in zip(("overall", "bulk", "surface"), axes.flat[:-1]):
            cls.plot_histogram(ax, *histograms[k], *kdes[k], color=colors[k])

            # Plot the mean
            ax.plot(
                means[k], 0, color=colors[k], ms=4,
                markeredgecolor="k", marker="o", mew=0.5,
                label=f"Mean = {means[k]:.4f} %"
            )

        # Plot the density histograms for bulk and surface on the same subplot
        for k in ("bulk_density", "surface_density"):
            cls.plot_histogram(
                axes[3], *histograms[k], *kdes[k], color=colors[k]
            )

            axes[3].plot(
                means[k], 0, color=colors[k], ms=4,
                markeredgecolor="k", marker="o", mew=0.5,
                label=f"Mean = {means[k]:.4f} %"
            )

        for ax in axes.flat[:-1]:
            ax.set_ylabel(r"Counts")
            handles, labels = ax.get_legend_handles_labels()
            handles = handles[1:-1]
            labels = labels[1:-1]
            ax.legend(
                handles, labels,
                frameon=False,
                loc="upper center",  bbox_to_anchor=(0.25, 0.75, 0.5, 0.5),
                fontsize=6, markerscale=0.7,
                ncols=1
            )

        axes[3].set_ylabel(r"Normalised counts")
        handles, labels = axes[3].get_legend_handles_labels()
        handles = [handles[i] for i in [1, 2, 4, 5]]
        labels = [labels[i] for i in [1, 2, 4, 5]]
        axes[3].legend(
            handles, labels,
            frameon=False,
            loc="right", bbox_to_anchor=(1.5, 0.25, 0.5, 0.5),
            fontsize=6, markerscale=0.7
        )
        axes[3].set_title("Density distributions")

        axes[0].set_xlabel(
            r"Overall strain, $\varepsilon$ (%)", color=colors["overall"]
        )
        axes[1].set_xlabel(
            r"Bulk strain, $\varepsilon_{\text{bulk}}$ (%)",
            color=colors["bulk"]
        )
        axes[2].set_xlabel(
            r"Surface strain, $\varepsilon_{\text{surface}}$ (%)",
            color=colors["surface"]
        )
        axes[3].set_xlabel(
            r"$\varepsilon_{\text{bulk}}$, $\varepsilon_{\text{surface}}$ (%)"
        )

        return figure, axes
