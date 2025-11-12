import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from cdiutils.analysis.stats import get_histogram, plot_histogram
from cdiutils.plot.formatting import (
    add_colorbar,
    add_labels,
    save_fig,
    set_plot_configs,
    white_interior_ticks_labels,
)
from cdiutils.plot.slice import plot_contour, plot_volume_slices
from cdiutils.utils import num_to_nan, symmetric_pad


class PipelinePlotter:
    """A class to provide key plotting methods used in (Bcdi)Pipeline."""

    @classmethod
    def detector_data(
        cls,
        det_data: np.ndarray,
        voxels: dict = None,
        full_det_data: np.ndarray = None,
        integrate: bool = False,
        title: str = "",
        save: str = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        def sub_get(voxels, k1, k2):
            """
            Handle the voxels dictionary. Check whenever a value
            of a sub-dictionary is provided and return it.
            """
            if voxels is None:
                return None
            if k1 not in voxels:
                return None
            sub = voxels.get(k1)
            if sub is None:
                return None
            return sub.get(k2)

        norm = LogNorm()

        plot_params = {
            "norm": norm,
            "slice_shift": (0, 0, 0),
            "show": False,
            "views": ("z-", "y+", "x+"),  # natural views,
            "integrate": integrate,
        }
        vox = sub_get(voxels, "cropped", "ref")
        if vox is not None:
            plot_params["slice_shift"] = [
                p - s // 2 for s, p in zip(det_data.shape, vox)
            ]
        _, axes1 = plot_volume_slices(det_data, **plot_params)

        if full_det_data is not None:
            plot_params["slice_shift"] = (0, 0, 0)
            vox = sub_get(voxels, "full", "ref")
            if vox is not None:
                plot_params["slice_shift"] = [
                    p - s // 2 for s, p in zip(full_det_data.shape, vox)
                ]
            _, axes2 = plot_volume_slices(full_det_data, **plot_params)
            fig, axes = plt.subplots(2, 3, layout="tight", figsize=(6, 4))
            old_axes = [axes1, axes2]
            for i, frame in enumerate(("cropped", "full")):
                for new_ax, old_ax in zip(axes[i].flat, old_axes[i].flat):
                    im = old_ax.get_images()[0]
                    new_ax.imshow(
                        im.get_array(),
                        cmap=im.get_cmap(),
                        norm=norm,
                    )

                for ax, p in zip(axes[i].flat, ((2, 1), (2, 0), (0, 1))):
                    cls._plot_markers(
                        ax, *[voxels[frame]["ref"][i] for i in p]
                    )
                    for style in ("max", "com"):
                        if voxels[frame][style] is not None:
                            cls._plot_markers(
                                ax,
                                *[voxels[frame][style][i] for i in p],
                                style=style,
                            )

            axes[0, 1].set_title("Cropped detector data")
            axes[1, 1].set_title("Raw detector data")
            for i in range(2):
                axes[i, 0].set_xlabel(r"axis$_{2}$, det. horiz.")
                axes[i, 0].set_ylabel(r"axis$_{1}$, det. vert.")

                axes[i, 1].set_xlabel(r"axis$_{2}$, det. horiz.")
                axes[i, 1].set_ylabel(r"axis$_{0}$, rocking curve")

                axes[i, 2].set_xlabel(r"axis$_{0}$, rocking curve")
                axes[i, 2].set_ylabel(r"axis$_{1}$, det. vert.")
            axes[1, 1].legend(
                loc="center",
                ncol=2,
                frameon=False,
                bbox_to_anchor=(0.5, 0.5),
                bbox_transform=fig.transFigure,
            )

        else:
            fig, axes = plt.subplots(1, 3, layout="tight", figsize=(6, 2))
            for new_ax, ax in zip(axes.flat, axes1.flat):
                im = ax.get_images()[0]
                new_ax.imshow(im.get_array(), cmap=im.get_cmap(), norm=norm)

            for ax, p in zip(axes.flat, ((2, 1), (2, 0), (0, 1))):
                cls._plot_markers(
                    ax, *[voxels["cropped"]["ref"][i] for i in p]
                )
                vox = sub_get(voxels, "cropped", "max")
                if vox is not None:
                    cls._plot_markers(ax, *[vox[i] for i in p], style="max")
                vox = sub_get(voxels, "cropped", "com")
                if vox is not None:
                    cls._plot_markers(ax, *[vox[i] for i in p], style="com")

            axes[0].set_xlabel(r"axis$_{2}$, det. horiz.")
            axes[0].set_ylabel(r"axis$_{1}$, det. vert.")

            axes[1].set_xlabel(r"axis$_{2}$, det. horiz.")
            axes[1].set_ylabel(r"axis$_{0}$, rocking curve")

            axes[2].set_xlabel(r"axis$_{0}$, rocking curve")
            axes[2].set_ylabel(r"axis$_{1}$, det. vert.")
            axes[1].legend(
                loc="upper center",
                ncol=2,
                frameon=False,
                bbox_to_anchor=(0.5, 1.3),
            )
        for ax in axes.flat:
            add_colorbar(ax)
        fig.suptitle(title)

        if save is not None:
            save_fig(fig, save, transparent=False)
        return fig, axes

    @staticmethod
    def _plot_markers(
        ax: plt.Axes, x: int, y: int, style: str = "ref"
    ) -> None:
        if style.lower() not in ("ref", "max", "com"):
            raise ValueError("style must be in ('ref', 'max', 'com').")
        if style == "ref":
            shape = ax.get_images()[0].get_array().shape
            ax.plot(
                np.repeat(x, 2),
                y + np.array([-0.1, 0.1]) * shape[0],
                color="w",
                lw=0.5,
            )
            ax.plot(
                x + np.array([-0.1, 0.1]) * shape[1],
                np.repeat(y, 2),
                color="w",
                lw=0.5,
            )
        else:
            plot_params = {
                "marker": "x",
                "markersize": 4,
                "linestyle": "None",
                "color": "green",
                "label": "com",
            }

            if style == "max":
                plot_params["color"] = "red"
                plot_params["label"] = "max"
            ax.plot(x, y, **plot_params)

    @staticmethod
    def ortho_detector_data(
        det_data: np.ndarray,
        ortho_data: np.ndarray,
        q_grid: np.ndarray,
        title: str = "",
        save: str = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        q_spacing = [q[1] - q[0] for q in q_grid]
        q_centre = (q_grid[0].mean(), q_grid[1].mean(), q_grid[2].mean())

        _, raw_axes = plot_volume_slices(
            det_data,
            views=("z-", "y+", "x+"),  # natural views,
            norm=LogNorm(),
            show=False,
        )
        _, ortho_axes = plot_volume_slices(
            ortho_data,
            voxel_size=q_spacing,
            data_centre=q_centre,
            norm=LogNorm(),
            convention="xu",
            show=False,
        )
        fig, axes = plt.subplots(2, 3, layout="tight", figsize=(6, 4))
        for new_axes, old_axes in zip(axes, (raw_axes, ortho_axes)):
            for new_ax, old_ax in zip(new_axes.flat, old_axes.flat):
                # replot with same configuration
                im = old_ax.get_images()[0]
                new_ax.imshow(
                    im.get_array(),
                    cmap=im.get_cmap(),
                    norm=LogNorm(),
                    extent=im.get_extent(),
                    origin=im.origin,
                )
                new_ax.axis(old_ax.axis())
        axes[0, 0].set_xlabel(r"axis$_{2}$, det. horiz.")
        axes[0, 0].set_ylabel(r"axis$_{1}$, det. vert.")

        axes[0, 1].set_xlabel(r"axis$_{2}$, det. horiz.")
        axes[0, 1].set_ylabel(r"axis$_{0}$, rocking curve")

        axes[0, 2].set_xlabel(r"axis$_{0}$, rocking curve")
        axes[0, 2].set_ylabel(r"axis$_{1}$, det. vert.")
        add_labels(axes[1], space="rcp", convention="xu")

        axes[0, 1].set_title("Raw data in detector frame")
        axes[1, 1].set_title("Orthogonalised data in reciprocal lab frame")

        fig.suptitle(title)
        if save is not None:
            save_fig(fig, save, transparent=False)
        return fig, axes

    @staticmethod
    def summary_plot(
        title: str = None,
        support: np.ndarray = None,
        table_info: dict = None,
        voxel_size: tuple = None,
        save: str = None,
        unique_vmin: float = None,
        unique_vmax: float = None,
        cmap: str = None,
        figsize: tuple = (6, 4),
        convention: str = "cxi",
        **to_plot,
    ) -> tuple[plt.Figure, plt.Axes]:
        _, _, PLOT_CONFIGS = set_plot_configs()

        fig, axes = plt.subplots(3, len(to_plot), figsize=figsize)
        if convention.lower() == "cxi":
            slice_names = (
                r"(xy)$_\mathrm{CXI}$ slice",
                r"(xz)$_\mathrm{CXI}$ slice",
                r"(zy)$_\mathrm{CXI}$ slice",
            )
        else:
            slice_names = (
                r"(yz)$_\mathrm{XU}$ slice",
                r"(xz)$_\mathrm{XU}$ slice",
                r"(xy)$_\mathrm{XU}$ slice",
            )
        for i in range(3):
            axes[i, 0].annotate(
                slice_names[i],
                xy=(0.2, 0.5),
                xytext=(-axes[i, 0].yaxis.labelpad - 2, 0),
                xycoords=axes[i, 0].yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
            )
        mappables = {}
        support = num_to_nan(support)
        for i, (key, array) in enumerate(to_plot.items()):
            if support is not None and key != "amplitude":
                array = support * array

            if key in PLOT_CONFIGS:
                cmap = PLOT_CONFIGS[key]["cmap"]
                # check if vmin and vmax are given | not
                if unique_vmin is None or unique_vmax is None:
                    if support is not None:
                        if key in ("dspacing", "lattice_parameter"):
                            vmin = np.nanmin(array)
                            vmax = np.nanmax(array)
                        elif key == "amplitude":
                            vmin = 0
                            vmax = np.nanmax(array)
                        else:
                            vmax = np.nanmax(np.abs(array))
                            vmin = -vmax
                    else:
                        vmin = PLOT_CONFIGS[key]["vmin"]
                        vmax = PLOT_CONFIGS[key]["vmax"]
                else:
                    vmin = unique_vmin
                    vmax = unique_vmax
            else:
                vmin = unique_vmin
                vmax = unique_vmax
                cmap = cmap if cmap else "turbo"
            if key == "amplitude":
                cmap = plt.get_cmap(cmap)
                cmap.set_bad("#30123bff")
            shape = array.shape

            norm = Normalize(vmin, vmax)
            plot_params = {
                "norm": norm,
                "data_centre": (0, 0, 0),
                "support": support,
                "show": False,
                "cmap": cmap,
                "voxel_size": voxel_size,
                "convention": convention,
            }
            _, helper_axes = plot_volume_slices(array, **plot_params)
            for ax, helper_ax in zip(axes[:, i].flat, helper_axes.flat):
                im = helper_ax.get_images()[0]
                ax.imshow(
                    im.get_array(),
                    cmap=im.get_cmap(),
                    norm=norm,
                    extent=im.get_extent(),
                    origin=im.origin,
                )
            mappables[key] = axes[2, i].get_images()[0]

            if key == "amplitude":
                if convention.lower() == "cxi":
                    plot_contour(
                        axes[0, i],
                        support[shape[0] // 2],
                        color="k",
                        pixel_size=(voxel_size[1], voxel_size[2]),
                        data_centre=(0, 0),
                    )
                    plot_contour(
                        axes[1, i],
                        support[:, shape[1] // 2, :],
                        color="k",
                        pixel_size=(voxel_size[0], voxel_size[2]),
                        data_centre=(0, 0),
                    )
                    plot_contour(
                        axes[2, i],
                        np.swapaxes(support[..., shape[2] // 2], 0, 1),
                        color="k",
                        pixel_size=(voxel_size[1], voxel_size[0]),
                        data_centre=(0, 0),
                    )
                else:
                    # TODO: XU case, to be implemented.
                    pass

        if table_info:
            table_ax = fig.add_axes([0.25, -0.05, 0.5, 0.15])
            table_ax.axis("tight")
            table_ax.axis("off")
            for k in table_info:
                table_info[k] = round(table_info[k], 4)
            cell_text = [
                [
                    np.array2string(
                        np.array(voxel_size),
                        formatter={
                            "float_kind": lambda x: f"{x:.2f}",
                        },
                    )
                ]
            ] + [[v] for v in table_info.values()]
            n_cols = len(list(table_info.keys())) + 1
            table = table_ax.table(
                cellText=np.transpose(cell_text),
                colLabels=["Voxel size (nm)"] + list(table_info.keys()),
                colWidths=[1 / n_cols] * n_cols,
                loc="center",
                cellLoc="center",
            )
            table.scale(1.5, 1.5)
            table.set_fontsize(10)

        fig.subplots_adjust(hspace=0.01, wspace=0.02)

        for i, key in enumerate(to_plot.keys()):
            l, _, w, _ = axes[0, i].get_position().bounds
            cax = fig.add_axes([l + 0.01, 0.93, w - 0.02, 0.02])
            try:
                cax.set_title(PLOT_CONFIGS[key]["title"])
            except KeyError:
                cax.set_title(key)
            fig.colorbar(
                mappables[key],
                cax=cax,
                extend="both",
                orientation="horizontal",
            )
            cax.tick_params(axis="x", which="major", pad=1)

        fig.canvas.draw()
        for i, ax in enumerate(axes.ravel()):
            ax.set_aspect("equal")
            if (
                i % len(to_plot) == 0
                and list(to_plot.keys())[i % len(to_plot)] == "amplitude"
            ):
                ax.locator_params(nbins=5)
                white_interior_ticks_labels(ax, -10, -5)

            else:
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])

        fig.suptitle(title, y=1.050)

        # save the figure
        if save:
            save_fig(fig, save, transparent=False)
        return fig

    @staticmethod
    def plot_final_object_fft(
        obj: np.ndarray,
        voxel_size: tuple,
        q_space_shift: tuple,
        exp_ortho_data: np.ndarray,
        exp_data_q_grid: np.ndarray,
        title: str = None,
        save: str = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        # Prepare the object fft
        shape = exp_ortho_data.shape
        q_voxel_size = 2 * np.pi / (10 * np.multiply(voxel_size, shape))
        obj_q_grid = [
            np.arange(-shape[i] // 2, shape[i] // 2) * q_voxel_size[i]
            for i in range(3)
        ]

        obj = symmetric_pad(obj, output_shape=shape)
        obj_fft = np.abs(np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(obj))))
        obj_fft **= 2  # We want to plot the intensity

        q_spacing = [q[1] - q[0] for q in obj_q_grid]
        q_centre = tuple(obj_q_grid[i].mean() for i in range(3))

        _, object_fft_axes = plot_volume_slices(
            obj_fft,
            voxel_size=q_spacing,
            data_centre=q_space_shift,
            norm=LogNorm(),
            convention="xu",
            show=False,
        )

        q_spacing = [q[1] - q[0] for q in exp_data_q_grid]
        q_centre = tuple(exp_data_q_grid[i].mean() for i in range(3))

        _, exp_axes = plot_volume_slices(
            exp_ortho_data,
            voxel_size=q_spacing,
            data_centre=q_centre,
            norm=LogNorm(),
            convention="xu",
            show=False,
        )
        fig, axes = plt.subplots(2, 3, layout="tight", figsize=(6, 4))
        for new_axes, old_axes in zip(axes, (object_fft_axes, exp_axes)):
            for new_ax, old_ax in zip(new_axes.flat, old_axes.flat):
                # replot with same configuration
                im = old_ax.get_images()[0]
                new_ax.imshow(
                    im.get_array(),
                    cmap=im.get_cmap(),
                    norm=LogNorm(),
                    extent=im.get_extent(),
                    origin=im.origin,
                )
                new_ax.axis(old_ax.axis())

        add_labels(axes[0], space="rcp", convention="xu")
        add_labels(axes[1], space="rcp", convention="xu")

        axes[0, 1].set_title("FFT of the final object")
        axes[1, 1].set_title("Orthogonalised experimental data")

        fig.suptitle(title)
        if save is not None:
            save_fig(fig, save, transparent=False)
        return fig, axes

    @staticmethod
    def strain_statistics(
        strain: np.ndarray,
        support: np.ndarray,
        bins: np.ndarray | int = 50,
        colors: dict = None,
        title: str = "",
        save: str = None,
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
        # Get the histograms with density=False to get counts
        histograms, kdes, means, _ = get_histogram(
            data=strain,
            support=support,
            bins=bins,
            density=False,
            region="all",
        )

        # Get the histogram with density=True to get probability density
        density_histograms, density_kdes, density_means, _ = get_histogram(
            data=strain, support=support, bins=bins, density=True, region="all"
        )
        # Update the histograms, kdes and means dictionnary
        histograms["bulk_density"] = density_histograms["bulk"]
        histograms["surface_density"] = density_histograms["surface"]
        kdes["bulk_density"] = density_kdes["bulk"]
        kdes["surface_density"] = density_kdes["surface"]
        means["bulk_density"] = density_means["bulk"]
        means["surface_density"] = density_means["surface"]

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
                axes[key],
                *histograms[e],
                *kdes[e],
                color=colors[e],
                bar_args={"edgecolor": "w"},
                kde_args={
                    "fill": True,
                    "fill_alpha": 0.45,
                    "color": "k",
                    "lw": 0.2,
                },
            )

            # Plot the mean
            axes[key].plot(
                means[e],
                0,
                color=colors[e],
                ms=4,
                markeredgecolor="k",
                marker="o",
                mew=0.5,
                label=f"Mean = {means[e]:.4f} %",
            )

        # Plot the density histograms for bulk and surface on the same subplot
        for e in ("bulk_density", "surface_density"):
            fwhms[e] = plot_histogram(
                axes["D"],
                *histograms[e],
                *kdes[e],
                color=colors[e],
                bar_args={"edgecolor": "w"},
                kde_args={
                    "fill": True,
                    "fill_alpha": 0.45,
                    "color": "k",
                    "lw": 0.2,
                },
            )

            axes["D"].plot(
                means[e],
                0,
                color=colors[e],
                ms=4,
                markeredgecolor="k",
                marker="o",
                mew=0.5,
                label=f"Mean = {means[e]:.4f} %",
            )

        for key in ("A", "B", "C"):
            axes[key].set_ylabel(r"Counts")
            handles, labels = axes[key].get_legend_handles_labels()
            handles = handles[1:-1]
            labels = labels[1:-1]
            axes[key].legend(
                handles,
                labels,
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.25, 0.8, 0.5, 0.5),
                fontsize=6,
                markerscale=0.7,
                ncols=1,
            )

        axes["D"].set_ylabel(r"Normalised counts")
        handles, labels = axes["D"].get_legend_handles_labels()
        handles = [handles[i] for i in [1, 2, 4, 5]]
        labels = [labels[i] for i in [1, 2, 4, 5]]
        # Shrink current axis by 20%
        box = axes["D"].get_position()
        axes["D"].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        axes["D"].legend(
            handles,
            labels,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            # loc="right", bbox_to_anchor=(1.5, 0.25, 0.5, 0.5),
            fontsize=6,
            markerscale=0.7,
        )
        axes["D"].set_title("Density distributions")

        axes["A"].set_xlabel(
            r"Overall strain, $\varepsilon$ (%)", color=colors["overall"]
        )
        axes["B"].set_xlabel(
            r"Bulk strain, $\varepsilon_{\text{bulk}}$ (%)",
            color=colors["bulk"],
        )
        axes["C"].set_xlabel(
            r"Surface strain, $\varepsilon_{\text{surface}}$ (%)",
            color=colors["surface"],
        )
        axes["D"].set_xlabel(
            r"$\varepsilon_{\text{bulk}}$, $\varepsilon_{\text{surface}}$ (%)"
        )

        figure.suptitle(title)
        if save:
            save_fig(figure, path=save, transparent=False)

        return figure, axes, means, fwhms
