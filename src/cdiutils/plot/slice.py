import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import AxesGrid

from cdiutils.plot.formatting import (
    CXI_VIEW_PARAMETERS,
    NATURAL_VIEW_PARAMETERS,
    XU_VIEW_PARAMETERS,
    add_colorbar,
    get_figure_size,
    get_x_y_limits_extents,
    set_x_y_limits_extents,
)
from cdiutils.utils import (
    extract_reduced_shape,
    get_centred_slices,
    nan_to_zero,
)


def plot_volume_slices(
    data: np.ndarray,
    support: np.ndarray = None,
    voxel_size: tuple | list = None,
    data_centre: tuple | list = None,
    views: tuple[str] = None,
    convention: str = None,
    title: str = None,
    equal_limits: bool = True,
    slice_shift: tuple | list = None,
    integrate: bool = False,
    opacity: np.ndarray = None,
    plot_type: str = "imshow",
    contour_levels: int = 100,
    show: bool = True,
    figsize: tuple | list =(6, 2),
    label_size: int = 6,
    **plot_params,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Generic function for plotting 2D slices (cross section or sum, with
    option 'integrate') of 3D volumes. The slices are plotted according
    to the specified views and conventions. If not specified, natural
    views are plotted in matrix convention (x-axis: 2nd dim, y-axis:
    1st dim), i.e:
    * first slice: taken at the centre of axis0
    * second slice: taken at the centre of axis1
    * third slice: taken at the centre of axis2

    Args:
        data (np.ndarray): the data to plot.
        support (np.ndarray, optional): a support for the data. Defaults
            to None.
        voxel_size (tuple | list, optional): the voxel size to modify
            the aspect ratio accordingly. Defaults to None.
        data_centre (tuple | list, optional): the centre to take the
            data at. Defaults to None.
        views (tuple[str], optional): the views for each plot according
            to the provided convention. If None default views of the
            specified convention are plotted. Defaults to None.
        convention (str, optional): the convention employed to plot the
            multiple slices, if views not specified, will set the
            default views for the specified convention, i.e.:
            ("x-", "y+", "z-") for XU convention and ("z+", "y-", "x+")
            for the CXI convention. If None, natural views are plotted.
            Defaults to None.
        title (str, optional): the title of the plot. Defaults to None.
        equal_limits (bool, optional): whether to have the same limit
            extend for all axes. Defaults to True.
        slice_shift (tuple | list, optional): the shift in the slice
            selection, by default will use the centre for each dim.
            Defaults to None.
        integrate (bool, optional): whether to sum the data instead of
            taking the slice. Defaults to False.
        opacity (np.ndarray, optional): the opacity 3D array of the
            data. Defaults to None. If constant opacity is required, use
            the 'alpha' parameter.
        plot_type (str, optional): Type of plot to use. Options are
            'imshow' or 'contourf'. Defaults to 'imshow'.
        contour_levels (int, optional): Number of contour levels when
            using 'contourf' plot type. Defaults to 100.
        show (bool, optional): whether to show the plot. Defaults to
            True. False might be useful if the function is only used for
            generating the axes that are then redrawn afterwards.
        **plot_params: additional plot params that will be parsed into
            the matplotlib imshow() function.

    Returns:
        tuple[plt.Figure, plt.Axes]: the generated figure and axes.
    """
    _plot_params = {"cmap": "turbo"}

    if plot_params:
        _plot_params.update(plot_params)

    view_params = CXI_VIEW_PARAMETERS.copy()
    if convention is None:
        if views is None:  # Simplest case, no swapping, no flipping etc.
            # For the default behaviour we use the 'natural views'
            view_params = NATURAL_VIEW_PARAMETERS.copy()
            views = ("dim0", "dim1", "dim2")
    elif convention.lower() in ("xu", "lab"):
        view_params = XU_VIEW_PARAMETERS.copy()  # overwrite the params
        if views is None:
            views = ("x-", "y+", "z-")
    elif convention.lower() == "cxi":
        if views is None:
            views = ("z+", "y-", "x+")

    slices = get_centred_slices(data.shape, shift=slice_shift)

    shape = data.shape
    if support is not None:
        shape = extract_reduced_shape(support)

    if voxel_size is not None:
        extents = get_x_y_limits_extents(data.shape, voxel_size, data_centre)
        limits = get_x_y_limits_extents(
            shape, voxel_size, data_centre, equal_limits=equal_limits
        )

    figure, axes = plt.subplots(1, 3, layout="tight", figsize=figsize)
    for i, v in enumerate(views):
        plane = view_params[v]["plane"]
        to_plot = data.sum(axis=i) if integrate else data[slices[i]]
        _plot_params["alpha"] = np.ones_like(to_plot)
        if opacity is not None:
            _plot_params["alpha"] = opacity[slices[i]]
        if plane[0] > plane[1]:
            to_plot = np.swapaxes(to_plot, 1, 0)
            _plot_params["alpha"] = np.swapaxes(_plot_params["alpha"], 1, 0)

        if view_params[v]["xaxis_points_left"]:
            to_plot = to_plot[np.s_[:, ::-1]]
            _plot_params["alpha"] = _plot_params["alpha"][np.s_[:, ::-1]]

        # Handle plot type
        if plot_type in ("contourf", "contour"):
            ny, nx = to_plot.shape
            if voxel_size is not None:
                y_coords = np.linspace(
                    extents[plane[0]][0], extents[plane[0]][1], ny
                )
                x_coords = np.linspace(
                    extents[plane[1]][0], extents[plane[1]][1], nx
                )
                if view_params[v]["xaxis_points_left"]:
                    x_coords = np.flip(x_coords)
                X, Y = np.meshgrid(x_coords, y_coords)
            else:
                X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

            alpha = _plot_params.pop("alpha", None)
            im = axes[i].contourf(
                X, Y, to_plot, levels=contour_levels, **_plot_params
            )

            # 2D array of opacity is not supported in contourf, so we
            # need a workaround: we add a contourf with the alpha values
            if opacity is not None:
                whites = [
                    (1, 1, 1, 1 - i / (contour_levels - 1))
                    for i in range(contour_levels)
                ]
                axes[i].contourf(
                    X, Y, alpha, levels=contour_levels, colors=whites
                )
            add_colorbar(axes[i], im,label_size=label_size)
            axes[i].set_aspect("equal")

        elif plot_type == "imshow":
            im = axes[i].imshow(to_plot, **_plot_params)
            add_colorbar(axes[i], im,label_size=label_size)

            if voxel_size is not None:
                set_x_y_limits_extents(
                    axes[i],
                    extents,
                    limits,
                    plane,
                    view_params[v]["xaxis_points_left"],
                )
        else:
            raise ValueError(
                f"Unknown plot type '{plot_type}'. "
                "Options are 'imshow' or 'contourf'."
            )

    figure.suptitle(title)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure, axes


def plot_multiple_volume_slices(
    *data_arrays: np.ndarray,
    data_labels: list[str] = None,
    supports: list[np.ndarray] = None,
    voxel_sizes: list = None,
    data_centres: list = None,
    slice_shifts: list = None,
    data_stacking: str = "horizontal",
    pvs_args: dict = None,
    cbar_args: dict = None,
    xlim: tuple = None,
    ylim: tuple = None,
    remove_ticks: bool = False,
    figsize: tuple = None,
    title: str = None,
    show: bool = True,
    **plot_params,
) -> plt.Figure:
    """
    Plot 2D slices of multiple 3D volumes with customizable layout.
    This function uses plot_volume_slices as a building block to create
    a composite figure comparing multiple datasets.

    Args:
        *data_arrays (np.ndarray): Multiple 3D arrays to plot.
        data_labels (list[str], optional): Labels for each dataset.
        supports (list[np.ndarray], optional): Support masks for each
            dataset.
        voxel_sizes (list, optional): List of voxel sizes for each
            dataset.
        data_centres (list, optional): List of data centers for each
            dataset.
        slice_shifts (list, optional): List of slice shifts for each
            dataset.
        data_stacking (str, optional): How to arrange plots ("vertical"
            or "horizontal"). Defaults to "vertical".
        pvs_args (dict, optional): Dictionary of parameters for
            plot_volume_slices.
        cbar_args (dict, optional): Dictionary of colorbar parameters.
        xlim (tuple, optional): Custom x-axis limits (min, max) to apply
            to all plots.
        ylim (tuple, optional): Custom y-axis limits (min, max) to apply
            to all plots.
        remove_ticks (bool, optional): Whether to remove ticks between
            subplots. Defaults to False.
        figsize (tuple, optional): Figure size. If None, calculated
            based on data.
        title (str, optional): Overall figure title.
        show (bool, optional): Whether to display the figure. Defaults
            to True.
        **plot_params: Additional plotting parameters passed to
            plot_volume_slices.

    Returns:
        plt.Figure: The generated figure.
    """
    # Validate inputs
    n_datasets = len(data_arrays)
    if n_datasets == 0:
        raise ValueError("At least one dataset must be provided")

    # Setup pvs_args with defaults
    _pvs_args = {
        "views": None,
        "convention": None,
        "equal_limits": True,
        "integrate": False,
        "show": False,  # Always False since we manage display ourselves
    }
    if pvs_args:
        _pvs_args.update(pvs_args)

    # Add any additional plot parameters
    _pvs_args.update(plot_params)

    # Determine the view labels based on convention
    convention = _pvs_args.get("convention")
    convention = convention.lower() if convention is not None else None

    if convention == "cxi":  # Default CXI views
        view_labels = ["z+ (-x, y)", "y- (-x, z)", "x+ (z, y)"]
    elif convention in ("xu", "lab"):  # Default XU views
        view_labels = ["x- (y, z)", "y- (x, z)", "z- (x, y)"]
    else:
        # Fallback to generic labels for unknown convention
        view_labels = [f"View {i + 1}" for i in range(3)]

    # Use specified views if provided
    if _pvs_args.get("views") is not None:
        view_labels = _pvs_args["views"]

    # Setup input lists with proper defaults
    input_lists = _prepare_input_lists(
        n_datasets,
        data_labels,
        supports,
        voxel_sizes,
        data_centres,
        slice_shifts,
    )

    # Set up colorbar arguments
    show_cbar = False
    if cbar_args:
        show_cbar = True  # if cbar_args is provided, we show cbar
        if "show" in cbar_args:
            show_cbar = cbar_args.pop("show")  # Override "show" if specified
        _cbar_args = {
            "location": "right",
            "title": None,
            "extend": "both",
            "ticks": None,
            "size": "5%",
            "pad": "3%",
        }
        if cbar_args.get("location", "right") == "bottom":
            _cbar_args["pad"] = "10%"
        _cbar_args.update(cbar_args)

    # Determine layout parameters
    stacking_vertical = data_stacking.lower() in ("vertical", "v")

    # First get individual plots using plot_volume_slices
    individual_plots = _generate_individual_plots(
        data_arrays, input_lists, _pvs_args
    )

    # the number of views is 3 because we are plotting 3 slices
    n_views = 3

    # Determine grid layout based on stacking direction
    nrows, ncols = (
        (n_datasets, n_views) if stacking_vertical else (n_views, n_datasets)
    )

    # Determine figure size if not provided
    if figsize is None:
        figsize = (n_datasets, n_views)  # horizontal stacking
        if stacking_vertical:
            figsize = (n_views, n_datasets)

    # Create the composite figure using AxesGrid
    composite_fig = plt.figure(figsize=figsize)
    grid = AxesGrid(
        composite_fig,
        111,
        nrows_ncols=(nrows, ncols),
        axes_pad=0.05,
        share_all=False,
        cbar_mode="single" if show_cbar else None,
        cbar_location=_cbar_args["location"] if show_cbar else "right",
        cbar_size=_cbar_args["size"] if show_cbar else None,
        cbar_pad=_cbar_args["pad"] if show_cbar else None,
    )

    # Track global min/max for colorbar
    vmin_global, vmax_global = float("inf"), float("-inf")
    images = []

    # Copy each plot from individual figures to the composite figure
    for dataset_idx, (fig, axes) in enumerate(individual_plots):
        for view_idx, ax in enumerate(axes):
            if stacking_vertical:
                grid_idx = dataset_idx * n_views + view_idx
            else:  # horizontal stacking
                grid_idx = view_idx * n_datasets + dataset_idx

            target_ax = grid[grid_idx]
            target_im = _copy_image_to_axes(ax.get_images()[0], target_ax)
            images.append(target_im)

            # Update global min/max for colorbar
            vmin, vmax = target_im.get_clim()
            vmin_global = min(vmin_global, vmin)
            vmax_global = max(vmax_global, vmax)

            # Copy axis limits and other properties
            target_ax.set_xlim(ax.get_xlim())
            target_ax.set_ylim(ax.get_ylim())

            # Apply custom x and y limits if provided
            if (
                xlim is not None
                and input_lists["voxel_sizes"][dataset_idx] is not None
            ):
                target_ax.set_xlim(xlim)
            if (
                ylim is not None
                and input_lists["voxel_sizes"][dataset_idx] is not None
            ):
                target_ax.set_ylim(ylim)

            # Add dataset and view labels
            _add_axis_labels(
                target_ax,
                stacking_vertical,
                view_idx,
                dataset_idx,
                input_lists["data_labels"][dataset_idx],
                view_labels[view_idx]
                if view_idx < len(view_labels)
                else f"View {view_idx + 1}",
            )

        # Close the individual figure to free memory
        plt.close(fig)

    # Ensure all images use the same color scale
    for im in images:
        im.set_clim(vmin_global, vmax_global)

    # Remove ticks between subplots
    for i in range(nrows):
        for j in range(ncols):
            ax = grid[i * ncols + j]
            if remove_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                if j != 0:  # Remove y-ticks for all columns except the first
                    ax.tick_params(
                        axis="y",
                        which="both",
                        left=False,
                        right=False,
                        labelleft=False,
                    )
                if i != nrows - 1:  # Remove x-ticks for all rows except last
                    ax.tick_params(
                        axis="x",
                        which="both",
                        bottom=False,
                        top=False,
                        labelbottom=False,
                    )

    # Add colorbar if requested
    if show_cbar and images:
        # Check for unsupported colorbar positions
        if _cbar_args["location"] in ("left", "top"):
            raise NotImplementedError(
                "Colorbar location 'top' and 'left' are not currently "
                "supported. Please use 'right', 'bottom'."
            )

        cbar = grid.cbar_axes[0].colorbar(
            images[0], extend=_cbar_args["extend"]
        )

        if _cbar_args["title"]:
            orientation = (
                "horizontal"
                if _cbar_args["location"] == "bottom"
                else "vertical"
            )

            if orientation == "vertical":
                # For vert. colorbar, rotate the title and position it properly
                if _cbar_args["location"] == "right":
                    cbar.ax.set_ylabel(
                        _cbar_args["title"],
                        rotation=270,
                        labelpad=10,
                        va="bottom",
                    )
                    # Adjust the y label position to align with colorbar
                    cbar.ax.yaxis.set_label_position("right")
            else:  # horizontal
                if _cbar_args["location"] == "bottom":
                    cbar.ax.set_xlabel(
                        _cbar_args["title"], ha="center", labelpad=5
                    )

        if _cbar_args["ticks"] is not None:
            cbar.set_ticks(_cbar_args["ticks"])

    # Add overall title if provided
    if title:
        composite_fig.suptitle(title, fontsize=10, y=1.02)

    # Show or close the figure
    if show:
        plt.show()
    else:
        plt.close(composite_fig)

    return composite_fig


def _prepare_input_lists(
    n_datasets: int,
    data_labels: list[str] | None,
    supports: list[np.ndarray] | None,
    voxel_sizes: list[tuple[float, float, float]] | None,
    data_centres: list[tuple[float, float, float]] | None,
    slice_shifts: list[tuple[int, int, int]] | None,
) -> dict[str, list]:
    """Prepare lists of inputs with proper defaults."""
    result = {
        "data_labels": (
            data_labels
            if data_labels and len(data_labels) == n_datasets
            else [f"Dataset {i + 1}" for i in range(n_datasets)]
        ),
        "supports": (
            supports
            if supports and len(supports) == n_datasets
            else [None] * n_datasets
        ),
        "voxel_sizes": (
            voxel_sizes
            if voxel_sizes and len(voxel_sizes) == n_datasets
            else [None] * n_datasets
        ),
        "data_centres": (
            data_centres
            if data_centres and len(data_centres) == n_datasets
            else [None] * n_datasets
        ),
        "slice_shifts": (
            slice_shifts
            if slice_shifts and len(slice_shifts) == n_datasets
            else [None] * n_datasets
        ),
    }
    return result


def _copy_image_to_axes(
    src_image: AxesImage, target_ax: plt.Axes
) -> AxesImage:
    """Copy an image from one axes to another, preserving properties."""
    array = src_image.get_array()
    cmap = src_image.get_cmap()
    norm = src_image.norm
    extent = src_image.get_extent()
    origin = src_image.origin
    new_image = target_ax.imshow(
        array, cmap=cmap, norm=norm, extent=extent, origin=origin
    )
    return new_image


def _add_axis_labels(
    ax: plt.Axes,
    stacking_vertical: bool,
    view_idx: int,
    dataset_idx: int,
    data_label: str,
    view_label: str,
) -> None:
    """Add dataset and view labels to the appropriate axes."""
    if stacking_vertical:
        # Always show data labels on the left side
        if view_idx == 0:
            ax.annotate(
                data_label,
                xy=(0, 0.5),
                xytext=(-12, 0),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="right",
                va="center",
                fontweight="bold",
            )

        # View labels at the top instead of the bottom for vert. stacking
        if dataset_idx == 0:  # First row (top) instead of last row (bottom)
            ax.annotate(
                view_label,
                xy=(0.5, 1),
                xytext=(0, 12),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
    else:
        # For horizontal stacking, keep as is
        if view_idx == 0:
            ax.annotate(
                data_label,
                xy=(0.5, 1),
                xytext=(0, 12),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        if dataset_idx == 0:
            ax.annotate(
                view_label,
                xy=(0, 0.5),
                xytext=(-12, 0),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="right",
                va="center",
            )


def _generate_individual_plots(
    data_arrays: list[np.ndarray], input_lists: dict, pvs_args: dict
) -> list[tuple[plt.Figure, np.ndarray]]:
    """Generate individual plots using plot_volume_slices."""
    individual_plots = []
    for i, data in enumerate(data_arrays):
        dataset_params = pvs_args.copy()
        dataset_params.update(
            {
                "support": input_lists["supports"][i],
                "voxel_size": input_lists["voxel_sizes"][i],
                "data_centre": input_lists["data_centres"][i],
                "slice_shift": input_lists["slice_shifts"][i],
            }
        )
        fig, axes = plot_volume_slices(data, **dataset_params)
        individual_plots.append((fig, axes))
    return individual_plots


def plot_slices(
    *data: list[np.ndarray],
    slice_labels: list = None,
    figsize: tuple[float] = None,
    data_stacking: str = "vertical",
    nan_supports: list = None,
    vmin: float = None,
    vmax: float = None,
    alphas: list = None,
    origin: str = "lower",
    cmap: str = "turbo",
    show_cbar: bool = True,
    cbar_title: str = None,
    cbar_location: str = "top",
    cbar_extend: str = "both",
    norm: matplotlib.colors.Normalize = None,
    cbar_ticks: list = None,
    slice_name: str = None,
    suptitle: str = None,
    show: bool = True,
) -> matplotlib.figure.Figure:
    """Plot 2D slices of the provided data."""

    if figsize is None:
        if data_stacking in ("vertical", "v"):
            figsize = (6, 4 * len(data))
        else:
            figsize = (6 * len(data), 4)
    if data_stacking in ("vertical", "v"):
        nrows_ncols = (len(data), 1)
    elif data_stacking in ("horizontal", "h"):
        nrows_ncols = (1, len(data))
    else:
        raise ValueError("data_stacking should be 'vertical' or 'horizontal'.")
    if slice_labels is None:
        slice_labels = [None for i in range(len(data))]
    elif len(slice_labels) != len(data):
        print(
            "Number of slice_labels should be identical to number of *data.\n"
            "slice_labels won't be displayed."
        )
        slice_labels = ["" for i in range(len(data))]

    if figsize is None:
        figsize = get_figure_size()

    figure = plt.figure(figsize=figsize)
    grid = AxesGrid(
        figure,
        111,
        nrows_ncols=nrows_ncols,
        axes_pad=0.05,
        cbar_mode="single" if show_cbar else None,
        cbar_location=cbar_location,
        cbar_pad=0.25 if show_cbar else None,
        cbar_size=0.2 if show_cbar else None,
    )

    for i, to_plot in enumerate(data):
        if nan_supports is not None:
            if isinstance(nan_supports, list):
                to_plot = to_plot * nan_supports[i]
            else:
                to_plot = to_plot * nan_supports

        im = grid[i].matshow(
            to_plot,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            origin=origin,
            norm=norm,
            alpha=None if alphas is None else alphas[i],
        )

        if data_stacking in ("vertical", "v"):
            grid[i].annotate(
                slice_labels[i] if slice_labels is not None else "",
                xy=(0.2, 0.5),
                xytext=(-grid[i].yaxis.labelpad - 2, 0),
                xycoords=grid[i].yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
            )
        else:
            grid[i].annotate(
                slice_labels[i] if slice_labels is not None else "",
                xy=(0.5, 0.9),
                xytext=(0, -grid[i].xaxis.labelpad - 2),
                xycoords=grid[i].xaxis.label,
                textcoords="offset points",
                ha="center",
                va="top",
            )

    if data_stacking in ("vertical", "v"):
        grid[len(data) - 1].annotate(
            slice_name,
            xy=(0.5, 0.2),
            xytext=(0, -grid[len(data) - 1].xaxis.labelpad - 2),
            xycoords=grid[len(data) - 1].xaxis.label,
            textcoords="offset points",
            ha="center",
            va="top",
        )
    else:
        grid[0].annotate(
            slice_name,
            xy=(0.2, 0.5),
            xytext=(-grid[0].yaxis.labelpad - 2, 0),
            xycoords=grid[0].yaxis.label,
            textcoords="offset points",
            ha="right",
            va="center",
        )
    for i, ax in enumerate(grid):
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
    if show_cbar:
        ticklocation = (
            "bottom" if cbar_location in ("top", "bottom") else "auto"
        )
        cbar = grid.cbar_axes[0].colorbar(
            im, extend=cbar_extend, ticklocation=ticklocation
        )
        grid.cbar_axes[0].set_title(cbar_title)
        if cbar_ticks:
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_ticks)
    figure.suptitle(suptitle)
    figure.tight_layout()
    if show:
        plt.show()
    return figure


def plot_contour(
    ax, support_2d, linewidth=1, color="k", pixel_size=None, data_centre=None
):
    shape = support_2d.shape
    x_range = np.arange(0, shape[1])
    y_range = np.arange(0, shape[0])
    if pixel_size is not None:
        x_range = x_range * pixel_size[1]
        y_range = y_range * pixel_size[0]
    if data_centre is not None:
        x_range = x_range - x_range.mean() + data_centre[1]
        y_range = y_range - y_range.mean() + data_centre[0]

    X, Y = np.meshgrid(x_range, y_range)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        ax.contour(
            X,
            Y,
            nan_to_zero(support_2d),
            levels=[0, 1],
            linewidths=linewidth,
            colors=color,
        )
