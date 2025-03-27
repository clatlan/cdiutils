"""
A function for plotting stereographic projections of reciprocal space
data.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.interpolate import griddata

from cdiutils.plot import (
    save_fig,
    add_colorbar,
    add_labels,
    plot_volume_slices,
)


def pole_figure(
        intensity: np.ndarray,
        grid: list,
        axis: str = "2",
        radius: float = None,
        dr: float = None,
        resolution: int = 250,
        figsize: tuple = (4, 4),
        title: str = None,
        verbose: bool = False,
        save: str = None,
        **plot_params,
) -> tuple:
    """
    Generate a crystallographic pole figure using stereographic
    projection of 3D diffraction intensity.

    A stereographic projection maps points on a sphere to a plane by
    projecting from the opposite pole. In crystallography, this is used
    to visualise the distribution of crystal directions.

    Standard convention:
    - When axis in ("0", "1", "2"): The upper hemisphere is projected
    onto the equatorial plane with projection lines extending towards
    the south pole (observer position)
    - When axis in ("-0", "-1", "-2"): The lower hemisphere is projected
    onto the equatorial plane with projection lines extending towards
    the north pole (observer position)

    Args:
        intensity (np.ndarray): 3D array of intensity values.
        grid (list): list of 1D arrays representing the orthogonal grid.
            ex: [x_coords, y_coords, z_coords]
        axis (str, optional): Projection axis and hemisphere selection:
            Positive values ("0","1","2") select the upper hemisphere
            (highest coordinates), projecting from the opposite pole.
            Negative values ("-0","-1","-2") select the lower hemisphere
            (lowest coordinates), projecting from the opposite pole.

            The absolute value indicates which axis (xu convention):
                |axis|=0: project onto yz-plane (normal to x-axis)
                |axis|=1: project onto xz-plane (normal to y-axis)
                |axis|=2: project onto xy-plane (normal to z-axis)

            Defaults to "2", giving the standard upper hemisphere
            projection onto the xy-plane.

        radius (float, optional): Radius of the spherical shell to
            select data from, centered at origin.
            If None, uses 0.25 * the maximum radial distance in the data
            Defaults to None.
        dr (float, optional): Thickness of the spherical shell.
            If None, uses 0.01 * radius. Defaults to None.
        resolution (int, optional): Resolution of the output 2D grid
            (number of points per dimension). Defaults to 250.
        figsize(tuple, optional): Size of the figure. Defaults to
            (4, 4).
        title (str, optional): Title for the plot. Defaults to None.
        verbose (bool, optional): Whether to print and plot additional
            information. Defaults to False.
        save (str, optional): File path to save the plot. Defaults to
            None.
        **plot_params (dict, optional): Additional parameters for the
            plotting function.

    Raises:
        ValueError: If axis is invalid or no data points found in shell

    Returns:
        tuple: (grid_x, grid_y, projected_intensity): The projected 2D
            grid coordinates and intensity values. Also returns
            (fig, ax): The figure and axis objects.

    Example:
        >>> import numpy as np
        >>> from cdiutils.analysis import pole_figure
        >>> intensity = np.random.random((100, 100, 100))
        >>> grid = [np.linspace(-1, 1, 100)] * 3

        >>> (grid_x, grid_y, projected_intensity), (fig, ax) = pole_figure(
                intensity, grid, axis="2"
            )
        >>> plt.show()
    """
    # parse the axis parameter - handle as a string first
    if not isinstance(axis, str):
        axis = str(axis)

    # check if we're selecting upper or lower hemisphere
    select_upper_hemisphere = not axis.startswith("-")

    # convert to absolute value for determining projection axis
    if axis.startswith("-"):
        projection_axis = int(axis[1:])
    else:
        projection_axis = int(axis)

    if projection_axis >= 3:
        raise ValueError("Axis must be in ('-2', '-1', '-0', '0', '1', '2')")

    observer_position = "South" if select_upper_hemisphere else "North"
    hemisphere = "upper" if select_upper_hemisphere else "lower"

    if verbose:
        print(
            f"Projection axis: {projection_axis}, selecting {hemisphere} "
            f"hemisphere with observer at {observer_position} Pole"
        )

    # calculate center points for the full grid
    centres = [np.mean(g) for g in grid]

    # Determine which part of the data to keep based on hemisphere selection
    slices = [slice(None), slice(None), slice(None)]
    if select_upper_hemisphere:
        # lower hemisphere will be zeroed out (relative to projection axis)
        slices[projection_axis] = slice(
            None, intensity.shape[projection_axis] // 2
        )
    else:
        # upper hemisphere will be zeroed out
        slices[projection_axis] = slice(
            intensity.shape[projection_axis] // 2, None
        )

    # select the hemisphere data and zero out the other hemisphere
    hemisphere_intensity = intensity.copy()
    hemisphere_intensity[tuple(slices)] = 0

    # make the meshgrid from the grid
    coordinate_meshgrids = np.meshgrid(*grid, indexing='ij')

    # make a spherical shell mask
    radii = np.sqrt(sum(
        (coordinate_meshgrids[i] - centres[i]) ** 2
        for i in range(3)
    ))

    # set default radius and thickness if not provided
    if radius is None:
        radius = 0.25 * np.max(radii)  # A quarter of the max radius
    if dr is None:
        dr = 0.01 * radius

    if verbose:
        print(
            f"Selected radius: {radius:.3f} and spherical "
            f"shell thickness: {dr:.5f}"
        )

    shell_mask = np.logical_and(
        radii > (radius - dr/2),
        radii < (radius + dr/2)
    )

    # plot the filtered data if requested
    if verbose:
        _, debug_axes = plt.subplots(2, 2, layout="tight", figsize=figsize)
        params = {
            "norm": LogNorm(),
            "cmap": "turbo",
            "convention": "xu",
            "voxel_size": [np.diff(g).mean() for g in grid],
            "data_centre": [g.mean() for g in grid],
            "show": False
        }
        params["cmap"] = plt.get_cmap(params["cmap"]).copy()
        params["cmap"].set_bad(params["cmap"](0))
        params["cmap"].set_under(params["cmap"](0))

        for col, to_plot in enumerate(
                (
                    hemisphere_intensity,
                    np.ma.masked_where(~shell_mask, hemisphere_intensity)
                )
        ):
            _, old_axes = plot_volume_slices(to_plot, **params)
            add_labels(old_axes, convention="xu")
            axis_indexes = [0, 1, 2]
            axis_indexes.remove(projection_axis)
            old_axes = old_axes[axis_indexes]
            for new_ax, old_ax in zip(debug_axes[:, col].flat, old_axes.flat):
                im = old_ax.get_images()[0]
                new_ax.imshow(
                    im.get_array(), cmap=im.get_cmap(), norm=LogNorm(),
                    extent=im.get_extent(), origin=im.origin
                )
                new_ax.axis(old_ax.axis())
                new_ax.set_xlabel(old_ax.get_xlabel())
                new_ax.set_ylabel(old_ax.get_ylabel())
            debug_axes[0, col].set_title(
                f"Intensity of the {hemisphere} hemisphere" if col == 0
                else "Shell-masked intensity"
            )

    # extract the masked data
    shell_intensity = hemisphere_intensity[shell_mask]

    # check if we have valid data to work with
    if shell_intensity.size == 0:
        raise ValueError(
            "No data points found in the specified radius shell. "
            "Try adjusting radius or dr."
        )

    # calculate stereographic projection coordinates
    # determine which coordinates to use for the projection
    equatorial_plane_axes = [i for i in range(3) if i != projection_axis]

    # Get the coordinates for points in the mask
    masked_coordinates = [
        coordinate_meshgrids[i][shell_mask] - centres[i]
        for i in range(3)
    ]

    # Stereographic projection formula depends on hemisphere selection.
    # for upper hemisphere, project from south pole (opposite pole)
    # for lower hemisphere, project from north pole (opposite pole)
    if select_upper_hemisphere:
        denominator = radius + masked_coordinates[projection_axis]
    else:
        denominator = radius - masked_coordinates[projection_axis]

    # avoid division by zero
    safe_denominator = np.where(
        np.abs(denominator) < 1e-10, 1e-10, denominator
    )

    # Calculate projections for the two equatorial plane axes0.
    # The negative sign ensures correct orientation.
    # Scaling by 90 maps to stereographic degrees (0° at center, 90° at rim).
    projected_coordinates = [
        -masked_coordinates[equatorial_plane_axes[0]] / safe_denominator * 90,
        -masked_coordinates[equatorial_plane_axes[1]] / safe_denominator * 90
    ]

    # make the target 2D grid for interpolation
    grid_x, grid_y = np.mgrid[-90:90:resolution*1j, -90:90:resolution*1j]

    # interpolate the intensity values onto the regular 2D grid
    projected_intensity = griddata(
        (projected_coordinates[0], projected_coordinates[1]),  # source points
        shell_intensity,  # values at those points
        (grid_x, grid_y),  # target grid points
        method="cubic",                
        fill_value=np.nan
    )

    # generate a plot
    _plot_params = {
        "cmap": "cet_CET_C9s_r",
        "norm": LogNorm(1, 0.8 * np.max(projected_intensity)),
        "interpolation": "nearest",
        "origin": "lower",
        "extent": (-90, 90, -90, 90)
    }
    if plot_params:
        _plot_params.update(plot_params)

    # a colormap that handles nan values properly
    _plot_params["cmap"] = plt.get_cmap(_plot_params["cmap"]).copy()
    _plot_params["cmap"].set_bad("k")
    _plot_params["cmap"].set_under("k")

    fig, ax = plt.subplots(figsize=figsize, layout="tight")
    ax.axis("off")

    # plot the interpolated data
    im = ax.imshow(projected_intensity, **_plot_params)
    add_colorbar(ax, im, label="Intensity", extend="max")  # add a colorbar

    # add circles with labels along the diagonal to upper right
    angles = [30, 45, 60, 75]

    # position for the labels in radians 45 degrees (upper right diagonal)
    label_pos = np.deg2rad(45)
    y_shift = 7  # slight shift to avoid overlap with the circle

    circle_params = {
        "color": "white",
        "fill": False,
        "linestyle": "dotted",
        "linewidth": 0.75,
    }

    for angle in angles:
        circle = plt.Circle((0, 0), angle, **circle_params)
        ax.add_patch(circle)

        # calculate label position along the diagonal
        label_x = angle * np.cos(label_pos)
        label_y = angle * np.sin(label_pos)

        ax.text(
            label_x,
            label_y + y_shift,
            f"{angle}°",
            color="white",
            ha="center",
            va="center",
            fontsize=7
        )

    # add the primitive circle (90°) with a different style and label
    circle_params["linestyle"] = "solid"
    primitive_circle = plt.Circle((0, 0), 90, **circle_params)
    ax.add_patch(primitive_circle)

    # label for primitive circle along the same diagonal
    ax.text(
        (90 * np.cos(label_pos)) * 1.15,
        (90 * np.sin(label_pos)) * 1.05,
        "90°\n(equator)",
        color="white",
        ha="center",
        va="bottom",
        fontsize=7,
    )

    # add angle lines (azimuths)
    for azimuth in range(0, 360, 45):
        rad = np.deg2rad(azimuth)
        ax.plot(
            [0, 90 * np.cos(rad)],
            [0, 90 * np.sin(rad)],
            color="white",
            linestyle="dashed",
            linewidth=0.75,
        )

    # set axis labels based on the projection axis
    if title is None:
        title = (
            f"Pole Figure: {hemisphere} hemisphere along axis{projection_axis}"
            f"\n(Stereographic projection from {observer_position} Pole)"
        )
    ax.set_title(title, fontsize=7)

    if save is not None:
        save_fig(fig, save, transparent=False)

    return (grid_x, grid_y, projected_intensity), (fig, ax)
