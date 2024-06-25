import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import numpy as np
import warnings

from cdiutils.plot.formatting import (
    get_figure_size,
    get_extent,
    CXI_VIEW_PARAMETERS
)
from cdiutils.utils import (
    find_suitable_array_shape,
    CroppingHandler,
    nan_to_zero
)


def plot_3d_voxels(
        data: np.ndarray,
        support: np.ndarray,
        view: str = "y+",
        convention: str = "cxi",
        **plot_params
) -> plt.Figure:
    """
    Plot a 3D volumetric representation of data. Voxel are plotted as
    voxels! No triangulation/interpolation. The voxels to plot are based
    on the provided support, while the colouring is generated from the
    data variable.

    Args:
        data (np.ndarray): the quantity to plot.
        support (np.ndarray): the support of the data to plot.
        view (str, optional): the initial view of the 3D plot. Can be
        "x+-/y+-/z+-". Defaults to "y+".
        convention (str, optional): The convention the provided data
        follow, eitheir XU or CXI. Defaults to "cxi".

    Raises:
        ValueError: if convention in invalid.

    Returns:
        plt.Figure: the matpltolib figure the data were drawn in.
    """
    _plot_params = {
        "cmap": plt.get_cmap("turbo"),
        "norm": Normalize(data.min(), data.max()),
        "figsize": (6, 2)
    }
    if plot_params is not None:
        _plot_params.update(plot_params)

    if convention.lower() == "cxi":
        data = np.swapaxes(data, axis1=0, axis2=2)
        support = np.swapaxes(support, axis1=0, axis2=2)
        views = {
            "x+": (180, 0, 90),
            "y+": (0, -90, 0),
            "z+": (-90, 90, 0),
            "x-": (-180, -180, -90),
            "y-": (0, 90, 0),
            "z-": (90, -90, 0),
        }
    elif convention.lower() == "xu":
        raise ValueError("'XU' not implemented yet.")
    else:
        raise ValueError("Invalid convention, can be 'CXI' or 'XU'.")

    if isinstance(_plot_params["cmap"], str):
        _plot_params["cmap"] = plt.get_cmap(_plot_params["cmap"])

    colors = _plot_params["cmap"](_plot_params["norm"](data))

    figure = plt.figure(layout="tight", figsize=_plot_params["figsize"])
    ax = figure.add_subplot(projection="3d")

    ax.set_xlabel(r"$x_{\text{cxi}}$")
    ax.set_ylabel(r"$y_{cxi}$")
    ax.set_zlabel(r"$z_{cxi}$")
    ax.view_init(elev=views[view][0], azim=views[view][1], roll=views[view][2])

    ax.voxels(
        support,
        facecolors=colors,
        edgecolors=np.clip(2*colors-0.85, 0, 1)
    )
    # ax.set_box_aspect(None, zoom=1.25)
    return figure


def hemisphere_projection(
        data: np.ndarray,
        support: np.ndarray,
        axis: int,
        looking_from_downstream: bool = True
) -> np.ndarray:
    """Compute the hemisphere projection of a volume along one axis.

    Args:
        data (np.ndarray): the volume data to project.
        support (np.ndarray): the support of the reconstructed data.
        axis (int): the axis along which to project.
        looking_from_downstream (bool, optional): The direction along
            axis, positive-going (True) or negative-going (False).
            Defaults to True.

    Returns:
        np.ndarray: the 2D array corresponding to the projection.
    """
    # Make sure we have 0 values instead of nan
    support = nan_to_zero(support)

    # Find the support surface
    if looking_from_downstream:
        support_surface = np.cumsum(support, axis=axis)
    else:
        slices = tuple(
            [np.s_[:]] * axis + [np.s_[::-1]] + [np.s_[:]] * (2 - axis)
        )
        support_surface = np.cumsum(support[slices], axis=axis)[slices]

    support_surface = np.where(support_surface > 1, 0, support_surface)
    half_shell_strain = np.where(support_surface == 0, np.nan, data)

    # Some warning is expecting here as mean of empty slices may occur
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # project the half shell strain along the direction provided
        # by axis
        return np.nanmean(half_shell_strain, axis=axis)


def plot_3d_surface_projections(
        data: np.ndarray,
        support: np.ndarray,
        voxel_size: tuple | list | np.ndarray,
        view_parameters: dict = None,
        figsize: tuple = None,
        title: str = None,
        cbar_title: str = None,
        **plot_params
) -> plt.Figure:
    """Plot 3D projected views from a 3D object.

    Args:
        data (np.ndarray): the data to plot.
        support (np.ndarray): the support of the reconstructed object.
        voxel_size (tuple | list | np.ndarray): the voxel size of
            the data to plot.
        view_parameters (dict, optional): some parameters required for
            setting the plot views. Defaults to CXI_VIEW_PARAMETERS.
        figsize (tuple, optional): the size of the figure. Defaults to
            None.
        title (str, optional): the title of the figure. Defaults to
            None.
        cbar_title (str, optional): the title of the colour bar.
            Defaults to None.

    Returns:
        matplotlib.figure.Figure: the figure.
    """
    if view_parameters is None:
        view_parameters = CXI_VIEW_PARAMETERS.copy()

    if figsize is None:
        figsize = get_figure_size(subplots=(3, 3))

    cbar_size, cbar_pad = 0.07, 0.4
    figure, axes = plt.subplots(
        2, 3,
        layout="tight",
        figsize=figsize,
        gridspec_kw={'height_ratios': [1/(1-(cbar_pad+cbar_size)), 1]}
    )
    shape = find_suitable_array_shape(support, symmetrical_shape=False)

    cropped_support,  _, _, roi = CroppingHandler.chain_centering(
        support,
        output_shape=shape,
        methods=["com"],
    )

    cropped_data = data[CroppingHandler.roi_list_to_slices(roi)]

    for v in view_parameters:
        looking_from_downstream = False
        row = 0
        if v.endswith("+"):
            looking_from_downstream = True
            row = 1

        ax = axes[row, view_parameters[v]["axis"]]

        projection = hemisphere_projection(
            cropped_data,
            cropped_support,
            axis=view_parameters[v]["axis"],
            looking_from_downstream=looking_from_downstream
        )

        # Swap axes for matshow if the first plane axis is less than the
        # second, ensuring correct orientation where the first plane
        # corresponds to the y-axis and the seconde plane to the x-axis.
        # If first plane axis > second plane axis, the default orientation is
        # correct, and no swapping is needed.
        if view_parameters[v]["plane"] != sorted(view_parameters[v]["plane"]):
            projection = np.swapaxes(projection, axis1=0, axis2=1)

        # to handle extent and origin please refer to
        # https://matplotlib.org/stable/users/explain/artists/imshow_extent.html#imshow-extent
        extent = get_extent(
            shape,
            voxel_size,
            view_parameters[v]["plane"]
        )

        if view_parameters[v]["xaxis_points_left"]:
            # flip the horizontal extent, and the image horizontally
            extent = (extent[1], extent[0], *extent[2:])
            projection = projection[np.s_[:, ::-1]]

        image = ax.imshow(
            projection,
            extent=extent,
            origin="lower",
            **plot_params
        )
        ax.set_title(v, y=0.95)

        # Set a new boolean for whether y-axis should be right or left
        yaxis_left = view_parameters[v]["xaxis_points_left"]

        # Remove the useless spines
        ax.spines[
            ["top", "left" if yaxis_left else "right"]].set_visible(False)

        # Set the position of the spines
        ax.spines["right" if yaxis_left else "left"].set_position(
                ("axes", yaxis_left)
        )

        # Customize ticks and tick labels
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("right" if yaxis_left else "left")
        ax.yaxis.set_label_position("right" if yaxis_left else "left")

        # Plot the shaft of the axis
        ax.plot(
            yaxis_left,
            1,
            "^k",
            transform=ax.transAxes,
            clip_on=False
        )
        ax.plot(
            1-yaxis_left, 0,
            "<k" if yaxis_left else ">k",
            transform=ax.transAxes,
            clip_on=False
        )
        xlabel = view_parameters[v]["xlabel"]
        ylabel = view_parameters[v]["ylabel"]

        ax.set_xlabel(xlabel, labelpad=1)
        ax.set_ylabel(ylabel, labelpad=1)
        ax.tick_params(axis='both', which='major', pad=1.5)

        ax.locator_params(nbins=5)

    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes("top", size=cbar_size, pad=cbar_pad)
    figure.colorbar(
        image,
        cax=cax,
        extend="both",
        orientation="horizontal",
    )
    cax.set_title(cbar_title)

    figure.suptitle(title)
    return figure


def plot_3d_object(
        data,
        support=None,
        cmap="turbo",
        title="",
        vmin=None,
        vmax=None,
        show=True,
        marker="H",
        alpha=1
):

    """
    Plot a 3D object.

    :param data: the 3D array (np.array) to plot.
    :param support: 3D array (np.array) with the same shape as data.
    Support is the shape of the 3D data to plot, coordinates whose
    values <= 0 won't be plotted. Coordinates whose values > 0 are
    considred to be part of the object to plot.
    :param cmap: the matplotlib colormap (str) used for the colorbar
    (default: "jet").
    :param title: title (str) of the figure. Default is empty string.
    :param vmin: the minimum value (float) for the color scale
    (default: None).
    :param vmax: the maximum value (float) for the color scale
    (default: None).
    :param show: whether or not to show the figure (bool). If False, the
    figure is not displayed but returned.
    :return: None if show is True, otherwise the figure.
    """

    if support is None:
        support = np.ones(shape=data.shape)

    data_of_interest = np.where(support > 0, data, 0)
    nonzero_coordinates = data_of_interest.nonzero()
    nonzero_data = data_of_interest[(nonzero_coordinates[0],
                                     nonzero_coordinates[1],
                                     nonzero_coordinates[2])]
    if vmin is None:
        vmin = np.min(nonzero_data)
    if vmax is None:
        vmax = np.max(nonzero_data)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    p = ax.scatter(
        nonzero_coordinates[0],
        nonzero_coordinates[1],
        nonzero_coordinates[2],
        c=nonzero_data,
        cmap=cmap,
        marker=marker,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha
    )
    fig.colorbar(p)
    fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()
        
    return fig


def plot_3d_vector_field(
        data,
        support,
        arrow=True,
        scale=5,
        cmap="jet",
        title="",
        vmin=None,
        vmax=None,
        verbose=False):
    """
    Plot a 3D vector field represented by arrows.

    :param data: the (4-)3D data (np.array). If the object to plot has
    a shape m * n * l, then the data must be 3 * m * n * l. Each voxel
    must contain 3 chanels that describe the vector to plot.
    :param support: 3D array (np.array) with the same shape as data but
    without the chanels (therefore m * n * l). Support is the shape of
    the 3D data to plot, coordinates whose values <= 0 won't be plotted.
    Coordinates whose values > 0 are considred to be part of the object
    to plot.
    :param arrow: whether or not to used arrows for field representation
    (bool). If False, marker "o" is plotted instead and color represents
    norm of the arrow.
    :param cmap:ScalarMappable the matplotlib colormap (str) used for
    the colorbar (default: "turbo").
    :param title: title (str) of the figure. Default is empty string.
    :param vmin: the minimum value (float) for the color scale
    (default: None).
    :param vmax: the maximum value (float) for the color scale
    (default: None).
    :param verbose: whether or not to print out the min and max values
    of the absolute vector field (bool).
    """

    nonzero_coordinates = np.where(support > 0)
    data_of_interest = data[nonzero_coordinates[0],
                            nonzero_coordinates[1],
                            nonzero_coordinates[2],
                            ...]

    norm = np.empty(data_of_interest.shape[0])

    for i in range(data_of_interest.shape[0]):
        norm[i] = np.linalg.norm(data_of_interest[i, ...])
    if vmin is None:
        vmin = np.min(norm)
    if vmax is None:
        vmax = np.max(norm)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(projection="3d")
    if arrow:
        colormap = plt.get_cmap(cmap)
        colors = (norm.ravel() - norm.min()) / norm.ptp()
        colors = np.concatenate((colors, np.repeat(colors, 2)))
        colors = colormap(colors)

        sm = plt.cm.ScalarMappable(cmap=colormap, norm=None)

        q = ax.quiver(
            nonzero_coordinates[0],
            nonzero_coordinates[1],
            nonzero_coordinates[2],
            data_of_interest[..., 0],
            data_of_interest[..., 1],
            data_of_interest[..., 2],
            arrow_length_ratio=0.2,
            normalize=True,
            length=scale,
            colors=colors
            )

        sm.set_array(np.linspace(vmin, vmax))
        fig.colorbar(sm, ax=ax, orientation='vertical')
        q.set_edgecolor(colors)
        q.set_facecolor(colors)

    else:
        p = ax.scatter(
            nonzero_coordinates[0],
            nonzero_coordinates[1],
            nonzero_coordinates[2],
            c=norm,
            cmap=cmap,
            marker='o',
            vmin=vmin,
            vmax=vmax
            )

        fig.colorbar(p)

    fig.suptitle(title)
    fig.tight_layout()

    if verbose:
        print("Minimum value is {}".format(vmin))
        print("Maximum value is {}".format(vmax))
