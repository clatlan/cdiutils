import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_3D_object(
        data,
        support=None,
        cmap="jet",
        title="",
        vmin=None,
        vmax=None,
        show=True,
        marker="H"):

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
        vmax=vmax
        )
    fig.colorbar(p)
    fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()
        return None
    else:
        return fig


def plot_3D_vector_field(
        data,
        support,
        arrow=True,
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
    :param cmap:ScalarMappable the matplotlib colormap (str) used for the colorbar
    (default: "jet").
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
        colormap = mpl.cm.get_cmap(cmap)
        colors = (norm.ravel() - norm.min()) / norm.ptp()
        colors = np.concatenate((colors, np.repeat(colors, 2)))
        colors = colormap(colors)

        sm = mpl.cm.ScalarMappable(cmap=colormap, norm=None)

        q = ax.quiver(
            nonzero_coordinates[0],
            nonzero_coordinates[1],
            nonzero_coordinates[2],
            data_of_interest[..., 0],
            data_of_interest[..., 1],
            data_of_interest[..., 2],
            arrow_length_ratio=0.2,
            normalize=True,
            length=5,
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
