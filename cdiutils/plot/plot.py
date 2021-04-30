import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

mpl.rcParams["mpl_toolkits.legacy_colorbar"] = False


def plot_slices(*data, titles=None, figsize=(6, 4), cmap="viridis",
                log_scale=False, suptitle=None):
    fig = plt.figure(figsize=figsize)

    if log_scale:
            data = np.log(data)
    vmin = np.min(data)
    vmax = np.max(data)

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(len(data), 3),
                    axes_pad=0.05,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.2)

    for i, plot in enumerate(data):
        shape = plot.shape
        im = grid[3 * i].matshow(plot[shape[0]//2, ...],
                                 cmap=cmap, vmin=vmin, vmax=vmax)
        grid[3*i].annotate(titles[i] if titles is not None else "",
                           xy=(0.2, 0.5),
                           xytext=(-grid[3*i].yaxis.labelpad - 2, 0),
                           xycoords=grid[3*i].yaxis.label,
                           textcoords='offset points',
                           size="small", ha='right', va='center')
        grid[3 * i + 1].matshow(plot[: , shape[1]//2, :],
                                cmap=cmap, vmin=vmin, vmax=vmax)
        grid[3 * i + 2].matshow(plot[..., shape[2]//2],
                                cmap=cmap, vmin=vmin, vmax=vmax)

    for i, ax in enumerate(grid):
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    cbar = grid.cbar_axes[0].colorbar(im)
    fig.suptitle(suptitle)
    plt.show()


def fancy_plot(data, title=None, log_scale=False, figsize=None, cmap='jet'):
    fig = plt.figure(figsize=figsize)

    if log_scale:
        data = np.where(data <= 1, 1, data)
        data = np.log10(data)
    vmin = np.min(data)
    vmax = np.max(data)

    shape = data.shape
    plt.imshow(data[:, shape[1] // 2, :], vmin=vmin, vmax=vmax)
    fig.suptitle(title)
    plt.show()


def plot_3D_object(data, support=None, complex_object=False,
                   cmap="jet", title="", vmin=None, vmax=None):
    if support is None:
        support = np.ones(shape=data.shape)
    if complex_object:
        data_of_interest = np.where(support > 0, np.abs(data), 0)
    else:
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
    p = ax.scatter(nonzero_coordinates[0], nonzero_coordinates[1],
                   nonzero_coordinates[2], c=nonzero_data,
                   cmap=cmap, marker="o", vmin=vmin, vmax=vmax)
    fig.colorbar(p)
    fig.suptitle(title)
    fig.tight_layout()


def plot_3D_vector_field(data, support, arrow=True,
                         cmap="jet", title="",
                         vmin=None, vmax=None,
                         verbose=False):

    nonzero_coordinates = np.where(support > 0)
    data_of_interest = data[nonzero_coordinates[0], nonzero_coordinates[1],
                            nonzero_coordinates[2], ...]

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

        q = ax.quiver(nonzero_coordinates[0],
                      nonzero_coordinates[1],
                      nonzero_coordinates[2],
                      data_of_interest[..., 0],
                      data_of_interest[..., 1],
                      data_of_interest[..., 2],
                      arrow_length_ratio=0.2,
                      normalize=True,
                      length=5,
                      colors=colors)
        sm.set_array(np.linspace(vmin, vmax))
        fig.colorbar(sm, ax=ax, orientation='vertical')
        q.set_edgecolor(colors)
        q.set_facecolor(colors)

    else:
        p = ax.scatter(nonzero_coordinates[0],
                       nonzero_coordinates[1],
                       nonzero_coordinates[2],
                       c=norm,
                       cmap=cmap,
                       marker='o',
                       vmin=vmin, vmax=vmax)
        fig.colorbar(p)

    fig.suptitle("3D Displacement Field\n" + title)
    fig.tight_layout()

    if verbose:
        print("Minimum value is {}".format(vmin))
        print("Maximum value is {}".format(vmax))
