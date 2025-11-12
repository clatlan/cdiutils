import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev, splrep
from scipy.ndimage import rotate

from cdiutils.utils import (
    crop_at_center,
    nan_to_zero,
    normalise,
    to_bool,
)


def plot_deviation(
    ax,
    x,
    y_pos,
    deviation,
    scale=1,
    arrow=False,
    attribute=None,
    reference_line=True,
    vmin=None,
    vmax=None,
    centered=True,
    cmap="jet",
    interpolate=2,
    linewidth=0.7,
    **kwargs,
):
    colormap = plt.get_cmap(cmap)

    support = to_bool(deviation, nan_value=True)
    if reference_line:
        ax.plot(
            x,
            y_pos * support,
            c="grey",
            ls="--",
            lw=linewidth / 2,
        )

    support = np.repeat(support, interpolate)

    if np.isnan(deviation).any():
        deviation = nan_to_zero(deviation)

    if interpolate:
        spl_deviation = splrep(
            x,
            deviation,
            # s=(x.shape[0]-np.sqrt(2*x.shape[0]), x.shape[0]+np.sqrt(2*x.shape[0]))
        )
        if attribute is not None:
            spl_attribute = splrep(x, attribute)
        x = np.linspace(0, np.max(x), x.shape[0] * interpolate)
        deviation = splev(x, spl_deviation)
        if attribute is not None:
            attribute = splev(x, spl_attribute)

    y = support * deviation * scale + y_pos

    if vmin and vmax:
        if centered and (vmin < 0 and vmax >= 0):
            vmin, vmax = -np.max([-vmin, vmax]), np.max([-vmin, vmax])
        if attribute is not None:
            normalised_attribute = (attribute - vmin) / (vmax - vmin)
        else:
            normalised_attribute = (deviation - vmin) / (vmax - vmin)
    else:
        normalised_attribute = normalise(
            data=attribute if attribute is not None else deviation,
            zero_centered=centered,
        )

    c = colormap(normalised_attribute)

    length = len(x) if type(x) is list else x.shape[0]

    for i in range(length - 1):
        ax.plot(
            [x[i], x[i + 1]],
            [y[i], y[i + 1]],
            c=c[i],
            linewidth=linewidth,
            zorder=kwargs["zorder"],
        )
        if arrow and i % interpolate == 0 and not np.isnan(support[i]):
            ax.quiver(
                x[i],
                y_pos,
                0,
                deviation[i],
                color=c[i],
                scale=1 / scale,
                scale_units="xy",
                angles="xy",
                **kwargs,
            )

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=None)
    sm.set_array(attribute if attribute is not None else deviation)
    sm.set_clim(vmin, vmax)

    return ax, sm


def quiver_plot(
    ax,
    support,
    disp,
    strain,
    target_shape,
    slice_pos,
    displacement_scale=1,
    min_max_strain=(-0.1, 0.1),
    min_max_disp=(-0.01, 0.01),
    background_cmap=plt.get_cmap("cet_CET_D13"),
    foreground_cmap=plt.get_cmap("cet_CET_D8"),
    crop_fit=[1, -1],
    rotation=0,
    flip=False,
    interpolate=1,
    linewidth=0.7,
    contour_linewidth=1,
    reference_line=True,
    return_colorbar=False,
    no_background=False,
    no_foreground=False,
    background_opacity=1,
    aspect_ratio="equal",
    **kwargs,
):
    if rotation != 0:
        strain = rotate(strain, rotation, axes=(0, 2))
        disp = rotate(disp, rotation, axes=(0, 2))
        support = rotate(support, rotation, axes=(0, 2))

    disp = crop_at_center(disp, final_shape=target_shape)
    strain = crop_at_center(strain, final_shape=target_shape)
    support = crop_at_center(support, final_shape=target_shape)

    # Flip the slice if needed
    if flip:
        disp = np.flip(disp, axis=2)
        strain = np.flip(strain, axis=2)
        support = np.flip(support, axis=2)

    # Make the last minor slice adjustement to have a tight layout plot
    disp = disp[..., crop_fit[0] : crop_fit[-1]]
    strain = strain[..., crop_fit[0] : crop_fit[-1]]
    support = support[..., crop_fit[0] : crop_fit[-1]]

    X, Y = np.meshgrid(
        np.arange(0, support.shape[2]), (np.arange(0, support.shape[1]))
    )

    ax.contour(
        X,
        Y,
        support[slice_pos, ...],
        levels=[0, 0.009],
        linewidths=contour_linewidth,
        colors="k",
        zorder=2.3,
    )
    if not no_background:
        # strain = strain * support
        background = ax.matshow(
            strain[slice_pos, ...],
            origin="lower",
            cmap=background_cmap,
            vmin=min_max_strain[0],
            vmax=min_max_strain[1],
            alpha=background_opacity,
            zorder=1,
            aspect=aspect_ratio,
        )
    else:
        background = None

    # sized_up_support = size_up_support(support)
    # ax.matshow(
    #     np.where(sized_up_support == 1, np.nan, 0)[slice_pos, ...],
    #     origin="lower",
    #     cmap=matplotlib.colors.ListedColormap(['white']),
    #     alpha=1,
    #     zorder=2.2
    # )
    if not no_foreground:
        for z in np.arange(0, disp.shape[1]):
            ax, sm = plot_deviation(
                ax,
                x=np.arange(0, disp.shape[2]),
                y_pos=z,
                deviation=disp[slice_pos, z, :],
                scale=displacement_scale,
                vmin=min_max_disp[0],
                vmax=min_max_disp[1],
                centered=True,
                cmap=foreground_cmap,
                arrow=True,
                reference_line=reference_line,
                interpolate=interpolate,
                linewidth=linewidth,
                zorder=2.1,
                **kwargs,
            )
    else:
        sm = None
    if not return_colorbar:
        return ax
    else:
        return ax, background, sm
