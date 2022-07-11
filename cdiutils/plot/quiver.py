import numpy as np
import matplotlib.cm as cm
import matplotlib
from scipy.interpolate import splev, splrep
from scipy.ndimage import rotate

from cdiutils.utils import normalize, size_up_support, crop_at_center, nan_to_zero


def plot_deviation(
        ax,
        x,
        y_pos,
        deviation,
        scale=1,
        arrow=False,
        attribute=None,
        vmin=None,
        vmax=None,
        centered=True,
        cmap="jet",
        interpolate=2,
        linewidth=0.7,
        **kwargs
):
    colormap = cm.get_cmap(cmap)

    if np.isnan(deviation).any():
        deviation = nan_to_zero(deviation)

    if interpolate:
        spl_deviation = splrep(
            x,
            deviation,
            s=(x.shape[0]-np.sqrt(2*x.shape[0]), x.shape[0]+np.sqrt(2*x.shape[0]))
        )
        if attribute is not None:
            spl_attribute = splrep(x, attribute)
        x = np.linspace(0, np.max(x), x.shape[0] * interpolate)
        deviation = splev(x, spl_deviation)
        if attribute is not None:
            attribute = splev(x, spl_attribute)

    y = deviation * scale + y_pos

    if vmin and vmax:
        if centered and (vmin < 0 and vmax >= 0):
            vmin, vmax = -np.max([-vmin, vmax]), np.max([-vmin, vmax])
        if attribute is not None:
            normalised_attribute = (attribute - vmin) / (vmax - vmin)
        else:
            normalised_attribute = (deviation - vmin) / (vmax - vmin)
    else:
        normalised_attribute = normalize(
            data=attribute if attribute is not None else deviation,
            zero_centered=centered
            )

    c = colormap(normalised_attribute)

    length = len(x) if type(x) == list else x.shape[0]
    for i in range(length-1):

        ax.plot(
            [x[i], x[i+1]],
            [y[i], y[i+1]],
            c=c[i],
            linewidth=linewidth,
        )
        if arrow and i % interpolate == 0:
            ax.quiver(
                x[i],
                y_pos,
                0,
                deviation[i],
                color=c[i],
                scale=1/scale,
                scale_units="xy",
                angles="xy",
                **kwargs
            )

    sm = cm.ScalarMappable(cmap=colormap, norm=None)
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
        background_cmap=cm.get_cmap("cet_CET_D13"),
        foreground_cmap=cm.get_cmap("cet_CET_D8"),
        crop_fit=[1, -1],
        rotation=0,
        flip=False,
        interpolate=1,
        linewidth=0.7,
        contour_linewidth=1,
        hline=True,
        return_colorbar=False,
        no_background=False,
        **kwargs
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
    disp = disp[..., crop_fit[0]: crop_fit[-1]]
    strain = strain[..., crop_fit[0]: crop_fit[-1]]
    support = support[..., crop_fit[0]: crop_fit[-1]]

    X, Y = np.meshgrid(
        np.arange(0, support.shape[2]), (np.arange(0, support.shape[1])))

    ax.contour(
        X,
        Y,
        support[slice_pos, ...],
        levels=[0, .009],
        linewidths=contour_linewidth,
        colors="k",
        zorder=2.2
    )
    if not no_background:
        background = ax.matshow(
            strain[slice_pos, ...],
            origin="lower",
            cmap=background_cmap,
            vmin=min_max_strain[0],
            vmax=min_max_strain[1],
            alpha=0.6,
            zorder=1
        )
    else:
        background = None
    
    sized_up_support = size_up_support(support)
    ax.matshow(
        np.where(sized_up_support == 1, np.nan, 0)[slice_pos, ...],
        origin="lower",
        cmap=matplotlib.colors.ListedColormap(['white']),
        alpha=1,
        zorder=2.1
    )

    for z in np.arange(0, disp.shape[1]):
        
        if hline:
            ax.axhline(
                y=z,
                xmin=0.1,
                color="grey",
                ls=":",
                linewidth=.8,
            )
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
            interpolate=interpolate,
            linewidth=linewidth,
            # zorder=3,
            **kwargs
        )
    if not return_colorbar:
        return ax
    else:
        return ax, background, sm