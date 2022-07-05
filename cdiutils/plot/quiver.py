import numpy as np
import matplotlib.cm as cm
from scipy.interpolate import splev, splrep

import sys
from cdiutils.utils import normalize


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

    if interpolate:
        spl_deviation = splrep(x, deviation)
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
