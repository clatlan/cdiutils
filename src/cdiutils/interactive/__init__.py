"""
Interactive visualisation and data exploration tools for BCDI.

This package provides interactive widgets and plotting utilities for
Bragg Coherent Diffractive Imaging (BCDI) data analysis.

Components have different optional dependencies:
- Plotter: requires ipywidgets
- Plotting functions: require ipywidgets, bokeh, panel
- ThreeDViewer: requires ipywidgets, plotly, scikit-image, scipy
- TabPlotData: requires ipywidgets, h5glance
- plot_3d_isosurface: requires ipywidgets, plotly
- VolumeViewer (PyVista): requires pyvista, trame (separate install)

Install interactive dependencies: pip install cdiutils[interactive]
Install PyVista dependencies: pip install cdiutils[pyvista]
"""

# check for different dependency groups
_DEPS = {}

# core interactive dependency (needed by most components)
try:
    import ipywidgets  # noqa: F401

    _DEPS["ipywidgets"] = True
except ImportError:
    _DEPS["ipywidgets"] = False

# plotting dependencies (for plotting.py)
try:
    import bokeh  # noqa: F401
    import panel  # noqa: F401

    _DEPS["plotting"] = True
except ImportError:
    _DEPS["plotting"] = False

# 3D visualisation dependencies (for viewer_3d.py - now Plotly based)
# note: plotly is checked separately below
_DEPS["3d"] = _DEPS.get("plotly", False)

# data browser dependencies (for data_browser.py)
try:
    import h5glance  # noqa: F401

    _DEPS["browser"] = True
except ImportError:
    _DEPS["browser"] = False

# Plotly dependencies (for volume.py and viewer_3d.py)
try:
    import plotly  # noqa: F401

    _DEPS["plotly"] = True
except ImportError:
    _DEPS["plotly"] = False

# PyVista/Trame dependencies (for volume.py - VolumeViewer only)
try:
    import pyvista  # noqa: F401
    from pyvista.trame.ui import vuetify3  # noqa: F401

    _DEPS["pyvista"] = True
except ImportError:
    _DEPS["pyvista"] = False

# convenience flags
IS_INTERACTIVE_AVAILABLE = _DEPS["ipywidgets"]
IS_FULL_INTERACTIVE = all(
    _DEPS[k] for k in ["ipywidgets", "plotting", "3d", "browser", "plotly"]
)


# factory function for creating helpful error classes
def _make_unavailable_class(name, missing_deps):
    """Create a class that raises ImportError when instantiated."""

    class UnavailableClass:
        def __init__(self, *args, **kwargs):
            deps_str = ", ".join(missing_deps)
            raise ImportError(
                f"{name} requires the following packages: {deps_str}\n"
                f"Install with: pip install {' '.join(missing_deps)}\n"
                f"Or install all interactive dependencies: pip install cdiutils[interactive]"
            )

    UnavailableClass.__name__ = name
    UnavailableClass.__qualname__ = name
    return UnavailableClass


def _make_unavailable_func(fname, deps):
    def unavailable(*args, **kwargs):
        deps_str = ", ".join(deps)
        raise ImportError(
            f"{fname}() requires: {deps_str}\n"
            f"Install with: pip install {' '.join(deps)}\n"
            f"Or: pip install cdiutils[interactive]"
        )

    unavailable.__name__ = fname
    return unavailable


# conditional imports with helpful error messages
# plotter - requires only ipywidgets
if _DEPS["ipywidgets"]:
    from .plotter import Plotter
else:
    Plotter = _make_unavailable_class("Plotter", ["ipywidgets"])

# plotting functions - require ipywidgets, bokeh, panel
if _DEPS["ipywidgets"] and _DEPS["plotting"]:
    from .plotting import plot_2d_image, plot_3d_slices, plot_data
else:
    missing = []
    if not _DEPS["ipywidgets"]:
        missing.append("ipywidgets")
    if not _DEPS["plotting"]:
        missing.extend(["bokeh", "panel"])

    plot_2d_image = _make_unavailable_func("plot_2d_image", missing)
    plot_3d_slices = _make_unavailable_func("plot_3d_slices", missing)
    plot_data = _make_unavailable_func("plot_data", missing)

# ThreeDViewer - requires ipywidgets, plotly, scikit-image, scipy
if _DEPS["ipywidgets"] and _DEPS["plotly"]:
    from .viewer_3d import ThreeDViewer
else:
    missing = []
    if not _DEPS["ipywidgets"]:
        missing.append("ipywidgets")
    if not _DEPS["plotly"]:
        missing.extend(["plotly", "scikit-image", "scipy"])
    ThreeDViewer = _make_unavailable_class("ThreeDViewer", missing)

# TabPlotData - requires ipywidgets, h5glance
if _DEPS["ipywidgets"] and _DEPS["browser"]:
    from .data_browser import TabPlotData
else:
    missing = []
    if not _DEPS["ipywidgets"]:
        missing.append("ipywidgets")
    if not _DEPS["browser"]:
        missing.append("h5glance")
    TabPlotData = _make_unavailable_class("TabPlotData", missing)

# VolumeViewer - requires pyvista, trame (separate optional dependency)
if _DEPS["pyvista"]:
    from .volume import VolumeViewer
else:
    VolumeViewer = _make_unavailable_class(
        "VolumeViewer", ["pyvista", "trame", "trame-vuetify", "trame-vtk"]
    )

# plot_3d_isosurface - requires ipywidgets, plotly (part of interactive)
if _DEPS["ipywidgets"] and _DEPS["plotly"]:
    from .volume import plot_3d_isosurface
else:
    missing = []
    if not _DEPS["ipywidgets"]:
        missing.append("ipywidgets")
    if not _DEPS["plotly"]:
        missing.extend(["plotly", "kaleido"])

    plot_3d_isosurface = _make_unavailable_func("plot_3d_isosurface", missing)


__all__ = [
    "TabPlotData",
    "Plotter",
    "ThreeDViewer",
    "VolumeViewer",
    "plot_data",
    "plot_2d_image",
    "plot_3d_slices",
    "plot_3d_isosurface",
    "IS_INTERACTIVE_AVAILABLE",
    "IS_FULL_INTERACTIVE",
]
