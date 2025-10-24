"""
Interactive visualisation and data exploration tools for BCDI.

This package provides interactive widgets and plotting utilities for
Bragg Coherent Diffractive Imaging (BCDI) data analysis.

Components have different optional dependencies:
- Plotter: requires ipywidgets
- Plotting functions: require ipywidgets, bokeh, panel
- ThreeDViewer: requires ipywidgets, ipyvolume, tornado
- TabPlotData: requires ipywidgets, h5glance

Install all with: pip install cdiutils[interactive]
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

# 3D visualisation dependencies (for viewer_3d.py)
try:
    import ipyvolume  # noqa: F401
    import tornado  # noqa: F401

    _DEPS["3d"] = True
except ImportError:
    _DEPS["3d"] = False

# data browser dependencies (for data_browser.py)
try:
    import h5glance  # noqa: F401

    _DEPS["browser"] = True
except ImportError:
    _DEPS["browser"] = False

# convenience flags
IS_INTERACTIVE_AVAILABLE = _DEPS["ipywidgets"]
IS_FULL_INTERACTIVE = all(_DEPS.values())


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

    plot_2d_image = _make_unavailable_func("plot_2d_image", missing)
    plot_3d_slices = _make_unavailable_func("plot_3d_slices", missing)
    plot_data = _make_unavailable_func("plot_data", missing)

# ThreeDViewer - requires ipywidgets, ipyvolume, tornado
if _DEPS["ipywidgets"] and _DEPS["3d"]:
    from .viewer_3d import ThreeDViewer
else:
    missing = []
    if not _DEPS["ipywidgets"]:
        missing.append("ipywidgets")
    if not _DEPS["3d"]:
        missing.extend(["ipyvolume", "tornado"])
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


__all__ = [
    "TabPlotData",
    "Plotter",
    "ThreeDViewer",
    "plot_data",
    "plot_2d_image",
    "plot_3d_slices",
    "IS_INTERACTIVE_AVAILABLE",
    "IS_FULL_INTERACTIVE",
]
