Changelog
=========

This document tracks all notable changes to CDIutils.

Unreleased
----------

**MultiVolumeViewer refactor** (``cdiutils.interactive.multiviewer_3d``):

* Split into the ``cdiutils.interactive.multiviewer`` subpackage
  (``geometry``, ``analysis``, ``rendering``, ``export``, ``ui``, ``tools``,
  ``anim``, ``actions``, ``help``); ``multiviewer_3d`` keeps the class, and
  the import path, the constructor and ``Plotter(..., plot="layers")`` are
  unchanged.
* One trace builder for the live view, animation exports and mesh files:
  exports keep layer transforms, clip geometry and colour ranges; exports no
  longer modify the live layers.
* Geometry caching and in-place trace updates; derived layers reference
  volumes instead of copying them.
* Rendering backend tested lazily on first export; ``imageio-ffmpeg`` only
  needed for MP4; ``rendering_mode`` respected by every export; VTK >= 9.2
  compatibility.
* New: several conditions with a slider, difference and surface-shell/core
  layers, smooth planar clip caps, facet-family colouring and statistics,
  probe clicks and line profiles, orthoslices, dislocation lines, arrows,
  crystal-axes triad and scale bar, linked cameras, mesh export
  (STL/PLY/OBJ/VTK), JSON save/load of the view, iso-level and layer-normal
  animations, PNG-frame export.
* ``MultiVolumeViewer`` is exported from ``cdiutils.interactive``.
* Scripted actions: short step format (``{"show": "phase"}``), whole-script
  check with suggestions before running, idempotent layer creation, style keys
  on create, camera presets, chained exports, live report, YAML/JSON scripts,
  ``script()`` builder, ``to_actions()`` / ``save_actions()``; former action
  names still work.
* Built-in ``help(topic)`` and a guided ``tutorial()`` on the user's data.
* Built-in software renderer (``multiviewer.raster`` + ``multiviewer.overlay``,
  numpy + matplotlib) is the default export backend: stills and animations
  work on any server without a browser, display or GPU, with Plotly's camera,
  axis padding and lighting model. Kaleido and VTK are optional and fall back
  to it when unavailable. Export options *Smooth edges* (2x rendering) and
  *Flat colours* (one colour per triangle); renderer selector in the panel.
* Lighter exports: GIF written with Pillow and MP4 with the ffmpeg binary of
  ``imageio-ffmpeg`` (no system ffmpeg needed). With
  ``rendering_mode="fast"`` the render workers stay alive between exports,
  receive the scene once per export and then only the per-frame change
  (camera or swept layer); decoded triangles of unchanged layers are reused
  (also in ``safe`` mode). Workers stop with ``viewer.close()``.
* Colour ranges: presets ``minmax``, ``p1`` / ``p5`` (percentiles),
  ``sym`` / ``sym99`` (symmetric around zero) and ``manual``, in the panel
  and in scripts (``{"edit": "strain", "range": "sym"}``); *same for all
  conditions* shares one range across conditions. An explicit manual range
  widens the slider bounds.
* Units on colorbars: ``field_units=`` / ``set_units()`` / ``{"units": ...}``
  with an optional display scale (``("%", 100)``); ``rad`` by default for
  phases; derived layers use their source's unit; kept in state files and
  action scripts. New help topic ``help("colours")``.
* Switching conditions keeps the geometry already computed for the others.
* Animation through conditions: ``start_condition_export()``, panel mode
  *Conditions* and script action ``{"conditions": "all", "hold": 20}``;
  frames held per condition are rendered once; shared colour range and
  fixed axes by default; optional camera orbit. The condition label
  (``{label}``, ``{index}``, ``{n}``) is written on condition animations and,
  with several conditions, on stills and other animations.
* Software renderer: long colorbar titles are wrapped/shrunk to fit.
* ``MultiVolumeViewer.from_cdiutils(paths)``: opens cdiutils results
  (``S*_postprocessed_data.cxi``, ``S*_structural_properties.npz`` or
  ``.vti``; a file, scan folder or folder of scans); several scans become
  conditions (common grid, labels from scan, a parameter or given); voxel
  size, pipeline isosurface and cdiutils units (nm, %, Å, rad) are used;
  values outside the support masked; the configuration is printed first.
* ``from_cdiutils(scans_as="layers")``: every scan as its own layers
  (``"S1 | amplitude"``), side by side in a grid or overlaid, e.g.
  different particles; ``source="diffraction"``: measured intensity of
  ``S*_preprocessed_data.cxi`` (detector frame or orthogonalised q grid,
  log/sqrt/linear); ``label_from="sample"`` (``Cu_BCDI S1``). Repeated
  default labels (``S1`` in several samples) get the sample name. Scans
  with different voxel sizes are resampled (linear, NaN-aware) to the
  finest one (``resample=False`` to keep the arrays).
* ``fit_aspect()``: true proportions sized to the view (Plotly's default
  switches to a cube when an axis is >4x longer); used for side-by-side
  layers.
* Shell/core panel: "automatic" is now 0 instead of NaN (NaN values broke
  the widget messages to the browser).
* Responsiveness: in a notebook, panel edits (iso level, slice position,
  clip offset, transforms, visibility, condition slider) are computed in a
  worker thread; rapid edits are merged and the view is updated once
  ("updating…" shown meanwhile; ``await viewer.wait_for_refresh()``).
  Scripts and API calls stay synchronous (``background=False`` disables).
  Colormap and colour-range changes update the trace in place without
  recomputing or resending the geometry. Caches of isosurfaces and slices
  are bounded in memory (``cache_mb=2048``; ``cache_info()``,
  ``clear_caches()``).
* New layout: toolbar (camera views, grid, theme, rotation, condition
  slider, figure size preset, quick PNG / HTML, hide panel), status bar,
  and a tabbed panel: *Layers* (list with visibility, colormap swatch and
  type; click to edit), *Add* (tiles), *Analyze*, *Export*, *Script* (the
  view as an editable script: check, run, open, save) and *Help*.
  ``panel_side`` / ``panel_width``; ``figure_preset`` / ``figure_presets``
  and ``set_figure_preset()`` (screen, paper 1200x900, slide 1600x900,
  poster 2400x1800). Panel colours follow the Jupyter theme.
* Facet patches: edge-connected faces of the same crystal plane form a
  patch; ``facet_patches()`` gives area, area fraction, mean normal and its
  deviation, (hkl) of the mean normal (``cdiutils.facetanalysis`` rule),
  centre and field mean / std per patch; ``color_by="__patch__"``, (hkl)
  labels on the facets facing the camera, ``save_facet_csv()``, the
  ``{"patches": ...}`` action and *group by: patches* in the Analyze tab.
* Example notebook ``examples/multivolume_viewer.ipynb`` (synthetic data).
* ``MultiVolumeViewer.launcher(folder)``: notebook form (results, fields,
  labels, voxel size, units, options, start script) that opens the viewer
  in its place, with *Show configuration* and *Back*.
* ``export_html()``, *Save HTML* button and ``{"html": ...}`` action:
  interactive Plotly page, optionally with a slider/play over conditions.
* Isosurfaces coloured by a field defined only inside the support (cdiutils
  strain, phase): interpolation ignores undefined corners, so surface
  colours are no longer lost on the support edge.

Version 0.2.0 (Current)
------------------------

**Released:** January 2025

**New Features:**

* **Interactive 3D Visualisation Overhaul:**
  
  * ``ThreeDViewer`` migrated from ipyvolume to Plotly for modern, performant 3D rendering
  * New ``plot_3d_isosurface`` function for quick interactive isosurface visualisation
  * Support for 20+ colormaps including perceptually uniform options (viridis, turbo, colorcet)
  * Advanced colorbar controls: auto-scale, symmetric mode, manual limits (vmin/vmax)
  * NaN handling: replace NaN values with mean to avoid visualisation artefacts
  * Rotation animation and theme toggle (light/dark) in ``ThreeDViewer``
  * Improved zoom sensitivity and camera controls
  
* Comprehensive API documentation with Sphinx
* Professional PyData documentation theme
* Tutorial notebooks for BCDI workflows
* Template notebooks for common analysis tasks
* Improved type hints throughout the codebase

**Improvements:**

* **Better Dependency Management:**
  
  * Optional dependencies now properly organised: ``[interactive]``, ``[pyvista]``, ``[vtk]``
  * Graceful fallback when optional packages are not installed
  * Clear error messages indicating which packages to install
  
* **Code Quality:**
  
  * Google-style docstrings throughout
  * Type hints for all public functions
  * 79-character line length for better readability
  
* Better error handling and validation
* Optimised memory usage for large datasets
* Improved CXI file handling
* Enhanced strain analysis algorithms
* Better integration with PyNX and other tools

**Bug Fixes:**

* Fixed normalise/normalize function naming consistency
* Resolved import issues with optional dependencies
* Fixed documentation build configuration
* Corrected type annotation compatibility
* Fixed colorbar tick label visibility in 3D plots
* Resolved NaN propagation issues in min/max calculations

**Documentation:**

* Complete API reference documentation
* Step-by-step tutorials for beginners
* Example notebooks with real datasets including ``interactive_features.ipynb``
* Installation and setup guides with optional dependency information
* Contributing guidelines

**Breaking Changes:**

* ``ThreeDViewer`` now requires ``plotly``, ``scikit-image``, and ``scipy`` instead of ``ipyvolume``
* Install with: ``pip install cdiutils[interactive]`` for full interactive functionality

Version 0.1.x
--------------

**Initial Development Releases**

* Core functionality for BCDI analysis
* Basic plotting and visualisation tools
* CXI file format support
* Integration with synchrotron beamlines
* Phase retrieval post-processing
* Strain analysis capabilities

Development Notes
-----------------

CDIutils follows semantic versioning (SemVer). 

* **Major versions** (x.0.0): Breaking API changes
* **Minor versions** (0.x.0): New features, backwards compatible
* **Patch versions** (0.0.x): Bug fixes, backwards compatible

For the complete development history, see the `GitHub commit log <https://github.com/clatlan/cdiutils/commits/master>`_.

Future Roadmap
--------------

* Additional beamline support
* Machine learning integration
* Real-time analysis capabilities
* Cloud computing integration
* Advanced visualisation features
