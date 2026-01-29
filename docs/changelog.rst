Changelog
=========

This document tracks all notable changes to CDIutils.

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
