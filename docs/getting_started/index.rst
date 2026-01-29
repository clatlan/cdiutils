Getting Started
===============

Welcome to CDIutils! This section will help you get up and running quickly
with Bragg Coherent X-ray Diffraction Imaging (BCDI) data analysis.

.. toctree::
   :maxdepth: 2

   quickstart
   concepts
   first_analysis

What is BCDI?
-------------

Bragg Coherent Diffraction Imaging is a lensless imaging technique that
provides 3D maps of lattice displacement and strain in crystalline
nanostructures with nanometre spatial resolution.

CDIutils provides:

* **Automated workflows** via :class:`~cdiutils.pipeline.BcdiPipeline`
* **Beamline-specific data loaders** (ID01, P10, SIXS, NanoMAX, etc.)
* **Coordinate system handling** for multiple experimental geometries
* **Phase retrieval integration** with PyNX
* **Post-processing tools** for strain, displacement analysis
* **Interactive visualisation** for 3D data exploration

Quick Links
-----------

**New users:**
  Start with :doc:`quickstart` for a 5-minute example

**Example notebooks:**
  See :doc:`../examples/index` for interactive Jupyter examples

**API reference:**
  Browse :doc:`../api` for detailed class documentation

**BCDI concepts:**
  Read :doc:`concepts` for coordinate systems and reciprocal space

Next Steps
----------

1. :doc:`../installation` - Install CDIutils and dependencies
2. :doc:`quickstart` - Run your first analysis in 5 minutes
3. :doc:`concepts` - Understand BCDI workflow and conventions
4. :doc:`first_analysis` - Complete analysis from raw data to strain maps
