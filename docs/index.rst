CDIutils Documentation
======================

**CDIutils** is a Python package for Bragg Coherent X-ray Diffraction Imaging (BCDI) data processing, analysis, and visualization workflows.

The package is designed to handle the three primary stages of a BCDI data processing workflow:

* **Pre-processing** (data centering and cropping)
* **Phase retrieval** using PyNX for accurate phasing
* **Post-processing** (orthogonalization, phase manipulation, strain computation)

Key Features
------------

* **Flexibility in Hardware**: GPU support for phase retrieval, CPU support for pre/post-processing
* **Multiple Beamlines**: Support for various synchrotron beamlines (ID01, P10, SIXS, etc.)
* **Comprehensive Analysis**: Full toolkit for strain analysis, phase manipulation, and visualization
* **Publication-Ready Plots**: High-quality figures suitable for scientific publications

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Quick Start
===========

After :doc:`installation <installation>`, you can start using CDIutils:

.. code-block:: python

   import cdiutils
   from cdiutils import Geometry
   
   # Create a geometry object for a specific beamline
   geom = Geometry.from_beamline('P10_PETRA_III')
   
   # Set sample orientation
   geom.sample_orientation = 'horizontal'
   
   # Generate notebook templates
   from cdiutils.scripts import prepare_bcdi_notebooks
   prepare_bcdi_notebooks.main()

Getting Help
============

* **Documentation**: https://cdiutils.readthedocs.io/
* **Issues & Bug Reports**: https://github.com/clatlan/cdiutils/issues
* **Source Code**: https://github.com/clatlan/cdiutils
* **PyPI Package**: https://pypi.org/project/cdiutils/

Citation
========

If you use CDIutils in your research, please cite:

.. code-block:: bibtex

   @software{cdiutils,
     author = {Atlan, Clement and others},
     title = {CDIutils: A Python package for Bragg Coherent X-ray Diffraction Imaging},
     url = {https://github.com/clatlan/cdiutils},
     version = {0.2.0},
     year = {2024}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
