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
   :caption: Getting Started

   installation

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

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

   # create a loader object for a specific experiment
   loader = cdiutils.Loader.from_setup(
       "id01",
       experiment_file_path="path/to/experiment_file_path.h5",
       sample_name="sample_name"
   )

   # load detector scan data
   scan = 1  # specify the scan number
   detector_data = loader.load_detector_data(scan)
   angles = loader.load_motor_positions(scan)
   energy = loader.load_energy(scan)
   det_calib_params = loader.load_det_calib_params(scan)

   # load the geometry information
   geometry = cdiutils.Geometry.from_setup("id01")

   # initialise the space converter for data transformation
   converter = cdiutils.SpaceConverter(geometry, det_calib_params, energy=energy)
   converter.init_q_space(**angles)

   # convert the detector data to lab frame
   ortho_data = converter.orthogonalise_to_q_lab(detector_data)

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
