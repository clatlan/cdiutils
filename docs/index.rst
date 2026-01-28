CDIutils Documentation
======================

**CDIutils** is a Python package for Bragg Coherent X-ray Diffraction Imaging (BCDI) data processing, analysis, and visualization workflows.

The package is designed to handle the three primary stages of a BCDI data processing workflow:

* **Pre-processing** (data centering and cropping)
* **Phase retrieval** using PyNX for accurate phasing
* **Post-processing** (orthogonalization, phase manipulation, strain computation)

Key Features
------------

* **Modern Interactive 3D Visualisation**: Plotly-based tools (``ThreeDViewer``, ``plot_3d_isosurface``) with extensive controls for colormaps, colorbar settings, and NaN handling
* **Flexibility in Hardware**: GPU support for phase retrieval, CPU support for pre/post-processing
* **Multiple Beamlines**: Support for various synchrotron beamlines (ID01, P10, SIXS, etc.)
* **Comprehensive Analysis**: Full toolkit for strain analysis, phase manipulation, and visualization
* **Publication-Ready Plots**: High-quality figures suitable for scientific publications
* **Optional Dependencies**: Modular installation - install only what you need (``[interactive]``, ``[pyvista]``, ``[vtk]``)

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   getting_started/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

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

Get started with a minimal working example in a Jupyter notebook:

.. code-block:: python

   import os
   import cdiutils
   
   # Setup parameters
   beamline_setup = "id01"
   experiment_file_path = "/path/to/file.h5"
   sample_name = "MySample"
   scan = 42
   dump_dir = os.getcwd() + f"/results/{sample_name}/S{scan}/"
   
   # Create pipeline
   params = cdiutils.pipeline.get_params_from_variables(dir(), globals())
   pipeline = cdiutils.BcdiPipeline(params=params)
   
   # Run workflow
   pipeline.preprocess(preprocess_shape=(200, 200, 200))
   pipeline.phase_retrieval(nb_run=50)
   pipeline.analyse_phasing_results()
   pipeline.mode_decomposition()
   pipeline.postprocess(voxel_size=5)
   
   # Visualize
   pipeline.show_3d_final_result()

See :doc:`getting_started/quickstart` for detailed explanation.

For complete examples with real data, check :doc:`examples/index`.

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
