CDIutils Documentation
======================

**CDIutils** is a Python package for Bragg Coherent X-ray Diffraction Imaging (BCDI) data analysis.

Main Capabilities
-----------------

**Complete BCDI pipeline**
  Handle pre-processing, phasing (PyNX backend), and post-processing to extract quantitative strain information. Jupyter notebooks provide step-by-step workflows:
  
  * :download:`bcdi_pipeline_example.ipynb <bcdi_pipeline_example.ipynb>`
  * :download:`step_by_step_bcdi_analysis.ipynb <../step_by_step_bcdi_analysis.ipynb>`

**Multiple beamline geometries**
  Support for ID01, P10, SIXS, NanoMAX, and ID27 beamlines. Coordinate transformations use :doc:`CXI convention <user_guide/coordinate_systems>` and xrayutilities backends.

**Publication-ready figures**
  Utility functions for creating publication-quality plots. See :download:`bcdi_reconstruction_analysis.ipynb <bcdi_reconstruction_analysis.ipynb>` for examples.

**Interactive 3D visualisation**
  Tools for exploring reconstruction results interactively. See :doc:`api/interactive` for available classes.

**CXI file management**
  :class:`~cdiutils.io.CXIFile` manager simplifies CXI file creation. :class:`~cdiutils.io.CXIExplorer` provides interactive inspection of CXI files

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

Minimal working example:

.. code-block:: python

   import cdiutils
   
   # define parameters
   params = cdiutils.pipeline.get_params_from_variables(
       beamline_setup="id01",
       experiment_file_path="/path/to/data.h5",
       sample_name="MySample",
       scan=42
   )
   
   # create and run pipeline
   pipeline = cdiutils.BcdiPipeline(params=params)
   pipeline.preprocess(preprocess_shape=(200, 200, 200))
   pipeline.phase_retrieval(nb_run=50)
   pipeline.postprocess(voxel_size=5)
   pipeline.show_3d_final_result()

See :doc:`getting_started/quickstart` for details.

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
