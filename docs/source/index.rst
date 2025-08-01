.. CdiUtils documentation master file, created by
   sphinx-quickstart on Fri Feb 14 15:01:29 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CdiUtils Documentation
======================

CdiUtils is a Python package for Coherent Diffraction Imaging (CDI) data processing and analysis.

Features
--------

* Coherent X-ray diffraction data processing
* Support for multiple beamline configurations (ID01, P10, SIXS, etc.)
* Conversion between CXI and XU coordinate conventions
* Geometry handling for diffraction experiments
* Data visualization and analysis tools

Quick Start
-----------

Install CdiUtils using pip:

.. code-block:: bash

   pip install cdiutils

Or for development:

.. code-block:: bash

   git clone https://github.com/clatlan/cdiutils.git
   cd cdiutils
   pip install -e .

Basic Usage
-----------

.. code-block:: python

   from cdiutils.geometry import Geometry
   
   # Create a geometry for ID01 beamline
   geometry = Geometry.from_setup("id01", sample_orientation="horizontal")
   
   # Display geometry information
   print(geometry)

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development:

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

