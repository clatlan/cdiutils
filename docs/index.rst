CdiUtils Documentation
=====================

CdiUtils is a Python package for Coherent Diffraction Imaging (CDI) data analysis and visualization.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   api

Installation
============

Install CdiUtils using pip:

.. code-block:: bash

   pip install cdiutils

Or from source:

.. code-block:: bash

   git clone https://github.com/clatlan/cdiutils.git
   cd cdiutils
   pip install -e .

Quick Start
===========

.. code-block:: python

   from cdiutils import Geometry
   
   # Create a geometry object for a specific beamline
   geom = Geometry.from_beamline('P10_PETRA_III')
   
   # Set sample orientation
   geom.sample_orientation = 'horizontal'

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
