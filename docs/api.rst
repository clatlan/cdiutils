API Reference
=============

This page provides complete documentation for all public classes and functions in CDIutils.

.. contents:: Quick Navigation
   :local:
   :depth: 2

Overview
--------

CDIutils is organized into several key modules:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`api/geometry`
     - Beamline geometry and coordinate systems
   * - :doc:`api/converter`
     - Coordinate transformations between frames
   * - :doc:`api/pipeline`
     - Automated processing workflows
   * - :doc:`api/process`
     - Phase retrieval and post-processing
   * - :doc:`api/io`
     - Data loading and file I/O
   * - :doc:`api/plot`
     - Visualization and plotting tools
   * - :doc:`api/analysis`
     - Statistical analysis utilities
   * - :doc:`api/interactive`
     - Interactive 3D visualization
   * - :doc:`api/wavefront`
     - Wavefront propagation and probe analysis
   * - :doc:`api/simulation`
     - Data simulation tools

Core Classes
------------

Essential classes for BCDI workflows:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Description
   * - :doc:`api/classes/Geometry`
     - Beamline geometry, sample orientation, coordinate conventions
   * - :doc:`api/classes/SpaceConverter`
     - Detector ↔ reciprocal ↔ direct space transformations
   * - :doc:`api/classes/BcdiPipeline`
     - Complete BCDI pipeline (preprocessing → phasing → postprocessing)
   * - :doc:`api/classes/PyNXPhaser`
     - PyNX phase retrieval wrapper with result analysis
   * - :doc:`api/classes/PostProcessor`
     - Strain, displacement, phase manipulation tools


Data I/O Classes
----------------

Loading data from different beamlines:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Description
   * - :doc:`api/classes/Loader`
     - Base loader class with factory method
   * - :doc:`api/classes/ID01Loader`
     - ESRF ID01 beamline (BLISS format)
   * - :doc:`api/classes/P10Loader`
     - PETRA III P10 beamline
   * - :doc:`api/classes/SIXSLoader`
     - SOLEIL SIXS beamline
   * - :doc:`api/classes/NanoMaxLoader`
     - MAX IV NanoMAX beamline
   * - :doc:`api/classes/CXIFile`
     - CXI file format reader/writer
   * - :doc:`api/classes/CXIExplorer`
     - Interactive CXI file explorer widget

Interactive Visualization
-------------------------

3D interactive tools:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class/Function
     - Description
   * - :doc:`api/classes/ThreeDViewer`
     - Advanced 3D isosurface viewer with full controls
   * - :doc:`api/classes/Plotter`
     - Interactive 2D slice viewer widget
   * - :doc:`api/functions/plotting`
     - Quick 3D visualization function

Class Reference
---------------

Detailed documentation by class:

.. toctree::
   :maxdepth: 1
   :caption: Core Classes

   api/classes/Geometry
   api/classes/SpaceConverter
   api/classes/BcdiPipeline
   api/classes/Pipeline
   api/classes/PyNXPhaser
   api/classes/PostProcessor
   api/classes/PhasingResultAnalyser

.. toctree::
   :maxdepth: 1
   :caption: I/O Classes

   api/classes/Loader
   api/classes/ID01Loader
   api/classes/P10Loader
   api/classes/SIXSLoader
   api/classes/NanoMaxLoader
   api/classes/CXIFile
   api/classes/CXIExplorer

.. toctree::
   :maxdepth: 1
   :caption: Interactive Classes

   api/classes/ThreeDViewer
   api/classes/Plotter

Function Reference
------------------

Key functions organized by category:

.. toctree::
   :maxdepth: 1
   :caption: By Category

   api/functions/plotting
   api/functions/utilities
   api/functions/analysis
   api/functions/io

Module Reference
----------------

Detailed documentation by module:

.. toctree::
   :maxdepth: 1
   :caption: By Module

   api/geometry
   api/converter
   api/pipeline
   api/process
   api/io
   api/plot
   api/analysis
   api/interactive
   api/wavefront
   api/simulation
   api/utils

Quick Links
-----------

**Most Common Use Cases:**

* **Loading data**: :doc:`api/classes/Loader`, :doc:`api/classes/ID01Loader`
* **Coordinate transformations**: :doc:`api/classes/SpaceConverter`
* **Full pipeline**: :doc:`api/classes/BcdiPipeline`
* **Phase retrieval**: :doc:`api/classes/PyNXPhaser`
* **Post-processing**: :doc:`api/classes/PostProcessor`
* **3D visualization**: :doc:`api/classes/ThreeDViewer`, :doc:`api/functions/plotting`
* **Plotting slices**: :doc:`api/functions/plotting`

Search
------

Use the search box in the sidebar to quickly find specific classes, functions, or parameters.

