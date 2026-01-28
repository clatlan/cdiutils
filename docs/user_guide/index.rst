User Guide
==========

Comprehensive guides for using CDIutils in your BCDI experiments.

.. toctree::
   :maxdepth: 2
   :caption: Core Workflows

   pipeline
   beamlines
   coordinate_systems
   wavefront_analysis

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   detector_calibration
   phase_retrieval_tuning
   reciprocal_space_gridding
   strain_analysis
   facet_analysis

.. toctree::
   :maxdepth: 1
   :caption: Reference

   troubleshooting
   performance_tips
   data_formats

Overview
--------

This user guide is organized by workflow:

**For typical users:**
  :doc:`pipeline` covers the automated :class:`~cdiutils.pipeline.BcdiPipeline`

**For beamline scientists:**
  :doc:`beamlines` explains beamline-specific configurations

**For manual processing:**
  :doc:`coordinate_systems` shows how to use converters directly

**For optimizing results:**
  :doc:`phase_retrieval_tuning` and :doc:`detector_calibration`

Quick Links by Task
-------------------

**"How do I..."**

* Load data from my beamline? → :doc:`beamlines`
* Set up detector calibration? → :doc:`detector_calibration`  
* Improve phase retrieval quality? → :doc:`phase_retrieval_tuning`
* Handle coordinate transformations? → :doc:`coordinate_systems`
* Analyse the reconstructed probe? → :doc:`wavefront_analysis`
* Calculate strain correctly? → :doc:`strain_analysis`
* Find crystallographic facets? → :doc:`facet_analysis`
* Reduce memory usage? → :doc:`performance_tips`
* Debug failed reconstructions? → :doc:`troubleshooting`
