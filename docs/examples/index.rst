Examples
========

This section contains practical examples and Jupyter notebooks demonstrating CDIutils functionality.

.. toctree::
   :maxdepth: 2
   :caption: Example Notebooks

   ../bcdi_pipeline_example
   ../interactive_features
   ../bcdi_reconstruction_analysis
   ../explore_cxi_file
   ../pole_figure

Jupyter Notebook Examples
--------------------------

The following Jupyter notebooks provide hands-on examples of CDIutils usage:

Interactive 3D Visualisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Download:** :download:`interactive_features.ipynb <../../examples/interactive_features.ipynb>`

**New in v0.2.0!** Learn to use the modern interactive 3D visualisation tools:

* ``plot_3d_isosurface`` - Quick interactive isosurface plotting with multiple quantities
* ``ThreeDViewer`` - Widget for exploring complex 3D arrays (amplitude + phase)
* ``TabPlotData`` - Interactive browser for CDI reconstruction results
* Colormap selection and colorbar control (auto-scale, symmetric, manual limits)
* NaN handling for clean visualisations
* Theme toggle and rotation animations

BCDI Pipeline Example
~~~~~~~~~~~~~~~~~~~~~

**Download:** :download:`bcdi_pipeline_example.ipynb <../../examples/bcdi_pipeline_example.ipynb>`

**Complete end-to-end pipeline** This comprehensive example demonstrates a full BCDI analysis workflow:

* Setting up and running the ``BcdiPipeline`` class
* Automated data processing and phase retrieval
* Post-processing and strain analysis
* **Interactive 3D visualisation** with ``plot_3d_isosurface`` (new in v0.2.0)
* Advanced plotting and figure generation

This example showcases the complete workflow from raw data to publication-ready visualisations, including the new Plotly-based interactive 3D plots.

BCDI Reconstruction Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Download:** :download:`bcdi_reconstruction_analysis.ipynb <../../examples/bcdi_reconstruction_analysis.ipynb>`

This notebook demonstrates comprehensive BCDI reconstruction analysis including:

* Loading experimental data
* Phase retrieval with PyNX
* Post-processing and strain analysis
* Visualisation and interpretation

CXI File Exploration
~~~~~~~~~~~~~~~~~~~~~

**Download:** :download:`explore_cxi_file.ipynb <../../examples/explore_cxi_file.ipynb>`

Learn how to:

* Navigate CXI file structure
* Extract reconstruction results
* Analyse metadata and experimental parameters
* Export data for further processing

Pole Figure Analysis
~~~~~~~~~~~~~~~~~~~~

**Download:** :download:`pole_figure.ipynb <../../examples/pole_figure.ipynb>`

This example covers:

* Stereographic projection analysis
* Crystal orientation mapping
* Pole figure generation and interpretation
* Advanced crystallographic analysis

Template Notebooks
-------------------

Ready-to-use templates are available in the `templates directory <https://github.com/clatlan/cdiutils/tree/master/src/cdiutils/templates>`_:

* **bcdi_pipeline.ipynb** - Complete BCDI analysis pipeline
* **step_by_step_bcdi_analysis.ipynb** - Detailed step-by-step reconstruction
* **detector_calibration.ipynb** - Detector calibration procedures

Getting Started
---------------

1. Download the notebook examples above
2. Install CDIutils following the :doc:`../installation` guide
3. Run the notebooks in your Jupyter environment
4. Adapt the examples to your specific datasets
