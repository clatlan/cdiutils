Installation
============

CDIutils requires Python 3.10 or later and has been tested on Linux, macOS, and Windows.

.. _pypi_install:

Install from PyPI (Recommended)
--------------------------------

The simplest way to install CDIutils is from PyPI:

.. code-block:: bash

   pip install cdiutils

.. _conda_install:

Install with Conda (Recommended for Dependency Management)
-----------------------------------------------------------

For better dependency management using conda's solver, you have two options:

**Option 1: Create a new conda environment**

.. code-block:: bash

   # Create conda environment directly from GitHub
   conda env create -f https://raw.githubusercontent.com/clatlan/cdiutils/master/environment.yml
   
   # Activate the environment
   conda activate cdiutils-env
   
   # Install CDIutils from PyPI
   pip install cdiutils

**Option 2: Install dependencies in your existing conda environment**

.. code-block:: bash

   # Download and install dependencies using conda solver
   conda env update -f https://raw.githubusercontent.com/clatlan/cdiutils/master/environment.yml
   
   # Install CDIutils from PyPI
   pip install cdiutils

.. _development_install:

Development Installation
------------------------

For developers who want to contribute or use the latest features:

.. code-block:: bash

   # Create development environment directly from GitHub
   conda env create -f https://raw.githubusercontent.com/clatlan/cdiutils/master/environment-dev.yml
   
   # Activate the environment
   conda activate cdiutils-dev-env
   
   # Clone and install in development mode
   git clone https://github.com/clatlan/cdiutils.git
   cd cdiutils
   pip install -e .

.. _github_install:

Install from GitHub (Latest Development Version)
-------------------------------------------------

To get the latest development version directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/clatlan/cdiutils.git

.. note::
   The development version may contain new features but is not guaranteed to be stable.

Dependencies
------------

CDIutils has a minimal set of core dependencies for basic functionality. Additional features require optional dependencies.

Core Dependencies
~~~~~~~~~~~~~~~~~

* numpy
* scipy >= 1.8.0
* matplotlib >= 3.10
* h5py >= 3.6.0
* hdf5plugin >= 3.2.0
* pandas >= 1.4.2
* scikit-image >= 0.19.2
* scikit-learn >= 1.1.3
* silx
* xrayutilities >= 1.7.3
* colorcet >= 3.0.0
* tabulate

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

Install additional features using pip extras:

**Interactive Visualisation** (``pip install cdiutils[interactive]``):

* ipywidgets - Interactive widgets for Jupyter notebooks
* ipyvolume - Legacy 3D visualisation support
* bokeh - Interactive 2D plotting
* panel - Dashboard creation
* plotly >= 5.0.0 - Modern 3D visualisation (``ThreeDViewer``, ``plot_3d_isosurface``)
* kaleido >= 0.2.0 - Static image export for Plotly
* h5glance - Interactive HDF5 file browser
* tornado, pytables, anywidget, ipykernel - Supporting packages

**PyVista 3D Rendering** (``pip install cdiutils[pyvista]``):

* pyvista >= 0.43.0 - Advanced 3D visualisation
* trame >= 3.0.0 - Web-based rendering framework
* trame-vuetify >= 2.0.0 - UI components
* trame-vtk >= 2.0.0 - VTK integration

**VTK Support** (``pip install cdiutils[vtk]``):

* vtk >= 9.0.1 - Visualisation Toolkit

**All Optional Dependencies** (``pip install cdiutils[all]``):

Installs all optional dependencies for full functionality.

For a complete list of dependencies and versions, see the `pyproject.toml <https://github.com/clatlan/cdiutils/blob/master/pyproject.toml>`_ file.

Verification
------------

To verify your installation, run:

.. code-block:: python

   import cdiutils
   print(cdiutils.__version__)

You can also test the command-line tools:

.. code-block:: bash

   prepare_bcdi_notebooks --help
   prepare_detector_calibration --help
