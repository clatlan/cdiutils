Installation
============

Requirements
------------

CdiUtils requires Python 3.10 or higher and the following packages:

* numpy
* scipy
* matplotlib
* h5py
* xrayutilities
* fabio

Install from PyPI
-----------------

.. code-block:: bash

   pip install cdiutils

Install from Source
-------------------

To install the latest development version:

.. code-block:: bash

   git clone https://github.com/clatlan/cdiutils.git
   cd cdiutils
   pip install -e .

Development Installation
------------------------

For development, also install the test and documentation dependencies:

.. code-block:: bash

   git clone https://github.com/clatlan/cdiutils.git
   cd cdiutils
   pip install -e ".[dev]"

Testing Installation
--------------------

To verify your installation, run the test suite:

.. code-block:: bash

   pytest tests/

Conda Environment
-----------------

For a complete environment, you can use conda:

.. code-block:: bash

   conda create -n cdiutils python>=3.10
   conda activate cdiutils
   pip install cdiutils
