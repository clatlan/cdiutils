Contributing
============

We welcome contributions to CdiUtils! This guide will help you get started.

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

   git clone https://github.com/yourusername/cdiutils.git
   cd cdiutils

3. Install in development mode:

.. code-block:: bash

   pip install -e ".[dev]"

Running Tests
-------------

Run the test suite with:

.. code-block:: bash

   pytest tests/

For coverage reports:

.. code-block:: bash

   pytest --cov=cdiutils tests/

Code Style
----------

We follow PEP 8 style guidelines. Please ensure your code is formatted properly:

.. code-block:: bash

   black src/
   flake8 src/

Documentation
-------------

Documentation is built with Sphinx. To build the docs locally:

.. code-block:: bash

   cd docs/
   make html

Submitting Changes
------------------

1. Create a new branch for your feature or bugfix
2. Make your changes and add tests
3. Ensure all tests pass
4. Submit a pull request

Bug Reports
-----------

Please use the GitHub issue tracker to report bugs or request features.
