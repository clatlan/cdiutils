Contributing to CDIutils
========================

Thank you for your interest in contributing to CDIutils! This document provides guidelines for contributing to the project.

Getting Started
---------------

1. **Fork the repository** on GitHub
2. **Clone your fork** locally::

    git clone https://github.com/yourusername/cdiutils.git
    cd cdiutils

3. **Create a development environment**::

    conda env create -f environment-dev.yml
    conda activate cdiutils-dev

4. **Install in development mode**::

    pip install -e .

Development Workflow
--------------------

1. **Create a feature branch**::

    git checkout -b feature/your-feature-name

2. **Make your changes** with appropriate tests
3. **Run the test suite**::

    pytest tests/

4. **Update documentation** if needed
5. **Commit your changes** with descriptive messages::

    git commit -m "Add feature: describe your changes"

6. **Push to your fork**::

    git push origin feature/your-feature-name

7. **Create a pull request** on GitHub

Code Standards
--------------

* Follow PEP 8 style guidelines
* Add docstrings to all public functions and classes
* Include type hints for function parameters and return values
* Write unit tests for new functionality
* Update documentation for API changes

Testing
-------

Run the test suite before submitting changes::

    pytest tests/

Add tests for new features in the ``tests/`` directory.

Documentation
-------------

Build documentation locally::

    cd docs/
    make html

The built documentation will be in ``docs/build/html/``.

Reporting Issues
----------------

* Use the GitHub issue tracker
* Include a minimal reproducible example
* Specify your Python and CDIutils versions
* Describe expected vs. actual behaviour

Community Guidelines
--------------------

* Be respectful and constructive
* Help others learn and contribute
* Follow the project's code of conduct

Questions?
----------

* Open an issue for bug reports or feature requests
* Contact the maintainers for development questions
* Check existing issues before creating new ones

Thank you for contributing to CDIutils!
