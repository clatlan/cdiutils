CXI File Explorer Tutorial
==========================

This tutorial provides a simple example to show you how to use the CXI file browser. The ``CXIExplorer`` provides powerful tools for exploring the hierarchical structure of CXI files and understanding the data they contain.

Setup and Initialisation
-------------------------

Start by importing the necessary libraries and setting up plotting parameters:

.. code-block:: python

    import cdiutils

    cdiutils.plot.update_plot_params()

Set Up the File Path
---------------------

First, provide the path to the CXI file you want to explore:

.. code-block:: python

    # load the data
    # replace with the actual path to your data file
    path = "path/to/your/data.cxi"

Basic Usage
-----------

Create an explorer instance and print a summary of the file contents.

The ``CXIExplorer`` provides four main methods for exploring CXI files:

1. ``summarise()`` - Provides an overview of the file including file size, number of groups and datasets
2. ``tree(max_depth=None, show_attributes=False)`` - Displays the hierarchical structure of the file in a tree format
3. ``search(pattern)`` - Finds datasets, groups, or attributes matching a specific pattern
4. ``show(path)`` - Visualises a specific dataset, with automatic plotting for array data

Additionally, the ``explore()`` method launches an interactive widget-based browser:

.. code-block:: python

    explorer = cdiutils.io.CXIExplorer(path)

    explorer.summarise()

This provides an overview of the CXI file contents. For example:

.. code-block:: text

    CXI File Summary
    ================
    File: /path/to/your/data.cxi
    Size: 45.2 MB
    Groups: 12
    Datasets: 28
    Attributes: 15

View Hierarchical Structure
---------------------------

Display the file's hierarchical structure as a tree, limiting depth for clarity:

.. code-block:: python

    explorer.tree(max_depth=2)

This displays the file structure as a tree, showing groups and datasets. For example:

.. code-block:: text

    entry_1/
    ├── data_1/
    │   ├── data
    │   └── mask
    ├── instrument_1/
    │   ├── detector_1/
    │   └── source_1/
    └── result_1/
        ├── amplitude
        ├── phase
        └── support

Search Functionality
--------------------

Search for specific path using keywords.

Example: you know that the dataset you are looking for contains the word "strain" in its name:

.. code-block:: python

    explorer.search("strain")

Display Specific Datasets
--------------------------

View the contents of a specific dataset by its path:

.. code-block:: python

    explorer.show("entry_1/dspacing")

For numerical datasets, this automatically generates visualisation plots. For example, viewing strain data produces multi-slice plots showing different cross-sections of the 3D volume.

Interactive Exploration
------------------------

Launch an interactive browser to navigate through the file:

.. code-block:: python

    explorer.explore()

Closing the Explorer
--------------------

Always close the explorer when finished to release file resources. When you delete the explorer (``del explorer``), the file will also be closed automatically:

.. code-block:: python

    explorer.close()

Using Context Manager
---------------------

A cleaner approach is to use the context manager, which automatically closes the file when done. Note that using the context manager prevents using the interactive ``explore()`` method after exiting the context:

.. code-block:: python

    with cdiutils.io.CXIExplorer(path) as explorer:
        # print a summary of the file
        print("File summary:")
        explorer.summarise()

        # print the hierarchical tree structure
        print("\nTree:")
        explorer.tree(max_depth=1)

        # search for specific datasets
        print("\nSearch specific key word:")
        explorer.search("strain")

        # show a specific dataset
        print("\nShow specific dataset:")
        explorer.show("entry_1/dspacing")

Direct Use of CXIFile Class
----------------------------

The ``CXIExplorer`` can also be accessed directly from the ``CXIFile`` class using ``get_explorer()``, which returns a ready-to-use ``CXIExplorer`` instance:

.. code-block:: python

    cxi_file = cdiutils.CXIFile(path)

    # quick interactive exploration
    print("Summary:")
    cxi_file.get_explorer().summarise()

    # or:
    explorer = cxi_file.get_explorer()
    print("\nTree:")
    explorer.tree(max_depth=1)

    # when finished
    cxi_file.close()

Load Data from a CXI File
-------------------------

There are several ways to open a CXI file. The most common way is to use the ``CXIFile`` class, which provides a simple interface for reading and writing CXI files.

You can load data from a CXI file using the classic approach:

.. code-block:: python

    cxi_file = cdiutils.CXIFile(path)
    cxi_file.open()
    data = cxi_file["entry_1/amplitude"]
    cxi_file.close()  # don't forget to close the file!

Or you can use the context manager, i.e. the ``with`` statement to automatically close the file when done:

.. code-block:: python

    with cdiutils.CXIFile(path) as cxi_file:
        data = cxi_file["entry_1/amplitude"]

Finally, you can use the ``cdiutils.io.load_cxi`` function to conveniently load data from CXI files. The ``cdiutils`` library provides a convenient function to load data from CXI files. It requires the path to the CXI file and a dataset name to load. If the dataset name is not the exact full "key path", say ``"voxel_size"`` instead of ``"entry_1/result_1/voxel_size"``, the function will find it for you anyway. Note that you can provide as much as keys as you want, and the function will return a dictionary with the keys as the dataset names and the values as the data loaded from the CXI file:

.. code-block:: python

    voxel_size = cdiutils.io.load_cxi(path, "voxel_size")  # simple value

    data = cdiutils.io.load_cxi(path, "amplitude", "het_strain")  # dictionary

    print(data.keys())

Credits
-------

This notebook was created by Clément Atlan, ESRF, 2025. It is part of the ``cdiutils`` package, which provides tools for BCDI data analysis and visualisation.

If you have used this notebook or the ``cdiutils`` package in your research, please consider citing the package: https://github.com/clatlan/cdiutils/

You'll find the citation information in the ``cdiutils`` package documentation.

.. code-block:: bibtex

    @software{Atlan_Cdiutils_A_python,
    author = {Atlan, Clement},
    doi = {10.5281/zenodo.7656853},
    license = {MIT},
    title = {{Cdiutils: A python package for Bragg Coherent Diffraction Imaging processing, analysis and visualisation workflows}},
    url = {https://github.com/clatlan/cdiutils},
    version = {0.2.0}
    }
