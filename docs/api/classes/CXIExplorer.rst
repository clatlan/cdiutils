CXIExplorer
===========

.. currentmodule:: cdiutils.io

.. autoclass:: CXIExplorer
   :members:
   :undoc-members:
   :show-inheritance:

   The CXIExplorer class provides an interactive widget for exploring CXI files.
   See the CXI explorer tutorial for usage examples.

Examples
--------

Exploring CXI file::

    from cdiutils.io import CXIExplorer

    # Create explorer (opens file automatically)
    explorer = CXIExplorer("results.cxi")

    # Show interactive widget in Jupyter notebook
    # This displays file tree and interactive data viewer
    explorer.show()
    
    # Close when done
    explorer.close()
    
    # Or use context manager
    with CXIExplorer("results.cxi") as explorer:
        explorer.show()

See Also
--------
:class:`CXIFile` : Low-level CXI file access
:func:`cdiutils.plot.plot_volume_slices` : 2D slice viewer for arrays
