VolumeViewer
============

.. currentmodule:: cdiutils.interactive

.. autoclass:: VolumeViewer
   :members:
   :undoc-members:
   :show-inheritance:

   The VolumeViewer class provides an interactive 3D isosurface viewer
   for volumetric data using Plotly.

Examples
--------

Interactive 3D visualization::

    from cdiutils.interactive import VolumeViewer
    import numpy as np

    # Create viewer with data
    viewer = VolumeViewer(
        data=reconstructed_object,
        voxel_size=(10e-9, 10e-9, 10e-9)  # in metres
    )

    # Show interactive widget in notebook
    viewer.show()

See Also
--------
:func:`cdiutils.plot.plot_volume_slices` : 2D slice plotting
:class:`CXIExplorer` : CXI file explorer
