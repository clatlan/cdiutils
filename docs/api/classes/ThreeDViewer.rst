VolumeViewer (3D Interactive)
==============================

.. note::
   This page documents the interactive 3D viewer. For 2D slice plotting,
   see :func:`cdiutils.plot.plot_volume_slices`.

.. currentmodule:: cdiutils.interactive

.. autoclass:: VolumeViewer
   :members:
   :undoc-members:
   :show-inheritance:

   The VolumeViewer class provides interactive 3D visualisation of
   reconstructed objects with isosurface rendering using Plotly.

Examples
--------

Interactive 3D visualisation::

    from cdiutils.interactive import VolumeViewer
    import numpy as np

    # Create viewer
    viewer = VolumeViewer(
        data=reconstructed_object,
        voxel_size=(10e-9, 10e-9, 10e-9)  # metres
    )

    # Show interactive widget (in Jupyter)
    viewer.show()

See Also
--------
:func:`cdiutils.plot.plot_3d_surface_projections` : Static matplotlib plots
:class:`VolumeViewer` : Same class, interactive 3D viewer
