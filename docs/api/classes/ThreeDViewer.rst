ThreeDViewer (3D Interactive)
=============================

.. note::
   This page documents the interactive 3D viewer (Plotly). For 2D slice
   plotting, see :func:`cdiutils.plot.plot_volume_slices`.

.. currentmodule:: cdiutils.interactive

.. autoclass:: ThreeDViewer
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Interactive 3D visualisation in a notebook::

    from IPython.display import display
    from cdiutils.interactive import ThreeDViewer

    # reconstructed_object: complex 3D array
    viewer = ThreeDViewer(reconstructed_object, voxel_size=(10, 10, 10))
    display(viewer)

See Also
--------
:func:`plot_3d_isosurface` : quick isosurface plot
:class:`Plotter` : static and layered plots of files and arrays
