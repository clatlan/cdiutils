Plotter
=======

.. currentmodule:: cdiutils.interactive

.. autoclass:: Plotter
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Slices of a reconstruction stored in a file::

    from cdiutils.interactive import Plotter

    Plotter("S1_postprocessed_data.cxi", plot="slices")

See Also
--------
:class:`ThreeDViewer` : interactive 3D isosurface viewer
:func:`cdiutils.plot.plot_volume_slices` : 2D slice plotting
