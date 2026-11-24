Interactive Module
==================

The :mod:`cdiutils.interactive` module provides interactive 3D visualisation tools.

.. currentmodule:: cdiutils.interactive

Classes
-------

:class:`ThreeDViewer` (:doc:`classes/ThreeDViewer`),
:class:`MultiVolumeViewer` (:doc:`classes/MultiVolumeViewer`) and
:class:`Plotter` (:doc:`classes/Plotter`) have their own pages.

.. autosummary::

   ThreeDViewer
   MultiVolumeViewer
   Plotter

.. autoclass:: VolumeViewer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TabPlotData
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autosummary::
   :toctree: ../generated/

   plot_3d_isosurface
   plot_data
   plot_2d_image
   plot_3d_slices

See Also
--------
:doc:`classes/ThreeDViewer` : 3D isosurface viewer documentation
:doc:`classes/MultiVolumeViewer` : multi-layer 3D viewer documentation
:doc:`classes/Plotter` : 2D slice viewer documentation
:doc:`plot` : Static plotting functions
