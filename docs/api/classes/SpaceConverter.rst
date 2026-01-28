SpaceConverter
==============

.. currentmodule:: cdiutils

.. autoclass:: SpaceConverter
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Coordinate Transformations

   .. autosummary::
      ~SpaceConverter.detector_to_lab
      ~SpaceConverter.lab_to_detector
      ~SpaceConverter.reciprocal_to_direct
      ~SpaceConverter.direct_to_reciprocal

   .. rubric:: Utility Methods

   .. autosummary::
      ~SpaceConverter.get_q_range
      ~SpaceConverter.get_voxel_size

See Also
--------
:class:`Geometry` : Beamline geometry configuration
:class:`BcdiPipeline` : Uses SpaceConverter for coordinate transformations
