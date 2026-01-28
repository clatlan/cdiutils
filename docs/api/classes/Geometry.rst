Geometry
========

.. currentmodule:: cdiutils

.. autoclass:: Geometry
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Attributes

   .. autosummary::
      ~Geometry.directbeam
      ~Geometry.sample_offsets
      ~Geometry.sample_inplane_angle
      ~Geometry.sample_outofplane_angle
      ~Geometry.tilt_angle
      ~Geometry.wavelength
      ~Geometry.energy
      ~Geometry.distance
      ~Geometry.pixel_size_horizontal
      ~Geometry.pixel_size_vertical

   .. rubric:: Methods

   .. autosummary::
      ~Geometry.get_sample_angles
      ~Geometry.get_q_lab
      ~Geometry.get_hkl

See Also
--------
:class:`SpaceConverter` : Coordinate transformations using Geometry
:class:`BcdiPipeline` : Full pipeline using Geometry configuration
