Utility Functions
=================

Index of the functions of :mod:`cdiutils.utils`; each entry links to its
full documentation on the :doc:`../utils` page.

Arrays and Shapes
-----------------

.. autosummary::

   cdiutils.utils.crop_at_center
   cdiutils.utils.symmetric_pad
   cdiutils.utils.center
   cdiutils.utils.find_max_pos
   cdiutils.utils.nan_center_of_mass
   cdiutils.utils.compute_distance_from_com
   cdiutils.utils.bin_along_axis
   cdiutils.utils.shape_for_safe_centred_cropping
   cdiutils.utils.find_suitable_array_shape
   cdiutils.utils.extract_reduced_shape
   cdiutils.utils.ensure_pynx_shape
   cdiutils.utils.is_valid_shape
   cdiutils.utils.adjust_to_valid_shape
   cdiutils.utils.get_prime_factors
   cdiutils.utils.get_centred_slices
   cdiutils.utils.transform_volume

Values and Masks
----------------

.. autosummary::

   cdiutils.utils.normalise
   cdiutils.utils.normalise_complex_array
   cdiutils.utils.num_to_nan
   cdiutils.utils.zero_to_nan
   cdiutils.utils.nan_to_zero
   cdiutils.utils.to_bool
   cdiutils.utils.hybrid_gradient

Supports
--------

.. autosummary::

   cdiutils.utils.make_support
   cdiutils.utils.fill_up_support
   cdiutils.utils.size_up_support
   cdiutils.utils.find_hull

Detector Data
-------------

.. autosummary::

   cdiutils.utils.hot_pixel_filter
   cdiutils.utils.cluster_bragg_signal
   cdiutils.utils.background_map
   cdiutils.utils.get_oversampling_ratios
   cdiutils.utils.oversampling_from_diffraction
   cdiutils.utils.get_reciprocal_voxel_size
   cdiutils.utils.compute_corrected_angles

Vectors, Energy and Helpers
---------------------------

.. autosummary::

   cdiutils.utils.unit_vector
   cdiutils.utils.angle
   cdiutils.utils.v1_to_v2_rotation_matrix
   cdiutils.utils.energy_to_wavelength
   cdiutils.utils.wavelength_to_energy
   cdiutils.utils.valid_args_only
