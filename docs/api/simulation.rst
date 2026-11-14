Simulation Module
=================

The :mod:`cdiutils.simulation` module provides tools for simulating BCDI
data and for fitting simple real-space models to measured intensity.

.. automodule:: cdiutils.simulation
   :members:
   :undoc-members:
   :show-inheritance:

.. currentmodule:: cdiutils.simulation

Objects and Diffraction
-----------------------

.. autosummary::

   make_box
   make_ellipsoid
   make_cylinder
   add_linear_phase
   add_quadratic_phase
   add_displacement_field
   add_random_phase
   simulate_diffraction

Noise and Measurement
---------------------

.. autosummary::

   add_noise
   BCDISimulator

Domain Models
-------------

Test phase steps, tilts and bending of a reconstruction against the
measured intensity (:mod:`cdiutils.simulation.domains`).

.. autosummary::

   segment_domains
   fill_domain_gaps
   measure_domain_phases
   fit_quadratic_phase
   DomainModelFit

See Also
--------
:doc:`../user_guide/domain_model` : Domain models, with a worked example
