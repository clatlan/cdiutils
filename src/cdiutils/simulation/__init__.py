"""
Simulation sub-package for BCDI experiments.

This sub-package provides comprehensive tools for simulating BCDI
measurements, from creating synthetic 3D objects to generating
realistic detector data with noise and experimental geometry.

Modules
-------
* ``objects``: object creation, phase manipulation and diffraction
  simulation.
* ``noise``: noise models for realistic detector simulation.
* ``detector``: end-to-end BCDI measurement simulator with detector
  geometry.
* ``domains``: domain models of a reconstruction (phase steps, tilts,
  bending) fitted to measured intensity.

Main Components
---------------
Object creation (from ``objects``):

* ``make_box``: create a 3D box (parallelepiped/cube).
* ``make_ellipsoid``: create a 3D ellipsoid or sphere.
* ``make_cylinder``: create a 3D cylinder.
* ``make_polyhedron``: create a 3D convex polyhedron from planes.
* ``get_winterbottom``: wulffpack Winterbottom particle (with a fix for
  interfaces along z, e.g. c-plane hcp).
* ``wulffpack_planes``: facet planes of a wulffpack particle.
* ``make_from_wulffpack``: voxelise a wulffpack particle.

Phase manipulation (from ``objects``):

* ``add_linear_phase``: add a linear phase gradient.
* ``add_quadratic_phase``: add a quadratic phase (defocus, strain).
* ``add_displacement_field``: add the phase of a displacement field.
* ``add_random_phase``: add random phase noise.

Diffraction simulation (from ``objects``):

* ``simulate_diffraction``: compute the diffraction pattern via FFT.

Domain models (from ``domains``):

* ``segment_domains``: split a reconstruction into domains separated by
  low-modulus planes, sharing its filled outline.
* ``fill_domain_gaps``: fill only the planes between domains.
* ``measure_domain_phases``: ramp, phase offset and coherence per
  domain.
* ``fit_quadratic_phase``: linear + quadratic phase of a region (bent or
  graded crystal).
* ``DomainModelFit``: fit measured intensity with domains carrying phase
  steps, tilts or a quadratic phase (Poisson deviance).

Noise models (from ``noise``):

* ``add_noise``: add Gaussian and/or Poisson noise to data.

Supported crystals (from ``epitaxy``):

* ``SupportedCrystal``: crystal on a substrate for experiment planning
  (Bragg angles vs energy, substrate nodes, reciprocal-space maps,
  Winterbottom shapes, simulated peaks).

Diffractometer view (from ``viewer``):

* ``DiffractometerViewer``: interactive lab-frame view of the reciprocal
  lattice, the Ewald sphere and the detector.

BCDI measurement simulation (from ``detector``):

* ``BCDISimulator``: complete BCDI measurement simulator.

Notes
-----
The diffraction simulation uses forward FFT convention (optics/signal
processing). For crystallographic conventions, manage phase signs in
your object definition rather than changing the FFT direction.

For realistic noise modelling, the :class:`BCDISimulator`
class provides end-to-end simulation including diffractometer
geometry, coordinate transformations, and detector effects.

Example
-------
>>> from cdiutils.simulation import (
...     make_box,
...     add_random_phase,
...     simulate_diffraction,
...     BCDISimulator,
... )
>>>
>>> # simple diffraction simulation
>>> obj = make_box((64, 64, 64), dimensions=20)
>>> obj = add_random_phase(obj, amplitude=0.1)
>>> intensity = simulate_diffraction(obj, photon_budget=1e9)
>>>
>>> # full BCDI measurement simulation
>>> sim = BCDISimulator(
...     energy=9000,
...     lattice_parameter=4.08e-10,
... )
>>> sim.simulate_object(shape=(100, 100, 100))
>>> detector_data = sim.to_detector_frame()
"""

# import from objects module
# import from detector module
from .detector import BCDISimulator
from .domains import (
    DomainModelFit,
    fill_domain_gaps,
    fit_quadratic_phase,
    measure_domain_phases,
    segment_domains,
)
from .epitaxy import SupportedCrystal

# import from noise module
from .noise import add_noise
from .objects import (
    add_displacement_field,
    add_linear_phase,
    add_quadratic_phase,
    add_random_phase,
    get_winterbottom,
    make_box,
    make_cylinder,
    make_ellipsoid,
    make_from_wulffpack,
    make_polyhedron,
    simulate_diffraction,
    wulffpack_planes,
)
from .viewer import DiffractometerViewer

__all__ = [
    # object creation
    "make_box",
    "make_ellipsoid",
    "make_cylinder",
    "make_polyhedron",
    "get_winterbottom",
    "wulffpack_planes",
    "make_from_wulffpack",
    # phase manipulation
    "add_linear_phase",
    "add_quadratic_phase",
    "add_displacement_field",
    "add_random_phase",
    # diffraction simulation
    "simulate_diffraction",
    "segment_domains",
    "fill_domain_gaps",
    "measure_domain_phases",
    "fit_quadratic_phase",
    "DomainModelFit",
    # noise models
    "add_noise",
    # BCDI measurement simulator
    "BCDISimulator",
    "SupportedCrystal",
    "DiffractometerViewer",
]
