"""
Simulation sub-package for BCDI experiments.

This sub-package provides comprehensive tools for simulating BCDI
measurements, from creating synthetic 3D objects to generating
realistic detector data with noise and experimental geometry.

Modules
-------
objects : Object creation, phase manipulation, and diffraction
    simulation.
noise : Noise models for realistic detector simulation.
detector : End-to-end BCDI measurement simulator with detector
    geometry.

Main Components
---------------
Object creation (from objects):
    make_box : Create a 3D box (parallelepiped/cube).
    make_ellipsoid : Create a 3D ellipsoid or sphere.
    make_cylinder : Create a 3D cylinder.
    make_polyhedron : Create a 3D convex polyhedron from planes.
    get_winterbottom : wulffpack Winterbottom particle (with a fix
        for interfaces along z, e.g. c-plane hcp).
    wulffpack_planes : Facet planes of a wulffpack particle.
    make_from_wulffpack : Voxelise a wulffpack particle.

Phase manipulation (from objects):
    add_linear_phase : Add linear phase gradient.
    add_quadratic_phase : Add quadratic phase (defocus, strain).
    add_displacement_field : Add phase from displacement field.
    add_random_phase : Add random phase noise.

Diffraction simulation (from objects):
    simulate_diffraction : Compute diffraction pattern via FFT.

Noise models (from noise):
    add_noise : Add Gaussian and/or Poisson noise to data.

Supported crystals (from epitaxy):
    SupportedCrystal : Crystal on a substrate for experiment
        planning (Bragg angles vs energy, substrate nodes,
        reciprocal-space maps, Winterbottom shapes, simulated peaks).

BCDI measurement simulation (from detector):
    BCDISimulator : Complete BCDI measurement simulator.

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
    # noise models
    "add_noise",
    # BCDI measurement simulator
    "BCDISimulator",
    "SupportedCrystal",
]
