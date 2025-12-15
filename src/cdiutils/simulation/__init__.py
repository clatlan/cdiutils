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

Phase manipulation (from objects):
    add_linear_phase : Add linear phase gradient.
    add_quadratic_phase : Add quadratic phase (defocus, strain).
    add_displacement_field : Add phase from displacement field.
    add_random_phase : Add random phase noise.

Diffraction simulation (from objects):
    simulate_diffraction : Compute diffraction pattern via FFT.

Noise models (from noise):
    add_noise : Add Gaussian and/or Poisson noise to data.

BCDI measurement simulation (from detector):
    BCDIMeasurementSimulator : Complete BCDI measurement simulator.

Notes
-----
The diffraction simulation uses forward FFT convention (optics/signal
processing). For crystallographic conventions, manage phase signs in
your object definition rather than changing the FFT direction.

For realistic noise modelling, the :class:`BCDIMeasurementSimulator`
class provides end-to-end simulation including diffractometer
geometry, coordinate transformations, and detector effects.

Example
-------
>>> from cdiutils.simulation import (
...     make_box,
...     add_random_phase,
...     simulate_diffraction,
...     BCDIMeasurementSimulator,
... )
>>>
>>> # simple diffraction simulation
>>> obj = make_box((64, 64, 64), dimensions=20)
>>> obj = add_random_phase(obj, amplitude=0.1)
>>> intensity = simulate_diffraction(obj, photon_budget=1e9)
>>>
>>> # full BCDI measurement simulation
>>> sim = BCDIMeasurementSimulator(
...     energy=9000,
...     lattice_parameter=4.08e-10,
... )
>>> sim.simulate_object(shape=(100, 100, 100))
>>> detector_data = sim.to_detector_frame()
"""

# import from objects module
# import from detector module
from .detector import BCDIMeasurementSimulator

# import from noise module
from .noise import add_noise
from .objects import (
    add_displacement_field,
    add_linear_phase,
    add_quadratic_phase,
    add_random_phase,
    make_box,
    make_cylinder,
    make_ellipsoid,
    simulate_diffraction,
)

__all__ = [
    # object creation
    "make_box",
    "make_ellipsoid",
    "make_cylinder",
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
    "BCDIMeasurementSimulator",
]
