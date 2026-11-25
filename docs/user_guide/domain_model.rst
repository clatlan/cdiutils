Domain Models: Phase Steps, Tilts or Bending
=============================================

This guide shows how to test the interpretation of a reconstruction
against the measured intensity with the :mod:`cdiutils.simulation.domains`
module.

**Use cases:**

* A reconstruction breaks into parallel bands or "domains" separated by
  low-amplitude planes, and you need to know whether the gaps are real
* A Bragg peak is split or streaked and several real-space pictures
  (phase steps, tilted grains, a bent crystal) could explain it
* Choosing between reconstructions obtained with different supports

Overview
--------

Phase retrieval from random starts does not always find the object that
produced the data. On a strongly deformed crystal, HIO/RAAR/ER often
converge to solutions made of parallel bands of amplitude, separated by
low-amplitude planes where the phase changes fastest. The same split
Bragg peak can be produced by:

1. **Domains with phase steps:** parts of the crystal whose lattice
   planes are offset along Q (a constant phase difference between
   domains).
2. **Tilted domains:** parts with slightly different orientations
   (different linear phase ramps).
3. **A continuous, smoothly deformed crystal:** a quadratic phase
   (bending, or a lattice-spacing gradient).

Instead of reading the reconstruction, the module builds each picture as
a forward model with a few parameters and fits it to the measured
intensity. Models are compared by their Poisson deviance on the same data
and mask.

Two checks that do not need a model are worth doing first:

* **Integrated intensity.** Phase steps and smooth deformations conserve
  the integrated Bragg intensity (Parseval). If the reconstructed volume
  drops by a large factor while the monitor-normalised integrated
  intensity of the scan does not, the "missing" volume is not missing
  material.
* **Fixed support.** Phase the same data with the support fixed to the
  outline of a less-deformed state of the same particle (many runs). If
  that solution fits the data as well or better, the gaps are not
  required by the data.

Conventions
-----------

* Arrays are in the **detector frame of the phasing**: the intensity is
  the ``pynx_input_data`` array, the object a reconstruction mode from the
  same phasing (before flip or orthogonalisation).
* The model intensity is ``|fftshift(fftn(ifftshift(obj)))|**2`` (PyNX
  convention), so it compares voxel by voxel with the data.
* Phases are written on coordinates centred on ``n // 2``, as in
  :func:`~cdiutils.simulation.add_linear_phase`. Linear coefficients are
  in rad/voxel, quadratic ones in rad/voxel².

Segmenting the Domains
----------------------

.. code-block:: python

   import numpy as np
   from cdiutils.simulation import (
       segment_domains,
       fill_domain_gaps,
       measure_domain_phases,
   )

   # mode: complex reconstruction mode (detector frame)
   domains, seeds = segment_domains(
       np.abs(mode),
       seed_threshold=0.25,    # relative modulus of the domain cores
       outline_threshold=0.2,  # relative modulus of the outline
   )

   # seeds: the domains with their gaps left empty ("voids")
   # domains: the closed, filled outline shared between the domains
   # fill_domain_gaps: the seeds plus only the planes between them
   filled = fill_domain_gaps(seeds, width=3.0)

   # ramp, constant phase and coherence of each domain
   for domain in measure_domain_phases(mode, seeds):
       print(domain["label"], domain["size"], domain["ramp"],
             domain["offset"], domain["coherence"])

A coherence close to 1 means the domain is well described by one ramp
plus a constant. Ramps that differ by much less than :math:`2\pi/N`
rad/voxel (N the array size along that axis) move the domains' Bragg
peaks by less than one pixel: they cannot produce a peak splitting of
several pixels.

Fitting the Models
------------------

.. code-block:: python

   from cdiutils.simulation import DomainModelFit

   ramps = np.array([d["ramp"] for d in measure_domain_phases(mode, seeds)])
   fitter = DomainModelFit(iobs, mask, seeds)

   no_steps = fitter.fit("C", ramps=ramps)
   steps = fitter.fit("A", ramps=ramps, n_starts=8)
   tilts = fitter.fit("B", ramps=ramps)

   for result in (no_steps, steps, tilts):
       print(result["model"], result["deviance"], result["n_parameters"])

The models are:

==========  =========================================================
Model       Description
==========  =========================================================
``"C"``     one common ramp, no phase steps, one amplitude per domain
``"A"``     one common ramp, free phase step per domain
``"B"``     no phase steps, free ramp per domain (tilted domains)
``"full"``  free phase steps and free ramps
``"bend"``  no steps, free ramp and quadratic phase per domain
==========  =========================================================

``equal_amplitudes=True`` forces the same density in every domain. Fits
with free phases use several random starts (``n_starts``); fits with
free ramps re-compute the Fourier transforms at every step and use
Powell's method, so they are slower.

Testing a Continuous, Bent Crystal
----------------------------------

Give the whole outline a single label and start the quadratic phase from
the reconstruction itself:

.. code-block:: python

   from scipy.ndimage import binary_fill_holes
   from cdiutils.simulation import fit_quadratic_phase

   # outline of a reconstruction obtained with a fixed support
   outline = binary_fill_holes(np.abs(rod_mode) / np.abs(rod_mode).max()
                               > 0.3)
   ramp, quadratic = fit_quadratic_phase(rod_mode, outline)

   rod_fitter = DomainModelFit(iobs, mask, outline.astype(int))
   bend = rod_fitter.fit("bend", ramps=ramp[None],
                         quadratics=quadratic[None])

:func:`~cdiutils.simulation.fit_quadratic_phase` fits the local phase
differences between neighbouring voxels, so it is insensitive to phase
wrapping. The deviance of ``bend`` compares directly with the domain
models above when the data and mask are the same.

Reading the Result
------------------

* Compare deviances, not reconstructions: a model is preferred when it
  lowers the deviance by much more than its extra number of parameters.
* Look at the model in the detector frame (``fitter.model_intensity``):
  the rocking profile through the peak and the (rocking, detector) slices
  show which features each model reproduces.
* Validate the selection on a **known answer**: simulate each model with
  the real mask and photon count, add Poisson noise, refit, and check that
  the right model wins (see the example notebook).

Example: a Cu particle under gas (ID01, 2026)
---------------------------------------------

Above 150 °C the free-support reconstruction of a ~900 nm Cu rod broke
into three bands, and only about a third of its volume remained above the
surface threshold. The tests above gave:

* integrated intensity constant within ~15 % (no lost material);
* the support fixed to the rod measured at 100 °C fitted as well or
  better than the free support, with a continuous phase;
* Poisson deviance on the measured intensity: no steps 2.2·10⁷, tilted
  domains 2.2·10⁷, phase steps 1.0–1.3·10⁷, one bent rod 0.69·10⁷.

The bands were the solutions random starts find on a bent crystal; the
bending vanished on cooling.

See Also
--------

* :doc:`../api/simulation` — API reference
* ``examples/domain_model_fit.ipynb`` — worked example with known answers
* :doc:`phase_retrieval_tuning` — support strategies
