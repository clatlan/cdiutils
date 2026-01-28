Wavefront Analysis and Probe Characterisation
==============================================

This guide covers wavefront propagation and probe analysis using the
:mod:`cdiutils.wavefront` module.

**Use cases:**

* Characterising focusing optics (KB mirrors, Fresnel zone plates)
* Determining focal plane position
* Measuring probe size and coherence
* Probe correction in forward models

Overview
--------

BCDI reconstructions provide both the **sample** and the **illuminating probe**
(or "exit wave"). Analysing the probe gives insights into:

* Beam focus quality
* Coherence properties
* Effective resolution
* Optics aberrations

The :mod:`cdiutils.wavefront` module provides tools for:

1. **Probe metrics:** FWHM, beam size, intensity profiles
2. **Propagation:** Angular spectrum method for near/far field
3. **Focus finding:** Automated focal plane determination

Basic Probe Metrics
-------------------

Extract Probe from Reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After phase retrieval with PyNX, the probe is stored in the CXI file:

.. code-block:: python

   from cdiutils.io import CXIFile
   from cdiutils.wavefront import probe_metrics
   
   # Load probe from PyNX output
   with CXIFile("result.cxi", mode="r") as cxi:
       probe = cxi["entry_1/probe_1/data"][()]  # Complex 2D array
   
   # Get pixel size from reconstruction
   pixel_size = (55e-9, 55e-9)  # metres (from detector)

Analyse Probe
~~~~~~~~~~~~~

.. code-block:: python

   # Compute metrics and plot
   fig, metrics = probe_metrics(
       probe=probe,
       pixel_size=pixel_size,
       zoom_factor="auto",  # or specific int
       probe_convention="pynx",
       centre_at_max=False,
       verbose=True
   )
   
   # Access computed metrics
   fwhm_x = metrics["fwhm_x"]["value"]  # metres
   fwhm_y = metrics["fwhm_y"]["value"]
   
   print(f"Probe FWHM: {fwhm_x*1e6:.2f} µm × {fwhm_y*1e6:.2f} µm")

**Output:**

* **fig:** Matplotlib figure with 2D probe and line profiles
* **metrics:** Dictionary with FWHM, FW10%M, Gaussian fit results

Wavefront Propagation
---------------------

Angular Spectrum Method
~~~~~~~~~~~~~~~~~~~~~~~

Propagate a 2D complex wavefront to a different plane:

.. code-block:: python

   from cdiutils.wavefront import angular_spectrum_propagation
   
   # Propagate downstream by 10 cm
   propagation_distance = 0.1  # metres
   wavelength = 1.5e-10        # metres (from energy)
   pixel_size = 55e-9          # metres
   
   propagated_probe = angular_spectrum_propagation(
       wavefront=probe,
       propagation_distance=propagation_distance,
       wavelength=wavelength,
       pixel_size=pixel_size,
       magnification=1.0,
       do_fftshift=True,
       verbose=True
   )

**Parameters:**

* **magnification:** Simulates effective magnification (for divergent beams)
* **do_fftshift:** Whether wavefront is centred (usually True)
* **verbose:** Prints near-field validity limits

**Validity range:** Printed by ``verbose=True``. Beyond this, use Fresnel or
Fraunhofer diffraction instead.

Finding the Focal Plane
------------------------

Automated Focus Sweep
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cdiutils.wavefront import probe_focus_sweep
   
   # Sweep through propagation distances
   propagation_range = (-0.5, 0.5)  # -50 cm to +50 cm
   num_steps = 100
   
   fig, focus_data = probe_focus_sweep(
       probe=probe,
       wavelength=wavelength,
       pixel_size=pixel_size,
       propagation_range=propagation_range,
       num_steps=num_steps,
       metric="peak_intensity"  # or "fwhm"
   )
   
   # Optimal distance
   optimal_z = focus_data["optimal_distance"]
   print(f"Focal plane at z = {optimal_z:.3f} m")

**Metrics:**

* ``"peak_intensity"``: Focus = maximum intensity (default)
* ``"fwhm"``: Focus = minimum FWHM

Get Focal Distances
~~~~~~~~~~~~~~~~~~~

For more control:

.. code-block:: python

   from cdiutils.wavefront import get_focal_distances
   
   focal_distances = get_focal_distances(
       probe=probe,
       wavelength=wavelength,
       pixel_size=pixel_size,
       propagation_range=(-0.5, 0.5),
       num_steps=100,
       verbose=True
   )
   
   print(f"Focal distance (x): {focal_distances['x']:.4f} m")
   print(f"Focal distance (y): {focal_distances['y']:.4f} m")

Focus Probe to Optimal Plane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cdiutils.wavefront import focus_probe
   
   # Automatically find and propagate to focus
   focused_probe, focal_distances = focus_probe(
       probe=probe,
       wavelength=wavelength,
       pixel_size=pixel_size,
       propagation_range=(-0.5, 0.5),
       num_steps=100
   )
   
   # Analyse focused probe
   fig, metrics_focused = probe_metrics(
       probe=focused_probe,
       pixel_size=pixel_size,
       verbose=True
   )

Visualising Propagated Probe
-----------------------------

.. code-block:: python

   from cdiutils.wavefront import plot_propagated_probe
   
   # Plot probe at multiple distances
   distances = [-0.2, -0.1, 0.0, 0.1, 0.2]  # metres
   
   fig = plot_propagated_probe(
       probe=probe,
       wavelength=wavelength,
       pixel_size=pixel_size,
       propagation_distances=distances,
       quantity="intensity"  # or "phase"
   )

Practical Examples
------------------

Example 1: Full Probe Characterisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cdiutils.io import CXIFile
   from cdiutils.wavefront import probe_metrics, focus_probe
   from cdiutils.utils import energy_to_wavelength
   
   # Load probe
   with CXIFile("result.cxi") as cxi:
       probe = cxi["entry_1/probe_1/data"][()]
       energy = cxi["entry_1/instrument_1/source_1/energy"][()]  # eV
   
   wavelength = energy_to_wavelength(energy)
   pixel_size = (55e-9, 55e-9)
   
   # Analyse at current plane
   print("=== Probe at reconstruction plane ===")
   fig1, metrics1 = probe_metrics(probe, pixel_size, verbose=True)
   
   # Find and focus to optimal plane
   print("\\n=== Finding focal plane ===")
   focused_probe, focal_dist = focus_probe(
       probe, wavelength, pixel_size,
       propagation_range=(-1.0, 1.0),
       num_steps=200
   )
   
   # Analyse at focus
   print("\\n=== Probe at focal plane ===")
   fig2, metrics2 = probe_metrics(focused_probe, pixel_size, verbose=True)
   
   print(f"\\nFocal position: {focal_dist['x']:.3f} m (x), "
         f"{focal_dist['y']:.3f} m (y)")

Example 2: Compare Probes from Different Runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cdiutils.io import CXIFile
   from cdiutils.wavefront import probe_metrics
   import matplotlib.pyplot as plt
   
   runs = [1, 2, 3, 4, 5]
   fwhm_values = []
   
   for run_id in runs:
       with CXIFile(f"result_run{run_id}.cxi") as cxi:
           probe = cxi["entry_1/probe_1/data"][()]
       
       _, metrics = probe_metrics(probe, pixel_size, verbose=False)
       fwhm_x = metrics["fwhm_x"]["value"]
       fwhm_values.append(fwhm_x * 1e6)  # Convert to µm
   
   # Plot FWHM variation
   plt.figure()
   plt.plot(runs, fwhm_values, 'o-')
   plt.xlabel("Run ID")
   plt.ylabel("FWHM (µm)")
   plt.title("Probe size across runs")
   plt.grid(True)
   plt.show()

Common Pitfalls
---------------

**1. Wrong probe convention**

PyNX stores probe in specific orientation. Use ``probe_convention="pynx"``:

.. code-block:: python

   # ✅ Correct
   probe_metrics(probe, pixel_size, probe_convention="pynx")

**2. Propagation outside validity range**

Check warnings from ``verbose=True``:

.. code-block:: python

   propagated = angular_spectrum_propagation(
       ..., verbose=True  # ✅ Prints limits
   )

If outside range, results are inaccurate.

**3. Mixed units**

Be consistent with metres:

.. code-block:: python

   # ❌ Wrong
   pixel_size = 55  # Forgot units
   wavelength = 1.5e-10
   
   # ✅ Correct
   pixel_size = 55e-9  # metres
   wavelength = 1.5e-10  # metres

References
----------

* **Angular Spectrum Method:** Goodman, "Introduction to Fourier Optics" (2005)
* **BCDI Probe Analysis:** Laulhé et al., J. Appl. Cryst. (2020)

See Also
--------

* :mod:`cdiutils.wavefront` - API reference
* :class:`~cdiutils.process.PostProcessor` - Uses probe for normalisation
