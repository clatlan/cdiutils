Detector Geometry Calibration
==============================

Accurate detector calibration is **critical** for quantitative BCDI strain
analysis. Small errors in detector position or orientation propagate directly
into strain measurements.

**This guide covers:**

* Why calibration matters
* Obtaining calibration parameters
* Using calibration in CDIutils
* Validating calibration quality

Why Calibration Matters
------------------------

Detector calibration defines:

1. **Sample-to-detector distance** (``distance``)
2. **Direct beam position** (``cch1``, ``cch2``)
3. **Pixel dimensions** (``pwidth1``, ``pwidth2``)
4. **Detector tilt angles** (``tilt``, ``tiltazimuth``)

**Impact of errors:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Effect of 1% error
   * - distance
     - ~1% error in Q, strain magnitude
   * - cch1, cch2
     - Systematic strain gradient artifacts
   * - pwidth1, pwidth2
     - Anisotropic strain errors
   * - tilt, tiltazimuth
     - Shear strain artifacts, coordinate rotation

**Bottom line:** Calibration errors fake strain.

Calibration Parameters
----------------------

The ``det_calib_params`` dictionary contains:

.. code-block:: python

   det_calib_params = {
       "distance": 1.2,         # sample-to-detector distance (metres)
       "pwidth1": 55e-6,        # pixel size, axis 0 (metres)
       "pwidth2": 55e-6,        # pixel size, axis 1 (metres)
       "cch1": 512.3,           # direct beam centre, axis 0 (pixels)
       "cch2": 511.7,           # direct beam centre, axis 1 (pixels)
       "tiltazimuth": 0.0,      # detector rotation azimuth (radians)
       "tilt": 0.01             # detector tilt from perpendicular (radians)
   }

**Typical values:**

* **distance:** 0.5–2.0 m (depends on beamline, detector size)
* **pwidth1, pwidth2:** 55 µm (Maxipix), 75 µm (Eiger), 172 µm (Pilatus)
* **cch1, cch2:** Usually near centre, but can be offset
* **tilt, tiltazimuth:** Small (~0–0.1 radians), often negligible

Obtaining Calibration Parameters
---------------------------------

Method 1: Beamline Motor Positions (Quick)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most beamlines record detector position in metadata:

.. code-block:: python

   from cdiutils.io import ID01Loader
   
   loader = ID01Loader(
       sample_name="S0001",
       scan=42,
       data_dir="/path/to/data"
   )
   
   # Load from metadata
   det_calib_params = loader.load_det_calib_params()
   print(det_calib_params)

**Pros:** Fast, automated

**Cons:** Motor positions may be inaccurate (backlash, calibration drift)

Method 2: Calibration Scan (Accurate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a standard sample (e.g., silver behenate, LaB6) to refine calibration.

**Procedure:**

1. Collect diffraction pattern from calibrant
2. Fit known peak positions to determine geometry
3. Use :mod:`xrayutilities` for fitting

.. code-block:: python

   from xrayutilities.analysis import sample_align
   
   # Load calibration scan
   calibration_image = ...  # 2D detector image
   
   # Known peak positions (for silver behenate)
   d_spacing = 58.38e-10  # metres
   energy = 9000  # eV
   
   # Fit detector parameters
   param, eps = sample_align.area_detector_calib(
       peak_positions_2d,  # List of (y, x) pixel coords
       detector_init_params,
       detector_model='area',
       plot=True
   )
   
   # Extract refined parameters
   det_calib_params_refined = {
       "distance": param[0],
       "cch1": param[3],
       "cch2": param[4],
       # ... etc
   }

**Pros:** Most accurate (~0.1% in distance)

**Cons:** Requires calibration scan, manual peak picking

Method 3: Direct Beam Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ``cch1, cch2`` only:

1. Collect image with **beam attenuated** (no sample)
2. Find peak position

.. code-block:: python

   import numpy as np
   from scipy.ndimage import center_of_mass
   
   # Load direct beam image (attenuated!)
   direct_beam = ...  # 2D array
   
   # Find centre
   cch1, cch2 = center_of_mass(direct_beam)
   print(f"Direct beam at: ({cch1:.2f}, {cch2:.2f})")

**Pros:** Simple, quick check

**Cons:** Only refines beam centre, not distance/tilt

Using Calibration in CDIutils
------------------------------

In BcdiPipeline (Automatic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pipeline loads calibration automatically:

.. code-block:: python

   from cdiutils.pipeline import BcdiPipeline
   
   pipeline = BcdiPipeline(param_file_path="config.yml")
   pipeline.preprocess()
   
   # Access calibration
   print(pipeline.converter.det_calib_params)

Override in config.yml if needed:

.. code-block:: yaml

   det_calib_params:
     distance: 1.234
     cch1: 512.5
     cch2: 511.8
     # ... other params

Manual SpaceConverter
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cdiutils import Geometry, SpaceConverter
   
   geometry = Geometry.from_setup(beamline="ID01")
   
   converter = SpaceConverter(
       geometry=geometry,
       energy=9000,
       det_calib_params=det_calib_params  # ← Your calibration
   )

Validating Calibration Quality
-------------------------------

Check 1: Reciprocal Space Peak Symmetry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After gridding to Q-space, Bragg peak should be symmetric:

.. code-block:: python

   from cdiutils.plot import plot_volume_slices
   
   # Grid data to Q-space
   data_q = converter.detector_to_lab_q(data)
   
   # Plot central slices
   plot_volume_slices(
       data_q,
       title="Q-space peak (should be symmetric)"
   )

**Bad calibration → asymmetric peak, elliptical cross-sections**

Check 2: Strain Map Artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After full reconstruction:

.. code-block:: python

   # Check for systematic gradients
   import numpy as np
   
   # Strain should be ~uniform in unstrained region
   strain_mean = np.nanmean(strain)
   strain_std = np.nanstd(strain)
   
   print(f"Strain: {strain_mean:.2e} ± {strain_std:.2e}")
   
   # Large gradients → calibration error

**Calibration errors create fake strain gradients**

Check 3: Multi-Scan Consistency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measure same sample, different scans:

.. code-block:: python

   strains = []
   for scan in [42, 43, 44]:
       pipeline = BcdiPipeline(param_file_path=f"config_scan{scan}.yml")
       pipeline.preprocess()
       pipeline.phase_retrieval()
       pipeline.postprocess()
       strains.append(np.nanmean(pipeline.strain))
   
   # Should be consistent
   print(f"Strain variation: {np.std(strains):.2e}")

**High variation → systematic calibration differences**

Advanced: Refining Calibration from Data
-----------------------------------------

If you suspect calibration errors, you can refine using the Bragg peak itself:

**Concept:** Misalignment creates asymmetric peak. Optimize parameters to
maximize symmetry.

.. code-block:: python

   from scipy.optimize import minimize
   
   def peak_asymmetry(params):
       """Cost function: measure peak asymmetry."""
       det_calib_params["distance"] = params[0]
       det_calib_params["cch1"] = params[1]
       det_calib_params["cch2"] = params[2]
       
       # Regrid with new params
       converter = SpaceConverter(geometry, energy, det_calib_params=det_calib_params)
       # ... grid data ...
       
       # Measure asymmetry (e.g., via moments)
       asymmetry = compute_asymmetry(data_q)
       return asymmetry
   
   # Optimize
   result = minimize(
       peak_asymmetry,
       x0=[det_calib_params["distance"], det_calib_params["cch1"], det_calib_params["cch2"]],
       method="Nelder-Mead"
   )
   
   print(f"Refined distance: {result.x[0]:.4f} m")

**Caution:** This is advanced and can overfit. Validate with known calibrant.

Common Pitfalls
---------------

**1. Mixing pixel conventions**

Some beamlines define pixel centre at (0.5, 0.5), others at (0, 0):

.. code-block:: python

   # Check beamline convention
   # ID01: typically (0.5, 0.5) offset
   # P10: typically (0, 0)

**2. Wrong pixel size**

Binned detectors have effective larger pixels:

.. code-block:: python

   # If detector binned 2×2:
   pwidth1_effective = pwidth1_native * 2
   pwidth2_effective = pwidth2_native * 2

**3. Ignoring detector tilts**

Even small tilts (~1°) affect strain. Don't assume ``tilt=0``.

See Also
--------

* :class:`~cdiutils.converter.SpaceConverter` - Uses calibration for transformations
* :doc:`coordinate_systems` - How calibration affects Q-space
* :doc:`strain_analysis` - Impact on strain quantification
* :mod:`xrayutilities.analysis.sample_align` - Calibration tools
