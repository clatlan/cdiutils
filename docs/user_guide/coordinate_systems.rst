Coordinate Systems and Transformations
======================================

This guide explains coordinate system handling in CDIutils, covering the
transformations between detector, reciprocal space, and direct space frames.

**Critical for:** Quantitative strain analysis, coordinate-dependent plotting,
multi-Bragg peak analysis.

Overview of Coordinate Systems
-------------------------------

CDIutils manages three primary coordinate systems:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - System
     - Convention
     - Usage
   * - **Detector Frame**
     - Matrix indices (i, j, k)
     - Raw data, ROI selection
   * - **Reciprocal Space (Lab)**
     - CXI or XU convention
     - Q-space gridding, resolution
   * - **Direct Space (Lab)**
     - CXI or XU convention
     - Reconstruction, strain fields

The :class:`~cdiutils.geometry.Geometry` and
:class:`~cdiutils.converter.SpaceConverter` classes handle all transformations.

Geometry Class
--------------

The :class:`~cdiutils.geometry.Geometry` class defines experimental geometry:

* Beam direction
* Sample rotation axes
* Detector rotation axes  
* Detector pixel orientation
* Sample surface normal (horizontal vs vertical mounting)

Factory Method (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :meth:`~cdiutils.geometry.Geometry.from_setup` for standard beamlines:

.. code-block:: python

   from cdiutils import Geometry
   
   # ID01 beamline, horizontal sample
   geometry = Geometry.from_setup(
       beamline="ID01",
       sample_orientation="horizontal"
   )
   
   # P10 beamline, vertical sample
   geometry = Geometry.from_setup(
       beamline="P10",
       sample_orientation="vertical"
   )
   
   # Supported beamlines:
   # "ID01", "P10", "SIXS", "NanoMAX", "CRISTAL", "ID27"

Custom Geometry
~~~~~~~~~~~~~~~

For custom or modified geometries:

.. code-block:: python

   geometry = Geometry(
       sample_circles=["x-", "y-"],          # eta, phi
       detector_circles=["y-", "x-"],        # nu, delta
       detector_axis0_orientation="y-",      # rows
       detector_axis1_orientation="x+",      # columns
       beam_direction=[1, 0, 0],             # along x
       sample_surface_normal=[0, 1, 0],      # vertical is y
       is_cxi=True                           # using CXI convention
   )

**Axis orientation notation:**

* ``"x+"``: Positive x direction
* ``"y-"``: Negative y direction
* ``"z+"``: Positive z direction

CXI ↔ xrayutilities Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The geometry can be converted between conventions:

.. code-block:: python

   # Start in CXI convention
   geometry = Geometry.from_setup(beamline="ID01")
   print(geometry.is_cxi)  # True
   
   # Convert to xrayutilities convention
   geometry.cxi_to_xu()
   print(geometry.is_cxi)  # False
   
   # Convert back
   geometry.xu_to_cxi()
   print(geometry.is_cxi)  # True

**When to convert:**

* **CXI:** For output, visualisation, CXI file writing
* **XU:** For :mod:`xrayutilities` gridding (internal to SpaceConverter)

SpaceConverter Class
--------------------

The :class:`~cdiutils.converter.SpaceConverter` performs coordinate
transformations using :mod:`xrayutilities` under the hood.

Initialization
~~~~~~~~~~~~~~

.. code-block:: python

   from cdiutils import Geometry, SpaceConverter
   
   geometry = Geometry.from_setup(beamline="ID01")
   
   converter = SpaceConverter(
       geometry=geometry,
       energy=9000,  # eV
       roi=[100, 400, 50, 450, 80, 380],  # detector ROI
       det_calib_params={
           "distance": 1.2,         # metres
           "pwidth1": 55e-6,        # metres
           "pwidth2": 55e-6,
           "cch1": 512,             # pixels
           "cch2": 512,
           "tiltazimuth": 0.0,      # radians
           "tilt": 0.0
       }
   )

Detector → Reciprocal Space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transform detector coordinates (pixels + angles) to reciprocal space (Qx, Qy, Qz):

.. code-block:: python

   import numpy as np
   
   # Set motor angles for the scan
   angles = {
       "eta": np.linspace(30.0, 32.0, 100),  # degrees
       "phi": 0.0,
       "nu": 35.5,
       "delta": 32.1
   }
   converter.angles = angles
   
   # Initialise Q-space transformation
   converter.init_qspace_transformations(
       scan_shape=data.shape  # (nz, ny, nx)
   )
   
   # Get Q coordinates for each detector pixel
   qx, qy, qz = converter.get_qspace_coords()
   
   # Or use interpolator for gridding
   converter.init_interpolator(
       target_voxel_size=0.05,  # 1/nm
       space="reciprocal"
   )
   
   # Grid to orthogonal Q-space
   data_q = converter.detector_to_lab_q(data)

Reciprocal → Direct Space
~~~~~~~~~~~~~~~~~~~~~~~~~~

After phase retrieval (which outputs in reciprocal space), transform to
direct space:

.. code-block:: python

   # Assuming 'obj' is complex 3D reconstruction from phasing
   
   # Initialise direct space interpolator
   converter.init_interpolator(
       target_voxel_size=5e-9,  # metres (5 nm)
       space="direct"
   )
   
   # Transform to orthogonal direct space
   obj_direct = converter.q_to_lab_direct(obj)
   
   # Access voxel size
   voxel_size = converter.direct_lab_voxel_size
   print(f"Resolution: {voxel_size[0]*1e9:.2f} nm/voxel")

Coordinate Transformations Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Detector → Reciprocal space (Q)
   data_q = converter.detector_to_lab_q(data_detector)
   
   # Reciprocal → Direct space (real space)
   obj_direct = converter.q_to_lab_direct(obj_q)
   
   # Direct → Reciprocal (for forward modelling)
   obj_q = converter.lab_direct_to_q(obj_direct)

Practical Examples
------------------

Example 1: Manual Coordinate Gridding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cdiutils import Geometry, SpaceConverter
   from cdiutils.io import ID01Loader
   import numpy as np
   
   # Load data
   loader = ID01Loader(
       sample_name="S0001",
       scan=42,
       data_dir="/path/to/data"
   )
   data, angles = loader.load_data(roi=[100, 400, 50, 450, 80, 380])
   energy = loader.load_energy()
   
   # Setup geometry
   geometry = Geometry.from_setup(beamline="ID01")
   
   # Create converter
   converter = SpaceConverter(
       geometry=geometry,
       energy=energy,
       roi=loader.roi,
       det_calib_params=loader.load_det_calib_params()
   )
   converter.angles = angles
   
   # Transform to Q-space
   converter.init_qspace_transformations(scan_shape=data.shape)
   converter.init_interpolator(target_voxel_size=0.05, space="reciprocal")
   data_q = converter.detector_to_lab_q(data)
   
   print(f"Q-space shape: {data_q.shape}")
   print(f"Q voxel size: {converter.q_lab_interpolator.target_voxel_size}")

Example 2: Save/Load Converter State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The converter can be saved to HDF5 for reproducibility:

.. code-block:: python

   # Save converter configuration
   converter.to_file("converter_config.h5")
   
   # Later, reload
   from cdiutils import SpaceConverter
   
   converter = SpaceConverter.from_file("converter_config.h5")
   
   # All settings preserved:
   # - Geometry
   # - Energy, ROI
   # - Detector calibration
   # - Transformation matrices

Example 3: Coordinate-Dependent Plotting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cdiutils.plot import plot_volume_slices
   
   # Plot in CXI convention with physical units
   plot_volume_slices(
       data=strain,
       voxel_size=converter.direct_lab_voxel_size,  # metres
       convention="cxi",
       views=("z+", "y-", "x+"),
       title="Strain field (CXI convention)"
   )
   
   # Plot in XU convention
   plot_volume_slices(
       data=strain,
       voxel_size=converter.direct_lab_voxel_size,
       convention="xu",
       views=("x-", "y+", "z-"),
       title="Strain field (XU convention)"
   )

Common Pitfalls
---------------

**1. Mixing conventions inconsistently**

❌ Wrong:

.. code-block:: python

   geometry = Geometry.from_setup(beamline="ID01")  # CXI
   geometry.cxi_to_xu()  # Convert to XU
   # ... later ...
   plot_volume_slices(data, convention="cxi")  # Mismatch!

✅ Correct:

.. code-block:: python

   geometry = Geometry.from_setup(beamline="ID01")
   # Keep in CXI throughout, or track conversions explicitly

**2. Incorrect sample orientation**

❌ Wrong:

.. code-block:: python

   # Sample mounted vertically, but declared horizontal
   geometry = Geometry.from_setup(
       beamline="ID01",
       sample_orientation="horizontal"  # ❌
   )

✅ Correct:

.. code-block:: python

   geometry = Geometry.from_setup(
       beamline="ID01",
       sample_orientation="vertical"  # ✅
   )

**3. Forgetting to set angles**

❌ Wrong:

.. code-block:: python

   converter = SpaceConverter(geometry=geometry, energy=9000)
   converter.init_qspace_transformations(scan_shape=data.shape)
   # ❌ converter.angles not set!

✅ Correct:

.. code-block:: python

   converter = SpaceConverter(geometry=geometry, energy=9000)
   converter.angles = angles  # ✅ Set angles first
   converter.init_qspace_transformations(scan_shape=data.shape)

**4. Incompatible voxel sizes**

When transforming reciprocal → direct space, voxel sizes are related by:

.. math::

   \\Delta x \\cdot \\Delta Q_x = 2\\pi / N_x

Requesting arbitrary voxel size may require interpolation.

Advanced Topics
---------------

Custom Sample Surface Normal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For non-standard sample orientations:

.. code-block:: python

   geometry = Geometry.from_setup(beamline="ID01")
   
   # Override sample surface normal (in CXI frame)
   geometry.sample_surface_normal = [0.707, 0.707, 0]  # 45° tilted
   
   converter = SpaceConverter(geometry=geometry, energy=9000)

Multi-Bragg Peak Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

For full 3D displacement, measure multiple Bragg peaks:

.. code-block:: python

   # Measure (111) peak
   converter_111 = SpaceConverter(geometry=geometry, energy=9000)
   # ... process ...
   displacement_111 = get_displacement(...)
   
   # Measure (200) peak  
   converter_200 = SpaceConverter(geometry=geometry, energy=9000)
   # ... process ...
   displacement_200 = get_displacement(...)
   
   # Combine to get full 3D displacement
   # (requires alignment and inversion, see literature)

See Also
--------

* :class:`~cdiutils.geometry.Geometry` - API reference
* :class:`~cdiutils.converter.SpaceConverter` - API reference
* :doc:`../getting_started/concepts` - Conceptual overview
* :doc:`detector_calibration` - Calibration procedures
* :doc:`strain_analysis` - Using coordinates for strain calculations
