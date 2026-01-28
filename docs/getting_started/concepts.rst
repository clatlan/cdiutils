BCDI Concepts and Conventions
==============================

This page explains key concepts, coordinate systems, and conventions used
in CDIutils for Bragg Coherent Diffraction Imaging.

BCDI Workflow Overview
----------------------

The complete BCDI data processing workflow consists of:

1. **Data Loading**
   
   * Read detector frames and motor angles
   * Apply flat-field correction and bad pixel masking
   * Determine experimental geometry

2. **Preprocessing**
   
   * Crop to region of interest (ROI) around Bragg peak
   * Bin data to reduce size (optional)
   * Transform to reciprocal space coordinates

3. **Phase Retrieval**
   
   * Iterative reconstruction (typically using PyNX)
   * Recover complex-valued object from intensity-only data
   * Multiple runs to avoid local minima

4. **Postprocessing**
   
   * Phase unwrapping and ramp removal
   * Calculate displacement from phase
   * Calculate strain from displacement gradient
   * Transform to laboratory frame

5. **Analysis & Visualization**
   
   * Facet analysis for crystallographic orientation
   * 3D interactive visualization
   * Statistical analysis of strain distributions

Coordinate Systems
------------------

CDIutils handles three main coordinate systems and their transformations:

Detector Frame (Matrix Indices)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Convention:** Matrix indexing (i, j, k)

* **axis0 (i):** First detector dimension (rows)
* **axis1 (j):** Second detector dimension (columns)  
* **axis2 (k):** Rocking curve (angular scan) direction

This is the native format of detector arrays. **Not physical coordinates.**

CXI Convention (Lab Frame)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Convention:** Right-handed Cartesian (x, y, z) in metres

* **x:** Beam direction (downstream)
* **y:** Typically vertical (up)
* **z:** Perpendicular to beam (completing right-handed system)

**Used for:**

* Reconstruction output
* Direct space (real space) object
* Displacement and strain fields

**Origin:** Centre of the reconstructed object

The CXI convention follows the Coherent X-ray Imaging data format
standard. See: https://github.com/cxidb/CXI

xrayutilities (XU) Convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Convention:** Right-handed Cartesian (x, y, z)

* **x:** Perpendicular to beam (typically towards the door)
* **y:** Typically vertical (up)
* **z:** Beam direction (downstream)

**Used for:**

* Reciprocal space gridding (via :mod:`xrayutilities`)
* Detector calibration
* Internal coordinate transformations

Transformations
~~~~~~~~~~~~~~~

CDIutils handles coordinate transformations via:

* :class:`~cdiutils.geometry.Geometry`: Defines beamline geometry
* :class:`~cdiutils.converter.SpaceConverter`: Executes transformations

.. code-block:: python

   from cdiutils import Geometry, SpaceConverter
   
   # Define geometry
   geometry = Geometry.from_setup(beamline="ID01")
   
   # Create converter
   converter = SpaceConverter(geometry=geometry, energy=9.0e3)
   
   # Convert CXI ↔ XU
   geometry.cxi_to_xu()  # Modifies geometry in-place
   geometry.xu_to_cxi()  # Reverse transformation

Reciprocal Space Concepts
--------------------------

Momentum Transfer Vector (Q)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The scattering vector **Q** connects the incident wavevector $\\mathbf{k}_i$
and scattered wavevector $\\mathbf{k}_f$:

.. math::

   \\mathbf{Q} = \\mathbf{k}_f - \\mathbf{k}_i

For elastic scattering: $|\\mathbf{k}_i| = |\\mathbf{k}_f| = 2\\pi/\\lambda$

Bragg Condition
~~~~~~~~~~~~~~~

Diffraction occurs when:

.. math::

   \\mathbf{Q} = \\mathbf{G}_{hkl}

where $\\mathbf{G}_{hkl}$ is a reciprocal lattice vector.

**BCDI measures 3D intensity around one Bragg peak**, providing:

* **Amplitude:** Related to electron density
* **Phase:** Contains lattice displacement information

Reciprocal Space Gridding
~~~~~~~~~~~~~~~~~~~~~~~~~~

Raw data is in **detector + rocking angle** coordinates. CDIutils transforms
this to **orthogonal reciprocal space (Qx, Qy, Qz)** using:

* Detector geometry (distance, pixel size, centre)
* Incident beam energy (wavelength)
* Sample and detector motor angles

**Trade-offs:**

* **Coarse grid:** Faster, but may alias features
* **Fine grid:** Accurate, but memory-intensive

See :doc:`../user_guide/reciprocal_space_gridding` for tuning.

Phase and Displacement
-----------------------

Relationship
~~~~~~~~~~~~

For a crystal lattice displaced by $\\mathbf{u}(\\mathbf{r})$, the phase is:

.. math::

   \\phi(\\mathbf{r}) = \\mathbf{Q} \\cdot \\mathbf{u}(\\mathbf{r})

**Key insight:** Phase encodes projection of displacement along **Q**.

Measuring only one Bragg peak gives **one component** of displacement. For
full 3D displacement, measure multiple reflections (multi-BCDI).

Strain Calculation
~~~~~~~~~~~~~~~~~~

Strain tensor components are computed from displacement gradients:

.. math::

   \\varepsilon_{ij} = \\frac{1}{2}\\left(\\frac{\\partial u_i}{\\partial x_j} + \\frac{\\partial u_j}{\\partial x_i}\\right)

CDIutils computes:

* **Normal strain:** $\\varepsilon_{xx}, \\varepsilon_{yy}, \\varepsilon_{zz}$
* **Shear strain:** $\\varepsilon_{xy}, \\varepsilon_{xz}, \\varepsilon_{yz}$
* **Volumetric strain:** $\\varepsilon_{\\text{vol}} = \\varepsilon_{xx} + \\varepsilon_{yy} + \\varepsilon_{zz}$

Via :meth:`~cdiutils.process.PostProcessor.get_structural_properties`.

Sample Orientation
------------------

The :class:`~cdiutils.geometry.Geometry` class defines:

* **sample_circles:** Motor rotation axes (e.g., eta, phi)
* **detector_circles:** Detector rotation axes (e.g., nu, delta)
* **sample_surface_normal:** Defines "horizontal" vs "vertical" sample

Example for horizontal sample at ID01:

.. code-block:: python

   geometry = Geometry.from_setup(
       beamline="ID01",
       sample_orientation="horizontal"  # or "h"
   )
   
   # For vertical sample
   geometry = Geometry.from_setup(
       beamline="ID01", 
       sample_orientation="vertical"  # or "v"
   )

**Critical for strain calculations:** Incorrect orientation → wrong strain
reference frame.

Detector Geometry
-----------------

Accurate detector calibration is essential for quantitative results.

Parameters (stored in ``det_calib_params`` dict):

* **distance:** Sample-to-detector distance (metres)
* **pwidth1, pwidth2:** Pixel sizes (metres)
* **cch1, cch2:** Direct beam position (pixels)
* **tiltazimuth, tilt:** Detector rotation (radians)

Obtain via:

* **Detector motor positions** (rough, often sufficient)
* **Calibration scan** (e.g., silver behenate powder)

See :doc:`../user_guide/detector_calibration` for procedures.

Support and Autocorrelation
----------------------------

Support Definition
~~~~~~~~~~~~~~~~~~

The **support** is a binary mask defining where the object exists:

* 1: Object present
* 0: Vacuum

**Critical for phase retrieval convergence.** Too loose → artifacts. Too
tight → loss of weak features.

Autocorrelation Thresholding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyNX can estimate support from data autocorrelation:

.. code-block:: yaml

   phasing:
     support_autocorrelation_threshold: 0.04  # Typical: 0.02-0.1

Higher threshold → tighter support.

See :doc:`../user_guide/phase_retrieval_tuning` for best practices.

Further Reading
---------------

* :doc:`../user_guide/coordinate_systems` - Detailed transformation math
* :doc:`../user_guide/detector_calibration` - Calibration procedures  
* :doc:`../user_guide/phase_retrieval_tuning` - Phasing parameter guide
* :doc:`../examples/index` - Interactive Jupyter notebook examples
