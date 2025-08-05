Detector Calibration Tutorial
=============================

This tutorial covers detector calibration procedures essential for accurate BCDI analysis, based on the actual ``detector_calibration.ipynb`` template notebook included with CDIutils. Proper calibration ensures correct geometric parameters, accurate strain measurements, and reliable data interpretation.

.. note::
   **Download the template**: :download:`detector_calibration.ipynb <../../../src/cdiutils/templates/detector_calibration.ipynb>`

Overview
--------

Detector calibration is crucial for accurate BCDI analysis. The CDIutils library provides automated calibration tools that determine essential geometric parameters from calibration measurements.

Prerequisites
-------------

Before starting calibration, you need:

* Access to calibration scan data (typically powder or single crystal measurements)
* Knowledge of your experimental setup and beamline configuration
* Approximate detector-sample distance estimate (optional but helpful)

Getting Started
---------------

Begin by importing CDIutils and setting up the basic parameters:

.. code-block:: python

   import cdiutils

Set up the experiment parameters that you'll need:

.. code-block:: python

   experiment_file_path = ""  # Path to your experiment HDF5 file
   energy = ""  # X-ray energy in eV
   sample_name = ""  # Sample identifier for calibration
   
   loader = cdiutils.io.ID01Loader(
       experiment_file_path=experiment_file_path,
       sample_name=sample_name,
   )

Loading Calibration Data
------------------------

Load the detector calibration frames and motor positions:

.. code-block:: python

   scan = ""  # Scan number for calibration measurement
   det_calib_frames = loader.load_detector_data(scan=scan)
   angles = loader.load_motor_positions(scan=scan)

The calibration data typically consists of:
- Detector frames showing diffraction patterns (powder rings or crystal spots)
- Motor positions defining the measurement geometry
- Energy information for the calibration measurement

Geometry Setup
--------------

Set up the beamline geometry configuration:

.. code-block:: python

   geometry = cdiutils.Geometry.from_setup(beamline_setup="")  # Specify your beamline
   geometry.cxi_to_xu()  # change to XU convention
   print(geometry)  # to check out the geometry

The geometry object contains:
- Beamline-specific coordinate systems and conventions
- Standard detector and sample motor configurations
- Default calibration parameters for the beamline

Running Detector Calibration
-----------------------------

Execute the automated detector calibration procedure:

.. code-block:: python

   det_calib_params = cdiutils.SpaceConverter.run_detector_calibration(
       det_calib_frames,
       detector_outofplane_angle=angles["detector_outofplane_angle"],
       detector_inplane_angle=angles["detector_inplane_angle"],
       xu_detector_circles=geometry.detector_circles,
       energy=energy,
       sdd_estimate=None  # Optional: provide estimated sample-detector distance
   )

This automated calibration determines:

* **Direct beam position**: ``cch1`` (vertical) and ``cch2`` (horizontal) pixel coordinates
* **Sample-to-detector distance**: Accurate distance measurement
* **Detector orientation**: Tilt and rotation corrections
* **Pixel size verification**: Confirms pixel size parameters

Calibration Results
-------------------

Review the calibration parameters that were determined:

.. code-block:: python

   print(
       "det_calib_params = {"
   )
   
   for k, v in det_calib_params.items():
       print(
           f'\t"{k}": {v},'
       )
   print("}")

The output provides a complete set of detector calibration parameters ready for use in BCDI analysis.

**Understanding the Parameters**

* ``cch1``, ``cch2``: Direct beam position on detector (pixels)
* ``distance``: Calibrated sample-to-detector distance (metres)
* ``pwidth1``, ``pwidth2``: Detector pixel sizes (metres)
* ``tilt``, ``tiltazimuth``: Detector tilt corrections
* ``detrot``: Detector rotation correction
* ``outerangle_offset``: Outer angle offset correction

Using Calibrated Parameters
---------------------------

The calibrated parameters can be used directly in your BCDI analysis workflow:

.. code-block:: python

   # Use in SpaceConverter for data analysis
   converter = cdiutils.SpaceConverter(
       geometry,
       det_calib_params,
       energy=energy,
       roi=roi  # Define your region of interest
   )

Quality Assessment
------------------

Assess the quality of your calibration:

**Visual Inspection**
- Check that powder rings appear circular and centred
- Verify that crystal spots show expected symmetries
- Look for systematic deviations that might indicate calibration issues

**Quantitative Checks**
- Compare calibrated distance with known experimental setup
- Verify that lattice parameters calculated from calibrated geometry match expected values
- Check consistency across multiple calibration measurements

Best Practices
--------------

**Calibration Frequency**
- Perform calibration for each experimental session
- Re-calibrate if detector position changes
- Use multiple calibration measurements for validation

**Calibration Standards**
- Use well-characterised reference materials
- Prefer powder patterns for detector geometry calibration
- Consider single crystal standards for specific applications

**Quality Control**
- Document calibration parameters and conditions
- Keep records of calibration measurements
- Validate calibration with independent measurements

**Troubleshooting Common Issues**

* **Poor ring quality**: Check sample preparation and exposure conditions
* **Asymmetric patterns**: May indicate detector misalignment or sample issues  
* **Inconsistent results**: Verify mechanical stability and measurement conditions

Next Steps
----------

With proper calibration, you can:

* Proceed with confidence to :doc:`pipeline_tutorial` for automated processing
* Apply manual control techniques from :doc:`step_by_step_tutorial`
* Analyze complex datasets in :doc:`../examples/bcdi_reconstruction_analysis`
* Achieve quantitative strain measurements with known uncertainties
