Detector Calibration Tutorial
=============================

This tutorial shows how to use calibration measurements to refine detector parameters using the authentic CDIutils API, based on the template notebook.

Setup and Data Loading
-----------------------

Import Required Modules
^^^^^^^^^^^^^^^^^^^^^^^

Start by importing the necessary libraries:

.. code-block:: python

    import cdiutils

Configure Experiment Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set up your experiment file path and parameters:

.. code-block:: python

    experiment_file_path = "/path/to/experiment.h5"
    energy = 9000  # eV - your beam energy  
    sample_name = "calibration_sample"
    
    loader = cdiutils.io.ID01Loader(
        experiment_file_path=experiment_file_path,
        sample_name=sample_name,
    )

Load Calibration Data
^^^^^^^^^^^^^^^^^^^^^

Load the calibration scan data and motor positions:

.. code-block:: python

    scan = 456  # your calibration scan number
    det_calib_frames = loader.load_detector_data(scan=scan)
    angles = loader.load_motor_positions(scan=scan)

Geometry Setup
--------------

Create and configure the geometry:

.. code-block:: python

    geometry = cdiutils.Geometry.from_setup(beamline_setup="ID01")
    geometry.cxi_to_xu()  # change to XU convention
    print(geometry)  # to check out the geometry

Run Detector Calibration
-------------------------

Execute the detector calibration using the SpaceConverter:

.. code-block:: python

    det_calib_params = cdiutils.SpaceConverter.run_detector_calibration(
        det_calib_frames,
        detector_outofplane_angle=angles["detector_outofplane_angle"],
        detector_inplane_angle=angles["detector_inplane_angle"],
        xu_detector_circles=geometry.detector_circles,
        energy=energy,
        sdd_estimate=None
    )

View Calibration Results
------------------------

Display the calibration parameters:

.. code-block:: python

    print(
        "det_calib_params = {"
    )
    
    for k, v in det_calib_params.items():
        print(
            f'\t"{k}": {v},'
        )
    print("}")

Next Steps
----------

- Apply your calibration to BCDI data analysis using :doc:`pipeline_tutorial`
- Explore advanced analysis with :doc:`step_by_step_tutorial`
- Check the API reference for :class:`~cdiutils.SpaceConverter`
- See template notebooks in ``src/cdiutils/templates/`` for working examples
