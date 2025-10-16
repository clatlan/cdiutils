Pipeline Tutorial
=================

This tutorial introduces the **cdiutils** pipeline functionality for automated BCDI data processing based on the authentic CDIutils API.

Overview
--------

The cdiutils package provides a powerful pipeline system that automates the complete BCDI reconstruction workflow from raw detector data to final quantitative analysis. The main class for this functionality is :class:`~cdiutils.BcdiPipeline`.

Getting Started
---------------

Basic Pipeline Setup
^^^^^^^^^^^^^^^^^^^^

First, import the necessary modules and set up the required parameters:

.. code-block:: python

    import os
    import cdiutils  # core library for BCDI processing

Parameter Configuration
^^^^^^^^^^^^^^^^^^^^^^^

Define the key parameters for accessing and saving data:

.. code-block:: python

    # define the key parameters (must be filled in by the user)
    beamline_setup: str = "ID01"  # provide the beamline setup
    experiment_file_path: str = "/path/to/experiment/file.h5"  # path to experiment file
    sample_name: str = "Sample_Pt"  # specify the sample name
    scan: int = 42  # specify the scan number
    
    # choose where to save the results (default: current working directory)
    dump_dir = os.getcwd() + f"/results/{sample_name}/S{scan}/"
    
    # load the parameters and parse them into the BcdiPipeline class instance
    params = cdiutils.pipeline.get_params_from_variables(dir(), globals())
    bcdi_pipeline = cdiutils.BcdiPipeline(params=params)

Data Preprocessing
^^^^^^^^^^^^^^^^^^

Configure and run the preprocessing step:

.. code-block:: python

    bcdi_pipeline.preprocess(
        preprocess_shape=(150, 150),  # define cropped window size
        voxel_reference_methods=["max", "com", "com"],  # centering method sequence
        hot_pixel_filter=False,  # remove isolated hot pixels
        background_level=None,  # background intensity level to remove
    )

Phase Retrieval
^^^^^^^^^^^^^^^

Configure and run the PyNX phase retrieval:

.. code-block:: python

    bcdi_pipeline.phase_retrieval(
        clear_former_results=True,
        nb_run=20,  # total number of runs
        nb_run_keep=10,  # number of reconstructions to keep
        # support=bcdi_pipeline.pynx_phasing_dir + "support.cxi"  # optionally use existing support
    )

Analysing Phase Retrieval Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyse the quality of the phase retrieval results:

.. code-block:: python

    bcdi_pipeline.analyse_phasing_results(
        sorting_criterion="mean_to_max",  # selects the sorting method
        
        # optional parameters
        # plot_phasing_results=False,  # uncomment to disable plotting
        # plot_phase=True,  # uncomment to enable phase plotting
    )

Selecting Best Candidates
^^^^^^^^^^^^^^^^^^^^^^^^^

Select the best reconstructions and perform mode decomposition:

.. code-block:: python

    # define how many of the best candidates to keep
    number_of_best_candidates: int = 5
    
    # select the best reconstructions based on the sorting criterion
    bcdi_pipeline.select_best_candidates(
        nb_of_best_sorted_runs=number_of_best_candidates
        # best_runs=[10]  # uncomment to manually select a specific run
    )
    
    # perform mode decomposition on the selected reconstructions
    bcdi_pipeline.mode_decomposition()

Post-processing
^^^^^^^^^^^^^^^

Run the complete post-processing pipeline:

.. code-block:: python

    bcdi_pipeline.postprocess(
        isosurface=0.3,  # threshold for isosurface
        voxel_size=None,  # use default voxel size if not provided
        flip=False        # whether to flip the reconstruction if you got the twin image
    )

Advanced Configuration
---------------------

Preprocessing Parameters
^^^^^^^^^^^^^^^^^^^^^^^^

The preprocessing step supports various parameters:

.. code-block:: python

    # main preprocessing parameters
    bcdi_pipeline.preprocess(
        preprocess_shape=(150, 150),  # shape of the cropped window
        voxel_reference_methods=[(70, 200, 200), "com", "com"],  # centering methods
        rocking_angle_binning=2,  # binning factor for rocking curve
        light_loading=True,  # load only ROI of data
        hot_pixel_filter=True,  # remove isolated hot pixels
        background_level=3,  # background intensity level to remove
        hkl=[1, 1, 1],  # Bragg reflection measured
    )

Phase Retrieval Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure detailed phase retrieval parameters:

.. code-block:: python
    
    # override defaults in phase_retrieval
    bcdi_pipeline.phase_retrieval(
        nb_run=50, 
        nb_run_keep=25,

        # algorithm recipe options
        algorithm = None  # or specify exact chain like "(Sup * (ER**20)) ** 10"
        nb_raar = 500
        nb_hio = 300
        nb_er = 200
        psf = "pseudo-voigt,1,0.05,20"
        
        # support-related parameters
        support = "auto"  # or path to existing support
        support_threshold = "0.15, 0.40"  # must be a string
        support_update_period = 20
        support_only_shrink = False
        support_post_expand = None  # ex: "-1,1" or "-1,2,-1"
        support_update_border_n = None
        support_smooth_width_begin = 2
        support_smooth_width_end = 0.5
        
        # other parameters
        positivity = False
        beta = 0.9  # β parameter in HIO and RAAR
        detwin = True
        rebin = "1, 1, 1"  # must be a string
    )

Generate Support for Further Phasing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optionally generate a support from the best reconstruction:

.. code-block:: python

    # generate support from best reconstruction
    # bcdi_pipeline.generate_support_from("best", fill=False)  # uncomment to generate

Available Sorting Criteria
^^^^^^^^^^^^^^^^^^^^^^^^^^

Choose from various sorting criteria for reconstruction analysis:

.. code-block:: python

    # available sorting criteria:
    # - "mean_to_max": difference between mean of Gaussian fit and maximum amplitude
    # - "sharpness": sum of amplitude within support raised to power of 4  
    # - "std": standard deviation of amplitude
    # - "llk": log-likelihood of reconstruction
    # - "llkf": free log-likelihood of reconstruction
    
    bcdi_pipeline.analyse_phasing_results(
        sorting_criterion="sharpness",  # try different criteria
        plot_phasing_results=True,
        plot_phase=False,
    )

Next Steps
----------

- Explore the :doc:`step_by_step_tutorial` for detailed manual processing
- Check the :doc:`detector_calibration_tutorial` for geometry calibration
- See the API reference for :class:`~cdiutils.BcdiPipeline`
- Examine the template notebooks in ``src/cdiutils/templates/`` for working examples
