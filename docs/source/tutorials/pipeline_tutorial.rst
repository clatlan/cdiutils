BCDI Pipeline Tutorial
=====================

This tutorial demonstrates how to use the complete BCDI pipeline for automated data processing using the ``BcdiPipeline`` class. The content is based on the actual ``bcdi_pipeline.ipynb`` template notebook included with CDIutils.

.. note::
   **Download the template**: :download:`bcdi_pipeline.ipynb <../../../src/cdiutils/templates/bcdi_pipeline.ipynb>`

Overview
--------

The BCDI pipeline provides a streamlined workflow for processing Bragg Coherent Diffraction Imaging data. The ``BcdiPipeline`` class handles the entire process, including:

* **Pre-processing** → Data preparation and corrections
* **Phase retrieval** → Running PyNX algorithms to reconstruct the phase
* **Post-processing** → Refining, analysing (getting the strain!), and visualising results

You can provide **either**:
* A **YAML parameter file** for full automation
* A **Python dictionary** for interactive control in the notebook

General Parameters Setup
------------------------

Here, define the key parameters for **accessing and saving data** before running the pipeline. These parameters must be set manually by the user before execution:

.. code-block:: python

   # Import required packages
   import os
   import cdiutils  # core library for BCDI processing
   
   # Define the key parameters (must be filled in by the user)
   beamline_setup: str = ""  # example: "ID01" (provide the beamline setup)
   experiment_file_path: str = ""  # example: "/path/to/experiment/file.h5"
   sample_name: str = ""  # example: "Sample_Pt" (specify the sample name)
   scan: int = 0  # example: 42 (specify the scan number)
   
   # Choose where to save the results (default: current working directory)
   dump_dir = os.getcwd() + f"/results/{sample_name}/S{scan}/"
   
   # Load the parameters and parse them into the BcdiPipeline class instance
   params = cdiutils.pipeline.get_params_from_variables(dir(), globals())
   bcdi_pipeline = cdiutils.BcdiPipeline(params=params)

Pre-processing Stage
--------------------

If you need to update specific parameters, you can **pass them directly** into the ``preprocess`` method. The pre-processing stage prepares your raw data for phase retrieval:

.. code-block:: python

   bcdi_pipeline.preprocess(
       preprocess_shape=(150, 150),  # define cropped window size
       voxel_reference_methods=["max", "com", "com"],  # centring method sequence
       hot_pixel_filter=False,  # remove isolated hot pixels
       background_level=None,  # background intensity level to remove
   )

**Main Parameters**

* ``preprocess_shape`` → The shape of the cropped window used throughout the processes. Can be a **tuple of 2 or 3 values**. If only **2 values**, the entire rocking curve is used.

* ``voxel_reference_methods`` → A ``list`` (or a single value) defining how to centre the data. Can include ``"com"``, ``"max"``, or a ``tuple`` of ``int`` (specific voxel position). Example:

  .. code-block:: python
  
     voxel_reference_methods = [(70, 200, 200), "com", "com"]
  
  This centres a box of size ``preprocess_shape`` around ``(70, 200, 200)``, then iteratively refines it using ``"com"`` (only computed within this box). Useful when ``"com"`` fails due to artefacts or ``"max"`` fails due to hot pixels. Default: ``["max", "com", "com"]``.

* ``rocking_angle_binning`` → If you want to bin in the **rocking curve direction**, provide a binning factor (e.g., ``2``).

* ``light_loading`` → If ``True``, loads only the **ROI of the data** based on ``voxel_reference_methods`` and ``preprocess_output_shape``.

* ``hot_pixel_filter`` → Removes isolated hot pixels. Default: ``False``.

* ``background_level`` → Sets the background intensity to be removed. Example: ``3``. Default: ``None``.

* ``hkl`` → Defines the **Bragg reflection** measured to extend *d*-spacing values to the lattice parameter. Default: ``[1, 1, 1]``.

Phase Retrieval Stage  
---------------------

The phase retrieval uses `PyNX <https://pynx.esrf.fr/en/latest/index.html>`_ algorithms to reconstruct the phase. See the `pynx.cdi <https://pynx.esrf.fr/en/latest/scripts/pynx-cdi-id01.html>`_ documentation for details on the phasing algorithms.

.. code-block:: python

   bcdi_pipeline.phase_retrieval(
       clear_former_results=True,
       nb_run=20,
       nb_run_keep=10,
       # support=bcdi_pipeline.pynx_phasing_dir + "support.cxi"
   )

**Algorithm Recipe**

You can either:
- Provide the exact chain of algorithms
- Specify the number of iterations for **RAAR**, **HIO**, and **ER**

.. code-block:: python

   algorithm = None  # ex: "(Sup * (ER**20)) ** 10, (Sup*(HIO**20)) ** 15, (Sup*(RAAR**20)) ** 25"
   nb_raar = 500
   nb_hio = 300
   nb_er = 200
   psf = "pseudo-voigt,1,0.05,20"

**Support-related Parameters**

.. code-block:: python

   support = "auto"  # ex: bcdi_pipeline.pynx_phasing_dir + "support.cxi" (path to an existing support)
   support_threshold = "0.15, 0.40"  # must be a string
   support_update_period = 20
   support_only_shrink = False
   support_post_expand = None  # ex: "-1,1" or "-1,2,-1"

.. note::
   If strain seems too large, don't use "auto" (autocorrelation) but use "circle" or "square", in combination with "support_size"

**Other Parameters**

.. code-block:: python

   positivity = False
   beta = 0.9  # β parameter in HIO and RAAR
   detwin = True
   rebin = "1, 1, 1"  # must be a string

**Number of Runs & Reconstructions to Keep**

.. code-block:: python

   nb_run = 20  # total number of runs
   nb_run_keep = 10  # number of reconstructions to keep

Analysing Phasing Results
-------------------------

This step evaluates the quality of the phase retrieval results by sorting reconstructions based on a ``sorting_criterion``:

.. code-block:: python

   bcdi_pipeline.analyse_phasing_results(
       sorting_criterion="mean_to_max",  # selects the sorting method
       
       # Optional parameters
       # plot_phasing_results=False,  # uncomment to disable plotting
       # plot_phase=True,  # uncomment to enable phase plotting
   )

**Available Sorting Criteria**

* ``"mean_to_max"`` → Difference between the mean of the **Gaussian fit of the amplitude histogram** and its maximum value. A **smaller difference** indicates a more homogeneous reconstruction.
* ``"sharpness"`` → Sum of the amplitude within the support raised to the power of 4. **Lower values** indicate greater homogeneity.
* ``"std"`` → **Standard deviation** of the amplitude.
* ``"llk"`` → **Log-likelihood** of the reconstruction.
* ``"llkf"`` → **Free log-likelihood** of the reconstruction.

Selecting Best Candidates and Mode Decomposition
-----------------------------------------------

Select the best reconstructions based on a **sorting criterion** and keep a specified number of top candidates:

.. code-block:: python

   # Define how many of the best candidates to keep
   number_of_best_candidates: int = 5  
   
   # Select the best reconstructions based on the sorting criterion
   bcdi_pipeline.select_best_candidates(
       nb_of_best_sorted_runs=number_of_best_candidates
       # best_runs=[10]  # uncomment to manually select a specific run
   )
   
   # Perform mode decomposition on the selected reconstructions
   bcdi_pipeline.mode_decomposition()

**Parameters**

* ``nb_of_best_sorted_runs`` → The number of best reconstructions to keep, selected based on the ``sorting_criterion`` used in the ``analyse_phasing_results`` method.
* ``best_runs`` → Instead of selecting based on sorting, you can manually specify a list of reconstruction numbers.

Once the best candidates are chosen, ``mode_decomposition`` analyses them to extract dominant features.

Generating Support (Optional)
-----------------------------

Optionally, generate a support for further phasing attempts:

.. code-block:: python

   # bcdi_pipeline.generate_support_from("best", fill=False)  # uncomment to generate a support

**Parameters**

* ``run`` → Set to either ``"best"`` to use the best reconstruction or an **integer** corresponding to the specific run you want.
* ``output_path`` → The location to save the generated support. By default, it will be saved in the ``pynx_phasing`` folder.
* ``fill`` → Whether to fill the support if it contains holes. Default: ``False``.
* ``verbose`` → Whether to print logs and display a plot of the support.

Post-processing Stage
--------------------

This stage includes several key operations: **orthogonalisation** of the reconstructed data, **phase manipulation** (phase unwrapping, phase ramp removal), **computation of physical properties** (displacement field, strain, d-spacing), and **visualisation**:

.. code-block:: python

   bcdi_pipeline.postprocess(
       isosurface=0.3,  # threshold for isosurface
       voxel_size=None,  # use default voxel size if not provided
       flip=False        # whether to flip the reconstruction if you got the twin image (enantiomorph)
   )

**Key Post-processing Features**

* **Strain calculation**: Compute strain tensor components
* **Phase unwrapping**: Remove 2π phase jumps  
* **Coordinate transformation**: Convert to orthogonal coordinates
* **Displacement fields**: Calculate atomic displacements
* **Visualisation**: Generate multiple plots for analysis

Complete Pipeline Execution
----------------------------

You can run the entire pipeline by calling each method sequentially, or use a YAML configuration file for full automation.

Output Files
------------

The pipeline generates several output files in the specified ``dump_dir``:

* Pre-processed data in CXI format
* Multiple PyNX reconstruction results
* Post-processed data with strain analysis
* VTK files for 3D visualisation
* Various plots and summary figures

Advanced Configuration with YAML
---------------------------------

For complex experiments, you can use YAML configuration files. The pipeline can be initialised from YAML parameters for full automation.

Troubleshooting
---------------

**Memory errors during processing**
   Reduce the ``preprocess_shape`` or use lighter loading options

**Poor phase retrieval results**
   Try different support methods or increase ``nb_run``

**Strain calculation fails**  
   Check data quality and support determination parameters

**File path errors**
   Ensure all paths are absolute and files exist

**Support issues**
   If strain seems too large, avoid "auto" support and use "circle" or "square" with appropriate ``support_size``

Next Steps
----------

After completing this tutorial, you can:

* Explore the :doc:`step_by_step_tutorial` for detailed control
* Check the :doc:`../examples/bcdi_reconstruction_analysis` for analysis examples
* Learn about :doc:`detector_calibration_tutorial` for geometric corrections
