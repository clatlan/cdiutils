Quickstart: 5-Minute BCDI Analysis
====================================

This quickstart guide shows you how to run a complete BCDI analysis using
:class:`~cdiutils.pipeline.BcdiPipeline` in a Jupyter notebook.

Prerequisites
-------------

* CDIutils installed with PyNX support (see :doc:`../installation`)
* BCDI scan data from a supported beamline
* Jupyter notebook environment

Minimal Working Example
-----------------------

**Step 1: Import and Set Parameters**

.. code-block:: python

   import os
   import cdiutils
   
   # define experiment parameters
   beamline_setup = "id01"
   experiment_file_path = "/path/to/experiment/file.h5"
   sample_name = "Sample_Pt"
   scan = 42
   
   # set output directory
   dump_dir = os.getcwd() + f"/results/{sample_name}/S{scan}/"
   
   # load parameters and create pipeline
   params = cdiutils.pipeline.get_params_from_variables(dir(), globals())
   pipeline = cdiutils.BcdiPipeline(params=params)

**Step 2: Preprocess Data**

.. code-block:: python

   # run preprocessing with optional parameter overrides
   pipeline.preprocess(
       preprocess_shape=(200, 200, 200),
       voxel_reference_methods=["max", "com", "com"]
   )

**Step 3: Phase Retrieval**

.. code-block:: python

   # run PyNX phase retrieval
   pipeline.phase_retrieval(
       nb_run=50,
       nb_run_keep=10,
       support_threshold="0.15, 0.40"
   )

**Step 4: Postprocess Results**

.. code-block:: python

   # analyse phasing results and run postprocessing
   pipeline.analyse_phasing_results()
   pipeline.select_best_candidates(nb_of_best_sorted_runs=5)
   pipeline.mode_decomposition()
   
   # calculate strain and displacement
   pipeline.postprocess(voxel_size=5)  # 5 nm voxels

**Step 5: Visualise Results**

.. code-block:: python

   # interactive 3D visualisation
   pipeline.show_3d_final_result()
   
   # or browse phasing results interactively
   pipeline.phase_retrieval_gui()

That's it! Your results are saved in the ``dump_dir`` including:

* CXI files with reconstruction
* Strain and displacement maps
* VTK files for visualisation
* Analysis figures

Understanding the Output
------------------------

After ``postprocess()``, access results programmatically:

.. code-block:: python

   # access structural properties dictionary
   props = pipeline.structural_props
   
   amplitude = props["amplitude"]  # reconstructed amplitude
   support = props["support"]  # binary support mask
   phase = props["phase"]  # unwrapped phase (radians)
   displacement = props["displacement"]  # atomic displacement (angstroms)
   het_strain = props["het_strain"]  # heterogeneous strain (%)
   
   # voxel size in metres
   voxel_size = props["voxel_size"]
   print(f"Resolution: {voxel_size[0]*1e9:.2f} nm/voxel")

Next Steps
----------

**Customise processing:**
  See :doc:`../user_guide/pipeline` for detailed parameter tuning

**Complete notebook example:**
  Check :download:`bcdi_pipeline_example.ipynb <../../examples/bcdi_pipeline_example.ipynb>`

**Beamline-specific examples:**
  - :download:`ID01 example <../../examples/bcdi_pipeline_example.ipynb>`
  - :download:`P10 example <../../examples/bcdi_pipeline_example_p10.ipynb>`
  - :download:`NanoMAX example <../../examples/bcdi_pipeline_example_nanomax.ipynb>`

**Advanced workflows:**
  Learn about :doc:`../user_guide/coordinate_systems` for manual processing

Common Issues
-------------

**PyNX not found:**
  Install PyNX: https://pynx.esrf.fr/en/latest/install.html

**Wrong detector geometry:**
  Verify ``beamline_setup`` matches your beamline (e.g., "id01", "p10", "sixs")

**Memory errors:**
  Reduce ``preprocess_shape`` or increase binning

**Phase retrieval fails:**
  Adjust ``support_threshold`` (try "0.20, 0.35" range)

See :doc:`../user_guide/troubleshooting` for more solutions.
