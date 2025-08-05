Step-by-Step BCDI Analysis Tutorial
====================================

This tutorial provides detailed manual control over each stage of BCDI processing, based on the actual ``step_by_step_bcdi_analysis.ipynb`` template notebook included with CDIutils. This approach allows you to understand and customise every step of the analysis workflow.

.. note::
   **Download the template**: :download:`step_by_step_bcdi_analysis.ipynb <../../../src/cdiutils/templates/step_by_step_bcdi_analysis.ipynb>`

Overview
--------

This notebook provides a step-by-step guide for users who intend to analyse their BCDI data. Unlike the automated pipeline, this approach gives you complete control over:

* Data loading and inspection with detailed parameter control
* Pre-processing methods and parameter selection
* Phase retrieval algorithm customisation
* Post-processing with advanced strain analysis
* Visualisation and output formatting

This manual approach is ideal for:

* Learning how BCDI processing works in detail
* Debugging problematic datasets
* Implementing custom processing steps
* Research and method development

Getting Started
---------------

Begin with importing the required libraries and setting up plotting parameters:

.. code-block:: python

   from IPython.display import display
   import matplotlib.pyplot as plt
   from matplotlib.colors import LogNorm, Normalize
   import numpy as np
   import os
   
   import cdiutils
   
   # Update the matplotlib parameters 
   cdiutils.plot.update_plot_params()

Data Loading and Inspection
----------------------------

Start by loading your experimental data and understanding its structure. First, inspect the data directory structure:

.. code-block:: python

   experiment_path = ""  # the path to the /RAW_DATA dir of the experiment
   # Ex: path = "/data/visitor/<experiment_name>/id01/<date>/RAW_DATA/"
   !tree -L 1 {experiment_path}

Then load the specific scan data:

.. code-block:: python

   experiment_name = ""  # Required
   experiment_file_path = f"{experiment_path}/{experiment_name}_id01.h5"
   print(f"Experiment file path: {experiment_file_path}")
   
   bliss_sample_name = ""  # Required
   bliss_dataset_name = ""  # Required
   scan = None  # Required
   
   sample_name = f"{bliss_sample_name}_{bliss_dataset_name}"
   
   loader = cdiutils.io.ID01Loader(
       experiment_file_path=experiment_file_path,
       sample_name=sample_name,
   )
   data = loader.load_detector_data(scan)
   
   print(f"Shape of the detector data is: {data.shape}")
   plt.figure(layout="tight")
   plt.imshow(data[data.shape[0]//2], norm=LogNorm())
   plt.colorbar()
   plt.show()

**Interactive Data Inspection**

Use the interactive plotter to explore your data:

.. code-block:: python

   # Inspect the detector data with an interactive slider plot
   cdiutils.plot.Plotter(data, plot="2D")

Data Cropping and Centring
--------------------------

Find the Bragg Peak position and crop the data appropriately:

.. code-block:: python

   # data.shape[0] represents the number of frames in the rocking curve
   output_shape = (data.shape[0], 150, 150)
   
   # Crop the data according to specified methods. 'methods' can be a list
   # of strings such as "com" or "max", or it can be a tuple specifying the
   # reference detector pixel. The list can be as long as needed. Note that
   # if there are hot pixels, "max" or "com" might not work effectively.
   (
       cropped_data,  # the output cropped data
       det_ref,  # the detector reference voxel in the full detector frame
       cropped_det_ref,  # the detector reference voxel in the cropped detector frame
       roi  # the region of interest (ROI) used to crop the data
   ) = cdiutils.utils.CroppingHandler.chain_centring(
       data,
       methods=["max", "com"],  # the list of methods used sequentially
       output_shape=output_shape,  # the output shape you want to work with
       verbose=True  # whether to print logs during the reference voxel search
   )
   
   # Plot the cropped detector data
   loader.plot_detector_data(cropped_data, f"Scan #{scan}", equal_limits=False)

Data Cleaning and Pre-processing
---------------------------------

Clean the data to improve its quality for analysis:

.. code-block:: python

   # Remove hot pixels using a median filter-based function
   cleaned_data, hot_pixel_mask = cdiutils.utils.hot_pixel_filter(cropped_data)
   
   # If you intend to remove background noise (e.g., fluorescence), set the
   # background level
   background_level = 4
   cleaned_data = np.where(
       cleaned_data - background_level > 0,
       cleaned_data - background_level, 0
   )
   
   # Plot the cleaned detector data
   loader.plot_detector_data(cleaned_data, f"Scan #{scan}")

**Optional: Apply Flat Field Correction**

If you have a flat field for the detector:

.. code-block:: python

   # If you have a flat field for the detector (at the correct energy), load and apply it.
   # flat_field_path = ""
   
   # with np.load(flat_field_path) as file:
   #     flat_field = file["arr_0"][
   #         cdiutils.utils.CroppingHandler.roi_list_to_slices(roi[2:])
   #     ]
   # cleaned_data = cropped_data * flat_field

**Load the Mask**

Load the appropriate detector mask:

.. code-block:: python

   mask = cdiutils.io.Loader.get_mask(
       detector_name=,  # Required,
       channel=cleaned_data.shape[0],  # or data.shape[0] depending on the cropping
       roi=roi
   )
   # mask *= hot_pixel_mask
   cdiutils.plot.plot_volume_slices(
       mask, title=f"Mask, scan #{scan}", norm=Normalize()
   )

Phase Retrieval with PyNX
-------------------------

This part requires the PyNX package for phase retrieval.

**Initialisation**

.. code-block:: python

   # Optionally load a support from a former analysis
   # good_run_path = (
   #     f"results/{sample_name}/S{scan}/pynx_phasing/.cxi"
   # )
   # with cdiutils.io.CXIFile(run_path) as file:
   #     good_support = file["entry_1/image_1/support"]

Initialise the PyNXPhaser, which is a wrapper to embed and initialise PyNX quickly:

.. code-block:: python

   # Set up phasing parameters
   params = {
       "support_update_period": 50,
       "support_threshold": "0.15, 0.35",
       "support_autocorrelation_threshold": (0.05, 0.11),
       "update_psf": 0,
       "psf": None,
       "show_cdi": 0,
       # "update_border_n": 4,
       # "post_expand": (-1,1),
       "rebin": "1, 1, 1",
       "scale_obj": "F",
       # "support_shape":"square",
       # "support_size": 20,
   }
   
   phaser = cdiutils.process.PyNXPhaser(
       iobs=cleaned_data,
       mask=mask,
       **params
   )
   
   # Initialise the CDI object. Support, or former cdi objects can be provided.
   phaser.init_cdi(
       # support=good_support,  # if you want to start from a known support
   )
   
   # Plot the first guess.
   phaser.plot_cdi(phaser.cdi)

**Running the Phase Retrieval**

.. code-block:: python

   # Define the recipe you'd like to run.
   recipe = "HIO**400, RAAR**500, ER**200"
   phaser.run_multiple_instances(run_nb=5, recipe=recipe)
   
   # The genetic phasing requires smaller recipes.
   # recipe = "HIO**50, RAAR**60, ER**40"
   # phaser.genetic_phasing(
   #     run_nb=5, genetic_pass_nb=10,
   #     recipe=recipe, selection_method="mean_to_max"
   # )

Plot the final results to get a first taste of the reconstructions:

.. code-block:: python

   for i, cdi in enumerate(phaser.cdi_list):
       phaser.plot_cdi(cdi, title=f"Run {i+1:04d}")

**Phasing Results Analysis**

The ``cdiutils.process.PhasingResultAnalyser`` class provides utility methods for analysing phase retrieval results:

.. code-block:: python

   analyser = cdiutils.process.PhasingResultAnalyser(cdi_results=phaser.cdi_list)
   
   analyser.analyse_phasing_results(
       sorting_criterion = "mean_to_max"
       # plot_phasing_results=False,  # Defaults to True
       # plot_phase=True,  # Defaults to False
   )

The ``sorting_criterion`` can be:

* ``"mean_to_max"``: The difference between the mean of the Gaussian fitting of the amplitude histogram and the maximum value of the amplitude
* ``"sharpness"``: The sum of the amplitude within the support raised to the power of 4
* ``"std"``: The standard deviation of the amplitude
* ``"llk"``: The log-likelihood of the reconstruction
* ``"llkf"``: The free log-likelihood of the reconstruction

**Select Best Candidates and Mode Decomposition**

.. code-block:: python

   analyser.select_best_candidates(
       # best_runs=[2, 5]
       nb_of_best_sorted_runs=3,
   )
   print(
       f"The best candidates selected are: {analyser.best_candidates}."
   )
   modes, mode_weight = analyser.mode_decomposition()
   
   mode = modes[0]  # Select the first mode

**Amplitude Distribution Analysis**

Check the amplitude distribution using the ``cdiutils.analysis.find_isosurface`` function:

.. code-block:: python

   isosurface, _ = cdiutils.analysis.find_isosurface(np.abs(mode), plot=True)

Define the support array and calculate oversampling ratio:

.. code-block:: python

   # isosurface = 0.45
   support = cdiutils.utils.make_support(np.abs(mode), isosurface=isosurface)
   
   ratios = cdiutils.utils.get_oversampling_ratios(support)
   print(
       "[INFO] The oversampling ratios in each direction are "
       + ", ".join(
           [f"axis{i}: {ratios[i]:.1f}" for i in range(len(ratios))]
       )
       + ".\nIf low-strain crystal, you can set PyNX 'rebin' parameter to "
               "(" + ", ".join([f"{r//2}" for r in ratios]) + ")"
   )

Plot the final amplitude and phase:

.. code-block:: python

   figure, axes = plt.subplots(2, 3, layout="tight", figsize=(6, 4))
   
   slices = cdiutils.utils.get_centred_slices(mode.shape)
   for i in range(3):
       amp_img = axes[0, i].imshow(np.abs(mode)[slices[i]])
       phase_img = axes[1, i].imshow(
           np.angle(mode)[slices[i]], cmap="cet_CET_C9s_r"
       )
   
       for ax in (axes[0, i], axes[1, i]):
           cdiutils.plot.add_colorbar(ax, ax.images[0])
           limits = cdiutils.plot.x_y_lim_from_support(support[slices[i]])
           ax.set_xlim(limits[0])
           ax.set_ylim(limits[1])

Orthogonalisation of Reconstructed Data
---------------------------------------

This part involves transforming from the detector frame to the XU/CXI frame, retrieving motor positions, building reciprocal space grids, and calculating transformation matrices.

**Load Motor Positions and Energy**

.. code-block:: python

   angles = loader.load_motor_positions(scan, roi=roi)
   energy = loader.load_energy(scan)  # or define it manually

**Detector Calibration Parameters**

To correctly compute the grid, detector calibration parameters are required:

.. code-block:: python

   det_calib_params = loader.load_det_calib_params(scan)  # depends on the beamline
   
   # Fill the detector calibration parameters manually if needed
   # det_calib_params = {
   #     "cch1": , # direct beam position vertical,
   #     "cch2": , # horizontal
   #     "pwidth1": 5.5e-05,  # detector pixel size in m, eiger: 7.5e-5, maxipix: 5.5e-5
   #     "pwidth2": 5.5e-05,  # detector pixel size in m
   #     "distance":,  # sample to detector distance in m
   #     "tiltazimuth": .0,
   #     "tilt": .0,
   #     "detrot": .0,
   #     "outerangle_offset": .0
   # }

**Define Geometry and Space Converter**

.. code-block:: python

   # Load the appropriate geometry
   geometry = cdiutils.Geometry.from_setup("ID01")
   
   # Initialise the space converter
   converter = cdiutils.SpaceConverter(
       geometry,
       det_calib_params,
       energy=energy,
       roi=roi[2:]
   )
   
   # The Q space area is initialised only for the selected roi used
   # before cropping the data
   converter.init_q_space(**angles)

**Verify Q-space Gridding**

Check if the Q-space gridding has worked properly:

.. code-block:: python

   # What Bragg reflection did you measure?
   hkl = [1, 1, 1]
   
   # Reminder: cropped_det_ref is the pixel reference chosen at the beginning. 
   # It is the very centre of the cropped data.
   q_lab_ref = converter.index_det_to_q_lab(cropped_det_ref)
   dspacing_ref = converter.dspacing(q_lab_ref)
   lattice_parameter_ref = converter.lattice_parameter(q_lab_ref, hkl)
   print(
       f"The d-spacing and 'effective' lattice parameter are respectively "
       f"{dspacing_ref:.4f} and {lattice_parameter_ref:.4f} angstroms.\n"
       "Is that what you expect?! -> If not, the detector calibration might "
       "be wrong."
   )

**Initialise Interpolators and Orthogonalise**

.. code-block:: python

   converter.init_interpolator(space="both", verbose=True)
   
   # This is the orthogonalised intensity
   ortho_intensity = converter.orthogonalise_to_q_lab(cleaned_data)
   
   # This is the regular Q-space grid
   qx, qy, qz = converter.get_q_lab_regular_grid()

Plot the intensity in the orthogonal Q-space:

.. code-block:: python

   q_spacing = [np.mean(np.diff(q)) for q in (qx, qy, qz)]
   q_centre = (qx.mean(), qy.mean(), qz.mean())
   
   figure, axes = cdiutils.plot.slice.plot_volume_slices(
       ortho_intensity,
       voxel_size=q_spacing,
       data_centre=q_centre,
       title="Orthogonalised intensity in the Q-lab frame",
       norm=LogNorm(),
       convention="xu",
       show=False
   )
   cdiutils.plot.add_labels(axes, space="rcp", convention="xu")
   display(figure)

**Orthogonalisation in Direct Space**

.. code-block:: python

   voxel_size = converter.direct_lab_voxel_size
   voxel_size = 20  # or define it manually
   
   ortho_obj = converter.orthogonalise_to_direct_lab(mode, voxel_size)
   voxel_size = converter.direct_lab_voxel_size
   print(f"The target voxel size is: {voxel_size} nm.")

Find isosurface and create support:

.. code-block:: python

   isosurface, _ = cdiutils.analysis.find_isosurface(np.abs(ortho_obj), plot=True)
   
   # isosurface = 0.3  # Choose the isosurface value if not happy with the estimated one
   ortho_support = cdiutils.utils.make_support(np.abs(ortho_obj), isosurface)

Visualise the orthogonalised data:

.. code-block:: python

   figures = {}
   axes = {}
   
   figures["amp"], axes["amp"] = cdiutils.plot.plot_volume_slices(
       np.abs(ortho_obj),
       support=ortho_support,
       voxel_size=voxel_size,
       data_centre=(0, 0, 0),
       convention="xu",
       title="Amplitude",
       show=False
   )
   cdiutils.plot.add_labels(axes["amp"], space="direct", convention="xu")
   
   figures["phase"], axes["phase"] = cdiutils.plot.plot_volume_slices(
       np.angle(ortho_obj) * ortho_support,
       support=ortho_support,
       data_centre=(0, 0, 0),
       voxel_size=voxel_size,
       cmap="cet_CET_C9s_r",
       convention="xu",
       vmin=-np.pi,
       vmax=np.pi,
       title="Phase (rad)",
       show=False
   )
   cdiutils.plot.add_labels(axes["phase"], space="direct", convention="xu")
   
   display(figures["amp"], figures["phase"])

**Convention Conversion**

Convert from XU to CXI convention:

.. code-block:: python

   cxi_ortho_obj = geometry.swap_convention(ortho_obj)
   cxi_ortho_support = geometry.swap_convention(ortho_support)
   cxi_voxel_size = geometry.swap_convention(voxel_size)

Extracting Structural Properties
--------------------------------

Use the ``PostProcessor`` class to extract quantitative structural properties. First, optionally flip and/or apodise the reconstruction:

.. code-block:: python

   # cxi_ortho_obj = cdiutils.process.PostProcessor.flip_reconstruction(cxi_ortho_obj)
   cxi_ortho_obj = cdiutils.process.PostProcessor.apodize(cxi_ortho_obj, "blackman")

Extract structural properties using the comprehensive method:

.. code-block:: python

   struct_props = cdiutils.process.PostProcessor.get_structural_properties(
       cxi_ortho_obj,
       isosurface=0.4,
       g_vector=geometry.swap_convention(q_lab_ref),
       hkl=hkl,
       voxel_size=cxi_voxel_size,
       handle_defects=False  # this is whether you expect a defect.
   )
   
   for prop, value in struct_props.items():
       print(f"{prop}: ", end="")
       if isinstance(value, (np.ndarray)) and value.ndim > 1:
           print(f"3D array of shape: {value.shape}")
       elif isinstance(value, (list, tuple)):
           if isinstance(value[0], np.ndarray):
               print(f"tuple or list of length = {len(value)}")
           else:
               print(value)
       else:
           print(value)

The ``PostProcessor`` provides methods including:

* ``flip_reconstruction``: Needed if you determine that you have the complex conjugate solution
* ``apodize``: Required to avoid high-frequency artefacts using 3D windows (blackman, hamming, hann, etc.)
* ``unwrap_phase``: Unwraps the phase for voxels within the support
* ``remove_phase_ramp``: Removes phase ramps using linear regression
* ``get_displacement``: Computes displacement using phase and reciprocal space node position
* ``get_het_normal_strain``: Extracts heterogeneous normal strain with multiple methods
* ``get_structural_properties``: Comprehensive method generating displacement, strain, d-spacing, and lattice parameter maps

Visualisation and Analysis
---------------------------

Create a comprehensive summary plot:

.. code-block:: python

   to_plot = {
       k: struct_props[k]
       for k in [
           "amplitude", "phase", "displacement", "het_strain", "lattice_parameter"
       ]
   }
   
   table_info = {
       "Isosurface": isosurface,
       "Averaged Lat. Par. (Å)":np.nanmean(struct_props["lattice_parameter"]),
       "Averaged d-spacing (Å)": np.nanmean(struct_props["dspacing"])
   }
   
   summary_fig = cdiutils.pipeline.PipelinePlotter.summary_plot(
       title=f"Summary figure, Scan #{scan}",
       support=struct_props["support"],
       table_info=table_info,
       voxel_size=cxi_voxel_size,
       **to_plot
   )

Create 3D strain visualisation:

.. code-block:: python

   fig = cdiutils.plot.volume.plot_3d_surface_projections(
       data=struct_props["het_strain"],
       support=struct_props["support"],
       voxel_size=cxi_voxel_size,
       cmap="cet_CET_D13",
       vmin=-np.nanmax(np.abs(struct_props["het_strain"])),
       vmax=np.nanmax(np.abs(struct_props["het_strain"])),
       cbar_title=r"Strain (%)",
       title=f"3D views of the strain, Scan #{scan}"
   )

Data Saving
-----------

Save the processed data in multiple formats:

.. code-block:: python

   # Provide the path of the directory you want to save the data in
   dump_dir = f"results/{sample_name}/S{scan}_step_by_step/" 
   
   if os.path.isdir(dump_dir):
       print(
           "[INFO] Dump directory already exists, results will be saved in\n",
           dump_dir
       )
   else:
       print(f"[INFO] Creating the dump directory at: {dump_dir}")
       os.makedirs(dump_dir, exist_ok=True)
   
   # Select the data you want to save
   to_save = {
       "isosurface": isosurface,
       "q_lab_reference": q_lab_ref,
       "dspacing_reference": dspacing_ref,
       "lattice_parameter_reference": lattice_parameter_ref
   }
   
   to_save.update(struct_props)
   
   # Save as .npz file
   np.savez(f"{dump_dir}/S{scan}_structural_properties.npz", **to_save)

Save as VTI file for 3D visualisation:

.. code-block:: python

   # Save as .vti file
   # This is for 3D visualisation, so we do not need to save everything.
   to_save_as_vti = {
       k: struct_props[k]
       for k in [
           "amplitude", "support", "phase", "displacement", "het_strain",
           "het_strain_from_dspacing", "lattice_parameter",
           "dspacing"
       ]
   }
   
   # Also, we want to avoid nan values as they will mess up the visualisation.
   # Therefore, nan value are replaced by average value of the quantity.
   for k in (
       "het_strain", "het_strain_from_dspacing", "dspacing",
       "lattice_parameter", "displacement"
   ):
       to_save_as_vti[k] = np.where(
           np.isnan(to_save_as_vti[k]),
           np.nanmean(to_save_as_vti[k]),
           to_save_as_vti[k]
       )
   
   cdiutils.io.save_as_vti(
       f"{dump_dir}/S{scan}_structural_properties.vti",
       voxel_size=voxel_size,
       cxi_convention=True,
       **to_save_as_vti
   )
   
   print("\n[INFO] data saved.")

Additional Visualisation
------------------------

Plot individual structural properties using CDIutils plotting functions:

.. code-block:: python

   _, _, plot_configs = cdiutils.plot.set_plot_configs()
   for prop in (
           "amplitude", "support", "phase",
           "displacement", "het_strain", "dspacing"
   ):
       figures[prop], axes[prop] = cdiutils.plot.slice.plot_volume_slices(
           struct_props[prop]*cdiutils.utils.zero_to_nan(struct_props["support"]),
           support=struct_props["support"],
           voxel_size=cxi_voxel_size,
           data_centre=(0, 0, 0),
           vmin=plot_configs[prop]["vmin"],
           vmax=plot_configs[prop]["vmax"],
           cmap=plot_configs[prop]["cmap"],
           title=prop,
           show=False
       )
       cdiutils.plot.add_labels(axes[prop])
       display(figures[prop])

Next Steps
----------

After mastering manual processing, you can:

* Automate frequently used workflows with the :doc:`pipeline_tutorial`
* Explore advanced analysis in :doc:`../examples/bcdi_reconstruction_analysis` 
* Learn detector calibration techniques in :doc:`detector_calibration_tutorial`
* Develop custom processing methods for your specific research needs
