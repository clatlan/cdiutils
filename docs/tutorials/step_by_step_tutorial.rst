Step-by-Step BCDI Analysis Tutorial
===================================

This tutorial provides a detailed, step-by-step guide for manual BCDI data analysis using **cdiutils**. This approach gives you full control over each processing step and is based on the authentic CDIutils template notebooks.

Overview
--------

This tutorial covers the complete BCDI analysis workflow:

1. **Preprocessing**: Loading, cropping, and cleaning detector data
2. **Phasing**: Running phase retrieval algorithms
3. **Orthogonalisation**: Converting to laboratory coordinate system
4. **Analysis**: Extracting quantitative structural properties
5. **Visualisation**: Creating plots and saving results

Unlike the automated pipeline approach, this manual method allows you to inspect and modify each step according to your specific needs.

Setup and Imports
-----------------

Initial Setup
^^^^^^^^^^^^^

Start by importing the necessary libraries and setting up plotting parameters:

.. code-block:: python

    from IPython.display import display
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    import numpy as np
    import os
    
    import cdiutils
    
    # update matplotlib parameters for better plots
    cdiutils.plot.update_plot_params()

1. Preprocessing
----------------

1.1 Loading the Data
^^^^^^^^^^^^^^^^^^^^

First, explore your experiment directory structure and load the detector data:

.. code-block:: python

    # set experiment parameters
    experiment_path = "/path/to/RAW_DATA"  # path to experiment RAW_DATA directory
    experiment_name = "experiment_name"    # your experiment name
    experiment_file_path = f"{experiment_path}/{experiment_name}_id01.h5"
    
    # define sample and scan information
    bliss_sample_name = "sample_name"      # your sample name in BLISS
    bliss_dataset_name = "dataset_name"    # your dataset name in BLISS
    scan = 123                             # scan number to analyse
    
    sample_name = f"{bliss_sample_name}_{bliss_dataset_name}"
    
    # create loader instance
    loader = cdiutils.io.ID01Loader(
        experiment_file_path=experiment_file_path,
        sample_name=sample_name,
    )
    
    # load detector data
    data = loader.load_detector_data(scan)
    print(f"Shape of the detector data is: {data.shape}")
    
    # quick visualisation
    plt.figure(layout="tight")
    plt.imshow(data[data.shape[0]//2], norm=LogNorm())
    plt.colorbar()
    plt.show()

Interactive Data Inspection
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use CDIutils' interactive plotting capabilities:

.. code-block:: python

    # interactive 2D plot with slider for 3D data inspection
    cdiutils.plot.Plotter(data, plot="2D")

1.2 Cropping the Data
^^^^^^^^^^^^^^^^^^^^^

Find and crop around the Bragg peak:

.. code-block:: python

    # Define output shape for cropped data
    output_shape = (data.shape[0], 150, 150)
    
    # Crop data using chain of centering methods
    # Methods can be "com" (center of mass), "max" (maximum), or coordinate tuples
    (
        cropped_data,       # Output cropped data
        det_ref,           # Detector reference voxel in full frame
        cropped_det_ref,   # Detector reference voxel in cropped frame
        roi                # Region of interest used for cropping
    ) = cdiutils.utils.CroppingHandler.chain_centring(
        data,
        methods=["max", "com"],    # Sequential methods for centering
        output_shape=output_shape,
        verbose=True              # Print progress information
    )
    
    # Visualize cropped data
    loader.plot_detector_data(cropped_data, f"Scan #{scan}", equal_limits=False)

1.3 Cleaning the Data
^^^^^^^^^^^^^^^^^^^^^

Remove hot pixels, apply flat field corrections, and remove background:

.. code-block:: python

    # Optional: Apply flat field correction
    # flat_field_path = "/path/to/flat_field.npz"
    # with np.load(flat_field_path) as file:
    #     flat_field = file["arr_0"][
    #         cdiutils.utils.CroppingHandler.roi_list_to_slices(roi[2:])
    #     ]
    # cleaned_data = cropped_data * flat_field
    
    # Remove hot pixels using median filter
    cleaned_data, hot_pixel_mask = cdiutils.utils.hot_pixel_filter(cropped_data)
    
    # Remove background (e.g., fluorescence)
    background_level = 4
    cleaned_data = np.where(
        cleaned_data - background_level > 0,
        cleaned_data - background_level, 0
    )
    
    # Plot cleaned data
    loader.plot_detector_data(cleaned_data, f"Scan #{scan}")

2. Phasing
----------

This section requires PyNX package for phase retrieval.

2.1 Load the Mask
^^^^^^^^^^^^^^^^^

Load the detector mask with the correct shape:

.. code-block:: python

    mask = cdiutils.io.Loader.get_mask(
        detector_name="eiger2m",  # or your detector name
        channel=cleaned_data.shape[0],  # or data.shape[0] depending on the cropping
        roi=roi
    )
    # mask *= hot_pixel_mask  # optionally combine with hot pixel mask
    cdiutils.plot.plot_volume_slices(
        mask, title=f"Mask, scan #{scan}", norm=Normalize()
    )

2.2 Initialize PyNX Phaser
^^^^^^^^^^^^^^^^^^^^^^^^^^

Set up the PyNXPhaser with parameters:

.. code-block:: python

    # Optionally load a support from a former analysis
    # good_run_path = f"results/{sample_name}/S{scan}/pynx_phasing/.cxi"
    # with cdiutils.io.CXIFile(run_path) as file:
    #     good_support = file["entry_1/image_1/support"]
    
    # Initialize the PyNXPhaser. It is wrapper to embed and initialize PyNX
    # quickly. iobs (observed intensity) and mask are required. The
    # parameters (params) are optional, since most of them have a default value.
    
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
    
    # Initialize the CDI object. Support, or former cdi objects can be provided.
    phaser.init_cdi(
        # support=good_support,  # if you want to start from a known support
    )
    
    # Plot the first guess
    phaser.plot_cdi(phaser.cdi)

2.3 Running Phase Retrieval
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Execute the phase retrieval algorithm:

.. code-block:: python

    # Define the recipe you'd like to run
    recipe = "HIO**400, RAAR**500, ER**200"
    phaser.run_multiple_instances(run_nb=5, recipe=recipe)
    
    # Alternative: The genetic phasing requires smaller recipes
    # recipe = "HIO**50, RAAR**60, ER**40"
    # phaser.genetic_phasing(
    #     run_nb=5, genetic_pass_nb=10,
    #     recipe=recipe, selection_method="mean_to_max"
    # )
    
    # Plot the final results
    for i, cdi in enumerate(phaser.cdi_list):
        phaser.plot_cdi(cdi, title=f"Run {i+1:04d}")

2.4 Phasing Results Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze the phase retrieval results:

.. code-block:: python

    analyser = cdiutils.process.PhasingResultAnalyser(cdi_results=phaser.cdi_list)
    
    analyser.analyse_phasing_results(
        sorting_criterion="mean_to_max"
        # plot_phasing_results=False,  # Defaults to True
        # plot_phase=True,  # Defaults to False
    )

2.5 Select Best Candidates and Mode Decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Select the best reconstructions and perform mode decomposition:

.. code-block:: python

    analyser.select_best_candidates(
        # best_runs=[2, 5]  # manually select runs
        nb_of_best_sorted_runs=3,  # or automatically select best 3
    )
    print(f"The best candidates selected are: {analyser.best_candidates}.")
    
    modes, mode_weight = analyser.mode_decomposition()
    mode = modes[0]  # Select the first mode

2.6 Check Amplitude Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze the amplitude distribution to find the appropriate isosurface:

.. code-block:: python

    isosurface, _ = cdiutils.analysis.find_isosurface(np.abs(mode), plot=True)
    
    # Define support array and calculate oversampling ratio
    # isosurface = 0.45  # or define manually if not happy with the estimate
    support = cdiutils.utils.make_support(np.abs(mode), isosurface=isosurface)
    
    ratios = cdiutils.utils.get_oversampling_ratios(support)
    print(
        "[INFO] The oversampling ratios in each direction are "
        + ", ".join([f"axis{i}: {ratios[i]:.1f}" for i in range(len(ratios))])
        + ".\nIf low-strain crystal, you can set PyNX 'rebin' parameter to "
        + "(" + ", ".join([f"{r//2}" for r in ratios]) + ")"
    )

2.7 Plot Final Reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualize the amplitude and phase of the final object:

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

3. Orthogonalisation
--------------------

Transform from the detector frame to the XU/CXI frame.

3.1 Load Motor Positions and Energy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Load the motor positions and beam energy:

.. code-block:: python

    angles = loader.load_motor_positions(scan, roi=roi)
    energy = loader.load_energy(scan)  # or define it manually

3.2 Load Detector Calibration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get the detector calibration parameters:

.. code-block:: python

    det_calib_params = loader.load_det_calib_params(scan)  # depends on the beamline
    
    # Or fill the detector calibration parameters manually
    # det_calib_params = {
    #     "cch1": ,  # direct beam position vertical,
    #     "cch2": ,  # horizontal
    #     "pwidth1": 5.5e-05,  # detector pixel size in m, eiger: 7.5e-5, maxipix: 5.5e-5
    #     "pwidth2": 5.5e-05,  # detector pixel size in m
    #     "distance": ,  # sample to detector distance in m
    #     "tiltazimuth": .0,
    #     "tilt": .0,
    #     "detrot": .0,
    #     "outerangle_offset": .0
    # }

3.3 Define Geometry and Initialize Space Converter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set up the experimental geometry:

.. code-block:: python

    # Load the appropriate geometry
    geometry = cdiutils.Geometry.from_setup("ID01")
    
    # Initialize the space converter
    converter = cdiutils.SpaceConverter(
        geometry,
        det_calib_params,
        energy=energy,
        roi=roi[2:]
    )
    
    # The Q space area is initialized only for the selected roi used
    # before cropping the data
    converter.init_q_space(**angles)

3.4 Check Q-space Gridding
^^^^^^^^^^^^^^^^^^^^^^^^^^

Verify the Q-space gridding has worked properly:

.. code-block:: python

    # What Bragg reflection did you measure?
    hkl = [1, 1, 1]
    
    # cropped_det_ref is the pixel reference chosen at the beginning
    # It is the very centre of the cropped data
    q_lab_ref = converter.index_det_to_q_lab(cropped_det_ref)
    dspacing_ref = converter.dspacing(q_lab_ref)
    lattice_parameter_ref = converter.lattice_parameter(q_lab_ref, hkl)
    print(
        f"The d-spacing and 'effective' lattice parameter are respectively "
        f"{dspacing_ref:.4f} and {lattice_parameter_ref:.4f} angstroms.\n"
        "Is that what you expect?! -> If not, the detector calibration might "
        "be wrong."
    )

3.5 Initialize Interpolators and Orthogonalize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set up interpolators and perform orthogonalization:

.. code-block:: python

    # Initialize the interpolators in both reciprocal and direct spaces
    converter.init_interpolator(space="both", verbose=True)
    
    # This is the orthogonalized intensity
    ortho_intensity = converter.orthogonalise_to_q_lab(cleaned_data)
    
    # This is the regular Q-space grid
    qx, qy, qz = converter.get_q_lab_regular_grid()
    
    # Plot the intensity in the orthogonal Q-space
    q_spacing = [np.mean(np.diff(q)) for q in (qx, qy, qz)]
    q_centre = (qx.mean(), qy.mean(), qz.mean())
    
    figure, axes = cdiutils.plot.slice.plot_volume_slices(
        ortho_intensity,
        voxel_size=q_spacing,
        data_centre=q_centre,
        title="Orthogonalized intensity in the Q-lab frame",
        norm=LogNorm(),
        convention="xu",
        show=False
    )
    cdiutils.plot.add_labels(axes, space="rcp", convention="xu")
    display(figure)

3.6 Orthogonalization in Direct Space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Convert to direct laboratory coordinates:

.. code-block:: python

    # The voxel size can be changed here (must be a float, tuple, list, or np.ndarray in nm)
    # If not specified, the previously determined voxel size will be used
    voxel_size = converter.direct_lab_voxel_size
    voxel_size = 20  # or define it manually
    
    ortho_obj = converter.orthogonalise_to_direct_lab(mode, voxel_size)
    voxel_size = converter.direct_lab_voxel_size
    print(f"The target voxel size is: {voxel_size} nm.")
    
    # Find isosurface for the orthogonalized object
    isosurface, _ = cdiutils.analysis.find_isosurface(np.abs(ortho_obj), plot=True)
    
    # Create support
    # isosurface = 0.3  # Choose the isosurface value if not happy with the estimated one
    ortho_support = cdiutils.utils.make_support(np.abs(ortho_obj), isosurface)

3.7 Plot Orthogonalized Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualize the orthogonalized amplitude and phase:

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

3.8 Convention Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^

Convert from XU to CXI convention:

.. code-block:: python

    # Convert from XU convention to CXI convention
    cxi_ortho_obj = geometry.swap_convention(ortho_obj)
    cxi_ortho_support = geometry.swap_convention(ortho_support)
    cxi_voxel_size = geometry.swap_convention(voxel_size)

4. Extracting Quantitative Properties
-------------------------------------

Use the PostProcessor class to extract structural properties.

4.1 Apply Optional Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optionally flip and/or apodize the reconstruction:

.. code-block:: python

    # Optionally flip reconstruction if you have the complex conjugate solution
    # cxi_ortho_obj = cdiutils.process.PostProcessor.flip_reconstruction(cxi_ortho_obj)
    
    # Apodize to avoid high-frequency artifacts
    cxi_ortho_obj = cdiutils.process.PostProcessor.apodize(cxi_ortho_obj, "blackman")

4.2 Extract Structural Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get all structural properties using the PostProcessor:

.. code-block:: python

    struct_props = cdiutils.process.PostProcessor.get_structural_properties(
        cxi_ortho_obj,
        isosurface=0.4,
        g_vector=geometry.swap_convention(q_lab_ref),
        hkl=hkl,
        voxel_size=cxi_voxel_size,
        handle_defects=False  # this is whether you expect a defect
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

5. Plotting and Visualization
-----------------------------

5.1 Summary Plot
^^^^^^^^^^^^^^^^

Create a comprehensive summary figure:

.. code-block:: python

    to_plot = {
        k: struct_props[k]
        for k in [
            "amplitude", "phase", "displacement", "het_strain", "lattice_parameter"
        ]
    }
    
    table_info = {
        "Isosurface": isosurface,
        "Averaged Lat. Par. (Å)": np.nanmean(struct_props["lattice_parameter"]),
        "Averaged d-spacing (Å)": np.nanmean(struct_props["dspacing"])
    }
    
    summary_fig = cdiutils.pipeline.PipelinePlotter.summary_plot(
        title=f"Summary figure, Scan #{scan}",
        support=struct_props["support"],
        table_info=table_info,
        voxel_size=cxi_voxel_size,
        **to_plot
    )

5.2 3D Strain Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create 3D surface projections of the strain:

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

5.3 Individual Property Plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Plot individual structural properties:

.. code-block:: python

    _, _, plot_configs = cdiutils.plot.set_plot_configs()
    figures = {}
    axes = {}
    
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

6. Saving Results
-----------------

6.1 Create Output Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set up the directory for saving results:

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

6.2 Save as NPZ Format
^^^^^^^^^^^^^^^^^^^^^^

Save the structural properties and metadata:

.. code-block:: python

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

6.3 Save as VTI Format for 3D Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save in VTI format for 3D visualization software:

.. code-block:: python

    # Save as .vti file
    # This is for 3D visualization, so we do not need to save everything
    to_save_as_vti = {
        k: struct_props[k]
        for k in [
            "amplitude", "support", "phase", "displacement", "het_strain",
            "het_strain_from_dspacing", "lattice_parameter", "dspacing"
        ]
    }
    
    # Avoid nan values as they will mess up the visualization
    # Therefore, nan values are replaced by average value of the quantity
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

Summary
-------

This step-by-step approach provides complete control over the BCDI analysis workflow. Key advantages include:

- **Flexibility**: Modify each step according to your data characteristics
- **Transparency**: Understand exactly what processing is applied
- **Debugging**: Easy to identify and fix issues at specific steps
- **Customization**: Add custom processing steps as needed

The manual approach is particularly useful for:

- Testing new processing methods
- Handling unusual or problematic data
- Research and development work
- Learning the BCDI analysis process

For routine processing of similar datasets, consider using the automated :doc:`pipeline_tutorial` approach instead.

Next Steps
----------

- Try the automated :doc:`pipeline_tutorial` for streamlined processing
- Learn about :doc:`detector_calibration_tutorial` for geometry optimization
- Explore the examples in ``examples/`` directory
- Check template notebooks in ``src/cdiutils/templates/`` for working code
