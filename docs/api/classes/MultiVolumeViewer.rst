MultiVolumeViewer (multi-layer 3D viewer)
=========================================

.. currentmodule:: cdiutils.interactive

.. autoclass:: MultiVolumeViewer
   :members: from_cdiutils, launcher, set_data, set_condition, get_volume,
      sample, add_slice, add_plane,
      add_clip, add_difference, add_region, add_lines, add_fitted_line,
      add_arrows, add_axes_triad, add_scale_bar, set_layer_params, set_units,
      rename_layer, delete_layer, probe, line_profile, facet_stats,
      facet_patches, save_facet_csv,
      region_stats, set_facet_settings, set_ortho_center, link_cameras,
      export_mesh, layer_mesh, get_view_state, set_view_state, save_state,
      load_state, run_actions, run_actions_async, validate_actions,
      wait_for_script, script, to_actions, save_actions, help, tutorial,
      write_image, export_html, fit_aspect, set_figure_preset, cache_info,
      clear_caches,
      wait_for_refresh,
      start_rotation_export, start_layer_export, start_condition_export,
      wait_for_export, show
   :show-inheritance:

   Jupyter viewer for several scalar volumes on a shared voxel grid
   (amplitude, phase, displacement, strain, ...), built on Plotly and
   ipywidgets. It is also what ``Plotter(..., plot="layers")`` opens.

Coordinates
-----------

Arrays are indexed ``(z, y, x)`` and ``voxel_size`` is ``(vz, vy, vx)``.
Normals, origins and axis names typed in the UI or passed to the ``add_*``
methods are in *plot* order, which follows ``PLOT_ORDER`` (``"zyx"`` by
default: plot x is the array's first axis). Line and arrow helpers accept
``coords="index"`` (voxel indices, e.g. from ``np.argwhere``),
``"physical"`` or ``"plot"``.

Examples
--------

cdiutils results (configuration printed first; several scans become
conditions)::

    viewer = MultiVolumeViewer.from_cdiutils(
        ["results/S101", "results/S102", "results/S103"],
        label_from="pressure",          # or labels=[...], default: scan
        fields=["amplitude", "phase", "het_strain"],
    )
    MultiVolumeViewer.launcher("results")   # the same, from a form


Several conditions, a clip coloured by strain and dislocation lines::

    from cdiutils.interactive import MultiVolumeViewer
    from cdiutils.analysis.dislocation import extract_structure, fit_line_3d

    viewer = MultiVolumeViewer(
        [data_0bar, data_1bar],            # dicts {name: 3D array}, same keys
        voxel_size=(5, 5, 5), unit="nm",
        condition_labels=["0 bar", "1 bar"],
        orientation_matrix=U,              # n_crystal = U @ n_lab
        facet_families="100, 110, 111",
    )
    viewer.add_clip("cut", "amplitude", normal=(1, 0, 0))
    viewer.set_layer_params("cut", color_by="het_strain", cmap="RdBu_r")
    centroid, direction = fit_line_3d(extract_structure(dislo_mask))
    viewer.add_fitted_line("dislocation", centroid, direction)
    viewer.add_difference("delta phase", "phase", ref=0)
    viewer.set_units(het_strain=("%", 100))            # strain shown in %
    viewer.set_layer_params("cut", range_mode="sym", range_all=True)
    viewer.show()

    viewer.facet_stats("amplitude", "het_strain")
    viewer.region_stats("het_strain", "amplitude", threshold=0.3, depth=10)

Exports (MP4, GIF or a folder of PNG frames) run in the background; in a
notebook, ``await`` them to chain several::

    await viewer.start_rotation_export("orbit_z", frames=180, fmt="mp4",
                                       name="rotation")
    await viewer.start_layer_export("cut", "offset", (-60, 60), points=90,
                                    fmt="gif", name="cut_sweep")
    await viewer.start_condition_export(hold=20, fmt="mp4", name="pressure")

Scripts, help and tutorial
--------------------------

Everything in the panels can be scripted. A step is the action name with its
main argument, then the other parameters; the script is checked before
anything runs, re-running a script replaces the layers it created, and steps
after an export wait for it::

    viewer.run_actions([
        {"only": "amplitude"},
        {"clip": "half", "source": "amplitude", "normal": [1, 0, 0],
         "color_by": "het_strain", "cmap": "RdBu_r"},
        {"camera": "iso", "zoom": 1.2},
        {"rotation": "orbit_z", "frames": 120, "name": "spin"},
    ])

    viewer.save_actions("view.yaml")     # the current view as a script
    viewer.run_actions("view.yaml")      # replay it
    viewer.script().only("amplitude").slice("zmid", source="phase").run()

``viewer.help()`` lists the help topics, ``viewer.help("actions")`` every
action, ``viewer.help("clip")`` one action with examples using your layer
names, and ``viewer.tutorial()`` is a guided tour on your own data (also in
the *Help & tutorial* section of the panel).

See Also
--------
:class:`ThreeDViewer` : single-volume isosurface viewer
:class:`Plotter` : ``plot="layers"`` opens a MultiVolumeViewer
