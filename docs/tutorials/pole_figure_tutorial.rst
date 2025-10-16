Pole Figure Tutorial
====================

This tutorial demonstrates how to create crystallographic Generate a pole figure with default parameters:

.. code-block:: python

    # basic usage with default parameters
    (grid_x, grid_y, projected_int), (fig, ax) = cdiutils.analysis.pole_figure(
        data,
        [qx, qy, qz],
        radius=0.020,
        dr=0.0002, 
        axis="2",
        norm=LogNorm(1, ),
        verbose=True,
    )

Setup and Initialisation
-------------------------

Start by importing the necessary libraries and setting up plotting parameters:

.. code-block:: python

    from matplotlib.colors import LogNorm
    import numpy as np

    import cdiutils

    cdiutils.plot.update_plot_params()

Load the Data
-------------

Specify the path to the data that contains the orthogonalised Bragg peak, along with the corresponding grid of q values.

If you have run the ``BcdiPipeline.preprocess()`` function, you can find the data in the results folder, in the ``"S<scan>_preprocessed_data.cxi"`` file:

.. code-block:: python

    path = ("path/to/data.cxi")

Load and Plot the Data
----------------------

Load the orthogonalised diffraction data and corresponding q-space grid:

.. code-block:: python

    with cdiutils.CXIFile(path, "r") as cxi:
        data = cxi["entry_1/data_2/data"]
        qx = cxi["entry_1/result_2/qx_xu"]
        qy = cxi["entry_1/result_2/qy_xu"]
        qz = cxi["entry_1/result_2/qz_xu"]
        shift = cxi["entry_1/result_2/q_space_shift"]

    print(qx.shape, qy.shape, qz.shape, data.shape)
    voxel_size = (
        np.diff(qx).mean(),
        np.diff(qy).mean(),
        np.diff(qz).mean()
    )

    fig, axes = cdiutils.plot.plot_volume_slices(
        data, voxel_size=voxel_size, data_centre=shift,
        norm=LogNorm(), convention="xu", show=False
    )

    cdiutils.plot.add_labels(axes, convention="xu")
    fig

Usage of ``cdiutils.analysis.pole_figure``
-------------------------------------------

The ``cdiutils.analysis.pole_figure`` function generates a crystallographic pole figure using stereographic projection. This method maps 3D diffraction intensity data onto a 2D plane, providing a visual representation of the crystallographic orientation distribution.

Parameters
^^^^^^^^^^

- **intensity (np.ndarray)**: A 3D array of intensity values representing the diffraction data.
- **grid (list)**: A list of 1D arrays defining the orthogonal grid (e.g., ``[x_coords, y_coords, z_coords]``).
- **axis (str, optional)**: Specifies the projection axis and hemisphere:
  
  - ``"0"``, ``"1"``, ``"2"``: Upper hemisphere projection onto the equatorial plane.
  - ``"-0"``, ``"-1"``, ``"-2"``: Lower hemisphere projection onto the equatorial plane.
  - The absolute value of the axis determines the normal plane:
    
    - ``|axis|=0``: Project onto the yz-plane (normal to x-axis).
    - ``|axis|=1``: Project onto the xz-plane (normal to y-axis).
    - ``|axis|=2``: Project onto the xy-plane (normal to z-axis).
    
  - Defaults to ``"2"`` (upper hemisphere projection onto the xy-plane).

- **radius (float, optional)**: Radius of the spherical shell for data selection. Defaults to ``None`` (0.25 * max radial distance).
- **dr (float, optional)**: Thickness of the spherical shell. Defaults to ``None`` (0.01 * radius).
- **resolution (int, optional)**: Resolution of the output 2D grid (number of points per dimension). Defaults to ``250``.
- **figsize (tuple, optional)**: Size of the output figure. Defaults to ``(4, 4)``.
- **title (str, optional)**: Title for the plot. Defaults to ``None``.
- **verbose (bool, optional)**: If ``True``, prints and plots additional information. Defaults to ``False``.
- **save (str, optional)**: File path to save the plot. Defaults to ``None``.
- **plot_params (dict, optional)**: Additional parameters for the plotting function.

Returns
^^^^^^^

- **tuple**: 
  
  - ``(grid_x, grid_y, projected_intensity)``: The 2D grid coordinates and intensity values.
  - ``(fig, ax)``: The figure and axis objects for the plot.

Example Usage
^^^^^^^^^^^^^

Basic usage with default parameters:

.. code-block:: python

    # basic usage with default parameters
    (grid_x, grid_y, projected_int), (fig, ax) = cdiutils.analysis.pole_figure(
        data,
        [qx, qy, qz],
        radius=0.020,
        dr=0.0002, 
        axis="2",
        norm=LogNorm(1, ),
        verbose=True,
    )

Advanced Usage
--------------

You can customise various aspects of the pole figure generation:

Projection Direction
^^^^^^^^^^^^^^^^^^^^

Change the projection axis to view different orientations:

.. code-block:: python

    # project onto different planes
    # upper hemisphere projection onto yz-plane (normal to x-axis)
    (grid_x, grid_y, projected_int), (fig, ax) = cdiutils.analysis.pole_figure(
        data, [qx, qy, qz], axis="0", radius=0.020, dr=0.0002
    )

    # lower hemisphere projection onto xy-plane (normal to z-axis)
    (grid_x, grid_y, projected_int), (fig, ax) = cdiutils.analysis.pole_figure(
        data, [qx, qy, qz], axis="-2", radius=0.020, dr=0.0002
    )

Spherical Shell Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

Adjust the spherical shell thickness and radius for different data characteristics:

.. code-block:: python

    # thicker shell for noisy data
    (grid_x, grid_y, projected_int), (fig, ax) = cdiutils.analysis.pole_figure(
        data, [qx, qy, qz], 
        radius=0.025,      # larger radius
        dr=0.005,          # thicker shell
        resolution=300,    # higher resolution
        verbose=True
    )

Custom Visualisation
^^^^^^^^^^^^^^^^^^^^

Customise the appearance and save the result:

.. code-block:: python

    # custom visualisation with saving
    (grid_x, grid_y, projected_int), (fig, ax) = cdiutils.analysis.pole_figure(
        data, [qx, qy, qz],
        radius=0.020,
        dr=0.0002,
        figsize=(6, 6),
        title="Custom Pole Figure",
        save="pole_figure.png",
        plot_params={
            "cmap": "plasma",
            "interpolation": "bilinear"
        }
    )

Interpretation
--------------

Pole figures help identify:

- **Crystallographic texture**: Preferred orientations in the sample
- **Symmetry**: Crystal symmetries reflected in the intensity distribution  
- **Defects**: Deviations from expected crystallographic patterns
- **Twinning**: Multiple crystallographic domains

The intensity distribution on the pole figure corresponds to the likelihood of finding crystallographic directions at specific orientations relative to the sample coordinate system.

Tips for Analysis
-----------------

1. **Choose appropriate radius**: Should encompass the Bragg peak but avoid noise
2. **Adjust shell thickness**: Balance between statistics and angular resolution
3. **Compare different projections**: Use multiple axes to get complete picture
4. **Consider symmetry**: Expected crystallographic symmetries should be visible
5. **Normalise intensity**: Use logarithmic scale for wide dynamic range

Credits
-------

This notebook was created by Clément Atlan, ESRF, 2025. It is part of the ``cdiutils`` package, which provides tools for BCDI data analysis and visualisation.

If you have used this notebook or the ``cdiutils`` package in your research, please consider citing the package: https://github.com/clatlan/cdiutils/

You'll find the citation information in the ``cdiutils`` package documentation.

.. code-block:: bibtex

    @software{Atlan_Cdiutils_A_python,
    author = {Atlan, Clement},
    doi = {10.5281/zenodo.7656853},
    license = {MIT},
    title = {{Cdiutils: A python package for Bragg Coherent Diffraction Imaging processing, analysis and visualisation workflows}},
    url = {https://github.com/clatlan/cdiutils},
    version = {0.2.0}
    }
