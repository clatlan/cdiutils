Interactive 3D Visualisation Tutorial
======================================

This tutorial demonstrates the interactive 3D visualisation tools in cdiutils, which provide modern, feature-rich interfaces for exploring volumetric data from BCDI reconstructions.

.. note::
   The interactive 3D features require additional dependencies. Install with:
   
   .. code-block:: bash
   
      pip install cdiutils[interactive]

Overview
--------

cdiutils provides two main tools for interactive 3D visualisation:

1. **plot_3d_isosurface** - Quick function for interactive isosurface plotting with multiple quantities
2. **ThreeDViewer** - Widget class for exploring complex 3D arrays (amplitude + phase)

Both tools use **Plotly** for modern, performant 3D rendering with extensive interactive controls.

Quick Start: plot_3d_isosurface
--------------------------------

The ``plot_3d_isosurface`` function is ideal for quickly visualising multiple quantities on an isosurface.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import cdiutils
   from cdiutils.interactive import plot_3d_isosurface
   
   # load reconstruction data
   data = cdiutils.io.load_cxi(
       "reconstruction.cxi",
       "amplitude",
       "phase",
       "het_strain",
       "support",
   )
   voxel_size = cdiutils.io.load_cxi("reconstruction.cxi", "voxel_size")
   
   # create interactive plot
   plot_3d_isosurface(
       data["amplitude"],
       data,
       voxel_size=voxel_size,
       cmaps={"amplitude": "turbo", "het_strain": "RdBu"},
   )

Interactive Controls
~~~~~~~~~~~~~~~~~~~~

The function provides several interactive widgets:

* **Threshold slider**: Control the isosurface level
* **Quantity dropdown**: Switch between different data arrays (amplitude, phase, strain, etc.)
* **Colormap dropdown**: Choose from 20+ colormaps including perceptually uniform options
* **Auto-scale checkbox**: Automatically scale colorbar to data range
* **Symmetric checkbox**: Force colorbar to be symmetric around zero (useful for strain)
* **Set limits checkbox**: Enable manual vmin/vmax input fields
* **Replace NaN checkbox**: Replace NaN values with mean to avoid visualisation artefacts

Colormap Selection
~~~~~~~~~~~~~~~~~~

Choose appropriate colormaps for your data type:

* **Sequential** (amplitude, intensity): ``turbo``, ``viridis``, ``inferno``, ``magma``
* **Diverging** (strain, phase): ``RdBu``, ``coolwarm``, ``twilight``
* **Perceptually uniform**: ``cet_CET_D13``, ``cet_CET_C9s_r``, ``cet_CET_D1A``
* **Constant/max chroma**: ``jch_const``, ``jch_max``

.. code-block:: python

   # example: different colormaps for different quantities
   plot_3d_isosurface(
       data["amplitude"],
       data,
       voxel_size=voxel_size,
       cmaps={
           "amplitude": "turbo",
           "phase": "twilight",
           "het_strain": "RdBu",
           "support": "Greys",
       },
   )

Advanced Usage: ThreeDViewer
-----------------------------

The ``ThreeDViewer`` class provides a widget for exploring complex 3D arrays with phase and amplitude information.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from cdiutils.interactive import ThreeDViewer
   import numpy as np
   
   # create complex 3D array
   complex_3d = data["amplitude"] * np.exp(1j * data["phase"])
   
   # create viewer
   viewer = ThreeDViewer(
       complex_3d,
       voxel_size=voxel_size,
       figsize=(9, 6),
   )
   viewer

Interactive Controls
~~~~~~~~~~~~~~~~~~~~

The viewer provides extensive controls:

* **Threshold slider**: Control isosurface level based on amplitude
* **Phase/Amplitude toggle**: Switch between phase and amplitude display
* **Colormap dropdown**: Choose from 20+ colormaps
* **Colorbar controls**: Auto-scale, symmetric mode, or manual limits
* **Replace NaN checkbox**: Handle NaN values gracefully
* **Rotation toggle**: Enable continuous rotation animation
* **Theme toggle**: Switch between light and dark themes

Phase Visualisation
~~~~~~~~~~~~~~~~~~~~

When displaying phase:

1. Toggle to "Phase" mode using the Phase/Amplitude button
2. Enable **symmetric colorbar** for proper phase visualisation
3. Use a diverging colormap like ``twilight`` or ``coolwarm``
4. Phase values are displayed in radians (-π to π)

.. code-block:: python

   # the viewer automatically handles phase extraction
   # just toggle between Phase/Amplitude mode in the widget

Amplitude Visualisation
~~~~~~~~~~~~~~~~~~~~~~~~

When displaying amplitude:

1. Toggle to "Amplitude" mode
2. Use sequential colormaps like ``turbo`` or ``viridis``
3. Auto-scale typically works well for amplitude data

Handling NaN Values
-------------------

NaN values in your data can cause visualisation artefacts (weird colours, missing regions). Both tools provide a **"Replace NaN with mean"** checkbox:

.. code-block:: python

   # if you see unusual colours or artefacts:
   # 1. Check the "Replace NaN with mean" checkbox
   # 2. The visualisation will replace NaN values with the mean value
   # 3. This provides cleaner visualisations while preserving data structure

This is particularly useful for:

* Strain fields computed near boundaries
* Phase unwrapping artefacts
* Support mask edges

Tips and Best Practices
-----------------------

Colorbar Settings
~~~~~~~~~~~~~~~~~

* **For strain data**: Enable symmetric colorbar and use diverging colormaps (``RdBu``, ``coolwarm``)
* **For phase data**: Use symmetric mode with ``twilight`` or phase-specific colormaps
* **For amplitude**: Auto-scale works well with sequential colormaps

Performance
~~~~~~~~~~~

* Start with a moderate threshold to reduce the number of vertices
* Use the threshold slider to explore different isosurface levels
* For very large datasets, consider downsampling before visualisation

Theme Selection
~~~~~~~~~~~~~~~

* **Light theme**: Better for presentations and well-lit environments
* **Dark theme**: Easier on the eyes in low-light conditions, better for focusing on the data

Camera Controls
~~~~~~~~~~~~~~~

* **Left mouse button**: Rotate view
* **Middle mouse button** (or Shift + left): Pan view  
* **Right mouse button** (or Ctrl + left): Zoom
* **Scroll wheel**: Zoom in/out
* Zoom sensitivity has been optimised for smooth interaction

Example Workflow
----------------

Here's a complete workflow for analysing a BCDI reconstruction:

.. code-block:: python

   import cdiutils
   from cdiutils.interactive import plot_3d_isosurface, ThreeDViewer
   import numpy as np
   
   # load data
   data_path = "path/to/reconstruction.cxi"
   data = cdiutils.io.load_cxi(
       data_path,
       "amplitude",
       "phase",
       "het_strain",
       "support",
   )
   voxel_size = cdiutils.io.load_cxi(data_path, "voxel_size")
   
   # quick exploration of multiple quantities
   plot_3d_isosurface(
       data["amplitude"],
       data,
       voxel_size=voxel_size,
       cmaps={"het_strain": "RdBu", "amplitude": "turbo"},
   )
   
   # detailed phase/amplitude exploration
   complex_3d = data["amplitude"] * np.exp(1j * data["phase"])
   viewer = ThreeDViewer(complex_3d, voxel_size=voxel_size)
   viewer

Troubleshooting
---------------

Import Errors
~~~~~~~~~~~~~

If you encounter import errors:

.. code-block:: python

   ImportError: ThreeDViewer requires the following packages: plotly, scikit-image, scipy

Install the interactive dependencies:

.. code-block:: bash

   pip install cdiutils[interactive]

Visualisation Artefacts
~~~~~~~~~~~~~~~~~~~~~~~

If you see strange colours or patterns:

1. Check for NaN values in your data: ``np.any(np.isnan(data))``
2. Enable the "Replace NaN with mean" checkbox
3. Adjust the colorbar limits manually using "Set limits"

Performance Issues
~~~~~~~~~~~~~~~~~~

For large datasets:

1. Increase the threshold to reduce vertex count
2. Consider downsampling: ``data[::2, ::2, ::2]``
3. Close other widgets/plots to free memory

Additional Resources
--------------------

* Example notebook: :download:`interactive_features.ipynb <../../examples/interactive_features.ipynb>`
* API reference: :doc:`../api`
* Plotly documentation: https://plotly.com/python/

