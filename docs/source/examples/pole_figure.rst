Pole Figure Visualization Example
=================================

This example demonstrates how to create crystallographic pole figures using stereographic projection. Pole figures are essential tools for visualizing 3D diffraction intensity distributions and understanding crystallographic orientation relationships.

.. note::
   **Download the notebook**: :download:`pole_figure.ipynb <../../../examples/pole_figure.ipynb>`

Overview
--------

Pole figures provide powerful visualization for:

* **Crystallographic Orientation Analysis**: Understanding preferred orientations and texture
* **3D Data Visualization**: Mapping 3D diffraction intensity onto 2D projections
* **Stereographic Projection**: Standard crystallographic projection methods
* **Bragg Peak Analysis**: Visualizing diffraction peak distributions in reciprocal space
* **Texture Analysis**: Quantifying crystallographic texture and preferred orientations

This example covers the complete workflow from loading 3D reciprocal space data to creating publication-quality pole figure visualizations.

Learning Objectives
-------------------

By working through this example, you will learn to:

1. Load and prepare 3D reciprocal space diffraction data
2. Understand pole figure geometry and stereographic projection principles
3. Create pole figures using CDIutils analysis tools
4. Customize pole figure appearance for different analysis needs
5. Interpret pole figure patterns for crystallographic analysis
6. Export pole figures in publication-ready formats

Key Concepts
------------

**Stereographic Projection**
   A method for mapping points on a sphere onto a plane, preserving angular relationships essential for crystallographic analysis.

**Reciprocal Space Coordinates**
   3D momentum transfer coordinates (qx, qy, qz) that describe the diffraction geometry and crystal orientation.

**Pole Figure Interpretation**
   Understanding how intensity distributions in pole figures relate to crystallographic orientations and preferred directions.

Getting Started
---------------

Begin with loading the preprocessed reciprocal space data:

.. code-block:: python

   from matplotlib.colors import LogNorm
   import numpy as np
   
   import cdiutils
   
   # Set plotting parameters for publication quality
   cdiutils.plot.update_plot_params()
   
   # Path to preprocessed data containing orthogonalized Bragg peak
   path = "path/to/your/S001_preprocessed_data.cxi"

The preprocessed data file contains:
- Orthogonalized diffraction intensity data
- Corresponding q-space coordinate grids
- Geometric transformation parameters

Data Loading and Preparation
-----------------------------

Load the 3D diffraction data and coordinate information:

.. code-block:: python

   # Load reciprocal space data and coordinates
   with cdiutils.CXIFile(path, "r") as cxi:
       # 3D diffraction intensity data
       data = cxi["entry_1/data_2/data"]
       
       # Reciprocal space coordinate grids
       qx = cxi["entry_1/result_2/qx_xu"]
       qy = cxi["entry_1/result_2/qy_xu"] 
       qz = cxi["entry_1/result_2/qz_xu"]
       
       # Q-space shift for proper centering
       shift = cxi["entry_1/result_2/q_space_shift"]
   
   print(f"Data shape: {data.shape}")
   print(f"Q-coordinates shape: {qx.shape}, {qy.shape}, {qz.shape}")
   
   # Calculate voxel size for proper scaling
   voxel_size = (
       np.diff(qx).mean(),
       np.diff(qy).mean(), 
       np.diff(qz).mean()
   )
   
   print(f"Voxel size: {voxel_size}")

Initial Visualization
---------------------

Visualize the 3D data with orthogonal slices before creating pole figures:

.. code-block:: python

   # Create overview plot of 3D diffraction data
   fig, axes = cdiutils.plot.plot_volume_slices(
       data, 
       voxel_size=voxel_size, 
       data_centre=shift,
       norm=LogNorm(),  # Logarithmic scaling for intensity
       convention="xu",  # Use XU convention for q-space
       show=False
   )
   
   # Add proper axis labels
   cdiutils.plot.add_labels(axes, convention="xu")
   
   # Display the figure
   fig.show()

This overview helps verify data quality and identify the Bragg peak position before pole figure analysis.

Creating Basic Pole Figures
----------------------------

Generate a standard pole figure using stereographic projection:

.. code-block:: python

   # Create pole figure with default parameters
   pole_figure_data, projection_info = cdiutils.analysis.pole_figure(
       data,
       qx=qx,
       qy=qy, 
       qz=qz,
       projection_type="stereographic",
       hemisphere="upper"  # Project to upper hemisphere
   )
   
   # Visualize the pole figure
   fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
   
   # Plot pole figure with proper scaling
   im = ax.imshow(
       pole_figure_data,
       extent=projection_info['extent'],
       origin='lower',
       norm=LogNorm(),
       cmap='viridis'
   )
   
   # Configure polar plot
   ax.set_title("Pole Figure - Upper Hemisphere")
   ax.grid(True)
   
   # Add colorbar
   plt.colorbar(im, ax=ax, label="Intensity")
   plt.show()

Advanced Pole Figure Customization
-----------------------------------

Create customized pole figures for specific analysis needs:

.. code-block:: python

   # Advanced pole figure with custom parameters
   pole_figure_advanced = cdiutils.analysis.pole_figure(
       data,
       qx=qx, qy=qy, qz=qz,
       projection_type="stereographic",
       hemisphere="both",  # Include both hemispheres
       resolution=500,     # High resolution for publication
       intensity_threshold=0.01,  # Filter weak intensities
       interpolation_method="cubic"
   )
   
   # Create dual-hemisphere visualization
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8),
                                  subplot_kw={'projection': 'polar'})
   
   # Upper hemisphere
   im1 = ax1.imshow(
       pole_figure_advanced['upper'],
       extent=pole_figure_advanced['extent'],
       norm=LogNorm(vmin=1e-3, vmax=1),
       cmap='hot'
   )
   ax1.set_title("Upper Hemisphere")
   
   # Lower hemisphere  
   im2 = ax2.imshow(
       pole_figure_advanced['lower'],
       extent=pole_figure_advanced['extent'],
       norm=LogNorm(vmin=1e-3, vmax=1),
       cmap='hot'
   )
   ax2.set_title("Lower Hemisphere")
   
   # Add coordinated colorbars
   plt.colorbar(im1, ax=ax1, label="Intensity")
   plt.colorbar(im2, ax=ax2, label="Intensity")
   
   plt.tight_layout()
   plt.show()

Multiple Pole Figure Analysis
-----------------------------

Compare pole figures from different experimental conditions:

.. code-block:: python

   # Load multiple datasets for comparison
   data_sets = {
       'Condition_A': load_dataset("S001_preprocessed_data.cxi"),
       'Condition_B': load_dataset("S002_preprocessed_data.cxi"),
       'Condition_C': load_dataset("S003_preprocessed_data.cxi")
   }
   
   # Generate pole figures for comparison
   pole_figures = {}
   
   for condition, (data, qx, qy, qz, shift) in data_sets.items():
       pole_fig, info = cdiutils.analysis.pole_figure(
           data, qx=qx, qy=qy, qz=qz,
           projection_type="stereographic",
           hemisphere="upper"
       )
       pole_figures[condition] = pole_fig
   
   # Create comparison plot
   fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                           subplot_kw={'projection': 'polar'})
   
   for i, (condition, pole_fig) in enumerate(pole_figures.items()):
       im = axes[i].imshow(
           pole_fig,
           extent=info['extent'],
           norm=LogNorm(),
           cmap='plasma'
       )
       axes[i].set_title(f"Pole Figure - {condition}")
       axes[i].grid(True)
       
       plt.colorbar(im, ax=axes[i], label="Intensity")
   
   plt.tight_layout()
   plt.show()

Quantitative Pole Figure Analysis
----------------------------------

Perform quantitative analysis of pole figure patterns:

.. code-block:: python

   # Quantitative analysis of pole figure characteristics
   def analyze_pole_figure(pole_figure_data, projection_info):
       """Quantitative analysis of pole figure features."""
       
       analysis_results = {}
       
       # Calculate intensity statistics
       analysis_results['max_intensity'] = np.max(pole_figure_data)
       analysis_results['mean_intensity'] = np.mean(pole_figure_data)
       analysis_results['intensity_std'] = np.std(pole_figure_data)
       
       # Find intensity peaks
       from scipy.ndimage import maximum_filter
       
       # Local maxima detection
       local_maxima = maximum_filter(pole_figure_data, size=20) == pole_figure_data
       peak_intensities = pole_figure_data[local_maxima]
       
       # Select significant peaks (above threshold)
       threshold = 0.1 * analysis_results['max_intensity']
       significant_peaks = peak_intensities[peak_intensities > threshold]
       
       analysis_results['num_peaks'] = len(significant_peaks)
       analysis_results['peak_intensities'] = significant_peaks
       
       # Calculate texture strength
       # Measure deviation from random orientation
       uniform_intensity = np.mean(pole_figure_data)
       texture_index = np.sqrt(np.mean((pole_figure_data - uniform_intensity)**2))
       analysis_results['texture_index'] = texture_index
       
       return analysis_results
   
   # Analyze pole figure
   analysis = analyze_pole_figure(pole_figure_data, projection_info)
   
   print("Pole Figure Analysis Results:")
   print(f"Maximum intensity: {analysis['max_intensity']:.3f}")
   print(f"Number of significant peaks: {analysis['num_peaks']}")
   print(f"Texture index: {analysis['texture_index']:.3f}")

Crystallographic Interpretation
-------------------------------

Interpret pole figure patterns in crystallographic context:

.. code-block:: python

   # Crystallographic interpretation tools
   def interpret_pole_figure(pole_figure_data, crystal_system, lattice_parameters):
       """Interpret pole figure in crystallographic context."""
       
       interpretation = {}
       
       # Expected peak positions for given crystal system
       if crystal_system == "cubic":
           # For cubic crystals, calculate expected pole positions
           expected_poles = calculate_cubic_poles(lattice_parameters)
       elif crystal_system == "hexagonal":
           expected_poles = calculate_hexagonal_poles(lattice_parameters)
       
       # Compare observed with expected
       observed_peaks = find_pole_figure_peaks(pole_figure_data)
       
       interpretation['expected_poles'] = expected_poles
       interpretation['observed_peaks'] = observed_peaks
       interpretation['deviation'] = calculate_peak_deviation(
           observed_peaks, expected_poles
       )
       
       return interpretation
   
   # Apply crystallographic interpretation
   crystal_info = {
       'system': 'cubic',
       'lattice_parameter': 3.615e-10  # meters, for example Ni
   }
   
   interpretation = interpret_pole_figure(
       pole_figure_data, 
       crystal_info['system'],
       crystal_info['lattice_parameter']
   )

Advanced Visualization Techniques
---------------------------------

Create specialized visualizations for specific research needs:

.. code-block:: python

   # Contour plot visualization
   def create_contour_pole_figure(pole_figure_data, projection_info):
       """Create contour-based pole figure visualization."""
       
       fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
       
       # Create coordinate meshes for contour plotting
       r = np.linspace(0, projection_info['max_radius'], pole_figure_data.shape[0])
       theta = np.linspace(0, 2*np.pi, pole_figure_data.shape[1])
       R, Theta = np.meshgrid(r, theta)
       
       # Contour levels
       levels = np.logspace(-3, 0, 20)
       
       # Create contour plot
       cs = ax.contour(Theta, R, pole_figure_data.T, levels=levels, cmap='viridis')
       ax.contourf(Theta, R, pole_figure_data.T, levels=levels, cmap='viridis', alpha=0.7)
       
       # Add contour labels
       ax.clabel(cs, inline=True, fontsize=8)
       
       # Formatting
       ax.set_title("Pole Figure - Contour Plot")
       ax.grid(True)
       
       return fig, ax
   
   # 3D visualization
   def create_3d_pole_figure(pole_figure_data, projection_info):
       """Create 3D surface plot of pole figure."""
       
       from mpl_toolkits.mplot3d import Axes3D
       
       fig = plt.figure(figsize=(12, 10))
       ax = fig.add_subplot(111, projection='3d')
       
       # Convert polar to cartesian coordinates
       r = np.linspace(0, projection_info['max_radius'], pole_figure_data.shape[0])
       theta = np.linspace(0, 2*np.pi, pole_figure_data.shape[1])
       R, Theta = np.meshgrid(r, theta)
       
       X = R * np.cos(Theta)
       Y = R * np.sin(Theta)
       Z = pole_figure_data.T
       
       # Surface plot
       surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
       
       # Formatting
       ax.set_xlabel('X')
       ax.set_ylabel('Y') 
       ax.set_zlabel('Intensity')
       ax.set_title('3D Pole Figure Surface')
       
       plt.colorbar(surf, ax=ax, label='Intensity')
       
       return fig, ax

Export and Documentation
------------------------

Export pole figures in various formats for publication:

.. code-block:: python

   # Export pole figure data and visualizations
   def export_pole_figure_results(pole_figure_data, projection_info, 
                                  output_dir, base_name):
       """Export pole figure results in multiple formats."""
       
       import os
       os.makedirs(output_dir, exist_ok=True)
       
       # Export raw data
       np.savez(
           os.path.join(output_dir, f"{base_name}_pole_figure_data.npz"),
           pole_figure=pole_figure_data,
           projection_info=projection_info
       )
       
       # Export visualization
       fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
       im = ax.imshow(pole_figure_data, norm=LogNorm(), cmap='viridis')
       ax.set_title(f"Pole Figure - {base_name}")
       plt.colorbar(im, ax=ax, label="Intensity")
       
       # Save in multiple formats
       fig.savefig(os.path.join(output_dir, f"{base_name}_pole_figure.png"), 
                  dpi=300, bbox_inches='tight')
       fig.savefig(os.path.join(output_dir, f"{base_name}_pole_figure.pdf"), 
                  bbox_inches='tight')
       
       plt.close(fig)
       
       # Export analysis summary
       analysis = analyze_pole_figure(pole_figure_data, projection_info)
       
       with open(os.path.join(output_dir, f"{base_name}_analysis.txt"), 'w') as f:
           f.write(f"Pole Figure Analysis Summary - {base_name}\n")
           f.write("=" * 50 + "\n")
           for key, value in analysis.items():
               f.write(f"{key}: {value}\n")

Integration with Processing Pipelines
-------------------------------------

Integrate pole figure analysis into automated workflows:

.. code-block:: python

   # Automated pole figure generation for multiple scans
   def batch_pole_figure_analysis(scan_list, base_path, output_dir):
       """Generate pole figures for multiple scans automatically."""
       
       results = {}
       
       for scan in scan_list:
           # Load scan data
           scan_path = os.path.join(base_path, f"S{scan:03d}_preprocessed_data.cxi")
           
           try:
               with cdiutils.CXIFile(scan_path, "r") as cxi:
                   data = cxi["entry_1/data_2/data"]
                   qx = cxi["entry_1/result_2/qx_xu"]
                   qy = cxi["entry_1/result_2/qy_xu"]
                   qz = cxi["entry_1/result_2/qz_xu"]
               
               # Generate pole figure
               pole_fig, info = cdiutils.analysis.pole_figure(
                   data, qx=qx, qy=qy, qz=qz
               )
               
               # Export results
               export_pole_figure_results(
                   pole_fig, info, output_dir, f"S{scan:03d}"
               )
               
               results[scan] = {'success': True, 'pole_figure': pole_fig}
               
           except Exception as e:
               results[scan] = {'success': False, 'error': str(e)}
               
       return results

Best Practices
--------------

**Data Quality**
   Ensure proper orthogonalization and centering of reciprocal space data before pole figure generation.

**Projection Choice**
   Use stereographic projection for most crystallographic applications; consider equal-area projection for texture analysis.

**Resolution**
   Balance computation time and visualization quality when setting pole figure resolution.

**Normalization**
   Apply appropriate intensity normalization for quantitative comparisons between different datasets.

**Interpretation**
   Always consider the specific crystal system and expected symmetries when interpreting pole figure patterns.

Next Steps
----------

After mastering pole figure visualization:

* Apply these techniques to your own BCDI reconstruction data
* Combine with strain analysis from :doc:`bcdi_reconstruction_analysis`
* Integrate into automated workflows using :doc:`../tutorials/pipeline_tutorial`
* Explore advanced crystallographic analysis using pole figure quantitative data

Related Examples
----------------

* :doc:`bcdi_reconstruction_analysis` - Comprehensive reconstruction analysis workflows
* :doc:`explore_cxi_file` - Understanding data structure for pole figure inputs
* :doc:`../tutorials/step_by_step_tutorial` - Manual processing leading to pole figure analysis
