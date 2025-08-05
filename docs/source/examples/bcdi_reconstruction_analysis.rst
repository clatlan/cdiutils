BCDI Reconstruction Analysis Example
=====================================

This example demonstrates comprehensive analysis and comparison of BCDI reconstruction results stored in CXI files. You'll learn advanced techniques for data exploration, visualization, and quantitative analysis of strain and amplitude distributions.

.. note::
   **Download the notebook**: :download:`bcdi_reconstruction_analysis.ipynb <../../../examples/bcdi_reconstruction_analysis.ipynb>`

Overview
--------

This example covers:

* **CXI File Exploration**: Interactive exploration of reconstruction file structure
* **Multi-dataset Loading**: Loading and comparing multiple reconstruction results  
* **Advanced Visualization**: Creating publication-quality plots and comparisons
* **Quantitative Analysis**: Statistical analysis of strain distributions and amplitude variations
* **Custom Plot Generation**: Tailoring visualizations for specific research needs

The notebook provides a complete workflow from raw CXI files to publication-ready analysis figures.

Learning Objectives
-------------------

By working through this example, you will learn to:

1. Navigate complex CXI file structures efficiently
2. Load and organize multiple reconstruction datasets
3. Create comparative visualizations across different conditions
4. Perform statistical analysis of 3D strain fields
5. Generate customized plots for specific analysis needs
6. Export results in various formats for further analysis

Key Concepts
------------

**CXI File Structure**
   CXI (Coherent X-ray Imaging) files provide a standardized format for storing coherent diffraction imaging data and results. Understanding the hierarchical structure is essential for effective data analysis.

**Reconstruction Comparison**
   Comparing reconstructions from different experimental conditions or processing parameters helps validate results and understand systematic effects.

**Strain Field Analysis**
   3D strain fields contain rich information about material properties and deformation mechanisms that require specialized visualization and analysis techniques.

Getting Started
---------------

The example begins with basic setup and data path configuration:

.. code-block:: python

   # Import required packages
   from IPython.display import display
   import matplotlib.pyplot as plt
   from matplotlib.colors import LogNorm
   import numpy as np
   
   import cdiutils
   
   # Set default plotting parameters for publication quality
   cdiutils.plot.update_plot_params()
   
   # Specify your data directory
   results_dir = "path/to/your/results/directory"

CXI File Exploration
--------------------

The notebook demonstrates interactive exploration of CXI file structure:

.. code-block:: python

   # Path to CXI file for exploration
   cxi_path = results_dir + "Sample_Name/S000/S000_postprocessed_data.cxi"
   
   # Initialize explorer for interactive investigation
   explorer = cdiutils.io.CXIExplorer(cxi_path)
   
   # Explore file structure interactively
   explorer.explore()
   
   # Get overview of available datasets
   datasets = explorer.list_datasets()
   print("Available datasets:")
   for dataset in datasets:
       print(f"  {dataset}")

This exploration reveals the organization of:
- Raw experimental data
- Reconstruction results
- Post-processed strain fields
- Metadata and processing parameters

Multi-Dataset Loading and Organization
--------------------------------------

The example shows how to systematically load multiple reconstruction results:

.. code-block:: python

   # Define multiple samples and scans for comparison
   samples_scans = {
       "Sample_A": ["S001", "S002", "S003"],
       "Sample_B": ["S101", "S102", "S103"],
   }
   
   # Load reconstruction data systematically
   reconstruction_data = {}
   
   for sample, scans in samples_scans.items():
       reconstruction_data[sample] = {}
       
       for scan in scans:
           cxi_file = f"{results_dir}/{sample}/{scan}/{scan}_postprocessed_data.cxi"
           
           # Load amplitude and phase data
           with h5py.File(cxi_file, 'r') as f:
               amplitude = f['/entry_1/image_1/amplitude'][:]
               phase = f['/entry_1/image_1/phase'][:]
               
           reconstruction_data[sample][scan] = {
               'amplitude': amplitude,
               'phase': phase,
               'file_path': cxi_file
           }

Advanced Visualization Techniques
---------------------------------

The notebook demonstrates sophisticated plotting for multi-dimensional data:

**3D Slice Visualization**

.. code-block:: python

   # Create comprehensive slice plots
   fig, axes = plt.subplots(2, 3, figsize=(15, 10))
   
   # Plot amplitude slices
   for i, (sample, data) in enumerate(reconstruction_data.items()):
       amplitude = data['S001']['amplitude']
       
       # X-Y slice at center
       axes[i, 0].imshow(amplitude[amplitude.shape[0]//2, :, :])
       axes[i, 0].set_title(f'{sample} - XY Slice')
       
       # X-Z slice at center  
       axes[i, 1].imshow(amplitude[:, amplitude.shape[1]//2, :])
       axes[i, 1].set_title(f'{sample} - XZ Slice')
       
       # Y-Z slice at center
       axes[i, 2].imshow(amplitude[:, :, amplitude.shape[2]//2])
       axes[i, 2].set_title(f'{sample} - YZ Slice')
   
   plt.tight_layout()
   plt.show()

**Strain Distribution Analysis**

.. code-block:: python

   # Analyze strain distributions across samples
   strain_stats = {}
   
   for sample, scans_data in reconstruction_data.items():
       strain_stats[sample] = {}
       
       for scan, data in scans_data.items():
           # Calculate strain from phase gradients
           phase = data['phase']
           strain = calculate_strain_from_phase(phase)
           
           # Compute statistics
           strain_stats[sample][scan] = {
               'mean': np.mean(strain),
               'std': np.std(strain),
               'max': np.max(strain),
               'min': np.min(strain)
           }
   
   # Visualize strain statistics
   plot_strain_statistics(strain_stats)

Comparative Analysis
--------------------

The example includes methods for quantitative comparison:

.. code-block:: python

   # Compare reconstructions across different conditions
   def compare_reconstructions(data1, data2, metric='correlation'):
       """Compare two reconstructions using specified metric."""
       
       if metric == 'correlation':
           # Calculate cross-correlation
           correlation = correlate_3d(data1, data2)
           return correlation
           
       elif metric == 'structural_similarity':
           # Calculate structural similarity index
           ssim = calculate_ssim_3d(data1, data2)
           return ssim
           
       elif metric == 'rmse':
           # Root mean square error
           rmse = np.sqrt(np.mean((data1 - data2)**2))
           return rmse
   
   # Perform systematic comparisons
   comparison_matrix = create_comparison_matrix(
       reconstruction_data,
       metrics=['correlation', 'ssim', 'rmse']
   )

Custom Plot Generation
----------------------

The notebook shows how to create specialized visualizations:

.. code-block:: python

   # Custom plotting function for research-specific needs
   def create_custom_analysis_plot(amplitude, phase, strain, title=""):
       """Create multi-panel analysis plot."""
       
       fig = plt.figure(figsize=(20, 12))
       gs = gridspec.GridSpec(3, 4, figure=fig)
       
       # Amplitude analysis
       ax1 = fig.add_subplot(gs[0, :2])
       plot_amplitude_analysis(ax1, amplitude)
       
       # Phase analysis  
       ax2 = fig.add_subplot(gs[1, :2])
       plot_phase_analysis(ax2, phase)
       
       # Strain analysis
       ax3 = fig.add_subplot(gs[2, :2]) 
       plot_strain_analysis(ax3, strain)
       
       # 3D renderings
       ax4 = fig.add_subplot(gs[:, 2:], projection='3d')
       plot_3d_reconstruction(ax4, amplitude, phase)
       
       plt.suptitle(title, fontsize=16)
       plt.tight_layout()
       
       return fig

Statistical Analysis Tools
--------------------------

Advanced statistical analysis of reconstruction results:

.. code-block:: python

   # Statistical analysis of strain distributions
   def analyze_strain_statistics(strain_field, support_mask=None):
       """Comprehensive statistical analysis of strain fields."""
       
       if support_mask is not None:
           strain_masked = strain_field[support_mask]
       else:
           strain_masked = strain_field.flatten()
       
       stats = {
           'mean': np.mean(strain_masked),
           'median': np.median(strain_masked),
           'std': np.std(strain_masked),
           'skewness': scipy.stats.skew(strain_masked),
           'kurtosis': scipy.stats.kurtosis(strain_masked),
           'percentiles': np.percentile(strain_masked, [5, 25, 75, 95])
       }
       
       return stats
   
   # Generate statistical summary across all samples
   statistical_summary = generate_statistical_summary(reconstruction_data)

Quality Assessment
------------------

Methods for assessing reconstruction quality:

.. code-block:: python

   # Quality metrics for reconstruction assessment
   def calculate_quality_metrics(amplitude, phase, original_data=None):
       """Calculate various quality metrics for reconstructions."""
       
       metrics = {}
       
       # Support fraction
       support = amplitude > 0.1 * amplitude.max()
       metrics['support_fraction'] = support.sum() / support.size
       
       # Phase consistency
       phase_wrapped = np.angle(np.exp(1j * phase))
       metrics['phase_consistency'] = calculate_phase_consistency(phase_wrapped)
       
       # Resolution estimate
       metrics['resolution'] = estimate_resolution(amplitude)
       
       # If original data available, calculate fidelity
       if original_data is not None:
           metrics['data_fidelity'] = calculate_data_fidelity(
               amplitude, phase, original_data
           )
       
       return metrics

Data Export and Further Analysis
--------------------------------

The notebook concludes with data export options:

.. code-block:: python

   # Export processed results for external analysis
   def export_analysis_results(reconstruction_data, output_dir):
       """Export analysis results in multiple formats."""
       
       for sample, scans_data in reconstruction_data.items():
           sample_dir = os.path.join(output_dir, sample)
           os.makedirs(sample_dir, exist_ok=True)
           
           for scan, data in scans_data.items():
               # Export to NumPy format
               np.savez(
                   os.path.join(sample_dir, f'{scan}_analysis.npz'),
                   amplitude=data['amplitude'],
                   phase=data['phase'],
                   strain=data.get('strain', None)
               )
               
               # Export to VTK for 3D visualization
               export_to_vtk(
                   os.path.join(sample_dir, f'{scan}_3d.vti'),
                   data['amplitude'], 
                   data['phase']
               )

Tips and Best Practices
-----------------------

**Memory Management**
   For large datasets, use memory mapping and process data in chunks to avoid memory issues.

**Visualization Optimization**
   Use appropriate colormaps and scaling for different data types (logarithmic for amplitude, linear for phase).

**Statistical Significance**
   When comparing conditions, perform appropriate statistical tests to assess significance of observed differences.

**Documentation**
   Keep detailed records of processing parameters and analysis methods for reproducibility.

Next Steps
----------

After mastering this analysis workflow:

* Explore the :doc:`explore_cxi_file` example for detailed CXI file investigation
* Learn 3D visualization techniques in :doc:`pole_figure`
* Apply these methods to your own reconstruction datasets
* Develop custom analysis pipelines for specific research questions

Related Examples
----------------

* :doc:`explore_cxi_file` - Detailed CXI file structure exploration
* :doc:`pole_figure` - Advanced 3D crystallographic visualization  
* :doc:`../tutorials/pipeline_tutorial` - Automated processing workflows
