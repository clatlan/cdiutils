CXI File Explorer Example
=========================

This example demonstrates how to use the CXI file explorer to investigate and navigate complex CXI (Coherent X-ray Imaging) file structures. Learn to efficiently find, visualize, and extract data from standardized diffraction imaging files.

.. note::
   **Download the notebook**: :download:`explore_cxi_file.ipynb <../../../examples/explore_cxi_file.ipynb>`

Overview
--------

The CXI file explorer provides powerful tools for:

* **File Structure Analysis**: Understanding the hierarchical organization of CXI files
* **Interactive Navigation**: Widget-based browsing through complex data structures
* **Smart Search**: Finding specific datasets using pattern matching
* **Data Visualization**: Automatic plotting and inspection of array data
* **Metadata Extraction**: Accessing attributes and experimental parameters

CXI files can contain dozens of datasets across multiple hierarchy levels. The explorer simplifies navigation and data discovery.

Learning Objectives
-------------------

By working through this example, you will:

1. Understand CXI file structure and organization principles
2. Master the four core explorer methods for file investigation
3. Use interactive widgets for efficient data browsing
4. Implement search strategies for finding specific datasets
5. Visualize data directly from the file structure
6. Extract metadata and experimental parameters

Key Features
------------

**Four Core Methods**
   The ``CXIExplorer`` provides four essential methods for file investigation:

   * ``summarise()`` - File overview with size and dataset counts
   * ``tree()`` - Hierarchical structure display with depth control
   * ``search()`` - Pattern-based dataset and group finding
   * ``show()`` - Direct visualization of datasets with automatic plotting

**Interactive Browser**
   The ``explore()`` method launches an interactive widget interface for point-and-click navigation through complex file structures.

**Smart Data Handling**
   Automatic detection of data types with appropriate visualization (1D plots, 2D images, 3D slices) and metadata display.

Getting Started
---------------

Begin with basic setup and file path configuration:

.. code-block:: python

   import cdiutils
   
   # Set plotting parameters for better visualization
   cdiutils.plot.update_plot_params()
   
   # Specify path to your CXI file
   path = "path/to/your/data.cxi"
   
   # Create explorer instance
   explorer = cdiutils.io.CXIExplorer(path)

File Overview and Summary
-------------------------

Start exploration with a high-level file summary:

.. code-block:: python

   # Get comprehensive file overview
   explorer.summarise()

This provides essential information:
- File size and format version
- Total number of groups and datasets
- Top-level structure overview
- Memory usage estimates

Example output::

   CXI File Summary
   ================
   File: reconstruction_results.cxi
   Size: 245.7 MB
   
   Structure Overview:
   - Groups: 15
   - Datasets: 42
   - Attributes: 128
   
   Top-level entries:
   - /entry_1/ (group)
   - /reconstruction_1/ (group)
   - /processing_1/ (group)

Hierarchical Structure Navigation
---------------------------------

Explore the file's tree structure with controlled depth:

.. code-block:: python

   # Display full tree structure
   explorer.tree()
   
   # Limit depth for large files
   explorer.tree(max_depth=2)
   
   # Include attributes in tree view
   explorer.tree(max_depth=3, show_attributes=True)

The tree view shows:
- Group hierarchies with proper indentation
- Dataset names with type and shape information
- Attributes when requested
- Path structure for direct access

Example tree structure::

   /
   ├── entry_1/
   │   ├── instrument/
   │   │   ├── detector/
   │   │   │   ├── distance (dataset: float64)
   │   │   │   └── pixel_size (dataset: float64)
   │   ├── sample/
   │   │   └── name (dataset: string)
   │   └── data_1/
   │       ├── intensity (dataset: (256, 256, 256) float32)
   │       └── phase (dataset: (256, 256, 256) float32)

Smart Search Functionality
---------------------------

Find datasets using flexible pattern matching:

.. code-block:: python

   # Search for strain-related datasets
   strain_datasets = explorer.search("strain")
   
   # Find phase information
   phase_data = explorer.search("phase")
   
   # Look for detector parameters
   detector_info = explorer.search("detector")
   
   # Search for specific scan numbers
   scan_data = explorer.search("S001")
   
   # Use wildcards for flexible matching
   amplitude_data = explorer.search("*amplitude*")

Search results include:
- Full paths to matching items
- Dataset shapes and types
- Group structures
- Attribute matches

Direct Data Visualization
-------------------------

Visualize datasets directly from the explorer:

.. code-block:: python

   # Show 2D intensity data
   explorer.show("/entry_1/data_1/intensity")
   
   # Visualize 3D amplitude with automatic slicing
   explorer.show("/reconstruction_1/amplitude")
   
   # Display 1D profiles
   explorer.show("/processing_1/radial_profile")
   
   # Show metadata and attributes
   explorer.show("/entry_1/instrument/detector/")

The ``show()`` method automatically:
- Detects data dimensions and types
- Chooses appropriate visualization (line plots, images, volume slices)
- Displays metadata and attributes
- Handles large datasets efficiently

Interactive Widget Browser
---------------------------

Launch the interactive browser for point-and-click navigation:

.. code-block:: python

   # Start interactive exploration
   explorer.explore()

The interactive browser provides:
- Collapsible tree structure
- Click-to-expand groups
- Automatic data preview
- Integrated plotting
- Copy-paste path functionality

Advanced Usage Patterns
------------------------

**Systematic Data Extraction**

.. code-block:: python

   # Extract all reconstruction results
   reconstruction_paths = explorer.search("reconstruction*")
   
   # Load multiple datasets systematically
   reconstruction_data = {}
   for path in reconstruction_paths:
       if "amplitude" in path:
           reconstruction_data['amplitude'] = explorer.get_dataset(path)
       elif "phase" in path:
           reconstruction_data['phase'] = explorer.get_dataset(path)

**Metadata Collection**

.. code-block:: python

   # Collect experimental parameters
   def collect_experimental_metadata(explorer):
       metadata = {}
       
       # Search for common parameter patterns
       energy_paths = explorer.search("*energy*")
       distance_paths = explorer.search("*distance*")
       detector_paths = explorer.search("*detector*")
       
       # Extract values
       for path in energy_paths:
           metadata['energy'] = explorer.get_dataset(path)
       
       return metadata

**Quality Assessment**

.. code-block:: python

   # Check data quality and completeness
   def assess_file_quality(explorer):
       report = {
           'completeness': {},
           'data_shapes': {},
           'missing_datasets': []
       }
       
       # Expected datasets for BCDI analysis
       expected_datasets = [
           'amplitude', 'phase', 'support',
           'strain', 'displacement'
       ]
       
       for dataset in expected_datasets:
           paths = explorer.search(f"*{dataset}*")
           if paths:
               report['completeness'][dataset] = True
               # Get shape information
               for path in paths:
                   try:
                       shape = explorer.get_dataset_info(path)['shape']
                       report['data_shapes'][dataset] = shape
                   except:
                       pass
           else:
               report['completeness'][dataset] = False
               report['missing_datasets'].append(dataset)
       
       return report

Comparison Across Files
-----------------------

Compare structures across multiple CXI files:

.. code-block:: python

   # Compare multiple reconstruction files
   def compare_cxi_structures(file_paths):
       structures = {}
       
       for file_path in file_paths:
           explorer = cdiutils.io.CXIExplorer(file_path)
           
           # Get all dataset paths
           all_datasets = []
           def collect_datasets(name, obj):
               if hasattr(obj, 'shape'):  # It's a dataset
                   all_datasets.append(name)
           
           structures[file_path] = {
               'datasets': all_datasets,
               'summary': explorer.summarise(return_dict=True)
           }
       
       return structures

Integration with Analysis Workflows
-----------------------------------

Use explorer results to guide automated analysis:

.. code-block:: python

   # Automated analysis pipeline based on exploration
   def create_analysis_pipeline(cxi_file_path):
       explorer = cdiutils.io.CXIExplorer(cxi_file_path)
       
       # Discover available data types
       amplitude_paths = explorer.search("*amplitude*")
       phase_paths = explorer.search("*phase*")
       strain_paths = explorer.search("*strain*")
       
       # Create processing pipeline based on available data
       pipeline_steps = []
       
       if amplitude_paths and phase_paths:
           pipeline_steps.append('reconstruction_analysis')
       
       if strain_paths:
           pipeline_steps.append('strain_analysis')
       
       # Additional steps based on metadata
       metadata = explorer.search("*metadata*")
       if metadata:
           pipeline_steps.append('metadata_extraction')
       
       return pipeline_steps

Error Handling and Troubleshooting
-----------------------------------

Handle common issues when exploring CXI files:

.. code-block:: python

   # Robust file exploration with error handling
   def safe_explore(file_path):
       try:
           explorer = cdiutils.io.CXIExplorer(file_path)
           
           # Test basic functionality
           summary = explorer.summarise()
           
           # Check for corrupted datasets
           problematic_datasets = []
           all_paths = explorer.search("*")
           
           for path in all_paths:
               try:
                   info = explorer.get_dataset_info(path)
               except Exception as e:
                   problematic_datasets.append((path, str(e)))
           
           return {
               'success': True,
               'summary': summary,
               'issues': problematic_datasets
           }
           
       except Exception as e:
           return {
               'success': False,
               'error': str(e)
           }

Best Practices
--------------

**Efficient Exploration**
   Start with ``summarise()`` and ``tree(max_depth=2)`` to get overview before detailed investigation.

**Search Strategy**
   Use specific keywords first, then broaden with wildcards if needed.

**Memory Management**
   Use ``show()`` for visualization rather than loading large datasets into memory.

**Documentation**
   Keep notes on dataset paths and structures for future reference.

**Validation**
   Always check dataset shapes and types before using in analysis pipelines.

Next Steps
----------

After mastering CXI file exploration:

* Apply discovered datasets in :doc:`bcdi_reconstruction_analysis` 
* Use file structure knowledge in :doc:`pole_figure` for 3D visualization
* Integrate exploration into automated :doc:`../tutorials/pipeline_tutorial` workflows
* Develop custom analysis scripts based on your specific CXI file structures

Related Examples
----------------

* :doc:`bcdi_reconstruction_analysis` - Analysis workflows using discovered datasets
* :doc:`pole_figure` - 3D visualization techniques for CXI data
* :doc:`../tutorials/step_by_step_tutorial` - Manual processing with CXI file loading
