# cdiutils

[![DOI](https://zenodo.org/badge/360442527.svg)](https://zenodo.org/badge/latestdoi/360442527)
[![PyPI version](https://badge.fury.io/py/cdiutils.svg)](https://badge.fury.io/py/cdiutils)
[![License](https://img.shields.io/github/license/clatlan/cdiutils)](https://github.com/clatlan/cdiutils/blob/main/LICENSE)

My python package to help X-ray Bragg Coherent Diffraction Imaging (BCDI) practitioners in their analysis and visualisation workflows. I developed the package during my PhD.

The package is designed to handle the three primary stages of a BCDI data processing workflow:

* **Pre-processing** (data centering and cropping)
* **Phase retrieval**: utilises  PyNX for accurate phasing (refer to [PyNX documentation](https://pynx.esrf.fr/en/latest/)).
* **Post-processing** (orthogonalisation, phase manipulation, strain computation etc.)

It is assumed that the phase retrieval is conducted using the PyNX package. The `BcdiPipeline` class runs all three stages and can manage connections to different machines, especially for GPU-based phase retrieval.

Some features of this package include:

* **Flexibility in Hardware:** While the phase retrieval stage may leverage GPUs, pre- and post-processing can be executed without GPU support.
* **Utility Functions:** The package provides utility functions to analyse processed data and generate plots suitable for potential publications.

For a visual wrap-up, see the associated poster presented at [XTOP24](https://xtop2024.sciencesconf.org/):
![xtop_poster](https://github.com/clatlan/cdiutils/blob/master/images/XTOP_24_cdiutils_poster_200_dpi.png)


## Installation

### Using pip (from PyPI - simplest way)

Install directly from PyPI:

```bash
pip install cdiutils
```

### Using conda (recommended for dependency management)

**Option 1: Create a new conda environment with all dependencies**

```bash
# create conda environment directly from GitHub
conda env create -f https://raw.githubusercontent.com/clatlan/cdiutils/master/environment.yml

# activate the environment
conda activate cdiutils-env

# install cdiutils from PyPI
pip install cdiutils
```

**Option 2: Install dependencies in your existing conda environment**

```bash
# download and install dependencies using conda solver
conda env update -f https://raw.githubusercontent.com/clatlan/cdiutils/master/environment.yml

# install cdiutils from PyPI
pip install cdiutils
```

For development (includes documentation and testing tools):

```bash
# create development environment directly from GitHub
conda env create -f https://raw.githubusercontent.com/clatlan/cdiutils/master/environment-dev.yml

# activate the environment
conda activate cdiutils-dev-env

# install cdiutils in development mode (requires cloning)
git clone https://github.com/clatlan/cdiutils.git
cd cdiutils
pip install -e .
```

### Directly from GitHub (development version)

You can also install the latest development version directly from GitHub:

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/clatlan/cdiutils.git
```

To update your environment with the latest commits:

```bash
pip install -I --no-deps git+https://github.com/clatlan/cdiutils.git
```

_Note: Check out the dev branch for the latest features and bug fixes. The dev branch is not guaranteed to be stable._


## Getting started

Once the package is installed, you can try it right away using the notebook template directly accessible with the command:

```bash
prepare_bcdi_notebooks
```

This will generate a notebook templates in your current directory.

## Processing BCDI data

Once data are processed, the `BcdiPipeline` instance saves the data in .npz, .cxi and .vti files following the CXI file format convention (see [https://www.cxidb.org/cxi.html]()). It also plots summary and debug figures such as:

* **Summary Slice Plot**
  ![summary](https://github.com/clatlan/cdiutils/blob/master/images/cdiutils_S311_summary_slice_plot.png)
* **Isosurface determination**
  ![isosurface](https://github.com/clatlan/cdiutils/blob/master/images/cdiutils_S311_amplitude_distribution_plot.png)
* **Different strain computation methods**
  ![strain](https://github.com/clatlan/cdiutils/blob/master/images/cdiutils_S311_different_strain_methods.png)

## BCDI reconstruction analysis
If want to analyse and compare your reconstructions, check out the example notebook [bcdi_reconstruction_analysis.ipynb](https://github.com/clatlan/cdiutils/blob/master/examples/bcdi_reconstruction_analysis.ipynb) in the `examples` folder. This notebook provides a comprehensive overview of the analysis process, including:
* **Slice plots of any quantity you like (here phase) across different conditions:**
  ![](https://github.com/clatlan/cdiutils/blob/master/images/multi_slice_plots_phase.png)


* **Reciprocal space plots in the orthogonal frame (lab frame)**
  ![](https://github.com/clatlan/cdiutils/blob/master/images/reciprocal_space_q_lab.png)

* **Histogram plots of any quantity you like across different conditions:**
  ![](https://github.com/clatlan/cdiutils/blob/master/images/strain_histograms.png)


## Cross section quiver
The cross section quiver is nice tool for visualising the strain and displacement fields and their relationship in BCDI data. 

* The cross section quiver allows to plot cross section of strain and displacement field on the same plot.
  ![Cross Section Quiver](https://github.com/clatlan/cdiutils/blob/master/images/cross_section_quiver.png)
* For different conditions
  ![Quivers](https://github.com/clatlan/cdiutils/blob/master/images/multi_cross_sections.png)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please open an issue on the [GitHub repository](https://github.com/clatlan/cdiutils/issues).

