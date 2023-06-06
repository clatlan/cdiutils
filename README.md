# cdiutils

[![DOI](https://zenodo.org/badge/360442527.svg)](https://zenodo.org/badge/latestdoi/360442527)

My python package to help Bragg Coherent X-ray Diffraction Imaging (BCDI) practitioners in their analysis and visualisation workflow. I developped the package during my PhD.

The package allows for the handling of the three main stages of a BCDI data processing workflow:

* the proprocessing (data centering and cropping)
* the phase retrieval using PyNX.
* the post processing (orthogonalization, phase manipulation, strain computation etc.)

It is assumed that the phase retrieval is carried out by the PyNX package (see [http://ftp.esrf.fr/pub/scisoft/PyNX/doc/]()). The BcdiPipeline class runs the three stages and can manage connection to different machines if required (GPU  for phase retrieval).

Pre- and post-processing do not require GPUs and can be run using the present package (cdiutils backend) or the `bcdi` package (see [https://github.com/carnisj/bcdi]()) (bcdi backend).

The package also provide utility fonctions to analyze processed data and plot them for potential publications.

## Installation

You can install the package with the following command:

```
pip install git+https://github.com/clatlan/cdiutils.git
```

Upgrade your environment with a new version of the package:

```
pip install -U --no-deps git+https://github.com/clatlan/cdiutils.git
```

## Getting started
Once the package installed, you can try it right away using the notebook template directly accessible with the command:

```
prepare_bcdi_notebook.py [path_to_destination]
```
This will generate a notebook template at the given destination.

## Processing BCDI data

Once data are processed, the BcdiPipeline instance saves the data in .npz, .h5 .cxi and .vti files following the CXI file format convention (see [https://www.cxidb.org/cxi.html]()). It also plots summary and debug figures such as:

* **Summary Slice Plot**
  ![summary](https://github.com/clatlan/cdiutils/blob/master/images/cdiutils_S311_summary_slice_plot.png)
* **Isosurface determination**
  ![isosurface](https://github.com/clatlan/cdiutils/blob/master/images/cdiutils_S311_amplitude_distribution_plot.png)
* **Different strain computation methods**
  ![strain](https://github.com/clatlan/cdiutils/blob/master/images/cdiutils_S311_different_strain_methods.png)
* **Orthogonalization in the direct space**
  ![ortho](https://github.com/clatlan/cdiutils/blob/master/images/cdiutils_S311_direct_lab_orthogonaliztion_plot.png)

## Slice plot

cdiutils.plot.slice.plot_3D_volume_slices function

* **Bragg electron density slice plot**

![Electron density](https://github.com/clatlan/cdiutils/blob/master/images/electron_density.png)

* **Comparing contour of support**

![Contour](https://github.com/clatlan/cdiutils/blob/master/images/contour.png)

* **Phase slice plot**

![Phase](https://github.com/clatlan/cdiutils/blob/master/images/phase.png)

## Cross section quiver

cdiutils.plot.quiver.quiver_plot

* The cross section quiver allows to plot cross section of strain and displacement field on the same plot.
  ![Cross Section Quiver](https://github.com/clatlan/cdiutils/blob/master/images/cross_section_quiver.png)
* For different conditions
  ![Quivers](https://github.com/clatlan/cdiutils/blob/master/images/multi_cross_sections.png)
* Can also be used to plot the curves/arrows only
  ![arrows](https://github.com/clatlan/cdiutils/blob/master/images/arrows.png)
* Can also be used to plot basic cross sections
  ![strain](https://github.com/clatlan/cdiutils/blob/master/images/strain.png)

## Diffraction pattern plots in the reciprocal space

cdiutils.plot.slice.plot_diffraction_patterns

![Diffraction Patterns](https://github.com/clatlan/cdiutils/blob/master/images/diffraction_patterns.png)
