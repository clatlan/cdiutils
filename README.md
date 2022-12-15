# cdiutils

My python package to help Bragg Coherent Diffraction Imaging (BCDI) practionionners in their analysis and visualisation workflow. This is a 'personal' package developped during my PhD. This provides utility functions to analyse ```bcdi```-processed data and to plot them for potential publication.

It requires the following libraries :

* bcdi
* xrayutilities
* silx
* numpy
* scipy
* matplotlib
* colorcet


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

cdiutils.plot.plot.plot_diffraction_patterns

![Diffraction Patterns](https://github.com/clatlan/cdiutils/blob/master/images/diffraction_patterns.png)
