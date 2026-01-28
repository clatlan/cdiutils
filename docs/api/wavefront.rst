Wavefront Module
================

The :mod:`cdiutils.wavefront` module provides functions for X-ray wavefront
analysis and propagation, particularly useful for characterising focusing
optics and probe analysis in BCDI experiments.

Functions
---------

Wavefront Propagation
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: cdiutils.wavefront

.. autofunction:: angular_spectrum_propagation

.. autofunction:: focus_probe

.. autofunction:: plot_propagated_probe


Probe Analysis
~~~~~~~~~~~~~~

.. autofunction:: probe_metrics

.. autofunction:: get_width_metrics

.. autofunction:: probe_focus_sweep

.. autofunction:: get_focal_distances


Examples
--------

Basic probe metrics analysis::

    from cdiutils.wavefront import probe_metrics
    import numpy as np
    
    # Assuming 'probe' is a 2D complex array from reconstruction
    pixel_size = (55e-9, 55e-9)  # metres
    
    # Analyse probe characteristics
    fig, metrics = probe_metrics(
        probe=probe,
        pixel_size=pixel_size,
        zoom_factor="auto",  # automatically determine zoom
        verbose=True
    )

Wavefront propagation::

    from cdiutils.wavefront import angular_spectrum_propagation
    
    # Propagate probe to focal plane
    propagation_distance = 0.1  # metres
    wavelength = 1.5e-10  # metres
    pixel_size = 55e-9  # metres
    
    propagated = angular_spectrum_propagation(
        wavefront=probe,
        propagation_distance=propagation_distance,
        wavelength=wavelength,
        pixel_size=pixel_size,
        magnification=1.0
    )

Find focal plane::

    from cdiutils.wavefront import get_focal_distances, focus_probe
    
    # Determine focal distances in x and y
    focal_distances = get_focal_distances(
        probe=probe,
        wavelength=wavelength,
        pixel_size=pixel_size,
        propagation_range=(-0.5, 0.5),  # metres
        num_steps=100
    )
    
    # Focus probe to optimal plane
    focused_probe = focus_probe(
        probe=probe,
        wavelength=wavelength,
        pixel_size=pixel_size,
        focal_distances=focal_distances
    )

See Also
--------
:class:`cdiutils.process.PostProcessor` : Uses probe for normalisation
:class:`cdiutils.pipeline.BcdiPipeline` : Access probe from reconstruction
