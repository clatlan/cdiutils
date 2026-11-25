Bragg Signal Clustering, Aliens and Background
===============================================

This guide shows how to separate the Bragg signal of a particle from
aliens (neighbouring particles, parasitic peaks) and from the diffuse
background, with :func:`~cdiutils.utils.cluster_bragg_signal` and
:func:`~cdiutils.utils.background_map`.

**Use cases:**

* A second Bragg peak (another particle) appears in the rocking scan
* The peak is off-centre after preprocessing because a second peak
  pulled the centre of mass
* Choosing a background subtraction for weak, high-q fringes

Overview
--------

The clustering follows E. Bellec's ``alienclustering`` method
(https://github.com/ewbellec/alienclustering):

1. rescale the intensity to a log scale bounded to [0, 1];
2. keep the voxels above a threshold (typically 0.3);
3. optionally merge nearby voxels with a maximum filter, so that the
   fringes and facet streaks of the peak join its cluster;
4. group the voxels with DBSCAN (``eps = sqrt(2)``, ``min_samples = 8``).

The cluster holding the maximum is the Bragg peak; the other clusters are
alien candidates.

Clustering the Bragg Signal
---------------------------

.. code-block:: python

   import numpy as np
   from cdiutils.utils import cluster_bragg_signal

   # data: intensity in photons, detector frame (frames, rows, columns)
   labels, bragg_label, bragg_region = cluster_bragg_signal(
       data,
       threshold=0.3,   # log-scale threshold
       merge_size=5,    # below the fringe spacing (pixels)
       dilation=3,      # margin around the Bragg cluster
       mask=mask,       # detector gaps, dead pixels
   )

* ``labels``: integer label of every clustered voxel (0 = not clustered)
* ``bragg_label``: label of the cluster holding the maximum
* ``bragg_region``: the Bragg cluster dilated by ``dilation`` voxels

**Choosing merge_size:** without merging, the fringe shells and facet
streaks of a strongly scattering particle are often split into tens of
separate clusters. Use a size a little below the fringe spacing so that
they join the Bragg cluster, but keep it smaller than the distance to a
real alien, otherwise the alien merges too.

Masking Aliens
--------------

A separate cluster is an alien only if it is far from the Bragg region;
closer clusters are detached pieces of its own fringes:

.. code-block:: python

   from scipy.ndimage import binary_dilation, distance_transform_edt

   sizes = np.bincount(labels.ravel())
   distance = distance_transform_edt(~bragg_region)
   aliens = [
       k for k in range(1, sizes.size)
       if k != bragg_label
       and sizes[k] >= 20                          # not a noise fragment
       and distance[labels == k].min() > 10        # far from the peak
   ]
   alien_mask = binary_dilation(np.isin(labels, aliens), iterations=3)
   alien_mask &= ~bragg_region

   phasing_mask = mask | alien_mask

**Mask aliens, do not set them to zero.** A masked voxel is free during
phase retrieval; a zeroed voxel imposes a wrong measurement.

When a second peak is present, also centre the reconstruction window on
the Bragg cluster's centre of mass, not on that of the whole frame.

Background
----------

The diffuse background around a Bragg peak (air, sample environment,
substrate) varies slowly across the detector and hardly at all over a
rocking scan. :func:`~cdiutils.utils.background_map` estimates it per
detector pixel from the frames where that pixel is outside the Bragg
region:

.. code-block:: python

   from cdiutils.utils import background_map

   background = background_map(
       data,
       bragg_region | alien_mask,
       mask=mask,
       sigma=10,          # smoothing in the detector plane (pixels)
       pure_pixels=True,  # only pixels outside the region in every frame
   )
   cleaned = np.clip(data - background[None], 0, None)

With ``pure_pixels=False`` a pixel contributes from all frames in which
it is outside the region. Weak fringes crossing that pixel in some frames
then leak into the estimate, and the map is overestimated near the peak.
With ``pure_pixels=True`` the map near the peak comes from the smoothing
of the surrounding pure pixels.

Recommendations
---------------

* Do not mask or zero everything outside the Bragg region by default: the
  weak high-q fringes and facet streaks are at the background level and
  carry the resolution.
* Compare background treatments on the reconstruction (agreement between
  runs, fit to the data, outline), not on the data alone.
* A constant subtraction above about one photon per voxel removes every
  single-photon event, most of the weak high-q signal.

Example: CITIUS data, ID01 (2026)
---------------------------------

On Cu particles measured with the CITIUS detector, the clustering without
``merge_size`` found ~20 "aliens" that were the facet streak and fringe
fragments; with ``merge_size=5`` it found none on one particle and four
genuine ones on another (including a weak isolated peak). No subtraction,
a constant of 0.4 photon, and the per-pixel map gave the same objects
within the run-to-run spread; masking everything outside the Bragg region
made the runs less consistent.

See Also
--------

* :doc:`../api/utils` — API reference
* ``examples/bragg_signal_clustering.ipynb`` — worked example with a
  synthetic peak, alien and background
* :func:`~cdiutils.utils.hot_pixel_filter`
