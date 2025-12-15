import numpy as np
from scipy.fft import fftn, fftshift

from .utils import get_reciprocal_voxel_size

__all__ = [
    # geometry creation
    "make_box",
    "make_ellipsoid",
    "make_cylinder",
    # phase manipulation
    "add_linear_phase",
    "add_quadratic_phase",
    "add_displacement_field",
    "add_random_phase",
    # diffraction simulation
    "simulate_diffraction",
    "add_noise_and_scale",
    "add_noise_frame_by_frame",
    # utilities (re-exported from utils)
    "get_reciprocal_voxel_size",
]


def simulate_diffraction(
    obj: np.ndarray,
    photon_budget: float | None = None,
    max_intensity: float | None = None,
    scale: float = 1.0,
    poisson_statistics: bool = True,
    convention: str | None = None,
) -> np.ndarray:
    """Simulate diffraction pattern from a real-space object.

    This uses a single, consistent numerical convention: a forward
    n-dimensional FFT followed by fftshift to place the Bragg peak
    at the centre of the array::

        reciprocal_obj = fftshift(fftn(obj))

    The ``convention`` argument is kept for future extension (e.g. axis
    re-ordering or additional phase factors) but does not change the
    underlying FFT operator. This keeps the implementation simple and
    avoids mixing ``fftn``/``ifftn`` conventions. If you need to match a
    sign convention used in analytical Bragg-CDI derivations or in
    external tools, do so via the definition of the q-grid (e.g. flipping
    an axis or using ``-phase`` consistently), not by swapping FFT
    directions here.

    Intensity is computed as ``|reciprocal_obj|**2``. Optionally, the
    result can be scaled to ``photon_budget`` and/or ``max_intensity``.
    Note that first scaling to ``max_intensity`` will discard photon
    budget scaling if both are provided. Scale is applied in
    any case as a multiplicative factor.

    For noise modelling refer to :func:`add_noise`.

    Args:
        obj: Real-space complex object to simulate.
        max_intensity: Maximum intensity value for final scaling.
            If None, no scaling is applied.
        photon_budget: Total photon budget for the exposure.
        scale: Multiplicative scale factor applied to intensity.
            Defaults to 1.0.
        poisson_statistics: If True, apply Poisson statistics to
            the final intensity (photon counting). Default is True.
        convention: Placeholder for future FFT/q-space conventions.
            Currently ignored except for being accepted for API
            compatibility.

    Returns:
        Simulated diffraction pattern (intensity).

    Raises:
        ValueError: If max_intensity or total_photons are negative.
    """
    reciprocal_obj = fftshift(fftn(obj))
    intensity = np.abs(reciprocal_obj) ** 2
    if photon_budget is not None:
        scale *= photon_budget / intensity.sum()
    if max_intensity is not None:
        scale *= max_intensity / intensity.max()

    intensity = intensity * scale

    if poisson_statistics:
        intensity = np.random.poisson(intensity)
    return intensity


def add_noise(
    data: np.ndarray,
    gaussian_mean: float = 0.0,
    gaussian_std: float = 0.0,
    poisson_background: np.ndarray | float | None = None,
    poisson_statistics: bool = False,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Add noise to data with configurable Gaussian and Poisson
    components.

    This general-purpose function allows flexible noise modelling
    by combining Gaussian and Poisson noise sources. It can be
    called multiple times to build up complex noise models from
    different physical sources (dark current, readout noise, air
    scattering, fluorescence, etc.).

    Args:
        data: input data array (any shape).
        gaussian_mean: mean of Gaussian noise to add. Default is
            0.0.
        gaussian_std: standard deviation of Gaussian noise.
            Default is 0.0.
        poisson_background: background for Poisson sampling. Can
            be:
            - None: no Poisson background added
            - float: uniform background value
            - np.ndarray: spatially-varying background (must match
              data shape)
            Default is None.
        poisson_statistics: if True, apply Poisson
            statistics to the data itself (photon counting).
            Default is False.
        scale: multiplicative factor applied to data before
            adding noise. Useful for unit conversions or intensity
            scaling. Default is 1.0.

    Returns:
        Data with added noise, same shape as input. Values are
        converted to float64 for accumulation.

    Raises:
        TypeError: if data is not a numpy array.
        ValueError: if gaussian_std is negative, or if
            poisson_background array shape doesn't match data
            shape, or if scale is not positive.

    Examples:
        >>> # add dark current (Gaussian thermal noise)
        >>> noisy = add_noise(data, gaussian_mean=0.5,
        ...                   gaussian_std=0.75)

        >>> # add readout noise (Gaussian electronics noise)
        >>> noisy = add_noise(noisy, gaussian_std=0.5)

        >>> # add uniform air scattering (Poisson on uniform)
        >>> noisy = add_noise(noisy, poisson_background=2.0)

        >>> # add spatially-varying air scattering (beam profile)
        >>> beam_profile = create_beam_profile(data.shape)
        >>> noisy = add_noise(noisy, poisson_background=beam_profile)

        >>> # apply Poisson statistics to signal
        >>> noisy = add_noise(data, poisson_statistics=True)

        >>> # scale data and add noise
        >>> noisy = add_noise(data, scale=1.5, gaussian_std=1.0)

        >>> # combine multiple sources in one call
        >>> noisy = add_noise(data, gaussian_mean=0.5,
        ...                   gaussian_std=0.75,
        ...                   poisson_background=2.0,
        ...                   scale=1.2)
    """
    # validate inputs
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be np.ndarray, got {type(data)}")

    if gaussian_std < 0:
        raise ValueError(f"gaussian_std must be >= 0, got {gaussian_std}")

    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    if poisson_background is not None:
        if isinstance(poisson_background, np.ndarray):
            if poisson_background.shape != data.shape:
                raise ValueError(
                    f"poisson_background shape {poisson_background.shape} "
                    f"must match data shape {data.shape}"
                )
            if np.any(poisson_background < 0):
                raise ValueError(
                    "poisson_background array contains negative values"
                )
        elif isinstance(poisson_background, (int, float)):
            if poisson_background < 0:
                raise ValueError(
                    f"poisson_background must be >= 0, got "
                    f"{poisson_background}"
                )
        else:
            raise TypeError(
                f"poisson_background must be np.ndarray, float, or None, "
                f"got {type(poisson_background)}"
            )

    # apply scale factor to data
    scaled_data = data * scale

    # start with scaled data as float for accumulation
    if poisson_statistics:
        # apply Poisson statistics to the data itself
        noisy_data = np.random.poisson(scaled_data).astype(float)
    else:
        noisy_data = scaled_data.astype(float)

    # add Gaussian noise
    if gaussian_mean != 0.0 or gaussian_std > 0:
        gaussian_noise = np.random.normal(
            gaussian_mean, gaussian_std, data.shape
        )
        noisy_data += gaussian_noise

    # add Poisson noise from background
    if poisson_background is not None:
        if isinstance(poisson_background, (int, float)):
            # uniform background
            background = np.full(data.shape, poisson_background)
        else:
            # spatially-varying background (already validated)
            background = poisson_background

        poisson_noise = np.random.poisson(background)
        noisy_data += poisson_noise

    return np.maximum(noisy_data, 0)
