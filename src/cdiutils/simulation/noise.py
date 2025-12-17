"""
Noise models for realistic detector simulation.

This module provides flexible noise modelling functions that can
combine multiple noise sources (Gaussian, Poisson, etc.) to simulate
realistic detector behaviour in BCDI experiments.
"""

import numpy as np


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
        data: Input data array (any shape).
        gaussian_mean: Mean of Gaussian noise to add. Default is
            0.0.
        gaussian_std: Standard deviation of Gaussian noise.
            Default is 0.0.
        poisson_background: Background for Poisson sampling. Can
            be:
            - None: no Poisson background added
            - float: uniform background value
            - np.ndarray: spatially-varying background (must match
              data shape)
            Default is None.
        poisson_statistics: If True, apply Poisson statistics to
            the data itself (photon counting). Default is False.
        scale: Multiplicative factor applied to data before adding
            noise. Useful for unit conversions or intensity
            scaling. Default is 1.0.

    Returns:
        Data with added noise, same shape as input. Values are
        converted to float64 for accumulation and clipped to
        non-negative.

    Raises:
        TypeError: If data is not a numpy array or if
            poisson_background has invalid type.
        ValueError: If gaussian_std is negative, or if
            poisson_background array shape doesn't match data
            shape, or if scale is not positive, or if background
            values are negative.

    Example:
        >>> # add dark current (Gaussian thermal noise)
        >>> noisy = add_noise(
        ...     data,
        ...     gaussian_mean=0.5,
        ...     gaussian_std=0.75,
        ... )
        >>>
        >>> # add readout noise (Gaussian electronics noise)
        >>> noisy = add_noise(noisy, gaussian_std=0.5)
        >>>
        >>> # add uniform air scattering (Poisson background)
        >>> noisy = add_noise(noisy, poisson_background=2.0)
        >>>
        >>> # add spatially-varying air scattering
        >>> beam_profile = create_beam_profile(data.shape)
        >>> noisy = add_noise(
        ...     noisy,
        ...     poisson_background=beam_profile,
        ... )
        >>>
        >>> # apply Poisson statistics to signal
        >>> noisy = add_noise(data, poisson_statistics=True)
        >>>
        >>> # scale data and add noise
        >>> noisy = add_noise(
        ...     data,
        ...     scale=1.5,
        ...     gaussian_std=1.0,
        ... )
        >>>
        >>> # combine multiple sources in one call
        >>> noisy = add_noise(
        ...     data,
        ...     gaussian_mean=0.5,
        ...     gaussian_std=0.75,
        ...     poisson_background=2.0,
        ...     scale=1.2,
        ... )

    Notes:
        This function can be called multiple times sequentially to
        build up complex noise models. Each call adds additional
        noise to the input data. The order matters when combining
        Poisson statistics with other noise sources.

        For frame-by-frame noise in rocking curves, use this
        function within a loop or use the
        :class:`BCDISimulator` class which handles
        realistic detector simulation automatically.
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
                    f"poisson_background shape "
                    f"{poisson_background.shape} must match data "
                    f"shape {data.shape}"
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
                f"poisson_background must be np.ndarray, float, or "
                f"None, got {type(poisson_background)}"
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

    # clip to non-negative values
    return np.maximum(noisy_data, 0)
