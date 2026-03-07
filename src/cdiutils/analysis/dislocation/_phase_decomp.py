import numpy as np

from ._ring import center_angles


## utils phase decomposition
def decompose_experimental_phase(theta, phi_exp):
    """
    Decompose an experimental phase signal into linear, low-frequency,
    and second-harmonic (2θ) oscillatory components.

    The procedure consists of:
    1) Removing a global linear trend from the experimental phase.
    2) Fitting and subtracting low-frequency angular components
       (cos θ, sin θ, and constant offset).
    3) Isolating and fitting the second-harmonic oscillation
       (cos 2θ, sin 2θ).
    4) Reconstructing filtered phase components with proper angular
       centering.

    Parameters
    ----------
    theta : array_like
        Angular coordinate (in radians) at which the phase is sampled.
    phi_exp : array_like
        Experimental phase values corresponding to `theta`.

    Returns
    -------
    f_oscillation_final : numpy.ndarray
        Experimental phase after removal of low-frequency components,
        retaining the dominant oscillatory content and linear trend.
    f_fitoscillation_final : numpy.ndarray
        Fitted second-harmonic (cos 2θ, sin 2θ) oscillatory component,
        centered in angular space.
    coeffs : numpy.ndarray
        Least-squares coefficients for the second-harmonic fit
        [A_cos2θ, A_sin2θ].
    f_linear : numpy.ndarray
        Linear trend reconstructed from the filtered phase.
    coeffs_linear : numpy.ndarray
        Coefficients of the final linear fit [slope, intercept].

    Notes
    -----
    - Least-squares fitting is performed using ``np.linalg.lstsq``.
    - Low-frequency contributions (cos θ, sin θ, constant) are explicitly
      removed to avoid contamination of the 2θ harmonic.
    - The function assumes that `theta` and `phi_exp` have the same shape.
    - The function `center_angles` is expected to wrap or center angular
      phase values consistently within a chosen interval (e.g. [-π, π]).

    Examples
    --------
    >>> f_phase, f_fit2theta, coeffs2, f_lin, lin_coeffs = \
    ...     decompose_experimental_phase(theta, phi_exp)
    """
    coeffs_linear = np.polyfit(theta, phi_exp, 1)
    f_linear = np.polyval(coeffs_linear, theta)
    f_oscillation_0 = phi_exp - f_linear

    X_full = np.column_stack(
        [
            np.cos(theta),
            np.sin(theta),
            np.cos(2 * theta),
            np.sin(2 * theta),
            np.ones_like(theta),
        ]
    )
    coeffs, *_ = np.linalg.lstsq(X_full, f_oscillation_0, rcond=None)
    low_freq_fit = X_full[:, [0, 1, 4]] @ coeffs[[0, 1, 4]]
    f_oscillation_final = center_angles(f_oscillation_0 - low_freq_fit)
    f_filterlowfreq_final = f_oscillation_final + f_linear
    coeffs_linear = np.polyfit(theta, f_filterlowfreq_final, 1)
    f_linear = np.polyval(coeffs_linear, theta)
    f_oscillation_final = f_filterlowfreq_final

    X_full = np.column_stack(
        [
            np.cos(2 * theta),
            np.sin(2 * theta),
        ]
    )
    coeffs, *_ = np.linalg.lstsq(X_full, f_oscillation_0, rcond=None)
    high_freq_fit = X_full @ coeffs
    f_fitoscillation_final = center_angles(high_freq_fit)
    return (
        f_oscillation_final,
        f_fitoscillation_final,
        coeffs,
        f_linear,
        coeffs_linear,
    )
