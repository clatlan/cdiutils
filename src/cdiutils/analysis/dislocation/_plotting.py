import matplotlib.pyplot as plt
from matplotlib import rcParams

from ._phase_decomp import decompose_experimental_phase
from ._ring import center_angles


# plotting
def plot_phase_data_comparison_exp_to_theo(
    exp_angle,
    exp_phase,
    theo_phases,
    labels,
    save_path=None,
    marker_size=5,
    line_width=5,
    alpha=0.7,
    ref_band=0.1,
    show_band=True,
    figsize=(9, 5),
    band_theo=False,
    show_slope=False,
    filter_low_freq=True,
    fix_exp_slope=None,
    font_size=12,
    offset_theta=None,
    ncol=3,
):
    '''
    Plot the experimental phase and the theoretical phase.
    Parameters
    ----------
    exp_angle : array-like
        The experimental angle.
    exp_phase : array-like
        The experimental phase.
    theo_phases : array-like
        The theoretical phases.
    labels : array-like
        The labels for the theoretical phases.
    save_path : str, optional
        The path to save the plot.
    marker_size : int, optional
        The size of the markers.
    line_width : int, optional
        The width of the lines.
    alpha : float, optional
        The alpha of the lines.
    ref_band : float, optional
        The reference band.
    show_band : bool, optional
        Whether to show the band.
    figsize : tuple, optional
        The size of the figure.
    band_theo : bool, optional
        Whether to show the band of the theoretical phases.
    show_slope : bool, optional
        Whether to show the slope of the experimental phase.
    filter_low_freq : bool, optional
        Whether to filter the low frequency.
    fix_exp_slope : float, optional
        The slope of the experimental phase.
    font_size : int, optional
        The size of the font.
    offset_theta : float, optional
        The offset of the theta.
    ncol : int, optional
        The number of columns in the legend.
    '''
    # setup plot
    rcParams["font.size"] = font_size
    rcParams.update(
        {
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "savefig.bbox": "tight",
        }
    )
    # Preprocess experimental phase
    filtred_phase, filter_fit, _, f_linear, coeffs_linear = (
        decompose_experimental_phase(exp_angle, exp_phase)
    )
    slope_exp, intercept_exp = coeffs_linear
    if fix_exp_slope is not None:
        slope_exp = fix_exp_slope
        f_linear = slope_exp * exp_angle + intercept_exp

    y_exp = (
        filtred_phase - f_linear if filter_low_freq else exp_phase - f_linear
    )

    if offset_theta is None:
        offset_theta = 0
    # Setup 2-row subplot (main + residual)
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    # === MAIN PLOT ===
    if show_slope:
        exp_legend = f"Exp (slope={slope_exp:.2f})"
    else:
        exp_legend = "Exp"

    h_exp = ax1.plot(
        exp_angle - offset_theta,
        y_exp,
        "^",
        markersize=marker_size,
        label=exp_legend,
        color="black",
        alpha=alpha,
    )[0]
    if show_band:
        ax1.fill_between(
            exp_angle - offset_theta,
            y_exp - ref_band,
            y_exp + ref_band,
            color="gray",
            alpha=0.2,
        )
    handles = [h_exp]
    labels_all = [exp_legend]

    for i, theo_phase in enumerate(theo_phases):
        predicted_phase, pred_fit, _, f_linear_theo, coeffs_linear = (
            decompose_experimental_phase(exp_angle, theo_phase)
        )
        y_theo = (
            predicted_phase - f_linear_theo
            if filter_low_freq
            else theo_phase - f_linear_theo
        )
        slope_theo, intercept_theo = coeffs_linear
        if show_slope:
            theo_legend = labels[i] + f" (slope={slope_theo:.2f})"
        else:
            theo_legend = labels[i]

        (line,) = ax1.plot(
            exp_angle - offset_theta,
            y_theo,
            "-",
            linewidth=line_width,
            label=theo_legend,
            alpha=alpha,
        )
        handles.append(line)
        labels_all.append(theo_legend)

        if show_band and band_theo:
            ax1.fill_between(
                exp_angle - offset_theta,
                y_theo - ref_band,
                y_theo + ref_band,
                color=line.get_color(),
                alpha=0.2,
            )

        # === RESIDUAL SUBPLOT ===
        y_diff = center_angles(y_exp - y_theo)
        ax2.plot(
            exp_angle - offset_theta,
            y_diff,
            "-",
            linewidth=line_width,
            color=line.get_color(),
            alpha=alpha,
        )

    # === Styling for Main Plot ===
    ax1.set_ylabel("Phase Residual (rad)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.tick_params(labelsize=font_size - 2)

    # === Styling for Residual Subplot ===
    ax2.set_xlabel("Polar Angle (rad)")
    ax2.set_ylabel("Diff.")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.tick_params(labelsize=font_size - 2)

    # === Unified Legend Above Plots ===
    fig.legend(
        handles,
        labels_all,
        loc="upper center",
        frameon=False,
        ncol=ncol,
        bbox_to_anchor=(0.5, 1.12),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    rcParams["font.size"] = 12

    plt.show()


