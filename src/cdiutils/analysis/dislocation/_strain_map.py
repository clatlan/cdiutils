import h5py
import numpy as np
from matplotlib import rcParams

from cdiutils.io.vtk import save_as_vti
from cdiutils.plot import plot_volume_slices
from cdiutils.utils import hybrid_gradient, nan_to_zero, zero_to_nan


def map_min_gradient(
    path: str = None,
    obj: np.ndarray = None,
    voxel_size: tuple | list = [1.0, 1.0, 1.0],
    nb_of_phase_to_test: int = 10,
    font_size: int = 12,
    path_to_save: str = "",
    save_filename_vti: str = "",
    plot_debug: bool = False,
    save_plot: bool = False,
    verbose: bool = False,
):
    """
    Map the minimum gradient of the phase data to the strain mask and strain amplitude.
    """
    def calculate_displacement_gradient(phase, voxel_size):
        return hybrid_gradient(phase, *voxel_size)

    def find_closest_to_zero(gradients):
        closest_to_zero_indices = np.argmin(
            np.abs(nan_to_zero(gradients)), axis=0
        )
        shape_data = gradients.shape
        i, j, k = np.meshgrid(
            np.arange(shape_data[1]),
            np.arange(shape_data[2]),
            np.arange(shape_data[3]),
            indexing="ij",
        )
        return gradients[closest_to_zero_indices, i, j, k]

    def calculate_strain(displacement_gradient_min):
        strain_amp = (
            displacement_gradient_min[0] ** 2
            + displacement_gradient_min[1] ** 2
            + displacement_gradient_min[2] ** 2
        ) ** 0.5
        strain_amp = strain_amp / np.nanmax(strain_amp)
        strain_mask = (
            (nan_to_zero(displacement_gradient_min) != 0.0)
            .astype(float)
            .sum(axis=0)
            != 0.0
        ).astype(float)
        return strain_amp, strain_mask

    if path is not None and path != "":
        obj_list = np.array(h5py.File(path)["entry_1/data_1/data"])[0]
    elif obj is not None:
        obj_list = obj
    else:
        print("no obj or path to mode are provided")
        obj_list = None
        return None, None
    if str(np.abs(obj_list).max()) == "nan":
        obj_list = nan_to_zero(np.abs(obj_list)) * np.exp(
            1j * nan_to_zero(np.angle(obj_list))
        )

    modulus = zero_to_nan(np.abs(obj_list))
    phase_0 = np.angle(np.exp(1j * zero_to_nan(np.angle(obj_list))))

    displacement_gradient_0 = calculate_displacement_gradient(
        phase_0, voxel_size
    )
    displacement_gradient_0 = np.asarray(displacement_gradient_0)

    all_gradients = [displacement_gradient_0]
    phase_futures = []
    for i_phase in np.linspace(-2 * np.pi, 2 * np.pi, nb_of_phase_to_test):
        phase_1 = np.angle(np.exp((phase_0 + i_phase) * 1j))
        phase_futures.append(
            calculate_displacement_gradient(phase_1, voxel_size)
        )

    all_gradients.extend(phase_futures)

    all_gradients_x = [grad[0] for grad in all_gradients]
    all_gradients_y = [grad[1] for grad in all_gradients]
    all_gradients_z = [grad[2] for grad in all_gradients]

    closest_futures = [
        find_closest_to_zero(np.array(all_gradients_x)),
        find_closest_to_zero(np.array(all_gradients_y)),
        find_closest_to_zero(np.array(all_gradients_z)),
    ]
    displacement_gradient_min = np.stack(closest_futures, axis=0)

    strain_amp, strain_mask = calculate_strain(displacement_gradient_min)

    shd_0, shd_x, sh_y, sh_z = displacement_gradient_min.shape
    if verbose:
        print(
            "Displacement Gradient Min shape:", displacement_gradient_min.shape
        )

        print(
            f"\nOriginal values at position ({shd_x // 2, sh_y // 2, sh_z // 2}) for x direction:"
        )
        print(
            f"Gradient 0: {displacement_gradient_0[0][shd_x // 2, sh_y // 2, sh_z // 2]}"
        )
        print(
            f"Min value:  {displacement_gradient_min[0][shd_x // 2, sh_y // 2, sh_z // 2]}"
        )

        print(
            f"\nOriginal values at position ({shd_x // 2, sh_y // 2, sh_z // 2}) for y direction:"
        )
        print(
            f"Gradient 0: {displacement_gradient_0[1][shd_x // 2, sh_y // 2, sh_z // 2]}"
        )
        print(
            f"Min value:  {displacement_gradient_min[1][shd_x // 2, sh_y // 2, sh_z // 2]}"
        )

        print(
            f"\nOriginal values at position ({shd_x // 2, sh_y // 2, sh_z // 2}) for z direction:"
        )
        print(
            f"Gradient 0: {displacement_gradient_0[2][shd_x // 2, sh_y // 2, sh_z // 2]}"
        )
        print(
            f"Min value:  {displacement_gradient_min[2][shd_x // 2, sh_y // 2, sh_z // 2]}"
        )

    if save_filename_vti:
        dict_to_vti = {
            "modulus": nan_to_zero(modulus),
            "phase_0": nan_to_zero(phase_0),
            "phase_1": zero_to_nan(phase_1),
            "displacement_gradient_min_x": nan_to_zero(
                displacement_gradient_min[0]
            ),
            "displacement_gradient_0_x": nan_to_zero(
                displacement_gradient_0[0]
            ),
            "displacement_gradient_min_y": nan_to_zero(
                displacement_gradient_min[1]
            ),
            "displacement_gradient_0_y": nan_to_zero(
                displacement_gradient_0[1]
            ),
            "displacement_gradient_min_z": nan_to_zero(
                displacement_gradient_min[2]
            ),
            "displacement_gradient_0_z": nan_to_zero(
                displacement_gradient_0[2]
            ),
            "strain_mask": strain_mask,
            "strain_amp": strain_amp,
        }
        save_as_vti(
            output_path=save_filename_vti,
            voxel_size=tuple(voxel_size),
            **dict_to_vti,
        )
    if plot_debug:
        rcParams["font.size"] = font_size
        rcParams.update(
            {
                "font.weight": "bold",
                "axes.titleweight": "bold",
                "axes.labelweight": "bold",
                "savefig.bbox": "tight",
            }
        )
        figure, _ = plot_volume_slices(
            zero_to_nan(phase_0),
            plot_type="contourf",
            figsize=(10, 4),
            label_size=12,
            cmap="jet",
            vmin=-np.pi,
            vmax=np.pi,
            title="original phase midlle slice",
        )
        if save_plot:
            figure.savefig(path_to_save + "original_phase.png")
        figure, _ = plot_volume_slices(
            zero_to_nan(phase_1),
            plot_type="contourf",
            figsize=(10, 4),
            label_size=12,
            cmap="jet",
            vmin=-np.pi,
            vmax=np.pi,
            title=f"phase + {np.round(i_phase, 4)} midlle slice",
        )
        if save_plot:
            figure.savefig(path_to_save + "plus_phase.png")
        figure, _ = plot_volume_slices(
            displacement_gradient_0[1],
            plot_type="contourf",
            figsize=(10, 4),
            label_size=12,
            cmap="jet",
            vmin=-0.3,
            vmax=0.3,
            title="original phase gradient midlle slice",
        )
        if save_plot:
            figure.savefig(path_to_save + "original_gradientphase.png")
        figure, _ = plot_volume_slices(
            displacement_gradient_min[1],
            plot_type="contourf",
            figsize=(10, 4),
            label_size=12,
            cmap="jet",
            vmin=-0.3,
            vmax=0.3,
            title=f"phase  + {np.round(i_phase, 4)} gradient midlle slice",
        )
        if save_plot:
            figure.savefig(path_to_save + "plus_gradientphase.png")
        rcParams["font.size"] = 12

    return nan_to_zero(strain_mask), nan_to_zero(strain_amp)

