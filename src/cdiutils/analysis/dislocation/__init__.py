# cdiutils/analysis/dislocation/__init__.py

# Core exports (should not import optional deps at import time)

from ._clustering import clusters_dislo_strain_map
from ._geometry import (
    create_circular_mask,
    extract_structure,
    fit_line_3d,
    generate_filled_cylinder,
    plot_phase_around_dislo,
)
from ._phase_decomp import decompose_experimental_phase
from ._ring import (
    center_angles,
    dislo_process_phase_ring,
    remove_large_jumps,
)
from ._strain_map import map_min_gradient
from ._theory import (
    angle_between_vectors,
    closest_to_zero_in_array,
    dislo_phase_model,
    dislo_rotation_matrix_real_to_theo,
    normalize_vector,
    normalize_vectors_3d,
    project_vector,
    signed_angle_3d,
    transform_known_vector_to_crystallographic,
)

__all__ = [
    "map_min_gradient",
    "clusters_dislo_strain_map",
    "extract_structure",
    "fit_line_3d",
    "generate_filled_cylinder",
    "create_circular_mask",
    "plot_phase_around_dislo",
    "dislo_process_phase_ring",
    "remove_large_jumps",
    "center_angles",
    "dislo_phase_model",
    "dislo_rotation_matrix_real_to_theo",
    "signed_angle_3d",
    "angle_between_vectors",
    "project_vector",
    "normalize_vector",
    "transform_known_vector_to_crystallographic",
    "normalize_vectors_3d",
    "closest_to_zero_in_array",
    "decompose_experimental_phase",
    "plot_phase_data_comparison_exp_to_theo",
]

def __getattr__(name: str):
    # Lazy import for plotting (or anything that may import heavy/optional deps)
    if name == "plot_phase_data_comparison_exp_to_theo":
        from ._plotting import plot_phase_data_comparison_exp_to_theo
        return plot_phase_data_comparison_exp_to_theo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
