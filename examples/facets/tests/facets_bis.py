#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.fft import fftn, ifftshift, fftshift, ifftn
import io
import logging

# ---- cdi plots on remote -- 
import cdiutils
import cdiutils.plot.slice as slice_module # Import the specific submodule
import functools
import matplotlib.pyplot as plt
import io

# --- Step 1: Store a reference to the original function ---
original_plot_volume_slices = slice_module.plot_volume_slices

# --- Step 2: Create a patched wrapper function ---
@functools.wraps(original_plot_volume_slices)
def patched_plot_volume_slices(*args, **kwargs):
    """
    A wrapper for plot_volume_slices that captures the figure object
    and prevents it from being closed when show=False.
    """
    # Force show=False so the function returns the figure object
    # instead of blocking with plt.show().
    kwargs['show'] = False

    # Temporarily replace plt.close with a function that does nothing.
    # This is the "monkey-patch" that prevents the figure from closing.
    original_close = plt.close
    
    def do_nothing_close(fig):
        # This function will be called by cdiutils, but it won't
        # actually close the figure we want to keep.
        # print(f"Monkey-patch: Intercepted and prevented close for figure {fig.number}")
        pass

    plt.close = do_nothing_close
    
    # Call the original function. It will now run, but its call
    # to plt.close() will be intercepted by our do_nothing_close function.
    fig, axes = original_plot_volume_slices(*args, **kwargs)
    
    # IMPORTANT: Restore the original plt.close function immediately
    # so we don't break other parts of matplotlib.
    plt.close = original_close
    
    # The figure is now left open, and we return it as the original did.
    return fig, axes

# --- Step 3: Apply the patch ---
# Replace the function in the cdiutils module with our patched version.
slice_module.plot_volume_slices = patched_plot_volume_slices

# You can also patch it on the higher-level namespace if you imported it there
cdiutils.plot.plot_volume_slices = patched_plot_volume_slices

print("Monkey-patched cdiutils.plot.plot_volume_slices to prevent auto-closing.")
# -- end -- #

#from pynx.scattering.fhkl import Fhkl_thread
import functools
import pyopencl as cl
from pynx.scattering import fhkl as fhkl_module # Get a handle to the module
from pynx.scattering.fhkl import Fhkl_thread as original_Fhkl_thread

# -- Step 1: Helper function to find the best GPU --
def find_best_gpu():
    """Finds the first available OpenCL GPU and returns its platform and device names."""
    for platform in cl.get_platforms():
        # Avoid the CPU-based Intel OpenCL platform if possible
        if 'intel' in platform.name.lower():
            continue
        try:
            gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
            if gpu_devices:
                # Found a platform with a GPU, return its info
                return platform.name, gpu_devices[0].name
        except cl.Error:
            continue
    raise RuntimeError("Monkey-patch could not find a suitable OpenCL GPU.")

# -- Step 2: Create the wrapper function --
# Use a simple class to store the discovered names so we only search once.
class GpuInfo:
    platform_name = None
    gpu_name = None

@functools.wraps(original_Fhkl_thread)
def patched_Fhkl_thread(*args, **kwargs):
    """
    Wrapper for Fhkl_thread that automatically discovers and sets
    the best cl_platform and gpu_name if they are not provided.
    """
    # Discover GPU info only on the first call
    if GpuInfo.platform_name is None:
        print("Monkey-patch: Discovering best GPU for Fhkl_thread...")
        GpuInfo.platform_name, GpuInfo.gpu_name = find_best_gpu()
        print(f"Monkey-patch: Defaulting to '{GpuInfo.gpu_name}' on platform '{GpuInfo.platform_name}'")

    # Set our defaults, but allow the user to override them
    kwargs.setdefault('language', 'OpenCL')
    kwargs.setdefault('cl_platform', GpuInfo.platform_name)
    kwargs.setdefault('gpu_name', GpuInfo.gpu_name)
    logging.info(GpuInfo.platform_name)
    logging.info(GpuInfo.gpu_name)
    
    # Call the original function with the (potentially modified) arguments
    return original_Fhkl_thread(*args, **kwargs)

# -- Step 3: Apply the patch --
fhkl_module.Fhkl_thread = patched_Fhkl_thread
Fhkl_thread = patched_Fhkl_thread
print("PyNX Fhkl_thread has been monkey-patched for automatic GPU selection.")

import cdiutils
from ewokscore import Task

# --- Setup and Helpers from Notebook & experiment.py ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

cdiutils.update_plot_params()

def capture_figures_to_bytes():
    """
    Captures all open matplotlib figures, converts them to PNG bytes,
    and closes them.
    Returns:
        list: A list of PNG byte strings.
    """
    figures_as_bytes = []
    fig_nums = plt.get_fignums()
    if fig_nums:
        logging.info(f"Capturing {len(fig_nums)} figure(s) as output data.")
    for num in fig_nums:
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        figures_as_bytes.append(buf.getvalue())
        plt.close(fig)
    return figures_as_bytes

def read_lammps_positions(
        file_path: str,
        separate_atoms: bool = False,
        centre: bool = True
) -> np.ndarray | dict:
    """
    Reads atom positions from a LAMMPS .xyz or .lmp (dump) file.

    Args:
        file_path (str): path to the input file.
        separate_atoms (bool, optional): if True, separates positions by
            atom type. If False, returns all positions in one array.
            Defaults to False.

    Returns:
        dict or np.ndarray:
            - If separate_atoms=True, returns a dictionary {atom_type:
                positions_array}.
            - If separate_atoms=False, returns a single array of all
            positions.

    Raises:
        ValueError: if the file type cannot be detected or the file
            cannot be parsed.
    """
    positions = []
    atom_types = []

    with open(file_path, 'r') as file:
        if file_path.endswith("xyz"):
            file.readline()
            file.readline()
            for line in file:
                split_line = line.strip().split()
                atom_type = split_line[0]
                position = list(map(float, split_line[1:4]))
                atom_types.append(atom_type)
                positions.append(position)
        elif file_path.endswith("lmp"):
            raise ValueError(
                "Reading LAMMPS dump files is not implemented yet. "
                "Please convert to .xyz format."
            )

    positions = np.array(positions)
    if centre:
        positions -= np.mean(positions, axis=0, keepdims=True)

    if separate_atoms:
        separated_positions = {atom_type: [] for atom_type in atom_types}
        for atom_type, position in zip(atom_types, positions):
            separated_positions[atom_type].append(position)
        for atom_type in separated_positions:
            separated_positions[atom_type] = np.array(
                separated_positions[atom_type]
            ).T
        return dict(separated_positions)
    else:
        return positions.T

def find_bragg_peak_centre(
    diffraction_pattern: np.ndarray,
    q_grid: tuple,
    search_radius: float = 0.1
) -> tuple:
    """
    Find the center of mass of the Bragg peak in reciprocal space.
    """
    total_intensity = np.sum(diffraction_pattern)
    q_com = []
    for q in q_grid:
        q_com.append(np.sum((diffraction_pattern * q)) / total_intensity)
    return tuple(q_com)

def refine_q_grid(
    positions: tuple,
    hkl: np.ndarray,
    lattice_parameter_guess: float,
    q_size: tuple = (200, 200, 200),
    step_nb: int = 400,
    max_iterations: int = 5,
    convert_to_angstrom: bool = True
):
    """
    Iteratively refine the q-grid to center the Bragg peak.
    """
    unit = "m"
    rcp_space_unit = "1/m"
    if convert_to_angstrom:
        lattice_parameter_guess *= 1e10
        unit = "angstrom"
        rcp_space_unit = "1/angstrom"

    current_lattice = lattice_parameter_guess
    current_center = hkl / current_lattice

    for iteration in range(max_iterations):
        logging.info(f"Refinement iteration {iteration + 1}")
        logging.info(f"Current lattice parameter: {current_lattice:.10f}")
        logging.info(f"Current center: {current_center}")

        dq = np.array([1, 1, 1]) / step_nb
        q_ranges = []
        for i in range(3):
            q_ranges.append(
                current_center[i] + (np.arange(q_size[i]) - (q_size[i] / 2)) * dq[i]
            )
        q_grid = np.meshgrid(*q_ranges, indexing="ij")

        scattered_amp, dt = Fhkl_thread(
            *q_grid, *positions, occ=None
            #*q_grid, *positions, occ=None, gpu_name="NVIDIA A40", language=""
        )
        diffraction_pattern = np.abs(scattered_amp) ** 2
        peak_center = find_bragg_peak_centre(diffraction_pattern, q_grid)
        logging.info(f"Found peak center at: {peak_center}")

        expected_center = [np.mean(q) for q in q_grid]
        shift = [pc - ec for pc, ec in zip(peak_center, expected_center)]
        logging.info(f"Shift from grid center: {shift} {rcp_space_unit}")
        current_center = peak_center

        q_magnitude = np.linalg.norm(peak_center)
        theoretical_q = np.linalg.norm(hkl) / current_lattice
        lattice_correction = theoretical_q / q_magnitude
        current_lattice *= lattice_correction
        current_center = [c / lattice_correction for c in current_center]

        logging.info(f"Refined lattice parameter: {current_lattice:.4f} {unit}")
        logging.info(f"Shift magnitude: {np.linalg.norm(shift):.6f} {rcp_space_unit}")
        
        if np.linalg.norm(shift) < dq[0] * 0.1:
            logging.info("Converged!")
            break

    return q_grid, scattered_amp, current_lattice

# --- Ewoks Tasks ---

class DisplayImages(Task, input_names=["image_data_list"], optional_input_names=[], output_names=[]):
    """An Ewoks task specifically for displaying image data in Jupyter."""
    def run(self):
        from IPython.display import Image, display
        image_data_list = self.inputs.image_data_list
        if not image_data_list:
            logging.warning("DisplayImages task received no images to display.")
            return
        logging.info(f"Displaying {len(image_data_list)} image(s) in Jupyter.")
        for image_data in image_data_list:
            if image_data:
                display(Image(data=image_data))


class ReadLammpsPositions(Task, input_names=["file_path", "centre"], output_names=["positions", "figures"]):
    def run(self):
        positions = read_lammps_positions(
            self.inputs.file_path,
            centre=self.inputs.centre
        )
        
        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, layout="tight")
        scatter_params = {
            "s": 1.5, "antialiased": False, "depthshade": True, "alpha": 0.2,
            "edgecolor": "k", "linewidth": 0.01, "c": "teal"
        }
        ax.scatter(*positions, **scatter_params)
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_zlabel("$z$");
        ax.set_xlim([-200, 200]); ax.set_ylim([-200, 200]); ax.set_zlim([-200, 200])

        self.outputs.positions = positions
        self.outputs.figures = capture_figures_to_bytes()


class InitialScattering(Task, 
    input_names=["positions", "hkl", "lattice_parameter", "q_size", "step_nb"], 
    output_names=["scattered_amp", "q_grid", "dt", "figures"]):
    def run(self):
        hkl = self.inputs.hkl
        positions = self.inputs.positions
        lattice_parameter = self.inputs.lattice_parameter
        q_size = self.inputs.q_size
        step_nb = self.inputs.step_nb
        
        original_vector = np.array(hkl) / np.linalg.norm(hkl)
        target_vector = np.array([0, 1, 0], dtype=float)
        target_vector /= np.linalg.norm(target_vector)
        rotation_axis = np.cross(original_vector, target_vector)
        rotation_axis /= np.linalg.norm(rotation_axis)
        angle = np.arccos(np.dot(original_vector, target_vector))
        rotation = Rotation.from_rotvec(angle * rotation_axis)
        rotated_hkl = rotation.apply(np.column_stack(hkl)).T

        dq = np.array([1, 1, 1]) / step_nb
        q_ranges = []
        for i in range(3):
            q_ranges.append(
                (rotated_hkl[i] + (np.arange(q_size[i]) - (q_size[i] / 2)) * dq[i]) / lattice_parameter
            )
        q_grid = np.meshgrid(*q_ranges, indexing="ij")

        scattered_amp, dt = Fhkl_thread(
            *q_grid, *positions, occ=None,# gpu_name="NVIDIA A40", language=""
        )
        logging.info("zzzzzzzzzzzzzzzzzzzzzzzzzzzz")
        if np.isnan(scattered_amp).any():
            logging.info(scattered_amp)
            st 
        elif (scattered_amp == 0).any():
            logging.info(scattered_amp)
            st
        diffraction_pattern = np.abs(scattered_amp)**2

        fig, axes = cdiutils.plot.plot_volume_slices(
            diffraction_pattern, norm="log", voxel_size=(dq[0], dq[1], dq[2]),
            data_centre=[np.mean(q) for q in q_grid], convention="xu", show=False
        )
        cdiutils.plot.add_labels(axes, space="rcp", convention="xu")
        
        self.outputs.scattered_amp = scattered_amp
        self.outputs.q_grid = q_grid
        self.outputs.dt = dt
        self.outputs.figures = capture_figures_to_bytes()
        

class RefineGrid(Task, 
    input_names=["positions", "hkl", "initial_lattice_guess", "q_size", "step_nb", "max_iterations"],
    output_names=["refined_q_grid", "scattered_amp", "refined_lattice", "figures"]):
    def run(self):
        hkl = self.inputs.hkl
        original_vector = np.array(hkl) / np.linalg.norm(hkl)
        target_vector = np.array([0, 1, 0], dtype=float)
        target_vector /= np.linalg.norm(target_vector)
        rotation_axis = np.cross(original_vector, target_vector)
        rotation_axis /= np.linalg.norm(rotation_axis)
        angle = np.arccos(np.dot(original_vector, target_vector)) 
        rotation = Rotation.from_rotvec(angle * rotation_axis)
        rotated_hkl = rotation.apply(np.column_stack(hkl))
        # extract rotated x, y, z arrays
        rotated_hkl = rotated_hkl.T
        rotated_hkl

        refined_q_grid, scattered_amp, refined_lattice = refine_q_grid(
            self.inputs.positions, rotated_hkl, self.inputs.initial_lattice_guess, 
            q_size=self.inputs.q_size, step_nb=self.inputs.step_nb, 
            max_iterations=self.inputs.max_iterations
        )
        
        dq = np.array([1, 1, 1]) / self.inputs.step_nb
        logging.info(dq)
        fig, axes = cdiutils.plot.plot_volume_slices(
            np.abs(scattered_amp) ** 2, norm="log", voxel_size=(dq[0], dq[1], dq[2]),
            data_centre=[np.mean(q) for q in refined_q_grid], convention="xu", show=False
        )
        cdiutils.plot.add_labels(axes, space="rcp", convention="xu")

        self.outputs.refined_q_grid = refined_q_grid
        self.outputs.scattered_amp = scattered_amp
        self.outputs.refined_lattice = refined_lattice
        self.outputs.figures = capture_figures_to_bytes()

class PostProcessScattering(Task, 
    input_names=["scattered_amp", "refined_q_grid", "hkl"],
    output_names=["struct_properties", "figures"]):
    def run(self):
        scattered_amp = self.inputs.scattered_amp
        refined_q_grid = self.inputs.refined_q_grid
        hkl = self.inputs.hkl

        voxel_size = tuple(1e-1 / np.ptp(q) for q in refined_q_grid)
        # TODO: issue?
        #obj = fftshift(fftn(ifftshift(scattered_amp)))
        obj = fftshift(fftn(ifftshift(scattered_amp)))
        support = cdiutils.utils.make_support(np.abs(obj), isosurface=0.75)
        
        plot_params = {
            "support": support, "convention": "xu", "voxel_size": voxel_size,
            "data_centre": (0, 0, 0), "show": True,
        }
        cdiutils.plot.plot_volume_slices(np.abs(obj), **plot_params)
        fig, axes = cdiutils.plot.plot_volume_slices(
            np.angle(obj), cmap="cet_CET_C9s_r", vmin=-np.pi, vmax=np.pi, **plot_params
        )
        cdiutils.plot.add_labels(axes, convention="xu")
        
        q_com = find_bragg_peak_centre(np.abs(scattered_amp)**2, refined_q_grid)
        obj = cdiutils.process.PostProcessor.apodize(obj)
        struct_properties = cdiutils.process.PostProcessor.get_structural_properties(
            obj, isosurface=0.65, g_vector=tuple(q * (2*np.pi) for q in q_com),
            hkl=hkl, voxel_size=voxel_size, phase_factor=1
        )
        
        for k in ("amplitude", "phase", "het_strain","het_strain_from_dspacing", "lattice_parameter"):
            # TODO
            plot_params.update(cdiutils.plot.get_plot_configs(k))
            #plot_params["vmin"], plot_params["vmax"] = None, None
            if "strain" in k:
                plot_params["cmap"] = "cet_CET_D13"
            cdiutils.plot.plot_volume_slices(struct_properties[k], **plot_params)
        
        cdiutils.pipeline.PipelinePlotter.summary_plot(
            title="Reconstruction Summary",  # 1. Add the missing title
            support=struct_properties["support"], # 2. Explicitly name the support
            voxel_size=voxel_size, 
            table_info=None,
            **{k: struct_properties[k] for k in (
                "amplitude", 
                "phase", 
                "displacement", 
                "het_strain", 
                "lattice_parameter"
            ) if k in struct_properties}
        )

        self.outputs.struct_properties = struct_properties
        self.outputs.figures = capture_figures_to_bytes()
