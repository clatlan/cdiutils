import os
from typing import Union, Optional

# from matplotlib import font_manager
import h5py
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import center_of_mass
import silx.io.h5py_utils
import textwrap
import vtk
from vtk.util import numpy_support

from cdiutils.utils import (
    center, crop_at_center, find_isosurface, zero_to_nan, find_max_pos,
    shape_for_safe_centered_cropping, make_support, symmetric_pad,
)
from cdiutils.load.bliss import BlissLoader
from cdiutils.load.spec import SpecLoader
from cdiutils.converter import SpaceConverter
from cdiutils.geometry import Geometry
from cdiutils.processing.phase import (
    get_structural_properties, blackman_apodize, flip_reconstruction
)
from cdiutils.processing.plot import (
    preprocessing_detector_data_plot, summary_slice_plot,
    plot_direct_lab_orthogonalization_process,
    plot_q_lab_orthogonalization_process,
    plot_final_object_fft
)
from cdiutils.plot.formatting import update_plot_params


def loader_factory(metadata: dict) -> Union[BlissLoader, SpecLoader]:
    """
    Load the right loader based on the beamline_setup parameter
    in the metadata dictionary
    """
    if metadata["beamline_setup"] == "ID01BLISS":
        return BlissLoader(
            experiment_file_path=metadata["experiment_file_path"],
            detector_name=metadata["detector_name"],
            sample_name=metadata["sample_name"],
            flatfield=metadata["flatfield_path"]

        )
    elif metadata["beamline_setup"] == "ID01SPEC":
        return SpecLoader(
            experiment_file_path=metadata["experiment_file_path"],
            detector_data_path=metadata["detector_data_path"],
            edf_file_template=metadata["edf_file_template"],
            detector_name=metadata["detector_name"]
        )
    else:
        raise NotImplementedError(
            "The provided beamline_setup is not known yet"
        )

# TODO: Redundancy of attributes and parameters
class BcdiProcessor:
    """
    A class to handle pre and post processing in a bcdi data analysis workflow
    """
    def __init__(
            self,
            parameters: dict
    ) -> None:
        self.parameters = parameters

        self.loader = None
        self.space_converter = None

        self.detector_data = None
        self.sample_outofplane_angle = None
        self.sample_inplane_angle = None
        self.detector_outofplane_angle = None
        self.detector_inplane_angle = None

        self.cropped_detector_data = None
        self.mask = None

        self.det_reference_voxel = None
        self.q_lab_reference = None
        self.q_lab_max = None
        self.q_lab_com = None
        self.dspacing_reference = None
        self.dspacing_max = None
        self.dspacing_com = None
        self.lattice_parameter_reference = None
        self.lattice_parameter_max = None
        self.lattice_parameter_com = None

        self.orthgonolized_data = None
        self.orthogonalized_intensity = None
        self.voxel_size = None
        self.structural_properties = {}
        self.averaged_dspacing = None
        self.averaged_lattice_parameter = None


        # initialize figures
        self.figures = {
            "preprocessing": {
                "name": "centering_cropping_detector_data_plot",
                "debug": False
            },
            "postprocessing":{
                "name": "summary_slice_plot",
                "debug": False
            },
            "strain": {
                "name": "different_strain_methods",
                "debug": True
            },
            "amplitude":{
                "name": "amplitude_distribution_plot",
                "debug": False
            },
            "direct_lab_orthogonalization": {
                "name": "direct_lab_orthogonaliztion_plot",
                "debug": True
            },
            "q_lab_orthogonalization": {
                "name": "q_lab_orthogonaliztion_plot",
                "debug": True
            },
            "final_object_fft": {
                "name": "final_object_fft",
                "debug": True
            }
        }
        for value in self.figures.values():
            value["figure"] = None

        # initialise the loader, the space converter and the plot parameters
        self._init_loader()
        self._init_space_converter()
        self._init_plot_parameters()

    def _init_loader(self) -> None:
        self.loader = loader_factory(self.parameters["metadata"])

    def _init_space_converter(self) -> None:
        self.space_converter = SpaceConverter(
            energy=self.parameters["energy"],
            roi=self.parameters["roi"],
            geometry=Geometry.from_name(
                self.parameters["metadata"]["beamline_setup"])
        )
        self.space_converter.init_q_space_area(
            self.parameters["det_calib_parameters"]
        )

    def _init_plot_parameters(self):
        update_plot_params(
            usetex=self.parameters["usetex"],
            **{
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "figure.titlesize": 18,
            }
        )

    def load_data(self) -> None:
        self.detector_data = self.loader.load_detector_data(
            scan=self.parameters["metadata"]["scan"],
            binning_along_axis0=self.parameters["binning_along_axis0"]
        )
        (
            self.sample_outofplane_angle, self.sample_inplane_angle, 
            self.detector_inplane_angle, self.detector_outofplane_angle
        ) = self.loader.load_motor_positions(
            scan=self.parameters["metadata"]["scan"],
            binning_along_axis0=self.parameters["binning_along_axis0"]
        )
        self.space_converter.set_q_space_area(
                self.sample_outofplane_angle,
                self.sample_inplane_angle,
                self.detector_outofplane_angle,
                self.detector_inplane_angle
        )

        self.mask = self.loader.get_mask(
            channel=self.detector_data.shape[0],
            detector_name=self.parameters["metadata"]["detector_name"]
        )

    def verbose_print(self, text: str, **kwargs) -> None:
        if self.parameters["verbose"]:
            wrapper = textwrap.TextWrapper(
                width=80,
                break_long_words=True,
                replace_whitespace=False
            )
            text = "\n".join(wrapper.wrap(text))
            print(text,  **kwargs)


    def _compute_dspacing_lattice(self) -> None:
        hkl = self.parameters["hkl"]
        qnorm_reference = np.linalg.norm(self.q_lab_reference)
        self.dspacing_reference = 2*np.pi/qnorm_reference
        self.lattice_parameter_reference = (
            np.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2)
            * self.dspacing_reference
        )
        qnorm_max = np.linalg.norm(self.q_lab_max)
        self.dspacing_max = 2*np.pi/qnorm_max
        self.lattice_parameter_max = (
            np.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2)
            * self.dspacing_max
        )
        qnorm_com = np.linalg.norm(self.q_lab_com)
        self.dspacing_com = 2*np.pi/qnorm_com
        self.lattice_parameter_com = (
            np.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2)
            * self.dspacing_com
        )

    def center_crop_data(self) -> None:
        final_shape = tuple(self.parameters["preprocessing_output_shape"])
        initial_shape = self.detector_data.shape
        # if the final_shape is 2D convert it in 3D
        if len(final_shape) == 2:
            final_shape = (self.detector_data.shape[0], ) + final_shape
        print(f"The preprocessing output shape is: {final_shape}")

        # Find the max and com voxel in the detector frame without
        # considering the method provided by the user
        # first determine the maximum value position in the det frame
        det_max_voxel = find_max_pos(self.detector_data)

        # find the com nearby the max using the given
        # preprocessing_output_shape
        max_centered_data, former_center = center(
                self.detector_data,
                where=det_max_voxel,
                return_former_center=True
        )

        max_cropped_data = crop_at_center(
            max_centered_data,
            final_shape=final_shape
        )
        cropped_com_voxel = center_of_mass(max_cropped_data)
        # convert the com coordinates in the detector frame coordinates
        det_com_voxel = (
            cropped_com_voxel
            + (np.array(initial_shape) - np.array(final_shape))//2
            - (np.array(initial_shape)//2 - former_center)
        )

        # determine the detector voxel reference according to the
        # provided method
        if (
                self.parameters["det_reference_voxel_method"] is None
                and isinstance(
                    self.parameters["det_reference_voxel"], (list, tuple)
                )
        ):
            det_reference_voxel = [
                int(e) for e in self.parameters["det_reference_voxel"]
            ]
            self.verbose_print(
                "Reference voxel provided by user: "
                f"{self.parameters['det_reference_voxel']}"
            )
        if self.parameters["det_reference_voxel_method"] == "max":
            det_reference_voxel = det_max_voxel
            self.verbose_print(
                "Method employed for the reference voxel determination is max"
            )
        elif self.parameters["det_reference_voxel_method"] == "com":
            det_reference_voxel = np.rint(det_com_voxel).astype(int)
            self.verbose_print(
                "Method employed for the reference voxel determination is com"
            )

        # convert numpy.int64 to int to make them serializable and
        # store the det_reference_voxel in the parameters which will be
        # saved later
        self.parameters["det_reference_voxel"] = [
            int(e) for e in det_reference_voxel
        ]

        # now proceed to the centering and cropping using
        # the det_reference_voxel as a reference
        centered_data = center(self.detector_data, where=det_reference_voxel)
        self.verbose_print(f"Shape before checking: {final_shape}")
        final_shape = shape_for_safe_centered_cropping(
            initial_shape, det_reference_voxel, final_shape
        )
        self.verbose_print(f"Shape after checking: {final_shape}")
        self.cropped_detector_data = crop_at_center(
            centered_data,
            final_shape=final_shape
        )

        # center and crop the mask
        self.mask = center(self.mask, where=det_reference_voxel)
        self.mask = crop_at_center(self.mask, final_shape=final_shape)

        # redefine the max and com voxel coordinates in the new cropped data
        # frame
        cropped_max_voxel = (
            det_max_voxel
            - (det_reference_voxel - np.array(initial_shape)//2)
            - (np.array(initial_shape)- final_shape)//2
        )
        cropped_com_voxel = (
            det_com_voxel
            - (det_reference_voxel - np.array(initial_shape)//2)
            - (np.array(initial_shape)- final_shape)//2
        )

        # save the voxel reference in the detector frame and q_lab frame
        self.det_reference_voxel = det_reference_voxel
        self.q_lab_reference = self.space_converter.index_det_to_q_lab(
            det_reference_voxel)
        self.q_lab_max = self.space_converter.index_det_to_q_lab(
            det_max_voxel
        )
        try:
            self.q_lab_com = self.space_converter.index_det_to_q_lab(
                np.rint(det_com_voxel).astype(int)
            )
        except IndexError:
            self.verbose_print(
                "COM has been found out of the detector frame, will be set to max"
            )
            self.q_lab_com = self.q_lab_max

        self._compute_dspacing_lattice()

        # print some info
        self.verbose_print(
            "[INFO]\n"
            f"\nMax in the full detector frame at {det_max_voxel}\n"
            "Max in the cropped detector frame at "
            f"{tuple(cropped_max_voxel)}\n"
            f"The max corresponds to a d-spacing of {self.dspacing_max:.4f} A "
            f"and a lattice parameter of {self.lattice_parameter_max:.4f} A\n\n"
            f"Com in the full detector frame at "
            f"{tuple(round(det_com_voxel[i], 2) for i in range(3))} "
            f"(based on a {final_shape[1], final_shape[2]} max-centered "
            "bounding box)\n"
            "Com in the cropped detector frame at "
            f"{tuple(round(cropped_com_voxel[i], 2) for i in range(3))}\n"
            f"The com corresponds to a d-spacing of {self.dspacing_com:.4f} A "
            f"and a lattice parameter of {self.lattice_parameter_com:.4f} A\n\n"
            f"The reference q_lab_reference corresponds "
            f"to a d-spacing of {self.dspacing_reference:.4f} A and a lattice "
            f"parameter of {self.lattice_parameter_reference:.4f} A\n"
        )
         # save the values ofthe q_labs in the parameters dict
        self.parameters.update(
            {
                "q_lab_reference": self.q_lab_reference,
                "q_lab_max": self.q_lab_max,
                "q_lab_com": self.q_lab_com
            }
        )

        # plot the detector data in the full detector frame and in the
        # final frame
        self.figures["preprocessing"]["figure"] = (
            preprocessing_detector_data_plot(
                detector_data=self.detector_data,
                cropped_data=self.cropped_detector_data,
                det_reference_voxel=det_reference_voxel,
                det_max_voxel=det_max_voxel,
                det_com_voxel=det_com_voxel,
                cropped_max_voxel=cropped_max_voxel,
                cropped_com_voxel=cropped_com_voxel,
                title=(
                    "Detector data preprocessing, "
                    f"S{self.parameters['metadata']['scan']}"
                )
            )
        )

    def save_preprocessed_data(self):
        if os.path.isdir(self.parameters["metadata"]["dump_dir"]):
            self.verbose_print(
                "\n[INFO] Dump directory already exists, results will be saved"
                f" in: {self.parameters['metadata']['dump_dir']}"
            )
        else:
            self.verbose_print(
                "[INFO] Creating the dump directory at: "
                f"{self.parameters['metadata']['dump_dir']}")
            os.mkdir(self.parameters['metadata']["dump_dir"])

        self.figures["preprocessing"]["figure"].savefig(
            (
                f"{self.parameters['metadata']['dump_dir']}/cdiutils_S"
                f"{self.parameters['metadata']['scan']}_"
                f"{self.figures['preprocessing']['name']}.png"
            ),
            bbox_inches="tight",
            dpi=200
        )

        np.savez(
            f"{self.parameters['metadata']['dump_dir']}/S"
            f"{self.parameters['metadata']['scan']}_pynx_input_data.npz",
            data=self.cropped_detector_data
        )
        np.savez(
            f"{self.parameters['metadata']['dump_dir']}/S"
            f"{self.parameters['metadata']['scan']}_pynx_input_mask.npz",
            mask=self.mask
        )

    def show_figures(self) -> None:
        """
        Show the figures that were plotted during the processing.
        """
        if any(value["figure"] for value in self.figures.values()):
            plt.show()

    def reload_preprocessing_parameters(self):
        self.q_lab_reference = self.parameters["q_lab_reference"]
        self.q_lab_max = self.parameters["q_lab_max"]
        self.q_lab_com = self.parameters["q_lab_com"]

        self._compute_dspacing_lattice()

    def orthogonalize(self):
        """
        Orthogonalize detector data to the lab frame.
        """

        if self.detector_data is None:
            raise ValueError(
                "Detector data are not loaded, cannot proceed to "
                "orthogonalization"
            )

        reconstruction_file_path = (
            self.parameters["metadata"]["reconstruction_file"]
        )
        if not os.path.isabs(reconstruction_file_path):
            reconstruction_file_path = (
                self.parameters["metadata"]["dump_dir"]
                + reconstruction_file_path
            )
        if not os.path.isfile(reconstruction_file_path):
            raise FileNotFoundError(
                f"File was not found at: {reconstruction_file_path}"
            )
        reconstructed_object = self._load_reconstruction_file(
            reconstruction_file_path
        )
        # check if the reconstructed object is correctly centered
        reconstructed_amplitude = np.abs(reconstructed_object)

        # Need to hard code the isosurface if it is < 0 or > 1
        isosurface = find_isosurface(reconstructed_amplitude)
        if isosurface < 0 or isosurface > 1:
            isosurface = 0.3
        support = make_support(
            reconstructed_amplitude,
            isosurface=isosurface
        )
        com = tuple(e for e in center_of_mass(support))
        reconstructed_amplitude = center(reconstructed_amplitude, where=com)
        reconstructed_phase = center(np.angle(reconstructed_object), where=com)

        support, c = center(support, where=com, return_former_center=True)

        # find the maximum of the Bragg peak, this is will be taken as
        # the space origin for the orthogonalization
        det_reference_voxel = find_max_pos(self.detector_data)
        det_reference_voxel = self.parameters["det_reference_voxel"]

        # find a safe shape that will enable centering and cropping the
        # q values without rolling them
        final_shape = shape_for_safe_centered_cropping(
            self.detector_data.shape,
            det_reference_voxel,
            reconstructed_object.shape
        )
        self.verbose_print(
            "[INFO] The shape of the reconstructed object is: "
            f"{reconstructed_object.shape}\n"
            "The shape for a safe centered cropping is: "
            f"{final_shape}"
        )
        # set the reference voxel and the cropped shape in the space
        # converter for further processing in the q lab space
        self.space_converter.reference_voxel = det_reference_voxel
        self.space_converter.cropped_shape = final_shape

        cropped_data = center(self.detector_data, where=det_reference_voxel)

        # crop the detector data and the reconstructed data to the same
        # shape
        self.cropped_detector_data = crop_at_center(cropped_data, final_shape)
        reconstructed_amplitude = crop_at_center(
            reconstructed_amplitude,
            final_shape
        )
        reconstructed_phase = crop_at_center(
            reconstructed_phase,
            final_shape
        )

        # initialize the interpolator in reciprocal and direct spaces
        # (note that the orthogonalization is done in the lab frame
        # with xrayutilities conventions. We switch to cxi convention
        # afterwards).
        self.verbose_print(
            "[INFO] Voxel size in the direct lab frame provided by user: "
            f"{self.parameters['voxel_size']} nm"
        )
        self.space_converter.init_interpolator(
            self.cropped_detector_data,
            final_shape,
            space="both",
            direct_space_voxel_size=self.parameters["voxel_size"]
        )

        orthogonalized_amplitude = (
            self.space_converter.orthogonalize_to_direct_lab(
                reconstructed_amplitude,
            )
        )
        orthogonalized_phase = (
            self.space_converter.orthogonalize_to_direct_lab(
                reconstructed_phase,
            )
        )

        self.orthgonolized_data = self.space_converter.lab_to_cxi_conventions(
            orthogonalized_amplitude
            * np.exp(1j * orthogonalized_phase)
        )

        self.voxel_size = self.space_converter.lab_to_cxi_conventions(
            self.space_converter.direct_lab_interpolator.target_voxel_size
        )
        self.verbose_print(
            f"[INFO] Voxel size finally used is: {self.voxel_size} nm "
            "in the CXI convention"
        )

        if self.parameters["debug"]:
            self.figures["direct_lab_orthogonalization"]["figure"] = (
                plot_direct_lab_orthogonalization_process(
                    reconstructed_amplitude,
                    orthogonalized_amplitude,
                    self.space_converter.get_direct_lab_regular_grid(),
                    title=(
                    r"From \textbf{detector frame} to "
                    r"\textbf{direct lab frame}, "
                    f"S{self.parameters['metadata']['scan']}"
                    )
                )
            )

            self.orthogonalized_intensity = (
                self.space_converter.orthogonalize_to_q_lab(
                    self.cropped_detector_data
                )
            )

            where_in_det_space = find_max_pos(self.cropped_detector_data)
            where_in_ortho_space = (
                self.space_converter.index_cropped_det_to_index_of_q_lab(
                    where_in_det_space
                )
            )
            q_lab_regular_grid = self.space_converter.get_q_lab_regular_grid()

            self.figures["q_lab_orthogonalization"]["figure"] = (
                plot_q_lab_orthogonalization_process(
                    self.cropped_detector_data,
                    self.orthogonalized_intensity,
                    q_lab_regular_grid,
                    where_in_det_space,
                    where_in_ortho_space,
                    title=(
                        r"From \textbf{detector frame} to \textbf{q lab frame}"
                        f", S{self.parameters['metadata']['scan']}"
                    )
                )
            )

    def _load_reconstruction_file(self, file_path: str) -> np.ndarray:
        with silx.io.h5py_utils.File(file_path) as h5file:
            reconstructed_object = h5file["entry_1/data_1/data"][0]
        return reconstructed_object


    def load_orthogonolized_data(self, file_path: str) -> None:
        if file_path.endswith(".npz"):
            with np.load(file_path) as file:
                self.orthgonolized_data = file["data"]
                self.voxel_size = file["voxel_sizes"]
        else:
            raise NotImplementedError("Please provide a .npz file")

    def postprocess(self) -> None:
        if self.parameters["flip"]:
            data = flip_reconstruction(self.orthgonolized_data)
        else:
            data = self.orthgonolized_data

        if self.parameters["apodize"]:
            self.verbose_print(
                "[PROCESSING] Apodizing the complex array: ",
                end=""
            )
            data = blackman_apodize(data, initial_shape=data.shape)
            self.verbose_print("done.")
        amplitude = np.abs(data)

        # first compute the histogram of the amplitude to get an
        # isosurface estimate
        self.verbose_print(
            "[PROCESSING] Finding an isosurface estimate based on the "
            "reconstructed Bragg electron density histogram: ",
            end=""
        )
        isosurface, self.figures["amplitude"]["figure"] = find_isosurface(
            amplitude,
            nbins=100,
            sigma_criterion=3,
            plot=True # plot in any case
        )
        self.verbose_print("done.")
        self.verbose_print(
            f"[INFO] isosurface estimated at {isosurface}")

        if self.parameters["isosurface"] is not None:
            self.verbose_print(
                "[INFO] Isosurface provided by user will be used: "
                f"{self.parameters['isosurface']}"
            )
            isosurface = self.parameters["isosurface"]
        
        elif isosurface < 0 or isosurface > 1:
            isosurface = 0.3
            self.verbose_print(
                "[INFO] isosurface < 0 or > 1 is set to 0.3")


        # store the the averaged dspacing and lattice constant in variables
        # so they can be saved later in the output file
        self.verbose_print(
            "[INFO] The theoretical probed Bragg peak reflection is "
            f"{self.parameters['hkl']}"
        )

        self.structural_properties = get_structural_properties(
            data,
            isosurface,
            q_vector=SpaceConverter.lab_to_cxi_conventions(
                self.q_lab_reference),
            hkl=self.parameters["hkl"],
            voxel_size=self.voxel_size,
            phase_factor=-1 # it came out of pynx, phase must be -phase
        )

        self.averaged_dspacing = np.nanmean(
            self.structural_properties["dspacing"]
        )
        self.averaged_lattice_parameter = np.nanmean(
            self.structural_properties["lattice_parameter"]
        )

        # plot the results
        final_plots = {
            k: self.structural_properties[k]
            for k in ["amplitude", "phase", "displacement",
                      "local_strain", "lattice_parameter"]
        }

        self.figures["postprocessing"]["figure"] = summary_slice_plot(
            title=f"Summary figure, S{self.parameters['metadata']['scan']}",
            support=zero_to_nan(self.structural_properties["support"]),
            dpi=200,
            voxel_size=self.voxel_size,
            isosurface=isosurface,
            det_reference_voxel=self.parameters["det_reference_voxel"],
            averaged_dspacing=self.averaged_dspacing,
            averaged_lattice_parameter=self.averaged_lattice_parameter,
            **final_plots
        )

        strain_plots = {
            k: self.structural_properties[k]
            for k in ["local_strain", "local_strain_from_dspacing",
                      "local_strain_from_dspacing", "numpy_local_strain",
                      "local_strain_with_ramp"]
        }

        self.figures["strain"]["figure"] = summary_slice_plot(
            title=f"Strain check figure, S{self.parameters['metadata']['scan']}",
            support=zero_to_nan(self.structural_properties["support"]),
            dpi=200,
            voxel_size=self.voxel_size,
            isosurface=isosurface,
            det_reference_voxel=self.parameters["det_reference_voxel"],
            averaged_dspacing=self.averaged_dspacing,
            averaged_lattice_parameter=self.averaged_lattice_parameter,
            single_vmin=-self.structural_properties["local_strain"].ptp()/2,
            single_vmax=self.structural_properties["local_strain"].ptp()/2,
            **strain_plots
        )

        if self.parameters["debug"]:
            shape = self.orthogonalized_intensity.shape
            # convert to lab conventions and pad the data
            final_object_fft = symmetric_pad(
                self.space_converter.cxi_to_lab_conventions(
                    self.structural_properties["amplitude"]
                    * np.exp(1j*self.structural_properties["phase"])
                ),
                final_shape=shape
            )
            final_object_fft = np.abs(
                np.fft.ifftshift(
                    np.fft.fftn(
                        np.fft.fftshift(final_object_fft)
                    )
                )
            )

            extension = np.multiply(self.voxel_size, shape)
            voxel_size_of_fft_object = (2*np.pi / (10 * extension))

            final_object_q_lab_grid = (
                np.arange(-shape[0]//2, shape[0]//2, 1) * voxel_size_of_fft_object[0],
                np.arange(-shape[1]//2, shape[1]//2, 1) * voxel_size_of_fft_object[1],
                np.arange(-shape[2]//2, shape[2]//2, 1) * voxel_size_of_fft_object[2],
            )            

            where_in_det_space = find_max_pos(self.cropped_detector_data)
            where_in_ortho_space = (
                self.space_converter.index_cropped_det_to_index_of_q_lab(
                    where_in_det_space
                )
            )
            exp_data_q_lab_grid = self.space_converter.get_q_lab_regular_grid()
            self.figures["final_object_fft"]["figure"] = plot_final_object_fft(
                final_object_fft,
                self.orthogonalized_intensity,
                final_object_q_lab_grid,
                exp_data_q_lab_grid,
                where_in_ortho_space=where_in_ortho_space,
                title=(
                    r"FFT of final object \textit{vs.} experimental data"
                    f", S{self.parameters['metadata']['scan']}"
                )
            )

    def save_postprocessed_data(self) -> None:

        # save the results in a npz file
        template_path = (
            f"{self.parameters['metadata']['dump_dir']}/"
            f"cdiutils_S{self.parameters['metadata']['scan']}"
        )
        self.verbose_print(
            "[INFO] Saving the results to the following path:\n"
            f"{template_path}_structural_properties.npz"
        )

        np.savez(
            f"{template_path}_structural_properties.npz",
            q_lab_reference=self.q_lab_reference,
            q_lab_max=self.q_lab_max,
            q_lab_com=self.q_lab_com,
            q_cxi_reference=SpaceConverter.lab_to_cxi_conventions(
                self.q_lab_reference
            ),
            q_cxi_max=SpaceConverter.lab_to_cxi_conventions(
                self.q_lab_max
            ),
            q_cxi_com=SpaceConverter.lab_to_cxi_conventions(
                self.q_lab_com
            ),
            dspacing_reference=self.dspacing_reference,
            dspacing_max=self.dspacing_max,
            dspacing_com=self.dspacing_com,
            lattice_parameter_reference=self.lattice_parameter_reference,
            lattice_parameter_max=self.lattice_parameter_max,
            lattice_parameter_com=self.lattice_parameter_com,
            averaged_dspacing=self.averaged_dspacing,
            averaged_lattice_parameter=self.averaged_lattice_parameter,
            **self.structural_properties,
        )

        if self.parameters["debug"]:
            np.savez(
                f"{template_path}_orthogonalized_intensity.npz",
                q_xlab=self.space_converter.get_q_lab_regular_grid()[0],
                q_ylab=self.space_converter.get_q_lab_regular_grid()[1],
                q_zlab=self.space_converter.get_q_lab_regular_grid()[2],
                orthogonalized_intensity=self.orthogonalized_intensity
            )

        to_save_as_vti = {
            k: self.structural_properties[k]
            for k in ["amplitude", "support", "phase", "displacement",
                      "local_strain", "local_strain_from_dspacing",
                      "lattice_parameter", "numpy_local_strain", "dspacing"]
        }
        # save to vti file
        self.save_to_vti(
            f"{template_path}_structural_properties.vti",
            voxel_size=self.voxel_size,
            cxi_convention=True,
            **to_save_as_vti
        )

        with h5py.File(f"{template_path}_structural_properties.h5", "w") as hf:
            volumes = hf.create_group("volumes")
            scalars = hf.create_group("scalars")

            volumes.create_dataset(
                "amplitude",
                data=self.structural_properties["amplitude"]
            )
            volumes.create_dataset(
                "support",
                data=self.structural_properties["support"]
            )
            volumes.create_dataset(
                "phase",
                data=self.structural_properties["phase"]
            )
            volumes.create_dataset(
                "displacement",
                data=self.structural_properties["displacement"]
            )
            volumes.create_dataset(
                "local_strain",
                data=self.structural_properties["local_strain"]
            )
            volumes.create_dataset(
                "numpy_local_strain",
                data=self.structural_properties["numpy_local_strain"]
            )
            volumes.create_dataset(
                "dspacing",
                data=self.structural_properties["dspacing"]
            )
            volumes.create_dataset(
                "lattice_parameter",
                data=self.structural_properties["lattice_parameter"]
            )
            scalars.create_dataset(
                "q_lab_reference",
                data=self.q_lab_reference
            )
            scalars.create_dataset(
                "q_lab_max",
                data=self.q_lab_max
            )
            scalars.create_dataset(
                "q_lab_com",
                data=self.q_lab_com
            )
            scalars.create_dataset(
                "q_cxi_reference",
                data=SpaceConverter.lab_to_cxi_conventions(
                    self.q_lab_reference
                )
            )
            scalars.create_dataset(
                "q_cxi_max",
                data=SpaceConverter.lab_to_cxi_conventions(
                    self.q_lab_max
                )
            )
            scalars.create_dataset(
                "q_cxi_com",
                data=SpaceConverter.lab_to_cxi_conventions(
                    self.q_lab_com
                )
            )
            scalars.create_dataset(
                "dspacing_reference",
                data=self.dspacing_reference
            )
            scalars.create_dataset(
                "dspacing_max",
                data=self.dspacing_max
            )
            scalars.create_dataset(
                "dspacing_com",
                data=self.dspacing_com
            )
            scalars.create_dataset(
                "lattice_parameter_reference",
                data=self.lattice_parameter_reference
            )
            scalars.create_dataset(
                "lattice_parameter_max",
                data=self.lattice_parameter_max
            )
            scalars.create_dataset(
                "lattice_parameter_com",
                data=self.lattice_parameter_com
            )
            scalars.create_dataset(
                "averaged_dspacing",
                data=self.averaged_dspacing
            )
            scalars.create_dataset(
                "averaged_lattice_parameter",
                data=self.averaged_lattice_parameter
            )
            scalars.create_dataset(
                "voxel_size",
                data=self.voxel_size
            )
            scalars.create_dataset(
                "hkl",
                data=self.parameters["hkl"]
            )
        if self.parameters["debug"]:
            os.makedirs(
                f"{self.parameters['metadata']['dump_dir']}/debug",
                exist_ok=True
            )

        for fig in self.figures.values():
            if fig["figure"] is not None:
                fig_path = (
                    f"{self.parameters['metadata']['dump_dir']}/"
                    f"{'/debug/' if fig['debug'] else ''}"
                    f"cdiutils_S{self.parameters['metadata']['scan']}_"
                    f"{fig['name']}.png"
                )
                fig["figure"].savefig(
                    fig_path,
                    dpi=200,
                    bbox_inches="tight"
                )

    @staticmethod
    def save_to_vti(
            output_path: str,
            voxel_size: Union[tuple, list, np.ndarray],
            cxi_convention: Optional[bool]=False,
            origin: Optional[tuple]=(0, 0, 0),
            **np_arrays: dict[np.ndarray]
    ) -> None :
        """
        Save numpy arrays to .vti file.
        """
        voxel_size = tuple(voxel_size)
        nb_arrays = len(np_arrays)

        # if cxi_convention:
        #     voxel_size = (voxel_size[2], voxel_size[1], voxel_size[0])

        if not nb_arrays:
            raise ValueError(
                "np_arrays is empty, please provide a dictionary of "
                "(fieldnames: np.ndarray) you want to save."
            )
        is_init = False
        for i, (key, array) in enumerate(np_arrays.items()):
            if not is_init:
                shape = array.shape
                image_data = vtk.vtkImageData()
                image_data.SetOrigin(origin)
                image_data.SetSpacing(voxel_size)
                image_data.SetExtent(
                    0, shape[0] - 1,
                    0, shape[1] - 1,
                    0, shape[2] - 1
                )
                point_data = image_data.GetPointData()
                is_init = True

            if cxi_convention:
                array = np.flip(array, 2).T
                # array = np.swapaxes(array, axis1=0, axis2=2)

            vtk_array = numpy_support.numpy_to_vtk(array.ravel())
            point_data.AddArray(vtk_array)
            point_data.GetArray(i).SetName(key)
            point_data.Update()

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(output_path)
        writer.SetInputData(image_data)
        writer.Write()


