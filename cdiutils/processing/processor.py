import os
from typing import Union, Optional

# from matplotlib import font_manager
import h5py
import matplotlib.pyplot as plt
import numpy as np
import ruamel.yaml
from scipy.ndimage import center_of_mass
import silx.io.h5py_utils
import textwrap
import yaml

from cdiutils.utils import (
    center, crop_at_center, find_isosurface, zero_to_nan, find_max_pos,
    shape_for_safe_centered_cropping, make_support
)
from cdiutils.load.bliss import BlissLoader
from cdiutils.load.spec import SpecLoader
from cdiutils.converter import SpaceConverter
from cdiutils.geometry import Geometry
from cdiutils.processing.phase import (
    get_structural_properties, blackman_apodize, flip_reconstruction
)
from cdiutils.processing.plot import (
    preprocessing_detector_data_plot, summary_slice_plot
)
from cdiutils.plot.formatting import update_plot_params


def update_parameter_file(file_path: str, updated_parameters: dict) -> None:
    """
    Update a parameter file with the provided dictionary that contains
    the parameters (keys, values) to uptade.
    """
    with open(file_path, "r", encoding="utf8") as file:
        config, ind, bsi = ruamel.yaml.util.load_yaml_guess_indent(file)

    for key in config.keys():
        for updated_key, updated_value in updated_parameters.items():
            if updated_key in config[key]:
                config[key][updated_key] = updated_value
            else:
                for sub_key in config[key].keys():
                    if (
                            isinstance(config[key][sub_key], dict)
                            and updated_key in config[key][sub_key]
                    ):
                        config[key][sub_key][updated_key] = updated_value

            # elif (
            #     isinstance(config[key][updated_key], dict)):
            #     config[key][updated_key] = updated_value

    yaml_file = ruamel.yaml.YAML()
    yaml_file.indent(mapping=ind, sequence=ind, offset=bsi) 
    with open(file_path, "w", encoding="utf8") as file:
        yaml_file.dump(config, file)


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


class BcdiProcessor:
    """
    A class to handle pre and post processing in a bcdi data analysis workflow
    """
    def __init__(
            self,
            parameter_file_path: Optional[str]=None,
            parameters: Optional[dict]=None
    ) -> None:
        self.parameter_file_path = parameter_file_path
        self.parameters = parameters

        if parameters is None:
            if parameter_file_path is None:
                raise ValueError(
                    "parameter_file_path or parameters must be provided"
                )
            self.load_parameters(parameter_file_path)

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
        self.voxel_size = None
        self.structural_properties = {}
        self.averaged_dspacing = None
        self.averaged_lattice_parameter = None


        # initialize figures
        self.figures = {
            "preprocessing_fig": {
                "name": "centering_cropping_detector_data_plot"
            },
            "postprocessing_fig":{
                "name": "summary_slice_plot"
            },
            "strain_fig": {
                "name": "different_strain_methods"
            },
            "amplitude_fig":{
                "name": "amplitude_distribution_plot"
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
        # print(font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))
        # available_fonts = {
        #     os.path.splitext(os.path.basename(p))[0]: p
        #     for p in font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        # }
        update_plot_params(
            usetex=True,
            **{
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "figure.titlesize": 18,
            }
        )


    def load_parameters(self, path: str) -> None:
        with open(path, "r", encoding="utf8") as file:
            args = yaml.load(file, Loader=yaml.FullLoader)["cdiutils"]
            args["preprocessing_output_shape"] = tuple(
                args["preprocessing_output_shape"]
            )

        self.parameters = args

    def load_data(self) -> None:
        self.detector_data = self.loader.load_detector_data(
            scan=self.parameters["metadata"]["scan"],
        )
        (
            self.sample_outofplane_angle, self.sample_inplane_angle, 
            self.detector_inplane_angle, self.detector_outofplane_angle
        ) = self.loader.load_motor_positions(
            scan=self.parameters["metadata"]["scan"],
        )
        self.space_converter.set_q_space_area(
                self.sample_outofplane_angle,
                self.sample_inplane_angle, 
                self.detector_outofplane_angle,
                self.detector_inplane_angle
        )
        self.mask = self.loader.get_mask(channel=self.detector_data.shape[0])

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

        final_shape = self.parameters["preprocessing_output_shape"]
        initial_shape = self.detector_data.shape
        # if the final_shape is 2D convert it in 3D
        if len(final_shape) == 2:
            final_shape = (self.detector_data.shape[0], ) +  final_shape
        print(f"The preprocessing output shape is: {final_shape}")

        # Find the max and com voxel in the detector frame without
        # considering the method provided by the user
        # first determine the maximum value position in the det frame
        det_max_voxel = np.unravel_index(
            self.detector_data.argmax(), initial_shape)

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
                and isinstance(self.parameters["det_reference_voxel"], list)
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

        # save the voxel reference in the detector frame and Q_lab frame
        self.det_reference_voxel = det_reference_voxel
        self.q_lab_reference = self.space_converter.index_det_to_q_lab(
            det_reference_voxel)
        self.q_lab_max = self.space_converter.index_det_to_q_lab(
            det_max_voxel
        )
        self.q_lab_com = self.space_converter.index_det_to_q_lab(
            np.rint(det_com_voxel).astype(int)
        )

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
            f"{tuple([round(det_com_voxel[i], 2) for i in range(3)])} "
            f"(based on a {final_shape[1], final_shape[2]} max-centered "
            "bounding box)\n"
            "Com in the cropped detector frame at "
            f"{tuple([round(cropped_com_voxel[i], 2) for i in range(3)])}\n"
            f"The com corresponds to a d-spacing of {self.dspacing_com:.4f} A "
            f"and a lattice parameter of {self.lattice_parameter_com:.4f} A\n\n"
            f"The reference q_lab_reference corresponds "
            f"to a d-spacing of {self.dspacing_reference:.4f} A and a lattice "
            f"parameter of {self.lattice_parameter_reference:.4f} A\n"
        )

        # plot the detector data in the full detector frame and in the
        # final frame
        self.figures["preprocessing_fig"]["figure"] = (
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
        
        template_path = (
            self.parameters["metadata"]["dump_dir"]
            + "/cdiutils_S"
            + str(self.parameters["metadata"]["scan"])
            + "_"
        )

        self.figures["preprocessing_fig"]["figure"].savefig(
            (
                f"{template_path}"
                f"{self.figures['preprocessing_fig']['name']}.png"
            ),
            bbox_inches="tight",
            dpi=200
        )

        np.savez(
            template_path + "pynx_input_data.npz",
            data=self.cropped_detector_data
        )
        np.savez(
            template_path + "pynx_input_mask.npz", mask=self.mask)
        
        # save the values of the det_reference_voxel and the q_labs
        update_parameter_file(
            self.parameter_file_path,
            {
                "det_reference_voxel": self.parameters["det_reference_voxel"],
                "q_lab_reference": self.q_lab_reference,
                "q_lab_max": self.q_lab_max,
                "q_lab_com": self.q_lab_com
            }
        )

    def show_figures(self) -> None:
        """
        Show the figures that were plotted during the processing.
        """
        print("Running who-figure function")
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
                "Detector data are not loader, cannot proceed to "
                "orthogonalization"
            )

        reconstruction_file_path = (
            self.parameters["metadata"]["reconstruction_file"]
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
        support = make_support(
            reconstructed_amplitude,
            isosurface=find_isosurface(reconstructed_amplitude)
        )
        com = tuple(int(e) for e in center_of_mass(support))
        reconstructed_amplitude = center(reconstructed_amplitude, where=com)
        reconstructed_phase = center(np.angle(reconstructed_object), where=com)
        
        # find the maximum of the Bragg peak, this is will be taken as
        # the space origin for the orthogonalization
        max_pos = find_max_pos(self.detector_data)

        # find a safe shape that will enable centering and cropping the
        # q values without rolling them
        final_shape = shape_for_safe_centered_cropping(
            self.detector_data.shape,
            max_pos,
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
        self.space_converter.reference_voxel = max_pos
        self.space_converter.cropped_shape = final_shape

        cropped_data = center(self.detector_data, where=max_pos)

        # crop the detector data and the reconstructed data to the same
        # shape
        cropped_data = crop_at_center(cropped_data, final_shape)
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
        #  with xrayutilities conventions. We switch to cxi convention
        # afterwards). 
        self.verbose_print(
            "[INFO] Voxel size in the direct lab frame provided by user: "
            f"{self.parameters['voxel_size']} nm"
        )
        self.space_converter.init_interpolator(
            cropped_data,
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
        isosurface, self.figures["amplitude_fig"]["figure"] = find_isosurface(
            amplitude,
            nbins=100,
            sigma_criterion=2,
            plot=self.parameters["show"]
        )
        self.verbose_print("done.")
        self.verbose_print(
            f"[INFO] isosurface estimated to be {isosurface}")
        
        if self.parameters["isosurface"] is not None:
            self.verbose_print(
                "[INFO] Isosurface provided by user will be used: "
                f"{self.parameters['isosurface']}"
            )
            isosurface = self.parameters["isosurface"]

        # store the the averaged dspacing and lattice constant in variables
        # so they can be saved later in the output file
        self.verbose_print(
            "[INFO] The theoretical probed Bragg peak reflection is "
            f"{self.parameters['hkl']}"
        )

        self.structural_properties = get_structural_properties(
            data,
            isosurface,
            q_vector=self.q_lab_reference,
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

        self.figures["postprocessing_fig"]["figure"] = summary_slice_plot(
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

        self.figures["strain_fig"]["figure"] = summary_slice_plot(
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

        self.figures["postprocessing_fig"]["figure"].savefig(
            (
                f"{template_path}_"
                f"{self.figures['postprocessing_fig']['name']}.png"
            ),
            dpi=200,
            bbox_inches="tight"
        )

        self.figures["strain_fig"]["figure"].savefig(
            (
                f"{template_path}_"
                f"{self.figures['strain_fig']['name']}.png"
            ),
            dpi=200,
            bbox_inches="tight"
        )

        # save the amplitude distriubtion figure
        self.figures["amplitude_fig"]["figure"].savefig(
            f"{template_path}_{self.figures['amplitude_fig']['name']}.png",
            dpi=200,
            bbox_inches="tight"
        )
