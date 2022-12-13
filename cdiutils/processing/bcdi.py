from typing import Union
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import ruamel.yaml
from scipy.ndimage import center_of_mass
import yaml

from cdiutils.utils import center, crop_at_center, find_isosurface, zero_to_nan
from cdiutils.load.bliss import BlissLoader
from cdiutils.converter import SpaceConverter
from cdiutils.processing.phase import (
    get_structural_properties, blackman_apodize, flip_reconstruction
)
from cdiutils.plot.slice import summary_slice_plot


def update_parameter_file(file_path: str, updated_parameters: dict) -> None:
    config, ind, bsi = ruamel.yaml.util.load_yaml_guess_indent(
        open(file_path))
    for key in config.keys():
        for sub_key, v in updated_parameters.items():
            if sub_key in config[key]:
                config[key][sub_key] = v
    yaml_file = ruamel.yaml.YAML()
    yaml_file.indent(mapping=ind, sequence=ind, offset=bsi) 
    with open(file_path, "w") as f:
        yaml_file.dump(config, f)


def loader_factory(beamline_name: str):
    if beamline_name == "ID01BLISS":
        return BlissLoader
    else:
        raise NotImplementedError(
            "The provided beamline name is not known yet"
        )


class BcdiProcessingHandler:
    """
    A class to handle pre and post processing in a bcdi data analysis workflow
    """
    def __init__(self, parameter_file_path: str):
        self.parameter_file_path = parameter_file_path
        self.parameters = None

        self.loader = None
        self.space_converter = None

        self.detector_data = None
        self.sample_outofplane_angle = None
        self.sample_inplane_angle = None
        self.detector_outofplane_angle = None
        self.detector_inplane_angle = None
        self.cropped_detector_data = None
        self.mask = None

        self.det_pixel_reference = None 
        self.q_lab_reference = None
        self.q_lab_max = None
        self.q_lab_com = None

        self.orthgonolized_data = None
        self.voxel_size = None
        self.structural_properties = {}
        self.averaged_dspacing = None
        self.averaged_lattice_constant = None

        self.preprocessing_figure = None
        self.postprocessing_figure = None
        self.amplitude_distribution_figure = None

        self.load_parameters(parameter_file_path)
        # initialise the loader and the space converter
        self._init_loader()
        self._init_space_converter()
        self._init_plot_parameters()


    
    def _init_loader(self) -> None:
        loader_class = loader_factory(self.parameters["beamline"])
        self.loader = loader_class(
            self.parameters["experiment_file_path"],
            flatfield=self.parameters["flatfield_path"]
        )

    def _init_space_converter(self) -> None:
        self.space_converter = SpaceConverter(
            energy=self.parameters["energy"],
            roi=self.parameters["roi"]
        )
        self.space_converter.init_Q_space_area(
            self.parameters["det_calib_parameters"]
        )
    
    def _init_plot_parameters(self):
        matplotlib.pyplot.rcParams.update({
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 18
        })


    def load_parameters(self, path: str) -> None:
        with open(path, "r") as file:
            args = yaml.load(file, Loader=yaml.FullLoader)["cdiutils"]
            args["preprocessing_output_shape"] = tuple(
                args["preprocessing_output_shape"]
            )

        self.parameters = args

    def load_data(self) -> None:
        self.detector_data = self.loader.load_detector_data(
            scan=self.parameters["scan"],
            sample_name=self.parameters["sample_name"]
        )
        self.initial_shape = self.detector_data.shape
        (
            self.sample_outofplane_angle, self.sample_inplane_angle, 
            self.detector_inplane_angle, self.detector_outofplane_angle
        ) = self.loader.load_motor_positions(
            scan=self.parameters["scan"],
            sample_name=self.parameters["sample_name"]
        )
        self.space_converter.set_Q_space_area(
                self.sample_outofplane_angle,
                self.sample_inplane_angle, 
                self.detector_outofplane_angle,
                self.detector_inplane_angle
        )
        self.mask = self.loader.get_mask(channel=self.detector_data.shape[0])
    
    def verbose_print(self, text: str, **kwargs) -> None:
        if self.parameters["verbose"]:
            print(text, **kwargs)         
    
    
    def center_crop_data(self) -> None:

        final_shape = self.parameters["preprocessing_output_shape"]
        initial_shape = self.detector_data.shape
        # if the final_shape is 2D convert it in 3D
        if len(final_shape) == 2:
            final_shape = (self.detector_data.shape[0], ) +  final_shape
        print(f"The preprocessing output shape is: {final_shape}")

        # Find the max and com pixel in the detector frame without
        # considering the method provided by the user
        # first determine the maximum value position in the det frame
        det_max_pixel = np.unravel_index(
            self.detector_data.argmax(), initial_shape)

        # find the com nearby the max using the given 
        # preprocessing_output_shape
        max_centered_data, former_center = center(
                self.detector_data,
                center_coordinates=det_max_pixel,
                return_former_center=True
        )

        max_cropped_data = crop_at_center(
            max_centered_data,
            final_shape=final_shape
        )
        cropped_com_pixel = center_of_mass(max_cropped_data)
        # convert the com coordinates in the detector frame
        det_com_pixel = (
            cropped_com_pixel 
            + (np.array(initial_shape) - np.array(final_shape))//2
            - (np.array(initial_shape)//2 - former_center)
        )

        # determine the detector pixel reference according to the 
        # provided method
        if self.parameters["det_pixel_reference_method"] == "max":
            det_pixel_reference = det_max_pixel
            self.verbose_print(
                "Method employed for the reference pixel determination is max"
            )
        elif self.parameters["det_pixel_reference_method"] == "com":
            det_pixel_reference = np.rint(det_com_pixel).astype(int)
            self.verbose_print(
                "Method employed for the reference pixel determination is com"
            )
            
        elif type(self.parameters["det_pixel_reference_method"]) == list:
            det_pixel_reference = [
                int(e) for e in self.parameters["det_pixel_reference_method"]
            ]
            self.verbose_print(
                "Reference pixel provided by user: "
                f"{self.parameters['det_pixel_reference_method']}"
            )
        
        # now proceed to the centering and cropping using
        # the det_pixel_reference as a reference
        centered_data = center(
                self.detector_data,
                center_coordinates=det_pixel_reference,
        )
        self.cropped_detector_data = crop_at_center(
            centered_data,
            final_shape=final_shape
        )

        # center and crop the mask
        self.mask = center(self.mask, center_coordinates=det_pixel_reference)
        self.mask = crop_at_center(self.mask, final_shape=final_shape)

        # redefine the max and com pixel coordinates in the new cropped data
        # system
        cropped_max_pixel = (
            det_max_pixel
            - (det_pixel_reference - np.array(initial_shape)//2)
            - (np.array(initial_shape)- final_shape)//2
        )
        cropped_com_pixel = (
            det_com_pixel 
            - (det_pixel_reference - np.array(initial_shape)//2)
            - (np.array(initial_shape)- final_shape)//2
        )

        # save the pixel reference in the detector frame and Q_lab frame
        self.det_pixel_reference = det_pixel_reference
        self.q_lab_reference = self.space_converter.det2lab(
            det_pixel_reference)
        self.q_lab_max = self.space_converter.det2lab(
            det_max_pixel
        )
        self.q_lab_com = self.space_converter.det2lab(
            np.rint(det_com_pixel).astype(int)
        )
        hkl = self.parameters["hkl"]
        qnorm_reference = np.linalg.norm(self.q_lab_reference)
        averaged_dspacing_reference = 2*np.pi/qnorm_reference
        averaged_lattice_constant_reference = (
            np.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2)
            * averaged_dspacing_reference
        )
        qnorm_max = np.linalg.norm(self.q_lab_max)
        averaged_dspacing_max = 2*np.pi/qnorm_max
        averaged_lattice_constant_max = (
            np.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2)
            * averaged_dspacing_max
        )
        qnorm_com = np.linalg.norm(self.q_lab_com)
        averaged_dspacing_com = 2*np.pi/qnorm_com
        averaged_lattice_constant_com = (
            np.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2)
            * averaged_dspacing_com
        )


        # print some info
        self.verbose_print(
            f"Max in the full detector frame at {det_max_pixel}\n"
            "Max in the cropped detector frame at "
            f"{tuple(cropped_max_pixel)}\n"
            f"Max corresponds to a qnorm of {qnorm_max} 1/A, "
            f"a d-spacing of {averaged_dspacing_max} A "
            f"and a lattice parameter of {averaged_lattice_constant_max} A\n\n"
            f"Com in the full detector frame at {det_com_pixel} "
            f"(based on a {final_shape[1], final_shape[2]} max-centered "
            "bounding box)\n"
            "Com in the cropped detector frame at "
            f"{tuple(cropped_com_pixel)}\n"
            f"Com corresponds to a qnorm of {qnorm_com} 1/A, "
            f"a d-spacing of {averaged_dspacing_com} A "
            f"and a lattice parameter of {averaged_lattice_constant_com} A\n\n"
            f"q_lab_reference is: {self.q_lab_reference} 1/A and corresponds "
            f"to a d-spacing of {averaged_dspacing_reference} A and a lattice "
            f"parameter of {averaged_lattice_constant_reference} A\n"
        )

        # plot the detector data in the full detector frame and in th
        # final frame
        self.preprocessing_figure = preprocessing_detector_data_plot(
            detector_data=self.detector_data,
            cropped_data=self.cropped_detector_data,
            det_pixel_reference=det_pixel_reference,
            det_max_pixel=det_max_pixel,
            det_com_pixel=det_com_pixel,
            cropped_max_pixel=cropped_max_pixel,
            cropped_com_pixel=cropped_com_pixel,
            title=f"Detector data preprocessing, S{self.parameters['scan']}"
        )
        
        
    def save_preprocessed_data(self):
        if os.path.isdir(self.parameters["dump_dir"]):
            self.verbose_print("[INFO] Dump directory already exists, results will be saved"
                f" in: {self.parameters['dump_dir']}"
            )
        else:
            self.verbose_print(
                "[INFO] Creating the dump directory at: "
                f"{self.parameters['dump_dir']}")
            os.mkdir(self.parameters["dump_dir"])
        
        self.preprocessing_figure.savefig(
            self.parameters["dump_dir"]
            + "/centering_cropping_detector_data.png",
            bbox_inches="tight",
            dpi=300
        )

        np.savez(
            self.parameters["dump_dir"]
            + f"/S{self.parameters['scan']}_pynx_input_data.npz",
            data=self.cropped_detector_data
        )
        np.savez(
            self.parameters["dump_dir"]
            + f"/S{self.parameters['scan']}_pynx_input_mask.npz",
            mask=self.
            mask
        )
        update_parameter_file(
            self.parameter_file_path, 
            {"q_lab_reference": self.q_lab_reference}
        )

    
    def orthgonolize(self, target_frame: str):
        pass

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
        isosurface, self.amplitude_distribution_figure = find_isosurface(
            amplitude,
            nbins=100,
            sigma_criterion=2,
            show=self.parameters["show"]
        )
        self.verbose_print("done.")
        self.verbose_print(f"[INFO] isosurface estimated to be {isosurface}")

        # store the the averaged dspacing and lattice constant in variables
        # so they can be saved later in the output file
        hkl = self.parameters["hkl"]
        self.verbose_print(
            f"[INFO] The theoretical probed Bragg peak reflection is {hkl}")
        qnorm = np.linalg.norm(self.q_lab_reference)

        self.averaged_dspacing = 2*np.pi/qnorm
        self.averaged_lattice_constant = (
            np.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2)
            * self.averaged_dspacing
        )

        self.structural_properties = get_structural_properties(
            data,
            isosurface,
            q_vector=self.q_lab_reference,
            hkl=hkl,
            voxel_size=self.voxel_size,
            phase_factor=-1, # it came out of pynx, phase must be -phase
            do_phase_ramp_removal=self.parameters["remove_phase_ramp"]
        )

        # plot the results
        to_be_plotted = {
            k: self.structural_properties[k]
            for k in ["amplitude", "phase", "displacement", 
                    "local_strain", "lattice_constant"]
        }

        self.postprocessing_figure = summary_slice_plot(
            comment=f"S{self.parameters['scan']}",
            support=zero_to_nan(self.structural_properties["support"]),
            save=(
                f"{self.parameters['dump_dir']}/S{self.parameters['scan']}"
                "_summary_slice_plot.png"
            ),
            show=self.parameters["show"],
            dpi=300,
            voxel_size=self.voxel_size,
            isosurface=isosurface,
            qnorm=qnorm,
            averaged_dspacing=self.averaged_dspacing,
            averaged_lattice_constant=self.averaged_lattice_constant,
            **to_be_plotted
        )

    def save_postprocessed_data(self) -> None:
        # save the results in a npz file
        template_path = (
            f"{self.parameters['dump_dir']}/S{self.parameters['scan']}"
        )
        self.verbose_print(
            "[INFO] Saving the results to the following path:\n"
            f"{template_path}_structural_properties.npz"
        )

        np.savez(
            f"{template_path}_structural_properties.npz",
            averaged_dspacing=self.averaged_dspacing,
            averaged_lattice_constant=self.averaged_lattice_constant,
            **self.structural_properties,
        )
        

        self.postprocessing_figure.savefig(
            f"{template_path}_summary_slice_plot.png",
            dpi=300,
            bbox_inches="tight"
        )

        # save the amplitude distriubtion figure
        self.amplitude_distribution_figure.savefig(
            f"{template_path}_amplitude_distribution.png",
            dpi=200,
            bbox_inches="tight"
        )


def preprocessing_detector_data_plot(
        detector_data: np.array,
        cropped_data: np.array,
        det_pixel_reference: Union[np.array, list, tuple],
        det_max_pixel: Union[np.array, list, tuple],
        det_com_pixel: Union[np.array, list, tuple],
        cropped_max_pixel: Union[np.array, list, tuple],
        cropped_com_pixel: Union[np.array, list, tuple],
        title: str=""

) -> matplotlib.figure.Figure:
    """
    Plot the detector data in the full detector data frame and in the 
    cropped/centered frame.fits

    :param detector_data: the raw detector data (np.array)
    :param cropped_data: the cropped/centered data (np.array)
    :det_pixel_reference: the pixel reference in the full detector frame
    (np.array, list or tuple)
    :det_max_pixel: the max pixel in the full detector frame
    (np.array, list or tuple)
    :det_com_pixel: the com pixel in the full detector fame (np.array,
    list, tuple)
    :cropped_max_pixel: the max pixel in the centered/cropped detector
    frame (np.array, list or tuple)
    :cropped_com_pixel: the com pixel in the centered/cropped detector
    frame (np.array, list or tuple)
    :title: the tile of the figure (string)

    :return: the matplotlib figure object
    """

    figure, axes = plt.subplots(2, 3, figsize=(14, 10))

    log_data = np.log10(detector_data +1)
    log_cropped_data = np.log10(cropped_data +1)
    vmin = 0
    vmax = np.max(log_data)

    initial_shape = detector_data.shape
    final_shape = cropped_data.shape


    axes[0, 0].matshow(
        log_data[det_max_pixel[0]],
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",
        origin="upper"
    )
    axes[0, 0].plot(
        np.repeat(det_pixel_reference[2], 2),
        det_pixel_reference[1] + np.array([-0.1*initial_shape[1], 0.1*initial_shape[1]]),
        color="w", 
        lw=0.5
    )
    axes[0, 0].plot(
        det_pixel_reference[2] + np.array([-0.1*initial_shape[2], 0.1*initial_shape[2]]),
        np.repeat(det_pixel_reference[1], 2),
        color="w", 
        lw=0.5
    )
    axes[0, 0].plot(
        det_com_pixel[2], 
        det_com_pixel[1],
        marker="x",
        markersize=10,
        linestyle="None",
        color="green",
        label="com",
    )
    axes[0, 0].plot(
        det_max_pixel[2], 
        det_max_pixel[1],
        marker="x",
        markersize=10,
        linestyle="None",
        color="red",
        label="max"
    )

    axes[0, 1].matshow(
        log_data[:, det_max_pixel[1], :],
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",
        origin="lower"
    )
    axes[0, 1].plot(
        np.repeat(det_pixel_reference[2], 2),
        det_pixel_reference[0] + np.array([-0.1*initial_shape[0], 0.1*initial_shape[0]]),
        color="w", 
        lw=0.5
    )
    axes[0, 1].plot(
        det_pixel_reference[2] + np.array([-0.1*initial_shape[2], 0.1*initial_shape[2]]),
        np.repeat(det_pixel_reference[0], 2),
        color="w", 
        lw=0.5
    )
    axes[0, 1].plot(
        det_com_pixel[2], 
        det_com_pixel[0],
        marker="x",
        markersize=10,
        linestyle="None",
        color="green",
        label="com",
    )
    axes[0, 1].plot(
        det_max_pixel[2], 
        det_max_pixel[0],
        marker="x",
        markersize=10,
        linestyle="None",
        color="red",
        label="max"
    )

    mappable = axes[0, 2].matshow(
        np.swapaxes(log_data[..., det_max_pixel[2]], axis1=0, axis2=1),
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",
        origin="upper"
    )
    axes[0, 2].plot(
        np.repeat(det_pixel_reference[0], 2),
        det_pixel_reference[1] + np.array(
            [- 0.1 * initial_shape[1],  + 0.1 * initial_shape[1]]),
        color="w", 
        lw=0.5
    )
    axes[0, 2].plot(
        det_pixel_reference[0] + np.array(
            [- 0.1 * initial_shape[0],  + 0.1 * initial_shape[0]]),
        np.repeat(det_pixel_reference[1], 2),
        color="w", 
        lw=0.5
    )
    axes[0, 2].plot(
        det_com_pixel[0], 
        det_com_pixel[1],
        marker="x",
        markersize=10,
        color="green",
        label="com"
    )
    axes[0, 2].plot(
        det_max_pixel[0], 
        det_max_pixel[1],
        marker="x",
        markersize=10,
        color="red",
        label="max",
    )

    axes[1, 0].matshow(
        log_cropped_data[cropped_max_pixel[0]],
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",
        origin="upper"
    )

    axes[1, 0].plot(
        np.repeat(final_shape[2]//2, 2),
        np.array([0.4*final_shape[1], 0.6*final_shape[1]]),
        color="w", 
        lw=0.5
    )
    axes[1, 0].plot(
        np.array([0.4*final_shape[2], 0.6*final_shape[2]]),
        np.repeat(final_shape[1]//2, 2),
        color="w", 
        lw=0.5
    )
    axes[1, 0].plot(
        cropped_com_pixel[2], 
        cropped_com_pixel[1],
        marker="x",
        markersize=10,
        color="green",
        label="com",
    )
    axes[1, 0].plot(
        cropped_max_pixel[2], 
        cropped_max_pixel[1],
        marker="x",
        markersize=10,
        color="red",
        label="max",
    )

    axes[1, 1].matshow(
        log_cropped_data[:, cropped_max_pixel[1], :],
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",
        origin="lower"
    )
    axes[1, 1].plot(
        np.repeat(final_shape[2]//2, 2),
        np.array([0.4*final_shape[0], 0.6*final_shape[0]]),
        color="w", 
        lw=0.5
    )
    axes[1, 1].plot(
        np.array([0.4*final_shape[2], 0.6*final_shape[2]]),
        np.repeat(final_shape[0]//2, 2),
        color="w", 
        lw=0.5
    )
    axes[1, 1].plot(
        cropped_com_pixel[2], 
        cropped_com_pixel[0],
        marker="x",
        markersize=10,
        linestyle="None",
        color="green",
        label="com",
    )
    axes[1, 1].plot(
        cropped_max_pixel[2], 
        cropped_max_pixel[0],
        marker="x",
        markersize=10,
        linestyle="None",
        color="red",
        label="max"
    )

    mappable = axes[1, 2].matshow(
        np.swapaxes(log_cropped_data[..., cropped_max_pixel[2]], axis1=0, axis2=1),
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",
        origin="upper"
    )
    axes[1, 2].plot(
        np.repeat(final_shape[0]//2, 2),
        np.array([0.4*final_shape[1], 0.6*final_shape[1]]),
        color="w", 
        lw=0.5
    )
    axes[1, 2].plot(
        np.array([0.4*final_shape[0], 0.6*final_shape[0]]),
        np.repeat(final_shape[1]//2, 2),
        color="w", 
        lw=0.5
    )
    axes[1, 2].plot(
        cropped_com_pixel[0], 
        cropped_com_pixel[1],
        marker="x",
        markersize=10,
        color="green",
        label="com"
    )
    axes[1, 2].plot(
        cropped_max_pixel[0], 
        cropped_max_pixel[1],
        marker="x",
        markersize=10,
        color="red",
        label="max",
    )

    # handle the labels
    axes[0, 0].set_xlabel("detector dim 2 axis")
    axes[0, 0].set_ylabel("detector dim 1 axis")

    axes[0, 1].set_xlabel("detector dim 2 axis")
    axes[0, 1].set_ylabel("rocking angle axis")

    axes[0, 2].set_xlabel("rocking angle axis")
    axes[0, 2].set_ylabel("detector dim 1 axis")

    axes[1, 0].set_xlabel("cropped dim 2 axis")
    axes[1, 0].set_ylabel("cropped dim 1 axis")

    axes[1, 1].set_xlabel("cropped dim 2 axis")
    axes[1, 1].set_ylabel("cropped rocking angle axis")

    axes[1, 2].set_xlabel("cropped rocking angle axis")
    axes[1, 2].set_ylabel("cropped dim 1 axis")

    axes[0, 1].set_title("raw detector data", size=18, y=1.8)
    axes[1, 1].set_title("cropped detector data", size=18, y=1.05)

    figure.canvas.draw()
    for ax in axes.ravel():
        ax.tick_params(axis="x",direction="in", pad=-15, colors="w")
        ax.tick_params(axis="y",direction="in", pad=-25, colors="w")
        ax.xaxis.set_ticks_position("bottom")

        xticks_loc, yticks_loc = ax.get_xticks(), ax.get_yticks()
        xticks_loc[1] = yticks_loc[1] = None
        
        xlabels, ylabels = ax.get_xticklabels(), ax.get_yticklabels()
        xlabels[1] = ylabels[1] = ""
        ax.xaxis.set_major_locator(mticker.FixedLocator(xticks_loc))
        ax.yaxis.set_major_locator(mticker.FixedLocator(yticks_loc))
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)

    # handle the colorbar
    l0, b0, w0, h0 = axes[0, 1].get_position().bounds
    l1, b1, w1, h1 = axes[1, 1].get_position().bounds
    center_y = (b0 + (b1+h1)) / 2
    # cax = figure.add_axes([l0, center_y, w0, 0.025])
    cax = figure.add_axes([l0, 0.52, w0, 0.020])
    cax.set_title("Log(Int.) (a.u.)")
    figure.colorbar(mappable, cax=cax, orientation="horizontal")

    # handle the legend
    axes[0, 1].legend(
        loc="center",
        ncol=2,
        bbox_to_anchor=((l0 + l0 + w0)/2, (b0 + center_y)/2),
        bbox_transform=figure.transFigure
    )

    figure.suptitle(title, y=0.95)

    return figure
