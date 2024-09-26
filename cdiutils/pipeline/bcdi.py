"""
Definition of the BcdiPipeline class.

Authors:
    * Clément Atlan, clement.atlan@esrf.fr - 09/2024
"""


# Built-in dependencies.
import glob
import os
import shutil
from string import Template
import subprocess

# Dependencies.
import numpy as np
import ruamel.yaml
from tabulate import tabulate
import yaml

# General cdiutils classes, to handle loading, beamline geometry and
# space conversion.
from cdiutils.converter import SpaceConverter
from cdiutils.geometry import Geometry
from cdiutils.load import Loader

# Plot function specifically made for the pipeline.
from .pipeline_plotter import PipelinePlotter

# Utility functions
from cdiutils.utils import (
    CroppingHandler,
    oversampling_from_diffraction,
    ensure_pynx_shape,
    hot_pixel_filter
)
from cdiutils.process.phaser import PhasingResultAnalyser
from cdiutils.process.facet_analysis import FacetAnalysisProcessor

# Base Pipeline class and pipeline-related functions.
from .base import Pipeline
from .parameters import check_params, convert_np_arrays, fill_pynx_params


class PynxScriptError(Exception):
    """Custom exception to handle pynx script failure."""
    pass


def update_parameter_file(file_path: str, updated_params: dict) -> None:
    """
    Update a parameter file with the provided dictionary that contains
    the parameters (keys, values) to update.
    """
    convert_np_arrays(updated_params)
    fill_pynx_params(updated_params)
    with open(file_path, "r", encoding="utf8") as file:
        config, ind, bsi = ruamel.yaml.util.load_yaml_guess_indent(file)

    for key in config.keys():
        for updated_key, updated_value in updated_params.items():
            if updated_key in config[key]:
                config[key][updated_key] = updated_value
            elif updated_key == key:
                config[key] = updated_value
            else:
                for sub_key in config[key].keys():
                    if (
                            isinstance(config[key][sub_key], dict)
                            and updated_key in config[key][sub_key]
                    ):
                        config[key][sub_key][updated_key] = updated_value

    yaml_file = ruamel.yaml.YAML()
    yaml_file.indent(mapping=ind, sequence=ind, offset=bsi)
    with open(file_path, "w", encoding="utf8") as file:
        yaml_file.dump(config, file)


class BcdiPipeline(Pipeline):
    """
    A class to handle the BCDI workflow, from pre-processing to
    post-processing, including phase retrieval (using PyNX package).
    Provide either a path to a parameter file or directly the parameter
    dictionary.

    Args:
        param_file_path (str, optional): the path to the
            parameter file. Defaults to None.
        parameters (dict, optional): the parameter dictionary.
            Defaults to None.

    """
    voxel_pos = ("ref", "max", "com")

    def __init__(
            self,
            params: dict = None,
            param_file_path: str = None,
    ):
        """
        Initialisation method.

        Args:
            param_file_path (str, optional): the path to the
                parameter file. Defaults to None.
            parameters (dict, optional): the parameter dictionary.
                Defaults to None.

        """
        super().__init__(params, param_file_path)

        check_params(self.params)

        self.scan = self.params["scan"]
        self.sample_name = self.params["sample_name"]
        self.pynx_phasing_dir = self.dump_dir + "/pynx_phasing/"

        # The dictionary of the Voxels Of Interest, in both the "full"
        # and "cropped" detector frame
        self.voi = {"full": {}, "cropped": {}}

        # Define base attributes
        self.detector_data: np.ndarray = None
        self.cropped_detector_data: np.ndarray = None
        self.mask: np.ndarray = None
        self.angles: dict = None

        self.result_analyser: PhasingResultAnalyser = None

        self.logger.info("BcdiPipeline initialised.")

    @Pipeline.process
    def preprocess(self) -> None:

        # Check whether the requested shape is permitted by pynx
        self.params["preprocess_shape"] = ensure_pynx_shape(
            self.params["preprocess_shape"], verbose=True
        )

        if self.params["light_loading"]:
            roi = self._light_load()
            # Filter the data
            self.cropped_detector_data = self._filter(
                self.cropped_detector_data
            )

        else:
            self._load()
            self._from_2d_to_3d_shape()
            self.logger.info(
                "The preprocessing output shape is: and "
                f"{self.params['preprocess_shape']} will be used for "
                "determining the ROI dimensions."
            )
            # Filter, crop and centre the detector data.
            self.cropped_detector_data, roi = self._crop_centre(
                self._filter(self.detector_data)
            )
        for r in roi:
            if r < 0:
                raise ValueError(
                    "The preprocess_shape and the detector voxel reference are"
                    f" not compatible: {self.params['preprocess_shape'] = }, "
                    f"{self.voi['full']['ref'] = }."
                )
        # print out the oversampling ratio and rebin factor suggestion
        ratios = oversampling_from_diffraction(
            self.cropped_detector_data
        )
        self.logger.info(
            "\nOversampling ratios calculated from diffraction pattern "
            "are: "
            + ", ".join(
                [f"axis{i}: {ratios[i]:.1f}" for i in range(len(ratios))]
            )
            + ". If low-strain crystal, you can set PyNX 'rebin' parameter to "
            "(" + ", ".join([f"{r//2}" for r in ratios]) + ")"
        )

        # position of the max and com in the cropped detector frame
        for pos in ("max", "com"):
            self.voi["cropped"][pos] = CroppingHandler.get_position(
                self.cropped_detector_data, pos
            )

        # initialise the space converter
        self.converter.energy = self.params["energy"]
        self.converter.init_q_space_area(
            roi=roi[2:],
            det_calib_params=self.params["det_calib_params"]
        )
        self.converter.set_q_space_area(**self.angles)

        # Initialise the fancy table with the columns
        table = [
            ["voxel", "uncrop. det. pos.", "crop. det. pos.",
             "dspacing (A)", "lat. par. (A)"]
        ]

        # get the position of the reference, max and det voxels in the
        # q lab space
        for pos in self.voxel_pos:
            key = f"q_lab_{pos}"
            det_voxel = self.voi["full"][pos]
            cropped_det_voxel = self.voi["cropped"][pos]

            self.params[key] = self.converter.index_det_to_q_lab(
                cropped_det_voxel
            )

            # compute the corresponding dpsacing and lattice parameter
            # for printing
            dspacing = self.converter.dspacing(self.params[key])
            lattice = self.converter.lattice_parameter(
                    self.params[key], self.params["hkl"]
            )
            table.append(
                [key.split('_')[-1], det_voxel, cropped_det_voxel,
                 f"{dspacing:.5f}", f"{lattice:.5f}"]
            )

        self._unwrap_logs()  # Turn off wrapping for structured output
        self.logger.info(
            "\nSummary table:\n"
            + tabulate(table, headers="firstrow", tablefmt="fancy_grid"),
        )
        self._wrap_logs()  # Turn wrapping back on for regular logs

        if self.params["orthogonalise_before_phasing"]:
            self.logger.info(
                "Orthogonalization required before phasing.\n"
                "Will use xrayutilities Fuzzy Gridding without linear "
                "approximation."
            )
            self.orthogonalised_intensity = (
                self.converter.orthogonalise_to_q_lab(
                     self.cropped_detector_data,
                     method="xrayutilities"
                )
            )
            # we must orthogonalise the mask and orthogonalised_intensity must
            # be saved as the pynx input
            self.mask = self.converter.orthogonalise_to_q_lab(
                self.mask,
                method="xrayutilities"
            )
            self.cropped_detector_data = self.orthogonalised_intensity
            q_lab_regular_grid = self.converter.get_xu_q_lab_regular_grid()
        else:
            self.logger.info(
                "Will linearise the transformation between detector and"
                " lab space."
            )
            # Initialise the interpolator so we won't need to reload raw
            # data during the post processing. The converter will be saved.
            self.converter.init_interpolator(
                self.cropped_detector_data,
                self.params["preprocess_shape"],
                space="both",
                direct_space_voxel_size=self.params["voxel_size"]
            )

            # Run the interpolation in the reciprocal space so we don't
            # do it later
            self.orthogonalised_intensity = (
                self.converter.orthogonalise_to_q_lab(
                    self.cropped_detector_data,
                    method="cdiutils"
                )
            )

            q_lab_regular_grid = self.converter.get_q_lab_regular_grid()

        # Update the preprocess_shape and the det_reference_voxel
        self.params["det_reference_voxel"] = self.voi["full"]["ref"]

        # plot and save the detector data in the full detector frame and
        # final frame
        dump_file_tmpl = (
            f"{self.dump_dir}/S{self.scan}_" + "detector_data_{}.png"
        )
        PipelinePlotter.detector_data(
            self.cropped_detector_data,
            full_det_data=self.detector_data,
            voxels=self.voi,
            title=(
                "Detector data preprocessing (slices), "
                f"{self.sample_name}, S{self.scan}"
            ),
            save=dump_file_tmpl.format("slices")
        )
        PipelinePlotter.detector_data(
            self.cropped_detector_data,
            full_det_data=self.detector_data,
            voxels=self.voi,
            integrate=True,
            title=(
                "Detector data preprocessing (sum), "
                f"{self.sample_name}, S{self.scan}"
            ),
            save=dump_file_tmpl.format("sum")
        )

        # Plot the reciprocal space data in the detector and lab frames
        PipelinePlotter.ortho_detector_data(
            self.cropped_detector_data,
            self.orthogonalised_intensity,
            q_lab_regular_grid,
            title=(
                r"From detector frame to q lab frame"
                f", {self.sample_name}, S{self.scan}"
            ),
            save=dump_file_tmpl.format("orthogonalisation")
        )

        # Save the data and the parameters in the dump directory
        self._save_preprocess_data()
        self._save_parameter_file()

    def _save_parameter_file(self) -> None:
        """
        Save the parameter file used during the analysis.
        """
        output_file_path = f"{self.dump_dir}/S{self.scan}_parameter_file.yml"

        if self.param_file_path is not None:
            try:
                shutil.copy(
                    self.param_file_path,
                    output_file_path
                )
            except shutil.SameFileError:
                self.logger.info(
                    "\nScan parameter file saved at:\n"
                    f"{output_file_path}"
                )

        else:
            convert_np_arrays(self.params)
            with open(output_file_path, "w", encoding="utf8") as file:
                yaml.dump(self.params, file)
            self.logger.info(
                "\nScan parameter file saved at:\n"
                f"{output_file_path}"
            )

    def _from_2d_to_3d_shape(self) -> tuple:
        if len(self.params["preprocess_shape"]) == 2:
            self.params["preprocess_shape"] = (
                (self.detector_data.shape[0], )
                + self.params["preprocess_shape"]
            )

    def _load(self, roi: tuple[slice] = None) -> None:
        """
        Load the raw detector data and motor positions.

        Args:
            roi (tuple[slice], optional): the region of interest on the
                detector frame. Defaults to None.

        Raises:
            ValueError: check whether detector data, motor positions and
                mask have been correctly loaded.
        """
        # Make the loader using the factory method, only parse the
        # parameters relevant to the Loader class, to make them explicit.
        loader_keys = (
            "beamline_setup", "scan", "sample_name", "experiment_file_path",
            "experiment_data_dir_path", "detector_data_path",
            "edf_file_template", "detector_name"
        )
        loader = Loader.from_setup(**{k: self.params[k] for k in loader_keys})

        if self.params.get("detector_name") is None:
            self.params["detector_name"] = loader.detector_name
            if self.params.get("detector_name") is None:
                raise ValueError(
                    "The automatic detection of the detector name is not"
                    "yet implemented for this setup"
                    f"({self.params['metadata']['setup']} = )."
                )

        shape = loader.load_detector_shape(self.scan)
        if shape is not None:
            self.logger.info(f"Raw detector data shape is: {shape}.")

        self.detector_data = loader.load_detector_data(
            scan=self.scan,
            roi=roi,
            rocking_angle_binning=self.params["rocking_angle_binning"]
        )
        self.angles = loader.load_motor_positions(
            scan=self.scan,
            roi=roi,
            rocking_angle_binning=self.params["rocking_angle_binning"]
        )
        self.mask = loader.get_mask(
            channel=self.detector_data.shape[0],
            detector_name=self.params["detector_name"],
            roi=(slice(None), roi[1], roi[2]) if roi else None
        )
        if any(
                data is None
                for data in (self.detector_data, self.angles)
        ):
            raise ValueError("Something went wrong during data loading.")

        if self.params["energy"] is None:
            self.params["energy"] = loader.load_energy(
                self.scan
            )
            if self.params["energy"] is None:
                raise ValueError(
                    "The automatic loading of energy is not yet implemented"
                    f"for this setup ({self.params['metadata']['setup']} = )."
                )
        if self.params["det_calib_params"] is None:
            self.logger.info(
                "\ndet_calib_params not provided, will try to find them. "
                "However, for a more accurate calculation, you'd better "
                "provide them."
            )
            self.params["det_calib_params"] = (
                loader.load_det_calib_params(self.scan)
            )
            if self.params["det_calib_params"] is None:
                raise ValueError(
                    "The automatic loading of det_calib_params is not yet "
                    "implemented for this setup "
                    f"({self.params['metadata']['setup']} = )."
                )

        # Initialise SpaceConverter, later use for orthogonalisation
        geometry = Geometry.from_setup(
            self.params["beamline_setup"]
        )
        self.converter = SpaceConverter(geometry=geometry)

    def _light_load(self) -> list:
        """
        Light load the detector data according to the provided
        preprocess_shape and voxel_reference_methods
        position. This allows to load only the region of interest
        determined by these two parameters.

        Raises:
            ValueError: When final_shape length is not 3 and
                 rocking_angle_binning are not provided.
            ValueError: If voxel_reference_methods is not provided.

        Returns:
            list: the roi associated to the cropped_detector_data.
        """
        if (
                len(self.params["preprocess_shape"]) != 3
                and self.params["rocking_angle_binning"] is None
        ):
            self.params["rocking_angle_binning"] = 1
            self.logger.warning(
                "When light loading is requested, preprocess_shape must "
                "include 3 axis lengths or rocking_angle_binning must be "
                "provided. Will set rocking_angle_binning to 1, but this might"
                " not be very efficient."
            )
        self.voi["full"]["ref"] = self.params["voxel_reference_methods"][-1]
        if isinstance(self.voi["full"]["ref"], str):
            raise ValueError(
                "When light loading, voxel_reference_methods must "
                "contain a tuple indicating the position of the voxel you "
                "want to crop the data at. Ex: [(100, 200, 200)]."
            )
        if (
                len(self.voi["full"]["ref"]) == 2
                and len(self.params["preprocess_shape"]) == 3
        ):
            self.params["preprocess_shape"] = (
                self.params["preprocess_shape"][1:]
            )
        elif (
                len(self.voi["full"]["ref"]) == 3
                and len(self.params["preprocess_shape"]) == 2
        ):
            self.voi["full"]["ref"] = self.voi["full"]["ref"][1:]

        roi = CroppingHandler.get_roi(
            self.params["preprocess_shape"], self.voi["full"]["ref"]
        )
        if len(roi) == 4:
            roi = [None, None, roi[0], roi[1], roi[2], roi[3]]

        self.logger.info(
            f"\nLight loading requested, will use ROI {roi} and "
            "bin along rocking curve direction by "
            f"{self.params['rocking_angle_binning']} during data loading."
        )
        self._load(roi=CroppingHandler.roi_list_to_slices(roi))

        # If user only specified position in 2D or provided a 2D
        # preprocess_shape, we must extend it to 3D and check the first
        # dimension length.
        if len(self.voi["full"]["ref"]) == 2:  # covers both cases
            self.voi["full"]["ref"] = (
                (self.detector_data.shape[0] // 2, )
                + tuple(self.voi["full"]["ref"])
            )
            self._from_2d_to_3d_shape()
            shape = ensure_pynx_shape(self.params["preprocess_shape"])
            roi = CroppingHandler.get_roi(shape, self.voi["full"]["ref"])

            # If the shape has changed due to PyNX conventions, then
            # only crop the data along the first axis direction.
            if shape != self.params["preprocess_shape"]:
                roi_1d = CroppingHandler.roi_list_to_slices(roi[:2])
                self.detector_data = self.detector_data[roi_1d]
                self.mask = self.mask[roi_1d]
                rocking_angle = Loader.get_rocking_angle(self.angles)
                self.angles[rocking_angle] = self.angles[rocking_angle][roi_1d]

            self.params["preprocess_shape"] = shape
            self.logger.info(f"New ROI is: {roi}.")

        # find the position in the cropped detector frame
        self.voi["cropped"]["ref"] = tuple(
                p - r if r else p  # if r is None, p-r must be p
                for p, r in zip(self.voi["full"]["ref"], roi[::2])
        )
        self.voi["full"]["max"], self.voi["full"]["com"] = None, None

        # Since we light load the data, the full detector data do not exist
        self.cropped_detector_data = self.detector_data.copy()
        self.detector_data = None

        return roi

    def _filter(self, data: np.ndarray) -> np.ndarray:
        if self.params["hot_pixel_filter"]:
            self.logger.info("hot_pixel_filter requested.")
            if isinstance(self.params["hot_pixel_filter"], tuple):
                self.logger.info(
                    "Hot pixel filter parameters are : "
                    f"{self.params['hot_pixel_filter']}"
                )

                data, hot_pixel_mask = hot_pixel_filter(
                    data, *self.params["hot_pixel_filter"]
                )
            else:
                self.logger.info(
                    "Will use defaults parameters: threshold = 1e2, "
                    "kernel_size = 3 "
                )
                data, hot_pixel_mask = hot_pixel_filter(data)
            self.mask *= hot_pixel_mask

        if self.params["background_level"]:
            self.logger.info(
                f"background_level set to {self.params['background_level']}, "
                "will remove the background."
            )
            data -= self.params["background_level"]
            data[data < 0] = 0
        return data

    def _crop_centre(self, detector_data) -> tuple[np.ndarray, list]:
        """
        Crop and centre the self.detector data and return the cropped
        detector data and the associated roi.
        """
        self.logger.info(
            "Method(s) employed for the voxel reference determination are "
            f"{self.params['voxel_reference_methods']}."
        )
        self._unwrap_logs()  # Turn off wrapping for structured output
        (
            cropped_detector_data,
            self.voi["full"]["ref"],
            self.voi["cropped"]["ref"],
            roi
        ) = CroppingHandler.chain_centring(
                detector_data,
                self.params["preprocess_shape"],
                methods=self.params["voxel_reference_methods"],
                verbose=True
        )
        self._wrap_logs()  # Turn wrapping back on for regular logs

        # position of the max and com in the full detector frame
        for pos in ("max", "com"):
            self.voi["full"][pos] = CroppingHandler.get_position(
                self.detector_data, pos
            )
        # convert numpy.int64 to int to make them serializable
        self.params["det_reference_voxel"] = tuple(
            int(e) for e in self.voi["full"]["ref"]
        )

        # center and crop the mask
        self.mask = self.mask[CroppingHandler.roi_list_to_slices(roi)]

        self.logger.info(
            "The reference voxel was found at "
            f"{self.voi['full']['ref']} in the uncropped data frame.\n"
            "The processing_out_put_shape being "
            f"{self.params['preprocess_shape']}, the roi used to crop "
            f"the data is {roi}.\n"
        )

        # set the q space area with the sample and detector angles
        # that correspond to the requested roi
        for key, value in self.angles.items():
            if isinstance(value, (list, np.ndarray)):
                self.angles[key] = value[np.s_[roi[0]:roi[1]]]

        return cropped_detector_data, roi

    def _save_preprocess_data(self) -> None:
        """Save the data generated during the preprocessing."""
        # Prepare dir in which pynx phasing results will be saved.
        os.makedirs(self.pynx_phasing_dir, exist_ok=True)

        for name, data in zip(
                ("data", "mask"),
                (self.cropped_detector_data, self.mask)
        ):
            path = f"{self.pynx_phasing_dir}S{self.scan}_pynx_input_{name}.npz"
            np.savez(path, data=data)

        if self.params["orthogonalise_before_phasing"]:
            regular_grid_func = self.converter.get_xu_q_lab_regular_grid
        else:
            regular_grid_func = self.converter.get_q_lab_regular_grid
            self.converter.save_interpolation_parameters(
                f"{self.dump_dir}/S{self.scan}_interpolation_parameters.npz"
            )
        np.savez(
            f"{self.dump_dir}/S{self.scan}_orthogonalised_intensity.npz",
            q_xlab=regular_grid_func()[0],
            q_ylab=regular_grid_func()[1],
            q_zlab=regular_grid_func()[2],
            orthogonalised_intensity=self.orthogonalised_intensity
        )

    def _make_slurm_file(self, template: str = None) -> None:
        # Make the pynx slurm file
        if template is None:
            template = (
                f"{os.path.dirname(__file__)}/pynx-id01-cdi_template.slurm"
            )
            self.logger.info(
                "Pynx slurm file template not provided, will take "
                f"the default: {template}"
            )
        else:
            self.logger.info(
                "Pynx slurm file template provided"
                f"{template = }."
            )
        with open(
                template, "r", encoding="utf8"
        ) as file:
            source = Template(file.read())
            pynx_slurm_text = source.substitute(
                {
                    "number_of_nodes": 2,
                    "data_path": self.pynx_phasing_dir,
                    "SLURM_JOBID": "$SLURM_JOBID",
                    "SLURM_NTASKS": "$SLURM_NTASKS"
                }
            )
        with open(
                self.pynx_phasing_dir + "/pynx-id01-cdi.slurm",
                "w",
                encoding="utf8"
        ) as file:
            file.write(pynx_slurm_text)

    @Pipeline.process
    def phase_retrieval(
            self,
            jump_to_cluster: bool = False,
            pynx_slurm_file_template: str = None,
            clear_former_results: bool = False,
            command: str = None,
    ) -> None:
        """
        Run the phase retrieval using pynx either by submitting a job to
        a slurm cluster or by running pynx script directly on the
        current machine.

        Args:
            jump_to_cluster (bool, optional): whether a job must be
                submitted to the cluster. Defaults to False.
            pynx_slurm_file_template (str, optional): the template for
                the pynx slurm file. Defaults to None.
            clear_former_results (bool, optional): whether ti clear the
                former results. Defaults to False.
            command (str, optional): the command to run when running
                pynx on the current machine. Defaults to None.

        Raises:
            PynxScriptError: if the pynx script failes.
            e: if the subprocess failes.
        """
        if clear_former_results:
            self.logger.info("Removing former results.\n")
            files = glob.glob(self.pynx_phasing_dir + "/*Run*.cxi")
            files += glob.glob(self.pynx_phasing_dir + "/*Run*.png")
            for f in files:
                os.remove(f)
            self.phasing_results = []

        pynx_input_path = (
            self.pynx_phasing_dir + "/pynx-cdi-inputs.txt"
        )

        # Update the data and mask paths in self.params.
        for name in ("data", "mask"):
            path = f"{self.pynx_phasing_dir}S{self.scan}_pynx_input_{name}.npz"
            self.params["pynx"][name] = path

        # Make the pynx input file.
        with open(pynx_input_path, "w", encoding="utf8") as file:
            for key, value in self.params["pynx"].items():
                file.write(f"{key} = {value}\n")

        if jump_to_cluster:
            self.logger.info("Jumping to cluster requested.")
            self._make_slurm_file(pynx_slurm_file_template)
            job_id, output_file = self.submit_job(
                job_file="pynx-id01-cdi.slurm",
                working_dir=self.pynx_phasing_dir
            )
            self.monitor_job(job_id, output_file)

        else:
            self.logger.info("Assuming the current machine is running PyNX.")
            if command is None:
                command = """
                    module load pynx
                    pynx-cdi-id01 pynx-cdi-inputs.txt
                """
            try:
                with subprocess.Popen(
                        ["bash", "-l", "-c", command],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=self.pynx_phasing_dir,  # Change to this directory
                        text=True,  # Ensures stdout/stderr are str, not bytes
                        # shell=True,
                        env=os.environ.copy(),
                        bufsize=1
                ) as proc:
                    # Stream stdout
                    for line in iter(proc.stdout.readline, ''):
                        self.logger.info(line.strip())

                    # Stream stderr
                    for line in iter(proc.stderr.readline, ''):
                        self.logger.error(line.strip())

                    # Wait for the process to complete and check the
                    # return code.
                    proc.wait()
                    if proc.returncode != 0:
                        self.logger.error(
                            "PyNX phasing process failed with return code "
                            f"{proc.returncode}"
                        )
                        raise PynxScriptError
                    else:
                        self.logger.info("Command completed successfully.")
            except subprocess.CalledProcessError as e:
                # Log the error if the job submission fails
                self.logger.error(
                    "Subprocess process failed with return code "
                    f"{e.returncode}: {e.stderr}"
                )
                raise e

    def analyse_phasing_results(
            self,
            sorting_criterion: str = "mean_to_max",
            plot: bool = True,
            plot_phasing_results: bool = True,
            plot_phase: bool = False,
            init_analyser: bool = True
    ) -> None:
        """
        Wrapper for analyse_phasing_results method of
        PhasingResultAnalyser class.

        Analyse the phase retrieval results by sorting them according to
        the sorting_criterion, which must be selected in among:
        * mean_to_max the difference between the mean of the
            Gaussian fitting of the amplitude histogram and the maximum
            value of the amplitude. We consider the closest to the max
            the mean is, the most homogeneous is the amplitude of the
            reconstruction, hence the best.
        * the sharpness the sum of the amplitude within support to
            the power of 4. For reconstruction with similar support,
            lowest values means greater amplitude homogeneity.
        * std the standard deviation of the amplitude.
        * llk the log-likelihood of the reconstruction.
        * llkf the free log-likelihood of the reconstruction.

        Args:
            sorting_criterion (str, optional): the criterion to sort the
                results with. Defaults to "mean_to_max".
            plot (bool, optional): whether or not to disable all plots.
            plot_phasing_results (bool, optional): whether to plot the
                phasing results. Defaults to True.
            plot_phase (bool, optional): whether the phase must be
                plotted. If True, will the phase is plotted with 
                amplitude as opacity. If False, amplitude is plotted
                instead. Defaults to False.
            init_analyser: (bool, optional): whether to force the
                reinitialisation of the PhasingResultAnalyser instance

        Raises:
            ValueError: if sorting_criterion is unknown.
        """
        if self.result_analyser is None or init_analyser:
            self.result_analyser = PhasingResultAnalyser(
                result_dir_path=self.pynx_phasing_dir
            )

        self.result_analyser.analyse_phasing_results(
            sorting_criterion=sorting_criterion,
            plot=plot,
            plot_phasing_results=plot_phasing_results,
            plot_phase=plot_phase
        )

    def select_best_candidates(
            self,
            nb_of_best_sorted_runs: int = None,
            best_runs: list = None
    ) -> None:
        """
        A function wrapper for
        PhasingResultAnalyser.select_best_candidates.
        Select the best candidates, two methods are possible. Either
        select a specific number of runs, provided they were analysed and
        sorted beforehand. Or simply provide a list of integers
        corresponding to the digit numbers of the best runs.

        Args:
            nb_of_best_sorted_runs (int, optional): the number of best
                runs to select, provided they were analysed beforehand.
                Defaults to None.
            best_runs (list[int], optional): the best runs to select.
                Defaults to None.

        Raises:
            ValueError: _description_
        """
        if not self.result_analyser:
            raise ValueError(
                "self.result_analyser not initialised yet. Run"
                " BcdiPipeline.analyse_phasing_results() first."
            )
        self.result_analyser.select_best_candidates(
            nb_of_best_sorted_runs,
            best_runs
        )

    @Pipeline.process
    def mode_decomposition(
            self,
            pynx_analysis_script: str = (
                "/cvmfs/hpc.esrf.fr/software/packages/"
                "ubuntu20.04/x86_64/pynx/2024.1/bin/pynx-cdi-analysis"
            ),
            run_command: str = None,
            machine: str = None,
            user: str = None,
            key_file_path: str = None
    ) -> None:
        """
        Run the mode decomposition using PyNX pynx-cdi-analysis.py
        script as a subprocess.

        Args:
            pynx_analysis_script (str, optional): Version of PyNX to
                use. Defaults to "2024.1".
            machine (str, optional): Remote machine to run the mode
                decomposition on. Defaults to None.
            user (str, optional): User for the remote machine. Defaults
                to None.
            key_file_path (str, optional): Path to the key file for SSH
                authentication. Defaults to None.
        """
        if run_command is None:
            run_command = (
                f"cd {self.pynx_phasing_dir};"
                f"{pynx_analysis_script} candidate_*.cxi --modes 1 "
                "--modes_output mode.h5 2>&1 | tee mode_decomposition.log"
            )

        if machine:
            print(f"[INFO] Remote connection to machine '{machine}'requested.")
            if user is None:
                user = os.environ["USER"]
                print(f"user not provided, will use '{user}'.")
            if key_file_path is None:
                key_file_path = os.environ["HOME"] + "/.ssh/id_rsa"
                print(
                    f"key_file_path not provided, will use '{key_file_path}'."
                )

            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                hostname=machine,
                username=user,
                pkey=paramiko.RSAKey.from_private_key_file(key_file_path)
            )

            _, stdout, stderr = client.exec_command(run_command)
            # read the standard output, decode it and print it
            formatted_stdout = stdout.read().decode("utf-8")
            formatted_stderr = stderr.read().decode("utf-8")
            print("[STDOUT FROM SSH PROCESS]\n")
            print(formatted_stdout)
            print("[STDERR FROM SSH PROCESS]\n")
            print(formatted_stderr)

            if stdout.channel.recv_exit_status():
                raise RuntimeError(
                    f"Error pulling the remote runtime {stderr.readline()}")
            client.close()

        # if no machine provided, run the mode decomposition as a subprocess
        else:
            with subprocess.Popen(
                    run_command,
                    shell=True,
                    executable="/usr/bin/bash",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
            ) as proc:
                stdout, stderr = proc.communicate()
                print("[STDOUT FROM SUBPROCESS]\n", stdout.decode("utf-8"))
                if proc.returncode:
                    print(
                        "[STDERR FROM SUBPROCESS]\n",
                        stderr.decode("utf-8")
                    )

            if self.param_file_path is not None:
                update_parameter_file(
                    self.param_file_path,
                    {"reconstruction_file":
                        f"{self.pynx_phasing_dir}mode.h5"}
                )
                self.params = self.load_parameters()

    @Pipeline.process
    def postprocess(self) -> None:

        if self.bcdi_processor is None:
            print("BCDI processor is not instantiated yet.")
            if any(
                    p not in self.params
                    or self.params[p] is None
                    for p in (
                        "q_lab_reference", "q_lab_max",
                        "q_lab_com", "det_reference_voxel",
                        "preprocess_shape"
                    )
            ):
                file_path = (
                        f"{self.dump_dir}"
                        f"S{self.scan}_parameter_file.yml"
                )
                print(f"Loading parameters from:\n{file_path}")
                preprocessing_params = self.load_parameters(
                    file_path=file_path)["cdiutils"]
                self.params.update(
                    {
                        "preprocess_shape": preprocessing_params[
                            "preprocess_shape"
                        ],
                        "det_reference_voxel": preprocessing_params[
                            "det_reference_voxel"
                        ],
                        "q_lab_reference": preprocessing_params[
                            "q_lab_reference"
                        ],
                        "q_lab_max": preprocessing_params["q_lab_max"],
                        "q_lab_com": preprocessing_params["q_lab_com"]
                    }
                )
            self.bcdi_processor = BcdiProcessor(
                parameters=self.params
            )

        self.bcdi_processor.orthogonalise()
        self.bcdi_processor.postprocess()
        self.bcdi_processor.save_postprocessed_data()
        self._save_parameter_file()
        if self.params["show"]:
            self.bcdi_processor.show_figures()


    def facet_analysis(self) -> None:
        facet_anlysis_processor = FacetAnalysisProcessor(
            parameters=self.params
        )
        facet_anlysis_processor.facet_analysis()
