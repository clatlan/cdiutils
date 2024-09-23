"""
Definition of the BcdiPipeline class.

Authors:
    * Clément Atlan, clement.atlan@esrf.fr - 09/2024
"""


# Built-in dependencies.
import glob
import logging
import os
import shutil
from string import Template
import subprocess
import sys
import time

# Dependencies.
import numpy as np
import paramiko
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
from cdiutils.utils import CroppingHandler, oversampling_from_diffraction
from cdiutils.process.phaser import PhasingResultAnalyser
from cdiutils.process.facet_analysis import FacetAnalysisProcessor

# Base Pipeline class and pipeline-related functions.
from .base import Pipeline, LoggerWriter
from .parameters import check_params, convert_np_arrays, fill_pynx_params


def make_scan_parameter_file(
        output_param_file_path: str,
        param_file_template_path: str,
        updated_params: dict
) -> None:
    """
    Create a scan parameter file given a template and the parameters
    to update.
    """

    with open(param_file_template_path, "r", encoding="utf8") as file:
        source = Template(file.read())

    scan_parameter_file = source.substitute(updated_params)

    with open(output_param_file_path, "w", encoding="utf8") as file:
        file.write(scan_parameter_file)


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
            param_file_path: str = None,
            params: dict = None,
    ):
        """
        Initialisation method.

        Args:
            param_file_path (str, optional): the path to the
                parameter file. Defaults to None.
            parameters (dict, optional): the parameter dictionary.
                Defaults to None.

        """
        super(BcdiPipeline, self).__init__(param_file_path, params)

        check_params(self.params)

        self.scan = self.params["scan"]
        self.sample_name = self.params["sample_name"]
        # The dictionary of the Voxels Of Interest, in both the "full"
        # and "cropped" detector frame
        self.voi = {"full": {}, "cropped": {}}

        self.result_analyser: PhasingResultAnalyser = None

        print("BcdiPipeline initialised.")

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
            print(f"Raw detector data shape is: {shape}.")

        self.detector_data = loader.load_detector_data(
            scan=self.scan,
            roi=roi,
            binning_along_axis0=self.params["binning_along_axis0"]
        )
        self.angles = loader.load_motor_positions(
            scan=self.scan,
            roi=roi,
            binning_along_axis0=self.params["binning_along_axis0"]
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
            print(
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

    def _light_load(self) -> None:
        """
        Args:
            roi (tuple[slice], optional): the region of interest on the
                detector frame. Defaults to None.
            binning_along_axis0 (int, optional): whether or not to bin
                the data along the axis0, ie the rocking curve direction.
                Defaults to None.

        Raises:
            ValueError: check whether detector data, motor positions and
                mask have been correctly loaded.
        """
        final_shape = tuple(self.params["preprocessing_output_shape"])
        if (
                len(final_shape) != 3
                and self.params["binning_along_axis0"] is None
        ):
            raise ValueError(
                "final_shape must include 3 axis lengths or you must "
                "provide binning_along_axis0 parameter if you want "
                "light loading."
            )
        self.voi["full"]["ref"] = self.params["det_reference_voxel_method"][-1]
        if isinstance(self.voi["full"]["ref"], str):
            raise ValueError(
                "When light loading, det_reference_voxel_method must "
                "contain a tuple indicating the position of the voxel you "
                "want to crop the data at. Ex: [(100, 200, 200)]."
            )
        if len(self.voi["full"]["ref"]) == 2 and len(final_shape) == 3:
            final_shape = final_shape[1:]
        elif len(self.voi["full"]["ref"]) == 3 and len(final_shape) == 2:
            self.voi["full"]["ref"] = self.voi["full"]["ref"][1:]

        roi = CroppingHandler.get_roi(final_shape, self.voi["full"]["ref"])
        if len(roi) == 4:
            roi = [None, None, roi[0], roi[1], roi[2], roi[3]]

        print(
            f"\nLight loading requested, will use ROI {roi} and "
            "bin along rocking curve direction by "
            f"{self.params['binning_along_axis0']} during data loading."
        )
        self._load(roi=CroppingHandler.roi_list_to_slices(roi))

        # if user only specify position in 2D, extend it in 3D
        if len(self.voi["full"]["ref"]) == 2:
            self.voi["full"]["ref"] = (
                (self.detector_data.shape[0] // 2, )
                + tuple(self.voi["full"]["ref"])
            )

            roi = CroppingHandler.get_roi(
                final_shape, self.voi["full"]["ref"]
            )
            self.cropped_detector_data = self.cropped_detector_data[
                CroppingHandler.roi_list_to_slices(roi)
            ]
            print(f"New ROI is: {roi}.")

        # if the final_shape is 2D convert it in 3D
        if len(final_shape) == 2:
            final_shape = (
                (self.cropped_detector_data.shape[0], )
                + final_shape
            )

        # find the position in the cropped detector frame
        self.voi["cropped"]["ref"] = tuple(
                p - r if r else p  # if r is None, p-r must be p
                for p, r in zip(self.voi["full"]["ref"], roi[::2])
        )
        self.voi["full"]["max"], self.voi["full"]["com"] = None, None

        return final_shape, roi

    @Pipeline.process
    def _preprocess(self) -> None:

        # Check whether the requested shape is permitted by pynx
        self.ensure_pynx_shape()
        final_shape = self.params["preprocessing_output_shape"]

        if self.params["light_loading"]:
            final_shape, roi = self._light_load()

            self.cropped_detector_data = self.detector_data.copy()
            self.detector_data = None

        else:
            self._load()
            roi = self._crop_and_centre(final_shape)

        # print out the oversampling ratio and rebin factor suggestion
        ratios = oversampling_from_diffraction(
            self.cropped_detector_data
        )
        print(
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
        print(
            "\nSummary table:\n"
            + tabulate(table, headers="firstrow", tablefmt="fancy_grid"),
        )
        self._wrap_logs()  # Turn wrapping back on for regular logs

        if self.params["orthogonalise_before_phasing"]:
            print(
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
            print(
                "Will linearise the transformation between detector and"
                " lab space."
            )
            # Initialise the interpolator so we won't need to reload raw
            # data during the post processing. The converter will be saved.
            self.converter.init_interpolator(
                self.cropped_detector_data,
                final_shape,
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

        # Update the preprocessing_output_shape and the det_reference_voxel
        self.params["preprocessing_output_shape"] = final_shape
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

    def _crop_and_centre(self, final_shape: tuple) -> list:
        print(
            "[SHAPE & CROPPING] Method(s) employed for the reference "
            "voxel determination are "
            f"{self.params['det_reference_voxel_method']}."
        )
        self._unwrap_logs()  # Turn off wrapping for structured output
        (
            self.cropped_detector_data,
            self.voi["full"]["ref"],
            self.voi["cropped"]["ref"],
            roi
        ) = CroppingHandler.chain_centring(
                self.detector_data,
                final_shape,
                methods=self.params["det_reference_voxel_method"],
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

        print(
            "\n[SHAPE & CROPPING] The reference voxel was found at "
            f"{self.voi['full']['ref']} in the uncropped data frame\n"
            f"The processing_out_put_shape being {final_shape}, the roi "
            f"used to crop the data is {roi}.\n"
        )

        # set the q space area with the sample and detector angles
        # that correspond to the requested roi
        for key, value in self.angles.items():
            if isinstance(value, (list, np.ndarray)):
                self.angles[key] = value[np.s_[roi[0]:roi[1]]]

        return roi

    def _save_preprocess_data(self) -> None:
        """Save the data generated during the preprocessing."""
        # Prepare dir in which pynx phasing results will be saved.
        self.pynx_phasing_dir = self.dump_dir + "/pynx_phasing/"
        os.makedirs(self.pynx_phasing_dir, exist_ok=True)

        for name, data in zip(
                ("data", "mask"),
                (self.cropped_detector_data, self.mask)
        ):
            path = f"{self.pynx_phasing_dir}S{self.scan}_pynx_input_{name}.npz"
            np.savez(path, data=data)
            self.params["pynx"][name] = path

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

    def ensure_pynx_shape(self) -> tuple:
        shape = tuple(self.params["preprocessing_output_shape"])
        # if the final_shape is 2D convert it in 3D
        if len(shape) == 2:
            shape = (self.detector_data.shape[0], ) + shape
        checked_shape = tuple(
            s-1 if s % 2 == 1 else s for s in shape
        )
        if checked_shape != shape:
            print(
                f"[SHAPE & CROPPING] PyNX needs even input dimensions, "
                f"requested shape {shape} will be cropped to "
                f"{checked_shape}."
            )

        print(
            "[SHAPE & CROPPING] The preprocessing output shape is: "
            f"{checked_shape} and will be used for ROI dimensions."
        )
        self.params["preprocessing_output_shape"] = checked_shape

    @Pipeline.process
    def preprocess(self) -> None:


        dump_dir = self.params["dump_dir"]
        if os.path.isdir(dump_dir):
            print(
                "\n[INFO] Dump directory already exists, results will be "
                f"saved in:\n{dump_dir}."
            )
        else:
            print(
                f"[INFO] Creating the dump directory at: {dump_dir}")
            os.makedirs(
                dump_dir,
                exist_ok=True
            )
        os.makedirs(self.pynx_phasing_dir, exist_ok=True)
        self.bcdi_processor = BcdiProcessor(
            parameters=self.params
        )
        self.bcdi_processor.preprocess_data()
        self.bcdi_processor.save_preprocessed_data()
        pynx_input_template = (
            f"{self.pynx_phasing_dir}/S*_pynx_input_data.npz"
        )
        pynx_mask_template = (
            f"{self.pynx_phasing_dir}/S*_pynx_input_mask.npz"
        )

        try:
            data_path = glob.glob(pynx_input_template)[0]
            mask_path = glob.glob(pynx_mask_template)[0]
        except IndexError as exc:
            raise FileNotFoundError(
                "[ERROR] file missing, something went"
                " wrong during preprocessing"
            ) from exc

        # update the parameters
        if self.param_file_path is not None:
            update_parameter_file(
                self.param_file_path,
                {
                    "data": data_path,
                    "mask": mask_path,
                    "cdiutils": self.bcdi_processor.params
                }
            )

        self.params.update(self.bcdi_processor.params)
        self.params["pynx"].update({"data": data_path})
        self.params["pynx"].update({"mask": mask_path})
        self._save_parameter_file()


    @Pipeline.process
    def phase_retrieval(
            self,
            machine: str = None,  # "slurm-nice-devel",
            user: str = os.environ["USER"],
            number_of_nodes: int = 2,
            key_file_path: str = os.environ["HOME"] + "/.ssh/id_rsa",
            pynx_slurm_file_template: str = None,
            clear_former_results: bool = False
    ) -> None:
        """
        Run the phase retrieval using pynx through ssh connection to a
        gpu machine.
        """

        if clear_former_results:
            print("[INFO] Removing former results.\n")
            files = glob.glob(self.pynx_phasing_dir + "/*Run*.cxi")
            files += glob.glob(self.pynx_phasing_dir + "/*Run*.png")
            for f in files:
                os.remove(f)
            self.phasing_results = []

        pynx_input_file_path = (
            self.pynx_phasing_dir + "/pynx-cdi-inputs.txt"
        )

        # Make the pynx input file
        with open(pynx_input_file_path, "w", encoding="utf8") as file:
            for key, value in self.params["pynx"].items():
                file.write(f"{key} = {value}\n")

        if machine is None:
            print(
                "[INFO] No machine provided, assuming PyNX is installed on "
                "the current machine.\n"
            )
            if os.uname()[1].lower().startswith(("p9", "scisoft16")):
                # import threading
                # import signal
                # import sys

                # def signal_handler(sig, frame):
                #     print(
                #         "Keyboard interrupt received, exiting main program...")
                #     sys.exit(0)

                # # Function to read and print subprocess output
                # def read_output(pipe):
                #     try:
                #         for line in iter(pipe.readline, ""):
                #             print(line.decode("utf-8"), end="")
                #     except KeyboardInterrupt:
                #         pass  # Catch KeyboardInterrupt to exit the thread

                # # Register signal handler for keyboard interrupt
                # signal.signal(signal.SIGINT, signal_handler)
                # process = subprocess.Popen(
                #         # "source /sware/exp/pynx/activate_pynx.sh;"
                #         f"cd {self.pynx_phasing_dir};"
                #         # "mpiexec -n 4 /sware/exp/pynx/devel.p9/bin/"
                #         "pynx-cdi-id01 pynx-cdi-inputs.txt",
                #         shell=True,
                #         executable="/bin/bash",
                #         stdout=subprocess.PIPE,
                #         stderr=subprocess.PIPE,
                # )
                with subprocess.Popen(
                        # "source /sware/exp/pynx/activate_pynx.sh;"
                        f"cd {self.pynx_phasing_dir};"
                        # "mpiexec -n 4 /sware/exp/pynx/devel.p9/bin/"
                        "pynx-cdi-id01 pynx-cdi-inputs.txt",
                        shell=True,
                        executable="/bin/bash",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                ) as proc:
                    stdout, stderr = proc.communicate()
                    print(
                        "[STDOUT FROM SUBPROCESS RUNNING PYNX]\n",
                        stdout.decode("utf-8")
                    )
                    if proc.returncode:
                        print(
                            "[STDERR FROM SUBPROCESS RUNNING PYNX]\n",
                            stderr.decode("utf-8")
                        )

                # # Start threads to read stdout and stderr
                # stdout_thread = threading.Thread(
                #     target=read_output, args=(process.stdout,)
                # )
                # stderr_thread = threading.Thread(
                #     target=read_output, args=(process.stderr,)
                # )

                # stdout_thread.start()
                # stderr_thread.start()

                # try:
                #     # Wait for the subprocess to complete
                #     process.wait()
                # except KeyboardInterrupt:
                #     print(
                #         "Keyboard interrupt received, terminating "
                #         "subprocess..."
                #     )
                #     process.terminate()  # Send SIGTERM to the subprocess
                #     process.wait()  # Wait for the subprocess to terminate
                    
                #     # Terminate the threads
                #     stdout_thread.join(timeout=1)  # Wait for stdout_thread to terminate
                #     if stdout_thread.is_alive():
                #         print("Forcefully terminating stdout_thread...")
                #         stdout_thread.cancel()
                    
                #     stderr_thread.join(timeout=1)  # Wait for stderr_thread to terminate
                #     if stderr_thread.is_alive():
                #         print("Forcefully terminating stderr_thread...")
                #         stderr_thread.cancel()

                # # Wait for the threads to complete
                # stdout_thread.join()
                # stderr_thread.join()
        else:
            # ssh to the machine and run phase retrieval
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                hostname=machine,
                username=user,
                pkey=paramiko.RSAKey.from_private_key_file(key_file_path)
            )

            print(f"[INFO] Connected to {machine}")
            if machine == "slurm-nice-devel":

                # Make the pynx slurm file
                if pynx_slurm_file_template is None:
                    pynx_slurm_file_template = (
                        f"{os.path.dirname(__file__)}/"
                        "pynx-id01cdi_template.slurm"
                    )
                    print(
                        "Pynx slurm file template not provided, will take "
                        f"the default: {pynx_slurm_file_template}")

                with open(
                        pynx_slurm_file_template, "r", encoding="utf8"
                ) as file:
                    source = Template(file.read())
                    pynx_slurm_text = source.substitute(
                        {
                            "number_of_nodes": number_of_nodes,
                            "data_path": self.pynx_phasing_dir,
                            "SLURM_JOBID": "$SLURM_JOBID",
                            "SLURM_NTASKS": "$SLURM_NTASKS"
                        }
                    )
                with open(
                        self.pynx_phasing_dir + "/pynx-id01cdi.slurm",
                        "w",
                        encoding="utf8"
                ) as file:
                    file.write(pynx_slurm_text)

                # submit job using sbatch slurm command
                _, stdout, _ = client.exec_command(
                    f"cd {self.pynx_phasing_dir};"
                    "sbatch pynx-id01cdi.slurm"
                )
                job_submitted = False
                time.sleep(0.5)

                # read the standard output, decode it and print it
                output = stdout.read().decode("utf-8")

                # get the job id and remove '\n' and space characters
                while not job_submitted:
                    try:
                        job_id = output.split(" ")[3].strip()
                        job_submitted = True
                        print(output)
                    except IndexError:
                        print("Job still not submitted...")
                        time.sleep(3)
                        print(output)
                    except KeyboardInterrupt as err:
                        print("User terminated job with KeyboardInterrupt.")
                        client.close()
                        raise err

                # while loop to check if job has terminated
                process_status = "PENDING"
                while process_status != "COMPLETED":
                    _, stdout, _ = client.exec_command(
                        f"sacct -j {job_id} -o state | head -n 3 | tail -n 1"
                    )

                    # python process needs to sleep here, otherwise it gets in
                    # trouble with standard output management. Anyway, we need
                    # to sleep in the while loop in order to wait for the
                    # remote  process to terminate.
                    time.sleep(2)
                    process_status = stdout.read().decode("utf-8").strip()
                    print(f"[INFO] process status: {process_status}")

                    if process_status == "RUNNING":
                        _, stdout, _ = client.exec_command(
                            f"cd {self.pynx_phasing_dir};"
                            f"cat pynx-id01cdi.slurm-{job_id}.out "
                            "| grep 'CDI Run:'"
                        )
                        time.sleep(1)
                        print(stdout.read().decode("utf-8"))

                    elif process_status == "CANCELLED+":
                        raise RuntimeError("[INFO] Job has been cancelled")
                    elif process_status == "FAILED":
                        raise RuntimeError(
                            "[ERROR] Job has failed. Check out logs at: \n",
                            f"{self.pynx_phasing_dir}/"
                            f"pynx-id01cdi.slurm-{job_id}.out"
                        )

                if process_status == "COMPLETED":
                    print(f"[INFO] Job {job_id} is completed.")

            else:
                _, stdout, stderr = client.exec_command(
                    "source /sware/exp/pynx/activate_pynx.sh 2022.1;"
                    f"cd {self.pynx_phasing_dir};"
                    "pynx-id01cdi.py pynx-cdi-inputs.txt "
                    f"2>&1 | tee phase_retrieval_{machine}.log"
                )
                if stdout.channel.recv_exit_status():
                    raise RuntimeError(
                        f"Error pulling the remote runtime {stderr.readline()}"
                    )
                for line in iter(lambda: stdout.readline(1024), ""):
                    print(line, end="")
            client.close()

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
                        "preprocessing_output_shape"
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
                        "preprocessing_output_shape": preprocessing_params[
                            "preprocessing_output_shape"
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
                print(
                    "\nScan parameter file saved at:\n"
                    f"{output_file_path}"
                )

        else:
            convert_np_arrays(self.params)
            with open(output_file_path, "w", encoding="utf8") as file:
                yaml.dump(self.params, file)
            print(
                "\nScan parameter file saved at:\n"
                f"{output_file_path}"
            )

    def facet_analysis(self) -> None:
        facet_anlysis_processor = FacetAnalysisProcessor(
            parameters=self.params
        )
        facet_anlysis_processor.facet_analysis()
