import glob
import os
import ruamel.yaml
import shutil
from string import Template
import subprocess
import sys
from typing import Callable, Dict, Optional
import time
import traceback

import paramiko
import yaml

try:
    from bcdi.preprocessing.preprocessing_runner import run as run_preprocessing
    from bcdi.postprocessing.postprocessing_runner import run as run_postprocessing
    from bcdi.utils.parser import ConfigParser
    IS_BCDI_AVAILABLE = True
except ModuleNotFoundError:
    print("The bcdi package is not installed. bcdi backend won't be available")
    IS_BCDI_AVAILABLE = False # is_bcdi_available
    BCDI_ERROR_TEXT = (
        "Cannot use 'bcdi' backend if bcdi package is not"
        "installed."
    )

from cdiutils.utils import pretty_print
from cdiutils.processing.find_best_candidates import find_best_candidates
from cdiutils.processing.processor import BcdiProcessor


def make_scan_parameter_file(
        output_parameter_file_path: str,
        parameter_file_template_path: str,
        updated_parameters: dict
) -> None:
    """
    Create a scan parameter file given a template and the parameters 
    to update.
    """

    with open(parameter_file_template_path, "r", encoding="utf8") as file:
        source = Template(file.read())

    scan_parameter_file = source.substitute(updated_parameters)

    with open(output_parameter_file_path, "w", encoding="utf8") as file:
        file.write(scan_parameter_file)


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

    yaml_file = ruamel.yaml.YAML()
    yaml_file.indent(mapping=ind, sequence=ind, offset=bsi)
    with open(file_path, "w", encoding="utf8") as file:
        yaml_file.dump(config, file)


if IS_BCDI_AVAILABLE:
    class BcdiPipelineParser(ConfigParser):
        def __init__(self, file_path: str) -> None:
            super().__init__(file_path)

        def load_arguments(self) -> Dict:
            raw_args = yaml.load(self.raw_config, Loader=yaml.SafeLoader)

            raw_args["preprocessing"].update(raw_args["general"])
            raw_args["postprocessing"].update(raw_args["general"])
            raw_args["pynx"].update(
                {"detector_distance": raw_args["general"]["detector_distance"]})

            self.arguments = {
                "preprocessing": self._check_args(raw_args["preprocessing"]),
                "pynx": raw_args["pynx"],
                "postprocessing": self._check_args(raw_args["postprocessing"]),
            }
            try:
                self.arguments["cdiutils"] = raw_args["cdiutils"]
            except KeyError:
                print("No cdiutils arguments given")
            return self.arguments
        
        def load_bcdi_parameters(self, procedure: str="preprocessing") -> Dict:
            raw_args = yaml.load(
                self.raw_config,
                Loader=yaml.SafeLoader
            )[procedure]
            raw_args.update(raw_args["general"])
            return self._check_args(raw_args)


def process(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception:
            print(
                "\n[ERROR] An error occured in the "
                f"'{func.__name__}' method... here is the traceback:\n"
            )
            traceback.print_exc()
            sys.exit(1)
    return wrapper

# TODO: better handling of parameter file, updating etc
class BcdiPipeline:
    """
    A class to handle the bcdi worfklow, from pre-processing to
    post-processing (bcdi package), including phase retrieval
    (pynx package).

    :param parameter_file_path: the path (str) of the scan parameter
    file that holds all the information related to the entire process.
    """
    def __init__(
            self,
            parameter_file_path: Optional[str]=None,
            parameters: Optional[dict]=None,
            backend: Optional[str]="cdiutils"
        ):

        self.parameter_file_path = parameter_file_path
        self.parameters = parameters

        if parameters is None:
            if parameter_file_path is None:
                raise ValueError(
                    "parameter_file_path or parameters must be provided"
                )
            self.parameters = self.load_parameters(backend)

        self.backend = backend

        if backend == "cdiutils":
            self.dump_directory = (
                self.parameters["cdiutils"]["metadata"]["dump_dir"]
            )
            self.scan = self.parameters["cdiutils"]["metadata"]["scan"]
        elif backend == "bcdi":
            if not IS_BCDI_AVAILABLE:
                raise ModuleNotFoundError(BCDI_ERROR_TEXT)
            self.dump_directory = self.parameters["preprocessing"][
                "save_dir"][0]
            self.scan = self.parameters['preprocessing']['scans'][0]
        else:
            raise ValueError(
                f"[ERROR] Unknwon backend value ({backend}), it must be either"
                " 'cdiutils' or 'bcdi'"
            )

        # the bcdi_processor attribute will be used only if backend
        # is cdiutils
        self.bcdi_processor = None

    def load_parameters(
            self,
            backend: Optional[str]=None,
            file_path: Optional[str]=None
    ) -> dict:
        """
        Load the paratemters from the configuration files.
        """
        if backend is None:
            backend = self.backend
        if file_path is None:
            file_path = self.parameter_file_path
        if backend  == "bcdi":
            return BcdiPipelineParser(
                file_path
            ).load_arguments()
        if backend == "cdiutils":
            with open(file_path, "r", encoding="utf8") as file:
                return yaml.load(
                    file,
                    Loader=yaml.FullLoader
                )
        else:
            raise ValueError(
                f"[ERROR] Unknwon backend value ({backend}), it must be either"
                " 'cdiutils' or 'bcdi'"
            )

    @process
    def preprocess(self, backend: Optional[str]=None) -> None:

        if backend is None:
            backend = self.backend

        if backend == "bcdi":
            if not IS_BCDI_AVAILABLE:
                raise ModuleNotFoundError(BCDI_ERROR_TEXT)
            os.makedirs(
                self.parameters["preprocessing"]["save_dir"][0],
                exist_ok=True
            )
            pretty_print(
                "[INFO] Proceeding to bcdi preprocessing using the bcdi "
                f"backend (scan {self.scan})"
            )
            run_preprocessing(prm=self.parameters["preprocessing"])
            pynx_input_template = "S*_pynx_norm_*.npz"
            pynx_mask_template = "S*_maskpynx_norm_*.npz"
            self.save_parameter_file()

        elif backend == "cdiutils":
            os.makedirs(
                self.parameters["cdiutils"]["metadata"]["dump_dir"],
                exist_ok=True
            )
            pretty_print(
                "[INFO] Proceeding to preprocessing using the cdiutils backend"
                f" (scan {self.parameters['cdiutils']['metadata']['scan']})"
            )
            self.bcdi_processor = BcdiProcessor(
                parameters=self.parameters["cdiutils"]
            )
            self.bcdi_processor.load_data()
            self.bcdi_processor.center_crop_data()
            self.bcdi_processor.save_preprocessed_data()
            pynx_input_template = "S*_pynx_input_data.npz"
            pynx_mask_template = "S*_pynx_input_mask.npz"
            if self.parameters["cdiutils"]["show"]:
                self.bcdi_processor.show_figures()
            # update the parameters
            self.parameters["cdiutils"].update(self.bcdi_processor.parameters)
            self.save_parameter_file()

        else:
            raise ValueError(
                f"[ERROR] Unknwon backend value ({backend}), it must be either"
                " 'cdiutils' or 'bcdi'"
            )

        try:
            data_path = glob.glob(
                f"{self.dump_directory}/{pynx_input_template}")[0]
            mask_path = glob.glob(
                    f"{self.dump_directory}/{pynx_mask_template}")[0]
        except IndexError as exc:
            raise FileNotFoundError(
                "[ERROR] file missing, something went"
                " wrong during preprocessing"
            ) from exc
        if self.parameter_file_path is not None:
            pretty_print("[INFO] Updating scan parameter file")
            update_parameter_file(
                self.parameter_file_path,
                {"data": data_path, "mask": mask_path}
            )
            self.parameters = self.load_parameters(backend)

    @process
    def phase_retrieval(
            self,
            machine: str="slurm-nice-devel",
            user: str=os.environ["USER"],
            number_of_nodes: int=2,
            key_file_path: str=os.environ["HOME"] + "/.ssh/id_rsa",
            pynx_slurm_file_template: str=None,
            remove_last_results: bool=False
    ) -> None:
        """
        Run the phase retrieval using pynx through ssh connection to a
        gpu machine.
        """

        pretty_print(
            "[INFO] Proceeding to PyNX phase retrieval "
            f"(scan {self.scan})"
        )

        if remove_last_results: 
            print("[INFO] Removing former results\n")
            files = glob.glob(self.dump_directory + "/*Run*.cxi")
            files += glob.glob(self.dump_directory + "/*Run*.png")
            for f in files:
                os.remove(f)

        pynx_input_file_path = (
            self.dump_directory + "/pynx-cdi-inputs.txt"
        )

        # Make the pynx input file
        with open(pynx_input_file_path, "w", encoding="utf8") as file:
            for key, value in self.parameters["pynx"].items():
                file.write(f"{key} = {value}\n")

        if machine == "slurm-nice-devel":
            # Make the pynx slurm file
            if pynx_slurm_file_template is None:
                pynx_slurm_file_template = (
                    f"{os.path.dirname(__file__)}/pynx-id01cdi_template.slurm"
                )
                print("Pynx slurm file template not provided, will take "
                      f"the default: {pynx_slurm_file_template}")

            with open(pynx_slurm_file_template, "r", encoding="utf8") as file:
                source = Template(file.read())
                pynx_slurm_text = source.substitute(
                    {
                        "number_of_nodes": number_of_nodes,
                        "data_path": self.dump_directory,
                        "SLURM_JOBID": "$SLURM_JOBID",
                        "SLURM_NTASKS": "$SLURM_NTASKS"
                    }
                )
            with open(
                    self.dump_directory + "/pynx-id01cdi.slurm",
                    "w",
                    encoding="utf8"
            ) as file:
                file.write(pynx_slurm_text)

        if os.uname()[1].startswith("p9"):
            with subprocess.Popen(
                    "source /sware/exp/pynx/activate_pynx.sh;"
                    f"cd {self.dump_directory};"
                    "mpiexec -n 4 /sware/exp/pynx/devel.p9/bin/"
                    "pynx-cdi-id01 pynx-cdi-inputs.txt",
                    shell=True,
                    executable="/bin/bash",
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
        # ssh to p9 machine and run phase retrieval
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=machine,
            username=user,
            pkey=paramiko.RSAKey.from_private_key_file(key_file_path)
        )

        print(f"[INFO] Connected to {machine}")
        if machine == "slurm-nice-devel":
            _, stdout, _= client.exec_command(
                f"cd {self.dump_directory};"
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
                    output = stdout.read().decode("utf-8")
                    print(output)

            # while loop to check if job has terminated
            process_status = "PENDING"
            while process_status != "COMPLETED":
                _, stdout, _ = client.exec_command(
                    f"sacct -j {job_id} -o state | head -n 3 | tail -n 1"
                )

                # python process needs to sleep here, otherwise it gets in
                # trouble with standard output management. Anyway, we need
                # to sleep in the while loop in order to wait for the remote
                # process to terminate.
                time.sleep(2)
                process_status = stdout.read().decode("utf-8").strip()
                print(f"[INFO] process status: {process_status}")

                if process_status == "RUNNING":
                    _, stdout, _ = client.exec_command(
                        f"cd {self.dump_directory};"
                        f"cat pynx-id01cdi.slurm-{job_id}.out "
                        "| grep 'CDI Run:'"
                    )
                    time.sleep(1)
                    print(stdout.read().decode("utf-8"))

                elif process_status == "CANCELLED+":
                    raise RuntimeError("[INFO] Job has been cancelled")
                elif process_status == "FAILED":
                    raise RuntimeError("[ERROR] Job has failed")

            if process_status == "COMPLETED":
                print(f"[INFO] Job {job_id} is completed.")

        else:
            _, stdout, stderr = client.exec_command(
                "source /sware/exp/pynx/activate_pynx.sh 2022.1;"
                f"cd {self.dump_directory};"
                "pynx-id01cdi.py pynx-cdi-inputs.txt "
                f"2>&1 | tee phase_retrieval_{machine}.log"
            )
            if stdout.channel.recv_exit_status():
                raise RuntimeError(
                    f"Error pulling the remote runtime {stderr.readline()}")
            for line in iter(lambda: stdout.readline(1024), ""):
                print(line, end="")
        client.close()

    def find_best_candidates(self, nb_to_keep: int=10) -> None:
        """Find the best candidates of the PyNX output"""
        pretty_print(
            "[INFO] Finding the best candidates of the PyNX run. "
            f"(scan {self.scan})"
        )
        # remove the previous candidates if needed
        files = glob.glob(self.dump_directory + "/candidate_*.cxi")
        if files:
            for f in files:
                os.remove(f)
        files = glob.glob(self.dump_directory + "/*Run*.cxi")
        if not files:
            print(
                "No PyNX output in the following directory: "
                f"{self.dump_directory} "
            )
        else:
            find_best_candidates(
                    files,
                    nb_to_keep=nb_to_keep,
                    criterion="mean_to_max"
                )

    @process
    def mode_decomposition(self) -> None:

        pretty_print(
            "[INFO] Running mode decomposition from /sware pynx "
            "installation "
            f"(scan {self.scan})"
        )

        # run the mode decomposition as a subprocesss
        with subprocess.Popen(
                "source /sware/exp/pynx/activate_pynx.sh 2022.1;"
                f"cd {self.dump_directory};"
                "pynx-cdi-analysis.py candidate_*.cxi modes=1 "
                "modes_output=mode.h5 2>&1 | tee mode_decomposition.log",
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

        if self.parameter_file_path is not None:
            pretty_print("[INFO] Updating scan parameter file")
            if self.backend == "bcdi":
                update_parameter_file(
                    self.parameter_file_path,
                    {"reconstruction_files": f"{self.dump_directory}mode.h5"}
                )
            else:
                update_parameter_file(
                    self.parameter_file_path,
                    {"reconstruction_file": f"{self.dump_directory}mode.h5"}
                )
            self.parameters = self.load_parameters()

    @process
    def postprocess(self, backend: Optional[str]=None) -> None:

        if backend is None:
            backend = self.backend

        if backend == "bcdi":
            if not IS_BCDI_AVAILABLE:
                raise ModuleNotFoundError(BCDI_ERROR_TEXT)
            pretty_print(
                "[INFO] Running post-processing from bcdi_strain.py "
                f"(scan {self.scan})"
            )

            run_postprocessing(prm=self.parameters["postprocessing"])
            self.save_parameter_file()

        elif backend == "cdiutils":

            pretty_print(
                "[INFO] Running post-processing using cdiutils backend "
                f"(scan {self.scan})"
            )

            if self.bcdi_processor is None:
                if any(
                        p not in self.parameters["cdiutils"].keys() 
                        for p in (
                            "q_lab_reference", "q_lab_max",
                            "q_lab_com", "det_reference_voxel"
                        )
                ):
                    preprocessing_parameters = self.load_parameters(
                        file_path=f"{self.dump_directory}/S{self.scan}_parameter_file.yml"
                    )["cdiutils"]
                    self.parameters["cdiutils"].update(
                        {
                            "det_reference_voxel": preprocessing_parameters[
                                "det_reference_voxel"
                            ],
                            "q_lab_reference": preprocessing_parameters[
                                "q_lab_reference"
                            ],
                            "q_lab_max": preprocessing_parameters["q_lab_max"],
                            "q_lab_com": preprocessing_parameters["q_lab_com"]
                        }
                    )
                self.bcdi_processor = BcdiProcessor(
                    parameters=self.parameters["cdiutils"]
                )
                self.bcdi_processor.reload_preprocessing_parameters()
                self.bcdi_processor.load_data()
            self.bcdi_processor.orthogonalize()
            self.bcdi_processor.postprocess()
            self.bcdi_processor.save_postprocessed_data()
            if self.parameters["cdiutils"]["show"]:
                self.bcdi_processor.show_figures()
            self.save_parameter_file()

        else:
            raise ValueError(
                f"[ERROR] Unknwon backend value ({backend}), it must be either"
                " 'cdiutils' or 'bcdi'"
            )

    def save_parameter_file(self) -> None:
        """
        Save the parameter file used during the analysis.
        """
        pretty_print(
            "Saving scan parameter file at the following location:\n"
            f"{self.dump_directory}"
        )
        output_file_path = (
            f"{self.dump_directory}/S{self.scan}_parameter_file.yml"
            # f"{os.path.basename(self.parameter_file_path)}"
        )
        if self.parameter_file_path is not None:
            shutil.copyfile(
                self.parameter_file_path,
                output_file_path
            )
        else:
            with open(output_file_path, "w", encoding="utf8") as file:
                yaml.dump(self.parameters, file)
