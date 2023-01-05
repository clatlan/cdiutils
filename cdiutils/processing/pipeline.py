import paramiko
import subprocess
import time
import glob
import os
import shutil
from string import Template
from typing import Callable, Dict
import traceback
import yaml
import warnings

from bcdi.preprocessing.preprocessing_runner import run as run_preprocessing
from bcdi.postprocessing.postprocessing_runner import run as run_postprocessing
from bcdi.utils.parser import ConfigParser

from cdiutils.utils import pretty_print
from cdiutils.processing.find_best_candidates import find_best_candidates
from cdiutils.processing.processor import (
    BcdiProcessor, update_parameter_file
)


def make_parameter_file_path(
        output_parameter_file_path: str,
        parameter_file_path_template: str,
        scan: int,
        sample_name: str,
        working_directory: str,
        data_dir: str=None 
) -> None:

    dump_directory = "/".join((
        working_directory, sample_name, f"S{scan}"))
    
    with open(parameter_file_path_template, "r") as f:
        source = Template(f.read())

    
    parameter_file_path = source.substitute(
        {
            "scan": scan,
            "save_dir": dump_directory,
            # "reconstruction_file": dump_directory + "/modes.h5",
            "sample_name": sample_name,
            "data": "$data",
            "mask": "$mask",
            "data_dir": data_dir
        }
    )

    with open(output_parameter_file_path, "w") as f:
        f.write(parameter_file_path)

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
    
    def load_bcdi_parameters(self, process="preprocessing") -> Dict:
        raw_args = yaml.load(
            self.raw_config,
            Loader=yaml.SafeLoader
        )[process]
        raw_args.update(raw_args["general"])
        return self._check_args(raw_args)
    
    def load_preprocessing_parameters(self) -> Dict:
        return self.load_bcdi_parameters(process="preprocessing")
    
    def load_postprocessing_parameters(self) -> Dict:
        return self.load_bcdi_parameters(process="postprocessing")

    def load_pynx_parameters(self) -> Dict:
        return yaml.load(self.raw_config, Loader=yaml.SafeLoader)["pynx"]


def process(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(
                "\n[ERROR] An error occured in the "
                f"'{func.__name__}' method... here is the traceback:\n"
            )
            traceback.print_exc()
            quit(1)
    return wrapper


# TODO:
# check find_best_candidates loads files twice -> should load 
# them only once
class BcdiPipeline:
    """
    A class to handle the bcdi worfklow, from pre-processing to
    post-processing (bcdi package), including phase retrieval
    (pynx package).
    
    :param parameter_file_path: the path (str) of the scan parameter
    file that holds all the information related to the entire process.
    """
    def __init__(
            self: Callable,
            parameter_file_path: str,
        ):

        self.parameter_file_path = parameter_file_path
        self.parameters = self.load_parameters()
        self.working_directory = self.parameters["preprocessing"][
            "save_dir"][0]
        
        # the bcdi_processor attribute will be used only if backend
        # is True
        self.bcdi_processor = None

    @process
    def preprocess(self: Callable, backend: str="bcdi") -> None:
        
        os.makedirs(
            self.parameters["preprocessing"]["save_dir"][0],
            exist_ok=True
        )

        if backend == "bcdi":
            pretty_print(
                "[INFO] Proceeding to bcdi preprocessing using the bcdi "
                f"backend (scan {self.parameters['preprocessing']['scans']})"
            )
            run_preprocessing(prm=self.parameters["preprocessing"])
            pynx_input_template = "S*_pynx_norm_*.npz"
            pynx_mask_template = "S*_maskpynx_norm_*.npz"
        elif backend == "cdiutils":
            pretty_print(
                "[INFO] Proceeding to preprocessing using the cdiutils backend"
                f" (scan {self.parameters['cdiutils']['metadata']['scan']})"
            )
            self.bcdi_processor = BcdiProcessor(
                parameter_file_path=self.parameter_file_path
            )
            self.bcdi_processor.load_data()
            self.bcdi_processor.center_crop_data()
            self.bcdi_processor.save_preprocessed_data()
            pynx_input_template = "*S*_pynx_input_data.npz"
            pynx_mask_template = "*S*_pynx_input_mask.npz"

        else:
            raise ValueError(
                f"[ERROR] Unknwon backend value ({backend}), 'bcdi' or "
                " 'cdiutils' can be used."
            )
            

        pretty_print("[INFO] Updating scan parameter file")

        try:
            data_path = glob.glob(
                f"{self.working_directory}/{pynx_input_template}")[0]
            mask_path = glob.glob(
                    f"{self.working_directory}/{pynx_mask_template}")[0]
        except IndexError:
            raise FileNotFoundError(
                "[ERROR] file missing, something went"
                " wrong during preprocessing"
            ) 
        update_parameter_file(
            self.parameter_file_path, {"data": data_path, "mask": mask_path}
        )
        self.parameters = self.load_parameters()

    def load_parameters(self):
        return BcdiPipelineParser(self.parameter_file_path).load_arguments()
    
    @process
    def phase_retrieval(
            self,
            machine: str="lid01pwr9",
            user: str=os.environ["USER"],
            key_file_name: str=f"{os.environ['HOME']}/.ssh/id_rsa",
            pynx_slurm_file_template=None,
            remove_last_results=False
    ) -> None:
        
        pretty_print(
            "[INFO] Proceeding to PyNX phase retrieval "
            f"(scan {self.parameters['preprocessing']['scans'][0]})"
        )

        if remove_last_results: 
            print("[INFO] Removing former results\n")
            files = glob.glob(self.working_directory + "/*Run*.cxi")
            files += glob.glob(self.working_directory + "/*Run*.png")
            for f in files:
                os.remove(f)

        pynx_input_file_path = (
            self.working_directory + "/pynx-cdi-inputs.txt"
        )

        # Make the pynx input file
        with open(pynx_input_file_path, "w") as f:
            for key, value in self.parameters["pynx"].items():
                f.write(f"{key} = {value}\n")

        if machine == "slurm-nice-devel":
            # Make the pynx slurm file
            if pynx_slurm_file_template == None:
                pynx_slurm_file_template = (
                    f"{os.path.dirname(__file__)}/pynx-id01cdi_template.slurm"
                )
                print("Pynx slurm file template not provided, will take "
                      f"the default: {pynx_slurm_file_template}")
            
            with open(pynx_slurm_file_template, "r") as f:
                source = Template(f.read())
                pynx_slurm_text = source.substitute(
                    {
                        "data_path": self.working_directory,
                        "SLURM_JOBID": "$SLURM_JOBID",
                        "SLURM_NTASKS": "$SLURM_NTASKS"
                    }
                )
            with open(
                self.working_directory + "/pynx-id01cdi.slurm", "w") as f:
                f.write(pynx_slurm_text)

        # ssh to p9 machine and run phase retrieval
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=machine,
            username=user, 
            pkey=paramiko.RSAKey.from_private_key_file(key_file_name)
        )
        # transport = client.get_transport()
        # chanel = transport.open_x11_channel()
        # import commands
        # def testFunc():
        #     cmd = "xterm"
        #     result = commands.getoutput(cmd)
        # chanel.request_x11() 
        print(f"[INFO] Connected to {machine}")
        if machine == "slurm-nice-devel":
            _, stdout, _= client.exec_command(
                f"cd {self.working_directory};"
                "sbatch pynx-id01cdi.slurm"
            )
            time.sleep(0.5)

            # read the standard output, decode it and print it
            output = stdout.read().decode("utf-8")
            print(output)

            # get the job id and remove '\n' and space characters
            job_id = output.split(" ")[3].strip() # 

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
                        f"cd {self.working_directory};"
                        f"cat pynx-id01cdi.slurm-{job_id}.out "
                        "| grep 'CDI Run:'"
                    )
                    time.sleep(1)
                    print(stdout.read().decode("utf-8"))
                
                elif process_status == "CANCELLED+":
                    raise Exception("[INFO] Job has been cancelled")
                elif process_status == "FAILED":
                    raise Exception("[ERROR] Job has failed")
            
            if process_status == "COMPLETED":
                print(f"[INFO] Job {job_id} is completed.")
            
        else:
            _, stdout, stderr = client.exec_command(
                "source /sware/exp/pynx/activate_pynx.sh;"
                f"cd {self.working_directory};"
                "pynx-id01cdi.py pynx-cdi-inputs.txt "
                f"2>&1 | tee phase_retrieval_{machine}.log"
            )
            if stdout.channel.recv_exit_status():
                raise Exception(
                    f"Error pulling the remote runtime {stderr.readline()}")
            for line in iter(lambda: stdout.readline(1024), ""):
                print(line, end="")
        client.close()

    def find_best_candidates(self: Callable, nb_to_keep=10):
        # Find the best candidates of the PyNX output
        pretty_print(
            "[INFO] Finding the best candidates of the PyNX run. "
            f"(scan {self.parameters['preprocessing']['scans'][0]})"
        )
        # remove the previous candidates if needed
        files = glob.glob(self.working_directory + "/candidate_*.cxi")
        if files:
            for f in files:
                os.remove(f)
        files = glob.glob(self.working_directory + "/*Run*.cxi")
        if not files:
            print(
                "No PyNX output in the following directory: "
                f"{self.working_directory} "
            )
        else:
            find_best_candidates(
                    files,
                    nb_to_keep=nb_to_keep,
                    criterion="std"
                )

    @process
    def mode_decomposition(self: Callable) -> None:

        pretty_print(
            "[INFO] Running mode decomposition from /sware pynx "
            "installation "
            f"(scan {self.parameters['preprocessing']['scans'][0]})"
        )

        # run the mode decomposition as a subprocesss
        with subprocess.Popen(
                "source /sware/exp/pynx/activate_pynx.sh;"
                f"cd {self.working_directory};"
                "pynx-cdi-analysis.py candidate_*.cxi modes=1 "
                "modes_output=mode.h5 | tee mode_decomposition.log",
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
        pretty_print("[INFO] Updating scan parameter file")
        update_parameter_file(
            self.parameter_file_path,
            {"reconstruction_files": f"{self.working_directory}mode.h5"}
        )
        self.parameters = self.load_parameters()
        print(
            "The mode file has been updated in the scan parameter file.\n"
            "Here is the path: \n"
            f"{self.parameters['postprocessing']['reconstruction_files'][0]}"
        )
    
    @process
    def postprocess(self: Callable, backend: str="bcdi") -> None:

        if backend == "bcdi":
            pretty_print(
                "[INFO] Running post-processing from bcdi_strain.py "
                f"(scan {self.parameters['preprocessing']['scans'][0]})"
            )
            
            run_postprocessing(prm=self.parameters["postprocessing"])

        elif (backend == "cdiutils"):
            pretty_print(
                "[INFO] bcdi package will be used for the orthogonalization "
                "only, cdiutils will be used for the phase manipulation"
            )

            pretty_print("[INFO] First, running orthogonalization")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                run_postprocessing(
                    prm=self.parameters["postprocessing"],
                    procedure="orthogonalization"
                )
            pretty_print(
                "[INFO] Now, processing the phase to get the structural "
                "properties."
            )

            if self.bcdi_processor is None:
                self.bcdi_processor = BcdiProcessor(
                    parameter_file_path=self.parameter_file_path
                )
                self.bcdi_processor.reload_preprocessing_parameters()
                
            self.bcdi_processor.load_orthogonolized_data(
                f"{self.parameters['postprocessing']['save_dir'][0]}/"
                f"S{self.parameters['postprocessing']['scans'][0]}"
                "_orthogonolized_reconstruction_"
                f"{self.parameters['postprocessing']['save_frame']}.npz"
            )
            self.bcdi_processor.postprocess()
            self.bcdi_processor.save_postprocessed_data()


    def save_parameter_file(self: Callable) -> None:
        pretty_print(
            "Saving scan parameter file at the following location:\n"
            f"{self.parameters['preprocessing']['save_dir'][0]}"
        )
        shutil.copyfile(
            self.parameter_file_path,
            f"{self.parameters['preprocessing']['save_dir'][0]}/"
            f"{os.path.basename(self.parameter_file_path)}"
        )