import paramiko
import subprocess
import time
import glob
import os
import ruamel.yaml
import shutil
from string import Template
from typing import Callable, Dict
import traceback

from bcdi.preprocessing.preprocessing_runner import run as run_preprocessing
from bcdi.postprocessing.postprocessing_runner import run as run_postprocessing

from cdiutils.bcdi.find_best_candidates import find_best_candidates
from cdiutils.load.load_parameters import BcdiPipelineParser


def process(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(
                "\n[ERROR] An error occured in the "
                f"'{func.__name__}' method... here is the traceback:\n"
                # f"caught:\n'{e}'"
            )
            traceback.print_exc()
            exit(1)
    return wrapper


# def update_file(file_path: str, updated_parameters: Dict) -> None:
#     # TODO: Not proud of that function at all...
#     new_lines = []
#     keys = list(updated_parameters.keys())  
#     with open(file_path, "r+") as f:
#         for line in f:
#             key_found = False
#             i = 0
#             while (not key_found and i < len(keys)):
#                 if line.startswith(f"  {keys[i]}: "):
#                     new_lines.append(
#                         f"  {keys[i]}: {updated_parameters[keys[i]]}\n"
#                     )
#                     key_found = True
#                 i += 1
#             if not key_found:
#                 new_lines.append(line)
#     # erase file content
#     f = open(file_path, 'r+')
#     f.write(''.join(new_lines))
#     f.close()

def update_parameter_file(file_path: str, updated_parameters: Dict) -> None:
    config, ind, bsi = ruamel.yaml.util.load_yaml_guess_indent(open(file_path))
    for key in config.keys():
        for sub_key, v in updated_parameters.items():
            if sub_key in config[key]:
                config[key][sub_key] = v
    yaml = ruamel.yaml.YAML()
    yaml.indent(mapping=ind, sequence=ind, offset=bsi) 
    with open(file_path, "w") as f:
        yaml.dump(config, f)


def make_scan_parameter_file(
        output_scan_parameter_file: str,
        scan_parameter_file_template: str,
        scan: int,
        sample_name: str,
        working_directory: str,
        data_dir: str=None 
) -> None:

    dump_directory = "/".join((
        working_directory, sample_name, f"S{scan}"))
    
    with open(scan_parameter_file_template, "r") as f:
        source = Template(f.read())

    
    scan_parameter_file = source.substitute(
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

    with open(output_scan_parameter_file, "w") as f:
        f.write(scan_parameter_file)


def pretty_print(text: str) -> None:
    """Print text with a frame of stars."""

    stars = "*" * 80
    print("\n" + stars)
    print("*", end="")
    for i in range((80-len(text))//2 - 1):
        print(" ", end="")
    print(text, end="")
    for i in range((80-len(text))//2 - 1 + len(text)%2):
        print(" ", end="")
    print("*")
    print(stars + "\n")

# TODO: rewrite update function
# find_best_candidates loads files twice -> should load them only once
class BcdiPipeline:
    """
    A class to handle the bcdi worfklow, from pre-processing to
    post-processing (bcdi package), including phase retrieval
    (pynx package).
    
    :param scan_parameter_file: the path (str) of the scan parameter
    file that holds all the information related to the entire process.
    """
    def __init__(
            self: Callable,
            scan_parameter_file: str,
        ):

        self.scan_parameter_file = scan_parameter_file
        self.parameters = self.load_parameters()
        self.working_directory = self.parameters["preprocessing"][
            "save_dir"][0]

    @process
    def preprocess(self: Callable) -> None:
        pretty_print(
            "[INFO] Proceeding to bcdi preprocessing "
            f"(scan {self.parameters['preprocessing']['scans']})"
        )

        run_preprocessing(prm=self.parameters["preprocessing"])

        pretty_print("[INFO] Update scan parameter file")
        update_parameter_file(
            self.scan_parameter_file,
            {
                "data": glob.glob(f"{self.working_directory}/S*_pynx_*npz")[0],
                "mask": glob.glob(
                    f"{self.working_directory}/S*_maskpynx_*npz")[0],
            }
        )
        self.parameters = self.load_parameters()
    
    # def load_parameters(self):
    #     return ArgumentHandler(
    #         self.scan_parameter_file,
    #         script_type="all"
    #     ).load_arguments()

    def load_parameters(self):
        return BcdiPipelineParser(self.scan_parameter_file).load_arguments()
    

    
    # def update_scan_parameter_file(self, updated_parameters: Dict):
    #     pretty_print("[INFO] Update scan parameter file")

    #     with open(self.scan_parameter_file, "r") as f:
    #         content = Template(f.read())  

    #     new_content = content.substitute(updated_parameters)
   
    #     with open(self.scan_parameter_file, 'w') as f:
    #         f.write(new_content)
    
    @process
    def phase_retrieval(
            self: Callable,
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
                    "/data/id01/inhouse/clatlan/pythonies/cdiutils/cdiutils/"
                    "bcdi/pynx-id01cdi_template.slurm"
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
                    self.working_directory + "/pynx-id01cdi.slurm", 
                    "w"
            ) as f:
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
            time.sleep(1)

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
                time.sleep(5)
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

    def find_best_candidates(self: Callable, nb_to_keep=10) -> None:
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
        ) as process:
            stdout, stderr = process.communicate()
            print("[STDOUT FROM SUBPROCESS]\n", stdout.decode("utf-8"))
            if process.returncode:
                    print(
                        "[STDERR FROM SUBPROCESS]\n",
                        stderr.decode("utf-8")
                    )
        pretty_print("[INFO] Updating scan parameter file")
        update_parameter_file(
            self.scan_parameter_file,
            {"reconstruction_files": f"{self.working_directory}mode.h5"}
        )
        self.parameters = self.load_parameters()
        print(
            "The mode file has been updated in the scan parameter file.\n"
            "Here is the path: \n"
            f"{self.parameters['postprocessing']['reconstruction_files'][0]}"
        )
    
    @process
    def postprocess(self: Callable) -> None:
        pretty_print(
            "[INFO] Running post-processing from bcdi_strain.py "
            f"(scan {self.parameters['preprocessing']['scans'][0]})"
        )
        
        run_postprocessing(prm=self.parameters["postprocessing"])
    
    def save_scan_parameter_file(self: Callable) -> None:
        pretty_print(
            "Saving scan parameter file at the following location:\n"
            f"{self.parameters['preprocessing']['save_dir'][0]}"
        )
        shutil.copyfile(
            self.scan_parameter_file,
            f"{self.parameters['preprocessing']['save_dir'][0]}/"
            f"{os.path.basename(self.scan_parameter_file)}"
        )




