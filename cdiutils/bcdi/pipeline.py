import paramiko
import subprocess
import time
import glob
import yaml
from string import Template
from typing import Callable

from bcdi.utils.parser import ConfigParser
from bcdi.preprocessing.preprocessing_runner import run as run_preprocessing
from bcdi.postprocessing.postprocessing_runner import run as run_postprocessing

from cdiutils.bcdi.find_best_candidates import find_best_candidates
from cdiutils.load.analysis import ArgumentHandler


def make_scan_file(
        output_scan_file,
        scan_file_template_path,
        scan,
        sample_name,
        working_directory,
):
    dump_directory = "/".join((
        working_directory, sample_name, f"S{scan}"))
    
    with open(scan_file_template_path, "r") as f:
        source = Template(f.read())
    
    scan_file = source.substitute(
        {
            "scan": scan,
            "save_dir": dump_directory,
            "reconstruction_file": dump_directory + "/modes.h5",
            "sample_name": sample_name,
        }
    )

    with open(output_scan_file, "w") as f:
        f.write(scan_file)


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


class BcdiPipeline():
    def __init__(self: Callable, scan_file: str, user_parameters: str):
        self.scan_file = scan_file
        self.user_parameters = user_parameters
        self.working_directory = ArgumentHandler(
            scan_file
        ).load_arguments()["save_dir"]

    def preprocess(self: Callable) -> None:
        pretty_print("[INFO] Proceeding to bcdi preprocessing")

        parser = ConfigParser(self.scan_file)
        run_preprocessing(prm=parser.load_arguments())
        self.update_scan_file()
    
    def update_scan_file(self):
        with open(self.scan_file, "r") as f:
            content = yaml.load(f, Loader=yaml.FullLoader)
        content["data"] = glob.glob(
            f"{self.working_directory}/S{content['scan']}_pynx_*npz"
        )[0]
        content["mask"] = glob.glob(
            f"{self.working_directory}/S{content['scan']}_maskpynx_*npz"
        )[0]
        with open(self.scan_file, 'w') as f:
            yaml.dump(content, f)

    def phase_retrieval(self: Callable, machine: str="lid01pwr9") -> None:
        pretty_print("[INFO] Proceeding to PyNX phase retrieval")

        pynx_input_file_path = (
            self.working_directory + "/pynx-cdi-inputs.txt"
        )

        # Make the pynx input file
        with open(pynx_input_file_path, "w") as f:
            for key, value in  ArgumentHandler(
                    self.scan_file,
                    script_type="pynx"
            ).load_arguments().items():
                f.write(f"{key} = {value}\n")

        if machine == "slurm-nice-devel":
            # Make the pynx slurm file
            with open(
                    self.user_parameters["pynx_slurm_template"],
                    "r"
            ) as f:
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
        ssh = paramiko.SSHClient()
        k = paramiko.RSAKey.from_private_key_file(
            self.user_parameters["key_file_name"]
        )

        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=machine,
            username=self.user_parameters["user_name"], 
            pkey=k
        )
        print(f"[INFO] Connected to {machine}")
        if machine == "slurm-nice-devel":
            cmd = (
                f"cd {self.working_directory};"
                "sbatch pynx-id01cdi.slurm"
            )
            stdin, stdout, stderr = ssh.exec_command(cmd)
            time.sleep(5)

            # read the standard output, decode it and print it
            output = stdout.read().decode("utf-8")
            print(output)

            # get the job id and remove '\n' and space characters
            job_id = output.split(" ")[3].strip() # 

            # while loop to check if job has terminated
            process_status = "PENDING"
            while process_status != "COMPLETED":
                _, stdout, _ = ssh.exec_command(
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
                    _, stdout, _ = ssh.exec_command(
                        f"cd {self.working_directory};"
                        f"cat pynx-id01cdi.slurm-{job_id}.out "
                        "| grep 'CDI Run:'"
                    )
                    time.sleep(1)
                    print(stdout.read().decode("utf-8"))
                
                elif process_status == "CANCELLED+":
                    print("[INFO] Job has been cancelled")
                    break
                elif process_status == "FAILED":
                    print("[ERROR] Job has failed")
                    break
            
            if process_status == "COMPLETED":
                print(f"[INFO] Job {job_id} is completed.")
            
        else:
            _, stdout, _ = ssh.exec_command(
                "source /sware/exp/pynx/activate_pynx.sh;"
                f"cd {self.working_directory};"
                "pynx-id01cdi.py pynx-cdi-inputs.txt "
                f"2>&1 | tee phase_retrieval_{machine}.log"
            )
            time.sleep(1)
            for line in iter(lambda: stdout.readline(2048), ""):
                print(line, end="")
        ssh.close()
        # TODO: Manage case where failure occurs, return 1 for instance

    def find_best_candidates(self: Callable) -> None:
        # Find the best candidates of the PyNX output
        pretty_print(
            "[INFO] Finding the best candidates of the PyNX run."
        )
        files = glob.glob(self.working_directory + "/*Run*.cxi")
        if not files:
            print(
                "No PyNX output in the following directory: "
                f"{self.working_directory} "
            )
            return
        find_best_candidates(
                files,
                nb_to_keep=self.user_parameters[
                    "nb_of_pynx_reconstructions_to_keep"],
                criterion="std"
            )

    def mode_decomposition(self: Callable) -> None:

        pretty_print(
            "[INFO] Running mode decomposition from /sware pynx "
            "installation"
        )

        # run the mode decomposition as a subprocesss
        with subprocess.Popen(
            "source /sware/exp/pynx/activate_pynx.sh;"
            f"cd {self.working_directory};"
            "pynx-cdi-analysis.py candidate_*.cxi modes=1 "
            "modes_output=modes.h5 > mode_decomposition.log",
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as process:
            stdout, stderr = process.communicate()
            stdout, stderr = process.communicate()
            print("[STDOUT FROM SUBPROCESS]\n", stdout.decode("utf-8"))
            if process.returncode:
                    print(
                        "[STDERR FROM SUBPROCESS]\n",
                        stderr.decode("utf-8")
                    )

        # TODO : Check if mode decomposition is better on p9gpu 
        # (or lid01pwr9 or lid01gpu1)
    
    def postprocess(self: Callable) -> None:
        pretty_print("[INFO] Running post-processing from bcdi_strain.py")

        parser = ConfigParser(self.scan_file)
        run_postprocessing(prm=parser.load_arguments())
