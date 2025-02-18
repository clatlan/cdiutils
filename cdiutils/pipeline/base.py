
from abc import ABC
from functools import wraps
import logging
import os
from typing import Callable
import signal
import subprocess
import sys
import textwrap
import time


import numpy as np
import yaml


from cdiutils.plot.formatting import update_plot_params


# Define a custom log level for JOB
JOB_LOG_LEVEL = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(JOB_LOG_LEVEL, "JOB")


# Create a method to log at the JOB level
def job(self, message, *args, **kwargs):
    if self.isEnabledFor(JOB_LOG_LEVEL):
        self._log(JOB_LOG_LEVEL, message, args, **kwargs)


logging.Logger.job = job


class LoggerWriter:
    """
    Custom stream to send stdout (print statements) directly to
    logger in real-time.
    """
    def __init__(self, logger, level, wrap=True):
        self.logger = logger
        self.level = level
        self.wrap = wrap

    def write(self, message):
        if message.strip():
            # Only log if there's something to log
            # (ignores empty messages)
            if self.wrap:
                # Wrap lines at 79 characters
                wrapped_message = textwrap.fill(message.strip(), width=79)
            else:
                wrapped_message = message.strip()
            self.logger.log(self.level, "\n" + wrapped_message + "\n")

    def flush(self):
        """
        Flush method is needed for compatibility with `sys.stdout`.
        """
        pass


class JobCancelledError(Exception):
    """Custom exception to handle job cancellations by the user."""
    pass


class JobFailedError(Exception):
    """Custom exception to handle job failure."""
    pass


class Pipeline(ABC):
    def __init__(
            self,
            params: dict = None,
            param_file_path: str = None
    ):
        """
        Initialisation method.

        Args:
            param_file_path (str, optional): the path to the
                parameter file. Defaults to None.
            parameters (dict, optional): the parameter dictionary.
                Defaults to None.

        """
        self.param_file_path = param_file_path
        self.params = params

        if params is None:
            if param_file_path is None:
                raise ValueError(
                    "param_file_path or parameters must be provided"
                )
            self.params = self.load_parameters()

        # Create the dump directory
        self.dump_dir = self.params["dump_dir"]
        self.make_dump_dir()

        # Initialise the logger
        self.logger = self._init_logger()

        self.interrupted = False  # Flag to check for keyboard interrupt

        # Set the printoptions legacy to 1.21, otherwise types are printed.
        np.set_printoptions(legacy="1.21")

        # update the plot parameters
        update_plot_params()

    def make_dump_dir(self) -> None:
        dump_dir = self.params["dump_dir"]
        if os.path.isdir(dump_dir):
            print(
                "\nDump directory already exists, results will be "
                f"saved in:\n{dump_dir}."
            )
        else:
            print(
                f"Creating the dump directory at: {dump_dir}")
            os.makedirs(dump_dir, exist_ok=True)

    def _init_logger(self) -> logging.Logger:
        # Remove all handlers associated with the root logger (Jupyter
        # default).
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logger = logging.getLogger("PipelineLogger")

        # Check if the logger already has handlers to avoid adding
        # multiple.
        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            console_formatter = logging.Formatter(
                fmt="[%(levelname)s] %(message)s",
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        return logger

    def _init_process_logger(self, process_name) -> logging.FileHandler:
        """
        Setup a new file handler for each process and overwrite the log
        file.
        """
        file_handler = logging.FileHandler(f"{process_name}.log", mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        return file_handler

    @staticmethod
    def process(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> None:

            # Setup a new log file for this process
            file_handler = self._init_process_logger(
                f"{self.dump_dir}/{func.__name__}_output"
            )
            msg = self.pretty_print(
                f"Starting process: {func.__name__}",
                do_print=False,
                return_text=True
            )
            self.logger.info(msg)

            # Redirect stdout to capture print statements in real time
            original_stdout = sys.stdout  # Save original stdout
            sys.stdout = LoggerWriter(self.logger, logging.INFO)

            try:
                func(self, *args, **kwargs)
                self.logger.info(
                    f"Process {func.__name__} completed successfully."
                )
            except Exception as e:
                self.logger.error(
                    "\nError occurred in the "
                    f"'{func.__name__}' process:\n{e}"
                )
                # traceback.print_exception(e)
                raise
            finally:
                # Restore original stdout and remove file handler
                sys.stdout = original_stdout
                self.logger.removeHandler(file_handler)
                file_handler.close()
        return wrapper

    def _unwrap_logs(self) -> None:
        """Bypass wrapping when printing logs."""
        sys.stdout = LoggerWriter(self.logger, logging.INFO, wrap=False)

    def _wrap_logs(self) -> None:
        """Enable wrapping."""
        sys.stdout = LoggerWriter(self.logger, logging.INFO, wrap=True)

    def submit_job(self, job_file: str, working_dir: str) -> tuple[str, str]:
        """Submit a job to SLURM as a subprocess."""

        # Set up signal handler for keyboard interrupt (Ctrl + C)
        signal.signal(
            signal.SIGINT, lambda sig, frame: self._handle_interrupt(job_id)
        )

        cmd = f"sbatch {job_file}"
        try:
            with subprocess.Popen(
                    ["bash", "-l", "-c", cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=working_dir,  # Change to this directory first
                    text=True,  # Ensures stdout/stderr are str, not bytes
                    # shell=True,
                    env=os.environ.copy()
            ) as proc:
                stdout, stderr = proc.communicate()

                # Check for errors based on the return code
                if proc.returncode != 0:
                    # An error occurred, log the stderr output
                    self.logger.error(
                        f"Error submitting job. Command returned: {stderr}"
                    )
                    raise subprocess.CalledProcessError(
                        proc.returncode,
                        proc.args,
                        output=stdout,
                        stderr=stderr
                    )

                # Extract job ID from the output
                job_id = self._get_job_id(stdout)
                if job_id:
                    self.logger.info(
                        f"Job submitted successfully. Job ID: {job_id}"
                    )
                    output_file = f"slurm-{job_id}.out"
                    return job_id, os.path.join(working_dir, output_file)
                else:
                    raise ValueError(
                        "Failed to extract job ID from sbatch output."
                    )

        except subprocess.CalledProcessError as e:
            # Log the error if the job submission fails
            self.logger.error(
                f"Subprocess failed with return code {e.returncode}: "
                f"{e.stderr}"
            )
            raise e

    def _get_job_id(self, stdout: str) -> str:
        """Extract the job ID from sbatch output."""
        for line in stdout.splitlines():
            if "Submitted batch job" in line:
                # The job ID is the last part of the line
                return line.split()[-1]
        return None

    def is_job_running(self, job_id: str) -> bool:
        """Check if the job is still running using squeue."""
        result = subprocess.run(
            ["squeue", "--job", job_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            self.logger.error(f"squeue error: {result.stderr}")
            raise subprocess.CalledProcessError(
                result.returncode,
                result.args,
                output=result.stdout,
                stderr=result.stderr
            )
        return job_id in result.stdout  # Job is running if job_id is found

    def stream_job_output(self, job_id: str, output_file: str) -> None:
        """Stream the job output in real time from the SLURM output file."""
        try:
            self.logger.info("Waiting for job output file...")

            # Wait until the output file is created (check every 2 seconds)
            while not os.path.exists(output_file):
                if self.interrupted:
                    self.logger.info(
                        "Job monitoring interrupted before file creation."
                    )
                    return
                time.sleep(0.5)

            self.logger.info(f"Streaming job output from {output_file}:\n\n")

            # Keep trying to read the output file until the job is done
            with open(output_file, "r") as f:
                while not self.interrupted:
                    # Check if the job is still in the queue before reading
                    if not self.is_job_running(job_id):
                        self.logger.info(
                            f"\n\nJob {job_id} is no longer running. "
                            "Stopping output streaming."
                        )
                        break

                    line = f.readline()
                    if line:
                        self.logger.job(line.strip())
                    else:
                        time.sleep(2)  # Sleep briefly before checking again

        except FileNotFoundError:
            self.logger.error(f"Output file {output_file} not found.")
            raise

    def monitor_job(self, job_id: str, output_file: str) -> None:
        """Monitor the job status using squeue and sacct."""
        try:
            # Start monitoring the job and streaming output
            while not self.interrupted:
                if not self.is_job_running(job_id):
                    self.logger.info("Checking final status...")
                    break

                # Job is still running, stream the output file
                # self.logger.info(f"Job {job_id} is running...")
                self.stream_job_output(job_id, output_file)

            # After job finishes, check final status
            if not self.interrupted:
                self.check_job_status(job_id)

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error monitoring job: {e.stderr}")
            raise

    def check_job_status(self, job_id: str) -> None:
        """Check the final job status using sacct."""
        try:
            result = subprocess.run(
                [
                    "sacct", "-j", job_id,
                    "--format=JobID,State,ExitCode", "--noheader"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode != 0:
                self.logger.error(f"sacct error: {result.stderr}")
                raise subprocess.CalledProcessError(
                    result.returncode,
                    result.args,
                    output=result.stdout,
                    stderr=result.stderr
                )

            # Parse the sacct output to check the job's final state
            for line in result.stdout.splitlines():
                if job_id in line:
                    state, exit_code = self._parse_sacct_output(line)
                    self.logger.info(
                        f"Job {job_id} finished with state: {state} and exit "
                        f"code: {exit_code}"
                    )

                    if state == "COMPLETED":
                        self.logger.info(
                            f"Job {job_id} completed successfully."
                        )
                    else:
                        self.logger.error(
                            f"Job {job_id} failed with state: {state} and exit"
                            f" code: {exit_code}"
                        )
                        raise JobFailedError
                        # Shall we raise an error here?
                    return

            # If no matching job ID is found in sacct
            self.logger.error(f"Failed to find job {job_id} in sacct output.")
            raise ValueError(f"Job {job_id} not found in sacct.")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error checking job status: {e.stderr}")
            raise

    def _parse_sacct_output(self, line: str) -> tuple[str, str]:
        """Parse the sacct output line and return the state and exit code."""
        parts = line.split()
        if len(parts) >= 3:
            state = parts[1]
            exit_code = parts[2]
            return state, exit_code
        return None, None

    def cancel_job(self, job_id: str) -> None:
        """Cancel the SLURM job using scancel."""
        try:
            self.logger.info(f"\n\nCancelling job {job_id}...")
            result = subprocess.run(
                ["scancel", job_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode != 0:
                self.logger.error(
                    f"Failed to cancel job {job_id}: {result.stderr}"
                )

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error cancelling job: {e.stderr}")
            raise

    def _handle_interrupt(self, job_id: str) -> None:
        """
        Handle a keyboard interrupt (Ctrl + C) by cancelling the
        job.
        """
        self.interrupted = True  # Set flag to interrupt monitoring
        self.cancel_job(job_id)
        # self.logger.info(f"Job {job_id} was cancelled by the user.")
        raise JobCancelledError(
            "Keyboard interruption. "
            f"Job {job_id} was cancelled by the user."
        )

    def load_parameters(
            self,
            file_path: str = None
    ) -> dict:
        """
        Load the parameters from the configuration files.
        """
        if file_path is None:
            file_path = self.param_file_path

        with open(file_path, "r", encoding="utf8") as file:
            params = yaml.safe_load(
                file,
                Loader=yaml.FullLoader
            )
        return params

    @staticmethod
    def pretty_print(
            text: str,
            max_char_per_line: int = 79,
            do_print: bool = True,
            return_text: bool = False
    ) -> None | str:
        """Print text with a frame of stars."""
        pretty_text = "\n".join(
            [
                "",
                "*" * (max_char_per_line),
                *[
                    f"* {w[::-1].center(max_char_per_line-4)[::-1]} *"
                    for w in textwrap.wrap(text, width=max_char_per_line - 4)
                ],
                "*" * max_char_per_line,
                "",
            ]
        )
        if do_print:
            print(pretty_text)
        if return_text:
            return pretty_text
        return None

