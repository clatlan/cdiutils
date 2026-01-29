import logging
import os
import signal
import subprocess
import sys
import textwrap
import time
from abc import ABC
from functools import wraps
from typing import Callable

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
    Custom stream redirecting stdout to logger in real-time.

    Captures print statements and routes them through the logging
    system with optional line wrapping at 79 characters.

    Args:
        logger (logging.Logger): target logger instance.
        level (int): logging level (e.g., logging.INFO).
        wrap (bool): enable line wrapping at 79 chars. Defaults to
            True.
    """

    def __init__(
        self, logger: logging.Logger, level: int, wrap: bool = True
    ) -> None:
        self.logger = logger
        self.level = level
        self.wrap = wrap

    def write(self, message: str) -> None:
        """
        Write message to logger with optional wrapping.

        Args:
            message (str): message to log.
        """
        if message.strip():
            # only log non-empty messages
            if self.wrap:
                wrapped_message = textwrap.fill(message.strip(), width=79)
            else:
                wrapped_message = message.strip()
            self.logger.log(self.level, "\n" + wrapped_message + "\n")

    def flush(self) -> None:
        """
        No-op flush method for sys.stdout compatibility.

        Required by the file-like object interface but performs no
        operation for logger streams.
        """
        pass


class JobCancelledError(Exception):
    """
    Exception raised when user cancels a SLURM job.

    Triggered by keyboard interrupts (Ctrl+C) during job monitoring.
    """


class JobFailedError(Exception):
    """
    Exception raised when a SLURM job fails.

    Indicates non-zero exit codes or failed job states detected via
    sacct.
    """


class Pipeline(ABC):
    """
    Abstract base class for CDI data processing pipelines.

    Provides infrastructure for parameter management, logging, job
    submission (SLURM), and subprocess execution. Not intended for
    direct instantiation—subclass for specific applications.

    Args:
        params (dict, optional): parameter dictionary. Defaults to
            None.
        param_file_path (str, optional): path to YAML parameter file.
            Defaults to None.

    Raises:
        ValueError: if neither params nor param_file_path is provided.
    """

    def __init__(
        self, params: dict = None, param_file_path: str = None
    ) -> None:
        """
        Initialise Pipeline with parameters from dict or file.

        Args:
            params (dict, optional): parameter dictionary. Defaults to
                None.
            param_file_path (str, optional): path to YAML parameter
                file. Defaults to None.

        Raises:
            ValueError: if neither params nor param_file_path is
                provided.
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
        """
        Create output directory specified in params['dump_dir'].

        Raises:
            ValueError: if dump_dir parameter is None.
        """
        dump_dir = self.params["dump_dir"]
        if dump_dir is None:
            raise ValueError("dump_dir parameter must be set.")
        if os.path.isdir(dump_dir):
            print(
                "\nDump directory already exists, results will be "
                f"saved in:\n{dump_dir}."
            )
        else:
            print(f"Creating the dump directory at: {dump_dir}")
            os.makedirs(dump_dir, exist_ok=True)

    @staticmethod
    def _init_logger() -> logging.Logger:
        """
        Initialise and configure logger for pipeline processes.

        Removes existing root handlers (e.g., Jupyter defaults) and
        sets up console logging at INFO level.

        Returns:
            logging.Logger: configured logger instance.
        """
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

    def _init_process_logger(self, process_name: str) -> logging.FileHandler:
        """
        Initialise file handler for process-specific logging.

        Creates a new log file (overwriting any existing one) and
        attaches a file handler to the logger with DEBUG level and
        timestamped formatting.

        Args:
            process_name (str): base name for log file (without
                extension).

        Returns:
            logging.FileHandler: configured file handler attached to
                logger.
        """
        file_handler = logging.FileHandler(f"{process_name}.log", mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        return file_handler

    @staticmethod
    def process(func: Callable) -> Callable:
        """
        Decorate pipeline methods to add logging and error handling.

        Wraps process methods with file logging, stdout redirection,
        and structured error reporting. Creates process-specific log
        files in dump_dir with format {func_name}_output.log.

        Args:
            func (Callable): pipeline method to decorate.

        Returns:
            Callable: wrapped function with logging infrastructure.

        Raises:
            Exception: re-raises any exception from decorated function
                after logging.

        Notes:
            Temporarily redirects sys.stdout to logger during execution
            to capture print statements. Original stdout is always
            restored in finally block.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs) -> None:
            # Setup a new log file for this process
            file_handler = self._init_process_logger(
                f"{self.dump_dir}/{func.__name__}_output"
            )
            msg = self.pretty_print(
                f"Starting process: {func.__name__}",
                do_print=False,
                return_text=True,
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
                    f"\nError occurred in the '{func.__name__}' process:\n{e}"
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
        """
        Disable line wrapping for logger output.

        Configures stdout redirection to bypass 79-character wrapping
        for cases requiring full-width output (e.g., tables, progress
        bars).
        """
        sys.stdout = LoggerWriter(self.logger, logging.INFO, wrap=False)

    def _wrap_logs(self) -> None:
        """
        Enable line wrapping for logger output.

        Configures stdout redirection to wrap lines at 79 characters
        for standard logging output.
        """
        sys.stdout = LoggerWriter(self.logger, logging.INFO, wrap=True)

    def _subprocess_run(
        self, cmd: str | list[str]
    ) -> subprocess.CompletedProcess:
        """
        Execute subprocess command with error handling.

        Runs command with captured stdout/stderr and validates return
        code. Logs errors and raises CalledProcessError on failure.

        Args:
            cmd (str | list[str]): command string or argument list.

        Returns:
            subprocess.CompletedProcess: completed process with
                stdout/stderr.

        Raises:
            subprocess.CalledProcessError: if command returns non-zero
                exit code.
        """
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            self.logger.error(f"Command {cmd} failed: {result.stderr}")
            raise subprocess.CalledProcessError(
                result.returncode,
                result.args,
                output=result.stdout,
                stderr=result.stderr,
            )
        return result

    def submit_job(self, job_file: str, working_dir: str) -> tuple[str, str]:
        """
        Submit SLURM job and return job ID with output file path.

        Executes sbatch command in bash login shell to ensure proper
        environment loading. Sets up keyboard interrupt handler for
        job cancellation.

        Args:
            job_file (str): path to SLURM batch script.
            working_dir (str): directory to execute sbatch from.

        Returns:
            tuple[str, str]: job ID and absolute path to output file
                (slurm-{job_id}.out).

        Raises:
            subprocess.CalledProcessError: if sbatch command fails.
            ValueError: if job ID cannot be extracted from sbatch
                output.

        Notes:
            Registers SIGINT handler that calls _handle_interrupt with
            job_id when Ctrl+C is pressed.
        """
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
                env=os.environ.copy(),
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
                        stderr=stderr,
                    )

                # Extract job ID from the output
                job_id = self._get_job_id(stdout)
                if job_id:
                    self.logger.info(
                        f"Job submitted successfully. Job ID: {job_id}"
                    )
                    output_file = f"slurm-{job_id}.out"
                    return job_id, os.path.join(working_dir, output_file)

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

    @staticmethod
    def _get_job_id(stdout: str) -> str:
        """
        Extract SLURM job ID from sbatch output.

        Parses sbatch stdout for line containing 'Submitted batch
        job' and returns the trailing job ID number.

        Args:
            stdout (str): sbatch command output.

        Returns:
            str: job ID as string, or None if not found.

        Examples:
            >>> _get_job_id("Submitted batch job 12345\\n")
            '12345'
        """
        for line in stdout.splitlines():
            if "Submitted batch job" in line:
                return line.split()[-1]  # last element of the line
        return None

    def is_job_running(self, job_id: str) -> bool:
        """
        Check if SLURM job is currently running.

        Queries squeue for job presence. Job is considered running if
        its ID appears in squeue output.

        Args:
            job_id (str): SLURM job ID to check.

        Returns:
            bool: True if job is in queue, False otherwise.

        Raises:
            subprocess.CalledProcessError: if squeue command fails.
        """
        result = self._subprocess_run(["squeue", "--job", job_id])
        return job_id in result.stdout  # Job is running if job_id is found

    def stream_job_output(self, job_id: str, output_file: str) -> None:
        """
        Stream SLURM job output in real-time.

        Waits for output file creation, then continuously reads and
        logs new lines until job stops running or interrupted flag is
        set. Logs at JOB level (custom level between INFO and
        WARNING).

        Args:
            job_id (str): SLURM job ID being monitored.
            output_file (str): path to slurm-{job_id}.out file.

        Raises:
            FileNotFoundError: if output file cannot be accessed after
                creation.

        Notes:
            Checks file existence every 0.5s until found. Polls running
            status and reads new lines with 0.5s interval. Respects
            self.interrupted flag for early termination.
        """
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
                        time.sleep(0.5)  # Sleep briefly before checking again

        except FileNotFoundError:
            self.logger.error(f"Output file {output_file} not found.")
            raise

    def monitor_job(
        self, job_id: str, output_file: str, retries: int = 10, delay: int = 1
    ) -> None:
        """
        Monitor SLURM job and verify final completion status.

        Streams job output in real-time and validates final state via
        sacct after job leaves queue. Retries state check if job shows
        RUNNING but is not in squeue (handles race conditions).

        Args:
            job_id (str): SLURM job ID to monitor.
            output_file (str): path to slurm-{job_id}.out file.
            retries (int): number of sacct retries for lingering
                RUNNING state. Defaults to 10.
            delay (int): seconds between retries. Defaults to 1.

        Raises:
            JobFailedError: if job terminates with FAILED state or
                non-zero exit code.

        Notes:
            Successfully completed jobs have state='COMPLETED' and
            exit_code='0:0'. Other terminal states log a warning but
            do not raise exceptions.
        """
        # Start monitoring the job and streaming output
        while not self.interrupted:
            if not self.is_job_running(job_id):
                self.logger.info("Checking final status...")
                break

            # Job is still running, stream the output file
            self.stream_job_output(job_id, output_file)

        # After job finishes, check final status
        if not self.interrupted:
            state, exit_code = self.get_job_state(job_id)
            attempt = 0
            while state == "RUNNING" and attempt < retries:
                self.logger.info(
                    f"Job {job_id} is still in RUNNING state but not "
                    f"found in queue. Rechecking the state in {delay} "
                    f"second(s)..."
                )
                time.sleep(delay)
                state, exit_code = self.get_job_state(job_id)
                attempt += 1
            if state == "COMPLETED":
                self.logger.info(
                    f"Job {job_id} completed successfully with "
                    f"exit code: {exit_code}"
                )
                return
            elif state == "FAILED":
                raise JobFailedError(
                    f"Job {job_id} failed with exit code: {exit_code}."
                    f"See {output_file} for more details."
                )
            else:
                self.logger.warning(
                    f"Job {job_id} finished with unexpected state: {state}."
                )

    def get_job_state(self, job_id: str) -> tuple[str, str]:
        """
        Retrieve SLURM job state and exit code via sacct.

        Queries sacct for job status information and parses output to
        extract state (e.g., COMPLETED, FAILED, RUNNING) and exit code
        (format: signal:status).

        Args:
            job_id (str): SLURM job ID to query.

        Returns:
            tuple[str, str]: job state and exit code (e.g.,
                ('COMPLETED', '0:0')).

        Raises:
            ValueError: if job ID not found in sacct output.
            subprocess.CalledProcessError: if sacct command fails.
        """
        result = self._subprocess_run(
            [
                "sacct",
                "-j",
                job_id,
                "--format=JobID,State,ExitCode",
                "--noheader",
            ]
        )
        state, exit_code = None, None

        # Parse the sacct output to check the job's final state
        for line in result.stdout.splitlines():
            if job_id in line:
                parts = line.split()
                if len(parts) >= 3:
                    state = parts[1]
                    exit_code = parts[2]
                    break
        if state is None or exit_code is None:
            raise ValueError(f"Job {job_id} not found in sacct.")
        return state, exit_code

    def cancel_job(self, job_id: str) -> None:
        """
        Cancel running SLURM job via scancel.

        Args:
            job_id (str): SLURM job ID to cancel.

        Raises:
            subprocess.CalledProcessError: if scancel command fails.
        """
        try:
            self.logger.info(f"\n\nCancelling job {job_id}...")
            _ = self._subprocess_run(["scancel", job_id])
            self.logger.info(f"Job {job_id} cancelled successfully.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to cancel job {job_id}: {e.stderr}")
            raise

    def _handle_interrupt(self, job_id: str) -> None:
        """
        Handle keyboard interrupt by cancelling job.

        Sets interrupted flag, cancels SLURM job via scancel, and
        raises JobCancelledError to terminate monitoring.

        Args:
            job_id (str): SLURM job ID to cancel.

        Raises:
            JobCancelledError: always raised after job cancellation.

        Notes:
            Called by SIGINT handler registered in submit_job. Sets
            self.interrupted=True to signal monitoring loops.
        """
        self.interrupted = True  # Set flag to interrupt monitoring
        self.cancel_job(job_id)
        raise JobCancelledError(
            f"Keyboard interruption. Job {job_id} was cancelled by the user."
        )

    def load_parameters(self, file_path: str = None) -> dict:
        """
        Load pipeline parameters from YAML configuration file.

        Uses yaml.full_load() to support Python-specific types like
        tuples that are serialised by yaml.dump().

        Args:
            file_path (str, optional): path to YAML parameter file.
                Defaults to None (uses self.param_file_path).

        Returns:
            dict: loaded parameter dictionary.

        Raises:
            FileNotFoundError: if parameter file does not exist.
            yaml.YAMLError: if file contains invalid YAML.
        """
        if file_path is None:
            file_path = self.param_file_path

        with open(file_path, "r", encoding="utf8") as file:
            params = yaml.full_load(file)
        return params

    @staticmethod
    def pretty_print(
        text: str,
        max_char_per_line: int = 79,
        do_print: bool = True,
        return_text: bool = False,
    ) -> None | str:
        """
        Format text with decorative star border.

        Creates a framed message with star borders and centred text
        wrapped to specified line width. Useful for logging section
        headers or important messages.

        Args:
            text (str): text to format.
            max_char_per_line (int): maximum line width including
                border. Defaults to 79.
            do_print (bool): whether to print formatted text. Defaults
                to True.
            return_text (bool): whether to return formatted string.
                Defaults to False.

        Returns:
            None | str: formatted text if return_text=True, else None.

        Examples:
            >>> pretty_print("Hello World", max_char_per_line=30)
            ******************************
            *        Hello World        *
            ******************************
        """
        pretty_text = "\n".join(
            [
                "",
                "*" * (max_char_per_line),
                *[
                    f"* {w[::-1].center(max_char_per_line - 4)[::-1]} *"
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
