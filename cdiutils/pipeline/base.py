
from abc import ABC
from functools import wraps
import logging
import os
from typing import Callable
import sys
# import traceback
import textwrap


import numpy as np
import yaml

from cdiutils.plot.formatting import update_plot_params


class LoggerWriter:
    """
    Custom stream to send stdout (print statements) directly to
    logger in real-time.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            # Only log if there's something to log
            # (ignores empty messages)
            self.logger.log(self.level, message.strip())

    def flush(self):
        """
        Flush method is needed for compatibility with `sys.stdout`.
        """
        pass


def process(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> None:

        # Setup a new log file for this process
        file_handler = self._init_process_logger(
            f"{self.dump_dir}/{func.__name__}_output"
        )
        self.logger.info(f"Starting process: {func.__name__}")

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


def pretty_print(text: str, max_char_per_line: int = 79) -> None:
    """Print text with a frame of stars."""

    pretty_text = "\n".join(
        [
            "",
            "*" * (max_char_per_line + 4),
            *[
                f"* {w[::-1].center(max_char_per_line)[::-1]} *"
                for w in textwrap.wrap(text, width=max_char_per_line)
            ],
            "*" * (max_char_per_line + 4),
            "",
        ]
    )
    print(pretty_text)


class Pipeline(ABC):
    def __init__(
            self,
            param_file_path: str = None,
            params: dict = None
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
            # self.params = self.load_parameters()

        self.dump_dir = self.params["dump_dir"]

        # Create the dump directory
        self._make_dump_dir()
        self.logger = self._init_logger()

        # Set the printoptions legacy to 1.21, otherwise types are printed.
        np.set_printoptions(legacy="1.21")

        # update the plot parameters
        update_plot_params()

    def _make_dump_dir(self) -> None:
        dump_dir = self.params["dump_dir"]
        if os.path.isdir(dump_dir):
            print(
                "\nDump directory already exists, results will be "
                f"saved in:\n{dump_dir}."
            )
        else:
            print(
                f"Creating the dump directory at: {dump_dir}")
            os.makedirs(
                dump_dir,
                exist_ok=True
            )

    def _init_logger(self) -> logging.Logger:
        logger = logging.getLogger("PipelineLogger")

        # Check if the logger already has handlers to avoid adding multiple
        # if not logger.hasHandlers():
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
            params = yaml.load(
                file,
                Loader=yaml.FullLoader
            )
        return params
