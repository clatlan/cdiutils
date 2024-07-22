#!/usr/bin/env python3

import argparse
import os
import shutil

import cdiutils


"""
    This is a simple script to handle the creation of the Jupyter
    notebooks required for BCDI analysis using cdiutils package.

    Raises:
        FileNotFoundError: If the provided directory does not exist.
        FileExistsError: If one of the file already exists and the
        --force option is not provided.
"""


if __name__ == "__main__":
    helptext = "try -h or --help to see usage."

    parser = argparse.ArgumentParser(
        prog="prepare_bcdi_notebook.py",
        description="Prepare the notebooks required for BCDI analysis.",
    )
    parser.add_argument(
        "-p", "--path",
        type=str,
        help="the directory path where the notebooks will be created."
    )
    parser.add_argument(
        "-f", "--force",
        default=False,
        action="store_true",
        help=(
            "whether or not to force the creations of the files if they "
            "already exist."
        )
    )

    args = parser.parse_args()
    print(args)

    notebook_path = os.path.abspath(
        os.path.dirname(cdiutils.__file__)
        + "/examples/bcdi_pipeline.ipynb"
    )
    step_by_step_notebook_path = os.path.abspath(
        os.path.dirname(cdiutils.__file__)
        + "/examples/step_by_step_bcdi_analysis.ipynb"
    )

    path = os.getcwd()
    if args.path:
        path = args.path
    if not path.endswith("/"):
        path += "/"

    if not os.path.exists(os.path.dirname(path)):
        raise FileNotFoundError(
            f"Diretory {path} does not exist.\n" + helptext
        )

    files = {
        notebook_path: path + os.path.split(notebook_path)[1],
        step_by_step_notebook_path: (
            path
            + os.path.split(step_by_step_notebook_path)[1]
        )
    }

    for source, dest in files.items():
        if os.path.isfile(dest):
            if args.force:
                print(
                    f"Force file creation requested, file '{dest}' "
                    "will be overwritten."
                )
                shutil.copy(source, dest)
            else:
                raise FileExistsError(
                    f"File {dest} already exists, rename the existing file or "
                    "use -f or --force option to force creation."
                )
        else:
            shutil.copy(source, dest)
