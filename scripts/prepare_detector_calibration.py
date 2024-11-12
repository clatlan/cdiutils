#!/usr/bin/env python3


"""
This is a simple script to handle the creation of the Jupyter
notebooks required for running the detector calibration using cdiutils
package.
"""

import argparse
import os
import shutil

import cdiutils


def main() -> None:
    helptext = "try -h or --help to see usage."

    parser = argparse.ArgumentParser(
        prog="prepare_detector_calibration.py",
        description=(
            "Prepare the notebooks required for the detector calibration."
        ),
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

    notebook_path = os.path.abspath(
        os.path.dirname(cdiutils.__file__)
        + "/examples/detector_calibration.ipynb"
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

    dest = path + os.path.split(notebook_path)[1],

    if os.path.isfile(dest):
        if args.force:
            print(
                f"Force file creation requested, file '{dest}' "
                "will be overwritten."
            )
            shutil.copy(notebook_path, dest)
        else:
            raise FileExistsError(
                f"File {dest} already exists, rename the existing file or "
                "use -f or --force option to force creation."
            )
    else:
        shutil.copy(notebook_path, dest)


if __name__ == "__main__":
    main()
