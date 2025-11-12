"""
This is a simple script to handle the creation of the Jupyter
notebooks required for running the detector calibration using the cdiutils
package.
"""

import argparse
import os
import shutil
from importlib import resources


def get_templates_path():
    """Locate the templates folder bundled with the package."""
    return str(resources.files("cdiutils").joinpath("templates"))


def main() -> None:
    helptext = "try -h or --help to see usage."

    parser = argparse.ArgumentParser(
        prog="prepare_detector_calibration",
        description=(
            "Prepare the notebooks required for the detector calibration."
        ),
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="The directory path where the notebooks will be created.",
    )
    parser.add_argument(
        "-f",
        "--force",
        default=False,
        action="store_true",
        help=(
            "Whether or not to force the creation of the files if they "
            "already exist."
        ),
    )

    args = parser.parse_args()

    # Locate the examples directory
    templates_dir = get_templates_path()

    # Path to the notebook in the examples directory
    notebook_path = os.path.join(templates_dir, "detector_calibration.ipynb")

    if not os.path.exists(notebook_path):
        raise FileNotFoundError(
            f"Notebook not found. Expected location: {notebook_path}\n"
            + helptext
        )

    path = os.getcwd()
    if args.path:
        path = args.path
    if not path.endswith("/"):
        path += "/"

    if not os.path.exists(os.path.dirname(path)):
        raise FileNotFoundError(
            f"Directory {path} does not exist.\n" + helptext
        )

    dest = os.path.join(path, os.path.basename(notebook_path))

    if os.path.isfile(dest):
        if args.force:
            print(
                f"Force file creation requested, file '{dest}' "
                "will be overwritten."
            )
            shutil.copy(notebook_path, dest)
        else:
            raise FileExistsError(
                f"File {dest} already exists. Rename the existing file or "
                "use -f or --force option to force creation."
            )
    else:
        shutil.copy(notebook_path, dest)

    print(
        f"Notebook copied to {path}.\n"
        "You can now run the notebook using Jupyter Notebook or Jupyter Lab."
    )


if __name__ == "__main__":
    main()
