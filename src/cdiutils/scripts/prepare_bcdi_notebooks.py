"""
This is a simple script to handle the creation of the Jupyter
notebooks required for BCDI analysis using cdiutils package.
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
        prog="prepare_bcdi_notebooks",
        description="Prepare the notebooks required for BCDI analysis.",
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

    # Update paths to notebooks in the examples directory
    bcdi_notebook = os.path.join(templates_dir, "bcdi_pipeline.ipynb")
    step_by_step_notebook = os.path.join(
        templates_dir, "step_by_step_bcdi_analysis.ipynb"
    )

    if not os.path.exists(bcdi_notebook) or not os.path.exists(
        step_by_step_notebook
    ):
        raise FileNotFoundError(
            "Examples notebooks not found. "
            f"Expected location: {templates_dir}\n" + helptext
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

    files = {
        bcdi_notebook: os.path.join(path, os.path.basename(bcdi_notebook)),
        step_by_step_notebook: os.path.join(
            path, os.path.basename(step_by_step_notebook)
        ),
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
                    f"File {dest} already exists. Rename the existing file or "
                    "use -f or --force option to force creation."
                )
        else:
            shutil.copy(source, dest)

    print(
        f"Notebooks copied to {path}.\n"
        "You can now run the notebooks using Jupyter Notebook or Jupyter Lab."
    )


if __name__ == "__main__":
    main()
