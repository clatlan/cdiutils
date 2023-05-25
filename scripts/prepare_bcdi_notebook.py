#!/usr/bin/env python3


import os
import shutil
import sys

import cdiutils


PATH, NAME = os.path.split(sys.argv[0])
helptext = (
    f"Usage: {NAME} path/to/destination.ipynb"
)

if __name__ == "__main__":
    notebook_path = os.path.abspath(
        os.path.dirname(cdiutils.__file__) 
        + "/examples/analyze_bcdi_data.ipynb"
    )
    if len(sys.argv) < 2:
        raise ValueError(helptext)
    for arg in sys.argv[1:]:
        if arg == ".":
            output_path = os.getcwd() + "/analyze_bcdi_data.ipynb"
        elif "/" in arg:
            path, name = os.path.split(arg)
            if not name.endswith(".ipynb"):
                name += ".ipynb"
            if not os.path.exists(os.path.dirname(path)):
                path = os.getcwd() + "/" + path 
                if not os.path.exists(os.path.dirname(path)):
                    raise FileExistsError(f"Diretory {path} does not exist.")
            output_path =  path + "/" + name
        else:
             if not arg.endswith(".ipynb"):
                 output_path = os.getcwd() + "/" + arg + ".ipynb"

    shutil.copy(notebook_path, output_path)