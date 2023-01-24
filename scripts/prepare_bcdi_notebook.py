#!/usr/bin/env python3


import os
import shutil
import sys

helptext =  (
    f"Usage: {sys.argv[0]} path/to/destination.ipynb"
)

if __name__ == "__main__":
    notebook_path = os.path.dirname(__file__) + "/analyze_bcdi_data.ipynb"
    for arg in sys.argv[1:]:
        if not arg.endswith(".ipynb"):
            arg += ".ipynb"
        print(arg, notebook_path)
        if os.path.exists(os.path.dirname(arg)):
            shutil.copy(notebook_path, arg)
        else:
            shutil.copy(notebook_path, os.getcwd() + "/" + arg)
            
    