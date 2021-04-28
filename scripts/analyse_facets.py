import numpy as np

from .cdiutils.utils import get_data_from_vtk

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("-f", "--file", required=True, type=str,
                    help="file to read")
    args = vars(ap.parse_args())
    get_data_from_vtk(args["file"])
