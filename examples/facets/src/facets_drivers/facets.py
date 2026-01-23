#!/usr/bin/env python

import os
from ewokscore import Task
import math
import facets.analyse as analyse

import vtk
from IPython.display import Image, display

class FullCircleAnalysis(Task, input_names=["scratch_directory", "hkl"], optional_input_names=["vti_filepath", "xyz_filepath"], output_names=["result"]):
    def run(self):
        scratch_directory = self.inputs.scratch_directory
        vti_filepath = self.get_input_value("vti_filepath", None)
        xyz_filepath = self.get_input_value("xyz_filepath", None)
        os.makedirs(scratch_directory, exist_ok=True)
        os.chdir(scratch_directory)
        if vti_filepath is not None:
            arguments = ["--hkl"] + [str(_) for _ in self.inputs.hkl] + ["--exp-qnorm", "3.2",
                         #"--exp-voxel-size", "5", "5", "5",
                         #"--exp-amp-key", "amp",
                         #"--final-shape", "64", "64", "64",
                         "--final-shape", "64", "64", "200",
                         "--phase-range", f"{math.pi/12}",
                         "--strain-range", "5e-4",
                         "--exp-data", vti_filepath]

        if xyz_filepath is not None:
            arguments = ["--hkl"] + [str(_) for _ in self.inputs.hkl] + [
                         #"--nstep", "800",
                         "--strain-range", "5e-4",          # remove to match Corentin’s
                         "--phase-range", f"{math.pi/12}",  # remove to match Corentin’s
                         "--input-file", xyz_filepath]

        result = analyse.main(arguments)
        self.outputs.result = result
