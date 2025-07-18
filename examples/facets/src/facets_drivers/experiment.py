#!/usr/bin/env python

import os
import cdiutils
import pickle
from ewokscore import Task

# Backups functions functions.
def rick_to_pickle(pipeline, directory, filename):
    output_path = os.path.join(directory, filename)
    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)
    return output_path

def pickle_to_rick(path):
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

class InitBcdiPipeline(Task, input_names=["parameters", "scratch_directory"], output_names=["pipeline", "scratch_directory"]):
    def run(self):
        params = self.inputs.parameters
        scratch_directory = self.inputs.scratch_directory
        os.makedirs(scratch_directory, exist_ok=True)
        params["dump_dir"] = scratch_directory

        pipeline = cdiutils.BcdiPipeline(params=params)

        output_path = rick_to_pickle(pipeline, scratch_directory, "pipeline.pkl")
        self.outputs.pipeline = pipeline
        self.outputs.scratch_directory = scratch_directory

class BcdiPreprocess(Task, input_names=["pipeline", "scratch_directory"], output_names=["pipeline", "scratch_directory"]):
    def run(self):
        scratch_directory = self.inputs.scratch_directory
        pipeline = self.inputs.pipeline

        pipeline.preprocess()

        output_path = rick_to_pickle(pipeline, scratch_directory, "pipeline.pkl")
        self.outputs.pipeline = pipeline
        self.outputs.scratch_directory = scratch_directory

class PhaseRetrieval(Task, input_names=["pipeline", "scratch_directory"], output_names=["pipeline", "scratch_directory"]):
    def run(self):
        pipeline = self.inputs.pipeline
        scratch_directory = self.inputs.scratch_directory

        pipeline.phase_retrieval(
            clear_former_results=True,
            nb_run=10,
            nb_run_keep=10,
        )

        output_path = rick_to_pickle(pipeline, scratch_directory, "pipeline.pkl")
        self.outputs.pipeline = pipeline
        self.outputs.scratch_directory = scratch_directory

class PhaseAnalysis(Task, input_names=["pipeline", "scratch_directory"], output_names=["pipeline", "scratch_directory"]):
    def run(self):
        pipeline = self.inputs.pipeline
        scratch_directory = self.inputs.scratch_directory

        pipeline.analyse_phasing_results(sorting_criterion="mean_to_max")

        output_path = rick_to_pickle(pipeline, scratch_directory, "pipeline.pkl")
        self.outputs.pipeline = pipeline
        self.outputs.scratch_directory = scratch_directory

class SelectingBest(Task, input_names=["pipeline", "scratch_directory"], output_names=["pipeline", "scratch_directory"]):
    def run(self):
        pipeline = self.inputs.pipeline
        scratch_directory = self.inputs.scratch_directory

        pipeline.select_best_candidates(3)

        output_path = rick_to_pickle(pipeline, scratch_directory, "pipeline.pkl")
        self.outputs.pipeline = pipeline
        self.outputs.scratch_directory = scratch_directory

class ModeDecomposition(Task, input_names=["pipeline", "scratch_directory"], output_names=["pipeline", "scratch_directory"]):
    def run(self):
        pipeline = self.inputs.pipeline
        scratch_directory = self.inputs.scratch_directory

        pipeline.mode_decomposition()

        output_path = rick_to_pickle(pipeline, scratch_directory, "pipeline.pkl")
        self.outputs.pipeline = pipeline
        self.outputs.scratch_directory = scratch_directory

class PostProcess(Task, input_names=["pipeline", "scratch_directory"], output_names=["pipeline", "scratch_directory"]):
    def run(self):
        pipeline = self.inputs.pipeline
        scratch_directory = self.inputs.scratch_directory

        pipeline.postprocess(
            isosurface=0.3,   # threshold for isosurface
            voxel_size=None,  # use default voxel size if not provided
            flip=False        # whether to flip the reconstruction if you got the twin image (enantiomorph)
        )

        output_path = rick_to_pickle(pipeline, scratch_directory, "pipeline.pkl")
        self.outputs.pipeline = pipeline
        self.outputs.scratch_directory = scratch_directory