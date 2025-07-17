#!/usr/bin/env python

import os
import cdiutils
import pickle
from ewokscore import Task

class DebugClass(Task, input_names=["path"], output_names=["yop"]):
    def run(self):
        print(self.inputs.path)
        self.outputs.yop = self.inputs.path

def pickle_to_rick(path):
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

def rick_to_pickle(pipeline, dump_dir, filename):
    output_path = os.path.join(dump_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)
    return output_path

class InitBcdiPipeline(Task, input_names=["parameters", "output_directory"], output_names=["pipeline_path"]):
    def run(self):
        params = self.inputs.parameters
        dump_dir = self.inputs.output_directory
        os.makedirs(dump_dir, exist_ok=True)
        params["dump_dir"] = dump_dir

        pipeline = cdiutils.BcdiPipeline(params=params)

        output_path = rick_to_pickle(pipeline, dump_dir, "0_init_pipeline.pkl")
        self.outputs.pipeline_path = output_path

class BcdiPreprocess(Task, input_names=["pipeline_path", "output_directory"], output_names=["pipeline_path"]):
    def run(self):
        input_path = self.inputs.pipeline_path
        dump_dir = self.inputs.output_directory
        pipeline = pickle_to_rick(input_path)

        pipeline.preprocess()

        output_path = rick_to_pickle(pipeline, dump_dir, "1_preprocess.pkl")
        self.outputs.pipeline_path = output_path

class PhaseRetrieval(Task, input_names=["pipeline_path", "output_directory"], output_names=["pipeline_path"]):
    def run(self):
        input_path = self.inputs.pipeline_path
        dump_dir = self.inputs.output_directory
        pipeline = pickle_to_rick(input_path)

        pipeline.phase_retrieval(
            clear_former_results=True,
            nb_run=10,
            nb_run_keep=10,
        )

        output_path = rick_to_pickle(pipeline, dump_dir, "2_phase_retrieval.pkl")
        self.outputs.pipeline_path = output_path

class PhaseAnalysis(Task, input_names=["pipeline_path", "output_directory"], output_names=["pipeline_path"]):
    def run(self):
        input_path = self.inputs.pipeline_path
        dump_dir = self.inputs.output_directory
        pipeline = pickle_to_rick(input_path)

        pipeline.analyse_phasing_results(sorting_criterion="mean_to_max")

        output_path = rick_to_pickle(pipeline, dump_dir, "3_phase_analysis.pkl")
        self.outputs.pipeline_path = output_path

class SelectingBest(Task, input_names=["pipeline_path", "output_directory"], output_names=["pipeline_path"]):
    def run(self):
        input_path = self.inputs.pipeline_path
        dump_dir = self.inputs.output_directory
        pipeline = pickle_to_rick(input_path)

        pipeline.select_best_candidates(3)

        output_path = rick_to_pickle(pipeline, dump_dir, "4_best_candidates.pkl")
        self.outputs.pipeline_path = output_path

class ModeDecomposition(Task, input_names=["pipeline_path", "output_directory"], output_names=["pipeline_path"]):
    def run(self):
        input_path = self.inputs.pipeline_path
        dump_dir = self.inputs.output_directory
        pipeline = pickle_to_rick(input_path)

        pipeline.mode_decomposition()

        output_path = rick_to_pickle(pipeline, dump_dir, "5_mode_decompose.pkl")
        self.outputs.pipeline_path = output_path

class PostProcess(Task, input_names=["pipeline_path", "output_directory"], output_names=["pipeline_path"]):
    def run(self):
        input_path = self.inputs.pipeline_path
        dump_dir = self.inputs.output_directory
        pipeline = pickle_to_rick(input_path)

        pipeline.postprocess(
            isosurface=0.3,   # threshold for isosurface
            voxel_size=None,  # use default voxel size if not provided
            flip=False        # whether to flip the reconstruction if you got the twin image (enantiomorph)
        )

        output_path = rick_to_pickle(pipeline, dump_dir, "6_postprocess.pkl")
        self.outputs.pipeline_path = output_path
