#!/usr/bin/env python

import os
import cdiutils
import pickle
import dask.config
import logging
from ewokscore import Task

# --- New Imports & Helpers ---
import matplotlib.pyplot as plt
import io

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def capture_figures_to_bytes():
    """
    Captures all open matplotlib figures, converts them to PNG bytes,
    and closes them.
    Returns:
        list: A list of PNG byte strings.
    """
    figures_as_bytes = []
    fig_nums = plt.get_fignums()
    if fig_nums:
        logging.info(f"Capturing {len(fig_nums)} figure(s) as output data.")
    for num in fig_nums:
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        figures_as_bytes.append(buf.getvalue())
        plt.close(fig)
    return figures_as_bytes

class DisplayImages(Task, input_names=["image_data_list"], optional_input_names=[], output_names=[]):
    """An Ewoks task specifically for displaying image data in Jupyter."""
    def run(self):
        from IPython.display import Image, display
        image_data_list = self.inputs.image_data_list
        if not image_data_list:
            logging.warning("DisplayImages task received no images to display.")
            return
        logging.info(f"Displaying {len(image_data_list)} image(s) in Jupyter.")
        for image_data in image_data_list:
            if image_data:
                display(Image(data=image_data))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def rick_to_pickle(pipeline, directory, filename):
    """Saves the pipeline object to a file."""
    output_path = os.path.join(directory, filename)
    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)
    logging.info(f"Saved checkpoint to: {output_path}")
    return output_path

def pickle_to_rick(path):
    """Loads the pipeline object from a file."""
    logging.info(f"Loading checkpoint from: {path}")
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

def _bcdi_pipeline_close(self):
    """
    Monkey-patched method to explicitly release large data arrays from the
    BcdiPipeline object, allowing for fast garbage collection.
    """
    logger = getattr(self, 'logger', logging.getLogger(__name__))
    logger.info(
        "Closing BcdiPipeline (via monkey patch) and releasing memory."
    )
    self.detector_data = None
    self.cropped_detector_data = None
    self.orthogonalised_intensity = None
    self.mask = None
    self.reconstruction = None
    self.structural_props = None
    self.phasing_results = []
    self.converter = None
    self.result_analyser = None
cdiutils.BcdiPipeline.close = _bcdi_pipeline_close

# --- MODIFIED Ewoks Tasks ---

# Note the addition of "figures" to the output_names list
class InitBcdiPipeline(Task, input_names=["parameters", "scratch_directory"], output_names=["pipeline", "scratch_directory", "figures"]):
    def run(self):
        params = self.inputs.parameters
        scratch_directory = self.inputs.scratch_directory
        os.makedirs(scratch_directory, exist_ok=True)
        params["dump_dir"] = scratch_directory
        pipeline = cdiutils.BcdiPipeline(params=params)
        self.outputs.pipeline = pipeline
        self.outputs.scratch_directory = scratch_directory
        self.outputs.figures = capture_figures_to_bytes() # Capture any figures

class BcdiPreprocess(Task, input_names=["pipeline", "scratch_directory"], output_names=["pipeline", "scratch_directory", "figures"]):
    def run(self):
        pipeline = self.inputs.pipeline
        pipeline.preprocess()
        self.outputs.pipeline = pipeline
        self.outputs.scratch_directory = self.inputs.scratch_directory
        self.outputs.figures = capture_figures_to_bytes()

class PhaseRetrieval(Task, input_names=["pipeline", "scratch_directory"], output_names=["pipeline", "scratch_directory", "figures"]):
    def run(self):
        pipeline = self.inputs.pipeline
        pipeline.phase_retrieval(clear_former_results=True, nb_run=10, nb_run_keep=10)
        self.outputs.pipeline = pipeline
        self.outputs.scratch_directory = self.inputs.scratch_directory
        self.outputs.figures = capture_figures_to_bytes()

class PhaseAnalysis(Task, input_names=["pipeline", "scratch_directory"], output_names=["pipeline", "scratch_directory", "figures"]):
    def run(self):
        pipeline = self.inputs.pipeline
        pipeline.analyse_phasing_results(sorting_criterion="mean_to_max")
        self.outputs.pipeline = pipeline
        self.outputs.scratch_directory = self.inputs.scratch_directory
        self.outputs.figures = capture_figures_to_bytes()

class SelectingBest(Task, input_names=["pipeline", "scratch_directory"], output_names=["pipeline", "scratch_directory", "figures"]):
    def run(self):
        pipeline = self.inputs.pipeline
        pipeline.select_best_candidates(3)
        self.outputs.pipeline = pipeline
        self.outputs.scratch_directory = self.inputs.scratch_directory
        self.outputs.figures = capture_figures_to_bytes()

class ModeDecomposition(Task, input_names=["pipeline", "scratch_directory"], output_names=["pipeline", "scratch_directory", "figures"]):
    def run(self):
        pipeline = self.inputs.pipeline
        pipeline.mode_decomposition()
        self.outputs.pipeline = pipeline
        self.outputs.scratch_directory = self.inputs.scratch_directory
        self.outputs.figures = capture_figures_to_bytes()

class PostProcess(Task, input_names=["pipeline", "scratch_directory"], output_names=["final_directory", "figures"]):
    def run(self):
        pipeline = self.inputs.pipeline
        pipeline.postprocess(isosurface=0.3, voxel_size=None, flip=False)
        pipeline.close()
        self.outputs.final_directory = self.inputs.scratch_directory
        self.outputs.figures = capture_figures_to_bytes()
