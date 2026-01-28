BcdiPipeline
============

.. currentmodule:: cdiutils.pipeline

.. autoclass:: BcdiPipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Main Methods

   .. autosummary::
      ~BcdiPipeline.preprocess
      ~BcdiPipeline.phase_retrieval
      ~BcdiPipeline.postprocess
      ~BcdiPipeline.facet_analysis

   .. rubric:: Configuration

   .. autosummary::
      ~BcdiPipeline.update_from_file

   .. rubric:: Visualisation

   .. autosummary::
      ~BcdiPipeline.phase_retrieval_gui
      ~BcdiPipeline.show_3d_final_result

   .. rubric:: Inherited from Pipeline

   .. autosummary::
      ~BcdiPipeline.setup_logger
      ~BcdiPipeline.setup_savedir
      ~BcdiPipeline.submit_to_slurm

Examples
--------

Basic usage::

    from cdiutils.pipeline import BcdiPipeline

    # Create pipeline from configuration file
    pipeline = BcdiPipeline(param_file_path="config.yml")

    # Run complete workflow
    pipeline.preprocess()
    pipeline.phase_retrieval()
    pipeline.postprocess()

See Also
--------
:class:`~cdiutils.pipeline.Pipeline` : Base pipeline class
:class:`~cdiutils.process.PyNXPhaser` : Phase retrieval engine
:class:`~cdiutils.process.PostProcessor` : Post-processing tools
