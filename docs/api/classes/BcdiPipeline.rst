BcdiPipeline
============

.. currentmodule:: cdiutils.pipeline

.. autoclass:: BcdiPipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

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
