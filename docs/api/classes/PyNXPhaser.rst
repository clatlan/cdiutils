PyNXPhaser
==========

.. currentmodule:: cdiutils.process

.. autoclass:: PyNXPhaser
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Phase Retrieval

   This class is deprecated. Use the BcdiPipeline.phase_retrieval() method instead,
   which handles PyNX execution automatically.

   See the :doc:`BcdiPipeline` documentation for current phase retrieval workflows.

Examples
--------

Phase retrieval should be performed using :class:`BcdiPipeline`::

    from cdiutils.pipeline import BcdiPipeline

    # Create pipeline with configuration
    pipeline = BcdiPipeline(param_file_path="config.yml")

    # Preprocess detector data
    pipeline.preprocess()

    # Run phase retrieval with PyNX
    pipeline.phase_retrieval()

See Also
--------
:class:`PhasingResultAnalyser` : Detailed analysis of phasing results
:class:`BcdiPipeline` : Complete pipeline using PyNXPhaser
