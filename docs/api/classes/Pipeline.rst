Pipeline
========

.. currentmodule:: cdiutils.pipeline

.. autoclass:: Pipeline
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Logging

   .. autosummary::
      ~Pipeline.setup_logger

   .. rubric:: File Management

   .. autosummary::
      ~Pipeline.setup_savedir

   .. rubric:: SLURM Integration

   .. autosummary::
      ~Pipeline.submit_to_slurm

Notes
-----
This is an abstract base class. Use :class:`BcdiPipeline` for BCDI workflows
or create your own subclass for custom pipelines.

See Also
--------
:class:`BcdiPipeline` : Concrete implementation for BCDI processing
