PostProcessor
=============

.. currentmodule:: cdiutils.process

.. autoclass:: PostProcessor
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Structural Properties

   .. autosummary::
      ~PostProcessor.get_displacement
      ~PostProcessor.get_het_normal_strain
      ~PostProcessor.get_structural_properties

   .. rubric:: Phase Manipulation

   .. autosummary::
      ~PostProcessor.unwrap_phase
      ~PostProcessor.remove_phase_ramp

   .. rubric:: Post-processing

   .. autosummary::
      ~PostProcessor.apodize
      ~PostProcessor.flip_reconstruction

Examples
--------

Post-processing is typically handled by :class:`BcdiPipeline`::

    from cdiutils.pipeline import BcdiPipeline

    pipeline = BcdiPipeline(param_file_path="config.yml")
    pipeline.preprocess()
    pipeline.phase_retrieval()
    
    # Post-processing includes strain, displacement calculation
    pipeline.postprocess()

See Also
--------
:class:`BcdiPipeline` : Uses PostProcessor for result analysis
:class:`PyNXPhaser` : Provides reconstructed objects for post-processing
