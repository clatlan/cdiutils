ID01Loader
==========

.. currentmodule:: cdiutils.io

.. autoclass:: ID01Loader
   :members:
   :undoc-members:
   :show-inheritance:

   The ID01Loader class handles data loading from the ESRF ID01 beamline.
   It is typically used internally by BcdiPipeline.

   See :class:`Loader` for inherited methods.

Examples
--------

The loader is used automatically by BcdiPipeline::

    from cdiutils.pipeline import BcdiPipeline

    # Configure pipeline for ID01 in YAML file:
    # beamline: "ID01"
    # sample_name: "S123"
    # scan: 42
    # data_dir: "/data/id01/inhouse/sample123"

    pipeline = BcdiPipeline(param_file_path="config.yml")
    pipeline.preprocess()  # automatically uses ID01Loader

See Also
--------
:class:`Loader` : Base loader class
:class:`BcdiPipeline` : Uses loaders to load data
