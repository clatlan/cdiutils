Loader
======

.. currentmodule:: cdiutils.io

.. autoclass:: Loader
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Use beamline-specific loaders::

    from cdiutils.io import ID01Loader

    # Create loader for ID01 beamline
    loader = ID01Loader(
        sample_name="S123",
        scan=42,
        data_dir="/path/to/data"
    )

    # Loader is used automatically by BcdiPipeline
    from cdiutils.pipeline import BcdiPipeline
    pipeline = BcdiPipeline(param_file_path="config.yml")
    pipeline.preprocess()  # loads data internally

See Also
--------
:class:`ID01Loader` : ESRF ID01 beamline
:class:`P10Loader` : PETRA III P10 beamline
:class:`SIXSLoader` : SOLEIL SIXS beamline
:class:`NanoMaxLoader` : MAX IV NanoMAX beamline
