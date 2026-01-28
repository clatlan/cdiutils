CXIFile
=======

.. currentmodule:: cdiutils.io

.. autoclass:: CXIFile
   :members:
   :undoc-members:
   :show-inheritance:

   The CXIFile class provides utilities for reading and writing CXI format files.
   CXI files are HDF5-based containers for coherent X-ray imaging data.

Examples
--------

Reading CXI file::

    from cdiutils.io import CXIFile

    with CXIFile("results.cxi", mode="r") as cxi:
        # Access data using dictionary-like syntax
        data = cxi["entry_1/data_1/data"][()]
        mask = cxi["entry_1/data_1/mask"][()]
        
        # Or use get_node method
        obj = cxi.get_node("entry_1/image_1/data")[()]

Writing CXI file::

    from cdiutils.io import CXIFile
    import numpy as np
    
    with CXIFile("output.cxi", mode="w") as cxi:
        # Set entry and create datasets
        entry = cxi.set_entry(1)
        cxi.file.create_dataset(f"{entry}/data", data=data)

See Also
--------
:class:`CXIExplorer` : Interactive CXI file explorer
:class:`BcdiPipeline` : Saves results to CXI format
