import numpy as np
# handling vtk case
try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    IS_VTK_AVAILABLE = True

except ImportError:
    print("vtk package is not installed.")
    IS_VTK_AVAILABLE = False


class VtkImportError(ImportError):
    """Custom exception to handle Vtk import error."""
    def __init__(self, msg: str = None) -> None:
        _msg = "vtk package is not installed."
        if msg is not None:
            _msg += "\n" + msg
        super().__init__(_msg)


def load_vtk(file_path: str):
    """
    Load a vtk file.

    Args:
        file_path (_type_): the path to the file to open.

    Raises:
        VtkImportError: if vtk is not installed.

    Returns:
        the reader output
    """
    if not IS_VTK_AVAILABLE:
        raise VtkImportError

    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(file_path)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllTensorsOn()
    reader.Update()

    return reader.GetOutput()


def save_as_vti(
        output_path: str,
        voxel_size: tuple | list | np.ndarray,
        cxi_convention: bool = False,
        origin: tuple = (0, 0, 0),
        **np_arrays: dict[np.ndarray]
) -> None:
    """Save numpy arrays to .vti file."""
    if not IS_VTK_AVAILABLE:
        raise VtkImportError
    voxel_size = tuple(voxel_size)
    nb_arrays = len(np_arrays)

    if not nb_arrays:
        raise ValueError(
            "np_arrays is empty, please provide a dictionary of "
            "(fieldnames: np.ndarray) you want to save."
        )
    is_init = False
    for i, (key, array) in enumerate(np_arrays.items()):
        if not is_init:
            shape = array.shape
            if cxi_convention:
                voxel_size = (voxel_size[2], voxel_size[1], voxel_size[0])
                shape = (shape[2], shape[1], shape[0])
            image_data = vtk.vtkImageData()
            image_data.SetOrigin(origin)
            image_data.SetSpacing(voxel_size)
            image_data.SetExtent(
                0, shape[0] - 1,
                0, shape[1] - 1,
                0, shape[2] - 1
            )
            point_data = image_data.GetPointData()
            is_init = True

        vtk_array = numpy_to_vtk(array.ravel())
        point_data.AddArray(vtk_array)
        point_data.GetArray(i).SetName(key)
        point_data.Update()

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(image_data)
    writer.Write()
