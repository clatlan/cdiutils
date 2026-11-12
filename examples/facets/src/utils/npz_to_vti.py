import vtk
from vtk.util import numpy_support
import numpy as np
import os
import sys

npz_path = sys.argv[1]
base_name = os.path.splitext(npz_path)[0]
vti_path = f"{base_name}_normalized.vti" # Added suffix to avoid overwriting

if not os.path.exists(npz_path):
    print(f"Error: Input file not found at '{npz_path}'")
else:
    print(f"Loading data from: {npz_path}")
    with np.load(npz_path) as npz_file:
        data_dict = {key: npz_file[key] for key in npz_file.keys()}

    if not data_dict:
        print("Error: The NPZ file is empty.")
    else:
        imageData = vtk.vtkImageData()

        first_key = list(data_dict.keys())[0]
        grid_shape = data_dict[first_key].shape

        imageData.SetDimensions(grid_shape[2], grid_shape[1], grid_shape[0])
        imageData.SetSpacing(5.0, 5.0, 5.0)
        imageData.SetOrigin(0.0, 0.0, 0.0)

        for key, array in data_dict.items():
            if array.shape != grid_shape:
                print(f"Warning: Skipping array '{key}' due to mismatched shape.")
                continue

            # === NORMALIZATION STEP FOR 'amp' ARRAY ===
            if key == 'amp':
                min_val = np.min(array)
                max_val = np.max(array)
                print(f"  > Normalizing '{key}' array from range [{min_val:.3f}, {max_val:.3f}] to [0, 1]")
                if (max_val - min_val) > 0:
                    # Apply min-max normalization
                    array = (array - min_val) / (max_val - min_val)
                else: # Handle case where all values are the same
                    array = np.zeros_like(array)
            # ============================================

            vtk_array = numpy_support.numpy_to_vtk(num_array=array.flatten(order='C'), deep=True)

            new_key = 'amplitude' if key == 'amp' else key
            vtk_array.SetName(new_key)

            imageData.GetPointData().AddArray(vtk_array)
            print(f"  > Added data array '{key}' as '{new_key}'")

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(vti_path)
        writer.SetInputData(imageData)
        writer.Write()

        print(f"\nSuccessfully converted and saved normalized data to '{vti_path}'")
