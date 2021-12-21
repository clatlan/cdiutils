import numpy as np
import xrayutilities as xu
from scipy.ndimage import center_of_mass

def diffraction_com_max(intensity, qx, qy, qz, maplog_min=3, verbose=True):

    if verbose:

        com_qx = np.sum(qx * np.sum(intensity, axis=(1, 2))) / np.sum(intensity)
        com_qy = np.sum(qy * np.sum(intensity, axis=(0, 2))) / np.sum(intensity)
        com_qz = np.sum(qz * np.sum(intensity, axis=(0, 1))) / np.sum(intensity)
        qcom = [com_qx, com_qy, com_qz]

        matrix_max = [c[0] for c in np.where(intensity == np.max(intensity))]
        qmax = qx[matrix_max[0]], qy[matrix_max[1]], qz[matrix_max[2]]

        print(
            "Max of intensity: \n"
            "In matrix coordinates: {}\n"
            "In reciprocal space coordinates: {} (1/angstroms)\n"
            "Center of mass of intensity: \n"
            "In reciprocal space coordinates: {} (1/angstroms)".format(
                matrix_max, qmax, qcom
            )
        )
        com_qx = np.sum(qx * np.sum(intensity, axis=(1, 2))) / np.sum(intensity)
        com_qy = np.sum(qy * np.sum(intensity, axis=(0, 2))) / np.sum(intensity)
        com_qz = np.sum(qz * np.sum(intensity, axis=(0, 1))) / np.sum(intensity)
        qcom = [com_qx, com_qy, com_qz]

    log_intensity = xu.maplog(intensity, maplog_min, 0)
    filtered_intensity = np.power(log_intensity, 10)

    matrix_com = [round(c) for c in center_of_mass(filtered_intensity)]
    com_qx = (
        np.sum(qx * np.sum(filtered_intensity, axis=(1, 2)))
        / np.sum(filtered_intensity)
        )
    com_qy = (
        np.sum(qy * np.sum(filtered_intensity, axis=(0, 2)))
        / np.sum(filtered_intensity)
        )
    com_qz = (
        np.sum(qz * np.sum(filtered_intensity, axis=(0, 1)))
        / np.sum(filtered_intensity)
    )
    qcom = [com_qx, com_qy, com_qz]

    matrix_max = [
        c[0] for c in np.where(
            filtered_intensity == np.max(filtered_intensity)
        )
    ]
    qmax = qx[matrix_max[0]], qy[matrix_max[1]], qz[matrix_max[2]]
    if verbose:
        print(
            "Max of intensity: \n"
            "In matrix coordinates: {}\n"
            "In reciprocal space coordinates: {} (1/angstroms)\n"
            "Center of mass of intensity: \n"
            "In reciprocal space coordinates: {} (1/angstroms)".format(
                matrix_max, qmax, qcom
            )
        )

    return matrix_max, qmax, matrix_com, qcom