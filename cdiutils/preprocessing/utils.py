import numpy as np
import xrayutilities as xu
from scipy.ndimage import center_of_mass

def diffraction_com_max(intensity, qx, qy, qz, maplog_min=3, verbose=True):

    if verbose:
        matrix_com = [round(c) for c in center_of_mass(intensity)]
        qcom = qx[matrix_com[0]], qy[matrix_com[1]], qz[matrix_com[2]]

        matrix_max = [c[0] for c in np.where(intensity == np.max(intensity))]
        qmax = qx[matrix_max[0]], qy[matrix_max[1]], qz[matrix_max[2]]

        print(
            "Max of intensity: \n"
            "In matrix coordinates: {}\n"
            "In reciprocal space coordinates: {} (1/angstroms)\n"
            "Center of mass of intensity: \n"
            "In matrix coordinates: {}\n"
            "In reciprocal space coordinates: {} (1/angstroms)".format(
                matrix_max, qmax, matrix_com, qcom
            )
        )

    log_intensity = xu.maplog(intensity, maplog_min, 0)
    filtered_intensity = np.power(log_intensity, 10)

    matrix_com = [round(c) for c in center_of_mass(filtered_intensity)]
    qcom = qx[matrix_com[0]], qy[matrix_com[1]], qz[matrix_com[2]]

    matrix_max = [
        c[0] for c in np.where(
            filtered_intensity == np.max(filtered_intensity)
        )
    ]
    qmax = qx[matrix_max[0]], qy[matrix_max[1]], qz[matrix_max[2]]
    if verbose:
        print(
            "\nAfter filtering\n"
            "Max of intensity: \n"
            "In matrix coordinates: {}\n"
            "In reciprocal space coordinates: {} (1/angstroms)\n"
            "Center of mass of intensity: \n"
            "In matrix coordinates: {}\n"
            "In reciprocal space coordinates: {} (1/angstroms)\n\n".format(
                matrix_max, qmax, matrix_com, qcom
            )
        )

    return matrix_max, qmax, matrix_com, qcom