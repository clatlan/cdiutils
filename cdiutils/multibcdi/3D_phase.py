import numpy as np

from utils import are_coplanar


# TODO: Check if the compute 3D phase functions need normalized vectors or not

def compute_3D_phase_V2(phases, q_vectors, support):
    # First load the vectors of the measurement frame.
    Q = [q1, q2, q3] = list(q_vectors.values())
    Q = []
    phi = []
    for key in phases.keys():
        Q.append(q_vectors[key])
        phi.append(phases[key])
    phi = np.array(phi)

    # check if vectors are coplanar. If they are, they cannot describe the
    # 3D space and then 3D phase cannot be computed
    if are_coplanar(q1, q2, q3):
        print("The given vectors are coplanar, this won't work.")
        return None
    Q_inverse = np.linalg.inv(np.array(Q))

    # Get only the coordinates of the voxel within the support.
    I, J, K = np.where(support > 0)

    # Initialise the 3D phase.
    shape = support.shape
    phase_3D = np.zeros([shape[0], shape[1], shape[2], 3])

    # Iteration over each voxel. 3D phase is computed for each voxel.
    for i, j, k in zip(I.tolist(), J.tolist(), K.tolist()):

        # Compute U values in the canonical basis.
        U = np.dot(Q_inverse, phi[..., i, j, k])
        phase_3D[i, j, k] = np.array(U)

    return phase_3D


def compute_3D_phase(phases, q_vectors, support):

    # First load the vectors of the measurement frame.
    Q = [q1, q2, q3] = q_vectors.values()

    # check if vectors are coplanar. If they are, the cannot describe the
    # 3D space and then 3D phase cannot be computed
    if are_coplanar(q1, q2, q3):
        print("The given vectors are coplanar, this won't work.")
        return None

    # Prepare the matrix M_inverse that is needed for retrieving the U values.
    # Theses U values are the actual components of the 3D phase for each q
    # vector.
    alpha = np.dot(q1, q2)
    beta = np.dot(q2, q3)
    gamma = np.dot(q1, q3)
    M = np.array([[1, alpha, gamma], [alpha, 1, beta],
                  [gamma, beta, 1]], np.float)
    M_inverse = np.linalg.inv(M)

    # Get the shape of the 3D phase and initialise it.
    shape = support.shape
    phase_3D = np.zeros([shape[0], shape[1], shape[2], 3])

    # Get only the coordinates of the voxel within the support.
    I, J, K = np.where(support > 0)
    # I, J, K = I.tolist(), J.tolist(), K.tolist()

    # Iteration over each voxel. 3D phase is computed for each voxel.
    for i, j, k in zip(I.tolist(), J.tolist(), K.tolist()):

        # Initialise the current voxel phase with zeros
        voxel_phase = np.zeros([3])

        # Phi values are the measured values. The phi list will contain
        # 3 values, one for each q_vector.
        phi = []
        for peak in phases.keys():
            phi.append(phases[peak][i, j, k])

        # U Values are computed here.
        U = np.dot(M_inverse, np.array(phi))

        # Add up the u vectors described in the canonical basis (u*q).
        # This will provide the 3D phase of the voxel.
        for u, q in zip(U.tolist(), Q):
            voxel_phase = np.add(voxel_phase, u * q)

        # Finally append the voxel phase to the global 3D phase
        phase_3D[i, j, k] = voxel_phase

    return phase_3D


def compute_projected_phase(phase_3D, q_projection):
    return np.dot(phase_3D, q_projection)
