import numpy as np
from scipy.ndimage.measurements import center_of_mass


def center_data(data, mask=None, max_size=256):
    shape = data.shape
    com = [int(round(c)) for c in center_of_mass(data)]
    com_box = [2 * min(c, shape[i] - c) for i, c in enumerate(com)]
    max_box = [min(c, max_size) for c in com_box]

    final_shape = smaller_primes((max_box[0], max_box[1], max_box[2]),
                                maxprime=7, required_dividers=(2,))

    centered_data = data[com[0]-final_shape[0]//2 : com[0]+final_shape[0]//2,
                         com[1]-final_shape[1]//2 : com[1]+final_shape[1]//2,
                         com[2]-final_shape[2]//2 : com[2]+final_shape[2]//2]
    if mask is not None:
        centered_mask = mask[com[0]-final_shape[0]//2:com[0]+final_shape[0]//2,
                             com[1]-final_shape[1]//2:com[1]+final_shape[1]//2,
                             com[2]-final_shape[2]//2:com[2]+final_shape[2]//2]
        return centered_data, centered_mask

    return centered_data


def center(data, com=None):
    shape = data.shape
    if com is None:
        com =[round(c) for c in center_of_mass(data)]
    centered_data = np.roll(data, shape[0] // 2 - com[0], axis=0)
    centered_data = np.roll(centered_data, shape[1] // 2 - com[1], axis=1)
    centered_data = np.roll(centered_data, shape[2] //2 - com[2], axis=2)

    return centered_data


def crop_at_center(data, final_shape=None):
    if final_shape is None:
        print("No final shape specified, did not proceed to cropping")
        return data

    shape = data.shape
    final_shape = np.array(final_shape)

    if not (final_shape <= data.shape).all():
        print("One of the axis of the final shape is larger than " \
              "the initial axis (initial shape: {}, final shape: {}).\n" \
              "Did not proceed to cropping.".format(shape, tuple(final_shape)))
        return data

    c = np.array(shape) // 2
    to_crop = final_shape // 2
    plus_one = np.where((final_shape %2 == 0), 0, 1)

    cropped = data[c[0] - to_crop[0] : c[0] + to_crop[0] + plus_one[0],
                   c[1] - to_crop[1] : c[1] + to_crop[1] + plus_one[1],
                   c[2] - to_crop[2] : c[2] + to_crop[2] + plus_one[2]]

    return cropped


def are_coplanar(q1, q2, q3, value=False):
    result =  np.dot(q1, np.cross(q2, q3))
    if value:
        return result, result == 0
    else:
        return result == 0
