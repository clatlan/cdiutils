import numpy as np
import matplotlib.pyplot as plt
import h5py


def find_hull(support, threshold=26):
    kernel = np.ones(shape=(3, 3, 3))
    convolved_support = ndimage.convolve(support, kernel,
                                         mode='constant', cval=0.0)
    hull = np.where(((0<convolved_support) & (convolved_support<=threshold)),
                    support, 0)
    return hull


def unit_vector(vector):
	return vector / np.linalg.norm(vector)


def normalize_complex_array(array):
    shifted_array = array - array.real.min() - 1j*array.imag.min()
    return shifted_array/np.abs(shifted_array).max()


def get_data_from_cxi(file, *items):

    data_dic = {}
    print("[INFO] Opening file:", file)

    try:
        data = h5py.File(file, "r")

        if "support" in items:
            data_dic["support"] = data["entry_1/image_1/support"][...]

        if "electronic_density" in items :
            data_dic["electronic_density"]= data["entry_1/data_1/data"][...]

        if "llkf" in items:
            data_dic["llkf"] = float(data["entry_1/image_1/process_1/results/" \
                                          "free_llk_poisson"][...])

        if "llk" in items:
            data_dic["llk"] = float(data["entry_1/image_1/process_1/results/" \
                                          "llk_poisson"][...])

        data.close()
        return data_dic

    except Exception as e:
        print("[ERROR] An error occured while opening the file:", f,
              "\n", e.__str__())
        return None


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
