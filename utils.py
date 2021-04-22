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
