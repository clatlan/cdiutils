import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def format_plane_name(list):
    name = ""
    for e in list:
        if e < 0:
            name += r"$\overline{"
            name += r"{}".format(-e)
            name += r"}$ "
        else:
            name += r"${}$ ".format(e)
    return name


def planes_111_110_100():

    planes111 = []
    planes110 = []
    planes100 = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                n = [i, j, k]
                if np.linalg.norm(n) == np.linalg.norm([1, 1, 1]):
                    planes111.append(n)
                elif np.linalg.norm(n) == np.linalg.norm([1, 1, 0]):
                    planes110.append(n)
                elif np.linalg.norm(n) == np.linalg.norm([1, 0, 0]):
                    planes100.append(n)
    return planes111, planes110, planes100


def get_rotation_matrix(u0, v0, u1, v1):
    """Get the rotation matrix between two frames.

    :param u0: 1st vector of frame 0
    :param v0: 2d vector of frame 0
    :param u1: 1st vector of frame 1
    :param v1: 2d vector of frame 1
    """

    w0 = np.cross(u0, v0)
    w1 = np.cross(u1, v1)

    # normalize each vector
    u0 = unit_vector(u0)
    v0 = unit_vector(v0)
    w0 = unit_vector(w0)
    u1 = unit_vector(u1)
    v1 = unit_vector(v1)
    w1 = unit_vector(w1)

    # compute rotation matrix from base 1 to base 0
    a = np.array([u0, v0, w0])
    b = np.linalg.inv(np.array([u1, v1, w1]))
    rotation_matrix = np.dot(np.transpose(a), np.transpose(b))

    return rotation_matrix


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(u, v):
    u, v = unit_vector(u), unit_vector(v)
    return np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))


def find_best_matching_normal_index(
        reference,
        normals,
        criterion="angle"):

    best_index = 0

    if criterion == "angle":
        lowest_angle = abs(angle_between(reference, normals[0]))
        for i, normal in enumerate(normals):
            angle = angle_between(reference, normal)
            if abs(angle) < lowest_angle:
                lowest_angle = abs(angle)
                best_index = i
        # print("lowest_angle is", lowest_angle)

    elif criterion == "difference":
        lowest_difference = np.linalg.norm(reference - normals[0])
        for i, normal in enumerate(normals[1:]):
            difference = np.linalg.norm(reference - normal)
            if difference < lowest_difference:
                lowest_difference = difference
                best_index = i
    return best_index


def get_miller_indices(normal):
    miller_indices = [None, None, None]
    for i in range(normal.shape[0]):
        if abs(normal[i]) < 0.2:
            miller_indices[i] = int(0)
        else:
            absolute_value = abs(1/normal[i])
            sign = -1 if normal[i] < 0 else 1
            if absolute_value > 1.41 and absolute_value < 2.5:
                absolute_value = 2
            miller_indices[i] = sign * int(round(absolute_value))
    miller_indices /= np.gcd.reduce(np.array(miller_indices))
    return [int(i) for i in miller_indices.tolist()]


def distance_between_parallel_planes(a, b, c, d1, d2):
    return abs(d2 - d1) / np.sqrt(a**2 + b**2 + c**2)
