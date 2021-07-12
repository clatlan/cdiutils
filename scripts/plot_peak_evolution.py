import numpy as np
import matplotlib as plt


if __name__ == "__main__":

    scan_1 = 199

    roi = []

    omega_scans = []

    delta_scans = []

    gamma_scans = []

    data_directory = "/data/id01/inhouse/richard/P10_2021/raw/  "

    thetas = {"theta1": [],
              "theta2": [],
              "theta3": [],
              "theta4": [],
              "theta5": [],
              "theta6": [],
              "theta7": [],
              "theta8": []
              }
    for scan in scans:
        intensities = []
        for theta in thetas.keys():
            crop = intensities

    scans = []
    for scan in scans:
        file = template.format(scan)
        data = pd.read_csv(path)

        intensity = data["intensity"]
        motor = data ["intensity"]
