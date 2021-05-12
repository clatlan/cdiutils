import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def find_best_matching_normal_index(reference, normals, criterion="angle"):

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
			sign = -1 if normal[i] < 0 else 1
			miller_indices[i] = sign * int(round(abs(1/normal[i])))
	miller_indices /= np.gcd.reduce(np.array(miller_indices))
	return [int(i) for i in miller_indices.tolist()]


if __name__ == "__main__":


	# Define the scan numbers
	scans = [180, 181, 182, 183, 184, 185]
	# scans = [180, 181]

	# Define the corresponding potentials
	potentials = ["E_OC", "E_OC", "0 V", "0.1 V", "0.2 V", "0.3 V"]
	potentials = [0.6, 0.6, 0, 0.1, 0.2, 0.3]

	path_pattern = "Results/S{}/FacetAnalysis/"
	file_pattern = "facet_iso=0.5_ref_0 0 1.npy"

	u0 = [0, 0, 1]
	v0 = [1, 0, 0]
	u1 = [0, 1, 0]
	v1 = [0, 0, 1]

	rotation_matrix = get_rotation_matrix(u0, v0, u1, v1)

	# initialize the nested data dictionary which will contain data from every
	# scan. Each scan is associated with a sub dictionary.

	data = {key: {subkey: [] for subkey in ["strain_mean", "disp_mean",
			"strain_std", "disp_std", "facet_normals", "facet_id",
			"facet_miller_indices"]} for key in scans}

	# chose the reference scan and load the corresponding data
	reference_scan = 180
	file_path = path_pattern.format(reference_scan) + file_pattern
	ref_data = np.load(file_path, allow_pickle=True).item()

	# store the facet normals of the reference scan as reference normals
	reference_normals = [np.array([ref_data["n0"][i],
						 ref_data["n1"][i],
						 ref_data["n2"][i]])
					 	 for i in range(len(ref_data["facet"]))]

	# store reference scan data into the main data dictionary
	data[reference_scan]["facet_normals"] = reference_normals
	data[reference_scan]["strain_mean"] = ref_data["strain_mean"]
	data[reference_scan]["disp_mean"] = ref_data["disp_mean"]
	data[reference_scan]["strain_std"] = ref_data["strain_std"]
	data[reference_scan]["disp_std"] = ref_data["disp_std"]
	data[reference_scan]["facet_id"] = ref_data["FacetIds"]
	data[reference_scan]["potential"] = potentials[0]

	# loop over every other scan
	for k, scan in enumerate(scans):
		if scan != reference_scan:

			file_path = path_pattern.format(scan) + file_pattern
			scan_data = np.load(file_path, allow_pickle=True).item()

			# print("Number of facet for scan {}".format(scan), len(scan_data["facet"]))


			# compute the normal for each facet of the current scan
			normals = [np.array([scan_data["n0"][i],
					scan_data["n1"][i],
					scan_data["n2"][i]])
					for i in range(len(scan_data["facet"]))]

			for i, reference in enumerate(reference_normals):

				index = find_best_matching_normal_index(reference, normals,
														criterion="angle")

				data[scan]["facet_normals"].append(normals[index])
				data[scan]["strain_mean"].append(scan_data["strain_mean"][index])
				data[scan]["disp_mean"].append(scan_data["disp_mean"][index])
				data[scan]["strain_std"].append(scan_data["strain_std"][index])
				data[scan]["disp_std"].append(scan_data["disp_std"][index])
				data[scan]["facet_id"].append(scan_data["FacetIds"][index])
				data[scan]["potential"] = potentials[k]

	# getting facet miller indices for each scan and each facet
	for scan_id, scan_data in data.items():
		data[scan_id]["facet_miller_indices"] = [get_miller_indices(n)
				for n in data[scan_id]["facet_normals"]]


	for scan_id, value in data.items():
		print("scan #", scan_id)
		print("\n facet miller indices:")
		print(value["facet_miller_indices"])
		print("\n facet normals:")
		print(value["facet_normals"])
		print("\n\n")


		# for key, value in data.items():
	# 			print("scan #", key)
	# 			for ke, v in value.items():
	# 				print("key", ke)
	# 				print(v)
	# 			print("\n\n")

	# Get the 111, 110, 100 plane families
	planes111, planes110, planes100 = planes_111_110_100()

	for plane_family in [planes111, planes110, planes100]:

		""" Plotting  facet average strain"""
		fig, axes = plt.subplots(2, 1)
		for plane in plane_family:
			# for each plane, new lists are created, they will contain
			# the necessary information for plotting
			facet_strain = []
			facet_strain_std = []

			facet_disp = []
			facet_disp_std = []

			plot_potentials = []
			scan_ids = []
			for scan_id, value in data.items():
				if plane in value["facet_miller_indices"]:
					# Find the index of the current plane
					facet_index = value["facet_miller_indices"].index(plane)

					# add strain, disp and their respective std to the lists
					facet_strain.append(value["strain_mean"][facet_index])
					facet_strain_std.append(value["strain_std"][facet_index])

					facet_disp.append(value["disp_mean"][facet_index])
					facet_disp_std.append(value["disp_std"][facet_index])

				else:
					facet_strain.append(np.nan)
					facet_strain_std.append(np.nan)
					facet_disp.append(np.nan)
					facet_disp_std.append(np.nan)

				# add the potential and scan number to the lists
				plot_potentials.append(value["potential"])
				scan_ids.append(scan_id)

			# make the label for legends
			label = "(" + str(plane).strip("[]") + ")"

			df = pd.DataFrame({"scans": scan_ids,
							   "potentials": plot_potentials,
							   "strain": facet_strain,
							   "strain_std": facet_strain_std,
							   "disp": facet_disp,
							   "disp_std": facet_disp_std})

			# sort the entire dataframe by potentials
			df.sort_values(by="potentials", inplace=True)
			# print("Plane {}".format(label), df)

			# plot strains
			line, = axes[0].plot(df["potentials"].fillna(method="ffill"),
						 	 	 df["strain"].fillna(method="ffill"), ls="--")
			axes[0].plot(df["potentials"], df["strain"], color=line.get_color(),
					 	 marker="o", label=label)

			# plot displacements
			line, = axes[1].plot(df["potentials"].fillna(method="ffill"),
						 	     df["disp"].fillna(method="ffill"), ls="--")
			axes[1].plot(df["potentials"], df["disp"], color=line.get_color(),
					 	 marker="o", label=label)

			# plt.errorbar(df["scans"],
			# 			 df["strain"],
			# 			 df["strain_std"],
			# 			 linestyle="-", marker="o",
			# 			 label=legend, color=line.get_color())

		axes[0].set_ylabel("Average strain")
		axes[1].set_xlabel("Potentials (V)")
		axes[1].set_ylabel("Average displacement ($\AA$)")
		fig.legend(handles=axes[0].get_legend_handles_labels()[0])
		fig.suptitle("Strain and displacement averages")

	plt.show()
