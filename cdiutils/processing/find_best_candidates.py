import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import shutil
import argparse

from cdiutils.load.load_data import get_data_from_cxi


def find_best_candidates(files, nb_to_keep=5, criterion="llkf", plot=False):

    if criterion == "llkf":

        print("\n[INFO] Candidates with the lowest llkf will be saved.\n")

        # make a dictionary with file names as keys and llkf as values
        LLKFs = {}
        LLKs = {}
        for f in files:
            data_dic = get_data_from_cxi(f, "llkf", "llk")
            LLKFs[f] = data_dic["llkf"]
            LLKs[f] = data_dic["llk"]

        print("\n[INFO]Sorting files...")

        # sort the dictionary
        sorted_LLKFs = dict(sorted(LLKFs.items(),
                            key=lambda item: item[1],
                            reverse=False))

        # pick only file names with the lowest llkf values
        files_of_interest = list(sorted_LLKFs.keys())[:nb_to_keep]

        # copy these files with a different name
        for i, f in enumerate(files_of_interest):
            dir_name, file_name = os.path.split(f)
            run_nb = file_name.split("Run")[1][2:4]
            scan_nb = file_name.split("_")[0]
            file_name = ("/candidate_{}-{}_".format(i+1, nb_to_keep)
                         + scan_nb + "_run_" + run_nb
                         + "_LLKF{:5f}_LLK{:5f}.cxi".format(LLKFs[f], LLKs[f]))
            shutil.copy(f, dir_name + file_name)

        print("Files saved.")

    elif criterion == "std":

        print("\n[INFO] Candidates with the lowest std will be saved.\n")

        # make a dictionary with file names as keys and std as values
        STDs = {f: 0 for f in files}
        LLKFs = {f: 0 for f in files}
        LLKs = {f: 0 for f in files}

        if len(files) <= nb_to_keep:
            print(
                "[INFO] did not proceed to sorting because the number of "
                "files is already satisfied"
            )
            files_of_interest = files
            nb_to_keep = len(files)
        else:
            for i, f in enumerate(files):
                print("[INFO] Opening file:", os.path.basename(f))
                data_dic = get_data_from_cxi(f, "support", "electronic_density",
                                            "llkf", "llk")

                support = data_dic["support"]
                eletronic_density = data_dic["electronic_density"]
                LLKFs[f] = data_dic["llkf"]
                LLKs[f] = data_dic["llk"]

                density_of_interest = eletronic_density[support > 0]
                modulus = np.abs(density_of_interest)
                STDs[f] = np.std(modulus)

            # sort the dictionary
            sorted_STDs = dict(sorted(STDs.items(),
                            key=lambda item: item[1],
                            reverse=False))

            # pick only file names with the lowest std values
            files_of_interest = list(sorted_STDs.keys())[:nb_to_keep]

        # copy these files with a different name
        for i, f in enumerate(files_of_interest):
            dir_name, file_name = os.path.split(f)
            run_nb = file_name.split("Run")[1][2:4]
            scan_nb = file_name.split("_")[0]
            file_name = ("/candidate_{}-{}_".format(i+1, nb_to_keep)
                         + scan_nb + "_run_" + run_nb
                         + "_STD{:5f}_".format(STDs[f])
                         + "_LLKF{:5f}_LLK{:5f}.cxi".format(LLKFs[f], LLKs[f]))
            shutil.copy(f, dir_name + file_name)

        print("Files saved.")

        if plot:
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax1)
            runs = [file.split("Run")[1][2:4] for file in STDs.keys()]
            ax1.plot(runs, STDs.values(), "ro")
            ax1.set_ylabel("Standard deviation")
            ax2.plot(runs, LLKFs.values(), "bo")
            ax2.set_ylabel("Free log likelihood")

            plt.show()


if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--scan-directory", required=True, type=str,
                    help="number of the scan to process")
    ap.add_argument("-n", "--number", default=5, type=int,
                    help="number of reconstructions to keep")
    ap.add_argument("-c", "--criterion", default="llkf", type=str,
                    help="choose on which criterion the candidates are "
                         "selected")
    ap.add_argument("-p", "--plot", default=False, action="store_true",
                    help="if std vs llks are plotted")

    args = vars(ap.parse_args())
    files = []
    # get the .cxi files that came from the pynx reconstruction
    try:
        files = glob.glob(args["scan_directory"] + "/candidate_*.cxi")
        if not files:
            try:
                files = glob.glob(args["scan_directory"] + "/*Run*.cxi")
            except Exception as e:
                print("[ERROR] Following exception occured:", e.__str__())
        else:
            if len(files) < args["number"]:
                for f in files:
                    os.remove(f)
                try:
                    files = glob.glob(args["scan_directory"] + "/*Run*.cxi")
                except Exception as e:
                    print("[ERROR] Following exception occured:", e.__str__())
            elif len(files) == args["number"]:
                print("[INFO] Best candidate already found, exiting script...")
                exit()

            elif len(files) > args["number"]:
                for f in files:
                    dir_name, file_name = os.path.split(f)
                    if int(file_name[10]) > args["number"]:
                        file.remove(f)
                exit()

    except Exception as e:
        print("[ERROR] Following exception occured:", e.__str__())

    find_best_candidates(files, nb_to_keep=args["number"],
                         criterion=args["criterion"], plot=args["plot"])
