import silx.io
import hdf5plugin


def get_positions(specfile_path, scan, beamline="ID01"):
    """
    Get the motor positions that were used during the scan.

    :param specfile_path: the path to the file that contains the motor
    positions of sample and detector.
    :param scan: scan number. int value must be provided.
    :param beamline: beamline where the measurement was performed.
    (Default value = "ID01").
    :returns: azimuth, out-of-plane, incidence, in-plane angles
    (floats), rockin angle (str), angle step (float).

    """

    if beamline == "ID01":
        specfile = silx.io.open(specfile_path)
        positioners = specfile["{}.1/instrument/positioners".format(scan)]

        # delta outofplane detector
        outofplane_angle = positioners["del"][...]
        # eta incidence sample
        incidence_angle = positioners["eta"][...]
        # nu inplane detector
        inplane_angle = positioners["nu"][...]
        # phi azimuth sample angle
        azimuth_angle = positioners["phi"][...]

        specfile.close()

        if incidence_angle.shape != ():
            angle_step = ((incidence_angle[-1] - incidence_angle[0])
                          / incidence_angle.shape[0])
            incidence_angle = (incidence_angle[-1] + incidence_angle[0]) / 2
            rocking_angle = "outofplane"

        elif azimuth_angle.shape != ():
            angle_step = ((azimuth_angle[-1] - azimuth_angle[0])
                          / azimuth_angle.shape[0])
            azimuth_angle = (azimuth_angle[-1] + azimuth_angle[0]) / 2
            rocking_angle = "inplane"

    elif beamline == "SIXS_2019":
        import bcdi.preprocessing.ReadNxs3 as nxsRead3
        beamline_geometry = "MED_V"

        if beamline_geometry == "MED_V":

            # scan_file_name ="$DATA_ROOT_FOLDER/$TEMPLATE_DATA_FILE"%int(scan)
            scan_file_name = specfile_path
            data = nxsRead3.DataSet(scan_file_name)

            # gamma outofplane detector angle
            outofplane_angle = data.gamma[0]
            # mu incidence sample angle
            incidence_angle = data.mu
            # delta inplane detector angle
            inplane_angle = data.delta[0]
            # omega azimuth sample angle
            azimuth_angle = data.omega

            if incidence_angle[0] != incidence_angle[1]:
                rocking_angle = "outofplane"
                angle_step = ((incidence_angle[-1] - incidence_angle[0])
                              / incidence_angle.shape[0])
                incidence_angle = ((incidence_angle[-1] + incidence_angle[0])
                                   / 2)

            elif azimuth_angle[0] != azimuth_angle[1]:
                rocking_angle = "inplane"
                angle_step = ((azimuth_angle[-1] - azimuth_angle[0])
                              / azimuth_angle.shape[0])

    elif beamline == "P10":
        # TO DO: implementation for rocking angle != "outofplane"

        with open(specfile_path, "r") as file:
            content = file.readlines()
            for i, line in enumerate(content):

                if line.startswith("user"):
                    scan_command = content[i - 1]
                    scanning_angle = scan_command.split(" ")[1]
                    if scanning_angle == "om":
                        rocking_angle = "outofplane"
                        angle_step = ((float(scan_command.split(" ")[3])
                                      - float(scan_command.split(" ")[2]))
                                      / float(scan_command.split(" ")[4]))

                if line.startswith("del"):
                    outofplane_angle = float(line.split(" = ")[1])

                elif line.startswith("om"):

                    incidence_angle = float(line.split(" = ")[1])

                elif line.startswith("gam"):
                    inplane_angle = float(line.split(" = ")[1])

                elif line.startswith("phi"):
                    azimuth_angle = float(line.split(" = ")[1])

    return (float(azimuth_angle),
            float(outofplane_angle),
            float(incidence_angle),
            float(inplane_angle),
            rocking_angle,
            float(angle_step))


if __name__ == '__main__':
    import argparse

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--specfile-path", required=True, type=str,
                    help="The absolute path to "
                         "the specfile of the measurement")
    ap.add_argument("--scan", required=True, type=int,
                    help="The scan to look at.")
    ap.add_argument("--beamline", default="ID01", type=str,
                    help="The beamline where the measurement was performed")

    args = vars(ap.parse_args())

    for res in get_positions(
            args["specfile_path"],
            args["scan"],
            args["beamline"]):
        print(res)
