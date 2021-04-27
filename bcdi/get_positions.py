

def get_positions(specfile_path, scan, beamline="ID01"):
    if beamline == "ID01":
        import silx.io
        specfile = silx.io.open(specfile_path)
        positioners = specfile[scan + ".1/instrument/positioners"]
        outofplane_angle = positioners["del"][...] # delta, outofplane detector
        incidence_angle = positioners["eta"][...] # eta, incidence sample
        inplane_angle = positioners["nu"][...] # nu, inplane detector
        azimuth_angle = positioners["phi"][...] # phi, azimuth sample angle

        specfile.close()
        if incidence_angle.shape != ():
            angle_step = (incidence_angle[-1] - incidence_angle[0]) \
                / incidence_angle.shape[0]
            incidence_angle = (incidence_angle[-1] + incidence_angle[0]) / 2
            rocking_angle = "outofplane"

        elif azimuth_angle.shape != ():
            angle_step = (azimuth_angle[-1] - azimuth_angle[0]) \
                / azimuth_angle.shape[0]
            rocking_angle = "inplane"

    elif beamline == "SIXS_2019":
        beamline_geometry = "MED_V"
        if beamline_geometry == "MED_V":
            import bcdi.preprocessing.ReadNxs3 as nxsRead3

            scan_file_name ="$DATA_ROOT_FOLDER/$TEMPLATE_DATA_FILE"%int(scan)
            data = nxsRead3.DataSet(scan_file_name)

            outofplane_angle = data.gamma[0] # gamma, outofplane detector angle
            incidence_angle = data.mu # mu, incidence sample angle
            inplane_angle = data.delta[0] # delta, inplane detector angle
            azimuth_angle = data.omega # omega, azimuth sample angle

            if incidence_angle[0] != incidence_angle[1]:
                rocking_angle = "outofplane"
                angle_step = (incidence_angle[-1] - incidence_angle[0]) / \
                              incidence_angle.shape[0]
                incidence_angle = (incidence_angle[-1] + incidence_angle[0]) / 2

            elif azimuth_angle[0] != azimuth_angle[1]:
                rocking_angle = "inplane"
                angle_step = (azimuth_angle[-1] - azimuth_angle[0]) /\
                              azimuth_angle.shape[0]
			# azimuth_angle = (azimuth_angle[-1] + azimuth_angle[0]) / 2
    return (float(outofplane_angle), float(incidence_angle),
            float(inplane_angle), rocking_angle, float(angle_step))



if __name__ == '__main__':
    import argparse

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--specfile-path", required=True, type=str,
                    help="The absolute path to the specfile of the measurement")
    ap.add_argument("--scan", required=True, type=str,
                    help="The scan to look at.")
    ap.add_argument("--beamline", default="ID01", type=str,
                    help="The beamline where the measurement was performed")

    args = vars(ap.parse_args())

    for res in get_positions(args["specfile_path"], args["scan"], args["beamline"]):
        print(res)
