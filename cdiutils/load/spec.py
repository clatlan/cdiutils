from typing import Union, Optional

import fabio
import numpy as np
import silx.io
import silx.io.specfile


def safe(func):
    def wrap(self, *args, **kwargs):
        with silx.io.open(self.experiment_file_path) as specfile:
            return func(self, specfile, *args, **kwargs)
    return wrap


class SpecLoader():
    def __init__(
            self,
            experiment_file_path: str,
            detector_data_path: str,
            edf_file_template: str,
            flatfield: Union[str, np.array]=None,
            alien_mask: Union[np.ndarray, str]=None,
            detector_name: str="mpx4inr"
    ):
        self.experiment_file_path = experiment_file_path
        if not detector_data_path.endswith("/"):
            detector_data_path += "/"
        self.detector_data_path = detector_data_path
        self.edf_file_template = edf_file_template
        self.detector_name = detector_name

        # load the flatfield
        if type(flatfield) == str and flatfield.endswith(".npz"):
            self.flatfield = np.load(flatfield)["arr_0"]
        elif type(flatfield) == np.ndarray:
            self.flatfield=flatfield
        elif flatfield is None:
            self.flatfield = None
        else:
            raise ValueError(
                "[ERROR] wrong value for flatfield parameter, provide a path, "
                "np.array or leave it to None"
            )
        
        if isinstance(alien_mask, str) and alien_mask.endswith(".npz"):
            self.alien_mask = np.load(alien_mask)["arr_0"]
        elif isinstance(alien_mask, np.ndarray):
            self.alien_mask=alien_mask
        elif alien_mask is None:
            self.alien_mask = None
        else:
            raise ValueError(
                "[ERROR] wrong value for alien_mask parameter, provide a path, "
                "np.ndarray or leave it to None"
            )

    @safe
    def load_detector_data(
            self,
            specfile: silx.io.specfile.SpecFile,
            scan: int,
            binning_along_axis0=None
    ):
        # TODO: implement flatfield consideration and binning_along_axis0
        frame_ids = specfile[f"{scan}.1/measurement/{self.detector_name}"][...]
        
        detector_data = []

        template = self.detector_data_path + self.edf_file_template

        for frame_id in frame_ids:
            with fabio.open(template % frame_id) as edf_data:
                detector_data.append(edf_data.data)
        
        return np.array(detector_data)
    
    @safe
    def load_motor_positions(
            self, 
            specfile: silx.io.specfile.SpecFile,
            scan: int,
            binning_along_axis0=None
    ):
        positioners = specfile[f"{scan}.1/instrument/positioners"]

        # delta outofplane detector
        outofplane_detector_angle = positioners["del"][...]
        # eta incidence sample
        outofplane_sample_angle = positioners["eta"][...]
        # nu inplane detector
        inplane_detector_angle = positioners["nu"][...]
        # phi azimuth sample angle
        inplane_sample_angle = positioners["phi"][...]

        # if outofplane_sample_angle.shape != ():
        #     # angular_step = (
        #     #     (incidence_angle[-1] - incidence_angle[0])
        #     #     / incidence_angle.shape[0]
        #     # )
        #     outofplane_sample_angle = (
        #         outofplane_sample_angle[-1]
        #         + outofplane_sample_angle[0]
        #         ) / 2
        #     # rocking_angle = "outofplane"

        # elif inplane_sample_angle.shape != ():
        #     # angular_step = (
        #     #     (azimuth_angle[-1] - azimuth_angle[0])
        #     #     / azimuth_angle.shape[0]
        #     # )
        #     inplane_sample_angle = (
        #         inplane_sample_angle[-1]
        #         + inplane_sample_angle[0]
        #         )/ 2
            # rocking_angle = "inplane"
        
        return (
            outofplane_sample_angle,
            inplane_sample_angle,
            inplane_detector_angle,
            outofplane_detector_angle
        )

    @staticmethod
    def get_mask(
            channel: Optional[int],
            detector_name: str="Maxipix"
    ) -> np.ndarray:
        """Load the mask of the given detector_name."""

        if detector_name in ("maxipix", "Maxipix", "mpxgaas", "mpx4inr", "mpx1x4"):
            mask = np.zeros(shape=(516, 516))
            mask[:, 255:261] = 1
            mask[255:261, :] = 1

        elif detector_name in ("Eiger2M", "eiger2m", "eiger2M", "Eiger2m"):
            mask = np.zeros(shape=(2164, 1030))
            mask[:, 255:259] = 1
            mask[:, 513:517] = 1
            mask[:, 771:775] = 1
            mask[0:257, 72:80] = 1
            mask[255:259, :] = 1
            mask[511:552, :] = 1
            mask[804:809, :] = 1
            mask[1061:1102, :] = 1
            mask[1355:1359, :] = 1
            mask[1611:1652, :] = 1
            mask[1905:1909, :] = 1
            mask[1248:1290, 478] = 1
            mask[1214:1298, 481] = 1
            mask[1649:1910, 620:628] = 1
        else:
            raise ValueError("Unknown detector_name")
        if channel:
            return np.repeat(mask[np.newaxis, :, :,], channel, axis=0)
        return mask


def get_positions(specfile_path, scan, beamline="ID01"):
    """
    Get the motor positions that were used during the scan.

    :param specfile_path: the path to the file that contains the motor
    positions of sample and detector.
    :param scan: scan number. int value must be provided.
    :param beamline: beamline where the measurement was performed.
    (Default value = "ID01").
    :returns: azimuth, out-of-plane, incidence, in-plane angles
    (floats), rockin angle (str), angular_step (float).

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
            angular_step = (
                (incidence_angle[-1] - incidence_angle[0])
                / incidence_angle.shape[0]
            )
            incidence_angle = (incidence_angle[-1] + incidence_angle[0]) / 2
            rocking_angle = "outofplane"

        elif azimuth_angle.shape != ():
            angular_step = (
                (azimuth_angle[-1] - azimuth_angle[0])
                / azimuth_angle.shape[0]
            )
            azimuth_angle = (azimuth_angle[-1] + azimuth_angle[0]) / 2
            rocking_angle = "inplane"

    elif beamline == "SIXS_2019":
        import bcdi.preprocessing.ReadNxs3 as nxsRead3
        beamline_geometry = "MED_V"

        if beamline_geometry == "MED_V":

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
                angular_step = (
                    (incidence_angle[-1] - incidence_angle[0])
                    / incidence_angle.shape[0]
                )
                incidence_angle = (
                    (incidence_angle[-1] + incidence_angle[0])
                    / 2
                )

            elif azimuth_angle[0] != azimuth_angle[1]:
                rocking_angle = "inplane"
                angular_step = (
                    (azimuth_angle[-1] - azimuth_angle[0])
                    / azimuth_angle.shape[0]
                )

    elif beamline == "P10":
        # TO DO: implementation for rocking angle != "outofplane"

        with open(specfile_path, "r", encoding="utf8") as file:
            content = file.readlines()
            for i, line in enumerate(content):

                if line.startswith("user"):
                    scan_command = content[i - 1]
                    scanning_angle = scan_command.split(" ")[1]
                    if scanning_angle == "om":
                        rocking_angle = "outofplane"
                        angular_step = (
                            (float(scan_command.split(" ")[3])
                            - float(scan_command.split(" ")[2]))
                            / float(scan_command.split(" ")[4])
                        )

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
            float(angular_step))


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
