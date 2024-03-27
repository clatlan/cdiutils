
import fabio
import numpy as np
import silx.io
import silx.io.specfile

from cdiutils.load import Loader


def safe(func):
    def wrap(self, *args, **kwargs):
        with silx.io.open(self.experiment_file_path) as specfile:
            return func(self, specfile, *args, **kwargs)
    return wrap


# TODO: Impelement roi parameter for detector, motors and mask methods
class SpecLoader(Loader):

    angle_names = {
        "sample_outofplane_angle": "eta",
        "sample_inplane_angle": "phi",
        "detector_outofplane_angle": "del",
        "detector_inplane_angle": "nu"
    }

    def __init__(
            self,
            experiment_file_path: str,
            detector_data_path: str,
            edf_file_template: str,
            detector_name: str,
            flat_field: str | np.ndarray = None,
            alien_mask: np.ndarray | str = None,
            **kwargs
    ) -> None:
        """
        Initialise SpecLoader with experiment data and detector
        information.

        Args:
            experiment_file_path (str): path to the spec master file
                used for the experiment.
            detector_data_path (str): the path to the directory
                containing the detector data.
            edf_file_template (str): the file name template of the
                detector data frame.
            detector_name (str): name of the detector.
            flat_field (str | np.ndarray, optional): flat field to
                account for the non homogeneous counting of the
                detector. Defaults to None.
            alien_mask (np.ndarray | str, optional): array to mask the
                aliens. Defaults to None.
        """
        super(SpecLoader, self).__init__(flat_field, alien_mask)
        self.experiment_file_path = experiment_file_path
        self.detector_data_path = detector_data_path
        self.edf_file_template = edf_file_template
        self.detector_name = detector_name

    @safe
    def load_detector_data(
            self,
            specfile: silx.io.specfile.SpecFile,
            scan: int,
            roi: tuple[slice] = None,
            binning_along_axis0=None
    ):
        if roi is None:
            roi = tuple(slice(None) for i in range(3))
        elif len(roi) == 2:
            roi = tuple([slice(None), roi[0], roi[1]])

        # TODO: implement flat_field consideration and binning_along_axis0
        frame_ids = specfile[f"{scan}.1/measurement/{self.detector_name}"][...]

        detector_data = []

        template = self.detector_data_path + self.edf_file_template

        for frame_id in frame_ids:
            with fabio.open(template % frame_id) as edf_data:
                detector_data.append(edf_data.data)

        return np.array(detector_data)[roi]

    @safe
    def load_motor_positions(
            self,
            specfile: silx.io.specfile.SpecFile,
            scan: int,
            roi: tuple[slice] = None,
            binning_along_axis0=None
    ):

        if roi is None or len(roi) == 2:
            roi = slice(None)
        elif len(roi) == 3:
            roi = roi[0]

        positioners = specfile[f"{scan}.1/instrument/positioners"]

        angles = {key: None for key in SpecLoader.angle_names.keys()}
        for angle, name in SpecLoader.angle_names.items():
            try:
                angles[angle] = positioners[name][roi]
            except ValueError:
                angles[angle] = angles[angle] = positioners[name][()]
        return angles


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
                            (
                                float(scan_command.split(" ")[3])
                                - float(scan_command.split(" ")[2])
                            )
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

    return (
        float(azimuth_angle),
        float(outofplane_angle),
        float(incidence_angle),
        float(inplane_angle),
        rocking_angle,
        float(angular_step)
    )
