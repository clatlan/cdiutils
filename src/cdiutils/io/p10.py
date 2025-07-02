import numpy as np
import silx.io.h5py_utils

from cdiutils.io import Loader


class P10Loader(Loader):
    """A class for loading data from P10 beamline experiments."""

    angle_names = {
        "sample_outofplane_angle": "om",
        "sample_inplane_angle": "phi",
        "detector_outofplane_angle": "del",
        "detector_inplane_angle": "gam"
    }
    authorised_detector_names = ("eiger4m", )

    def __init__(
            self,
            experiment_data_dir_path: str,
            scan: int = None,
            sample_name: str = None,
            detector_name: str = None,
            flat_field: np.ndarray | str = None,
            alien_mask: np.ndarray | str = None,
            hutch: str = "EH1",
            **kwargs
    ) -> None:
        """
        Initialise P10Loader with experiment data directory path and
        detector information.

        Args:
            experiment_data_dir_path (str): path to the experiment data
                directory.
            detector_name (str): name of the detector.
            sample_name (str, optional): name of the sample. Defaults to
                None.
            flat_field (np.ndarray | str, optional): flat field to
                account for the non homogeneous counting of the
                detector. Defaults to None.
            alien_mask (np.ndarray | str, optional): array to mask the
                aliens. Defaults to None.
        """
        self.experiment_data_dir_path = experiment_data_dir_path
        super().__init__(scan, sample_name, flat_field, alien_mask)
        self.detector_name = detector_name
        if self.detector_name is None:
            self.detector_name = "e4m"

        if hutch.lower() == "eh2":
            self.angle_names["sample_outofplane_angle"] = "samth"
            self.angle_names["detector_outofplane_angle"] = "e2_t02"
            self.angle_names["sample_inplane_angle"] = None
            self.angle_names["detector_inplane_angle"] = None
        elif hutch.lower() != "eh1":
            raise ValueError(
                f"Hutch name (hutch={hutch}) is not valid. Can be 'EH1' or "
                "'EH2'."
            )

    def _get_file_path(
            self,
            scan: int,
            sample_name: str,
            data_type: str = "detector_data"
    ) -> str:
        """
        Get the file path based on scan number, sample name, and data
        type.

        Args:
            scan (int): Scan number.
            sample_name (str): Name of the sample.
            data_type (str, optional): Type of data. Defaults to
                                       "detector_data".

        Returns:
            str: File path.
        """
        if data_type == "detector_data":
            return (
                self.experiment_data_dir_path
                + f"/{sample_name}_{scan:05d}"
                + f"/{self.detector_name}"
                + f"/{sample_name}_{scan:05d}_master.h5"
            )
        if data_type == "motor_positions":
            return (
                self.experiment_data_dir_path
                + f"/{sample_name}_{scan:05d}"
                + f"/{sample_name}_{scan:05d}.fio"
            )
        raise ValueError(
            f"data_type {data_type} is not valid. Must be either detector_data"
            " or motor_positions."
        )

    def load_detector_data(
            self,
            scan: int = None,
            sample_name: str = None,
            roi: tuple[slice] = None,
            rocking_angle_binning: int = None,
            binning_method: str = "sum"
    ) -> None:
        """
        Load detector data for a given scan and sample.

        Args:
            scan (int): Scan number. Defaults to None.
            sample_name (str, optional): Name of the sample. Defaults to
                None.
            roi (tuple, optional): Region of interest. Defaults to None.
            rocking_angle_binning (int, optional): Binning factor along
                axis 0. Defaults to None.
            binning_method (str, optional): Binning method. Defaults to
                "sum".

        Returns:
            numpy.ndarray: Loaded detector data.
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)

        path = self._get_file_path(scan, sample_name)
        key_path = "entry/data/data_000001"

        roi = self._check_roi(roi)

        with silx.io.h5py_utils.File(path) as h5file:
            if rocking_angle_binning:
                data = h5file[key_path][()]
            else:
                data = h5file[key_path][roi]

        data = self.bin_flat_mask(
            data,
            roi,
            self.flat_field,
            self.alien_mask,
            rocking_angle_binning,
            binning_method
        )

        # Must apply mask on P10 Eiger data
        mask = self.get_mask(
            channel=data.shape[0],
            detector_name=self.detector_name,
            roi=(slice(None), roi[1], roi[2])
        )
        data = data * np.where(mask, 0, 1)

        return data

    def load_motor_positions(
            self,
            scan: int = None,
            sample_name: str = None,
            roi: tuple[slice] = None,
            rocking_angle_binning: int = None,
    ) -> None:
        """
        Load motor positions for a given scan and sample.

        Args:
            scan (int): Scan number. Defaults to None.
            sample_name (str, optional): Name of the sample. Defaults
                to None.
            roi (tuple, optional): Region of interest. Defaults to None.
            rocking_angle_binning (int, optional): Binning factor along
                axis 0. Defaults to None.

        Returns:
            dict: Dictionary containing motor positions.
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)

        path = self._get_file_path(
            scan,
            sample_name,
            data_type="motor_positions"
        )
        if roi is None or len(roi) == 2:
            roi = slice(None)
        elif len(roi) == 3:
            roi = roi[0]

        angles = {name: None for name in self.angle_names.values()}

        rocking_angle_values = []

        with open(path, encoding="utf8") as fio_file:
            lines = fio_file.readlines()
            rocking_angle_column = None
            for line in lines:
                line = line.strip()
                words = line.split()

                for name in self.angle_names.values():
                    if name in words:
                        if "=" in words:
                            angles[name] = float(words[-1])
                        if "Col" in words and rocking_angle_column is None:
                            rocking_angle_column = int(words[1]) - 1
                            rocking_angle = words[2]

            for line in lines:
                line = line.strip()
                words = line.split()

                # check if the first word is numeric, if True the line
                # contains motor position values
                # if words[0].replace(".", "", 1).isdigit():
                if words[0].replace(".", "").replace("-", "").isnumeric():
                    rocking_angle_values.append(
                        float(words[rocking_angle_column])
                    )
                    if "e2_t02" in angles:
                        # This means that 'e2_t02' must be used as the
                        # detector out-of-plane angle.
                        angles["e2_t02"] = float(words[1])
        self.rocking_angle = rocking_angle
        angles[rocking_angle] = np.array(rocking_angle_values)
        for name in angles:
            if angles[name] is None:
                angles[name] = 0

        angles[self.rocking_angle] = self.bin_rocking_angle_values(
            angles[self.rocking_angle], rocking_angle_binning
        )
        if roi:
            angles[rocking_angle] = angles[rocking_angle][roi]

        return {
            angle: angles[name]
            for angle, name in self.angle_names.items()
        }

    def load_energy(self, scan: int = None, sample_name: str = None) -> float:
        scan, sample_name = self._check_scan_sample(scan, sample_name)
        path = self._get_file_path(
            scan,
            sample_name,
            data_type="motor_positions"
        )
        with open(path, encoding="utf8") as fio_file:
            lines = fio_file.readlines()
            for line in lines:
                line = line.strip()
                words = line.split()
                if "fmbenergy" in words:
                    return float(words[-1])
        return None

    def load_det_calib_params(self) -> dict:
        return None

    def load_detector_shape(self, scan: int = None) -> tuple:
        return None
