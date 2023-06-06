import hdf5plugin
import numpy as np
import silx.io.h5py_utils


class P10Loader:
    """
    A class for loading data from P10 beamline experiments.
    """

    angle_names = {
        "sample_outofplane_angle": "om",
        "sample_inplane_angle": "phi",
        "detector_outofplane_angle": "del",
        "detector_inplane_angle": "gam"
    }

    def __init__(
            self,
            experiment_data_dir_path: str,
            detector_name: str,
            sample_name: str = None,
            flatfield: np.ndarray or str = None,
            alien_mask: np.ndarray or str = None
    ) -> None:
        """
        Initialize P10Loader with experiment data directory path and
        detector information.

        Args:
            experiment_data_dir_path (str): Path to the experiment data
                                            directory.
            detector_name (str): Name of the detector.
            sample_name (str, optional): Name of the sample. Defaults to
                                         None.
            flatfield (numpy.ndarray or str, optional): Flatfield data.
                                                        Defaults to None.
            alien_mask (numpy.ndarray or str, optional): Alien mask data.
                                                         Defaults to None.
        """
        self.experiment_data_dir_path = experiment_data_dir_path
        self.detector_name = detector_name
        self.sample_name = sample_name
        self.h5file = None

        self.flatfield = flatfield
        self.alien_mask = alien_mask

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
            scan: int,
            sample_name: str = None,
            roi: tuple[slice] = None,
            binning_along_axis0: int = None,
            binning_method: str = "sum"
    ) -> None:
        """
        Load detector data for a given scan and sample.

        Args:
            scan (int): Scan number.
            sample_name (str, optional): Name of the sample. Defaults to
                                         None.
            roi (tuple, optional): Region of interest. Defaults to None.
            binning_along_axis0 (int, optional): Binning factor along
                                                 axis 0. Defaults to
                                                 None.
            binning_method (str, optional): Binning method. Defaults to
                                            "sum".

        Returns:
            numpy.ndarray: Loaded detector data.
        """
        if sample_name is None:
            sample_name = self.sample_name

        path = self._get_file_path(scan, sample_name)
        key_path = "entry/data/data_000001"

        if roi is None:
            roi = tuple(slice(None) for i in range(3))
        elif len(roi) == 2:
            roi = tuple([slice(None), roi[0], roi[1]])

        with silx.io.h5py_utils.File(path) as h5file:
            if binning_along_axis0:
                data = h5file[key_path][()]
            else:
                data = h5file[key_path][roi]

        if binning_along_axis0:
            original_dim0 = data.shape[0]
            nb_of_bins = original_dim0 // binning_along_axis0
            first_slices = nb_of_bins * binning_along_axis0
            last_slices = first_slices + original_dim0 % binning_along_axis0

            if binning_method == "sum":
                binned_data = [
                    np.sum(e, axis=0)
                    for e in np.array_split(data[:first_slices], nb_of_bins)
                ]
                if original_dim0 % binning_along_axis0 != 0:
                    binned_data.append(np.sum(data[last_slices:], axis=0))
                data = np.asarray(binned_data)

        if binning_along_axis0 and roi:
            data = data[roi]

        if self.flatfield is not None:
            data = data * self.flatfield[roi[1:]]

        if self.alien_mask is not None:
            data = data * self.alien_mask[roi[1:]]

        return data

    def load_motor_positions(
            self,
            scan: int,
            sample_name: str = None,
            roi: tuple[slice] = None,
            binning_along_axis0: int = None,
            binning_method: str = "mean"
    ) -> None:
        """
        Load motor positions for a given scan and sample.

        Args:
            scan (int): Scan number.
            sample_name (str, optional): Name of the sample. Defaults
                                         to None.
            roi (tuple, optional): Region of interest. Defaults to None.
            binning_along_axis0 (int, optional): Binning factor along
                                                 axis 0. Defaults to
                                                 None.
            binning_method (str, optional): Binning method. Defaults to
                                            "mean".

        Returns:
            dict: Dictionary containing motor positions.
        """
        if sample_name is None:
            sample_name = self.sample_name

        path = self._get_file_path(scan, sample_name, data_type="motor_positions")

        if roi is None or len(roi) == 2:
            roi = slice(None)
        elif len(roi) == 3:
            roi = roi[0]

        angles = {name: None for name in P10Loader.angle_names.values()}

        rocking_angle_values = []

        with open(path, encoding="utf8") as fio_file:
            lines = fio_file.readlines()

            for line in lines:
                line = line.strip()
                words = line.split()

                for name in P10Loader.angle_names.values():
                    if name in words:
                        if "=" in words:
                            angles[name] = float(words[-1])
                        elif "Col" in words:
                            column_index = int(words[1]) - 1
                            rocking_angle = words[2]
            for line in lines:
                line = line.strip()
                words = line.split()

                # check if the first word is numeric, if True the line
                # contains motor position values
                if words[0].replace(".", "", 1).isdigit():
                    rocking_angle_values.append(float(words[column_index]))

            angles[rocking_angle] = np.array(rocking_angle_values)

            if binning_along_axis0:
                original_dim0 = angles[rocking_angle].shape[0]
                nb_of_bins = original_dim0 // binning_along_axis0
                first_slices = nb_of_bins * binning_along_axis0
                last_slices = first_slices + original_dim0 % binning_along_axis0
                if binning_method == "mean":
                    if original_dim0 % binning_along_axis0 != 0:
                        binned_sample_outofplane_angle = [
                            np.mean(e, axis=0)
                            for e in np.split(
                                angles[rocking_angle][:first_slices],
                                nb_of_bins
                            )
                        ]
                        binned_sample_outofplane_angle.append(
                            np.mean(
                                angles[rocking_angle][last_slices-1:],
                                axis=0
                            )
                        )
                    else:
                        binned_sample_outofplane_angle = [
                            np.mean(e, axis=0)
                            for e in np.split(
                                angles[rocking_angle],
                                nb_of_bins
                            )
                        ]
                    angles[rocking_angle] = np.asarray(
                        binned_sample_outofplane_angle
                    )
            if binning_along_axis0 and roi:
                angles[rocking_angle] = angles[rocking_angle][roi]

        return {
            angle: angles[name]
            for angle, name in P10Loader.angle_names.items()
        }

    @staticmethod
    def get_mask(
            channel: int=None,
            detector_name: str="Maxipix",
            roi: tuple[slice]=None
    ) -> np.ndarray:
        """Load the mask of the given detector_name."""
        if roi is None:
            roi = tuple(slice(None) for i in range(3 if channel else 2))

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

        elif detector_name in ("Eiger4M", "eiger4m", "e4m"):
            mask = np.zeros(shape=(2167,2070))
            mask[:, 0:1] = 1
            mask[:, -1:] = 1
            mask[0:1, :] = 1
            mask[-1:, :] = 1
            mask[:, 1029:1041] = 1
            mask[513:552, :] = 1
            mask[1064:1103, :] = 1
            mask[1615:1654, :] = 1
        else:
            raise ValueError("Unknown detector_name")
        if channel:
            return np.repeat(mask[np.newaxis, :, :,], channel, axis=0)[roi]
        return mask[roi]