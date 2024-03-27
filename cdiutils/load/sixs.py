import hdf5plugin
import numpy as np
import silx.io.h5py_utils

from cdiutils.load import Loader


class SIXS2022Loader(Loader):
    """
    A class for loading data from SIXS beamline experiments.
    """

    angle_names = {
        "sample_outofplane_angle": "mu",
        "sample_inplane_angle": "omega",
        "detector_outofplane_angle": "gamma",
        "detector_inplane_angle": "delta"
    }

    def __init__(
            self,
            experiment_data_dir_path: str,
            detector_name: str,
            sample_name: str = None,
            flat_field: np.ndarray | str = None,
            alien_mask: np.ndarray | str = None,
            **kwargs
    ) -> None:
        """
        Initialize SIXSLoader with experiment data directory path and
        detector information.

        Args:
            experiment_data_dir_path (str): path to the experiment data
                directory.
            detector_name (str): name of the detector.
            sample_name (str, optional): name of the sample. Defaults to
                None.
            flat_field (numpy.ndarray or str, optional): flat field to
                account for the non homogeneous counting of the
                detector. Defaults to None.
            alien_mask (np.ndarray | str, optional): array to mask the
                aliens. Defaults to None.
        """
        super(SIXS2022Loader, self).__init__(flat_field, alien_mask)
        self.experiment_data_dir_path = experiment_data_dir_path
        self.detector_name = detector_name
        self.sample_name = sample_name

    def _get_file_path(
            self,
            scan: int,
            sample_name: str,
            data_type: str = "detector_motor_data"
    ) -> str:
        """
        Get the file path based on scan number, sample name, and data
        type. Only works for mu scans (out-of-plane RC).

        Args:
            scan (int): Scan number.
            sample_name (str): Name of the sample.
            data_type (str, optional): Type of data. Defaults to
                                       "detector_data".

        Returns:
            str: File path.
        """
        if data_type == "detector_motor_data":
            return (
                self.experiment_data_dir_path
                + f"/{sample_name}_ascan_mu_{scan:05d}.nxs"
            )

        raise ValueError(
            f"data_type {data_type} is not valid. Must be detector_motor_data."
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
        key_path = "com/scan_data/test_image"

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

        if self.flat_field is not None:
            data = data * self.flat_field[roi[1:]]

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
    ) -> dict:
        """
        Load the motor positions and return it as a dict of:
        - sample out of plane angle
        - sample in plane angle
        - detector out of plane angle
        - detector in plane angle
        """

        if sample_name is None:
            sample_name = self.sample_name

        # key_path = "_".join(
        #         (sample_name, str(scan))
        # ) + ".1/instrument/positioners/"

        path = self._get_file_path(scan, sample_name)
        key_path = "com/scan_data/"

        if roi is None or len(roi) == 2:
            roi = slice(None)
        elif len(roi) == 3:
            roi = roi[0]

        angles = {key: None for key in SIXS2022Loader.angle_names.keys()}

        with silx.io.h5py_utils.File(path) as h5file:
            for angle, name in SIXS2022Loader.angle_names.items():
                if binning_along_axis0:
                    angles[angle] = h5file[key_path + name][()]
                else:
                    try:
                        angles[angle] = h5file[key_path + name][roi]
                    except ValueError:
                        angles[angle] = h5file[key_path + name][()]

        if binning_along_axis0:
            original_dim0 = angles["sample_outofplane_angle"].shape[0]
            nb_of_bins = original_dim0 // binning_along_axis0
            first_slices = nb_of_bins * binning_along_axis0
            last_slices = first_slices + original_dim0 % binning_along_axis0
            if binning_method == "mean":
                if original_dim0 % binning_along_axis0 != 0:
                    binned_sample_outofplane_angle = [
                        np.mean(e, axis=0)
                        for e in np.split(
                                angles["sample_outofplane_angle"][
                                    :first_slices
                                ],
                                nb_of_bins
                            )
                    ]
                    binned_sample_outofplane_angle.append(
                        np.mean(
                            angles["sample_outofplane_angle"][last_slices-1:],
                            axis=0
                        )
                    )
                else:
                    binned_sample_outofplane_angle = [
                        np.mean(e, axis=0)
                        for e in np.split(
                                angles["sample_outofplane_angle"],
                                nb_of_bins
                            )
                    ]
                angles["sample_outofplane_angle"] = np.asarray(
                    binned_sample_outofplane_angle
                )
        if binning_along_axis0 and roi:
            for name, value in angles.items():
                try:
                    angles[name] = value[roi]
                except IndexError:
                    # note that it is not the same error as above
                    continue
        return angles
