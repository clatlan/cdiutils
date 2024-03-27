"""
A generic class for loaders
"""

import numpy as np
from typing import Callable
import silx.io.h5py_utils


def safe(func: Callable) -> Callable:
    """A wrapper to safely load data in h5 file"""
    def wrap(self, *args, **kwargs):
        with silx.io.h5py_utils.File(self.experiment_file_path) as self.h5file:
            return func(self, *args, **kwargs)
    return wrap


class Loader:
    def __init__(
            self,
            flat_field: np.ndarray | str = None,
            alien_mask: np.ndarray | str = None
    ) -> None:
        """
        The gerenic parent class for all loaders.

        Args:
            flat_field (np.ndarray | str, optional): flat field to
                account for the non homogeneous counting of the
                detector. Defaults to None.
            alien_mask (np.ndarray | str, optional): array to mask the
                aliens. Defaults to None.
        """

        self.flat_field = self._check_load(flat_field)
        self.alien_mask = self._check_load(alien_mask)

    @classmethod
    def from_setup(cls, beamline_setup: str, metadata: dict) -> "Loader":
        """
        Instantiate a child loader class given a the setup name,
        following the Factory Pattern.

        Args:
            beamline_setup (str): the name of the beamline setup.
            metadata (dict): the parameters defining the experimental
                setup.

        Raises:
            ValueError: If the beamline setup is invalid.

        Returns:
            Loader: the subclass loader according to the provided name.
        """
        if beamline_setup == "ID01BLISS":
            from . import BlissLoader
            return BlissLoader(**metadata)
        if beamline_setup == "ID01SPEC":
            from . import SpecLoader
            return SpecLoader(**metadata)
        if beamline_setup == "SIXS2022":
            from . import SIXS2022Loader
            return SIXS2022Loader(**metadata)
        if beamline_setup == "P10":
            from . import P10Loader
            return P10Loader(**metadata)
        raise ValueError(f"Invalid beamline setup: {beamline_setup}")

    @staticmethod
    def _check_load(self, data_or_path: np.ndarray | str) -> np.ndarray:
        """
        Private method to load mask or alien np.ndarray.

        Args:
            data_or_path (np.ndarray | str): the data or the path of the
                data.

        Raises:
            ValueError: If path or np.ndarray is not provided.

        Returns:
            np.ndarray: the numpy array.
        """
        if isinstance(data_or_path, str):
            if data_or_path.endswith(".npy"):
                return np.load(data_or_path)
            elif data_or_path.endswith(".npz"):
                with np.load(data_or_path, "r") as file:
                    return file["arr_0"]
        elif data_or_path is None or isinstance(data_or_path, np.ndarray):
            return data_or_path
        raise ValueError(
            "[ERROR] wrong value for flat_field and/or alien_mask "
            "parameter provide a path, np.ndarray or leave it to None"
        )

    @staticmethod
    def get_mask(
            channel: int = None,
            detector_name: str = "Maxipix",
            roi: tuple[slice] = None
    ) -> np.ndarray:
        """
        Load the mask of the given detector_name.

        Args:
            channel (int, optional): the size of the third (axis0)
                dimension. Defaults to None (2D in that case).
            detector_name (str, optional): The name of the detector.
                Defaults to "Maxipix".
            roi (tuple, optional): the region of interest associated to
                the data. Defaults to None.

        Raises:
            ValueError: If detector name is unknown or not implemented
                yet.

        Returns:
            np.ndarray: the 2D or 3D mask.
        """
        if roi is None:
            roi = tuple(slice(None) for i in range(3 if channel else 2))

        if detector_name in (
            "maxipix", "Maxipix", "mpxgaas", "mpx4inr", "mpx1x4"
        ):
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
            mask = np.zeros(shape=(2167, 2070))
            mask[:, 0:1] = 1
            mask[:, -1:] = 1
            mask[0:1, :] = 1
            mask[-1:, :] = 1
            mask[:, 1029:1041] = 1
            mask[513:552, :] = 1
            mask[1064:1103, :] = 1
            mask[1615:1654, :] = 1

        else:
            raise ValueError(f"Invalid detector name: {detector_name}")
        if channel:
            return np.repeat(mask[np.newaxis, :, :,], channel, axis=0)[roi]
        return mask[roi]
