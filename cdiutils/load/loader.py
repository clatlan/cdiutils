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
            experiment_file_path: str,
            detector_name: str,
            sample_name: str = None,
            flatfield: np.ndarray | str = None,
            alien_mask: np.ndarray | str = None
    ) -> None:

        self.experiment_file_path = experiment_file_path
        self.detector_name = detector_name
        self.sample_name = sample_name

        self.flatfield = _check_load(flatfield)
        self.alien_mask = _check_load(alien_mask)

        # TODO check whether this is needed
        import os
        self.experiment_root_directory = os.path.dirname(experiment_file_path)
    
    @classmethod
    def from_name(cls, beamline_setup: str) -> None:
         if beamline_setup == "ID01BLISS":
            return BlissLoader(
                experiment_file_path=metadata["experiment_file_path"],
                detector_name=metadata["detector_name"],
                sample_name=metadata["sample_name"],
                flatfield=metadata["flatfield_path"],
                # alien_mask=metadata["alien_mask"]

            )
    if metadata["beamline_setup"] == "ID01SPEC":
        return SpecLoader(
            experiment_file_path=metadata["experiment_file_path"],
            detector_data_path=metadata["detector_data_path"],
            edf_file_template=metadata["edf_file_template"],
            detector_name=metadata["detector_name"]
        )
    if metadata["beamline_setup"] == "SIXS2022":
        return SIXS2022Loader(
            experiment_data_dir_path=metadata["experiment_data_dir_path"],
            detector_name=metadata["detector_name"],
            sample_name=metadata["sample_name"],
        )
    if metadata["beamline_setup"] == "P10":
        return P10Loader(
            experiment_data_dir_path=metadata["experiment_data_dir_path"],
            detector_name=metadata["detector_name"],
            sample_name=metadata["sample_name"],
        )
    raise NotImplementedError("The provided beamline_setup is not valid.")



    def _check_load(data_or_path: np.ndarray | str) -> np.ndarray:
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
        elif isinstance(data_or_path, (np.ndarray, None)):
            return data_or_path
        else:
            raise ValueError(
                "[ERROR] wrong value for flatfield parameter, provide a path, "
                "np.ndarray or leave it to None"
            )

    @staticmethod
    def get_mask(
            channel: int = None,
            detector_name: str = "Maxipix"
    ) -> np.ndarray:
        """
        Load the mask of the given detector_name.

        Args:
            channel (int, optional): the size of the third (axis0)
                dimension. Defaults to None (2D in that case).
            detector_name (str, optional): The name of the detector.
                Defaults to "Maxipix".

        Raises:
            ValueError: If detector name is unknown or not implemented
                yet.

        Returns:
            np.ndarray: the 2D or 3D mask.
        """

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
        elif detector_name in ("Eiger4M", "eiger4m", "eiger4M", "Eiger4m"):
            pass
        
        else:
            raise ValueError("Unknown detector_name")
        if channel:
            return np.repeat(mask[np.newaxis, :, :,], channel, axis=0)
        return mask