"""A generic class for loaders."""

from abc import ABC, abstractmethod
from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import silx.io.h5py_utils

from cdiutils.utils import (
    CroppingHandler,
    get_centred_slices,
    bin_along_axis
)
from cdiutils.plot import add_colorbar


class Loader(ABC):
    """A generic class for loaders."""

    def __init__(
            self,
            flat_field: np.ndarray | str = None,
            alien_mask: np.ndarray | str = None
    ) -> None:
        """
        The generic parent class for all loaders.

        Args:
            flat_field (np.ndarray | str, optional): flat field to
                account for the non homogeneous counting of the
                detector. Defaults to None.
            alien_mask (np.ndarray | str, optional): array to mask the
                aliens. Defaults to None.
        """
        self.flat_field = self._check_load(flat_field)
        self.alien_mask = self._check_load(alien_mask)
        self.detector_name = None
        self.rocking_angle = "sample_outofplane_angle"

    def get_alien_mask(
            self,
            roi: tuple[slice, slice, slice] = None
    ) -> np.ndarray:
        if self.alien_mask is None:
            return None

        if roi is None:
            return self.alien_mask

        return self.alien_mask[roi]

    @classmethod
    def from_setup(cls, beamline_setup: str, **metadata) -> "Loader":
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
        if "id01" in beamline_setup.lower():
            if beamline_setup.lower() == "id01spec":
                from . import SpecLoader
                return SpecLoader(**metadata)
            from . import ID01Loader
            return ID01Loader(**metadata)

        if beamline_setup.lower() == "id01spec":
            from . import SpecLoader
            return SpecLoader(**metadata)

        if beamline_setup.lower() == "sixs2022":
            from . import SIXS2022Loader
            return SIXS2022Loader(**metadata)

        if "p10" in beamline_setup.lower():
            from . import P10Loader
            if beamline_setup.lower() == "p10eh2":
                return P10Loader(hutch="EH2", **metadata)
            return P10Loader(**metadata)

        if beamline_setup.lower() == "cristal":
            from . import CristalLoader
            return CristalLoader(**metadata)

        if beamline_setup.lower() == "nanomax":
            from . import NanoMAXLoader
            return NanoMAXLoader(**metadata)

        if beamline_setup.lower() == "id27":
            from . import ID27Loader
            return ID27Loader(**metadata)
        raise ValueError(f"Invalid beamline setup: {beamline_setup = }")  # noqa, E251

    @staticmethod
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
            if data_or_path.endswith(".npz"):
                with np.load(data_or_path, "r") as file:
                    for possible_key in (
                            "arr_0", "data", "mask", "flatfield", "flat_field"
                    ):
                        if possible_key in dict(file):
                            return file[possible_key]
                    raise KeyError(
                        f"Invalid file provided containing {file.keys()}."
                    )
        elif data_or_path is None or isinstance(data_or_path, np.ndarray):
            return data_or_path
        raise ValueError(
            "[ERROR] wrong value for flat_field and/or alien_mask "
            "parameter provide a path, np.ndarray or leave it to None"
        )

    @staticmethod
    def _check_roi(roi: tuple = None) -> tuple[slice]:
        """
        Utility function to check if a region of interest (roi) was
        parsed correctly.

        Args:
            roi (tuple, optional): the roi, a tuple of slices.
                len = 2 or len = 3 if tuple of slices. len = 4 or
                len = 6 if tuple of int. Defaults to None.

        Raises:
            ValueError: if roi does not correspond to tuple of slices
                with len 2 or 3 or tuple of int wit len 4 or 6.

        Returns:
            tuple[slice]: the prepared roi.
        """
        usage_text = (
            f"Wrong value for roi ({roi = }), roi should be:\n"  # noqa, E251
            "\t - either a tuple of slices with len = 2 or len = 3"
            "\t - either a tuple of int with len = 4 or len = 6"
        )
        if roi is None:
            return tuple(slice(None) for _ in range(3))
        if all(isinstance(e, slice) for e in roi):
            if len(roi) == 2:
                return (slice(None), roi[0], roi[1])
            elif len(roi) == 3:
                return roi
        if len(roi) == 4 or len(roi) == 6:
            if all(isinstance(e, (int, np.integer)) for e in roi):
                return CroppingHandler.roi_list_to_slices(roi)
        raise ValueError(usage_text)

    @staticmethod
    def bin_flat_mask(
            data: np.ndarray,
            roi: list = None,
            flat_field: np.ndarray = None,
            alien_mask: np.ndarray = None,
            rocking_angle_binning: int = None,
            binning_method: str = "sum",
    ) -> np.ndarray:
        """
        A generic method that takes care of binning, applying flat_field
        and alien mask to detector data.

        Args:
            data (np.ndarray): the data to bin, apply flat_field and
                mask.
            roi (list, optional): the region of interest to select.
                Defaults to None.
            flat_field (np.ndarray, optional): the flat field to apply.
                Defaults to None.
            alien_mask (np.ndarray, optional): the alien mask to apply.
                Defaults to None.
            rocking_angle_binning (int, optional): the binning factor
                along to rocking curve axis. Defaults to None.
            binning_method (str, optional): the method for the binning.
                Defaults to "sum".

        Returns:
            np.ndarray: the new data
        """
        if roi is None:
            roi = (slice(None), slice(None), slice(None))

        if rocking_angle_binning:
            data = bin_along_axis(
                data, rocking_angle_binning, binning_method, axis=0
            )
            # If binning, roi[1] and roi[2] have been applied already.
            data = data[roi[0]]

        if flat_field is not None:
            data = data * flat_field[roi[1:]]

        if alien_mask is not None:
            data = data * (1 - alien_mask[roi])
        return data

    @staticmethod
    def bin_rocking_angle_values(
            values: list | np.ndarray,
            binning_factor: int = None
    ) -> np.ndarray:
        """
        Bins the data along the rocking angle axis using the
        bin_along_axis function.

        Args:
            values (list | np.ndarray): the rocking angle values to be
                binned.
            binning_factor (int, optional): the number of data points to
                bin along the rocking angle axis. Defaults to None.

        Returns:
            np.ndarray: the binned values.
        """
        return bin_along_axis(values, binning_factor, binning_method="mean")

    @abstractmethod
    def load_energy(self):
        pass

    @abstractmethod
    def load_det_calib_params(self):
        pass

    def load_detector_shape(self):
        return None

    def get_detector_name(self) -> str:
        """By default, return the first authorised name of the class."""
        return self.authorised_detector_names[0]

    @staticmethod
    def get_rocking_angle(angles) -> str:
        outofplane = angles.get("sample_outofplane_angle")
        inplane = angles.get("sample_inplane_angle")

        if outofplane is not None and inplane is not None:
            if (
                    isinstance(outofplane, (np.ndarray, list))
                    and len(outofplane) > 1
            ):
                return "sample_outofplane_angle"
            if isinstance(inplane, ((np.ndarray, list))) and len(inplane) > 1:
                return "sample_inplane_angle"
            raise ValueError(
                "Could not find a rocking angle "
                f"({outofplane = }, {inplane = })"  # noqa, E251
            )
        raise ValueError(
            "sample_outofplane_angle and/or sample_inplane_angle missing in "
            "the provided angles dictionary."
        )

    @classmethod
    def get_mask(
            cls,
            detector_name: str = None,
            channel: int = None,
            roi: tuple[slice] = None
    ) -> np.ndarray:
        """
        Load the mask of the given detector_name.

        Args:
            channel (int, optional): the size of the third (axis0)
                dimension. Defaults to None (2D in that case).
            detector_name (str, optional): The name of the detector.
                Defaults to None.
            roi (tuple, optional): the region of interest associated to
                the data. Defaults to None.

        Raises:
            ValueError: If detector name is unknown or not implemented
                yet.

        Returns:
            np.ndarray: the 2D or 3D mask.
        """
        if detector_name is None:
            # Handling the case whenever the method is called as
            # a static method.
            local_params = locals()
            if isinstance(local_params[0], cls):
                detector_name = cls.detector_name
            else:
                raise ValueError(
                    "When called as a static method, detector_name must be "
                    "provided."
                )
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

        # Having a name such as "eiger" is super bad, it's id27...
        elif detector_name.lower() in ("eiger9m", "e9m", "eiger"):
            mask = np.zeros(shape=(3262, 3108))
            mask[:, 0:1] = 1
            mask[:, -1:] = 1
            mask[0:1, :] = 1
            mask[-1:, :] = 1
            mask[:, 513:515] = 1
            mask[:, 1028:1040] = 1
            mask[:, 1553:1555] = 1
            mask[:, 2068:2080] = 1
            mask[:, 2593:2595] = 1
            mask[512:550, :] = 1
            mask[1062:1100, :] = 1
            mask[1612:1650, :] = 1
            mask[2162:2200, :] = 1
            mask[2712:2750, :] = 1

        elif detector_name.lower() == "eiger500k":
            mask = np.zeros(shape=(512, 1028))
        elif detector_name.lower() == "merlin":
            mask = np.zeros(shape=(512, 512))
        else:
            raise ValueError(f"Invalid detector name: {detector_name}")
        if channel:
            mask = np.repeat(mask[np.newaxis, :, :,], channel, axis=0)
        return mask[roi]

    @staticmethod
    def plot_detector_data(
            data: np.ndarray,
            title: str = None,
            return_fig: bool = False,
            equal_limits: bool = False,
            **plot_params
    ) -> plt.Figure:
        _plot_params = {
            "norm": LogNorm(1),
            "origin": "upper",
            "cmap": "turbo"  # "PuBu_r"
        }
        if plot_params:
            _plot_params.update(plot_params)

        if data.ndim == 3:
            limits = [
                (s/2 - np.max(data.shape)/2, s/2 + np.max(data.shape)/2)
                for s in data.shape
            ]
            slices = get_centred_slices(data.shape)
            planes = ((1, 2), (0, 2), (1, 0))  # indexing convention

            fig, axes = plt.subplots(2, 3, layout="tight", figsize=(6, 4))
            for i, p in enumerate(planes):
                axes[0, i].imshow(
                    (
                        np.swapaxes(data[slices[i]], 0, 1)
                        if p[0] > p[1] else data[slices[i]]
                    ),
                    **_plot_params
                )
                axes[1, i].imshow(
                    (
                        np.swapaxes(data.sum(axis=i), 0, 1)
                        if p[0] > p[1] else data.sum(axis=i)
                    ),
                    **_plot_params,
                )
                for ax in (axes[0, i], axes[1, i]):
                    add_colorbar(ax, ax.images[0])
                    if equal_limits:
                        ax.set_xlim(limits[p[1]])
                        if _plot_params["origin"] == "upper":
                            ax.set_ylim(limits[p[0]][1], limits[p[0]][0])
                        ax.set_ylim(limits[p[0]])

            for i in range(2):
                axes[i, 0].set_xlabel(r"axis$_{2}$, det. horiz.")
                axes[i, 0].set_ylabel(r"axis$_{1}$, det. vert.")

                axes[i, 1].set_xlabel(r"axis$_{2}$, det. horiz.")
                axes[i, 1].set_ylabel(r"axis$_{0}$, rocking curve")

                axes[i, 2].set_xlabel(r"axis$_{0}$, rocking curve")
                axes[i, 2].set_ylabel(r"axis$_{1}$, det. vert.")

            axes[0, 1].set_title("Intensity slice")
            axes[1, 1].set_title("Intensity sum")
            fig.suptitle(title)
            if return_fig:
                return fig
            return None
        elif data.ndim == 2:
            pass
        raise ValueError(
            f"Invalid data shape (detector_data.shape={data.shape})."
            "Should be 2D or 3D."
        )


def h5_safe_load(func: Callable) -> Callable:
    """A wrapper to safely load data in h5 file"""
    def wrap(self, *args, **kwargs):
        with silx.io.h5py_utils.File(self.experiment_file_path) as self.h5file:
            return func(self, *args, **kwargs)
    return wrap


class H5TypeLoader(Loader):
    def __init__(
            self,
            experiment_file_path: str,
            sample_name: str = None,
            detector_name: str = None,
            flat_field: np.ndarray | str = None,
            alien_mask: np.ndarray | str = None,
    ) -> None:
        super().__init__(flat_field, alien_mask)
        self.experiment_file_path = experiment_file_path
        self.sample_name = sample_name
        if detector_name is None:
            self.detector_name = self.get_detector_name()
            # if sample_name is not None:
            #     self.detector_name = self.get_detector_name()
            #     print(
            #         "Detector name automatically found "
            #         f"('{self.detector_name}')."
            #     )
            # else:
            #     print(
            #         "detector_name is not provided, cannot automatically find "
            #         "it since sample_name is not provided either.\n"
            #         "Will set detector_name to "
            #         f"'{self.authorised_detector_names[0]}'."
            #     )
            #     self.detector_name = self.authorised_detector_names[0]
        else:
            self.detector_name = detector_name

    @h5_safe_load
    def load_angles(self, key_path: str) -> dict:
        angles = {}
        for name in self.angle_names.values():
            if name is not None:
                angles[name] = self.h5file[key_path + name][()]

        return angles
