"""A generic class for loaders."""

from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import silx.io.h5py_utils

from cdiutils.utils import CroppingHandler, get_centred_slices
from cdiutils.plot import add_colorbar


def h5_safe_load(func: Callable) -> Callable:
    """A wrapper to safely load data in h5 file"""
    def wrap(self, *args, **kwargs):
        with silx.io.h5py_utils.File(self.experiment_file_path) as self.h5file:
            return func(self, *args, **kwargs)
    return wrap


class Loader:
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
        if "P10" in beamline_setup:
            from . import P10Loader
            if beamline_setup == "P10EH2":
                return P10Loader(hutch="EH2", **metadata)
            else:
                return P10Loader(**metadata)
        raise ValueError(f"Invalid beamline setup: {beamline_setup}")

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
                    return file["arr_0"]
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
            "Wrong value for roi (roi={}), roi should be:\n"
            "\t - either a tuple of slices with len = 2 or len = 3"
            "\t - either a tuple of int with len = 4 or len = 6"
        )
        if roi is None:
            return tuple(slice(None) for _ in range(3))
        if len(roi) == 2 or len(roi) == 3:
            if all(isinstance(e, slice) for e in roi):
                return (slice(None), roi[0], roi[1])
        if len(roi) == 4 or len(roi) == 6:
            if all(isinstance(e, int) for e in roi):
                return CroppingHandler.roi_list_to_slices(roi)
        raise ValueError(usage_text.format(roi))

    def bin_flat_mask(
            self,
            data: np.ndarray,
            roi: list = None,
            binning_along_axis0: int = None,
            binning_method: str = "sum",
    ) -> np.ndarray:
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
        
        elif detector_name.lower() == "eiger500k":
            return None

        else:
            raise ValueError(f"Invalid detector name: {detector_name}")
        if channel:
            return np.repeat(mask[np.newaxis, :, :,], channel, axis=0)[roi]
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
