import copy
import glob
import os
import re
import shutil
from typing import Type

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import silx.io
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fft import fftn, fftshift, ifftshift
from scipy.ndimage import fourier_shift
from scipy.stats import gaussian_kde
from skimage.registration import phase_cross_correlation

try:
    from pynx.cdi import (
        CDI,
        ER,
        HIO,
        RAAR,
        AutoCorrelationSupport,
        DetwinHIO,
        DetwinRAAR,
        FourierApplyAmplitude,
        InitPSF,
        InitSupportShape,
        ScaleObj,
        SupportTooLarge,
        SupportTooSmall,
        SupportUpdate,
    )
    from pynx.cdi.selection import match2
    from pynx.utils.math import ortho_modes

    IS_PYNX_AVAILABLE = True
    CDI_Type = Type[CDI]

except ImportError:
    IS_PYNX_AVAILABLE = False
    CDI_Type = None

from cdiutils.plot import add_colorbar, get_plot_configs
from cdiutils.process.postprocessor import PostProcessor
from cdiutils.utils import CroppingHandler, get_centred_slices, valid_args_only

DEFAULT_PYNX_PARAMS = {
    # support-related params
    "support_threshold": (0.15, 0.40),
    "smooth_width": (2, 0.5, 900),  # (start, end, period)
    "post_expand": None,  # (-1, 1)
    "support_update_period": 50,
    "method": "rms",
    "force_shrink": False,
    "update_border_n": 0,
    "support": None,
    "obj": None,
    "amp_range": (0, 1),
    "scale_obj": "I",
    "phase_range": (-np.pi, np.pi),
    "all_random": False,
    "support_shape": None,
    "support_formula": None,
    "support_size": None,
    # object initialisation
    # "support": "auto",
    "support_autocorrelation_threshold": (0.09, 0.11),
    # operator parameters
    "beta": 0.9,  # HIO/RAAR feedback, as in pipeline/parameters.py
    "calc_llk": 50,
    "show_cdi": 0,
    "update_psf": 100,
    "fig_num": -1,
    "zero_mask": False,
    # others
    "psf": "pseudo-voigt,1,0.05",  # model,fwhm,eta
    "compute_free_llk": True,
    "positivity": False,
    "confidence_interval_factor_mask_min": 0.5,
    "confidence_interval_factor_mask_max": 1.2,
    "detwin": False,
    # TV regularisation
    "tv_er": 0.15,
    "tv_hio": 0.15,
    "tv_raar": 0.15,
    # extensions: per-run phase perturbation of a given
    # starting object, upper bound of the support, amplitude
    # homogenisation during HIO/RAAR (fraction pulled to the mean)
    "obj_noise": 0,
    "support_ceiling": None,
    "homogenise": 0,
}


class PyNXImportError(ImportError):
    """Custom exception to handle Pynx import error."""

    def __init__(self, msg: str = None) -> None:
        _msg = "'PyNX' is not installed on the current machine."
        if msg is not None:
            _msg += "\n" + msg
        super().__init__(_msg)


class PyNXPhaser:
    """
    A class for using PyNX's phasing algorithms without worrying
    too much about the initialisation and the parameters. This class
    uses generic PyNX parameters but they can also be provided upon
    instanciation.
    """

    def __init__(
        self,
        iobs: np.ndarray,
        mask: np.ndarray = None,
        operators: dict = None,
        **params,
    ) -> None:
        """
        Initialisation method.

        Args:
            iobs (np.ndarray): the observed intensity.
            mask (np.ndarray, optional): the mask, can include detector
                gap mask/hot pixels/aliens. Defaults to None.
            operators (dict, optional): the operators you might want to
                use. If not provided, ER, HIO and RAAR will be
                initialised anyway. Defaults to None.
            params (dict, optional): the PyNX parameters. If not
                provided, will use some generic parameters.

        Raises:
            ModuleNotFoundError: if PyNX is not installed, this class
             is of no use.
        """
        if not IS_PYNX_AVAILABLE:
            raise PyNXImportError
        self.iobs = fftshift(iobs)

        self.mask = None
        if mask is not None:
            self.mask = fftshift(mask)

        # Get the pynx parameters from the default and update them with
        # those provided by the user
        # not elegant, but works...
        self.params = copy.deepcopy(DEFAULT_PYNX_PARAMS)
        if params is not None:
            self.params.update(params)

        if operators is None:
            self.operators = self._init_operators()
        else:
            self.operators = operators

        self.support_update: SupportUpdate
        self._init_support_update()
        self.support_threshold_auto_tune_factor: float = 1.1

        self.cdi: CDI_Type = None
        self.init_cdi_params = dict
        self.cdi_list: list[CDI_Type] = []

        self.wrong_support_failure_tolerance = 5

        self.figure: matplotlib.figure.Figure = None

    def _init_operators(self) -> dict:
        """Initialise generic PynX operators."""
        operator_parameters = {
            key: self.params[key]
            for key in [
                "beta",
                "calc_llk",
                "show_cdi",
                "update_psf",
                "fig_num",
                "zero_mask",
                "confidence_interval_factor_mask_max",
                "confidence_interval_factor_mask_min",
            ]
        }
        return {
            # only the arguments each operator accepts (ER has no beta)
            "er": ER(
                **valid_args_only(operator_parameters, ER),
                tv=self.params["tv_er"],
            ),
            "hio": HIO(
                **valid_args_only(operator_parameters, HIO),
                tv=self.params["tv_hio"],
            ),
            "detwinhio": DetwinHIO(detwin_axis=1),
            "raar": RAAR(
                **valid_args_only(operator_parameters, RAAR),
                tv=self.params["tv_raar"],
            ),
            "detwinraar": DetwinRAAR(detwin_axis=1),
            "faa": FourierApplyAmplitude(
                **valid_args_only(operator_parameters, FourierApplyAmplitude)
            ),
        }

    def _init_support_update(self) -> None:
        """
        Initialise the SupportUpdate class from pynx, depending on the
        parameters of the instance.
        """

        # load default pynx values for missing parameters
        support_params = {
            k: self.params.get(k, default)
            for k, default in (
                ("threshold_relative", 0.2),
                ("smooth_width", 3),
                ("force_shrink", False),
                ("method", "rms"),
                ("post_expand", None),
                ("verbose", False),
                ("update_border_n", 0),
                ("min_fraction", 0),
                ("max_fraction", 1),
                ("lazy", False),
            )
        }

        # if support_threshold is provided, use it to set the
        # threshold_relative, unless threshold_relative is already provided
        if (
            "support_threshold" in self.params
            and "threshold_relative" not in self.params
        ):
            # take care of different types
            if isinstance(self.params["support_threshold"], (float, int)):
                support_params["threshold_relative"] = self.params[
                    "support_threshold"
                ]
            # if given as a string or list/tuple, assume it's a range
            elif isinstance(self.params["support_threshold"], str):
                # assuming it's a range in string format "min, max"
                support_params["threshold_relative"] = np.random.uniform(
                    *(
                        float(x)
                        for x in self.params["support_threshold"].split(",")
                    )
                )
            elif isinstance(self.params["support_threshold"], (list, tuple)):
                support_params["threshold_relative"] = np.random.uniform(
                    *self.params["support_threshold"]
                )

        print("Using support update parameters:", support_params)
        if self.params["support_update_period"]:
            self.support_update = SupportUpdate(**support_params)
        else:
            self.support_update = 1

    def init_cdi(
        self, verbose: bool = True, init_main_cdi: bool = True, **params
    ) -> CDI_Type:
        """
        Initialise the CDI object.

        Args:
            verbose (bool, optional): whether to print the
                initialisation steps. Defaults to True.
            init_main_cdi (bool, optional): whether to set the cdi
                attribute or not. Defaults to True.
            **params (optional): the parameters to update the
                default ones with.

        Returns:
            CDI_Type: the initialised CDI object.
        """
        if params is not None:
            for key in ["support", "obj"]:
                if key in params and isinstance(params[key], np.ndarray):
                    if verbose:
                        print(f"fftshifting {key}")
                    params[key] = fftshift(params[key])
            self.params.update(params)

        if self.params["positivity"]:
            self.params["phase_range"] = 0

        for key in ["amp_range", "phase_range"]:
            if isinstance(self.params[key], (int, float)):
                self.params[key] = (
                    -self.params[key] if key == "phase_range" else 0,
                    self.params[key],
                )

        cdi = CDI(
            self.iobs, self.params["support"], self.params["obj"], self.mask
        )
        if self.params["support"] is None and self.params["obj"] is None:
            if self.params["all_random"]:
                if verbose:
                    print("Full random initialisation requested.")
                amp = np.random.uniform(
                    low=self.params["amp_range"][0],
                    high=self.params["amp_range"][1],
                    size=self.iobs.size,
                ).reshape(self.iobs.shape)
                phase = np.random.uniform(
                    low=self.params["phase_range"][0],
                    high=self.params["phase_range"][1],
                    size=self.iobs.size,
                ).reshape(self.iobs.shape)
                cdi.set_obj(amp * np.exp(1j * phase))
            elif self.params["support_shape"] is not None:
                cdi = (
                    InitSupportShape(
                        shape=self.params["support_shape"],
                        size=self.params.get("support_size"),
                        formula=self.params.get("support_formula"),
                    )
                    * cdi
                )
            else:
                if verbose:
                    print("Support will be initialised using autocorrelation.")
                cdi = (
                    AutoCorrelationSupport(
                        threshold=self.params[
                            "support_autocorrelation_threshold"
                        ]
                    )
                    * cdi
                )
        if self.params["obj"] is None and not self.params["all_random"]:
            if verbose:
                print(
                    "obj is None, will initialise object with support, and:"
                    "\n\t-amp_range: "
                    + (
                        "scale to observed I"
                        if self.params["scale_obj"]
                        else f"{self.params['amp_range']}"
                    )
                    + f"\n\t-phase_range: {self.params['phase_range']}"
                )
            # Let's initialise the object from the support
            current_support = cdi.get_support()
            obj = current_support.astype(np.complex64)
            amp = np.random.uniform(
                low=self.params["amp_range"][0],
                high=self.params["amp_range"][1],
                size=np.flatnonzero(current_support).size,
            )
            phase = np.random.uniform(
                low=self.params["phase_range"][0],
                high=self.params["phase_range"][1],
                size=np.flatnonzero(current_support).size,
            )
            obj[current_support > 0] *= amp * np.exp(1j * phase)
            cdi.set_obj(obj)
            # cdi = InitObjRandom(
            #     amax=self.params["amp_range"],
            #     phirange=self.params["phase_range"]
            # ) * cdi
        if self.params["obj"] is not None and self.params["obj_noise"]:
            # each run starts from the given object with its own random
            # phase perturbation (obj_noise, rad rms), so that seeded runs
            # differ and can be compared
            obj = cdi.get_obj()
            noise = (
                np.random.standard_normal(obj.shape) * self.params["obj_noise"]
            )
            cdi.set_obj((obj * np.exp(1j * noise)).astype(np.complex64))
        if self.params["scale_obj"]:
            cdi = (
                ScaleObj(method=self.params["scale_obj"], verbose=verbose)
                * cdi
            )
        if isinstance(self.params["psf"], np.ndarray):
            # a PSF array given by the user (centred), e.g. bandwidth_psf();
            # padded here with its centre at n//2: PyNX's own padding of a
            # small odd kernel into an even axis shifts the origin by 1 px
            psf = self.pad_centred(
                self.params["psf"], ifftshift(self.iobs).shape
            )
            cdi = InitPSF(psf=psf.astype(np.float32)) * cdi
        elif self.params["psf"] is not None:
            model, fwhm, eta = self.params["psf"].split(",")
            cdi = InitPSF(model, float(fwhm), float(eta)) * cdi

        if self.params["compute_free_llk"]:
            cdi.init_free_pixels()

        if init_main_cdi:
            self.cdi = cdi
        return cdi

    @classmethod
    def read_instructions(cls, recipe: str) -> list[str]:
        """Read the instructions given in the recipe."""
        recipe = recipe.replace(" ", "")
        instructions = recipe.split(",")
        if "=" in instructions:
            pass
        return instructions

    def run(
        self, recipe: str, cdi: CDI_Type = None, init_cdi: bool = True
    ) -> None:
        """
        Run the reconstruction algorithm.

        Args:
            recipe (str): the instruction to run, i.e. the sequence of
                projection algorithms (ex:
                "ER**200, HIO**400, RAAR**800")
            cdi (CDI_Type, optional): a cdi object you might to work on,
                if not provided, will used the attribute self.cdi.
                Defaults to None.
            init_cdi (bool, optional): whether to initialise the cdi
                object or not. Defaults to True.

        Raises:
            ValueError: if instruction is not authorised.
            ValueError: if instruction is invalid
        """
        if cdi is None:
            cdi = self.cdi
            if init_cdi:
                cdi = self.init_cdi(init_main_cdi=False)

        instructions = self.read_instructions(recipe)
        for i, instruction in enumerate(instructions):
            print(f"Instruction #{i + 1}: {instruction}")

            if "**" in instruction:
                algo_name, iteration = instruction.split("**")
                iteration = int(iteration)

                try:
                    algo = self.operators[algo_name.lower()]
                except KeyError as exc:
                    raise ValueError(
                        f"Invalid operator name '{algo_name}', should be:\n"
                        f"{list(self.operators.keys())}."
                    ) from exc
                attempt = 1
                while attempt < self.wrong_support_failure_tolerance:
                    try:
                        support_update_period = (
                            self.params["support_update_period"]
                            if self.params["support_update_period"]
                            else 1
                        )
                        if (
                            self.params["support_ceiling"] is None
                            and not self.params["homogenise"]
                        ):
                            cdi = (
                                self.support_update
                                * (algo**support_update_period)
                                ** (iteration // support_update_period)
                            ) * cdi
                        else:
                            for _ in range(iteration // support_update_period):
                                cdi = (
                                    self.support_update
                                    * algo**support_update_period
                                ) * cdi
                                cdi = self._apply_constraints(
                                    cdi, algo_name.lower()
                                )
                        attempt = self.wrong_support_failure_tolerance

                    except SupportTooLarge:
                        print(
                            "Support is too large, shrinking it. "
                            f"Attempt #{attempt + 1}"
                        )
                        self.support_update.threshold_relative /= (
                            self.support_threshold_auto_tune_factor
                        )
                        attempt += 1
                    except SupportTooSmall:
                        print(
                            "Support is too small, making it bigger. "
                            f"Attempt #{attempt + 1}"
                        )
                        self.support_update.threshold_relative *= (
                            self.support_threshold_auto_tune_factor
                        )
                        attempt += 1
            elif instruction.lower() == "faa":
                cdi = self.operators["faa"] * cdi
            else:
                raise ValueError(f"Invalid instruction ({instruction}).")

    def run_multiple_instances(
        self, run_nb: int, recipe: str, init_cdi: bool = True
    ) -> None:
        """
        Run several reconstructions.

        Args:
            run_nb (int): the number of reconstructions
            recipe (str): the instruction to run, i.e. the sequence of
                projection algorithms (ex:
                "ER**200, HIO**400, RAAR**800")
            init_cdi (bool, optional):  whether to initialise the cdi
                object or not. Defaults to True.

        Raises:
            ValueError: if init_cdi is False but no initialisation was
                done with the init_cdi() method beforehand.
        """
        if init_cdi:
            self.cdi_list = []
            for i in range(run_nb):
                self.cdi_list.append(
                    self.init_cdi(init_main_cdi=False, verbose=False)
                )
        else:
            if self.cdi_list is None or not self.cdi_list:
                raise ValueError(
                    "CDI object are not initialised, init_cdi should be True."
                )

        # if support_threshold is given as a list/tuple/string, set a different
        # threshold for each run
        support_relative_must_be_updated = False
        if isinstance(self.params["support_threshold"], (list, tuple, str)):
            support_relative_must_be_updated = True

        for i, cdi in enumerate(self.cdi_list):
            print(f"Run #{i + 1}")

            # set support threshold if needed
            if support_relative_must_be_updated:
                if isinstance(self.params["support_threshold"], str):
                    threshold = np.random.uniform(
                        *(
                            float(x)
                            for x in self.params["support_threshold"].split(
                                ","
                            )
                        )
                    )
                else:
                    threshold = np.random.uniform(
                        *self.params["support_threshold"]
                    )
                self.support_update.threshold_relative = threshold
                print(f"Support threshold set to {threshold}")

            # run the recipe
            self.run(recipe, cdi, init_cdi=False)

    def _apply_constraints(self, cdi: CDI_Type, algo_name: str) -> CDI_Type:
        """
        Constraints applied after every support update period.

        - support_ceiling (natural orientation, same shape as the data):
          the support may only shrink inside it (shrink-wrap bounded by
          a model hull).
        - homogenise (0-1, HIO and RAAR only): the modulus inside the
          support is pulled towards its mean by this fraction, the phase
          is kept (modulus homogenisation, Newton et al. 2010; useful
          when the modulus collapses into part of a known support).

        Args:
            cdi (CDI_Type): the cdi object.
            algo_name (str): the current algorithm.

        Returns:
            CDI_Type: the constrained cdi object.
        """
        ceiling = self.params["support_ceiling"]
        if ceiling is not None:
            support = cdi.get_support(shift=True) > 0
            cdi.set_support(
                (support & (ceiling > 0)).astype(np.int8), shift=True
            )
        fraction = self.params["homogenise"]
        if fraction and algo_name in ("hio", "raar"):
            obj = cdi.get_obj(shift=True)
            support = cdi.get_support(shift=True) > 0
            amp = np.abs(obj)
            if support.any():
                target = amp[support].mean()
                new_amp = np.where(
                    support, amp + fraction * (target - amp), amp
                )
                cdi.set_obj(
                    (obj * new_amp / np.maximum(amp, 1e-20)).astype(
                        np.complex64
                    ),
                    shift=True,
                )
        return cdi

    @staticmethod
    def pad_centred(kernel: np.ndarray, shape: tuple) -> np.ndarray:
        """
        Zero-pad a kernel with its centre (index k//2) at n//2 of every
        axis, the pixel that fftshift maps to the origin, which is the
        PSF convention of PyNX's InitPSF.

        Args:
            kernel (np.ndarray): the kernel (odd sizes preferred).
            shape (tuple): the target shape.

        Returns:
            np.ndarray: the padded kernel.
        """
        if kernel.shape == tuple(shape):
            return kernel
        out = np.zeros(shape, kernel.dtype)
        out[
            tuple(
                slice(n // 2 - k // 2, n // 2 - k // 2 + k)
                for n, k in zip(shape, kernel.shape)
            )
        ] = kernel
        return out

    @staticmethod
    def crop_diffraction_data(
        data: np.ndarray, crop_factor: float
    ) -> np.ndarray:
        """
        Central 1/crop_factor of a (reciprocal-space) array along every
        axis. Same operation as crop_diffraction_data in E. Bellec's crop
        algorithm (cdi_dislo.ewen_utilities.Reconstruction_crop_algo).

        Args:
            data (np.ndarray): the array (natural orientation).
            crop_factor (float): > 1.

        Returns:
            np.ndarray: the cropped array.
        """
        crop = ((np.array(data.shape) * (1.0 - 1.0 / crop_factor)) / 2).astype(
            int
        )
        return data[tuple(slice(c, n - c) for c, n in zip(crop, data.shape))]

    @staticmethod
    def interpolate_object(obj: np.ndarray, shape: tuple) -> np.ndarray:
        """
        Resample a real or complex object onto another array size with
        the same field of view (linear interpolation on [-1, 1] grids,
        as interpolate_object of E. Bellec's crop algorithm).

        Args:
            obj (np.ndarray): the object (natural orientation).
            shape (tuple): the target shape.

        Returns:
            np.ndarray: the resampled object.
        """
        from scipy.interpolate import RegularGridInterpolator

        grids = tuple(np.linspace(-1, 1, n) for n in obj.shape)
        points = np.moveaxis(
            np.array(
                np.meshgrid(
                    *[np.linspace(-1, 1, n) for n in shape], indexing="ij"
                )
            ),
            0,
            -1,
        )
        if np.iscomplexobj(obj):
            real = RegularGridInterpolator(grids, obj.real)(points)
            imag = RegularGridInterpolator(grids, obj.imag)(points)
            return (real + 1j * imag).astype(np.complex64)
        return RegularGridInterpolator(grids, obj.astype(float))(points)

    @staticmethod
    def bandwidth_psf(
        q_lab_matrix: np.ndarray,
        q_bragg: np.ndarray,
        bandwidth: float = 1e-4,
        size: int = 15,
    ) -> np.ndarray:
        """
        Detector-frame PSF of the monochromator bandwidth (longitudinal
        coherence). At fixed angles a wavelength lambda' measures
        q lambda / lambda', so near the Bragg peak Q the pattern is
        blurred along Q by ``|Q| dlambda / lambda`` (Gaussian spectrum of that
        FWHM). Partial-coherence formalism: Clark et al., Nat. Commun.
        3, 993 (2012), eq. 3-4.

        Args:
            q_lab_matrix (np.ndarray): (3, 3) matrix q = B @ index of
                the scan (e.g. "transformation_matrices/q_lab" of the
                cdiutils preprocessed cxi).
            q_bragg (np.ndarray): the Bragg peak position in the same
                frame (e.g. "q_lab_shift").
            bandwidth (float, optional): relative bandwidth (FWHM).
                Defaults to 1e-4 (Si(111) double-crystal monochromator).
            size (int, optional): side of the PSF array. Defaults to 15.

        Returns:
            np.ndarray: the centred PSF, normalised to 1.
        """
        from scipy.ndimage import gaussian_filter

        line = (
            np.linalg.solve(
                np.asarray(q_lab_matrix, float), np.asarray(q_bragg, float)
            )
            * bandwidth
        )
        psf = np.zeros((size,) * 3)
        centre = np.full(3, size // 2, dtype=float)
        sigma = 1 / 2.3548
        for t in np.linspace(-3 * sigma, 3 * sigma, 241):
            pos = centre + t * line
            base = np.floor(pos).astype(int)
            frac = pos - base
            weight = np.exp(-(t**2) / (2 * sigma**2))
            for corner in np.ndindex(2, 2, 2):
                w = weight * np.prod(
                    [f if c else 1 - f for f, c in zip(frac, corner)]
                )
                psf[tuple(base + np.array(corner))] += w
        psf = gaussian_filter(psf, 0.3)
        return psf / psf.sum()

    def coarse_to_fine_phasing(
        self,
        run_nb: int,
        recipe: str,
        crop_factors: list,
        support: np.ndarray = None,
        obj: np.ndarray = None,
        final_recipe: str = None,
    ) -> None:
        """
        Coarse-to-fine phasing (E. Bellec's crop algorithm, cdi_dislo):
        each run is first phased on the central 1/f of reciprocal space
        (larger voxels, same field of view) for each f in crop_factors,
        the object is interpolated to the next size, and the run ends
        on the full data. The results are stored in self.cdi_list.

        Args:
            run_nb (int): the number of runs.
            recipe (str): the recipe at the cropped sizes.
            crop_factors (list): e.g. [3, 2].
            support (np.ndarray, optional): starting support (natural
                orientation, full size), resampled at each size.
            obj (np.ndarray, optional): starting object (idem).
            final_recipe (str, optional): the recipe on the full data.
                Defaults to recipe.
        """
        iobs = ifftshift(self.iobs)
        mask = None if self.mask is None else ifftshift(self.mask)
        params = {
            k: v
            for k, v in self.params.items()
            if k not in ("support", "obj", "support_ceiling")
        }
        self.cdi_list = []
        for i in range(run_nb):
            print(f"Run #{i + 1}")
            current = obj
            for f in crop_factors:
                data_f = self.crop_diffraction_data(iobs, f)
                sub = PyNXPhaser(
                    data_f,
                    None
                    if mask is None
                    else self.crop_diffraction_data(mask, f),
                    **params,
                )
                sub.init_cdi(
                    verbose=False,
                    support=None
                    if support is None
                    else (
                        self.interpolate_object(support, data_f.shape) > 0.5
                    ).astype(np.int8),
                    obj=None
                    if current is None
                    else self.interpolate_object(current, data_f.shape),
                )
                sub.run(recipe, sub.cdi, init_cdi=False)
                current = sub.cdi.get_obj(shift=True)
            cdi = self.init_cdi(
                verbose=False,
                init_main_cdi=False,
                support=None if support is None else support.copy(),
                obj=self.interpolate_object(current, iobs.shape),
            )
            self.run(final_recipe or recipe, cdi, init_cdi=False)
            self.cdi_list.append(cdi)

    def genetic_phasing(
        self,
        run_nb: int,
        genetic_pass_nb: int,
        recipe: str,
        selection_method: str = "sharpness",
        init_cdi: bool = True,
        mode: str = "amplitude",
        twin_align: bool = False,
    ):
        """
        Run 'genetic'-like phasing. It runs a number of reconstruction
        independently (defined by the recipe). The support of the best
        result based on the selection_method is used to replace other
        reconstruction support. This is repeated  genetic_pass_nb number
        of times.

        Args:
            run_nb (int): the number of reconstructions.
            genetic_pass_nb (int): the number of genetic pass, i.e. the
                number of times the recipe is applied.
            recipe (str): the instruction to run, i.e. the sequence of
                projection algorithms (ex:
                "ER**200, HIO**400, RAAR**800")
            selection_method (str, optional): The metric used to select
                the best reconstruction. Defaults to "sharpness".
            init_cdi (bool, optional): whether to initialise the cdi
                object or not. Defaults to True.
            mode (str, optional): "amplitude": each run keeps its phase
                and takes ``sqrt(|obj| |best|)`` as modulus (default);
                "complex": each run becomes ``sqrt(obj * best)`` as in
                E. Bellec's genetic scheme (cdi_dislo.genetic).
            twin_align (bool, optional): bring every run to the twin
                image and centre of the best one before combining.
                Defaults to False.

        Raises:
            ValueError: if init_cdi is False but no initialisation was
                done with the init_cdi() method beforehand.
            ValueError: selection method is unknown/invalid.
        """
        if init_cdi:
            self.cdi_list = []
            for i in range(run_nb):
                self.cdi_list.append(
                    self.init_cdi(init_main_cdi=False, verbose=False)
                )
        else:
            if self.cdi_list is None or self.cdi_list == []:
                raise ValueError(
                    "CDI object are not initialised, init_cdi should be True."
                )
        if selection_method not in ("sharpness", "mean_to_max"):
            raise ValueError(
                f"Invalid selection_method ({selection_method}), can be"
                "'sharpness' or 'mean_to_max'."
            )
        metrics = [None for _ in range(run_nb)]

        for i in range(genetic_pass_nb + 1):
            if i == 0:
                print("First reconstruction pass")
            else:
                print(f"\nGenetic pass #{i}.")
                for i in range(run_nb):
                    try:
                        metrics[i] = (
                            PhasingResultAnalyser.amplitude_based_metrics(
                                np.abs(self.cdi_list[i].get_obj(shift=True)),
                                self.cdi_list[i].get_support(shift=True),
                            )[selection_method]
                        )
                    except ValueError:
                        metrics[i] = np.nan

                indice = np.argsort(metrics)[0]
                print(
                    "Updating cdi objects with the best reconstruction "
                    f"(run # {indice + 1}).\n"
                )
                best = self.cdi_list[indice].get_obj(shift=True)
                for i in range(run_nb):
                    if i == indice:
                        continue
                    new_obj = self.cdi_list[i].get_obj(shift=True)
                    if twin_align:
                        new_obj = self._align_to(new_obj, best)
                    if mode == "complex":
                        new_obj = np.sqrt(new_obj * best)
                    else:
                        new_obj = np.sqrt(
                            np.abs(new_obj) * np.abs(best)
                        ) * np.exp(1j * np.angle(new_obj))
                    self.cdi_list[i].set_obj(
                        new_obj.astype(np.complex64), shift=True
                    )
            self.run_multiple_instances(run_nb, recipe, init_cdi=False)

    @staticmethod
    def _align_to(obj: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Bring obj to the twin image (conj(obj(-r))) and the centre of
        mass of reference, choosing the twin by modulus correlation.

        Args:
            obj (np.ndarray): object (natural orientation).
            reference (np.ndarray): reference object.

        Returns:
            np.ndarray: aligned object.
        """
        from scipy.ndimage import center_of_mass

        twin = np.conj(obj[::-1, ::-1, ::-1])
        ref = np.abs(reference).ravel()
        if (
            np.corrcoef(np.abs(twin).ravel(), ref)[0, 1]
            > np.corrcoef(np.abs(obj).ravel(), ref)[0, 1]
        ):
            obj = twin
        shift = np.round(
            np.array(center_of_mass(np.abs(reference) ** 2))
            - np.array(center_of_mass(np.abs(obj) ** 2))
        ).astype(int)
        return np.roll(obj, tuple(shift), axis=(0, 1, 2))

    @staticmethod
    def plot_cdi(
        cdi: CDI_Type,
        spaces: str = "both",
        axis: int = 0,
        title: str = None,
        axes: plt.Axes = None,
    ) -> None:
        """
        A staticmethod to plot cdi object main quantity.

        See ShowCDI operator to see how Vincent plots CDI objects.
        https://gitlab.esrf.fr/favre/PyNX/-/blob/master/pynx/cdi/cpu_operator.py?ref_type=heads

        Args:
            cdi (CDI_Type): the cdi object to plot the quantity from.
            spaces (str, optional): Whether reciprocal or direct space.
                Defaults to "both".
            axis (int, optional): What slice to plot. Defaults to 0.
            title (str, optional): A title for the plot. Defaults to None.
        """
        # Check the dimension of the object, whether 2D or 3D
        if cdi.get_obj().ndim == 2:
            the_slice = slice(None)
        else:
            # The slice to plot according to the given axis
            the_slice = get_centred_slices(cdi.get_obj().shape)[axis]

        # Get the support and amplitude
        support = cdi.get_support(shift=True)[the_slice]
        direct_space_obj = cdi.get_obj(shift=True)[the_slice]
        amplitude = np.abs(direct_space_obj)

        # Vincent's style to manage vmax value for amplitude plotting.
        # Basically: "max is at the 99 percentile relative to the number
        # of points inside the support"
        percent = 100 * (1 - 0.01 * support.sum() / support.size)
        max99 = np.percentile(amplitude, percent)

        # Initialise the quantities to plot, in any case direct space
        # amplitude and phase are required.
        quantities = {
            "amplitude": amplitude,
            "phase": np.ma.masked_array(
                np.angle(direct_space_obj), mask=(support == 0)
            ),
        }

        if spaces == "direct":
            quantities["support"] = support

        elif spaces == "both":
            quantities["calculated_intensity"] = (
                np.abs(ifftshift(fftn(cdi.get_obj().copy()))[the_slice]) ** 2
            )

            iobs = cdi.get_iobs(shift=True).copy()[the_slice]
            tmp = np.logical_and(iobs > -1e19, iobs < 0)
            if tmp.sum() > 0:
                # change back free pixels to their real intensity
                iobs[tmp] = -iobs[tmp] - 1
            iobs[iobs < 0] = 0

            quantities["observed_intensity"] = iobs

        nrows, ncols = (1, 3) if spaces == "direct" else (2, 2)
        update = False
        if axes is None:
            fig, axes = plt.subplots(
                nrows, ncols, layout="tight", figsize=(3, 3)
            )
        else:
            update = True
            if len(axes) != nrows * ncols:
                raise ValueError(
                    "Provided axes should have shape (1, 3) or (2, 2)."
                )

        for ax, key in zip(axes.flat, quantities):
            plot_params = get_plot_configs(key)

            # Refine the generic plot configs
            if key == "amplitude":
                plot_params["vmax"] = max99
                plot_params["cmap"] = "gray"
            if key == "calculated_intensity":
                plot_params["title"] = "Calculated Int. (a.u.)"
            if key == "observed_intensity":
                plot_params["title"] = "Observed Int. (a.u.)"
            if "intensity" in key:
                plot_params["norm"] = LogNorm(1)
                plot_params.pop("vmin"), plot_params.pop("vmax")

            # Set the ax title and remove it from the plotting params
            ax.set_title(plot_params.pop("title"))
            if update:
                ax.images[0].set_data(quantities[key])
            else:
                image = ax.matshow(
                    quantities[key], origin="lower", **plot_params
                )

            # if key in ("support", "phase") and support.sum() > 0:
            if key in ("amplitude", "support", "phase") and support.sum() > 0:
                ax.set_xlim(
                    np.nonzero(support.sum(axis=0))[0][[0, -1]]
                    + np.array([-10, 10])
                )
                ax.set_ylim(
                    np.nonzero(support.sum(axis=1))[0][[0, -1]]
                    + np.array([-10, 10])
                )

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="7.5%", pad=0.05)
            fig.colorbar(
                image, cax=cax, extend="both" if key == "phase" else None
            )

            ax.locator_params(nbins=5)
            ax.xaxis.set_ticks_position("bottom")
            ax.yaxis.set_ticks_position("left")
        fig.suptitle(title)
        return axes

    def __repr__(self) -> str:
        text = (
            "PyNXPhaser object:\n"
            f"PyNXPhaser with iobs shape: {self.iobs.shape}, "
            f"mask shape: {self.mask.shape if self.mask is not None else None}"
        )

        parameters_text = "\nPyNX parameters:"
        for key in sorted(self.params):
            parameters_text += f"\n\t{key}: {self.params[key]}"
        text += parameters_text
        return text


class PhasingResultAnalyser:
    """
    This class provided utility function for phase retrieval results
    analysis.
    """

    def __init__(
        self, cdi_results: list = None, result_dir_path: str = None
    ) -> None:
        """
        Init method.

        Args:
            cdi_results (list, optional): results as CDI objects.
                Defaults to None.
            result_dir_path (str, optional): the path of directory
                containing the .cxi phase retrieval results. Defaults
                to None.

        Raises:
            ValueError: if non of the parameters is provided or if both
                are.
        """
        message = (
            "cdi_results or result_dir_path cannot be provided simultaneously"
            ", but one of the two must be."
        )
        if cdi_results is None and result_dir_path is None:
            raise ValueError(message)

        if cdi_results is not None and result_dir_path is not None:
            raise ValueError(message)

        # Convert the parsed list into a dict whose keys are run numbers
        self.cdi_results = []
        if cdi_results:
            self.cdi_results = {
                f"Run{i + 1:04d}": cdi for i, cdi in enumerate(cdi_results)
            }
        self.result_dir_path = result_dir_path
        self._metrics = None
        self._sorted_phasing_results = None
        self.result_paths = []
        self.best_candidates = []

    @property
    def metrics(self) -> dict:
        if self._metrics is None:
            return None
        metrics = {m: {} for m in self._metrics}
        for m in self._metrics:
            for path in self._metrics[m]:
                if os.path.exists(path):
                    run = path.split("/")[-1]
                    metrics[m][run] = self._metrics[m][path]
                else:
                    metrics[m][path] = self._metrics[m][path]
        return metrics

    @property
    def sorted_phasing_results(self) -> dict:
        if self._sorted_phasing_results is None:
            return None
        sorted_results = {}
        for path in self._sorted_phasing_results:
            if os.path.exists(path):
                run = path.split("/")[-1]
                sorted_results[run] = self._sorted_phasing_results[path]
            else:
                sorted_results[path] = self._sorted_phasing_results[path]
        return sorted_results

    def find_phasing_results(self, search_pattern: str = "*Run*.cxi") -> None:
        """
        Find phasing results (.cxi files) that match the given pattern.

        Args:
            search_pattern (str, optional): Pattern to search for files.
                Uses glob syntax (not regex). Defaults to "*Run*.cxi".

        Raises:
            ValueError: If no files match the given pattern.
        """
        self.result_paths = []
        # Handle the path joining properly without duplicate slashes
        full_path_pattern = os.path.join(self.result_dir_path, search_pattern)
        fetched_paths = glob.glob(full_path_pattern)

        for path in fetched_paths:
            if os.path.isfile(path):
                self.result_paths.append(path)

        if not self.result_paths:
            raise ValueError(
                f"No result found that match the pattern '{search_pattern}' "
                f"in the result_dir_path ({self.result_dir_path})"
            )

    @staticmethod
    def amplitude_based_metrics(
        amplitude: np.ndarray, support: np.ndarray
    ) -> dict:
        """
        Compute the criteria based on amplitude analysis, namely:
            * the mean_to_max, mean of the gaussian fitting to max
                distance.
            * the std, the standard deviation of the amplitude
            * the sharpness, the sum or mean of the amplitude to the
                power of 4.

        Args:
            amplitude (np.ndarray): the amplitude to work on.
            support (np.ndarray): the associated support.

        Returns:
            dict: the dictionary containing the metrics.
        """
        if amplitude.shape != support.shape:
            raise ValueError("Amplitude and support must have the same shape.")
        if amplitude.size == 0 or support.size == 0:
            raise ValueError("Amplitude and support must not be empty arrays.")
        sharpness = np.mean((amplitude * support) ** 4)
        amplitude = amplitude[support > 0]
        std = np.std(amplitude)
        amplitude /= np.max(amplitude)

        # fit the amplitude distribution
        kernel = gaussian_kde(amplitude)
        x = np.linspace(0, 1, 100)
        fitted_counts = kernel(x)
        max_index = np.argmax(fitted_counts)
        return {
            "mean_to_max": 1 - x[max_index],
            "std": std,
            "sharpness": sharpness,
        }

    def analyse_phasing_results(
        self,
        sorting_criterion: str = "mean_to_max",
        search_pattern: str = "*Run*.cxi",
        plot: bool = True,
        plot_phasing_results: bool = True,
        plot_phase: bool = False,
    ) -> None:
        """
        Analyse the phase retrieval results by sorting them according to
        the sorting_criterion, which must be selected among:

        * mean_to_max the difference between the mean of the
            Gaussian fitting of the amplitude histogram and the maximum
            value of the amplitude. We consider the closest to the max
            the mean is, the most homogeneous is the amplitude of the
            reconstruction, hence the best.
        * the sharpness the sum of the amplitude within support to
            the power of 4. For reconstruction with similar support,
            lowest values means greater amplitude homogeneity.
        * std the standard deviation of the amplitude.
        * llk the log-likelihood of the reconstruction.
        * llkf the free log-likelihood of the reconstruction.

        Args:
            sorting_criterion (str, optional): the criterion to sort the
                results with. Defaults to "mean_to_max".
            search_pattern (str, optional): Pattern to search for files.
                Uses glob syntax (not regex). Defaults to "*Run*.cxi".
            plot (bool, optional): whether or not to disable all plots.
            plot_phasing_results (bool, optional): whether to plot the
                phasing results. Defaults to True.
            plot_phase (bool, optional): whether the phase must be
                plotted. If True, will the phase is plotted with
                amplitude as opacity. If False, amplitude is plotted
                instead. Defaults to False.

        Raises:
            ValueError: if sorting_criterion is unknown.
        """
        criteria = ["mean_to_max", "std", "llk", "llkf", "sharpness", "all"]
        if sorting_criterion not in criteria:
            raise ValueError(
                f"Provided criterion ({sorting_criterion}) is unknown. "
                f"Possible criteria are:\n{criteria}."
            )

        if self._sorted_phasing_results is None:
            print("[INFO] Computing metrics...")

            self._metrics = {
                m: {f: None for f in self.result_paths} for m in criteria
            }
            if self.cdi_results:
                for run in self.cdi_results:
                    self._metrics["llk"][run] = self.cdi_results[
                        run
                    ].llk_poisson
                    self._metrics["llkf"][run] = self.cdi_results[
                        run
                    ].llk_poisson_free

                    amplitude = np.abs(self.cdi_results[run].get_obj())
                    support = self.cdi_results[run].get_support()
                    # update the metric dictionary
                    try:
                        amplitude_based_metrics = self.amplitude_based_metrics(
                            amplitude, support
                        )
                    except ValueError:
                        amplitude_based_metrics = {
                            "mean_to_max": np.nan,
                            "std": np.nan,
                            "sharpness": np.nan,
                        }
                    for m in ["mean_to_max", "std", "sharpness"]:
                        self._metrics[m][run] = amplitude_based_metrics[m]
            else:
                self.find_phasing_results(search_pattern)

                for p in self.result_paths:
                    with silx.io.h5py_utils.File(p, "r") as file:
                        self._metrics["llk"][p] = file[
                            "entry_1/image_1/process_1/results/llk_poisson"
                        ][()]
                        self._metrics["llkf"][p] = file[
                            "entry_1/image_1/process_1/results/"
                            "free_llk_poisson"
                        ][()]

                        support = file["entry_1/image_1/support"][()]
                        amplitude = np.abs(file["entry_1/data_1/data"][()])
                    amplitude_based_metrics = self.amplitude_based_metrics(
                        amplitude, support
                    )
                    for m in ["mean_to_max", "std", "sharpness"]:
                        self._metrics[m][p] = amplitude_based_metrics[m]

            # normalise all the values so they can be compared in the same plot
            for m in ["mean_to_max", "std", "llk", "llkf", "sharpness"]:
                minimum_value = min(list(self._metrics[m].values()))
                ptp_value = np.ptp(np.array(list(self._metrics[m].values())))
                for k in self._metrics[m].keys():
                    self._metrics[m][k] = (
                        self._metrics[m][k] - minimum_value
                    ) / ptp_value

        # compute the average value over all metrics
        for run in self._metrics["llk"]:
            self._metrics["all"][run] = 0
            for m in ["mean_to_max", "std", "llk", "llkf", "sharpness"]:
                self._metrics["all"][run] += self._metrics[m][run]
            self._metrics["all"][run] /= 5

        # now sort the files according to the provided criterion
        self._sorted_phasing_results = dict(
            sorted(
                self._metrics[sorting_criterion].items(),
                key=lambda item: item[1],
                reverse=False,
            )
        )
        if self.cdi_results is not None:
            runs = [
                str(extract_run_info(run)[0]).zfill(2)
                for run in self._sorted_phasing_results
            ]
        else:
            runs = [
                str(extract_run_info(file)[0]).zfill(2)
                for file in self._sorted_phasing_results
            ]

        if plot:
            figure, ax = plt.subplots(1, 1, layout="tight", figsize=(6, 3))
            colors = {
                m: c
                for m, c in zip(
                    self._metrics,
                    [
                        "lightcoral",
                        "mediumslateblue",
                        "dodgerblue",
                        "plum",
                        "teal",
                        "crimson",
                    ],
                )
            }
            for m in self._metrics:
                ax.plot(
                    runs,
                    [
                        self._metrics[m][f]
                        for f in self._sorted_phasing_results
                    ],
                    label=m,
                    color=colors[m],
                    marker="o",
                    markersize=4,
                    markerfacecolor=colors[m],
                    markeredgewidth=0.5,
                    markeredgecolor="k",
                )
            ax.set_ylabel("Normalised metric")
            ax.set_xlabel("Run number")

            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(
                loc="center left", bbox_to_anchor=(1, 0.5), frameon=False
            )
            figure.suptitle("Phasing result analysis (the lower the better)\n")
        print(
            f"[INFO] the sorted list of runs using '{sorting_criterion}' "
            f"sorting_criterion is:\n{runs}."
        )

        if plot_phasing_results and plot:
            print("[INFO] Plotting phasing results...")
            for result in self._sorted_phasing_results:
                if self.cdi_results:
                    data = self.cdi_results[result].get_obj(shift=True)
                    support = self.cdi_results[result].get_support(shift=True)
                else:
                    with silx.io.h5py_utils.File(result, "r") as file:
                        data = file["entry_1/data_1/data"][()]
                        support = file["entry_1/image_1/support"][()]
                run = extract_run_info(result)[0]
                title = f"Phasing results, run {extract_run_info(result)[0]}"
                self.plot_phasing_result(data, support, title, plot_phase)

    @staticmethod
    def _register_spectrum(
        obj: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        """
        Spectrum of obj registered to reference: of obj and its twin
        image ``conj(obj(-r))``, the one with the larger complex overlap
        with reference after a sub-pixel translation (complex
        cross-correlation, Fourier shift theorem). The modulus alone
        cannot choose the twin of a nearly centrosymmetric crystal.

        Args:
            obj (np.ndarray): complex object, origin at the array
                centre.
            reference (np.ndarray): complex reference object.

        Returns:
            np.ndarray: registered spectrum, ``fftn(ifftshift(.))``
            convention (origin at index 0).
        """
        best_overlap, best_spectrum = -1.0, None
        for candidate in (obj, np.conj(obj[::-1, ::-1, ::-1])):
            shift = phase_cross_correlation(
                reference, candidate, upsample_factor=20
            )[0]
            spectrum = fourier_shift(fftn(ifftshift(candidate)), shift)
            registered = fftshift(np.fft.ifftn(spectrum))
            overlap = np.abs(np.vdot(reference, registered)) / (
                np.linalg.norm(reference) * np.linalg.norm(registered)
            )
            if overlap > best_overlap:
                best_overlap, best_spectrum = overlap, spectrum
        return best_spectrum

    @staticmethod
    def phase_retrieval_transfer_function(
        objects: list,
        intensity: np.ndarray,
        mask: np.ndarray = None,
        q_lab_matrix: np.ndarray = None,
        nb_shells: int = 40,
        nb_directions: int = 0,
        threshold: float = 1 / np.e,
        align: bool = True,
        shell_average: str = "amplitude",
        normalisation_q: float = None,
        background: np.ndarray = None,
        min_snr: float = 3.0,
    ) -> dict:
        """
        Phase retrieval transfer function (PRTF) of independent
        reconstructions against the measured intensity.

        The PRTF is ``|<F_i(q)>| / sqrt(I(q))``, where ``F_i`` is the
        Fourier transform of reconstruction i, scaled to the measured
        intensity, and the average is taken after aligning every
        reconstruction to the first one (twin image, sub-pixel
        translation and constant phase), see Chapman et al., J. Opt.
        Soc. Am. A 23, 1179 (2006), eq. 23. With ``nb_directions`` > 0,
        the PRTF is also computed in cones around quasi-uniform
        directions, since the resolution of Bragg CDI depends on the
        direction (Cherukara et al., Appl. Phys. Lett. 113, 203101
        (2018)).

        Two shell averages are available. "amplitude" averages
        ``|<F_i>|`` and ``sqrt(I)`` over each shell before the ratio
        (intensity clipped at zero). "intensity" takes
        ``sqrt(sum |<F_i>|^2 / sum I)`` over each shell, which accepts a
        background-subtracted, unclipped intensity: the noise then
        averages out instead of adding a positive bias to the
        denominator, so the curve is not set by the background level.

        With ``background`` (expected background counts per voxel, same
        shape), ``intensity`` holds the raw counts: the PRTF uses
        ``intensity - background`` and each shell (or cone x shell) gets
        the signal-to-noise ratio ``sum(I - B) / sqrt(sum I)`` (Poisson).
        The PRTF is left undefined (NaN) where it is below ``min_snr``:
        there the data, not the phasing, limit the resolution, and the
        half-period where the SNR first drops below ``min_snr`` is
        returned as ``signal_limit``.

        The resolution is the half-period ``pi / q`` at which the PRTF
        first falls below the threshold (1/e as in Cherukara et al.;
        Chapman et al. give 0.5 as a more conservative value).
        Reconstructions stuck in the same wrong solution give an
        optimistic PRTF: it measures reproducibility, not correctness.

        Args:
            objects (list): complex reconstructions in the detector
                frame, with the shape of intensity and the origin at the
                array centre (as PyNX outputs them).
            intensity (np.ndarray): measured intensity in photons,
                origin at the array centre; may be background-subtracted
                and negative with ``shell_average="intensity"``.
            mask (np.ndarray, optional): values > 0 flag masked voxels.
                Defaults to None.
            q_lab_matrix (np.ndarray, optional): (3, 3) matrix whose
                columns are the q vectors of one step along each array
                axis (``transformation_matrices/q_lab`` of the
                preprocessed cxi file, in 1/angstrom). Without it, q is
                in 2 pi / N per pixel along an axis of length N.
                Defaults to None.
            nb_shells (int, optional): number of shells up to the
                largest ``|q|`` reached along every axis. Defaults to
                40.
            nb_directions (int, optional): number of directions of the
                directional PRTF, 0 for none. Defaults to 0.
            threshold (float, optional): PRTF value that defines the
                resolution. Defaults to 1/e.
            align (bool, optional): whether to align the
                reconstructions to the first one. Defaults to True.
            shell_average (str, optional): "amplitude" or "intensity",
                see above. Defaults to "amplitude".
            normalisation_q (float, optional): scale each spectrum to
                the measured intensity summed over ``|q|`` below this
                value (signal-dominated region) instead of over the
                whole array, whose total includes the background.
                Defaults to None (whole array).
            background (np.ndarray, optional): expected background
                counts per voxel; intensity is then the raw counts.
                Defaults to None.
            min_snr (float, optional): shells (cones) whose net signal
                is below this SNR get no PRTF value; only used with
                background. Defaults to 3.

        Returns:
            dict: ``q`` (shell centres), ``prtf`` (one value per
            shell), ``nb_voxels`` (unmasked voxels per shell),
            ``resolution`` (half-period at the threshold, in the length
            unit of 2 pi / q, None if the PRTF stays above it) and,
            with directions, ``directions`` (unit vectors),
            ``prtf_directional`` (directions x shells) and
            ``resolution_directional``. With background, also ``snr``,
            ``signal_limit`` (half-period where the SNR first drops
            below min_snr, None if never) and their directional
            versions ``snr_directional`` and ``signal_limit_directional``.

        Raises:
            ValueError: if an object does not have the shape of the
                intensity, or if shell_average is unknown.
        """
        if shell_average not in ("amplitude", "intensity"):
            raise ValueError(
                f"shell_average must be 'amplitude' or 'intensity', not "
                f"{shell_average!r}."
            )
        intensity = np.asarray(intensity, dtype=float)
        valid = np.isfinite(intensity)
        if mask is not None:
            valid &= np.asarray(mask) == 0
        intensity = np.where(valid, intensity, 0)
        raw_counts = None
        if background is not None:
            raw_counts = np.clip(intensity, 0, None)
            intensity = intensity - np.where(valid, background, 0)

        # |q| of every voxel (origin at the array centre)
        shape = np.array(intensity.shape)
        if q_lab_matrix is None:
            q_lab_matrix = np.diag(2 * np.pi / shape)
        q_lab_matrix = np.asarray(q_lab_matrix, dtype=float)
        offsets = np.indices(intensity.shape, dtype=np.float32)
        offsets -= (shape // 2).reshape(3, 1, 1, 1)
        q_vectors = np.tensordot(q_lab_matrix, offsets, axes=(1, 0))
        del offsets
        q_norm = np.linalg.norm(q_vectors, axis=0)
        q_max = np.min(np.linalg.norm(q_lab_matrix, axis=0) * (shape // 2))
        edges = np.linspace(0, q_max, nb_shells + 1)
        shell = np.digitize(q_norm, edges) - 1
        inside = valid & (shell < nb_shells)
        centres = (edges[:-1] + edges[1:]) / 2

        # region where each spectrum is scaled to the data
        scaling = valid.copy()
        if normalisation_q is not None:
            scaling &= q_norm < normalisation_q
        measured_total = np.sum(intensity[scaling])

        # aligned and scaled spectra, summed (Chapman et al., eq. 23)
        reference = np.asarray(objects[0])
        reference_spectrum = None
        spectrum_sum = np.zeros(intensity.shape, dtype=np.complex128)
        for obj in objects:
            obj = np.asarray(obj, dtype=np.complex128)
            if obj.shape != intensity.shape:
                raise ValueError(
                    f"Object shape {obj.shape} differs from the intensity "
                    f"shape {intensity.shape}."
                )
            if align and reference_spectrum is not None:
                spectrum = PhasingResultAnalyser._register_spectrum(
                    obj, reference
                )
            else:
                spectrum = fftn(ifftshift(obj))
            spectrum = fftshift(spectrum)
            spectrum *= np.sqrt(
                measured_total / np.sum(np.abs(spectrum[scaling]) ** 2)
            )
            if align and reference_spectrum is not None:
                offset = np.angle(
                    np.sum(
                        np.conj(reference_spectrum[valid]) * spectrum[valid]
                    )
                )
                spectrum *= np.exp(-1j * offset)
            if reference_spectrum is None:
                reference_spectrum = spectrum
            spectrum_sum += spectrum
        calculated = np.abs(spectrum_sum) / len(objects)
        if shell_average == "amplitude":
            calculated_term = calculated[inside]
            measured_term = np.sqrt(np.clip(intensity[inside], 0, None))
        else:
            calculated_term = calculated[inside] ** 2
            measured_term = intensity[inside]

        def shell_snr(index, size):
            """Net-signal SNR per index (Poisson), None without
            background."""
            if raw_counts is None:
                return None
            net = np.bincount(index, intensity[inside], size)
            counts = np.bincount(index, raw_counts[inside], size)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(counts > 0, net / np.sqrt(counts), 0.0)

        def shell_prtf(index, size):
            """PRTF per index (shell sums, then ratio), NaN where the
            net signal is not significant."""
            numerator = np.bincount(index, calculated_term, size)
            denominator = np.bincount(index, measured_term, size)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(
                    denominator > 0, numerator / denominator, np.nan
                )
            if shell_average == "intensity":
                ratio = np.sqrt(ratio)
            snr = shell_snr(index, size)
            if snr is not None:
                ratio = np.where(snr >= min_snr, ratio, np.nan)
            return ratio

        def signal_limit(snr):
            """Half-period where the SNR first drops below min_snr."""
            below = np.nonzero(snr[1:] < min_snr)[0]
            if below.size == 0:
                return None
            return float(np.pi / centres[below[0] + 1])

        def crossing(prtf):
            """Half-period resolution at the threshold crossing (NaN
            shells, without significant signal, end the search)."""
            undefined = np.nonzero(~np.isfinite(prtf[1:]))[0]
            last = undefined[0] + 1 if undefined.size else prtf.size
            below = np.nonzero(prtf[1:last] < threshold)[0]
            if below.size == 0:
                return None
            k = below[0] + 1
            q_cross = centres[k]
            if np.isfinite(prtf[k - 1]) and prtf[k - 1] != prtf[k]:
                # linear interpolation between the two shell centres
                q_cross = centres[k - 1] + (
                    (prtf[k - 1] - threshold)
                    / (prtf[k - 1] - prtf[k])
                    * (centres[k] - centres[k - 1])
                )
            return float(np.pi / q_cross)

        prtf = shell_prtf(shell[inside], nb_shells)
        result = {
            "q": centres,
            "prtf": prtf,
            "nb_voxels": np.bincount(shell[inside], minlength=nb_shells),
            "resolution": crossing(prtf),
        }
        if raw_counts is not None:
            snr = shell_snr(shell[inside], nb_shells)
            result["snr"] = snr
            result["signal_limit"] = signal_limit(snr)
        if nb_directions > 0:
            # quasi-uniform (Fibonacci) directions, each voxel in the cone
            # of the closest one
            i = np.arange(nb_directions) + 0.5
            polar = np.arccos(1 - 2 * i / nb_directions)
            azimuth = np.pi * (1 + 5**0.5) * i
            directions = np.stack(
                [
                    np.cos(polar),
                    np.sin(polar) * np.sin(azimuth),
                    np.sin(polar) * np.cos(azimuth),
                ],
                axis=1,
            )
            units = q_vectors[:, inside] / np.maximum(q_norm[inside], 1e-12)
            cone = np.empty(units.shape[1], dtype=int)
            chunk = 1_000_000  # bounds the (directions x voxels) product
            for start in range(0, units.shape[1], chunk):
                cone[start : start + chunk] = np.argmax(
                    directions @ units[:, start : start + chunk], axis=0
                )
            directional = shell_prtf(
                cone * nb_shells + shell[inside], nb_directions * nb_shells
            ).reshape(nb_directions, nb_shells)
            result["directions"] = directions
            result["prtf_directional"] = directional
            result["resolution_directional"] = [
                crossing(p) for p in directional
            ]
            if raw_counts is not None:
                snr = shell_snr(
                    cone * nb_shells + shell[inside],
                    nb_directions * nb_shells,
                ).reshape(nb_directions, nb_shells)
                result["snr_directional"] = snr
                result["signal_limit_directional"] = [
                    signal_limit(s) for s in snr
                ]
        return result

    def select_best_candidates(
        self,
        nb_of_best_sorted_runs: int = None,
        best_runs: list[int] = None,
        search_pattern: str = "*Run*.cxi",
    ) -> None:
        """
        Select the best candidates, two methods are possible. Either
        select a specific number of runs, provided they were alaysed and
        sorted beforehand. Or simply provide a list of integers
        corresponding to the digit numbers of the best runs.

        Args:
            nb_of_best_sorted_runs (int, optional): the number of best
                runs to select, provided they were analysed beforehand.
                Defaults to None.
            best_runs (list[int], optional): the best runs to select.
                Defaults to None.
            search_pattern (str, optional): Pattern to search for files.
                Uses glob syntax (not regex). Defaults to "*Run*.cxi".

        Raises:
            ValueError: If nb_of_best_sorted_runs but reconstructions
            were not analysed before.
        """
        if nb_of_best_sorted_runs is None and best_runs is None:
            print(
                "Neither nb_of_best_sorted_runs nor best_runs are provided. "
                "Will select only the first best run."
            )
            nb_of_best_sorted_runs = 1

        # If selection is made by hand, i.e. best_runs is provided
        if best_runs:
            if self.cdi_results:
                # This is the notebook mode
                self.best_candidates = [f"Run{r:04d}" for r in best_runs]
            else:
                # This is the script/pipeline mode
                self.best_candidates = []
                if not self.result_paths:
                    self.find_phasing_results(search_pattern)
                for path in self.result_paths:
                    run_nb = extract_run_info(path)[0]
                    if run_nb in best_runs:
                        self.best_candidates.append(path)

        elif nb_of_best_sorted_runs:
            if self._sorted_phasing_results is None:
                raise ValueError("Phasing results have not been analysed yet.")
            self.best_candidates = list(self._sorted_phasing_results.keys())
            self.best_candidates = self.best_candidates[
                :nb_of_best_sorted_runs
            ]

        if not self.cdi_results:
            # This is the script/pipeline mode
            # Remove the previous candidate files
            for f in glob.glob(self.result_dir_path + "/candidate_*.cxi"):
                os.remove(f)
            printout_list = [
                str(extract_run_info(f)[0]).zfill(2)
                for f in self.best_candidates
            ]
            print(f"[INFO] Best candidates selected:\n{printout_list}")
            for i, f in enumerate(self.best_candidates):
                dir_name, file_name = os.path.split(f)
                run_nb = str(extract_run_info(file_name)[0]).zfill(2)
                scan_nb = file_name.split("_")[0]
                file_name = (
                    f"/candidate_{i + 1}-{len(self.best_candidates)}"
                    f"_{scan_nb}_run_{run_nb}.cxi"
                )
                shutil.copy(f, dir_name + file_name)

    # Note: this method only works if PyNX is installed. If not, use
    # the BcdiPipeline method.
    def mode_decomposition(
        self, verbose: bool = True, search_pattern: str = "*Run*.cxi"
    ) -> np.ndarray:
        """
        Run a mode decomposition à la PyNX. See pynx_cdi_analysis.py
        script. Note that this method only works if PyNX is installed.
        If not, use the BcdiPipeline method.

        Args:
            verbose (bool, optional): whether to print some logs.
                Defaults to True.
            search_pattern (str, optional): Pattern to search for files.
                Uses glob syntax (not regex). Defaults to "*Run*.cxi".

        Raises:
            ValueError: in script/notebook mode, if not results are
            found in the reconstruction folder.

        Returns:
            np.ndarray: the main mode.
        """
        if not IS_PYNX_AVAILABLE:
            raise PyNXImportError

        if self.best_candidates:
            result_keys = self.best_candidates

        else:
            print(
                "No best candidates selected, computation will be conducted "
                "on all reconstructions."
            )
            if self._sorted_phasing_results is not None:
                result_keys = self._sorted_phasing_results.keys()
            else:
                if verbose:
                    print(
                        "Results are not sorted, therefore matching won't be "
                        "done against the best result."
                    )
                if self.result_paths is None:
                    self.find_phasing_results(search_pattern)
                result_keys = self.result_paths
        results = []
        for r in result_keys:
            if self.cdi_results:
                results.append(self.cdi_results[r].get_obj(shift=True))
            else:
                with silx.io.h5py_utils.File(r, "r") as file:
                    results.append(file["entry_1/data_1/data"][()])

        match2_results = [results[0]]
        for i in range(1, len(results)):
            _, d2c, r = match2(results[0], results[i])
            match2_results.append(d2c)  # don't know if needed, that's PyNX way
            if verbose:
                print(f"R_match({i}) = {r * 100:6.3f} %")

        modes, mode_weights = ortho_modes(
            match2_results, nb_mode=1, return_weights=True
        )
        if verbose:
            print(f"First mode represents {mode_weights[0] * 100:6.3f} %")
        return modes, mode_weights

    @staticmethod
    def plot_phasing_result(
        data: np.ndarray,
        support: np.ndarray,
        title: str = None,
        plot_phase: bool = False,
    ) -> None:
        """
        Plot the reconstructed object in reciprocal and direct spaces.

        Args:
            data (np.ndarray): the reconstruction data to plot.
            support (np.ndarray): the support of the reconstruction.
            title (str, optional): the title of the plot. Defaults to None.
            plot_phase (bool, optional):  whether to plot the phase,
                if True, will plot the phase whit amplitude as opacity.
                If False, amplitude will be plotted.
                Defaults to False.
        """
        reciprocal_space_data = np.abs(ifftshift(fftn(fftshift(data)))) ** 2
        direct_space_amplitude = np.abs(data)
        normalised_direct_space_amplitude = (
            direct_space_amplitude - np.min(direct_space_amplitude)
        ) / np.ptp(direct_space_amplitude)
        direct_space_phase = np.angle(data)

        if data.ndim == 3:
            figure, axes = plt.subplots(
                2, 3, figsize=(6, 4), layout="constrained"
            )
            com = CroppingHandler.get_position(support, "com")
            shift = tuple(com[i] - support.shape[i] // 2 for i in range(3))
            slices = get_centred_slices(data.shape, shift)
            for i in range(3):
                rcp_im = axes[0, i].matshow(
                    np.sum(reciprocal_space_data, axis=i),
                    cmap="turbo",
                    norm=LogNorm(1),
                )
                if plot_phase:
                    direct_space_im = axes[1, i].matshow(
                        direct_space_phase[slices[i]],
                        vmin=-np.pi,
                        vmax=np.pi,
                        alpha=normalised_direct_space_amplitude[slices[i]],
                        cmap="cet_CET_C9s_r",
                    )
                else:
                    direct_space_im = axes[1, i].matshow(
                        direct_space_amplitude[slices[i]],
                        vmin=0,
                        vmax=direct_space_amplitude.max(),
                        cmap="turbo",
                    )

                if support[slices[i]].sum() > 0:
                    axes[1, i].set_xlim(
                        np.nonzero(support[slices[i]].sum(axis=0))[0][[0, -1]]
                        + np.array([-5, 5])
                    )
                    axes[1, i].set_ylim(
                        np.nonzero(support[slices[i]].sum(axis=1))[0][[0, -1]]
                        + np.array([-5, 5])
                    )
            figure.colorbar(rcp_im, ax=axes[0, 2], extend="both")
            figure.colorbar(direct_space_im, ax=axes[1, 2], extend="both")

            axes[0, 1].set_title("Intensity projection (a.u.)")
            axes[1, 1].set_title(
                "Phase (rad)" if plot_phase else "Amplitude (a.u.)"
            )

        if data.ndim == 2:
            figure, axes = plt.subplots(1, 2, figsize=(4, 2), layout="tight")

            rcp_im = axes[0].matshow(
                reciprocal_space_data, cmap="turbo", norm=LogNorm(1)
            )
            if plot_phase:
                direct_space_im = axes[1].matshow(
                    direct_space_phase,
                    vmin=-np.pi,
                    vmax=np.pi,
                    alpha=normalised_direct_space_amplitude,
                    cmap="cet_CET_C9s_r",
                )
            else:
                direct_space_im = axes[1].matshow(
                    direct_space_amplitude,
                    vmin=0,
                    vmax=direct_space_amplitude.max(),
                    cmap="turbo",
                )
            if support.sum() > 0:
                axes[1].set_xlim(
                    np.nonzero(support.sum(axis=0))[0][[0, -1]]
                    + np.array([-5, 5])
                )
                axes[1].set_ylim(
                    np.nonzero(support.sum(axis=1))[0][[0, -1]]
                    + np.array([-5, 5])
                )
            figure.colorbar(rcp_im, ax=axes[0])
            figure.colorbar(
                direct_space_im,
                ax=axes[1],
                extend="both" if plot_phase else None,
            )

            axes[0].set_title("Intensity sum (a.u.)")
            axes[1].set_title(
                "Phase (rad)" if plot_phase else "Amplitude (a.u.)"
            )

        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        figure.suptitle(title)

    plt.show()

    @staticmethod
    def twin_image_checkup(
        ref: np.ndarray | str,
        reconstruction: np.ndarray | str,
        phase_unwrap: bool = True,
        **plot_params,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Implementation of the check-up for the twin image problem as
        described by Manuel Sicairos (see https://www.researchgate.net/publication/233828110_Understanding_the_twin-image_problem_in_phase_retrieval)  # noqa
        The function takes two reconstructions, one is considered a
        reference. After the two reconstuctions being registered, their
        phase in the Fourier space are compared (difference). The same
        procedure is applied to the twin image of the second
        reconstruction.
        To cite the paper:
        'The smoothness of the retrieved Fourier phase differences can
        give an indication of the twin image problem. If either of the
        Fourier phase differences is smooth everywhere, this would be an
        indication that the reconstructions are free of the twin-image
        problem. In this test, the twin-image problem would be
        characterized by the appearance of smooth and random regions
        that appear complementary when comparing upright with twin
        images.'

        Args:
            ref (np.ndarray | str): the reconstruction considered as the
                reference.
            reconstruction (np.ndarray | str): the reconstruction that
                will be twinned and compared to the reference.
            phase_unwrap (bool, optional): whether to unwrap the phase
                upon plotting. Defaults to True.

        Returns:
            tuple[plt.Figure, plt.Axes]: the matpltolib figure and axes
                objects.
        """

        def check_load(data: str) -> np.ndarray:
            if isinstance(data, str):
                if data.endswith(".cxi"):
                    with silx.io.h5py_utils.File(data) as file:
                        return file["entry_1/data_1/data"][()]
                if data.endswith(".npz"):
                    with np.load(data) as file:
                        return file["arr_0"][()]
                raise NotImplementedError(
                    "File extension handling not implemented."
                )
            if isinstance(data, np.ndarray):
                return data
            raise ValueError(
                "ref and reconstruction must be either strings (path to file) "
                f"or np.ndarrays, not {type(data)}."
            )

        ref = check_load(ref)
        reconstruction = check_load(reconstruction)
        if ref.shape != reconstruction.shape:
            raise ValueError(
                f"Shapes of reference (ref.shape = {ref.shape}) and "
                "reconstruction (reconstruction.shape = "
                f"{reconstruction.shape}) must be "
                "identical."
            )

        # Register the two datasets
        shift, _, _ = phase_cross_correlation(
            np.abs(ref), np.abs(reconstruction), upsample_factor=100
        )
        print(
            "The shift calculated from the "
            f"phase_cross_correlation is: {shift}."
        )

        fourier_data = {
            "ref": ifftshift(fftn(fftshift(ref))),
            "reconstruction": ifftshift(
                fourier_shift(fftn(fftshift(reconstruction)), shift)
            ),
        }
        fourier_data["twin"] = np.conj(fourier_data["reconstruction"])

        fourier_phases = {}
        for k in fourier_data:
            phase = np.angle(fourier_data[k])
            centre_index = tuple(s // 2 for s in phase.shape)
            fourier_phases[k] = phase - phase[centre_index]

        phase_diffs = [
            PostProcessor.unwrap_phase(
                fourier_phases[k] - fourier_phases["ref"]
            )
            if phase_unwrap
            else fourier_phases[k] - fourier_phases["ref"]
            for k in ("reconstruction", "twin")
        ]

        _plot_params = {"cmap": "cet_CET_C9s_r"}

        if phase_unwrap:
            _plot_params["vmax"] = np.max(np.abs(np.asarray(phase_diffs)))
        else:
            _plot_params["vmax"] = 2 * np.pi
        _plot_params["vmin"] = -_plot_params["vmax"]
        if plot_params:
            _plot_params.update(plot_params)

        if ref.ndim == 3:
            figure, axes = plt.subplots(2, 3, layout="tight", figsize=(6, 3))
            slices = get_centred_slices(ref.shape)
            for i in range(2):
                for j, ax in enumerate(axes[i].flat):
                    ax.imshow(phase_diffs[i][slices[j]], **_plot_params)
            axes[0, 1].set_title(r"$\varphi - \varphi_{\text{ref}}$", y=1.1)
            axes[1, 1].set_title(
                r"$\varphi_{\text{twin}} - \varphi_{\text{ref}} "
                r"= -\varphi - \varphi_{\text{ref}}$",
                y=1.1,
            )
        if ref.ndim == 2:
            figure, axes = plt.subplots(1, 2, layout="tight")
            axes[0].imshow(phase_diffs[0], **_plot_params)
            axes[1].imshow(phase_diffs[1], **_plot_params)
            axes[0].set_title(r"$\varphi - \varphi_{\text{ref}}$")
            axes[1].set_title(
                r"$\varphi_{\text{twin}} - \varphi_{\text{ref}} "
                r"= -\varphi - \varphi_{\text{ref}}$",
            )
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            add_colorbar(ax, extend="both")

        return figure, axes


def extract_run_info(filename: str) -> tuple[int, str]:
    """
    Extract run number and scan info from a filename.

    Args:
        filename: Path or filename containing run information

    Returns:
        Run number and original run string
    """
    # Extract just the filename if a full path is given
    base_filename = os.path.basename(filename)

    # Try multiple patterns to extract run numbers
    run_patterns = [
        # Pattern for "Run0001" style
        (r"Run(\d+)", lambda m: int(m.group(1))),
        # Pattern for "Run_0001" style
        (r"Run_(\d+)", lambda m: int(m.group(1))),
        # Pattern for files with "_run_01" style
        (r"_run_(\d+)", lambda m: int(m.group(1))),
        # Pattern for "r0001" style
        (r"r(\d+)", lambda m: int(m.group(1))),
    ]

    for pattern, extractor in run_patterns:
        match = re.search(pattern, base_filename)
        if match:
            run_num = extractor(match)
            run_str = match.group(0)
            return run_num, run_str

    # If no pattern matches, return a default
    return 0, "unknown"
