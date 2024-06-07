import copy
import glob
import os
from typing import Type

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numpy.fft import fftn, fftshift, ifftshift
from scipy.stats import gaussian_kde
import silx.io

try:
    from pynx.cdi import (
        CDI,
        AutoCorrelationSupport,
        ScaleObj,
        SupportUpdate,
        HIO,
        RAAR,
        ER,
        FourierApplyAmplitude,
        SupportTooLarge,
        SupportTooSmall,
        InitPSF
    )
    from pynx.cdi.selection import match2
    from pynx.utils.math import ortho_modes
    IS_PYNX_AVAILABLE = True
    CDI_Type = Type[CDI]

except ImportError:
    IS_PYNX_AVAILABLE = False
    PYNX_ERROR_TEXT = (
        "'pynx' is not installed, PyNXPhaser is not available."
    )
    CDI_Type = None

from cdiutils.plot import get_plot_configs, get_figure_size
from cdiutils.utils import get_centred_slices, valid_args_only


DEFAULT_PYNX_PARAMS = {

    # support-related params
    "support_threshold": (0.15, 0.40),
    "smooth_width": (2, 0.5, 600),
    "post_expand": None,  # (-1, 1)
    "support_update_period": 50,
    "method": "rms",
    "force_shrink": False,
    "update_border_n": 0,

    # object initialisation
    # "support": "auto",
    "support_autocorrelation_threshold": (0.09, 0.11),

    # operator parameters
    "calc_llk": 50,
    "show_cdi": 0,
    "update_psf": 100,
    "fig_num": -1,
    "zero_mask": False,

    # others
    "psf": "pseudo-voigt,0.5,0.1,10",
    "compute_free_llk": True,
    "positivity": False,
    "confidence_interval_factor_mask_min": 0.5,
    "confidence_interval_factor_mask_max": 1.2,

}


class PyNXPhaser:

    def __init__(
            self,
            iobs: np.ndarray,
            mask: np.ndarray,
            params: dict = None,
            operators: dict = None,
    ) -> None:
        if not IS_PYNX_AVAILABLE:
            raise ModuleNotFoundError(PYNX_ERROR_TEXT)
        self.iobs = fftshift(iobs)
        self.mask = fftshift(mask)

        # get the pynx parameters from the default and update them with
        # those provided by the user
        # not elegant, but works...
        self.params = copy.deepcopy(DEFAULT_PYNX_PARAMS)
        if params is not None:
            self.params = self._update_params(self.params, new_params=params)

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

    def _update_params(self, target, new_params) -> dict:

        for key, value in new_params.items():
            if (
                    isinstance(value, dict)
                    and key in target
                    and isinstance(target[key], dict)
            ):
                # Recursively update sub-dictionaries
                self._update_params(target[key], value)
            else:
                # Update non-dictionary values
                target[key] = value
        return target

    def _init_operators(self) -> dict:
        operator_parameters = {
            key: self.params[key]
            for key in [
                    "calc_llk", "show_cdi", "update_psf", "fig_num",
                    "zero_mask", "confidence_interval_factor_mask_max",
                    "confidence_interval_factor_mask_min"
            ]
        }
        return {
            "er": ER(**operator_parameters),
            "hio": HIO(**operator_parameters),
            "raar": RAAR(**operator_parameters),
            "fap": FourierApplyAmplitude(
                **valid_args_only(operator_parameters, FourierApplyAmplitude)
            )
        }

    def _init_support_update(self) -> None:
        # Find which parameters are accepted using class inspection
        support_params = valid_args_only(self.params, SupportUpdate)

        if self.params["support_update_period"]:
            self.support_update = SupportUpdate(**support_params)
        else:
            self.support_update = 1

    def _init_cdi(self, verbose=False) -> CDI_Type:
        cdi = CDI(
            self.iobs, self.params["support"], self.params["obj"], self.mask
        )
        if (
                self.params["support"] is None
                and self.params["obj"] is None
        ):
            if self.params["all_random"]:
                if verbose:
                    print("Full random initialisation requested.")
                amp = np.random.uniform(
                    low=self.params["amp_range"][0],
                    high=self.params["amp_range"][1],
                    size=self.iobs.size
                ).reshape(self.iobs.shape)
                phase = np.random.uniform(
                    low=self.params["phase_range"][0],
                    high=self.params["phase_range"][1],
                    size=self.iobs.size
                ).reshape(self.iobs.shape)
                cdi.set_obj(amp * np.exp(1j * phase))
            else:
                if verbose:
                    print("Support will be initialised using autocorrelation.")
                cdi = AutoCorrelationSupport(
                    threshold=self.params["support_autocorrelation_threshold"]
                ) * cdi
        if self.params["obj"] is None and not self.params["all_random"]:
            if verbose:
                print(
                    "obj is None, will initialise object with support, and:"
                    "\n\t-amp_range: " + (
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
                size=np.flatnonzero(current_support).size
            )
            phase = np.random.uniform(
                low=self.params["phase_range"][0],
                high=self.params["phase_range"][1],
                size=np.flatnonzero(current_support).size
            )
            obj[current_support > 0] *= amp * np.exp(1j * phase)
            cdi.set_obj(obj)
            # cdi = InitObjRandom(
            #     amax=self.params["amp_range"],
            #     phirange=self.params["phase_range"]
            # ) * cdi
        if self.params["scale_obj"]:
            cdi = ScaleObj(method=self.params["scale_obj"]) * cdi
        if self.params["psf"] is not None:
            model, fwhm, eta, _ = self.params["psf"].split(",")
            cdi = InitPSF(model, float(fwhm), float(eta)) * cdi

        if self.params["compute_free_llk"]:
            cdi.init_free_pixels()

        return cdi

    def init_cdi(
            self,
            support: np.ndarray = None,
            obj: np.ndarray = None,
            phase_range: float | tuple | list | np.ndarray = np.pi,
            amp_range: float | tuple | list | np.ndarray = 100,
            all_random: bool = False,
            scale_obj: str = "I"
    ) -> None:
        # Store all method parameters into the self.params attribute
        local_parameters = locals()
        self.params.update(
            {
                param: value for param, value in local_parameters.items()
                if param != 'self'
            }
        )
        if self.params["positivity"]:
            self.params["phase_range"] = 0
        for key in ["support", "obj"]:
            if isinstance(self.params[key], np.ndarray):
                self.params[key] = fftshift(self.params[key])
        for key in ["amp_range", "phase_range"]:
            if isinstance(self.params[key], (int, float)):
                self.params[key] = (
                    -self.params[key] if key == "phase_range" else 0,
                    self.params[key]
                )

        self.cdi = self._init_cdi(verbose=True)

    @classmethod
    def read_instructions(cls, recipe: str) -> list[str]:
        recipe = recipe.replace(" ", "")
        instructions = recipe.split(",")
        if "=" in instructions:
            pass
        return instructions

    def run(
            self, recipe: str,
            cdi: CDI_Type = None,
            init_cdi: bool = True
    ) -> None:
        if cdi is None:
            cdi = self.cdi
            if init_cdi:
                cdi = self._init_cdi()

        instructions = self.read_instructions(recipe)
        for i, instruction in enumerate(instructions):
            print(f"Instruction #{i}: {instruction}")

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
                        cdi = (
                            self.support_update * (algo**support_update_period)
                            ** (iteration // support_update_period)
                        ) * cdi
                        attempt = self.wrong_support_failure_tolerance

                    except SupportTooLarge:
                        print(
                            "Support is too large, reduce it. "
                            f"Attempt #{attempt}"
                        )
                        self.support_update.threshold_relative /= (
                            self.support_threshold_auto_tune_factor
                        )
                        attempt += 1
                    except SupportTooSmall:
                        print(
                            "Support is too small, enlarge it. "
                            f"Attempt #{attempt}"
                        )
                        self.support_update.threshold_relative *= (
                            self.support_threshold_auto_tune_factor
                        )
                        attempt += 1
            elif instruction.lower() == "fap":
                self.operators["fap"] * cdi
            else:
                raise ValueError(f"Invalid instruction ({instruction}).")

    def run_multiple_instances(
            self,
            run_nb: int,
            recipe: str,
            init_cdi: bool = True
    ) -> None:
        if init_cdi:
            for i in range(run_nb):
                self.cdi_list.append(self._init_cdi())
        else:
            if self.cdi_list is None or self.cdi_list == []:
                raise ValueError(
                    "CDI object are not itialised, init_cdi should be True."

                )
        for i, cdi in enumerate(self.cdi_list):
            print(f"Run #{i}")
            self.run(recipe, cdi, init_cdi=False)

    # def sharpness(self, cdi):
    #     amplitude = np.abs(cdi.get_obj(shift=True)) * cdi.get_support(shift=True)
    #     return np.mean(amplitude ** 4)
 
    # def mean_to_max(self, cdi):
    #     metrics = PhasingResultAnalyser.amplitude_based_metrics(
    #         np.abs(cdi.get_obj(shift=True)),
    #          cdi.get_support(shift=True)
    #     )
    #     return metrics["mean_to_max"]

    def genetic_phasing(
            self,
            run_nb: int,
            genetic_pass_nb: int,
            recipe: str,
            selection_method: str = "sharpness",
            init_cdi: bool = True
    ):
        if init_cdi:
            for i in range(run_nb):
                self.cdi_list.append(self._init_cdi())
        else:
            if self.cdi_list is None or self.cdi_list == []:
                raise ValueError(
                    "CDI object are not itialised, init_cdi should be True."
                )
        if not selection_method in ("sharpness", "mean_to_max"):
            raise ValueError(
                f"Invalid selection_method ({selection_method}), can be"
                "'sharpness' or 'mean_to_max'."
            )
        metrics = [None for _ in range(run_nb)]

        for i in range(genetic_pass_nb + 1):
            if i == 0:
                print("First reconstruction pass")
            else:
                print(
                    f"Genetic pass #{i}.\n"
                    f"Updating cdi objects with best reconstruction ()."
                )
                for i in range(run_nb):
                    metrics[i] = PhasingResultAnalyser.amplitude_based_metrics(
                        np.abs(self.cdi_list[i].get_obj(shift=True)),
                        self.cdi_list[i].get_support(shift=True)
                    )[selection_method]

                indice = np.argsort(metrics)[0]
                amplitude_reference = np.abs(self.cdi_list[indice].get_obj())
                for i in range(run_nb):
                    if i == indice:
                        continue
                    new_obj = self.cdi_list[i].get_obj()
                    new_obj = np.sqrt(
                        np.abs(new_obj)
                        * amplitude_reference
                    ) * np.exp(1j * np.angle(new_obj))
                    self.cdi_list[i].set_obj(new_obj)
            self.run_multiple_instances(run_nb, recipe, init_cdi=False)

    @staticmethod
    def plot_cdi(
            cdi: CDI_Type,
            spaces: str = "both",
            axis: int = 0,
            title: str = None
    ) -> None:
        """
        See ShowCDI operator to see how Vincent plots CDI objects.
        https://gitlab.esrf.fr/favre/PyNX/-/blob/master/pynx/cdi/cpu_operator.py?ref_type=heads
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
            )
        }

        if spaces == "direct":
            quantities["support"] = support

        elif spaces == "both":
            quantities["calculated_intensity"] = np.log10(
                np.abs(fftshift(fftn(cdi.get_obj().copy()))[the_slice])**2
            )
            iobs = cdi.get_iobs(shift=True).copy()[the_slice]
            tmp = np.logical_and(iobs > -1e19, iobs < 0)
            if tmp.sum() > 0:
                # change back free pixels to their real intensity
                iobs[tmp] = -iobs[tmp] - 1
            iobs[iobs < 0] = 0

            quantities["observed_intensity"] = iobs

        nrows, ncols = (1, 3) if spaces == "direct" else (2, 2)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3, 3))
        for ax, key in zip(axes.ravel(), quantities.keys()):
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
                plot_params["norm"] = LogNorm(
                    vmin=0.5,
                    vmax=quantities[key].max()
                )
                plot_params.pop("vmin"), plot_params.pop("vmax")

            # Set the ax title and remove it from the plotting params
            ax.set_title(plot_params.pop("title"))

            image = ax.matshow(quantities[key], origin="lower", **plot_params)

            # if key in ("support", "phase") and support.sum() > 0:
            if key in ("amplitude", "support", "phase") and support.sum() > 0:
                ax.set_xlim(
                    np.nonzero(support.sum(axis=1))[0][[0, -1]]
                    + np.array([-10, 10])
                )
                ax.set_ylim(
                    np.nonzero(support.sum(axis=0))[0][[0, -1]]
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
        fig.tight_layout()


class PhasingResultAnalyser:
    def __init__(
            self,
            cdi_results: list = None,
            result_dir_path: str = None
    ) -> None:
        if cdi_results is None and result_dir_path is None:
            raise ValueError(
                "Both parameters cdi_results and result_dir_path cannot be "
                "None"
            )

        if cdi_results is not None and result_dir_path is not None:
            raise ValueError(
                "cdi_results and result_dir_path cannot be provided "
                "simultaneously. "
            )
        # Convert the parsed list into a dict whose keys are run numbers
        self.cdi_results = {
            f"Run{i+1:04d}": cdi for i, cdi in enumerate(cdi_results)
        }
        self.result_dir_path = result_dir_path
        self._metrics = None
        self._sorted_phasing_results = None
        self.result_paths = []

    def find_phasing_results(self) -> None:
        """
        Find the last phasing results (.cxi files) and add them to the
        given list if provided, otherwise create the list.
        """
        self.result_paths = []
        fetched_paths = glob.glob(self.result_dir_path + "/*Run*.cxi")
        for path in fetched_paths:
            if os.path.isfile(path):
                self.result_paths.append(path)

    @staticmethod
    def amplitude_based_metrics(
            amplitude: np.ndarray,
            support: np.ndarray
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
        sharpness = np.mean((amplitude * support)**4)
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
            "sharpness": sharpness
        }

    def analyse_phasing_results(
            self,
            sorting_criterion: str = "mean_to_max",
            plot_phasing_results: bool = True,
            plot_phase: bool = True,
    ):
        criteria = ["mean_to_max", "std", "llk", "llkf", "sharpness", "all"]
        if sorting_criterion not in criteria:
            raise ValueError(
                f"Provided criterion ({sorting_criterion}) is unknown. "
                f"Possible criteria are:\n{criteria}."
            )

        if self._sorted_phasing_results is None:
            print("[INFO] Computing metrics...")

            self._metrics = {
                m: {f: None for f in self.result_paths}
                for m in criteria
            }
            if self.cdi_results is not None:
                for run in self.cdi_results:
                    self._metrics["llk"][run] = (
                        self.cdi_results[run].llk_poisson
                    )
                    self._metrics["llkf"][run] = (
                        self.cdi_results[run].llk_poisson_free
                    )

                    amplitude = np.abs(self.cdi_results[run].get_obj())
                    support = self.cdi_results[run].get_support()
                    # update the metric dictionary
                    amplitude_based_metrics = self.amplitude_based_metrics(
                        amplitude, support
                    )
                    for m in ["mean_to_max", "std", "sharpness"]:
                        self._metrics[m][run] = amplitude_based_metrics[m]
            else:
                self.find_phasing_results()

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
                        (self._metrics[m][k] - minimum_value) / ptp_value
                    )

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
                    reverse=False
                )
            )
        if self.cdi_results is not None:
            runs = [
                run.split("Run")[1][2:4]
                for run in self._sorted_phasing_results
            ]
        else:
            runs = [
                file.split("Run")[1][2:4]
                for file in self._sorted_phasing_results
            ]

        figsize = get_figure_size(scale=0.75)
        figure, ax = plt.subplots(1, 1, figsize=figsize)
        colors = {
            m: c for m, c in zip(
                self._metrics,
                ["lightcoral", "mediumslateblue",
                 "dodgerblue", "plum", "teal", "crimson"]
            )
        }
        for m in self._metrics:
            ax.plot(
                runs,
                [self._metrics[m][f] for f in self._sorted_phasing_results],
                label=m,
                color=colors[m],
                marker='o',
                markersize=4,
                markerfacecolor=colors[m],
                markeredgewidth=0.5,
                markeredgecolor='k'
            )
        ax.set_ylabel("Normalised metric")
        ax.set_xlabel("Run number")
        figure.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.915),
            ncol=len(criteria)
        )
        figure.suptitle(
            "Phasing result analysis (the lower the better)\n"
        )
        figure.tight_layout()
        plt.show()
        print(
            "[INFO] the sorted list of runs using sorting_criterion "
            f"'{sorting_criterion}' is:\n{runs}"
        )

        if plot_phasing_results:
            print("[INFO] Plotting phasing results...")
            for result in self._sorted_phasing_results:
                if self.cdi_results is not None:
                    data = self.cdi_results[result].get_obj(shift=True)
                    support = self.cdi_results[result].get_support(shift=True)
                else:
                    with silx.io.h5py_utils.File(result, "r") as file:
                        data = file["entry_1/data_1/data"][()]
                        support = file["entry_1/image_1/support"][()]
                run = int(result.split("Run")[1][:4])
                title = (
                    f"Phasing results, run {int(result.split('Run')[1][:4])}"
                )
                self.plot_phasing_result(data, support, title, plot_phase)

    def decompose_into_one_mode(self, verbose: bool = True) -> np.ndarray:
        if not IS_PYNX_AVAILABLE:
            raise ValueError(PYNX_ERROR_TEXT)
        if self._sorted_phasing_results is not None:
            result_keys = self._sorted_phasing_results
        else:
            if verbose:
                print(
                    "Results are not sorted, therefore matching won't be "
                    "done against the best result."
                )
            if self.result_paths is None:
                self.find_phasing_results()
            result_keys = self.result_paths
        results = []
        for r in result_keys:
            if self.cdi_results is not None:
                results.append(self.cdi_results[r].get_obj(shift=True))
            else:
                with silx.io.h5py_utils.File(r, "r") as file:
                    results.append(file["entry_1/data_1/data"][()])

        first_result = results[0]
        match2_results = []
        for i in range(1, len(results)):
            _, d2c, r = match2(first_result, results[i])
            match2_results.append(d2c)  # don't know if needed, that's PyNX way
            if verbose:
                print(f"R_match({i}) = {r*100:6.3f} %")

        principle_mode, mode_weights = ortho_modes(
            match2_results, nb_mode=1, return_weights=True
        )
        if verbose:
            print(f"First mode represents {mode_weights[0] * 100:6.3f} %")
        return principle_mode[0]

    @staticmethod
    def plot_phasing_result(
            data: np.ndarray,
            support: np.ndarray,
            title: str = None,
            plot_phase: bool = False
    ) -> None:
        """
        Plot the reconstructed object in reciprocal and direct spaces.
        """
        reciprocal_space_data = np.abs(ifftshift(fftn(fftshift(data))))**2
        direct_space_amplitude = np.abs(data)
        normalised_direct_space_amplitude = (
            (direct_space_amplitude - np.min(direct_space_amplitude))
            / np.ptp(direct_space_amplitude)
        )
        direct_space_phase = np.angle(data)

        if data.ndim == 3:
            figsize = get_figure_size(scale=0.75, subplots=(3, 3))
            figure, axes = plt.subplots(2, 3, figsize=figsize)

            slices = get_centred_slices(data.shape)
            for i in range(3):
                rcp_im = axes[0, i].matshow(
                    np.sum(reciprocal_space_data, axis=i),
                    cmap="turbo",
                    norm=LogNorm()
                )
                if plot_phase:
                    direct_space_im = axes[1, i].matshow(
                        direct_space_phase[slices[i]],
                        vmin=-np.pi,
                        vmax=np.pi,
                        alpha=normalised_direct_space_amplitude[slices[i]],
                        cmap="cet_CET_C9s_r"
                    )
                else:
                    direct_space_im = axes[1, i].matshow(
                        direct_space_amplitude[slices[i]],
                        vmin=0,
                        vmax=direct_space_amplitude.max(),
                        cmap="turbo"
                    )

                if support.sum() > 0:
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

            axes[0, 1].set_title("Log. Proj. intensity (a.u.)")
            axes[1, 1].set_title(
                "Phase (rad)" if plot_phase else "Amplitude (a.u.)"
            )

        if data.ndim == 2:
            figsize = get_figure_size(scale=0.750, subplots=(1, 2))
            figure, axes = plt.subplots(1, 2, figsize=figsize)

            rcp_im = axes[0].matshow(
                reciprocal_space_data,
                cmap="turbo",
                norm=LogNorm()
            )
            if plot_phase:
                direct_space_im = axes[1].matshow(
                    direct_space_phase,
                    vmin=-np.pi,
                    vmax=np.pi,
                    alpha=normalised_direct_space_amplitude,
                    cmap="cet_CET_C9s_r"
                )
            else:
                direct_space_im = axes[1].matshow(
                    direct_space_amplitude,
                    vmin=0,
                    vmax=direct_space_amplitude.max(),
                    cmap="turbo"
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
            figure.colorbar(rcp_im, ax=axes[0], extend="both")
            figure.colorbar(direct_space_im, ax=axes[1], extend="both")

            axes[0].set_title("Log. Proj. intensity (a.u.)")
            axes[1].set_title(
                "Phase (rad)" if plot_phase else "Amplitude (a.u.)"
            )

        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        figure.suptitle(title)
        figure.tight_layout()
    plt.show()
