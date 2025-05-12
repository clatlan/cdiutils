from pynx.cdi import SupportUpdate, ScaleObj, AutoCorrelationSupport, \
    InitPSF, ShowCDI, HIO, RAAR, ER, SupportTooLarge, CDI, InitFreePixels, \
    InterpIobsMask
from pynx.cdi.runner.runner import default_params as params
from pynx.utils.math import smaller_primes

import numpy as np
import glob
import os
import operator as operator_lib
from datetime import datetime
from numpy.fft import fftshift
from scipy.ndimage import center_of_mass
from shlex import quote
from IPython.display import clear_output
from ast import literal_eval
from typing import Tuple, Union, Optional, List, Any
import h5py

import ipywidgets as widgets
from ipywidgets import interactive
from IPython.display import display


class TabPhaseRetrieval(widgets.VBox):
    """
    Regroups all the parameters needed to use the operators in a notebook.

    The goal is to have an interactive use of PyNX.
    """

    def __init__(self, box_style="", work_dir=None):
        """

        """
        super(TabPhaseRetrieval, self).__init__()

        # Brief header describing the tab
        self.header = 'Phase retrieval'
        self.box_style = box_style

        # Define widgets
        self.unused_label_data = widgets.HTML(
            value="<p style='font-weight: bold;font-size:1.2em'>\
            Data files",
            style={
                'description_width': 'initial'},
            layout=widgets.Layout(width='90%', height="35px")
        )

        if work_dir == None:
            work_dir = os.getcwd()

        self.parent_folder = widgets.Dropdown(
            options=sorted([x[0] + "/" for x in os.walk(work_dir)]),
            value=work_dir + "/",
            placeholder=work_dir + "/",
            description='Parent folder:',
            continuous_update=False,
            layout=widgets.Layout(width='90%'),
            style={'description_width': 'initial'}
        )

        self.iobs = widgets.Dropdown(
            options=[""]
            + sorted([os.path.basename(f) for f in
                      glob.glob(work_dir + "*.npz")],
                     key=os.path.getmtime),
            description='Dataset',
            layout=widgets.Layout(width='90%'),
            style={'description_width': 'initial'}
        )

        self.mask = widgets.Dropdown(
            options=[""]
            + sorted([os.path.basename(f) for f in
                      glob.glob(work_dir + "*.npz")],
                     key=os.path.getmtime),
            description='Mask',
            layout=widgets.Layout(width='90%'),
            style={'description_width': 'initial'}
        )

        self.support = widgets.Dropdown(
            options=[""]
            + sorted([os.path.basename(f) for f in
                      glob.glob(work_dir + "*.npz")],
                     key=os.path.getmtime),
            value="",
            description='Support',
            layout=widgets.Layout(width='90%'),
            style={'description_width': 'initial'}
        )

        self.obj = widgets.Dropdown(
            options=[""]
            + sorted([os.path.basename(f) for f in
                      glob.glob(work_dir + "*.npz")],
                     key=os.path.getmtime),
            value="",
            description='Object',
            layout=widgets.Layout(width='90%'),
            style={'description_width': 'initial'}
        )

        self.auto_center_resize = widgets.Checkbox(
            value=False,
            description='Auto center and resize',
            continuous_update=False,
            indent=False,
            layout=widgets.Layout(height="50px"),
            icon='check'
        )

        self.max_size = widgets.BoundedIntText(
            value=256,
            step=1,
            min=0,
            max=1000,
            layout=widgets.Layout(
                height="50px", width="30%"),
            continuous_update=False,
            description='Maximum array size for cropping:',
            readout=True,
            style={
                'description_width': 'initial'},
        )

        self.unused_label_support = widgets.HTML(
            value="<p style='font-weight: bold;font-size:1.2em'>\
            Support parameters",
            style={
                'description_width': 'initial'},
            layout=widgets.Layout(width='90%', height="35px")
        )

        self.support_threshold = widgets.Text(
            value="(0.23, 0.30)",
            placeholder="(0.23, 0.30)",
            description='Support threshold',
            layout=widgets.Layout(
                height="50px", width="30%"),
            continuous_update=False,
            style={'description_width': 'initial'}
        )

        self.support_only_shrink = widgets.Checkbox(
            value=False,
            description='Support only shrink',
            continuous_update=False,
            indent=False,
            layout=widgets.Layout(
                height="50px", width="15%"),
            icon='check'
        )

        self.support_update_period = widgets.BoundedIntText(
            value=20,
            max=500,
            step=5,
            layout=widgets.Layout(
                height="50px", width="30%"),
            continuous_update=False,
            description='Support update period:',
            readout=True,
            style={
                'description_width': 'initial'},
        )

        self.support_smooth_width = widgets.Text(
            value="(2, 1, 600)",
            placeholder="(2, 1, 600)",
            description='Support smooth width',
            layout=widgets.Layout(
                height="50px", width="30%"),
            continuous_update=False,
            style={'description_width': 'initial'}
        )

        self.support_post_expand = widgets.Text(
            value="(1, -2, 1)",
            placeholder="(1, -2, 1)",
            description='Support post expand',
            layout=widgets.Layout(width="30%"),
            continuous_update=False,
            style={'description_width': 'initial'}
        )

        self.support_method = widgets.Dropdown(
            options=["max", "average", "rms"],
            value="rms",
            description='Support method',
            layout=widgets.Layout(
                height="25px", width='30%'),
            continuous_update=False,
            style={'description_width': 'initial'}
        )

        self.support_autocorrelation_threshold = widgets.Text(
            value="(0.10)",
            placeholder="(0.10)",
            description='Support autocorrelation threshold',
            layout=widgets.Layout(
                height="50px", width="40%"),
            continuous_update=False,
            style={'description_width': 'initial'}
        )

        self.unused_label_psf = widgets.HTML(
            value="<p style='font-weight: bold;font-size:1.2em'>\
            Point spread function parameters",
            style={
                'description_width': 'initial'},
            layout=widgets.Layout(width='90%', height="35px")
        )

        self.psf = widgets.Checkbox(
            value=True,
            description='Use point spread function',
            continuous_update=False,
            indent=False,
            layout=widgets.Layout(width="30%", height="50px"),
            icon='check'
        )

        self.psf_model = widgets.Dropdown(
            options=[
                "gaussian", "lorentzian", "pseudo-voigt"],
            value="pseudo-voigt",
            description='PSF peak shape',
            continuous_update=False,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width="30%", height="25px"),
        )

        self.fwhm = widgets.FloatText(
            value=0.5,
            step=0.01,
            min=0,
            continuous_update=False,
            description="FWHM:",
            layout=widgets.Layout(
                width='15%', height="50px"),
            style={
                'description_width': 'initial'}
        )

        self.eta = widgets.FloatText(
            value=0.05,
            step=0.01,
            max=1,
            min=0,
            continuous_update=False,
            description='Eta:',
            layout=widgets.Layout(
                width='15%', height="50px"),
            readout=True,
            style={
                'description_width': 'initial'}
        )

        self.psf_filter = widgets.Dropdown(
            options=["None", "hann", "tukey"],
            value="None",
            description='PSF filter',
            layout=widgets.Layout(width='15%'),
            continuous_update=False,
            style={'description_width': 'initial'}
        )

        self.update_psf = widgets.BoundedIntText(
            value=20,
            step=5,
            continuous_update=False,
            description='Update PSF every:',
            layout=widgets.Layout(
                width='35%', height="50px"),
            readout=True,
            style={
                'description_width': 'initial'}
        )

        self.unused_label_algo = widgets.HTML(
            value="<p style='font-weight: bold;font-size:1.2em'>\
            Iterative algorithms parameters",
            style={
                'description_width': 'initial'},
            layout=widgets.Layout(width='90%', height="35px")
        )

        self.nb_raar = widgets.BoundedIntText(
            value=1000,
            min=0,
            max=9999,
            step=10,
            continuous_update=False,
            description='Nb of RAAR:',
            layout=widgets.Layout(
                height="35px", width="20%"),
            readout=True,
            style={
                'description_width': 'initial'},
        )

        self.nb_hio = widgets.BoundedIntText(
            value=400,
            min=0,
            max=9999,
            step=10,
            continuous_update=False,
            description='Nb of HIO:',
            layout=widgets.Layout(
                height="35px", width="20%"),
            readout=True,
            style={
                'description_width': 'initial'},
        )

        self.nb_er = widgets.BoundedIntText(
            value=300,
            min=0,
            max=9999,
            step=10,
            continuous_update=False,
            description='Nb of ER:',
            layout=widgets.Layout(
                height="35px", width="20%"),
            readout=True,
            style={
                'description_width': 'initial'},
        )

        self.nb_ml = widgets.BoundedIntText(
            value=0,
            min=0,
            max=9999,
            step=10,
            continuous_update=False,
            description='Nb of ML:',
            layout=widgets.Layout(
                height="35px", width="20%"),
            readout=True,
            style={
                'description_width': 'initial'},
        )

        self.nb_run = widgets.BoundedIntText(
            value=30,
            min=0,
            max=100,
            continuous_update=False,
            description='Number of run:',
            layout=widgets.Layout(width="20%", height="50px"),
            readout=True,
            style={
                'description_width': 'initial'},
        )

        self.unused_label_filtering = widgets.HTML(
            value="<p style='font-weight: bold;font-size:1.2em'>\
            Filtering criteria for reconstructions",
            style={
                'description_width': 'initial'},
            layout=widgets.Layout(width='90%', height="35px")
        )

        self.filter_criteria = widgets.Dropdown(
            options=[
                ("No filtering",
                 "no_filtering"),
                ("Standard deviation",
                    "standard_deviation"),
                ("Log-likelihood (FLLK)", "FLLK"),
                ("FLLK > Standard deviation",
                    "FLLK_standard_deviation"),
                # ("Standard deviation > FLLK", "standard_deviation_FLLK"),
            ],
            value="FLLK_standard_deviation",
            description='Filtering criteria',
            layout=widgets.Layout(width='50%'),
            style={'description_width': 'initial'}
        )

        self.nb_run_keep = widgets.BoundedIntText(
            value=10,
            continuous_update=False,
            description='Number of run to keep:',
            layout=widgets.Layout(width='30%', height="50px"),
            readout=True,
            style={
                'description_width': 'initial'},
        )

        self.unused_label_options = widgets.HTML(
            value="<p style='font-weight: bold;font-size:1.2em'>\
            Options",
            style={
                'description_width': 'initial'},
            layout=widgets.Layout(width='90%', height="35px")
        )

        self.live_plot = widgets.BoundedIntText(
            value=200,
            step=10,
            max=500,
            min=0,
            continuous_update=False,
            description='Plot every:',
            readout=True,
            layout=widgets.Layout(
                height="50px", width="20%"),
            style={
                'description_width': 'initial'},
        )

        self.plot_axis = widgets.Dropdown(
            options=[0, 1, 2],
            value=0,
            description='Axis used for plots',
            layout=widgets.Layout(width='20%'),
            style={'description_width': 'initial'}
        )

        self.verbose = widgets.BoundedIntText(
            value=100,
            min=10,
            max=300,
            continuous_update=False,
            description='Verbose:',
            layout=widgets.Layout(width='20%', height="50px"),
            readout=True,
            style={
                'description_width': 'initial'},
        )

        self.rebin = widgets.Text(
            value="(1, 1, 1)",
            placeholder="(1, 1, 1)",
            description='Rebin',
            layout=widgets.Layout(width='20%', height="50px"),
            continuous_update=False,
            style={'description_width': 'initial'}
        )

        self.positivity = widgets.Checkbox(
            value=False,
            description='Force positivity',
            continuous_update=False,
            indent=False,
            style={
                'description_width': 'initial'},
            layout=widgets.Layout(
                height="50px", width="15%"),
            icon='check'
        )

        self.beta = widgets.FloatText(
            value=0.9,
            step=0.01,
            max=1,
            min=0,
            continuous_update=False,
            description='Beta parameter for RAAR and HIO:',
            layout=widgets.Layout(
                width='35%', height="50px"),
            readout=True,
            style={
                'description_width': 'initial'},
        )

        self.detwin = widgets.Checkbox(
            value=False,
            description='Detwinning',
            continuous_update=False,
            indent=False,
            style={
                'description_width': 'initial'},
            layout=widgets.Layout(
                height="50px", width="15%"),
            icon='check'
        )

        self.calc_llk = widgets.BoundedIntText(
            value=50,
            min=0,
            max=100,
            continuous_update=False,
            description='Log likelihood update interval:',
            layout=widgets.Layout(width="25%", height="50px"),
            readout=True,
            style={
                'description_width': 'initial'},
        )

        self.unused_label_mask_options = widgets.HTML(
            value="<p style='font-weight: bold;font-size:1.2em'>\
            Mask options</p>",
            style={
                'description_width': 'initial'},
            layout=widgets.Layout(width='90%', height="35px")
        )

        self.zero_mask = widgets.Dropdown(
            options=("True", "False", 'auto'),
            value='False',
            description='Force mask pixels to zero',
            continuous_update=False,
            indent=False,
            style={
                'description_width': 'initial'},
            layout=widgets.Layout(width="40%"),
            icon='check'
        )

        self.mask_interp = widgets.Text(
            value="(8, 2)",
            description='Mask interp.',
            layout=widgets.Layout(
                height="50px", width="40%"),
            continuous_update=False,
            style={'description_width': 'initial'}
        )

        self.unused_label_run_options = widgets.HTML(
            value="<p style='font-weight: bold;font-size:1.2em'>\
            Job options</p>",
            style={
                'description_width': 'initial'},
            layout=widgets.Layout(width='90%', height="35px")
        )


        self.run_phase_retrieval = widgets.ToggleButtons(
            options=[
                ('No phase retrieval', False),
                ('Run batch job (slurm)', "batch"),
                ("Run script locally", "local_script"),
                ("Use operators", "operators"),
            ],
            value=False,
            tooltips=[
                "Click to be able to change parameters",
                "Collect parameters to run a job on slurm, will \
                automatically apply a std deviation filter and run modes \
                decomposition, freed the kernel",
                "Run script on jupyter notebook environment, uses notebook\
                 kernel, will be performed in background also but more \
                 slowly, good if you cannot use jobs.",
                r"Use operators on local environment, if using PSF, it is \
                activated after 50\% of RAAR cycles"
            ],
            continuous_update=False,
            button_style='',
            layout=widgets.Layout(
                width='100%', height="75x"),
            style={
                'description_width': 'initial'},
            icon='fast-forward',
            description="I. Click below to run the phase retrieval"
        )

        self.run_pynx_tools = widgets.ToggleButtons(
            options=[
                ('No tool running', False),
                ("Modes decomposition",
                    "modes"),
                ("Filter reconstructions",
                    "filter")
            ],
            value=False,
            tooltips=[
                "Click to be able to change parameters",
                "Run modes decomposition in data folder, selects *FLLK*.cxi\
                 files",
                "Filter reconstructions"
            ],
            continuous_update=False,
            button_style='',
            layout=widgets.Layout(
                width='100%', height="75px"),
            style={
                'description_width': 'initial'},
            icon='fast-forward',
            description="II. Click below to filter your solutions or create a \
            single solution after phase retrieval."
        )

        # Define children
        self.children = (
            self.unused_label_data,
            self.parent_folder,
            self.iobs,
            self.mask,
            self.support,
            self.obj,
            widgets.HBox([
                self.auto_center_resize,
                self.max_size,
            ]),
            self.unused_label_support,
            widgets.HBox([
                self.support_threshold,
                self.support_only_shrink
            ]),
            widgets.HBox([
                self.support_update_period,
                self.support_smooth_width,
                self.support_post_expand,
            ]),
            widgets.HBox([
                self.support_method,
                self.support_autocorrelation_threshold,
            ]),
            self.unused_label_psf,
            widgets.HBox([
                self.psf,
                self.psf_model,
                self.fwhm,
                self.eta,
            ]),
            widgets.HBox([
                self.psf_filter,
                self.update_psf,
            ]),
            self.unused_label_algo,
            widgets.HBox([
                self.nb_hio,
                self.nb_raar,
                self.nb_er,
                self.nb_ml,
            ]),
            self.nb_run,
            self.unused_label_filtering,
            widgets.HBox([
                self.filter_criteria,
                self.nb_run_keep,
            ]),
            self.unused_label_options,
            widgets.HBox([
                self.live_plot,
                self.plot_axis,
                self.verbose,
            ]),
            widgets.HBox([
                self.rebin,
                self.positivity,
            ]),
            widgets.HBox([
                self.beta,
                self.detwin,
                self.calc_llk,
            ]),
            self.unused_label_mask_options,
            widgets.HBox([
                self.zero_mask,
                self.mask_interp,
            ]),
            self.unused_label_run_options,
            self.run_phase_retrieval,
            self.run_pynx_tools,
        )

        # Assign handler
        self.parent_folder.observe(
            self.pynx_folder_handler, names="value")
        self.psf.observe(
            self.pynx_psf_handler, names="value")
        self.psf_model.observe(
            self.pynx_peak_shape_handler, names="value")
        self.run_phase_retrieval.observe(
            self.run_pynx_handler, names="value")
        self.run_pynx_tools.observe(
            self.run_pynx_handler, names="value")

    # Define handlers
    def pynx_folder_handler(self, change: Any) -> None:
        """
        Handles changes related to the pynx folder.

        Parameters:
        ----------
        change (Any): The change event triggered by the observer.

        Returns:
        -------
        None
        """
        if hasattr(change, "new"):
            change = change.new

        list_all_npz = [os.path.basename(f) for f in sorted(
            glob.glob(change + "/*.npz"), key=os.path.getmtime)]

        list_probable_iobs_files = [os.path.basename(f) for f in sorted(
            glob.glob(change + "/*_pynx_*.npz"), key=os.path.getmtime)]

        list_probable_mask_files = [os.path.basename(f) for f in sorted(
            glob.glob(change + "/*mask*.npz"), key=os.path.getmtime)]

        # support list
        self.support.options = [""]\
            + [os.path.basename(f) for f in sorted(
                glob.glob(change + "/*.npz"), key=os.path.getmtime)]

        # obj list
        self.obj.options = [""]\
            + [os.path.basename(f) for f in sorted(
                glob.glob(change + "/*.npz"), key=os.path.getmtime)]

        # Find probable iobs file
        temp_list = list_all_npz.copy()
        for f in list_probable_iobs_files:
            try:
                temp_list.remove(f)
            except ValueError:
                # Not in list
                pass
        sorted_iobs_list = list_probable_iobs_files + temp_list + [""]

        # Find probable mask file
        temp_list = list_all_npz.copy()
        for f in list_probable_mask_files:
            try:
                temp_list.remove(f)
            except ValueError:
                # not in list
                pass
        sorted_mask_list = list_probable_mask_files + temp_list + [""]

        # iobs list
        self.iobs.options = sorted_iobs_list

        # mask list
        self.mask.options = sorted_mask_list

    def pynx_psf_handler(self, change: Any) -> None:
        """
        Handles changes related to the psf.

        The function takes the `change` argument, which is expected to contain information
        related to the change event. If `change` has a `new` attribute, the value of `change`
        is set to `change.new`.

        The function disables or enables a number of widget objects (`self.psf_model`,
        `self.fwhm`, `self.eta`, `self.psf_filter`, and `self.update_psf`) based on the value
        of `change`. If `change` is truthy, the widgets are enabled. If `change` is falsy,
        the widgets are disabled.

        The function also calls `self.pynx_peak_shape_handler` with the `change` argument set
        to `self.psf_model.value`.

        Parameters
        ----------
        change :
            The new value of the change event.

        Returns
        -------
        None
        """
        if hasattr(change, "new"):
            change = change.new

        for w in [
            self.psf_model,
            self.fwhm,
            self.eta,
            self.psf_filter,
            self.update_psf,
        ]:
            if change:
                w.disabled = False
            else:
                w.disabled = True

        self.pynx_peak_shape_handler(
            change=self.psf_model.value)

    def pynx_peak_shape_handler(self, change: Any) -> None:
        """
        Handles changes related to psf the peak shape.

        Parameters
        ----------
        change : Any
            The new value of the change event.

        Returns
        -------
        None
        """
        if hasattr(change, "new"):
            change = change.new

        if change != "pseudo-voigt":
            self.eta.disabled = True

        else:
            self.eta.disabled = False

    def run_pynx_handler(self, change: Any) -> None:
        """
        Handles changes related to the phase retrieval.

        Parameters
        ----------
        change : Any
            The new value of the change event.

        Returns
        -------
        None
        """
        if change.new:
            for w in self.children:
                if isinstance(w, widgets.widgets.widget_box.HBox):
                    for wc in w.children:
                        wc.disabled = True
                else:
                    w.disabled = True

            if isinstance(self.run_phase_retrieval.value, str):
                self.run_phase_retrieval.disabled = False
            elif isinstance(self.run_pynx_tools.value, str):
                self.run_pynx_tools.disabled = False

        elif not change.new:
            for w in self.children:
                if isinstance(w, widgets.widgets.widget_box.HBox):
                    for wc in w.children:
                        wc.disabled = False
                else:
                    w.disabled = False

            self.pynx_psf_handler(
                change=self.psf.value)

    def stand_alone(self, energy, detector_distance, pixel_size_detector):
        init_phase_retrieval_tab_gui = interactive(
            init_phase_retrieval_tab,
            parent_folder=self.parent_folder,
            iobs=self.iobs,
            mask=self.mask,
            support=self.support,
            obj=self.obj,
            auto_center_resize=self.auto_center_resize,
            max_size=self.max_size,
            support_threshold=self.support_threshold,
            support_only_shrink=self.support_only_shrink,
            support_update_period=self.support_update_period,
            support_smooth_width=self.support_smooth_width,
            support_post_expand=self.support_post_expand,
            support_method=self.support_method,
            support_autocorrelation_threshold=self.support_autocorrelation_threshold,
            psf=self.psf,
            psf_model=self.psf_model,
            fwhm=self.fwhm,
            eta=self.eta,
            psf_filter=self.psf_filter,
            update_psf=self.update_psf,
            nb_hio=self.nb_hio,
            nb_raar=self.nb_raar,
            nb_er=self.nb_er,
            nb_ml=self.nb_ml,
            nb_run=self.nb_run,
            filter_criteria=self.filter_criteria,
            nb_run_keep=self.nb_run_keep,
            live_plot=self.live_plot,
            plot_axis=self.plot_axis,
            verbose=self.verbose,
            rebin=self.rebin,
            positivity=self.positivity,
            beta=self.beta,
            detwin=self.detwin,
            calc_llk=self.calc_llk,
            zero_mask=self.zero_mask,
            mask_interp=self.mask_interp,
            run_phase_retrieval=self.run_phase_retrieval,
            run_pynx_tools=self.run_pynx_tools,
            energy=energy,
            detector_distance=detector_distance,
            pixel_size_detector=pixel_size_detector,
        )

        window = widgets.VBox([
            self,
            init_phase_retrieval_tab_gui.children[-1]
        ])

        display(window)


def init_phase_retrieval_tab(
    parent_folder,
    iobs,
    mask,
    support,
    obj,
    auto_center_resize,
    max_size,
    support_threshold,
    support_only_shrink,
    support_update_period,
    support_smooth_width,
    support_post_expand,
    support_method,
    support_autocorrelation_threshold,
    psf,
    psf_model,
    fwhm,
    eta,
    psf_filter,
    update_psf,
    nb_hio,
    nb_raar,
    nb_er,
    nb_ml,
    nb_run,
    filter_criteria,
    nb_run_keep,
    live_plot,
    plot_axis,
    verbose,
    rebin,
    positivity,
    beta,
    detwin,
    calc_llk,
    zero_mask,
    mask_interp,
    run_phase_retrieval,
    run_pynx_tools,
    energy,
    detector_distance,
    pixel_size_detector,
):
    """
    Get parameters values from widgets and run phase retrieval Possible
    to run phase retrieval via the CLI (with ot without MPI) Or directly in
    python using the operators.

    :param parent_folder: folder in which the raw data files are, and where the
        output will be saved
    :param iobs: 2D/3D observed diffraction data (intensity).
        Assumed to be corrected and following Poisson statistics, will be
        converted to float32. Dimensions should be divisible by 4 and have a
        prime factor decomposition up to 7. Internally, the following special
        values are used:
        * values<=-1e19 are masked. Among those, values in ]-1e38;-1e19] are
            estimated values, stored as -(iobs_est+1)*1e19, which can be used
            to make a loose amplitude projection.
            Values <=-1e38 are masked (no amplitude projection applied), just
            below the minimum float32 value
        * -1e19 < values <= 1 are observed but used as free pixels
            If the mask is not supplied, then it is assumed that the above
            special values are used.
    :param support: initial support in real space (1 = inside support,
        0 = outside)
    :param obj: initial object. If None, it should be initialised later.
    :param mask: mask for the diffraction data (0: valid pixel, >0: masked)
    :param auto_center_resize: if used (command-line keyword) or =True,
        the input data will be centered and cropped  so that the size of the
        array is compatible with the (GPU) FFT library used. If 'roi' is used,
        centering is based on ROI. [default=False]
    :param max_size=256: maximum size for the array used for analysis,
        along all dimensions. The data will be cropped to this value after
        centering. [default: no maximum size]
    :param support_threshold: must be between 0 and 1. Only points with
        object amplitude above a value equal to relative_threshold *
        reference_value are kept in the support.
        reference_value can use the fact that when converged, the square norm
        of the object is equal to the number of recorded photons (normalized
        Fourier Transform). Then: reference_value = sqrt((abs(obj)**2).sum()/
        nb_points_support)
    :param support_smooth_width: smooth the object amplitude using a gaussian
        of this width before calculating new support.
        If this is a scalar, the smooth width is fixed to this value.
        If this is a 3-value tuple (or list or array), i.e. 'smooth_width=2,
        0.5,600', the smooth width will vary with the number of cycles
        recorded in the CDI object (as cdi.cycle), varying exponentially from
        the first to the second value over the number of cycles specified by
        the last value.
        With 'smooth_width=a,b,nb':
        - smooth_width = a * exp(-cdi.cycle/nb*log(b/a)) if cdi.cycle < nb
        - smooth_width = b if cdi.cycle >= nb
    :param support_only_shrink: if True, the support can only shrink
    :param support_post_expand=1: after the new support has been calculated,
        it can be processed using the SupportExpand operator, either one or
        multiple times, in order to 'clean' the support:
        - 'post_expand=1' will expand the support by 1 pixel
        - 'post_expand=-1' will shrink the support by 1 pixel
        - 'post_expand=(-1,1)' will shrink and then expand the support by
            1 pixel
        - 'post_expand=(-2,3)' will shrink and then expand the support by
            respectively 2 and 3 pixels
    :param support_method: either 'max' or 'average' or 'rms' (default), the
        threshold will be relative to either the maximum amplitude in the
        object, or the average or root-mean-square amplitude (computed inside
        support)
    :param support_autocorrelation_threshold: if no support is given, it will
        be estimated from the intensity auto-correlation, with this relative
        threshold. A range can also be given, e.g.
        support_autocorrelation_threshold=0.09,0.11 and the actual threshold
        will be randomly chosen between the min and max.
    :param psf: e.g. True
        whether or not to use the PSF, partial coherence point-spread function,
        estimated with 50 cycles of Richardson-Lucy
    :param psf_model: "lorentzian", "gaussian" or "pseudo-voigt", or None
        to deactivate
    :param psf_filter: either None, "hann" or "tukey": window type to
        filter the PSF update
    :param fwhm: the full-width at half maximum, in pixels
    :param eta: the eta parameter for the pseudo-voigt
    :param update_psf: how often the psf is updated
    :param nb_raar: number of relaxed averaged alternating reflections
        cycles, which the algorithm will use first. During RAAR and HIO, the
        support is updated regularly
    :param nb_hio: number of hybrid input/output cycles, which the
        algorithm will use after RAAR. During RAAR and HIO, the support is
        updated regularly
    :param nb_er: number of error reduction cycles, performed after HIO,
        without support update
    :param nb_ml: number of maximum-likelihood conjugate gradient to
        perform after ER
    :param nb_run: number of times to run the optimization
    :param nb_run_keep: number of best run results to keep, according to
        filter_criteria.
    :param filter_criteria: e.g. "FLLK"
        criteria onto which the best solutions will be chosen
    :param live_plot: a live plot will be displayed every N cycle
    :param plot_axis: for 3D data, the axis along which the cut plane will be
        selected
    :param beta: the beta value for the HIO operator
    :param positivity: True or False
    :param zero_mask: if True, masked pixels (iobs<-1e19) are forced to
        zero, otherwise the calculated complex amplitude is kept with an
        optional scale factor.
        'auto' is only valid if using the command line
    :param mask_interp: e.g. 16,2: interpolate masked pixels from surrounding
        pixels, using an inverse distance weighting. The first number N
        indicates that the pixels used for interpolation range from i-N to i+N
        for pixel i around all dimensions. The second number n that the weight
        is equal to 1/d**n for pixels with at a distance n.
        The interpolated values iobs_m are stored in memory as -1e19*(iobs_m+1)
        so that the algorithm knows these are not trul observations, and are
        applied with a large confidence interval.
    :param detwin: if set (command-line) or if detwin=True (parameters
        file), 10 cycles will be performed at 25% of the total number of
        RAAR or HIO cycles, with a support cut in half to bias towards one
        twin image
    :param calc_llk: interval at which the different Log Likelihood are
        computed
    :param pixel_size_detector: detector pixel size (meters)
    :param wavelength: experiment wavelength (meters)
    :param detector_distance: detector distance (meters)
    """
    # Assign attributes
    params = dict()
    params["parent_folder"] = parent_folder
    params["iobs"] = parent_folder + iobs
    if mask != "":
        params["mask"] = parent_folder + mask
    else:
        params["mask"] = ""
    if support != "":
        params["support"] = parent_folder + support
    else:
        params["support"] = ""
    if obj != "":
        params["obj"] = parent_folder + obj
    else:
        params["obj"] = ""
    params["auto_center_resize"] = auto_center_resize
    params["max_size"] = max_size

    params["support_only_shrink"] = support_only_shrink
    params["support_update_period"] = support_update_period
    params["support_method"] = support_method

    params["psf"] = psf
    params["psf_model"] = psf_model
    params["fwhm"] = fwhm
    params["eta"] = eta
    params["psf_filter"] = None
    params["update_psf"] = update_psf

    params["nb_raar"] = nb_raar
    params["nb_hio"] = nb_hio
    params["nb_er"] = nb_er
    params["nb_ml"] = nb_ml
    params["nb_run"] = nb_run

    params["filter_criteria"] = filter_criteria
    params["nb_run_keep"] = nb_run_keep
    params["live_plot"] = live_plot
    params["verbose"] = verbose
    params["positivity"] = positivity
    params["beta"] = beta
    params["detwin"] = detwin
    params["calc_llk"] = calc_llk
    params["zero_mask"] = zero_mask

    # Extract dict, list and tuple from strings
    params["support_threshold"] = literal_eval(support_threshold)
    params["support_autocorrelation_threshold"] = literal_eval(
        support_autocorrelation_threshold)
    params["support_smooth_width"] = literal_eval(support_smooth_width)
    params["support_post_expand"] = literal_eval(support_post_expand)
    params["rebin"] = literal_eval(rebin)
    params["mask_interp"] = literal_eval(mask_interp)
    # Convert zero_mask parameter
    params["zero_mask"] = {"True": True, "False": False, "auto": False}[
            zero_mask]

    if params["live_plot"] == 0:
        params["live_plot"] = False
    params["plot_axis"] = plot_axis

    params["energy"] = energy # KEv
    params["wavelength"] = 1.2399 * 1e-6 / params["energy"]
    params["detector_distance"] = detector_distance # m
    params["pixel_size_detector"] = np.round(
        pixel_size_detector * 1e-6, 6)

    # Run PR with operators
    if run_phase_retrieval and not run_pynx_tools:
        # Get scan nb
        try:
            scan = int(parent_folder.split("/")[-3].replace("S", ""))
            params["scan"] = scan
            print("Scan n°", scan)
        except:
            print("Could not get scan nb...")
            scan = 0

        print("\tCXI input: Energy = %8.2f eV" % params["energy"])
        print(f"\tCXI input: Wavelength = {params['wavelength']*1e10} A")
        print("\tCXI input: detector distance = %8.2f m" %
              params["detector_distance"])
        print(
            f"\tCXI input: detector pixel size = {params['pixel_size_detector']} m")
        print(
            f"\tLog likelihood is updated every {calc_llk} iterations."
        )

        # Keep a list of the resulting scans
        reconstruction_file_list = []

        try:
            # Initialise the cdi operator
            raw_cdi = initialize_cdi_operator(
                iobs=params["iobs"],
                mask=params["mask"],
                support=params["support"],
                obj=params["obj"],
                rebin=params['rebin'],
                auto_center_resize=params["auto_center_resize"],
                max_size=params["max_size"],
                wavelength=params["wavelength"],
                pixel_size_detector=params["pixel_size_detector"],
                detector_distance=params["detector_distance"],
            )

            # Run phase retrieval for nb_run
            for i in range(nb_run):
                print(
                    "\n###########################################"
                    "#############################################"
                    f"\nRun {i}"
                )

                # Make a copy to gain time
                cdi = raw_cdi.copy()

                # Save input data as cxi
                if i == 0:
                    cxi_filename = "{}/pynx_input_operator_{}.cxi".format(
                        parent_folder,
                        iobs.split("/")[-1].split(".")[0]
                    )

                    save_cdi_operator_as_cxi(
                        cdi_operator=cdi,
                        path_to_cxi=cxi_filename,
                    )

                if i > 4:
                    print("Stopping liveplot to go faster\n")
                    params["live_plot"] = False

                # Change support threshold for supports update
                if isinstance(params["support_threshold"], float):
                    threshold_relative = params["support_threshold"]
                elif isinstance(params["support_threshold"], tuple):
                    threshold_relative = np.random.uniform(
                        params["support_threshold"][0],
                        params["support_threshold"][1]
                    )
                print(f"Threshold: {threshold_relative}")

                # Create support object
                sup = SupportUpdate(
                    threshold_relative=threshold_relative,
                    smooth_width=params["support_smooth_width"],
                    force_shrink=params["support_only_shrink"],
                    method=params["support_method"],
                    post_expand=params["support_post_expand"],
                )

                # Initialize the free pixels for FLLK
                cdi = InitFreePixels() * cdi

                # Interpolate the detector gaps
                if params["live_plot"]:
                    cdi = ShowCDI(plot_axis=params["plot_axis"]) * InterpIobsMask(
                        params["mask_interp"][0],
                        params["mask_interp"][1],
                    ) * cdi
                else:
                    cdi = InterpIobsMask(
                        params["mask_interp"][0],
                        params["mask_interp"][1],
                    ) * cdi
                    print("test3")

                # Initialize the support with autocorrelation, if no
                # support given
                if not support:
                    params["sup_init"] = "autocorrelation"
                    if not params["live_plot"]:
                        cdi = ScaleObj() * AutoCorrelationSupport(
                            threshold=params["support_autocorrelation_threshold"],
                            verbose=True) * cdi

                    else:
                        cdi = ShowCDI(plot_axis=params["plot_axis"]) * ScaleObj() \
                            * AutoCorrelationSupport(
                            threshold=params["support_autocorrelation_threshold"],
                            verbose=True) * cdi
                else:
                    params["sup_init"] = "support"

                # Begin phase retrieval
                try:
                    if psf:
                        if support_update_period == 0:
                            cdi = HIO(
                                beta=params["beta"],
                                calc_llk=params["calc_llk"],
                                show_cdi=params["live_plot"],
                                plot_axis=params["plot_axis"],
                                positivity=params["positivity"],
                                zero_mask=params["zero_mask"],
                            ) ** params["nb_hio"] * cdi
                            cdi = RAAR(
                                beta=params["beta"],
                                calc_llk=params["calc_llk"],
                                show_cdi=params["live_plot"],
                                plot_axis=params["plot_axis"],
                                positivity=params["positivity"],
                                zero_mask=params["zero_mask"],
                            ) ** (params["nb_raar"] // 2) * cdi

                            # PSF is introduced at 66% of HIO and RAAR
                            if psf_model != "pseudo-voigt":
                                cdi = InitPSF(
                                    model=params["psf_model"],
                                    fwhm=params["fwhm"],
                                    filter=None,  # None for now bc experimental
                                ) * cdi

                            elif psf_model == "pseudo-voigt":
                                cdi = InitPSF(
                                    model=params["psf_model"],
                                    fwhm=params["fwhm"],
                                    eta=params["eta"],
                                    filter=None,
                                ) * cdi

                            cdi = RAAR(
                                beta=params["beta"],
                                calc_llk=params["calc_llk"],
                                show_cdi=params["live_plot"],
                                update_psf=params["update_psf"],
                                plot_axis=params["plot_axis"],
                                positivity=params["positivity"],
                                psf_filter=None,
                                zero_mask=params["zero_mask"],
                            ) ** (nb_raar // 2) * cdi
                            cdi = ER(
                                calc_llk=params["calc_llk"],
                                show_cdi=params["live_plot"],
                                update_psf=params["update_psf"],
                                plot_axis=params["plot_axis"],
                                positivity=params["positivity"],
                                psf_filter=None,
                                zero_mask=params["zero_mask"],
                            ) ** nb_er * cdi

                        else:
                            hio_power = params["nb_hio"] \
                                // params["support_update_period"]
                            raar_power = (
                                params["nb_raar"] // 2) \
                                // params["support_update_period"]
                            er_power = params["nb_er"] \
                                // params["support_update_period"]

                            cdi = (sup * HIO(
                                beta=params["beta"],
                                calc_llk=params["calc_llk"],
                                show_cdi=params["live_plot"],
                                plot_axis=params["plot_axis"],
                                positivity=params["positivity"],
                                psf_filter=None,
                                zero_mask=params["zero_mask"],
                            )**params["support_update_period"]
                            ) ** hio_power * cdi
                            cdi = (sup * RAAR(
                                beta=params["beta"],
                                calc_llk=params["calc_llk"],
                                show_cdi=params["live_plot"],
                                plot_axis=params["plot_axis"],
                                positivity=params["positivity"],
                                psf_filter=None,
                                zero_mask=params["zero_mask"],
                            )**params["support_update_period"]
                            ) ** raar_power * cdi

                            # PSF is introduced after half the HIO cycles
                            if psf_model != "pseudo-voigt":
                                cdi = InitPSF(
                                    model=params["psf_model"],
                                    fwhm=params["fwhm"],
                                    filter=None,
                                ) * cdi

                            elif psf_model == "pseudo-voigt":
                                cdi = InitPSF(
                                    model=params["psf_model"],
                                    fwhm=params["fwhm"],
                                    eta=params["eta"],
                                    filter=None,
                                ) * cdi

                            cdi = (sup * RAAR(
                                beta=params["beta"],
                                calc_llk=params["calc_llk"],
                                show_cdi=params["live_plot"],
                                update_psf=params["update_psf"],
                                plot_axis=params["plot_axis"],
                                positivity=params["positivity"],
                                psf_filter=None,
                                zero_mask=params["zero_mask"],
                            )**params["support_update_period"]
                            ) ** raar_power * cdi
                            cdi = (sup * ER(
                                calc_llk=params["calc_llk"],
                                show_cdi=params["live_plot"],
                                update_psf=params["update_psf"],
                                plot_axis=params["plot_axis"],
                                positivity=params["positivity"],
                                psf_filter=None,
                                zero_mask=params["zero_mask"],
                            )**params["support_update_period"]
                            ) ** er_power * cdi

                    if not psf:
                        if support_update_period == 0:
                            cdi = HIO(
                                beta=params["beta"],
                                calc_llk=params["calc_llk"],
                                show_cdi=params["live_plot"],
                                plot_axis=params["plot_axis"],
                                positivity=params["positivity"],
                                zero_mask=params["zero_mask"],
                            ) ** params["nb_hio"] * cdi
                            cdi = RAAR(
                                beta=params["beta"],
                                calc_llk=params["calc_llk"],
                                show_cdi=params["live_plot"],
                                plot_axis=params["plot_axis"],
                                positivity=params["positivity"],
                                zero_mask=params["zero_mask"],
                            ) ** params["nb_raar"] * cdi
                            cdi = ER(
                                calc_llk=params["calc_llk"],
                                show_cdi=params["live_plot"],
                                plot_axis=params["plot_axis"],
                                positivity=params["positivity"],
                                zero_mask=params["zero_mask"],
                            ) ** params["nb_er"] * cdi

                        else:
                            hio_power = params["nb_hio"] \
                                // params["support_update_period"]
                            raar_power = params["nb_raar"] \
                                // params["support_update_period"]
                            er_power = params["nb_er"] \
                                // params["support_update_period"]

                            cdi = (sup * HIO(
                                beta=params["beta"],
                                calc_llk=params["calc_llk"],
                                show_cdi=params["live_plot"],
                                plot_axis=params["plot_axis"],
                                positivity=params["positivity"],
                                zero_mask=params["zero_mask"],
                            )**params["support_update_period"]
                            ) ** hio_power * cdi
                            cdi = (sup * RAAR(
                                beta=params["beta"],
                                calc_llk=params["calc_llk"],
                                show_cdi=params["live_plot"],
                                plot_axis=params["plot_axis"],
                                positivity=params["positivity"],
                                zero_mask=params["zero_mask"],
                            )**params["support_update_period"]
                            ) ** raar_power * cdi
                            cdi = (sup * ER(
                                calc_llk=params["calc_llk"],
                                show_cdi=params["live_plot"],
                                plot_axis=params["plot_axis"],
                                positivity=params["positivity"],
                                zero_mask=params["zero_mask"],
                            )**params["support_update_period"]
                            ) ** er_power * cdi

                    fn = "{}/result_scan_{}_run_{}_FLLK_{:.4}_support_threshold_{:.4}_shape_{}_{}_{}_{}.cxi".format(
                        params["parent_folder"],
                        params["scan"],
                        i,
                        cdi.get_llk(normalized=True)[
                            3],  # check pynx for this
                        threshold_relative,
                        cdi.iobs.shape[0],
                        cdi.iobs.shape[1],
                        cdi.iobs.shape[2],
                        params["sup_init"],
                    )

                    reconstruction_file_list.append(fn)
                    cdi.save_obj_cxi(fn)
                    print(
                        f"\nSaved as {fn}."
                        "\n###########################################"
                        "#############################################"
                    )

                except SupportTooLarge:
                    print(
                        "Threshold value probably too low, support too large too continue")

            # If filter, filter data
            if filter_criteria:
                filter_reconstructions(
                    folder=params["parent_folder"],
                    nb_run=None,  # Will take the amount of cxi files found
                    nb_run_keep=params["nb_run_keep"],
                    filter_criteria=params["filter_criteria"]
                )

        except KeyboardInterrupt:
            clear_output(True)
            print(
                "Phase retrieval stopped by user, `.cxi` file list below."
            )

        cxi_files_list = list_files(
            folder=params["parent_folder"],
            glob_pattern="*.cxi",
            verbose=True,
        )

    # Modes decomposition and solution filtering
    if run_pynx_tools and not run_phase_retrieval:
        if run_pynx_tools == "modes":
            run_modes_decomposition(
                folder=params["parent_folder"],
                path_scripts="/home/esrf/simonne/.conda/envs/p9.cdiutils/bin/",
            )

        elif run_pynx_tools == "filter":
            filter_reconstructions(
                folder=params["parent_folder"],
                nb_run=None,  # Will take the amount of cxi files found
                nb_run_keep=params["nb_run_keep"],
                filter_criteria=params["filter_criteria"]
            )

    # Clean output
    if not run_phase_retrieval and not run_pynx_tools:
        print("Cleared output.")
        clear_output(True)

        cxi_files_list = list_files(
            folder=params["parent_folder"],
            glob_pattern="*.cxi",
            verbose=True,
        )


def initialize_cdi_operator(
    iobs: str,
    mask: Optional[str] = None,
    support: Optional[str] = None,
    obj: Optional[str] = None,
    rebin: Tuple[int, int, int] = (1, 1, 1),
    auto_center_resize: bool = False,
    max_size: Optional[int] = None,
    wavelength: Optional[float] = None,
    pixel_size_detector: Optional[float] = None,
    detector_distance: Optional[float] = None,
) -> Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """
    Initialize the CDI operator by processing the possible inputs:
        - iobs
        - mask
        - support
        - obj
    Will also crop and center the data if specified.

    :param iobs: path to npz or npy that stores the intensity observations data
    :param mask: path to npz or npy that stores the mask data
    :param support: path to npz or npy that stores the support data
    :param obj: path to npz or npy that stores the object data
    :param rebin: tuple, applied to all the arrays, e.g. (1, 1, 1)
    :param auto_center_resize: flag to automatically crop and center the data
    :param max_size: maximum size of the cropped data, optional
    :param wavelength: wavelength of the data, optional
    :param pixel_size_detector: pixel size of the detector, optional
    :param detector_distance: detector distance, optional

    :return: cdi operator or None if initialization fails
    """
    if os.path.isfile(str(iobs)):
        if iobs.endswith(".npy"):
            iobs = np.load(iobs)
            print("\tCXI input: loading data")
        elif iobs.endswith(".npz"):
            try:
                iobs = np.load(iobs)["data"]
                print("\tCXI input: loading data")
            except KeyError:
                print("\t\"data\" key does not exist.")
                return None
        if rebin != (1, 1, 1):
            iobs = bin_data(iobs, rebin)
            print("\tBinned data.")

        iobs = fftshift(iobs)
    else:
        iobs = None
        print("At least iobs must exist.")
        return None

    if os.path.isfile(str(mask)):
        if mask.endswith(".npy"):
            mask = np.load(mask).astype(np.int8)
            nb = mask.sum()
            mask_percentage = nb * 100 / mask.size
            print(
                f"\tCXI input: loading mask, "
                f"with {nb} pixels masked ({mask_percentage:0.3f}%)"
            )
        elif mask.endswith(".npz"):
            try:
                mask = np.load(mask)[
                    "mask"].astype(np.int8)
                nb = mask.sum()
                mask_percentage = nb * 100 / mask.size
                print(
                    f"\tCXI input: loading mask, "
                    f"with {nb} pixels masked ({mask_percentage:0.3f}%)"
                )
            except KeyError:
                print("\t\"mask\" key does not exist.")

        if rebin != (1, 1, 1):
            mask = bin_data(mask, rebin)
            print("\tBinned mask.")

        mask = fftshift(mask)

    else:
        mask = None

    if os.path.isfile(str(support)):
        if support.endswith(".npy"):
            support = np.load(support)
            print("\tCXI input: loading support")
        elif support.endswith(".npz"):
            try:
                support = np.load(support)["data"]
                print("\tCXI input: loading support")
            except (FileNotFoundError, ValueError):
                print("\tFile not supported or does not exist.")
            except KeyError:
                print("\t\"data\" key does not exist.")
                try:
                    support = np.load(support)["support"]
                    print("\tCXI input: loading support")
                except KeyError:
                    print("\t\"support\" key does not exist.")
                    try:
                        support = np.load(support)["obj"]
                        print("\tCXI input: loading support")
                    except KeyError:
                        print(
                            "\t\"obj\" key does not exist."
                            "\t--> Could not load support array."
                        )

        if rebin != (1, 1, 1):
            support = bin_data(support, rebin)
            print("\tBinned support.")

        support = fftshift(support)

    else:
        support = None

    if os.path.isfile(str(obj)):
        if obj.endswith(".npy"):
            obj = np.load(obj)
            print("\tCXI input: loading object")
        elif obj.endswith(".npz"):
            try:
                obj = np.load(obj)["data"]
                print("\tCXI input: loading object")
            except KeyError:
                print("\t\"data\" key does not exist.")

        if rebin != (1, 1, 1):
            obj = bin_data(obj, rebin)
            print("\tBinned obj.")

        obj = fftshift(obj)

    else:
        obj = None

    # Center and crop data
    if auto_center_resize:
        if iobs.ndim == 3:
            nz0, ny0, nx0 = iobs.shape

            # Find center of mass
            z0, y0, x0 = center_of_mass(iobs)
            print("Center of mass at:", z0, y0, x0)
            iz0, iy0, ix0 = int(round(z0)), int(round(y0)), int(round(x0))

            # Max symmetrical box around center of mass
            nx = 2 * min(ix0, nx0 - ix0)
            ny = 2 * min(iy0, ny0 - iy0)
            nz = 2 * min(iz0, nz0 - iz0)

            if max_size is not None:
                nx = min(nx, max_size)
                ny = min(ny, max_size)
                nz = min(nz, max_size)

            # Crop data to fulfill FFT size requirements
            nz1, ny1, nx1 = smaller_primes(
                (nz, ny, nx),
                maxprime=7,
                required_dividers=(2,)
            )

            print(
                f"Centering & reshaping data: ({nz0}, {ny0}, {nx0}) -> "
                f"({nz1}, {ny1}, {nx1})"
            )
            iobs = iobs[
                iz0 - nz1 // 2:iz0 + nz1 // 2,
                iy0 - ny1 // 2:iy0 + ny1 // 2,
                ix0 - nx1 // 2:ix0 + nx1 // 2]
            if mask is not None:
                mask = mask[
                    iz0 - nz1 // 2:iz0 + nz1 // 2,
                    iy0 - ny1 // 2:iy0 + ny1 // 2,
                    ix0 - nx1 // 2:ix0 + nx1 // 2]
                print(
                    f"Centering & reshaping mask: ({nz0}, {ny0}, {nx0}) -> "
                    f"({nz1}, {ny1}, {nx1})"
                )

        else:
            ny0, nx0 = iobs.shape

            # Find center of mass
            y0, x0 = center_of_mass(iobs)
            iy0, ix0 = int(round(y0)), int(round(x0))
            print("Center of mass (rounded) at:", iy0, ix0)

            # Max symmetrical box around center of mass
            nx = 2 * min(ix0, nx0 - ix0)
            ny = 2 * min(iy0, ny0 - iy0)
            if max_size is not None:
                nx = min(nx, max_size)
                ny = min(ny, max_size)
                nz = min(nz, max_size)

            # Crop data to fulfill FFT size requirements
            ny1, nx1 = smaller_primes(
                (ny, nx), maxprime=7, required_dividers=(2,))

            print(
                f"Centering & reshaping data: ({ny0}, {nx0}) -> ({ny1}, {nx1})"
            )
            iobs = iobs[iy0 - ny1 // 2:iy0 + ny1 //
                        2, ix0 - nx1 // 2:ix0 + nx1 // 2]

            if mask is not None:
                mask = mask[iy0 - ny1 // 2:iy0 + ny1 //
                            2, ix0 - nx1 // 2:ix0 + nx1 // 2]

    # Create cdi object with data and mask, load the main parameters
    cdi = CDI(
        iobs,
        support=support,
        obj=obj,
        mask=mask,
        wavelength=wavelength,
        pixel_size_detector=pixel_size_detector,
        detector_distance=detector_distance,
    )

    return cdi


def save_cdi_operator_as_cxi(
    cdi_operator,
    path_to_cxi,
):
    """
    We need to create a dictionnary with the parameters to save in the
    cxi file.

    :param cdi_operator: cdi object
     created with PyNX
    :param path_to_cxi: path to future cxi data
     Below are parameters that are saved in the cxi file
        - filename: the file name to save the data to
        - iobs: the observed intensity
        - wavelength: the wavelength of the experiment (in meters)
        - detector_distance: the detector distance (in meters)
        - pixel_size_detector: the pixel size of the detector (in meters)
        - mask: the mask indicating valid (=0) and bad pixels (>0)
        - sample_name: optional, the sample name
        - experiment_id: the string identifying the experiment, e.g.:
          'HC1234: Siemens star calibration tests'
        - instrument: the string identifying the instrument, e.g.:
         'ESRF id10'
        - iobs_is_fft_shifted: if true, input iobs (and mask if any)
        have their origin in (0,0[,0]) and will be shifted back to
        centered-versions before being saved.
        - process_parameters: a dictionary of parameters which will
          be saved as a NXcollection

    :return: Nothing, a CXI file is created.
    """
    print(
        "\nSaving phase retrieval parameters selected "
        "in the PyNX tab in the cxi file ..."
    )
    cdi_operator.save_data_cxi(
        filename=path_to_cxi,
        process_parameters=params,
    )


def list_files(
    folder: str,
    glob_pattern: str = "*FLLK*.cxi",
    verbose: bool = False
) -> List[str]:
    """List all files in a specified folder that match a specified glob pattern and sort by creation time.

    Args:
        folder (str): The path to the folder where the files are located.
        glob_pattern (str, optional): A string that specifies the pattern of the filenames to match. Default is "*FLLK*.cxi".
        verbose (bool, optional): If set to True, the function will print the filenames and their creation timestamps to the console. Default is False.

    Returns:
        list: A list of file paths that match the specified pattern and are sorted by creation time (most recent first).

    Example:
        file_list = list_files("/path/to/folder", verbose=True)
    """
    file_list = sorted(
        glob.glob(folder + "/" + glob_pattern),
        key=os.path.getmtime,
        reverse=True,
    )

    if verbose:
        print(
            "################################################"
            "################################################"
        )
        for f in file_list:
            file_timestamp = datetime.fromtimestamp(
                os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M:%S')
            print(
                f"\nFile: {os.path.basename(f)}"
                f"\n\tCreated: {file_timestamp}"
            )
        print(
            "################################################"
            "################################################"
        )

    return file_list


def filter_reconstructions(
    folder: str,
    nb_run_keep: int,
    nb_run: Optional[int] = None,
    filter_criteria: str = "FLLK"
) -> None:
    """Filter the phase retrieval output based on a specified parameter.

    The function filters phase retrieval output based on specified criteria,
    such as "FLLK" or "standard deviation". The user can run multiple
    reconstructions and the function will automatically keep the "best"
    ones according to the specified criteria. If "standard_deviation" and
    "FLLK" are specified as the criteria, half of the `nb_run_keep` files will
    be filtered based on the first criteria and the remaining files will be
    filtered based on the second criteria.

    The parameters are specified in the phase retrieval tab.

    Args:
        folder (str): Parent folder to cxi files
        nb_run_keep (int): The number of the best run results to keep in the end
            according to the `filter_criteria`.
        nb_run (Optional[int], optional): The number of times to run the optimization.
            If `None`, it is equal to the number of files detected. Defaults to `None`.
        filter_criteria (str, optional): The criteria based on which the best solutions
            will be chosen. Possible values are "standard_deviation", "FLLK",
            "standard_deviation_FLLK", "FLLK_standard_deviation". Defaults to "FLLK".

    Returns:
        None
    """
    def filter_by_std(
        cxi_files: List[str],
        nb_run_keep: int
    ) -> None:
        """
        Use the standard deviation of the reconstructed object as filtering criteria.

        The function computes the standard deviation of the object modulus for each of the input `cxi_files`, and
        removes the `cxi_files` with the highest standard deviations until only `nb_run_keep` remain. The files are
        removed by removing the corresponding file on disk.

        Parameters
        ----------
        cxi_files : List[str]
            A list of strings representing the paths to the `cxi` files to be filtered.
        nb_run_keep : int
            The number of `cxi` files to keep after filtering. The files with the lowest standard deviation will be kept.

        Returns
        -------
        None

        """
        filtering_criteria_value = {}

        print(
            "\n###################"
            "#####################"
            "#####################"
            "#####################"
        )
        print("Computing standard deviation of object modulus for scans:")
        for filename in cxi_files:
            print(f"\t{os.path.basename(filename)}")
            with h5py.File(filename, "r") as f:
                data = f["entry_1/image_1/data"][...]
                amp = np.abs(data)
                # Skip values near 0
                meaningful_data = amp[amp > 0.05 * amp.max()]
                filtering_criteria_value[filename] = np.std(
                    meaningful_data
                )

        # Sort files
        sorted_dict = sorted(
            filtering_criteria_value.items(),
            key=operator_lib.itemgetter(1)
        )

        # Remove files
        print("\nRemoving scans:")
        for filename, filtering_criteria_value in sorted_dict[nb_run_keep:]:
            print(f"\t{os.path.basename(filename)}")
            os.remove(filename)
        print(
            "#####################"
            "#####################"
            "#####################"
            "###################\n"
        )

    def filter_by_FLLK(
        cxi_files: List[str],
        nb_run_keep: int
    ) -> None:
        """Filter `cxi_files` using the free log-likelihood (FLLK) values.

        The `cxi_files` are filtered based on the FLLK values calculated from
        the reconstructed object using poisson statistics. The files with the
        lowest FLLK values are kept.

        Args:
            cxi_files: A list of paths to CXI files.
            nb_run_keep: The number of files to keep, based on their FLLK values.

        Returns:
            None. The function modifies the list of `cxi_files` in place by
            removing some of the files based on the FLLK values.
        """
        # Keep filtering criteria of reconstruction modules in dictionnary
        filtering_criteria_value = {}

        print(
            "\n###################"
            "#####################"
            "#####################"
            "#####################"
        )
        print("Extracting FLLK value (poisson statistics) for scans:")
        for filename in cxi_files:
            print(f"\t{os.path.basename(filename)}")
            with h5py.File(filename, "r") as f:
                fllk = f["entry_1/image_1/process_1/results/free_llk_poisson"][...]
                filtering_criteria_value[filename] = fllk

        # Sort files
        sorted_dict = sorted(
            filtering_criteria_value.items(),
            key=operator_lib.itemgetter(1)
        )

        # Remove files
        print("\nRemoving scans:")
        for filename, filtering_criteria_value in sorted_dict[nb_run_keep:]:
            print(f"\t{os.path.basename(filename)}")
            os.remove(filename)
        print(
            "#####################"
            "#####################"
            "#####################"
            "###################\n"
        )

    # Main function supporting different cases
    try:
        glob_pattern = "*FLLK*.cxi"
        print(
            "\n########################################"
            "##########################################"
        )
        print("Iterating on files matching:")
        print(f"\t{folder}/{glob_pattern}")
        cxi_files = list_files(
            folder=folder,
            glob_pattern=glob_pattern,
        )
        print(
            "##########################################"
            "########################################\n"
        )

        if cxi_files == []:
            print(
                f"No match for {folder}/*FLLK*.cxi"
                f"\nTrying with {folder}/*LLK*.cxi"
            )
            glob_pattern = "*LLK*.cxi"
            cxi_files = list_files(
                folder=folder,
                glob_pattern=glob_pattern,
            )

        # only standard_deviation
        if filter_criteria == "standard_deviation":
            filter_by_std(cxi_files, nb_run_keep)

        # only FLLK
        elif filter_criteria == "FLLK":
            filter_by_FLLK(cxi_files, nb_run_keep)

        # standard_deviation then FLLK
        elif filter_criteria == "standard_deviation_FLLK":
            if nb_run is None:
                nb_run = len(cxi_files)

            filter_by_std(cxi_files, nb_run_keep +
                          (nb_run - nb_run_keep) // 2)

            print("Iterating on remaining files.")

            cxi_files = list_files(
                folder=folder,
                glob_pattern=glob_pattern,
            )

            if cxi_files == []:
                print(
                    f"No {glob_pattern} files remaining in {folder}")
            else:
                filter_by_FLLK(cxi_files, nb_run_keep)

        # FLLK then standard_deviation
        elif filter_criteria == "FLLK_standard_deviation":
            if nb_run is None:
                nb_run = len(cxi_files)

            filter_by_FLLK(cxi_files, nb_run_keep +
                           (nb_run - nb_run_keep) // 2)

            print("Iterating on remaining files.")

            cxi_files = list_files(
                folder=folder,
                glob_pattern=glob_pattern,
            )

            if cxi_files == []:
                print(
                    f"No {glob_pattern} files remaining in {folder}")
            else:
                filter_by_std(cxi_files, nb_run_keep)

        else:
            print("No filtering")
    except KeyboardInterrupt:
        print("File filtering stopped by user ...")


def run_modes_decomposition(
    path_scripts: str,
    folder: str
) -> None:
    """
    Decomposes several phase retrieval solutions into modes, saves only
    the first mode to save space.

    All files corresponding to *FLLK* pattern are loaded, if no files are
    loaded, trying with *LLK* pattern.

    Args:
    - path_scripts (str): absolute path to the script containing the folder
    - folder (str): path to the folder in which the reconstructions are stored

    Returns:
    None

    Raises:
    - KeyboardInterrupt: if the decomposition into modes is stopped by the user

    Example:
    >>> run_modes_decomposition("/path/to/scripts", "/path/to/folder")
    """
    glob_pattern = "*FLLK*.cxi"
    cxi_files_list = list_files(
        folder=folder,
        glob_pattern=glob_pattern,
    )

    if cxi_files_list == []:
        glob_pattern = "*LLK*.cxi"
        cxi_files_list = list_files(
            folder=folder,
            glob_pattern=glob_pattern,
        )
        if cxi_files_list == []:
            print(
                "Could not find any files matching the *LLK*.cxi* "
                "or *FLLK*.cxi patterns."
            )
            glob_pattern = False

    if isinstance(glob_pattern, str):
        print(
            "\n###########################################"
            "#############################################"
            f"\nUsing {path_scripts}/pynx-cdi-analysis"
            f"\nUsing {folder}/{glob_pattern} files."
            f"\nRunning: $ pynx-cdi-analysis {glob_pattern} -- modes 1"
            f"\nOutput in {folder}/modes_gui.h5"
            "\n###########################################"
            "#############################################"
        )
    try:
        os.system(
            "{}/pynx-cdi-analysis {}/{} --modes 1 --modes_output {}/modes_gui.h5".format(
                quote(path_scripts),
                quote(folder),
                glob_pattern,
                quote(folder),
            )
        )
    except KeyboardInterrupt:
        print("Decomposition into modes stopped by user...")