import numpy as np
import glob
import os
from datetime import datetime
from scipy.fft import fftshift
from IPython.display import clear_output, display
from ast import literal_eval
import ipywidgets as widgets
from ipywidgets import interactive, Layout, Button

from cdiutils.process.phaser import PyNXImportError

try:
    from pynx.cdi import (
        CDI,
        AutoCorrelationSupport,
        ScaleObj,
        SupportUpdate,
        HIO,
        # DetwinHIO,
        RAAR,
        # DetwinRAAR,
        ER,
        # FourierApplyAmplitude,
        SupportTooLarge,
        SupportTooSmall,
        InitPSF,
        InterpIobsMask,
        InitFreePixels,
        ShowCDI,
    )
    from pynx.utils.array import rebin

    IS_PYNX_AVAILABLE = True

except ImportError:
    IS_PYNX_AVAILABLE = False


class PhaseRetrievalGUI(widgets.VBox):
    """
    A widget-based graphical user interface (GUI) for interactive phase
    retrieval using PyNX.

    This class provides a comprehensive set of widgets to configure and
    execute phase retrieval algorithms in a Jupyter Notebook environment.
    It allows users to set parameters, select input files, and run
    iterative algorithms for coherent diffraction imaging (CDI). The GUI
    is designed to facilitate an interactive workflow for phase retrieval
    tasks.

    Attributes:
        header: A brief header describing the purpose of the tab.
        box_style: The CSS style applied to the widget container.
        parent_folder: Dropdown to select the parent folder containing
            input files.
        iobs: Dropdown to select the observed intensity dataset file.
        mask: Dropdown to select the mask file.
        support: Dropdown to select the support file.
        obj: Dropdown to select the object file.
        support_threshold: Input field to specify the support threshold.
        support_only_shrink: Checkbox to enable or disable support
            shrinking.
        support_update_period: Input field to specify the period for
            support updates.
        support_smooth_width: Input field to specify the smoothing width
            for support updates.
        support_post_expand: Input field to specify post-expansion
            parameters for the support.
        support_method: Dropdown to select the method for support updates
            (e.g., "max", "average", "rms").
        psf: Checkbox to enable or disable the use of a point spread
            function (PSF).
        psf_model: Dropdown to select the PSF model (e.g., "gaussian",
            "lorentzian", "pseudo-voigt").
        fwhm: Input field to specify the full-width at half maximum
            (FWHM) for the PSF.
        eta: Input field to specify the eta parameter for the pseudo-voigt
            PSF model.
        psf_filter: Dropdown to select the PSF filter type (e.g., "None",
            "hann", "tukey").
        update_psf: Input field to specify the frequency of PSF updates.
        nb_hio: Input field to specify the number of Hybrid Input-Output
            (HIO) iterations.
        nb_raar: Input field to specify the number of Relaxed Averaged
            Alternating Reflections (RAAR) iterations.
        nb_er: Input field to specify the number of Error Reduction (ER)
            iterations.
        nb_ml: Input field to specify the number of Maximum Likelihood
            (ML) iterations.
        nb_run: Input field to specify the number of phase retrieval
            runs.
        live_plot: Input field to specify the frequency of live plotting
            during phase retrieval.
        plot_axis: Dropdown to select the axis used for live plots.
        verbose: Input field to specify the verbosity level of the output.
        binning: Input field to specify rebinning parameters for the input
            data.
        positivity: Checkbox to enable or disable positivity constraints.
        beta: Input field to specify the beta parameter for HIO and RAAR
            algorithms.
        detwin: Checkbox to enable or disable detwinning.
        calc_llk: Input field to specify the interval for log-likelihood
            calculations.
        zero_mask: Dropdown to specify whether to force mask pixels to
    """

    def __init__(
            self,
            pipeline_instance,
            box_style: str = None,
            work_dir: str = None,
            search_pattern: str = None
    ):
        """
        Initialize the PhaseRetrievalGUI class.

        This method sets up the graphical user interface (GUI) for phase
        retrieval by defining and initializing various widgets.
        These widgets allow users to configure parameters, select input
        files, and control the execution of phase retrieval algorithms.
        The GUI is designed to work in a Jupyter Notebook environment.

        Args:
            pipeline_instance: An instance of BcdiPipeline to handle the
                handle the analysis of phase retrieval results.
            box_style: The CSS style applied to the widget container.
                Default is an empty string.
            work_dir: The working directory where input files are located.
                If not provided, the current working directory is used.
            search_pattern: A string defining the search pattern for
                input files. If not provided, a default pattern for
                `.cxi` files is used (`*run*.cxi`).

        Attributes:
            header: A brief header describing the purpose of the tab.
            box_style: The CSS style applied to the widget container.
            parent_folder: Dropdown to select the parent folder containing
                input files.
            iobs: Dropdown to select the observed intensity dataset file.
            mask: Dropdown to select the mask file.
            support: Dropdown to select the support file.
            obj: Dropdown to select the object file.
            support_threshold: Input field to specify the support threshold.
            support_only_shrink: Checkbox to enable or disable support
                shrinking.
            support_update_period: Input field to specify the period for
                support updates.
            support_smooth_width: Input field to specify the smoothing width
                for support updates.
            support_post_expand: Input field to specify post-expansion
                parameters for the support.
            support_method: Dropdown to select the method for support updates
                (e.g., "max", "average", "rms").
            psf: Checkbox to enable or disable the use of a point spread
                function (PSF).
            psf_model: Dropdown to select the PSF model (e.g., "gaussian",
                "lorentzian", "pseudo-voigt").
            fwhm: Input field to specify the full-width at half maximum
                (FWHM) for the PSF.
            eta: Input field to specify the eta parameter for the
                pseudo-voigt PSF model.
            psf_filter: Dropdown to select the PSF filter type (e.g.,
                "None", "hann", "tukey").
            update_psf: Input field to specify the frequency of PSF updates.
            nb_hio: Input field to specify the number of Hybrid Input-Output
                (HIO) iterations.
            nb_raar: Input field to specify the number of Relaxed Averaged
                Alternating Reflections (RAAR) iterations.
            nb_er: Input field to specify the number of Error Reduction (ER)
                iterations.
            nb_ml: Input field to specify the number of Maximum Likelihood
                (ML) iterations.
            nb_run: Input field to specify the number of phase retrieval
                runs.
            live_plot: Input field to specify the frequency of live plotting
                during phase retrieval.
            plot_axis: Dropdown to select the axis used for live plots.
            verbose: Input field to specify the verbosity level of the
                output.
            binning: Input field to specify rebinning parameters for the
                input data.
            positivity: Checkbox to enable or disable positivity constraints.
            beta: Input field to specify the beta parameter for HIO and RAAR
                algorithms.
            detwin: Checkbox to enable or disable detwinning.
            calc_llk: Input field to specify the interval for log-likelihood
                calculations.
            zero_mask: Dropdown to specify whether to force mask pixels to
                zero.
            mask_interp: Input field to specify interpolation parameters for
                the mask.
            run_phase_retrieval: Toggle buttons to start or stop the phase
                retrieval process.
            run_pynx_tools: Toggle buttons to run additional PyNX tools
                (e.g., modes decomposition, filtering).
            filter_criteria: Dropdown to select the criteria for filtering
                reconstruction results.

        Notes:
            - The method also assigns event handlers to widgets to
              dynamically update the GUI based on user interactions.
            - The `children` attribute is populated with all the widgets,
              defining the layout of the GUI.
        """
        super().__init__()

        # brief header describing the tab
        self.header = "Phase retrieval"

        if box_style is None:
            box_style = ""
        self.box_style = box_style

        # define the pipeline instance
        self.pipeline = pipeline_instance

        # define global search pattern for cxi files
        self.search_pattern = search_pattern
        if self.search_pattern is None:
            self.search_pattern = "*run*.cxi"

        # define future attributes
        self.modes = None
        self.mode_weights = None

        if work_dir is None:
            work_dir = os.getcwd()

        # define widgets
        self.unused_label_data = widgets.HTML(
            value="<p style='font-weight: bold;font-size:1.2em'>\
            Data files",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="90%", height="35px"),
        )

        self.parent_folder = widgets.Dropdown(
            options=sorted([x[0] + "/" for x in os.walk(work_dir)]),
            value=work_dir + "/",
            placeholder=work_dir + "/",
            description="Parent folder:",
            continuous_update=False,
            layout=widgets.Layout(width="90%"),
            style={"description_width": "initial"},
        )

        self.iobs = widgets.Dropdown(
            options=[""]
            + sorted(
                [os.path.basename(f) for f in glob.glob(work_dir + "*.npz")],
                key=os.path.getmtime,
            ),
            description="Dataset",
            layout=widgets.Layout(width="90%"),
            style={"description_width": "initial"},
        )

        self.mask = widgets.Dropdown(
            options=[""]
            + sorted(
                [os.path.basename(f) for f in glob.glob(work_dir + "*.npz")],
                key=os.path.getmtime,
            ),
            description="Mask",
            layout=widgets.Layout(width="90%"),
            style={"description_width": "initial"},
        )

        self.support = widgets.Dropdown(
            options=[""]
            + sorted(
                [os.path.basename(f) for f in glob.glob(work_dir + "*.npz")],
                key=os.path.getmtime,
            ),
            value="",
            description="Support",
            layout=widgets.Layout(width="90%"),
            style={"description_width": "initial"},
        )

        self.obj = widgets.Dropdown(
            options=[""]
            + sorted(
                [os.path.basename(f) for f in glob.glob(work_dir + "*.npz")],
                key=os.path.getmtime,
            ),
            value="",
            description="Object",
            layout=widgets.Layout(width="90%"),
            style={"description_width": "initial"},
        )

        self.unused_label_support = widgets.HTML(
            value="<p style='font-weight: bold;font-size:1.2em'>\
            Support parameters",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="90%", height="35px"),
        )

        self.support_threshold = widgets.Text(
            value="(0.23, 0.30)",
            placeholder="(0.23, 0.30)",
            description="Support threshold",
            layout=widgets.Layout(height="50px", width="30%"),
            continuous_update=False,
            style={"description_width": "initial"},
        )

        self.support_only_shrink = widgets.Checkbox(
            value=False,
            description="Support only shrink",
            continuous_update=False,
            indent=False,
            layout=widgets.Layout(height="50px", width="35%"),
            icon="check",
        )

        self.support_update_period = widgets.BoundedIntText(
            value=20,
            max=500,
            step=5,
            layout=widgets.Layout(height="50px", width="30%"),
            continuous_update=False,
            description="Support update period:",
            readout=True,
            style={"description_width": "initial"},
        )

        self.support_smooth_width = widgets.Text(
            value="(2, 1, 600)",
            placeholder="(2, 1, 600)",
            description="Support smooth width",
            layout=widgets.Layout(height="50px", width="30%"),
            continuous_update=False,
            style={"description_width": "initial"},
        )

        self.support_post_expand = widgets.Text(
            value="(1, -2, 1)",
            placeholder="(1, -2, 1)",
            description="Support post expand",
            layout=widgets.Layout(width="30%"),
            continuous_update=False,
            style={"description_width": "initial"},
        )

        self.support_method = widgets.Dropdown(
            options=["max", "average", "rms"],
            value="rms",
            description="Support method",
            layout=widgets.Layout(height="25px", width="30%"),
            continuous_update=False,
            style={"description_width": "initial"},
        )

        self.support_autocorrelation_threshold = widgets.Text(
            value="(0.10)",
            placeholder="(0.10)",
            description="Support autocorrelation threshold",
            layout=widgets.Layout(height="50px", width="40%"),
            continuous_update=False,
            style={"description_width": "initial"},
        )

        self.unused_label_psf = widgets.HTML(
            value="<p style='font-weight: bold;font-size:1.2em'>\
            Point spread function parameters",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="90%", height="35px"),
        )

        self.psf = widgets.Checkbox(
            value=True,
            description="Use point spread function",
            continuous_update=False,
            indent=False,
            layout=widgets.Layout(width="30%", height="50px"),
            icon="check",
        )

        self.psf_model = widgets.Dropdown(
            options=["gaussian", "lorentzian", "pseudo-voigt"],
            value="pseudo-voigt",
            description="PSF peak shape",
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="30%", height="25px"),
        )

        self.fwhm = widgets.FloatText(
            value=0.5,
            step=0.01,
            min=0,
            continuous_update=False,
            description="FWHM:",
            layout=widgets.Layout(width="15%", height="50px"),
            style={"description_width": "initial"},
        )

        self.eta = widgets.FloatText(
            value=0.05,
            step=0.01,
            max=1,
            min=0,
            continuous_update=False,
            description="Eta:",
            layout=widgets.Layout(width="15%", height="50px"),
            readout=True,
            style={"description_width": "initial"},
        )

        self.psf_filter = widgets.Dropdown(
            options=["None", "hann", "tukey"],
            value="None",
            description="PSF filter",
            layout=widgets.Layout(width="15%"),
            continuous_update=False,
            style={"description_width": "initial"},
        )

        self.update_psf = widgets.BoundedIntText(
            value=20,
            step=5,
            continuous_update=False,
            description="Update PSF every:",
            layout=widgets.Layout(width="35%", height="50px"),
            readout=True,
            style={"description_width": "initial"},
        )

        self.unused_label_algo = widgets.HTML(
            value="<p style='font-weight: bold;font-size:1.2em'>\
            Iterative algorithms parameters",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="90%", height="35px"),
        )

        self.nb_raar = widgets.BoundedIntText(
            value=1000,
            min=0,
            max=9999,
            step=10,
            continuous_update=False,
            description="Nb of RAAR:",
            layout=widgets.Layout(height="35px", width="20%"),
            readout=True,
            style={"description_width": "initial"},
        )

        self.nb_hio = widgets.BoundedIntText(
            value=400,
            min=0,
            max=9999,
            step=10,
            continuous_update=False,
            description="Nb of HIO:",
            layout=widgets.Layout(height="35px", width="20%"),
            readout=True,
            style={"description_width": "initial"},
        )

        self.nb_er = widgets.BoundedIntText(
            value=300,
            min=0,
            max=9999,
            step=10,
            continuous_update=False,
            description="Nb of ER:",
            layout=widgets.Layout(height="35px", width="20%"),
            readout=True,
            style={"description_width": "initial"},
        )

        self.nb_ml = widgets.BoundedIntText(
            value=0,
            min=0,
            max=9999,
            step=10,
            continuous_update=False,
            description="Nb of ML:",
            layout=widgets.Layout(height="35px", width="20%"),
            readout=True,
            style={"description_width": "initial"},
        )

        self.nb_run = widgets.BoundedIntText(
            value=30,
            min=0,
            max=100,
            continuous_update=False,
            description="Number of run:",
            layout=widgets.Layout(width="20%", height="50px"),
            readout=True,
            style={"description_width": "initial"},
        )

        self.unused_label_options = widgets.HTML(
            value="<p style='font-weight: bold;font-size:1.2em'>\
            Options",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="90%", height="35px"),
        )

        self.live_plot = widgets.BoundedIntText(
            value=200,
            step=10,
            max=500,
            min=0,
            continuous_update=False,
            description="Plot every:",
            readout=True,
            layout=widgets.Layout(height="50px", width="20%"),
            style={"description_width": "initial"},
        )

        self.plot_axis = widgets.Dropdown(
            options=[0, 1, 2],
            value=0,
            description="Axis used for plots",
            layout=widgets.Layout(width="20%"),
            style={"description_width": "initial"},
        )

        self.verbose = widgets.BoundedIntText(
            value=100,
            min=10,
            max=300,
            continuous_update=False,
            description="Verbose:",
            layout=widgets.Layout(width="20%", height="50px"),
            readout=True,
            style={"description_width": "initial"},
        )

        self.binning = widgets.Text(
            value="(1, 1, 1)",
            placeholder="(1, 1, 1)",
            description="Binning",
            layout=widgets.Layout(width="20%", height="50px"),
            continuous_update=False,
            style={"description_width": "initial"},
        )

        self.positivity = widgets.Checkbox(
            value=False,
            description="Force positivity",
            continuous_update=False,
            indent=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(height="50px", width="15%"),
            icon="check",
        )

        self.beta = widgets.FloatText(
            value=0.9,
            step=0.01,
            max=1,
            min=0,
            continuous_update=False,
            description="Beta parameter for RAAR and HIO:",
            layout=widgets.Layout(width="35%", height="50px"),
            readout=True,
            style={"description_width": "initial"},
        )

        self.detwin = widgets.Checkbox(
            value=False,
            description="Detwinning",
            continuous_update=False,
            indent=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(height="50px", width="15%"),
            icon="check",
        )

        self.calc_llk = widgets.BoundedIntText(
            value=50,
            min=0,
            max=100,
            continuous_update=False,
            description="Log likelihood update interval:",
            layout=widgets.Layout(width="40%", height="50px"),
            readout=True,
            style={"description_width": "initial"},
        )

        self.unused_label_mask_options = widgets.HTML(
            value="<p style='font-weight: bold;font-size:1.2em'>\
            Mask options</p>",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="90%", height="35px"),
        )

        self.zero_mask = widgets.Dropdown(
            options=("True", "False", "auto"),
            value="False",
            description="Force mask pixels to zero",
            continuous_update=False,
            indent=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="40%"),
            icon="check",
        )

        self.mask_interp = widgets.Text(
            value="(8, 2)",
            description="Mask interp.",
            layout=widgets.Layout(height="50px", width="40%"),
            continuous_update=False,
            style={"description_width": "initial"},
        )

        self.unused_label_run_options = widgets.HTML(
            value="<p style='font-weight: bold;font-size:1.2em'>\
            Job options</p>",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="90%", height="35px"),
        )

        self.unused_label_reconstruction_analysis = widgets.HTML(
            value="<p style='font-weight: bold;font-size:1.2em'>\
            Options for reconstruction analysis</p>",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="90%", height="35px"),
        )

        self.filter_criteria = widgets.Dropdown(
            options=[
                ("No filtering", "no_filtering"),
                ("Mean to Max", "mean_to_max"),
                ("Standard deviation", "std"),
                ("Free Log-likelihood (FLLK)", "llkf"),
                ("Log-likelihood (FLLK)", "llk"),
                ("FLLK > Standard deviation", "FLLK_standard_deviation"),
                ("Average of all", "all"),
                # ("Standard deviation > FLLK", "standard_deviation_FLLK"),
            ],
            value="mean_to_max",
            description="Filtering criteria",
            layout=widgets.Layout(width="30%"),
            style={"description_width": "initial"},
        )

        self.plot_metrics_checkbox = widgets.Checkbox(
            value=True,
            description="Plot metrics",
            continuous_update=False,
            indent=False,
            layout=widgets.Layout(width="20%", height="50px"),
            icon="check",
        )

        self.plot_reconstruction_checkbox = widgets.Checkbox(
            value=True,
            description="Plot reconstruction",
            continuous_update=False,
            indent=False,
            layout=widgets.Layout(width="20%", height="50px"),
            icon="check",
        )

        self.plot_reconstruction_phase_checkbox = widgets.Checkbox(
            value=False,
            description="Plot reconstruction phase",
            continuous_update=False,
            indent=False,
            layout=widgets.Layout(width="20%", height="50px"),
            icon="check",
        )

        self.unused_label_reconstructions_selection = widgets.HTML(
            value="<p style='font-weight: bold;font-size:1.2em'>\
            Options for reconstructions selection</p>",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="90%", height="35px"),
        )
        self.best_runs_widget = widgets.SelectMultiple(
            options=[],
            rows=5,
            description="Best runs:",
            style={"description_width": "initial"},
            layout=Layout(display="flex", flex_flow="column", width="50%"),
        )
        self.number_of_best_ordered_runs_widget = widgets.BoundedIntText(
            value=5,
            layout=widgets.Layout(height="50px", width="30%"),
            continuous_update=False,
            description="Number of best ordered runs:",
            readout=True,
            style={"description_width": "initial"},
        )
        self.button_update_cxi_file_list = Button(
            description="Update cxi file list",
            layout=Layout(width="15%", height="35px"),
        )

        @self.button_update_cxi_file_list.on_click
        def ActionSaveName(selfbutton):
            self.best_runs_widget.options = [
                os.path.basename(f)
                for f in list_files(
                    self.parent_folder.value,
                    self.search_pattern,
                )
            ]

            self.best_runs_widget.value = ()

        # Job buttons
        self.run_phase_retrieval = widgets.ToggleButtons(
            options=[
                ("No phase retrieval", False),
                # ('Run batch job (slurm)', "batch"),
                # ("Run script locally", "local_script"),
                ("Run phase retrieval", "operators"),
            ],
            value=False,
            tooltips=[
                "Click to be able to change parameters",
                "Use operators on local environment, if using PSF, it is "
                "activated after 50 % of RAAR cycles",
            ],
            continuous_update=False,
            button_style="",
            layout=widgets.Layout(width="100%", height="75x"),
            style={"description_width": "initial"},
            icon="fast-forward",
            description="I. Click below to run the phase retrieval",
        )

        self.run_pynx_tools = widgets.ToggleButtons(
            options=[
                ("No tool running", False),
                ("I. Analyse results", "analyse_results"),
                ("II. Select reconstructions", "select"),
                ("III. Modes decomposition", "mode_decomposition"),
            ],
            value=False,
            tooltips=[
                "Click to be able to change parameters",
                "Analyse reconstructions",
                "Select good reconstructions",
                "Modes decomposition",  # need to have selected first
            ],
            continuous_update=False,
            button_style="",
            layout=widgets.Layout(width="100%", height="75px"),
            style={"description_width": "initial"},
            icon="fast-forward",
            description=(
                "II. Click below to analyse and select your reconstructions or"
                " create a single solution after phase retrieval."
            ),
        )

        # Define children
        self.children = (
            self.unused_label_data,
            self.parent_folder,
            self.iobs,
            self.mask,
            self.support,
            self.obj,
            self.unused_label_support,
            widgets.HBox([self.support_threshold, self.support_only_shrink]),
            widgets.HBox(
                [
                    self.support_update_period,
                    self.support_smooth_width,
                    self.support_post_expand,
                ]
            ),
            widgets.HBox(
                [
                    self.support_method,
                    self.support_autocorrelation_threshold,
                ]
            ),
            self.unused_label_psf,
            widgets.HBox(
                [
                    self.psf,
                    self.psf_model,
                    self.fwhm,
                    self.eta,
                ]
            ),
            widgets.HBox(
                [
                    self.psf_filter,
                    self.update_psf,
                ]
            ),
            self.unused_label_algo,
            widgets.HBox(
                [
                    self.nb_hio,
                    self.nb_raar,
                    self.nb_er,
                    self.nb_ml,
                ]
            ),
            self.nb_run,
            self.unused_label_options,
            widgets.HBox(
                [
                    self.live_plot,
                    self.plot_axis,
                    self.verbose,
                ]
            ),
            widgets.HBox(
                [
                    self.binning,
                    self.positivity,
                ]
            ),
            widgets.HBox(
                [
                    self.beta,
                    self.detwin,
                    self.calc_llk,
                ]
            ),
            self.unused_label_mask_options,
            widgets.HBox(
                [
                    self.zero_mask,
                    self.mask_interp,
                ]
            ),
            self.unused_label_reconstruction_analysis,
            widgets.HBox(
                [
                    self.filter_criteria,
                    self.plot_metrics_checkbox,
                    self.plot_reconstruction_checkbox,
                    self.plot_reconstruction_phase_checkbox,
                ]
            ),
            self.unused_label_reconstructions_selection,
            widgets.HBox(
                [
                    self.best_runs_widget,
                    self.number_of_best_ordered_runs_widget,
                    self.button_update_cxi_file_list,
                ]
            ),
            self.unused_label_run_options,
            self.run_phase_retrieval,
            self.run_pynx_tools,
        )

        # Assign handler
        self.parent_folder.observe(self.pynx_folder_handler, names="value")
        self.psf.observe(self.pynx_psf_handler, names="value")
        self.psf_model.observe(self.pynx_peak_shape_handler, names="value")
        self.run_phase_retrieval.observe(self.run_pynx_handler, names="value")
        self.run_pynx_tools.observe(self.run_pynx_handler, names="value")

    # Define handlers
    def pynx_folder_handler(self, change) -> None:
        """
        Handles changes related to the pynx folder.

        Args:
            change: The change event triggered by the observer.

        Returns:
            None
        """
        if hasattr(change, "new"):
            change = change.new

        list_all_npz = [
            os.path.basename(f)
            for f in sorted(glob.glob(change + "/*.npz"), key=os.path.getmtime)
        ]

        list_probable_iobs_files = [
            os.path.basename(f)
            for f in sorted(
                glob.glob(change + "/*_pynx_*.npz"), key=os.path.getmtime
            )
        ]

        list_probable_mask_files = [
            os.path.basename(f)
            for f in sorted(
                glob.glob(change + "/*mask*.npz"), key=os.path.getmtime
            )
        ]

        # support list
        self.support.options = [""] + [
            os.path.basename(f)
            for f in sorted(glob.glob(change + "/*.npz"), key=os.path.getmtime)
        ]

        # obj list
        self.obj.options = [""] + [
            os.path.basename(f)
            for f in sorted(glob.glob(change + "/*.npz"), key=os.path.getmtime)
        ]

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

        # cxi file list
        self.best_runs_widget.options = [
            os.path.basename(f)
            for f in list_files(
                self.parent_folder.value,
                self.search_pattern,
            )
        ]

    def pynx_psf_handler(self, change) -> None:
        """
        Handles changes related to the psf.

        The function takes the `change` argument, which is expected to
        contain information related to the change event. If `change` has
        a `new` attribute, the value of `change` is set to `change.new`.

        The function disables or enables a number of widget objects
        (`self.psf_model`, `self.fwhm`, `self.eta`, `self.psf_filter`,
        and `self.update_psf`) based on the value of `change`.
        If `change` is truthy, the widgets are enabled.
        If `change` is falsy, the widgets are disabled.

        The function also calls `self.pynx_peak_shape_handler` with
        the `change` argument set to `self.psf_model.value`.

        Args:
            change: The new value of the change event.

        Returns:
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

        self.pynx_peak_shape_handler(change=self.psf_model.value)

    def pynx_peak_shape_handler(self, change) -> None:
        """
        Handles changes related to psf the peak shape.

        Args:

            change: The new value of the change event.

        Returns:
            None
        """
        if hasattr(change, "new"):
            change = change.new

        if change != "pseudo-voigt":
            self.eta.disabled = True

        else:
            self.eta.disabled = False

    def run_pynx_handler(self, change) -> None:
        """
        Handles changes related to the phase retrieval.

        Args:
            change: The new value of the change event.

        Returns:
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

            self.pynx_psf_handler(change=self.psf.value)

    def show(self):
        """
        Display the PhaseRetrievalGUI GUI as a standalone widget in a Jupyter
        Notebook.

        This method creates an interactive widget interface for configuring and
        running phase retrieval algorithms. It combines the PhaseRetrievalGUI
        widget with an interactive function (`init_phase_retrieval_tab`) that
        collects user inputs and executes the phase retrieval process.

        Returns:
            None
                Displays the GUI in the Jupyter Notebook environment.
        """
        if not IS_PYNX_AVAILABLE:
            raise PyNXImportError

        init_phase_retrieval_tab_gui = interactive(
            self.init_phase_retrieval_tab,
            parent_folder=self.parent_folder,
            iobs=self.iobs,
            mask=self.mask,
            support=self.support,
            obj=self.obj,
            support_threshold=self.support_threshold,
            support_only_shrink=self.support_only_shrink,
            support_update_period=self.support_update_period,
            support_smooth_width=self.support_smooth_width,
            support_post_expand=self.support_post_expand,
            support_method=self.support_method,
            support_autocorrelation_threshold=(
                self.support_autocorrelation_threshold
            ),
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
            live_plot=self.live_plot,
            plot_axis=self.plot_axis,
            verbose=self.verbose,
            binning=self.binning,
            positivity=self.positivity,
            beta=self.beta,
            detwin=self.detwin,
            calc_llk=self.calc_llk,
            zero_mask=self.zero_mask,
            mask_interp=self.mask_interp,
            run_phase_retrieval=self.run_phase_retrieval,
            run_pynx_tools=self.run_pynx_tools,
            filter_criteria=self.filter_criteria,
            plot_metrics_checkbox=self.plot_metrics_checkbox,
            plot_reconstruction_checkbox=self.plot_reconstruction_checkbox,
            plot_reconstruction_phase_checkbox=(
                self.plot_reconstruction_phase_checkbox
            ),
            best_runs_widget=self.best_runs_widget,
            number_of_best_ordered_runs_widget=(
                self.number_of_best_ordered_runs_widget
            ),
        )
        # Use children architecture defined in __init__.py
        window = widgets.VBox(
            [self, init_phase_retrieval_tab_gui.children[-1]]
        )
        display(window)

    def init_phase_retrieval_tab(
        self,
        parent_folder,
        iobs,
        mask,
        support,
        obj,
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
        live_plot,
        plot_axis,
        verbose,
        binning,
        positivity,
        beta,
        detwin,
        calc_llk,
        zero_mask,
        mask_interp,
        run_phase_retrieval,
        run_pynx_tools,
        filter_criteria,
        plot_metrics_checkbox,
        plot_reconstruction_checkbox,
        plot_reconstruction_phase_checkbox,
        best_runs_widget,
        number_of_best_ordered_runs_widget,
    ):
        """
        Get parameters values from widgets and run phase retrieval

        Possible to run phase retrieval via the CLI (with ot without MPI)
        Or directly in python using the operators.

        Args:
            parent_folder: folder in which the raw data files are, and
                where the output will be saved
            iobs: 2D/3D observed diffraction data (intensity). Assumed to
                be corrected and following Poisson statistics, will be
                converted to float32. Dimensions should be divisible by 4
                and have a prime factor decomposition up to 7. Internally,
                the following special values are used:
                * values<=-1e19 are masked. Among those, values in
                  ]-1e38;-1e19] are estimated values, stored as
                  -(iobs_est+1)*1e19, which can be used to make a loose
                  amplitude projection. Values <=-1e38 are masked (no
                  amplitude projection applied), just below the minimum
                  float32 value
                * -1e19 < values <= 1 are observed but used as free pixels
                  If the mask is not supplied, then it is assumed that the
                  above special values are used.
            support: initial support in real space (1 = inside support,
                0 = outside)
            obj: initial object. If None, it should be initialised later.
            mask: mask for the diffraction data (0: valid pixel, >0: masked)
            support_threshold: must be between 0 and 1. Only points with
                object amplitude above a value equal to relative_threshold *
                reference_value are kept in the support. reference_value
                can use the fact that when converged, the square norm of
                the object is equal to the number of recorded photons
                (normalized Fourier Transform). Then:
                reference_value = sqrt((abs(obj)**2).sum()/nb_points_support)
            support_smooth_width: smooth the object amplitude using a
                gaussian of this width before calculating new support.
                If this is a scalar, the smooth width is fixed to this value.
                If this is a 3-value tuple (or list or array), i.e.
                'smooth_width=2,0.5,600', the smooth width will vary with the
                number of cycles recorded in the CDI object (as cdi.cycle),
                varying exponentially from the first to the second value over
                the number of cycles specified by the last value.
                With 'smooth_width=a,b,nb':
                - smooth_width = a * exp(-cdi.cycle/nb*log(b/a)) if
                  cdi.cycle < nb
                - smooth_width = b if cdi.cycle >= nb
            support_only_shrink: if True, the support can only shrink
            support_post_expand: after the new support has been calculated,
                it can be processed using the SupportExpand operator, either
                one or multiple times, in order to 'clean' the support:
                - 'post_expand=1' will expand the support by 1 pixel
                - 'post_expand=-1' will shrink the support by 1 pixel
                - 'post_expand=(-1,1)' will shrink and then expand the
                  support by 1 pixel
                - 'post_expand=(-2,3)' will shrink and then expand the
                  support by respectively 2 and 3 pixels
            support_method: either 'max' or 'average' or 'rms' (default),
                the threshold will be relative to either the maximum
                amplitude in the object, or the average or root-mean-square
                amplitude (computed inside support)
            support_autocorrelation_threshold: if no support is given, it
                will be estimated from the intensity auto-correlation, with
                this relative threshold. A range can also be given, e.g.
                support_autocorrelation_threshold=0.09,0.11 and the actual
                threshold will be randomly chosen between the min and max.
            psf: e.g. True whether or not to use the PSF, partial coherence
                point-spread function, estimated with 50 cycles of
                Richardson-Lucy
            psf_model: "lorentzian", "gaussian" or "pseudo-voigt", or None
                to deactivate
            psf_filter: either None, "hann" or "tukey": window type to
                filter the PSF update
            fwhm: the full-width at half maximum, in pixels
            eta: the eta parameter for the pseudo-voigt
            update_psf: how often the psf is updated
            nb_raar: number of relaxed averaged alternating reflections
                cycles, which the algorithm will use first. During RAAR and
                HIO, the support is updated regularly
            nb_hio: number of hybrid input/output cycles, which the
                algorithm will use after RAAR. During RAAR and HIO, the
                support is updated regularly
            nb_er: number of error reduction cycles, performed after HIO,
                without support update
            nb_ml: number of maximum-likelihood conjugate gradient to
                perform after ER
            nb_run: number of times to run the optimization
            filter_criteria: e.g. "FLLK" criteria onto which the best
                solutions will be chosen
            live_plot: a live plot will be displayed every N cycle
            plot_axis: for 3D data, the axis along which the cut plane will
                be selected
            beta: the beta value for the HIO operator
            positivity: True or False
            zero_mask: if True, masked pixels (iobs<-1e19) are forced to
                zero, otherwise the calculated complex amplitude is kept with
                an optional scale factor. 'auto' is only valid if using the
                command line
            mask_interp: e.g. 16,2: interpolate masked pixels from
                surrounding pixels, using an inverse distance weighting.
                The first number N indicates that the pixels used for
                interpolation range from i-N to i+N for pixel i around all
                dimensions. The second number n that the weight is equal to
                1/d**n for pixels with at a distance n. The interpolated
                values iobs_m are stored in memory as -1e19*(iobs_m+1) so
                that the algorithm knows these are not true observations, and
                are applied with a large confidence interval.
            detwin: if set (command-line) or if detwin=True (parameters
                file), 10 cycles will be performed at 25% of the total
                number of RAAR or HIO cycles, with a support cut in half to
                bias towards one twin image
            calc_llk: interval at which the different Log Likelihood are
                computed
        """
        # Assign attributes
        params = {
            "parent_folder": parent_folder,
            "iobs": parent_folder + iobs,
            "mask": parent_folder + mask if mask != "" else "",
            "support": parent_folder + support if support != "" else "",
            "obj": parent_folder + obj if obj != "" else "",
            "support_only_shrink": support_only_shrink,
            "support_update_period": support_update_period,
            "support_method": support_method,
            "psf": psf,
            "psf_model": psf_model,
            "fwhm": fwhm,
            "eta": eta,
            "psf_filter": psf_filter,
            "update_psf": update_psf,
            "nb_raar": nb_raar,
            "nb_hio": nb_hio,
            "nb_er": nb_er,
            "nb_ml": nb_ml,
            "nb_run": nb_run,
            "verbose": verbose,
            "positivity": positivity,
            "beta": beta,
            "detwin": detwin,
            "calc_llk": calc_llk,
            "support_threshold": literal_eval(support_threshold),
            "support_autocorrelation_threshold": literal_eval(
                support_autocorrelation_threshold
            ),
            "support_smooth_width": literal_eval(support_smooth_width),
            "support_post_expand": literal_eval(support_post_expand),
            "binning": literal_eval(binning),
            "mask_interp": literal_eval(mask_interp),
            "zero_mask": {"True": True, "False": False, "auto": False}[
                zero_mask
            ],
            "live_plot": live_plot if live_plot != 0 else False,
            "plot_axis": plot_axis,
            "filter_criteria": filter_criteria,
            "plot_metrics_checkbox": plot_metrics_checkbox,
            "plot_reconstruction_checkbox": plot_reconstruction_checkbox,
            "plot_reconstruction_phase_checkbox": (
                plot_reconstruction_phase_checkbox
            ),
            "best_runs_widget": best_runs_widget,
            "number_of_best_ordered_runs_widget": (
                number_of_best_ordered_runs_widget
            ),
        }

        # Run PR with operators
        if run_phase_retrieval and not run_pynx_tools:
            # Get scan nb
            try:
                scan = int(parent_folder.split("/")[-3].replace("S", ""))
                params["scan"] = scan
                print("Scan n°", scan)
            except Exception as E:
                print(E)
                print("Could not get scan nb.")
                scan = 0

            # Keep a list of the resulting scans
            reconstruction_file_list = []

            try:
                # Initialise the cdi operator
                raw_cdi = initialize_cdi_operator(
                    iobs=params["iobs"],
                    mask=params["mask"],
                    support=params["support"],
                    obj=params["obj"],
                    binning=params["binning"],
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
                            parent_folder, iobs.split("/")[-1].split(".")[0]
                        )

                        cxi_filename = (
                            f"{parent_folder}/pynx_input_operator_"
                            f"{iobs.split('/')[-1].split('.')[0]}.cxi"
                        )

                        save_cdi_operator_as_cxi(
                            cdi_operator=cdi,
                            path_to_cxi=cxi_filename,
                            params=params,
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
                            params["support_threshold"][1],
                        )
                    print(f"Threshold: {threshold_relative:.4f}")

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
                        cdi = (
                            ShowCDI(plot_axis=params["plot_axis"])
                            * InterpIobsMask(
                                params["mask_interp"][0],
                                params["mask_interp"][1],
                            )
                            * cdi
                        )
                    else:
                        cdi = (
                            InterpIobsMask(
                                params["mask_interp"][0],
                                params["mask_interp"][1],
                            )
                            * cdi
                        )
                        print("test3")

                    # Initialize the support with autocorrelation, if no
                    # support given
                    if not support:
                        params["sup_init"] = "autocorrelation"
                        if not params["live_plot"]:
                            cdi = (
                                ScaleObj()
                                * AutoCorrelationSupport(
                                    threshold=params[
                                        "support_autocorrelation_threshold"
                                    ],
                                    verbose=True,
                                )
                                * cdi
                            )

                        else:
                            cdi = (
                                ShowCDI(plot_axis=params["plot_axis"])
                                * ScaleObj()
                                * AutoCorrelationSupport(
                                    threshold=params[
                                        "support_autocorrelation_threshold"
                                    ],
                                    verbose=True,
                                )
                                * cdi
                            )
                    else:
                        params["sup_init"] = "support"

                    # Begin phase retrieval
                    try:
                        if psf:
                            if support_update_period == 0:
                                cdi = (
                                    HIO(
                                        beta=params["beta"],
                                        calc_llk=params["calc_llk"],
                                        show_cdi=params["live_plot"],
                                        plot_axis=params["plot_axis"],
                                        positivity=params["positivity"],
                                        zero_mask=params["zero_mask"],
                                    )
                                    ** params["nb_hio"]
                                    * cdi
                                )
                                cdi = (
                                    RAAR(
                                        beta=params["beta"],
                                        calc_llk=params["calc_llk"],
                                        show_cdi=params["live_plot"],
                                        plot_axis=params["plot_axis"],
                                        positivity=params["positivity"],
                                        zero_mask=params["zero_mask"],
                                    )
                                    ** (params["nb_raar"] // 2)
                                    * cdi
                                )

                                # PSF is introduced at 66% of HIO and RAAR
                                if psf_model != "pseudo-voigt":
                                    cdi = (
                                        InitPSF(
                                            model=params["psf_model"],
                                            fwhm=params["fwhm"],
                                            filter=None,
                                        )
                                        * cdi
                                    )

                                elif psf_model == "pseudo-voigt":
                                    cdi = (
                                        InitPSF(
                                            model=params["psf_model"],
                                            fwhm=params["fwhm"],
                                            eta=params["eta"],
                                            filter=None,
                                        )
                                        * cdi
                                    )

                                cdi = (
                                    RAAR(
                                        beta=params["beta"],
                                        calc_llk=params["calc_llk"],
                                        show_cdi=params["live_plot"],
                                        update_psf=params["update_psf"],
                                        plot_axis=params["plot_axis"],
                                        positivity=params["positivity"],
                                        psf_filter=params["psf_filter"],
                                        zero_mask=params["zero_mask"],
                                    )
                                    ** (nb_raar // 2)
                                    * cdi
                                )
                                cdi = (
                                    ER(
                                        calc_llk=params["calc_llk"],
                                        show_cdi=params["live_plot"],
                                        update_psf=params["update_psf"],
                                        plot_axis=params["plot_axis"],
                                        positivity=params["positivity"],
                                        psf_filter=params["psf_filter"],
                                        zero_mask=params["zero_mask"],
                                    )
                                    ** nb_er
                                    * cdi
                                )

                            else:
                                hio_power = (
                                    params["nb_hio"]
                                    // params["support_update_period"]
                                )
                                raar_power = (
                                    params["nb_raar"] // 2
                                ) // params["support_update_period"]
                                er_power = (
                                    params["nb_er"]
                                    // params["support_update_period"]
                                )

                                cdi = (
                                    sup
                                    * HIO(
                                        beta=params["beta"],
                                        calc_llk=params["calc_llk"],
                                        show_cdi=params["live_plot"],
                                        plot_axis=params["plot_axis"],
                                        positivity=params["positivity"],
                                        psf_filter=params["psf_filter"],
                                        zero_mask=params["zero_mask"],
                                    )
                                    ** params["support_update_period"]
                                ) ** hio_power * cdi
                                cdi = (
                                    sup
                                    * RAAR(
                                        beta=params["beta"],
                                        calc_llk=params["calc_llk"],
                                        show_cdi=params["live_plot"],
                                        plot_axis=params["plot_axis"],
                                        positivity=params["positivity"],
                                        psf_filter=params["psf_filter"],
                                        zero_mask=params["zero_mask"],
                                    )
                                    ** params["support_update_period"]
                                ) ** raar_power * cdi

                                # PSF is introduced after half the HIO cycles
                                if psf_model != "pseudo-voigt":
                                    cdi = (
                                        InitPSF(
                                            model=params["psf_model"],
                                            fwhm=params["fwhm"],
                                            filter=params["psf_filter"],
                                        )
                                        * cdi
                                    )

                                elif psf_model == "pseudo-voigt":
                                    cdi = (
                                        InitPSF(
                                            model=params["psf_model"],
                                            fwhm=params["fwhm"],
                                            eta=params["eta"],
                                            filter=params["psf_filter"],
                                        )
                                        * cdi
                                    )

                                cdi = (
                                    sup
                                    * RAAR(
                                        beta=params["beta"],
                                        calc_llk=params["calc_llk"],
                                        show_cdi=params["live_plot"],
                                        update_psf=params["update_psf"],
                                        plot_axis=params["plot_axis"],
                                        positivity=params["positivity"],
                                        psf_filter=params["psf_filter"],
                                        zero_mask=params["zero_mask"],
                                    )
                                    ** params["support_update_period"]
                                ) ** raar_power * cdi
                                cdi = (
                                    sup
                                    * ER(
                                        calc_llk=params["calc_llk"],
                                        show_cdi=params["live_plot"],
                                        update_psf=params["update_psf"],
                                        plot_axis=params["plot_axis"],
                                        positivity=params["positivity"],
                                        psf_filter=params["psf_filter"],
                                        zero_mask=params["zero_mask"],
                                    )
                                    ** params["support_update_period"]
                                ) ** er_power * cdi

                        if not psf:
                            if support_update_period == 0:
                                cdi = (
                                    HIO(
                                        beta=params["beta"],
                                        calc_llk=params["calc_llk"],
                                        show_cdi=params["live_plot"],
                                        plot_axis=params["plot_axis"],
                                        positivity=params["positivity"],
                                        zero_mask=params["zero_mask"],
                                    )
                                    ** params["nb_hio"]
                                    * cdi
                                )
                                cdi = (
                                    RAAR(
                                        beta=params["beta"],
                                        calc_llk=params["calc_llk"],
                                        show_cdi=params["live_plot"],
                                        plot_axis=params["plot_axis"],
                                        positivity=params["positivity"],
                                        zero_mask=params["zero_mask"],
                                    )
                                    ** params["nb_raar"]
                                    * cdi
                                )
                                cdi = (
                                    ER(
                                        calc_llk=params["calc_llk"],
                                        show_cdi=params["live_plot"],
                                        plot_axis=params["plot_axis"],
                                        positivity=params["positivity"],
                                        zero_mask=params["zero_mask"],
                                    )
                                    ** params["nb_er"]
                                    * cdi
                                )

                            else:
                                hio_power = (
                                    params["nb_hio"]
                                    // params["support_update_period"]
                                )
                                raar_power = (
                                    params["nb_raar"]
                                    // params["support_update_period"]
                                )
                                er_power = (
                                    params["nb_er"]
                                    // params["support_update_period"]
                                )

                                cdi = (
                                    sup
                                    * HIO(
                                        beta=params["beta"],
                                        calc_llk=params["calc_llk"],
                                        show_cdi=params["live_plot"],
                                        plot_axis=params["plot_axis"],
                                        positivity=params["positivity"],
                                        zero_mask=params["zero_mask"],
                                    )
                                    ** params["support_update_period"]
                                ) ** hio_power * cdi
                                cdi = (
                                    sup
                                    * RAAR(
                                        beta=params["beta"],
                                        calc_llk=params["calc_llk"],
                                        show_cdi=params["live_plot"],
                                        plot_axis=params["plot_axis"],
                                        positivity=params["positivity"],
                                        zero_mask=params["zero_mask"],
                                    )
                                    ** params["support_update_period"]
                                ) ** raar_power * cdi
                                cdi = (
                                    sup
                                    * ER(
                                        calc_llk=params["calc_llk"],
                                        show_cdi=params["live_plot"],
                                        plot_axis=params["plot_axis"],
                                        positivity=params["positivity"],
                                        zero_mask=params["zero_mask"],
                                    )
                                    ** params["support_update_period"]
                                ) ** er_power * cdi

                        fn = (
                            f"{params['parent_folder']}/"
                            f"result_scan_{params['scan']}_run_{i}_"
                            f"FLLK_{cdi.get_llk(normalized=True)[3]:.4f}_"
                            f"support_threshold_{threshold_relative:.4f}_"
                            f"shape_{cdi.iobs.shape[0]}_{cdi.iobs.shape[1]}"
                            f"_{cdi.iobs.shape[2]}_"
                            f"{params['sup_init']}.cxi"
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
                            "The support is too large to continue,"
                            "try increasing the threshold value."
                        )

                    except SupportTooSmall:
                        print(
                            "The support is too small to continue,"
                            "try lowering the threshold value."
                        )

            except KeyboardInterrupt:
                clear_output(True)
                print("Phase retrieval stopped by user.")

        # Modes decomposition and solution filtering
        if run_pynx_tools and not run_phase_retrieval:
            if run_pynx_tools == "analyse_results":
                self.pipeline.analyse_phasing_results(
                    sorting_criterion=params["filter_criteria"],
                    plot=params["plot_metrics_checkbox"],
                    plot_phasing_results=(
                        params["plot_reconstruction_checkbox"]
                    ),
                    plot_phase=params["plot_reconstruction_phase_checkbox"],
                    search_pattern=self.search_pattern,
                )
            elif run_pynx_tools == "select":
                options = [
                    os.path.basename(f)
                    for f in list_files(
                        params["parent_folder"],
                        self.search_pattern,
                    )
                ]
                indices = [
                    options.index(item) for item in params["best_runs_widget"]
                ]

                self.pipeline.select_best_candidates(
                    nb_of_best_sorted_runs=(
                        params["number_of_best_ordered_runs_widget"]
                    ),
                    best_runs=indices,
                    search_pattern=self.search_pattern,
                )
            elif run_pynx_tools == "mode_decomposition":
                self.pipeline.mode_decomposition()

        # Clean output
        if not run_phase_retrieval and not run_pynx_tools:
            print("Cleared output.")
            clear_output(True)


# Functions below shall not be in this module

def initialize_cdi_operator(
    iobs: str,
    mask: str | None,
    support: str | None,
    obj: str | None,
    binning: tuple[int, int, int] = (1, 1, 1),
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
    """
        Initialize the CDI operator by processing the possible inputs:
            - iobs
            - mask
            - support
            - obj
        Will also crop and center the data if specified.

        Args:
            iobs: path to npz or npy that stores the intensity observations
                data
            mask: path to npz or npy that stores the mask data
            support: path to npz or npy that stores the support data
            obj: path to npz or npy that stores the object data
            binning: tuple, applied to all the arrays, e.g. (1, 1, 1)

        Returns:
            cdi operator or None if initialization fails
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
                print('\t"data" key does not exist.')
                return None
        if binning != (1, 1, 1):
            iobs = rebin(iobs, binning)
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
            for key in ["mask", "data"]:
                try:
                    mask = np.load(mask)[key].astype(np.int8)
                    nb = mask.sum()
                    mask_percentage = nb * 100 / mask.size
                    print(
                        f"\tCXI input: loading mask, "
                        f"with {nb} pixels masked ({mask_percentage:0.3f}%)"
                    )
                    break
                except KeyError:
                    print(f'\t"{key}" key does not exist.')
            else:
                print("\t--> Could not load mask array.")

        if binning != (1, 1, 1):
            mask = rebin(mask, binning)
            print("\tBinned mask.")

        mask = fftshift(mask)

    else:
        mask = None

    if os.path.isfile(str(support)):
        if support.endswith(".npy"):
            support = np.load(support)
            print("\tCXI input: loading support")
        elif support.endswith(".npz"):
            for key in ["data", "support", "obj"]:
                try:
                    support = np.load(support)[key]
                    print("\tCXI input: loading support")
                    break
                except KeyError:
                    print(f'\t"{key}" key does not exist.')
            else:
                print("\t--> Could not load support array.")

        if binning != (1, 1, 1):
            support = rebin(support, binning)
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
                print('\t"data" key does not exist.')

        if binning != (1, 1, 1):
            obj = rebin(obj, binning)
            print("\tBinned obj.")

        obj = fftshift(obj)

    else:
        obj = None

    # Create cdi object with data and mask, load the main parameters
    cdi = CDI(
        iobs,
        support=support,
        obj=obj,
        mask=mask,
    )

    return cdi


def save_cdi_operator_as_cxi(
    cdi_operator,
    path_to_cxi: str,
    params: dict | None,
):
    """
    We need to create a dictionnary with the parameters to save in the
    cxi file.

    Args:
        cdi_operator: cdi object created with PyNX
        path_to_cxi: path to future cxi data
            Below are parameters that can be saved in the cxi file
            - filename: the file name to save the data to
            - iobs: the observed intensity
            - mask: the mask indicating valid (=0) and bad pixels (>0)
            - params: a dictionary of parameters which will be saved as a
              NXcollection
        params: dictionary of additional parameters to save

    Returns:
        None
    """
    print(
        "\nSaving phase retrieval parameters selected "
        "in the PyNX tab in the cxi file."
    )
    cdi_operator.save_data_cxi(
        filename=path_to_cxi,
        process_parameters=params,
    )


def list_files(
    folder: str, glob_pattern: str, verbose: bool = False
) -> list[str]:
    """
    List all files in a specified folder that match a specified
     glob pattern, and sort by creation time.

    Args:
        folder (str): The path to the folder where the files are located.
        glob_pattern (str, optional): A string that specifies the pattern
            of the filenames to match.
        verbose (bool, optional): If set to True, the function will print
            the filenames and their creation timestamps to the console.
            Default is False.

    Returns:
        list: A list of file paths that match the specified pattern and
            are sorted by creation time (most recent first).

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
                os.path.getmtime(f)
            ).strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"\nFile: {os.path.basename(f)}\n\tCreated: {file_timestamp}"
            )
        print(
            "################################################"
            "################################################"
        )

    return file_list
