"""
Definition of the BcdiPipeline class.

Authors:
    * Clément Atlan, clement.atlan@esrf.fr - 09/2024
"""

# Built-in dependencies.
import glob
import os
from string import Template
import subprocess

# Dependencies.
import h5py
import numpy as np
from tabulate import tabulate
import yaml

# General cdiutils classes, to handle loading, beamline geometry and
# space conversion.
from cdiutils.converter import SpaceConverter
from cdiutils.geometry import Geometry
from cdiutils.load import Loader
from cdiutils.cxi import CXIFile

# Utility functions
from cdiutils.utils import (
    CroppingHandler,
    oversampling_from_diffraction,
    ensure_pynx_shape,
    hot_pixel_filter,
    get_oversampling_ratios,
    find_isosurface,
)

# Plot function specifically made for the pipeline.
from .pipeline_plotter import PipelinePlotter
from cdiutils.plot.colormap import RED_TO_TEAL
from cdiutils.plot.volume import plot_3d_surface_projections

from cdiutils.process.phaser import PhasingResultAnalyser, PynNXImportError
from cdiutils.process.postprocessor import PostProcessor
from cdiutils.process.facet_analysis import FacetAnalysisProcessor

# Base Pipeline class and pipeline-related functions.
from .base import Pipeline, IS_VTK_AVAILABLE
from .parameters import check_params, convert_np_arrays

# to save version in files:
from cdiutils import __version__


class PyNXScriptError(Exception):
    """Custom exception to handle pynx script failure."""
    pass


class BcdiPipeline(Pipeline):
    """
    A class to handle the BCDI workflow, from pre-processing to
    post-processing, including phase retrieval (using PyNX package).
    Provide either a path to a parameter file or directly the parameter
    dictionary.

    Args:
        param_file_path (str, optional): the path to the
            parameter file. Defaults to None.
        parameters (dict, optional): the parameter dictionary.
            Defaults to None.

    """
    voxel_pos = ("ref", "max", "com")
    class_isosurface = 0.1

    def __init__(
            self,
            params: dict = None,
            param_file_path: str = None,
    ):
        """
        Initialisation method.

        Args:
            param_file_path (str, optional): the path to the
                parameter file. Defaults to None.
            parameters (dict, optional): the parameter dictionary.
                Defaults to None.

        """
        super().__init__(params, param_file_path)

        check_params(self.params)

        self.scan = self.params["scan"]
        self.sample_name = self.params["sample_name"]
        self.pynx_phasing_dir = self.dump_dir + "/pynx_phasing/"

        # The dictionary of the Voxels Of Interest, in both the "full"
        # and "cropped" detector frames.
        self.voi = {"full": {}, "cropped": {}}

        # The dictionary of the VOI associated q_lab positions
        self.q_lab_pos = {key: None for key in self.voxel_pos}

        # The dictionary of the atomic parameters (d-spacing and lattice
        # parameter) for the various voxel positions.
        self.atomic_params = {
            "dspacing": {k: None for k in self.voxel_pos},
            "lattice_parameter": {k: None for k in self.voxel_pos},
        }

        # Define base attributes
        self.detector_data: np.ndarray = None
        self.cropped_detector_data: np.ndarray = None
        self.mask: np.ndarray = None
        self.angles: dict = None
        self.converter: SpaceConverter = None
        self.result_analyser: PhasingResultAnalyser = None
        self.reconstruction: np.ndarray = None
        self.structural_props: dict = None

        # For storing data that later saved in the cxi files
        self.extra_info: dict = {}

        self.logger.info("BcdiPipeline initialised.")

    @classmethod
    def from_file(cls, path: str) -> "BcdiPipeline":
        """
        Factory method to create a BcdiPipeline instance from a file.
        """
        if path.endswith(".cxi"):
            params, converter = cls.load_from_cxi(path)
        elif path.endswith(".yml") or path.endswith(".yaml"):
            raise ValueError("Loading from yaml files not yet implemented.")
        else:
            raise ValueError("File format not supported.")
        instance = cls(params)
        instance.converter = converter
        if "q_lab_ref" not in params:
            raise ValueError("q_lab_ref is missing in the parameters.")
        instance.q_lab_pos["ref"] = params["q_lab_ref"]
        return instance

    def update_from_file(self, path: str) -> None:
        """
        Update the current instance with parameters loaded from a
        CXI file.
        """
        if path.endswith(".cxi"):
            params, converter = self.load_from_cxi(path)
        elif path.endswith(".yml") or path.endswith(".yaml"):
            raise ValueError("Loading from yaml files not yet implemented.")
        else:
            raise ValueError("File format not supported.")
        self.params.update(params)
        self.converter = converter
        if "q_lab_ref" not in params:
            raise ValueError("q_lab_ref is missing in the parameters.")
        self.q_lab_pos["ref"] = params["q_lab_ref"]

    @classmethod
    def load_from_cxi(cls, path: str) -> tuple[dict, SpaceConverter]:
        if not path.endswith(".cxi"):
            raise ValueError("CXI file expected.")

        with CXIFile(path, "r") as cxi:
            params = cxi["entry_1/parameters_1"]
            converter = cls._build_converter_from_cxi(cxi)
        return params, converter

    @staticmethod
    def _build_converter_from_cxi(cxi: CXIFile):
        converter_params = {
            "geometry": Geometry.from_setup(
                cxi["entry_1/geometry_1/name"]
            ),
            "det_calib_params": cxi["entry_1/detector_1/calibration"],
            "roi": cxi["entry_1/result_1/roi"],
            "energy": cxi["entry_1/source_1/energy"],
            "shape": cxi["entry_1/image_1/image_size"],
            "q_space_shift": cxi["entry_1/result_2/q_space_shift"],
            "q_lab_matrix": cxi[
                "entry_1/result_2/transformation_matrices/q_lab"
            ],
            "direct_lab_matrix": cxi[
                "entry_1/result_2/transformation_matrices/direct_lab"
            ],
            "direct_lab_voxel_size": cxi[
                "entry_1/result_2/direct_lab_voxel_size"
            ],
        }
        converter = SpaceConverter(**converter_params)
        converter.init_q_space(**cxi["entry_1/geometry_1/angles"])
        return converter

    @Pipeline.process
    def preprocess(self, **params) -> None:
        """
        Main method to handle the preprocessing of the BCDI data. It
        takes care of the data loading, centring, cropping and gets the
        orthogonalisation parameters.

        Arguments:
            **params (optional): additional parameters provided by the
                user. They will overwrite those parsed upon instance
                initialisation.

        Raises:
            ValueError: if the requested shape and the voxel reference
                are not compatible.
        """
        if params:
            self.logger.info(
                "Additional parameters provided, will update the current "
                "dictionary of parameters."
            )
            self.params.update(params)

        # If voxel_reference_methods is not a list, make it a list
        if not isinstance(self.params["voxel_reference_methods"], list):
            self.params["voxel_reference_methods"] = [
                self.params["voxel_reference_methods"]
            ]

        # Check whether the requested shape is permitted by pynx
        self.params["preprocess_shape"] = ensure_pynx_shape(
            self.params["preprocess_shape"], verbose=True
        )

        if self.params["light_loading"]:
            roi = self._light_load()
            # Filter the data
            self.cropped_detector_data = self._filter(
                self.cropped_detector_data
            )

        else:
            self._load()
            self._from_2d_to_3d_shape()
            self.logger.info(
                "The preprocessing output shape is: and "
                f"{self.params['preprocess_shape']} will be used for the "
                "determination of the ROI dimensions."
            )
            # Filter, crop and centre the detector data.
            self.cropped_detector_data, roi = self._crop_centre(
                self._filter(self.detector_data)
            )
        for r in roi:
            if r < 0:
                raise ValueError(
                    "The preprocess_shape and the detector voxel reference are"
                    f" not compatible: {self.params['preprocess_shape'] = }, "  # noqa E251, E202
                    f"{self.voi['full']['ref'] = }."  # noqa E251, E202
                )
        # print out the oversampling ratio and rebin factor suggestion
        ratios = oversampling_from_diffraction(
            self.cropped_detector_data
        )
        if ratios is None:
            self.logger.info("Could not estimate the oversampling.")
        else:
            self.logger.info(
                "\nOversampling ratios calculated from diffraction pattern "
                "are: "
                + ", ".join(
                    [f"axis{i}: {ratios[i]:.1f}" for i in range(len(ratios))]
                )
                + ". If low-strain crystal, you can set PyNX 'rebin' parameter"
                " to (" + ", ".join([f"{r//2}" for r in ratios]) + ")"
            )

        # position of the max and com in the cropped detector frame
        for pos in ("max", "com"):
            self.voi["cropped"][pos] = CroppingHandler.get_position(
                self.cropped_detector_data, pos
            )

        # Initialise SpaceConverter, later used for orthogonalisation
        geometry = Geometry.from_setup(self.params["beamline_setup"])
        self.converter = SpaceConverter(
            geometry=geometry,
            det_calib_params=self.params["det_calib_params"],
            energy=self.params["energy"],
            roi=roi[2:]
        )
        self.converter.init_q_space(**self.angles)

        # Initialise the fancy table with the columns
        table = [
            ["voxel", "uncrop. det. pos.", "crop. det. pos.",
             "dspacing (A)", "lat. par. (A)"]
        ]

        # get the position of the reference, max and det voxels in the
        # q lab space
        for pos in self.voxel_pos:
            det_voxel = self.voi["full"][pos]
            cropped_det_voxel = self.voi["cropped"][pos]
            if any(np.isnan(e) for e in cropped_det_voxel):
                self.q_lab_pos[pos] = None
                self.atomic_params["dspacing"][pos] = None
                self.atomic_params["lattice_parameter"][pos] = None
            else:
                self.q_lab_pos[pos] = self.converter.index_det_to_q_lab(
                    cropped_det_voxel
                )

                # compute the corresponding dpsacing and lattice parameter
                # for printing
                self.atomic_params["dspacing"][pos] = self.converter.dspacing(
                    self.q_lab_pos[pos]
                )
                self.atomic_params["lattice_parameter"][pos] = (
                    self.converter.lattice_parameter(
                        self.q_lab_pos[pos], self.params["hkl"]
                    )
                )
            table.append(
                [pos, det_voxel, cropped_det_voxel,
                 f"{self.atomic_params['dspacing'][pos]:.5f}",
                 f"{self.atomic_params['lattice_parameter'][pos]:.5f}"]
            )

        self._unwrap_logs()  # Turn off wrapping for structured output
        self.logger.info(
            "\nSummary table:\n"
            + tabulate(table, headers="firstrow", tablefmt="fancy_grid"),
        )
        self._wrap_logs()  # Turn wrapping back on for regular logs

        if self.params["orthogonalise_before_phasing"]:
            self.logger.info(
                "Orthogonalization required before phasing.\n"
                "Will use xrayutilities Fuzzy Gridding without linear "
                "approximation."
            )
            self.orthogonalised_intensity = (
                self.converter.orthogonalise_to_q_lab(
                     self.cropped_detector_data, method="xrayutilities"
                )
            )
            # we must orthogonalise the mask and orthogonalised_intensity must
            # be saved as the pynx input
            self.mask = self.converter.orthogonalise_to_q_lab(
                self.mask, method="xrayutilities"
            )
            self.cropped_detector_data = self.orthogonalised_intensity
            q_lab_regular_grid = self.converter.get_xu_q_lab_regular_grid()
        else:
            self.logger.info(
                "Will linearise the transformation between detector and"
                " lab space."
            )
            # Initialise the interpolator so we won't need to reload raw
            # data during the post processing. The converter will be saved.
            self.converter.init_interpolator(
                direct_lab_voxel_size=self.params["voxel_size"],
                space="both",
                verbose=True
            )
            if self.params["voxel_size"] is not None:
                self.logger.info(
                    "Voxel size provided by user will be saved for the direct "
                    "space orthogonalisation."
                )

            # Run the interpolation in the reciprocal space so we don't
            # do it later
            self.orthogonalised_intensity = (
                self.converter.orthogonalise_to_q_lab(
                    self.cropped_detector_data,
                    method="cdiutils"
                )
            )

            q_lab_regular_grid = self.converter.get_q_lab_regular_grid()

        # Update the preprocess_shape and the det_reference_voxel
        self.params["det_reference_voxel"] = self.voi["full"]["ref"]

        # plot and save the detector data in the full detector frame and
        # final frame
        dump_file_tmpl = (
            f"{self.dump_dir}/S{self.scan}_" + "detector_data_{}.png"
        )
        PipelinePlotter.detector_data(
            self.cropped_detector_data,
            full_det_data=self.detector_data,
            voxels=self.voi,
            title=(
                "Detector data preprocessing (slices), "
                f"{self.sample_name}, S{self.scan}"
            ),
            save=dump_file_tmpl.format("slices")
        )
        PipelinePlotter.detector_data(
            self.cropped_detector_data,
            full_det_data=self.detector_data,
            voxels=self.voi,
            integrate=True,
            title=(
                "Detector data preprocessing (sum), "
                f"{self.sample_name}, S{self.scan}"
            ),
            save=dump_file_tmpl.format("sum")
        )

        # Plot the reciprocal space data in the detector and lab frames
        PipelinePlotter.ortho_detector_data(
            self.cropped_detector_data,
            self.orthogonalised_intensity,
            q_lab_regular_grid,
            title=(
                r"From detector frame to q lab frame"
                f", {self.sample_name}, S{self.scan}"
            ),
            save=dump_file_tmpl.format("orthogonalisation")
        )

        # Save the data and the parameters in the dump directory, and
        # save the q_lab reference position as a parameter.
        self.params["q_lab_ref"] = self.q_lab_pos["ref"]
        self._save_preprocessed_data()
        self._save_parameter_file()

    def _from_2d_to_3d_shape(self) -> tuple:
        if len(self.params["preprocess_shape"]) == 2:
            self.params["preprocess_shape"] = (
                (self.detector_data.shape[0], )
                + self.params["preprocess_shape"]
            )

    def _load(self, roi: tuple[slice] = None) -> None:
        """
        Load the raw detector data and motor positions.

        Args:
            roi (tuple[slice], optional): the region of interest on the
                detector frame. Defaults to None.

        Raises:
            ValueError: check whether detector data, motor positions and
                mask have been correctly loaded.
        """
        # Make the loader using the factory method, only parse the
        # parameters relevant to the Loader class, to make them explicit.
        loader_keys = (
            "beamline_setup", "scan", "sample_name", "experiment_file_path",
            "experiment_data_dir_path", "detector_data_path",
            "edf_file_template", "detector_name", "alien_mask", "flat_field"
        )
        loader = Loader.from_setup(**{k: self.params[k] for k in loader_keys})

        if self.params.get("detector_name") is None:
            self.params["detector_name"] = loader.detector_name
            if self.params.get("detector_name") is None:
                raise ValueError(
                    "The automatic detection of the detector name is not"
                    "yet implemented for this setup"
                    f"({self.params['setup']} = )."
                )

        self.detector_data = loader.load_detector_data(
            roi=roi,
            rocking_angle_binning=self.params["rocking_angle_binning"]
        )
        if roi is not None:
            shape = loader.load_detector_shape()
            if shape is not None:
                self.logger.info(f"Raw detector data shape is: {shape}.")
        else:
            self.logger.info(
                f"Raw detector data shape is: {self.detector_data.shape}."
            )

        self.angles = loader.load_motor_positions(
            roi=roi,
            rocking_angle_binning=self.params["rocking_angle_binning"]
        )
        self.mask = loader.get_mask(
            channel=self.detector_data.shape[0],
            detector_name=self.params["detector_name"],
            roi=(slice(None), roi[1], roi[2]) if roi else None
        )
        if loader.get_alien_mask() is not None:
            self.logger.info("Alien mask provided. Will update detector mask.")
            alien_mask = loader.get_alien_mask(roi)
            self.mask = np.where(self.mask + alien_mask > 0, 1, 0)

        if any(
                data is None
                for data in (self.detector_data, self.angles)
        ):
            raise ValueError("Something went wrong during data loading.")

        if self.params["energy"] is None:
            self.params["energy"] = loader.load_energy(
                self.scan
            )
            if self.params["energy"] is None:
                raise ValueError(
                    "The automatic loading of energy is not yet implemented"
                    f"for this setup ({self.params['setup']} = )."
                )
            else:
                self.logger.info(
                    f"Energy successfully loaded ({self.params['energy']} eV)."
                )
        if self.params["det_calib_params"] is None:
            self.logger.info(
                "\ndet_calib_params not provided, will try to find them. "
                "However, for a more accurate calculation, you'd better "
                "provide them."
            )
            self.params["det_calib_params"] = (
                loader.load_det_calib_params(self.scan)
            )
            if self.params["det_calib_params"] is None:
                raise ValueError(
                    "The automatic loading of det_calib_params is not yet "
                    "implemented for this setup "
                    f"('{self.params['beamline_setup']}'), you must provide "
                    "them."
                )
            else:
                self.logger.info(
                    "det_calib_params successfully loaded:\n"
                    f"{self.params['det_calib_params']}"
                    )

    def _light_load(self) -> list:
        """
        Light load the detector data according to the provided
        preprocess_shape and voxel_reference_methods
        position. This allows to load only the region of interest
        determined by these two parameters.

        Raises:
            ValueError: When final_shape length is not 3 and
                 rocking_angle_binning are not provided.
            ValueError: If voxel_reference_methods is not provided.

        Returns:
            list: the roi associated to the cropped_detector_data.
        """
        if (
                len(self.params["preprocess_shape"]) != 3
                and self.params["rocking_angle_binning"] is None
        ):
            self.params["rocking_angle_binning"] = 1
            self.logger.warning(
                "When light loading is requested, preprocess_shape must "
                "include 3 axis lengths or rocking_angle_binning must be "
                "provided. Will set rocking_angle_binning to 1, but this might"
                " not be very efficient."
            )
        self.voi["full"]["ref"] = self.params["voxel_reference_methods"][-1]
        if isinstance(self.voi["full"]["ref"], str):
            raise ValueError(
                "When light loading, voxel_reference_methods must "
                "contain a tuple indicating the position of the voxel you "
                "want to crop the data at. Ex: [(100, 200, 200)]."
            )
        if (
                len(self.voi["full"]["ref"]) == 2
                and len(self.params["preprocess_shape"]) == 3
        ):
            self.params["preprocess_shape"] = (
                self.params["preprocess_shape"][1:]
            )
        elif (
                len(self.voi["full"]["ref"]) == 3
                and len(self.params["preprocess_shape"]) == 2
        ):
            self.voi["full"]["ref"] = self.voi["full"]["ref"][1:]

        roi = CroppingHandler.get_roi(
            self.params["preprocess_shape"], self.voi["full"]["ref"]
        )
        if len(roi) == 4:
            roi = [None, None, roi[0], roi[1], roi[2], roi[3]]

        self.logger.info(
            f"\nLight loading requested, will use ROI {roi} and "
            "bin along rocking curve direction by "
            f"{self.params['rocking_angle_binning']} during data loading."
        )
        self._load(roi=CroppingHandler.roi_list_to_slices(roi))

        # If user only specified position in 2D or provided a 2D
        # preprocess_shape, we must extend it to 3D and check the first
        # dimension length.
        if len(self.voi["full"]["ref"]) == 2:  # covers both cases
            self.voi["full"]["ref"] = (
                (self.detector_data.shape[0] // 2, )
                + tuple(self.voi["full"]["ref"])
            )
            self._from_2d_to_3d_shape()
            shape = ensure_pynx_shape(self.params["preprocess_shape"])
            roi = CroppingHandler.get_roi(shape, self.voi["full"]["ref"])

            # If the shape has changed due to PyNX conventions, then
            # only crop the data along the first axis direction.
            if shape != self.params["preprocess_shape"]:
                roi_1d = CroppingHandler.roi_list_to_slices(roi[:2])
                self.detector_data = self.detector_data[roi_1d]
                self.mask = self.mask[roi_1d]
                rocking_angle = Loader.get_rocking_angle(self.angles)
                self.angles[rocking_angle] = self.angles[rocking_angle][roi_1d]

            self.params["preprocess_shape"] = shape
            self.logger.info(f"New ROI is: {roi}.")

        # find the position in the cropped detector frame
        self.voi["cropped"]["ref"] = tuple(
                p - r if r else p  # if r is None, p-r must be p
                for p, r in zip(self.voi["full"]["ref"], roi[::2])
        )
        self.voi["full"]["max"], self.voi["full"]["com"] = None, None

        # Since we light load the data, the full detector data do not exist
        self.cropped_detector_data = self.detector_data.copy()
        self.detector_data = None

        return roi

    def _filter(self, data: np.ndarray) -> np.ndarray:
        if self.params["hot_pixel_filter"]:
            self.logger.info("hot_pixel_filter requested.")
            if isinstance(self.params["hot_pixel_filter"], tuple):
                self.logger.info(
                    "Hot pixel filter parameters are : "
                    f"{self.params['hot_pixel_filter']}"
                )

                data, hot_pixel_mask = hot_pixel_filter(
                    data, *self.params["hot_pixel_filter"]
                )
            else:
                self.logger.info(
                    "Will use defaults parameters: threshold = 1e2, "
                    "kernel_size = 3 "
                )
                data, hot_pixel_mask = hot_pixel_filter(data)
            self.mask = np.where(hot_pixel_mask + self.mask > 0, 1, 0)

        if self.params["background_level"]:
            self.logger.info(
                f"background_level set to {self.params['background_level']}, "
                "will remove the background."
            )
            data -= self.params["background_level"]
            data[data < 0] = 0
        return data

    def _crop_centre(self, detector_data) -> tuple[np.ndarray, list]:
        """
        Crop and centre the self.detector data and return the cropped
        detector data and the associated roi.
        """
        self.logger.info(
            "Method(s) employed for the voxel reference determination are "
            f"{self.params['voxel_reference_methods']}."
        )
        self._unwrap_logs()  # Turn off wrapping for structured output
        (
            cropped_detector_data,
            self.voi["full"]["ref"],
            self.voi["cropped"]["ref"],
            roi
        ) = CroppingHandler.chain_centring(
                detector_data,
                self.params["preprocess_shape"],
                methods=self.params["voxel_reference_methods"],
                verbose=True
        )
        self._wrap_logs()  # Turn wrapping back on for regular logs

        # position of the max and com in the full detector frame
        for pos in ("max", "com"):
            self.voi["full"][pos] = CroppingHandler.get_position(
                self.detector_data, pos
            )
        # convert numpy.int64 to int to make them serializable
        self.params["det_reference_voxel"] = tuple(
            int(e) for e in self.voi["full"]["ref"]
        )

        # center and crop the mask
        self.mask = self.mask[CroppingHandler.roi_list_to_slices(roi)]

        self.logger.info(
            "The reference voxel was found at "
            f"{self.voi['full']['ref']} in the uncropped data frame.\n"
            "The processing_out_put_shape being "
            f"{self.params['preprocess_shape']}, the roi used to crop "
            f"the data is {roi}.\n"
        )

        # set the q space area with the sample and detector angles
        # that correspond to the requested roi
        for key, value in self.angles.items():
            if isinstance(value, (list, np.ndarray)):
                self.angles[key] = value[np.s_[roi[0]:roi[1]]]

        return cropped_detector_data, roi

    def _save_parameter_file(self) -> None:
        """
        Save the parameters used during the analysis.
        """
        output_file_path = f"{self.dump_dir}/S{self.scan}_parameters.yml"

        self.params = convert_np_arrays(**self.params)
        with open(output_file_path, "w", encoding="utf8") as file:
            yaml.dump(self.params, file)
        self.logger.info(
            f"\nScan parameter file saved at:\n{output_file_path}"
        )

    def _save_preprocessed_data(self) -> None:
        """Save the data generated during the preprocessing."""
        # Prepare dir in which pynx phasing results will be saved.
        os.makedirs(self.pynx_phasing_dir, exist_ok=True)

        # Save the cropped detector data and mask in pynx phasing dir
        for name, data in zip(
                ("data", "mask"),
                (self.cropped_detector_data, self.mask)
        ):
            path = f"{self.pynx_phasing_dir}S{self.scan}_pynx_input_{name}.npz"
            np.savez(path, data=data)

        # Save all outputs of the preprocessing stage in a .cxi file
        dump_path = f"{self.dump_dir}/S{self.scan}_preprocessed_data.cxi"
        with CXIFile(dump_path, "w") as cxi:
            cxi.stamp()
            cxi.set_entry()

            path = cxi.create_cxi_group(
                "process",
                command="cdiutils.BcdiPipeline.preprocess()",
                comment="Pipeline preprocessing step"
            )
            cxi.softlink(f"{path}/program", "/creator")
            cxi.softlink(f"{path}/version", "/version")

            geometry_to_parse = self.converter.geometry.to_dict()
            geometry_to_parse.update({"angles": self.angles})
            geo_path = cxi.create_cxi_group("geometry", **geometry_to_parse)

            detector = {
                "description": self.params["detector_name"],
                "mask": self.mask[0],
                "calibration": self.params["det_calib_params"]
            }
            path = cxi.create_cxi_group("detector", **detector)
            cxi.softlink(f"{path}/distance", f"{path}/calibration/distance")
            cxi.softlink(f"{path}/x_pixel_size", f"{path}/calibration/pwidth2")
            cxi.softlink(f"{path}/y_pixel_size", f"{path}/calibration/pwidth1")
            cxi.softlink(f"{path}/geometry_1", geo_path)

            msg = """Raw detector data centring and cropping. The Region of
Interest is given by the roi entry. The Voxels of interest are given by the voi
entry."""
            path = cxi.create_cxi_group(
                "result", voi=self.voi, roi=self.converter.roi, description=msg
            )
            cxi.softlink(path + "/process_1", "entry_1/process_1")

            msg = """The orthogonalisation allows to make table of
correspondence between the detector pixels and their associated positions in
the reciprocal space. From this, the average d-spacing and lattice parameter
are computed. The matrix to orthogonalise the reconstructed object after phase
retrieval is also computed and will be used in the post-processing stage."""
            results = self.converter.to_dict()
            results = {k: results[k] for k in (
                "q_space_shift", "transformation_matrices",
                "direct_lab_voxel_size"
            )}
            atomic_params = self.atomic_params.copy()
            atomic_params["units"] = "angstrom"
            q_lab = self.q_lab_pos.copy()
            q_lab["units"] = "1/angstrom"
            if self.params["orthogonalise_before_phasing"]:
                qx, qy, qz = self.converter.get_xu_q_lab_regular_grid()
            else:
                qx, qy, qz = self.converter.get_q_lab_regular_grid()
            results.update({
                "atomic_parameters": atomic_params, "q_lab": q_lab,
                "description": msg, "qx_xu": qx, "qy_xu": qy, "qz_xu": qz
            })
            path = cxi.create_cxi_group("result", **results)
            cxi.softlink(path + "/process_1", "entry_1/process_1")
            exp_path = self.params["experiment_file_path"]
            cxi.create_cxi_group(
                "sample",
                "sample_name",
                sample_name=self.sample_name,
                experiment_file_path=exp_path,
                experiment_identifier=exp_path.split("/")[-1].split(".")[0]
            )
            cxi.create_cxi_group("parameters", **self.params)
            cxi.create_cxi_group(
                "source", energy=self.params["energy"], units="eV"
            )
            cxi.create_cxi_image(
                self.cropped_detector_data,
                data_type="cropped detector data",
                data_space="reciprocal",
                mask=self.mask[0],
                process_1="process_1"
            )
            cxi.create_cxi_image(
                self.orthogonalised_intensity,
                data_type="orthogonalised detector data",
                data_space="reciprocal",
                process_1="process_1"
            )
        self.logger.info(f"Pre-processed data file saved at:\n{dump_path}")

    def _make_slurm_file(self, template: str = None) -> None:
        # Make the pynx slurm file
        if template is None:
            template = (
                f"{os.path.dirname(__file__)}/pynx-id01-cdi_template.slurm"
            )
            self.logger.info(
                "Pynx slurm file template not provided, will take "
                f"the default: {template}"
            )
        else:
            self.logger.info(
                f"Pynx slurm file template provided {template = }."  # noqa, E251, E202
            )
        with open(template, "r", encoding="utf8") as file:
            source = Template(file.read())
            pynx_slurm_text = source.substitute(
                {
                    "number_of_nodes": 2,
                    "data_path": self.pynx_phasing_dir,
                    "SLURM_JOBID": "$SLURM_JOBID",
                    "SLURM_NTASKS": "$SLURM_NTASKS"
                }
            )
        with open(
                self.pynx_phasing_dir + "/pynx-id01-cdi.slurm",
                "w",
                encoding="utf8"
        ) as file:
            file.write(pynx_slurm_text)

    @Pipeline.process
    def phase_retrieval(
            self,
            jump_to_cluster: bool = False,
            pynx_slurm_file_template: str = None,
            clear_former_results: bool = False,
            cmd: str = None,
            **pynx_params
    ) -> None:
        """
        Run the phase retrieval using pynx either by submitting a job to
        a slurm cluster or by running pynx script directly on the
        current machine.

        Args:
            jump_to_cluster (bool, optional): whether a job must be
                submitted to the cluster. Defaults to False.
            pynx_slurm_file_template (str, optional): the template for
                the pynx slurm file. Defaults to None.
            clear_former_results (bool, optional): whether ti clear the
                former results. Defaults to False.
            cmd (str, optional): the command to run when running
                pynx on the current machine. Defaults to None.
            **pynx_params: additional pynx parameters.

        Raises:
            PyNXScriptError: if the pynx script fails.
            e: if the subprocess fails.
        """
        if pynx_params is not None:
            self.params["pynx"].update(pynx_params)
        if clear_former_results:
            self.logger.info("Removing former results.\n")
            files = glob.glob(self.pynx_phasing_dir + "/*Run*.cxi")
            files += glob.glob(self.pynx_phasing_dir + "/*Run*.png")
            for f in files:
                os.remove(f)
            self.phasing_results = []

        pynx_input_path = (
            self.pynx_phasing_dir + "/pynx-cdi-inputs.txt"
        )

        # Update the data and mask paths in self.params.
        for name in ("data", "mask"):
            path = f"{self.pynx_phasing_dir}S{self.scan}_pynx_input_{name}.npz"
            self.params["pynx"][name] = path

        # Make the pynx input file.
        with open(pynx_input_path, "w", encoding="utf8") as file:
            for key, value in self.params["pynx"].items():
                file.write(f"{key} = {value}\n")

        if jump_to_cluster:
            self.logger.info("Jumping to cluster requested.")
            self._make_slurm_file(pynx_slurm_file_template)
            job_id, output_file = self.submit_job(
                job_file="pynx-id01-cdi.slurm",
                working_dir=self.pynx_phasing_dir
            )
            self.monitor_job(job_id, output_file)

        else:
            self.logger.info(
                "Assuming the current machine is running PyNX. Will run the "
                "provided command."
            )
            if cmd is None:
                cmd = "pynx-cdi-id01 pynx-cdi-inputs.txt"
                self.logger.info(
                    f"No command provided. Will use the default: {cmd}"
                )
            self._run_cmd(cmd, self.pynx_phasing_dir)

    def _run_cmd(self, cmd: str, cwd: str) -> None:
        try:
            with subprocess.Popen(
                    ["bash", "-l", "-c", cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=cwd,  # Change to this directory
                    text=True,  # Ensures stdout/stderr are str, not bytes
                    # shell=True,
                    env=os.environ.copy(),
                    bufsize=1
            ) as proc:
                # Stream stdout
                for line in iter(proc.stdout.readline, ""):
                    self.logger.info(line.strip())

                # Stream stderr
                for line in iter(proc.stderr.readline, ""):
                    self.logger.error(line.strip())

                # Wait for the process to complete and check the
                # return code.
                proc.wait()
                if proc.returncode != 0:
                    self.logger.error(
                        "PyNX phasing process failed with return code "
                        f"{proc.returncode}"
                    )
                    raise PyNXScriptError
        except subprocess.CalledProcessError as e:
            # Log the error if the job submission fails
            self.logger.error(
                "Subprocess process failed with return code "
                f"{e.returncode}: {e.stderr}"
            )
            raise e

    def analyse_phasing_results(
            self,
            sorting_criterion: str = "mean_to_max",
            plot: bool = True,
            plot_phasing_results: bool = True,
            plot_phase: bool = False,
            init_analyser: bool = True
    ) -> None:
        """
        Wrapper for analyse_phasing_results method of
        PhasingResultAnalyser class.

        Analyse the phase retrieval results by sorting them according to
        the sorting_criterion, which must be selected in among:
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
            plot (bool, optional): whether or not to disable all plots.
            plot_phasing_results (bool, optional): whether to plot the
                phasing results. Defaults to True.
            plot_phase (bool, optional): whether the phase must be
                plotted. If True, will the phase is plotted with
                amplitude as opacity. If False, amplitude is plotted
                instead. Defaults to False.
            init_analyser: (bool, optional): whether to force the
                reinitialisation of the PhasingResultAnalyser instance

        Raises:
            ValueError: if sorting_criterion is unknown.
        """
        if self.result_analyser is None or init_analyser:
            self.result_analyser = PhasingResultAnalyser(
                result_dir_path=self.pynx_phasing_dir
            )

        self.result_analyser.analyse_phasing_results(
            sorting_criterion=sorting_criterion,
            plot=plot,
            plot_phasing_results=plot_phasing_results,
            plot_phase=plot_phase
        )

    def select_best_candidates(
            self,
            nb_of_best_sorted_runs: int = None,
            best_runs: list = None
    ) -> None:
        """
        A function wrapper for
        PhasingResultAnalyser.select_best_candidates.
        Select the best candidates, two methods are possible. Either
        select a specific number of runs, provided they were analysed
        and sorted beforehand. Or simply provide a list of integers
        corresponding to the digit numbers of the best runs.

        Args:
            nb_of_best_sorted_runs (int, optional): the number of best
                runs to select, provided they were analysed beforehand.
                Defaults to None.
            best_runs (list[int], optional): the best runs to select.
                Defaults to None.

        Raises:
            ValueError: If the results have not been analysed yet.
        """
        if not self.result_analyser:
            raise ValueError(
                "self.result_analyser not initialised yet. Run"
                " BcdiPipeline.analyse_phasing_results() first."
            )
        self.result_analyser.select_best_candidates(
            nb_of_best_sorted_runs,
            best_runs
        )

    @Pipeline.process
    def mode_decomposition(
            self,
            cmd: str = None,
    ) -> None:
        """
        Run the mode decomposition using PyNX pynx-cdi-analysis.py
        script as a subprocess.

        Args:
            cmd
        """
        if self.result_analyser is None:
            self.result_analyser = PhasingResultAnalyser(
                result_dir_path=self.pynx_phasing_dir
            )
        try:
            modes, mode_weights = self.result_analyser.mode_decomposition()
            self._save_pynx_results(modes=modes, mode_weights=mode_weights)

        except PynNXImportError:
            self.logger.info(
                "PyNX is not installed on the current machine. Will try to "
                "run the provided command instead."
            )
            if cmd is None:
                cmd = (
                    "pynx-cdi-analysis candidate_*.cxi --modes 1 "
                    "--modes_output mode.h5"
                )
                self.logger.info(
                    f"No command provided will use the default: {cmd}"
                )
            self._run_cmd(cmd, self.pynx_phasing_dir)
            self._save_pynx_results(self.pynx_phasing_dir + "mode.h5")

    def _save_pynx_results(
            self,
            mode_path: str = None,
            modes: list = None,
            mode_weights: list = None
    ) -> None:
        if mode_path is not None:
            with h5py.File(mode_path) as file:
                modes = file["entry_1/image_1/data"][()]
                mode_weights = file["entry_1/data_2/data"][()]
        else:
            if modes is None:
                raise ValueError("mode_path or modes required.")

        best_candidates, sorted_results, metrics = None, None, None
        if self.result_analyser is not None:
            best_candidates = self.result_analyser.best_candidates
            sorted_results = self.result_analyser.sorted_phasing_results
            metrics = self.result_analyser.metrics

        path = f"{self.dump_dir}/S{self.scan}_pynx_reconstruction_mode.cxi"
        with CXIFile(path, "w") as cxi:
            cxi.stamp()
            cxi.set_entry()
            # Copy the information in PyNX file
            path = cxi.create_cxi_group(
                "process",
                comment="PyNX phasing.",
                description="""Process information for the original CDI
reconstruction (best solution)."""
            )
            if best_candidates:
                with h5py.File(best_candidates[0], "r") as f:
                    if "/entry_last/image_1/process_1" in f:
                        for key in f["/entry_last/image_1/process_1"]:
                            f.copy(
                                f"/entry_last/image_1/process_1/{key}",
                                cxi.get_node(path),
                                name=key
                            )

            path = cxi.create_cxi_group(
                "result",
                description=f"Check results out in {self.pynx_phasing_dir}"
            )
            cxi.softlink(f"{path}/process_1", "/entry_1/process_1")

            cxi.create_cxi_group(
                "process",
                command="cdiutils.BcdiPipeline.analyse_phasing_results()",
                comment="Pipeline phase retrieval results analysis step."
            )
            path = cxi.create_cxi_group(
                "result",
                description="Sort the phasing results according to criterion.",
                sorted_results=sorted_results,
                metrics=metrics
            )
            cxi.softlink(f"{path}/process_2", "/entry_1/process_2")

            cxi.create_cxi_group(
                "process",
                command="cdiutils.BcdiPipeline.select_best_candidates()",
                comment="Best candidate selection"
            )
            path = cxi.create_cxi_group(
                "result", description="Only selected best candidates",
                best_candidates=best_candidates
            )
            cxi.softlink(f"{path}/process_3", "/entry_1/process_3")

            cxi.create_cxi_group(
                "process",
                command=(
                    "pynx-cdi-analysis / "
                    "cdiutils.pipeline.BcdiPipeline.mode_decomposition()"
                ),
                comment="Mode decomposition"
            )
            path = cxi.create_cxi_group(
                "result",
                description="The weights of each calculated mode.",
                mode_weights=mode_weights
            )
            cxi.softlink(f"{path}/process_4", "/entry_1/process_4")

            path = cxi.create_cxi_image(data=modes, process_4="process_4")
            cxi.get_node(path).attrs["description"] = "Mode decomposition"

    def _save_mode_as_h5(
            self,
            mode: np.ndarray,
            mode_weights: float,
            candidates: list[str]
    ) -> None:
        path = self.pynx_phasing_dir + "mode.h5"
        self.logger.info(f"Saving mode analysis to {path}")

        with h5py.File(path, "w") as file:
            # NeXus
            file.attrs["default"] = "entry_1"
            file.attrs["creator"] = "cdiutils"
            file.attrs["HDF5_Version"] = h5py.version.hdf5_version
            file.attrs["h5py_version"] = h5py.version.version

            entry_1 = file.create_group("entry_1")
            entry_1.create_dataset(
                "program_name", data=f"cdiutils {__version__}"
            )
            entry_1.attrs["NX_class"] = "NXentry"
            entry_1.attrs["default"] = "data_1"

            image_1 = entry_1.create_group("image_1")
            image_1.create_dataset(
                "data",
                data=mode,
                chunks=True,
                shuffle=True,
                compression="gzip"
            )
            image_1.attrs["NX_class"] = "NXdata"
            image_1.attrs["interpretation"] = "image"
            image_1["title"] = "Solutions modes"

            data_1 = file["/entry_1/image_1"]
            process_1 = data_1.create_group("process_1")
            process_1.attrs["NX_class"] = "NXprocess"
            process_1.attrs["label"] = "Mode decomposition information"
            process_1.create_dataset("program", data="cdiutils")
            process_1.create_dataset("version", data=f"{__version__}")
            process_1.create_dataset("candidates", data=candidates)
            if candidates[0].endswith(".cxi"):
                # Try to copy the original pynx process information
                with h5py.File(candidates[0]["file_name"], "r") as h0:
                    if "/entry_last/image_1/process_1" in h0:
                        h0.copy(
                            "/entry_last/image_1/process_1",
                            data_1, name="process_2"
                        )
                        data_1["process_2"].attrs["label"] = (
                            "Process information for the Original CDI "
                            "reconstruction (best solution)"
                        )

            # Add shortcut to the main data
            data_1 = entry_1.create_group("data_1")
            data_1["data"] = h5py.SoftLink("/entry_1/image_1/data")
            data_1.attrs["NX_class"] = "NXdata"
            data_1.attrs["signal"] = "data"
            data_1.attrs["interpretation"] = "image"
            data_1.attrs["label"] = "modes"
            data_1["title"] = "Solutions modes"

            # Add weights
            data_2 = entry_1.create_group("data_2")
            ds = data_2.create_dataset("data", data=mode_weights)
            ds.attrs["long_name"] = "Relative weights of modes"
            data_2.attrs["NX_class"] = "NXdata"
            data_2.attrs["signal"] = "data"
            data_2.attrs["interpretation"] = "spectrum"
            data_2.attrs["label"] = "modes relative weights"
            data_2["title"] = "Modes relative weights"

    @Pipeline.process
    def postprocess(self, **params) -> None:
        _path = f"{self.dump_dir}S{self.scan}_preprocessed_data.cxi"
        # Whether to reload the pre-processing .cxi file.
        if self.q_lab_pos.get("ref") is None or self.converter is None:
            self.logger.info(f"Loading parameters from:\n{_path}")
            self.update_from_file(_path)
        if params:
            self.logger.info(
                f"Additional parameters provided {params}, will update the "
                "current dictionary of parameters."
            )
            self.params.update(params)
            if "voxel_size" in params and params["voxel_size"] is None:
                with CXIFile(_path, "r") as cxi:
                    self.converter = self._build_converter_from_cxi(cxi)

        # Load the reconstruction mode
        _path = f"{self.dump_dir}/S{self.scan}_pynx_reconstruction_mode.cxi"
        self.reconstruction = self._load_reconstruction(_path, centre=True)

        # Handle the voxel size
        self._check_voxel_size()

        if not self.params["orthogonalise_before_phasing"]:
            self.reconstruction = self.converter.orthogonalise_to_direct_lab(
                self.reconstruction
            )
        # Change convention of the reconstruction if necessary.
        if self.params["orientation_convention"].lower() == "cxi":
            self.reconstruction = self.converter.xu_to_cxi(self.reconstruction)

        self.logger.info(
            f"Voxel size finally used is: {self.params['voxel_size']} nm in "
            f"the {self.params['orientation_convention'].upper()} convention."
        )

        # Handle flipping and apodization
        if self.params["flip"]:
            self.reconstruction = PostProcessor.flip_reconstruction(
                self.reconstruction
            )

        if self.params["apodize"]:
            self.logger.info(
                "Apodizing the complex array using "
                f"{self.params['apodize']} filter."
            )
            self.reconstruction = PostProcessor.apodize(
                self.reconstruction,
                window_type=self.params["apodize"]
            )

        # First compute the histogram of the amplitude to get an
        # isosurface estimate
        self.logger.info(
            "Finding an isosurface estimate based on the "
            "reconstructed Bragg electron density histogram:",
        )
        isosurface, _ = find_isosurface(
            np.abs(self.reconstruction),
            nbins=100,
            sigma_criterion=3,
            plot=True,  # plot and return the figure,
            save=f"{self.dump_dir}/S{self.scan}_amplitude_distribution.png"
        )
        # store the estimated isosurface
        self.extra_info["estimated_isosurface"] = isosurface
        self.logger.info(f"Isosurface estimated at {isosurface}.")

        if self.params["isosurface"] is not None:
            self.logger.info(
                "Isosurface provided by user will be used: "
                f"{self.params['isosurface']}."
            )

        elif isosurface < 0 or isosurface > 1:
            self.params["isosurface"] = self.class_isosurface
            self.logger.info(
                f"isosurface estimate has a wrong value ({isosurface}) and "
                f"will be set set to {self.class_isosurface = }."  # noqa, E251
            )
        else:
            self.params["isosurface"] = isosurface

        self.logger.info(
            "Computing the structural properties:"
            "\n\t- phase \n\t- displacement\n\t- het. (heterogeneous) strain"
            "\n\t- d-spacing\n\t- lattice parameter."
            "\nhet. strain maps are computed using various methods, either"
            " phase ramp removal or d-spacing method.\n"
            f"The theoretical Bragg peak is {self.params['hkl']}."
        )
        if self.params["handle_defects"]:
            self.logger.info("Defect handling requested.")

        if self.params["orientation_convention"].lower() == "cxi":
            g_vector = SpaceConverter.xu_to_cxi(self.q_lab_pos["ref"])
        else:
            g_vector = self.q_lab_pos["ref"]
        self.structural_props = PostProcessor.get_structural_properties(
            self.reconstruction,
            support_parameters=None,
            isosurface=self.params["isosurface"],
            g_vector=g_vector,
            hkl=self.params["hkl"],
            voxel_size=self.params["voxel_size"],
            phase_factor=-1,  # it came out of pynx.cdi
            handle_defects=self.params["handle_defects"]
        )

        to_plot = {
            k: self.structural_props[k]
            for k in ["amplitude", "phase", "displacement",
                      "het_strain", "lattice_parameter"]
        }
        # plot and save the detector data in the full detector frame and
        # final frame
        dump_file_tmpl = f"{self.dump_dir}/S{self.scan}_" + "{}.png"
        sample_scan = f"{self.sample_name}, S{self.scan}"
        self.extra_info["averaged_lattice_parameter"] = np.nanmean(
            self.structural_props["lattice_parameter"]
        )
        self.extra_info["averaged_dspacing"] = np.nanmean(
            self.structural_props["dspacing"]
        )
        table_info = {
            "Isosurface": self.params["isosurface"],
            "Averaged Lat. Par. (Å)": (
                self.extra_info["averaged_lattice_parameter"]
            ),
            "Averaged d-spacing (Å)": self.extra_info["averaged_dspacing"]
        }
        PipelinePlotter.summary_plot(
            title=f"Summary figure, {sample_scan}",
            support=self.structural_props["support"],
            table_info=table_info,
            voxel_size=self.params["voxel_size"],
            convention=self.params["orientation_convention"],
            save=dump_file_tmpl.format("summary_plot"),
            **to_plot
        )
        PipelinePlotter.summary_plot(
            title=f"Strain check figure, {sample_scan}",
            support=self.structural_props["support"],
            voxel_size=self.params["voxel_size"],
            convention=self.params["orientation_convention"],
            save=dump_file_tmpl.format("strain_methods"),
            **{
                k: self.structural_props[k]
                for k in [
                    "het_strain", "numpy_het_strain",
                    "het_strain_from_dspacing", "het_strain_with_ramp"
                ]
            }
        )

        axis_names = [r"z_{cxi}", r"y_{cxi}", r"x_{cxi}"]
        denom = (
            "du_"
            + "{" + f"{''.join([str(e) for e in self.params['hkl']])}"
            + "}"
        )
        titles = [f"${denom}/d{axis_names[i]}$" for i in range(3)]
        displacement_gradient_plots = {
            titles[i]: self.structural_props["displacement_gradient"][i]
            for i in range(3)
        }
        ptp_value = (
            np.nanmax(self.structural_props["displacement_gradient"][0])
            - np.nanmin(self.structural_props["displacement_gradient"][0])
        )
        PipelinePlotter.summary_plot(
            title=f"Shear displacement, {sample_scan}",
            support=self.structural_props["support"],
            voxel_size=self.params["voxel_size"],
            convention=self.params["orientation_convention"],
            save=dump_file_tmpl.format("shear_displacement"),
            unique_vmin=-ptp_value/2,
            unique_vmax=ptp_value/2,
            cmap=RED_TO_TEAL,
            **displacement_gradient_plots
        )
        _, _, means, fwhms = PipelinePlotter.strain_statistics(
            self.structural_props["het_strain_from_dspacing"],
            self.structural_props["support"],
            title=f"Strain statistics, {sample_scan}",
            save=dump_file_tmpl.format("strain_statistics")
        )
        self.extra_info["strain_means"] = means
        self.extra_info["strain_fwhms"] = fwhms
        plot_3d_surface_projections(
            data=self.structural_props["het_strain"],
            support=self.structural_props["support"],
            voxel_size=self.params["voxel_size"],
            cmap="cet_CET_D13",
            vmin=-np.nanmax(np.abs(self.structural_props["het_strain"])),
            vmax=np.nanmax(np.abs(self.structural_props["het_strain"])),
            cbar_title=r"Strain (%)",
            title=f"3D views of the strain, {sample_scan}",
            save=dump_file_tmpl.format("3d_strain_views")
        )

        # Load the orthogonalised peak
        path = f"{self.dump_dir}/S{self.scan}_preprocessed_data.cxi"
        with CXIFile(path, "r") as cxi:
            ortho_exp_intensity = cxi["entry_1/data_2/data"][()]
            exp_q_grid = tuple(
                cxi[f"entry_1/result_2/{k}_xu"][()]
                for k in ("qx", "qy", "qz")
            )
        # To compare, we must make sure we are back in XU convention.
        obj = (
            self.structural_props["amplitude"]
            * np.exp(-1j*self.structural_props["phase"])
        )
        voxel_size = self.params["voxel_size"]
        if self.params["orientation_convention"].lower() == "cxi":
            obj = SpaceConverter.cxi_to_xu(obj)
            voxel_size = SpaceConverter.cxi_to_xu(voxel_size)

        PipelinePlotter.plot_final_object_fft(
            obj,
            voxel_size,
            self.converter.q_space_shift,
            ortho_exp_intensity,
            exp_q_grid,
            title=f"FFT of final object vs. experimental data, {sample_scan}",
            save=f"{self.dump_dir}/S{self.scan}_final_object_fft.png"
        )

        self._save_postprocessed_data()
        self._save_parameter_file()

    def _load_reconstruction(
            self,
            path: str,
            centre: bool = False,
            isosurface: float = None
    ) -> np.ndarray:
        with CXIFile(path, "r") as cxi:
            reconstruction = cxi["entry_1/data_1/data"][0]

        self._get_oversampling_ratios(reconstruction)

        if centre:
            if isosurface is None:
                isosurface = self.class_isosurface
            amp = np.abs(reconstruction) / np.max(np.abs(reconstruction))
            support = np.where(amp >= isosurface, 1, 0)
            com = CroppingHandler.get_position(support, "com")
            reconstruction = CroppingHandler.force_centred_cropping(
                reconstruction, where=com
            )
        return reconstruction

    def _get_oversampling_ratios(self, data: np.ndarray) -> np.ndarray:
        amp = np.abs(data) / np.max(np.abs(data))

        isosurface = self.params["isosurface"]
        # if not provided, isosurface is hardcoded at this stage
        if isosurface is None:
            isosurface = self.class_isosurface
        support = np.where(amp >= isosurface, 1, 0)
        # Print the oversampling ratio
        ratios = get_oversampling_ratios(support)
        self.logger.info(
            "The oversampling ratios in each direction (original frame) are "
            + ", ".join(
                [f"axis{i}: {ratios[i]:.1f}" for i in range(len(ratios))]
            )
        )
        self.extra_info["oversampling_ratios"] = ratios

        if support.shape != tuple(self.params["preprocess_shape"]):
            self.logger.warning(
                f"Shapes before {self.params['preprocess_shape']} "
                f"and after {support.shape} Phase Retrieval are different.\n"
                "Check out PyNX parameters (ex.: auto_center_resize)."
            )

    def _check_voxel_size(self) -> None:
        if self.params["orientation_convention"].lower() == "cxi":
            self.extra_info["voxel_size_from_extent"] = (
                SpaceConverter.xu_to_cxi(self.converter.direct_lab_voxel_size)
            )  # if cxi requested, convert the voxel size from extent

        if self.params["voxel_size"] is None:
            self.params["voxel_size"] = self.converter.direct_lab_voxel_size

            # In the SpaceConverter, the convention is XU.
            if self.params["orientation_convention"].lower() == "cxi":
                self.params["voxel_size"] = SpaceConverter.xu_to_cxi(
                    self.params["voxel_size"]
                )
        else:
            # We consider voxel_size is given with the same convention
            # as the one specified in 'orientation_convention'.
            # if 1D, make it 3D
            if isinstance(
                self.params["voxel_size"],
                (float, int, np.floating, np.integer)
            ):
                self.params["voxel_size"] = tuple(np.repeat(
                    self.params["voxel_size"], self.reconstruction.ndim
                ))
            if self.params["orientation_convention"].lower() == "cxi":
                # Set the direct space interpolator voxel size with XU
                # convention.
                self.converter.direct_lab_voxel_size = (
                    SpaceConverter.cxi_to_xu(self.params["voxel_size"])
                )

    def _save_postprocessed_data(self) -> None:
        dump_path = f"{self.dump_dir}/S{self.scan}_post_processed_data.cxi"
        with CXIFile(dump_path, "w") as cxi:
            cxi.stamp()
            msg = """Post-processing of the data including:
- orthogonalisation
- isosurface estimation
- apodization
- structural properties computation"""
            path = cxi.create_cxi_group(
                "process",
                description=msg,
                comment="Data post-processing",
                command="cdiutils.BcdiPipeline.postprocess()"
            )
            cxi.softlink(f"{path}/program", "/creator")
            cxi.softlink(f"{path}/version", "/version")

            path = cxi.create_cxi_group(
                "result",
                description="Orthogonalisation procedure",
                voxel_size_from_reciprocal_space_extent=(
                    self.extra_info["voxel_size_from_extent"]
                ),
                voxel_size=self.params["voxel_size"],
                units="nm"
            )
            cxi.softlink(f"{path}/process_1", "entry_1/process_1")

            path = cxi.create_cxi_group(
                "result",
                description="Surface determination",
                estimated_isosurface=self.extra_info["estimated_isosurface"],
                used_isosurface=self.params["isosurface"],
            )
            cxi.softlink(f"{path}/process_1", "entry_1/process_1")
            path = cxi.create_cxi_group(
                "result",
                description="Oversampling estimation",
                oversampling_ratios=self.extra_info["oversampling_ratios"]
            )
            cxi.softlink(f"{path}/process_1", "entry_1/process_1")

            path = cxi.create_cxi_group(
                "result",
                description="Averaged lattice parameter and d-spacing",
                dspacing=self.extra_info["averaged_dspacing"],
                lattice_parameter=self.extra_info[
                    "averaged_lattice_parameter"
                ],
                units="angstrom"
            )
            cxi.softlink(f"{path}/process_1", "entry_1/process_1")

            path = cxi.create_cxi_group(
                "result",
                description="Strain statistics",
                strain_averages=self.extra_info["strain_means"],
                strain_fwhms=self.extra_info["strain_fwhms"],
                units="%"
            )
            cxi.softlink(f"{path}/process_1", "entry_1/process_1")

            # Copy entries from the preprocessed_data file
            prep_path = f"{self.dump_dir}/S{self.scan}_preprocessed_data.cxi"
            if os.path.isfile(prep_path):
                with CXIFile(prep_path, "r") as f:
                    for group in (
                            "detector_1", "geometry_1", "sample_1", "source_1"
                    ):
                        try:
                            f.copy(
                                f"/entry_1/{group}", cxi, f"/entry_1/{group}"
                            )
                        except KeyError:
                            print(f"Cannot found {group} in {prep_path} file.")
            # We do not copy the parameters from preprocessed_data
            # because they might have changed.
            cxi.create_cxi_group("parameters", **self.params)

            for key, data in self.structural_props.items():
                if isinstance(data, np.ndarray):
                    path = cxi.create_cxi_image(
                        data=data,
                        data_space="Direct space",
                        title=key,
                        process_1="process_1"
                    )
                    cxi.softlink(f"entry_1/{key}", path)

        self.logger.info(f"Post-processed data file saved at:\n{dump_path}")

        # Save as vti
        if IS_VTK_AVAILABLE:
            to_save_as_vti = {
                k: self.structural_props[k]
                for k in [
                    "amplitude", "support", "phase", "displacement",
                    "het_strain", "het_strain_from_dspacing",
                    "lattice_parameter", "numpy_het_strain", "dspacing"
                ]
            }

            # add the dspacing average and lattice constant average around
            # the NP to avoid nan values that are annoying for 3D
            # visualisation
            to_save_as_vti["dspacing"] = np.where(
                np.isnan(to_save_as_vti["dspacing"]),
                self.extra_info["averaged_dspacing"],
                to_save_as_vti["dspacing"]
            )
            to_save_as_vti["lattice_parameter"] = np.where(
                np.isnan(to_save_as_vti["lattice_parameter"]),
                self.extra_info["averaged_lattice_parameter"],
                to_save_as_vti["lattice_parameter"]
            )

            # save to vti file
            self.save_to_vti(
                f"{self.dump_dir}/S{self.scan}_structural_properties.vti",
                voxel_size=self.params["voxel_size"],
                cxi_convention=(
                    self.params["orientation_convention"].lower() == "cxi"
                ),
                **to_save_as_vti
            )
        else:
            self.logger.info(
                "vtk package not available, will not save the vti file."
            )

    def facet_analysis(self) -> None:
        facet_anlysis_processor = FacetAnalysisProcessor(
            self.params["facets"],
            self.params["support"]["support_method"],
            self.dump_dir,
        )
        facet_anlysis_processor.facet_analysis()
