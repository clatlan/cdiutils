"""
Integration tests using simulated ID01 data.

These tests verify the complete BCDI pipeline using simulated detector
data generated from synthetic objects. This approach allows testing the
full pipeline (preprocessing, phasing, and postprocessing) with known
ground truth.
"""

import os

import numpy as np
import pytest

cdiutils = pytest.importorskip("cdiutils", reason="cdiutils not installed")
from cdiutils.pipeline import BcdiPipeline  # noqa: E402


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.simulation
class TestPipelineWithSimulatedData:
    """
    Test BcdiPipeline using simulated ID01 data.

    This test class uses the BCDISimulator to create realistic
    detector data from a synthetic object, then runs the full pipeline
    to validate preprocessing, phasing, and postprocessing stages.
    """

    def test_preprocessing_with_simulated_data(
        self,
        simulated_id01_data,
    ):
        """
        Test preprocessing stage with simulated ID01 data.

        This test verifies that the pipeline can successfully load and
        preprocess simulated detector data, including:
        - Data loading from HDF5 structure
        - Detector calibration
        - Cropping and centring
        - Q-space coordinate computation

        Args:
            simulated_id01_data: fixture providing simulated
                experimental data and metadata.
        """
        # extract simulation metadata
        experiment_file = simulated_id01_data["experiment_file"]
        sample_name = simulated_id01_data["sample_name"]
        scan_number = simulated_id01_data["scan_number"]
        dump_dir = simulated_id01_data["dump_dir"]

        # create pipeline parameters
        params = {
            "beamline_setup": "id01",
            "experiment_file_path": str(experiment_file),
            "sample_name": sample_name,
            "scan": scan_number,
            "dump_dir": str(dump_dir),
        }

        # initialise pipeline
        pipeline = BcdiPipeline(params=params)

        # run preprocessing
        pipeline.preprocess(
            preprocess_shape=(200, 200, 200),
            voxel_reference_methods=["max", "com", "com"],
            hot_pixel_filter=False,
            background_level=1,
        )

        # verify preprocessing outputs
        assert pipeline.cropped_detector_data is not None
        assert pipeline.mask is not None
        assert pipeline.converter is not None

        # verify shapes
        assert pipeline.cropped_detector_data.shape == (200, 200, 200)
        assert pipeline.mask.shape == (200, 200, 200)

        # verify VOI was computed
        assert "full" in pipeline.voi
        assert "cropped" in pipeline.voi

        # verify q_lab positions were computed
        assert pipeline.q_lab_pos["ref"] is not None

        # verify preprocessed data file was created
        preprocessed_file = dump_dir / f"S{scan_number}_preprocessed_data.cxi"
        assert preprocessed_file.exists()

        # verify pynx input files were created
        pynx_dir = dump_dir / "pynx_phasing"
        assert pynx_dir.exists()
        assert (pynx_dir / f"S{scan_number}_pynx_input_data.npz").exists()
        assert (pynx_dir / f"S{scan_number}_pynx_input_mask.npz").exists()

    @pytest.mark.gpu
    def test_full_pipeline_with_simulated_data(
        self,
        simulated_id01_data,
    ):
        """
        Test complete pipeline including phase retrieval.

        This test runs the full BCDI pipeline from preprocessing
        through phase retrieval to postprocessing. It requires GPU
        and PyNX to be available.

        Note: This is a slow test and should typically be run
        separately from the main test suite.

        Args:
            simulated_id01_data: fixture providing simulated
                experimental data and metadata.
        """

        # extract simulation metadata
        experiment_file = simulated_id01_data["experiment_file"]
        sample_name = simulated_id01_data["sample_name"]
        scan_number = simulated_id01_data["scan_number"]
        dump_dir = simulated_id01_data["dump_dir"]

        # create pipeline parameters
        params = {
            "beamline_setup": "id01",
            "experiment_file_path": str(experiment_file),
            "sample_name": sample_name,
            "scan": scan_number,
            "dump_dir": str(dump_dir),
        }

        # initialise pipeline
        pipeline = BcdiPipeline(params=params)

        # run preprocessing
        pipeline.preprocess(
            preprocess_shape=(200, 200, 200),
            voxel_reference_methods=["max", "com", "com"],
            hot_pixel_filter=False,
            background_level=1,
        )

        # run phase retrieval (reduced number of runs for testing)
        # use PYNX_BIN env var if set to construct full paths
        pynx_bin = os.environ.get("PYNX_BIN", "")
        pynx_prefix = f"{pynx_bin}/" if pynx_bin else ""
        pipeline.phase_retrieval(
            cmd=f"{pynx_prefix}pynx-cdi-id01 pynx-cdi-inputs.txt",
            nb_run=10,
            nb_run_keep=5,
            clear_former_results=True,
        )

        # analyse phasing results
        pipeline.analyse_phasing_results(
            sorting_criterion="mean_to_max",
            plot=False,
        )

        # select best candidates
        pipeline.select_best_candidates(nb_of_best_sorted_runs=3)

        # mode decomposition
        pipeline.mode_decomposition(
            cmd=f"{pynx_prefix}pynx-cdi-analysis candidate_*.cxi --modes 1 --modes_output mode.h5"
        )

        # postprocess
        pipeline.postprocess(
            isosurface=0.25,
            flip=False,
            handle_defects=False,
            apodize=False,
        )

        # verify final results exist
        assert pipeline.structural_props is not None
        assert "amplitude" in pipeline.structural_props
        assert "phase" in pipeline.structural_props

        # verify postprocessed data file was created
        postprocessed_file = (
            dump_dir / f"S{scan_number}_postprocessed_data.cxi"
        )
        assert postprocessed_file.exists()

    def test_preprocessing_parameter_validation(
        self,
        simulated_id01_data,
    ):
        """
        Test that preprocessing validates parameters correctly.

        Args:
            simulated_id01_data: fixture providing simulated
                experimental data and metadata.
        """
        experiment_file = simulated_id01_data["experiment_file"]
        sample_name = simulated_id01_data["sample_name"]
        scan_number = simulated_id01_data["scan_number"]
        dump_dir = simulated_id01_data["dump_dir"]

        params = {
            "beamline_setup": "id01",
            "experiment_file_path": str(experiment_file),
            "sample_name": sample_name,
            "scan": scan_number,
            "dump_dir": str(dump_dir),
        }

        pipeline = BcdiPipeline(params=params)

        # test invalid voxel_reference_methods
        with pytest.raises(ValueError):
            pipeline.preprocess(
                preprocess_shape=(200, 200, 200),
                voxel_reference_methods=["invalid", "com", "com"],
            )


@pytest.mark.integration
@pytest.mark.simulation
class TestSimulationDataGeneration:
    """
    Test the simulation data generation utilities.

    These tests verify that the ID01 simulation functions correctly
    create the expected HDF5 file structure and can be loaded by
    ID01Loader.
    """

    def test_simulated_data_structure(self, simulated_id01_data):
        """
        Verify simulated data has correct HDF5 structure.

        Args:
            simulated_id01_data: fixture providing simulated
                experimental data and metadata.
        """
        experiment_file = simulated_id01_data["experiment_file"]
        sample_name = simulated_id01_data["sample_name"]
        scan_number = simulated_id01_data["scan_number"]

        # verify master file exists
        assert experiment_file.exists()

        # verify sample directory structure
        sample_dir = experiment_file.parent / sample_name
        assert sample_dir.exists()

        # verify scan folder
        dataset_dir = sample_dir / sample_name
        assert dataset_dir.exists()

        scan_dir = dataset_dir / f"scan{scan_number:04d}"
        assert scan_dir.exists()

    def test_simulated_data_loading(self, simulated_id01_data):
        """
        Test that simulated data can be loaded by ID01Loader.

        Args:
            simulated_id01_data: fixture providing simulated
                experimental data and metadata.
        """
        from cdiutils.io.id01 import ID01Loader

        experiment_file = simulated_id01_data["experiment_file"]
        sample_name = simulated_id01_data["sample_name"]
        scan_number = simulated_id01_data["scan_number"]
        detector_name = simulated_id01_data["detector_name"]

        # create loader
        loader = ID01Loader(
            experiment_file_path=str(experiment_file),
            sample_name=sample_name,
            scan=scan_number,
            detector_name=detector_name,
        )

        # load detector data
        detector_data = loader.load_detector_data()

        # verify data shape and type
        assert detector_data is not None
        assert isinstance(detector_data, np.ndarray)
        assert detector_data.ndim == 3

        # verify data has expected number of frames
        expected_num_frames = simulated_id01_data["num_frames"]
        assert detector_data.shape[0] == expected_num_frames

        # verify detector shape
        expected_detector_shape = simulated_id01_data["detector_shape"]
        assert detector_data.shape[1:] == expected_detector_shape

    def test_motor_positions_loading(self, simulated_id01_data):
        """
        Test that motor positions can be loaded from simulated data.

        Args:
            simulated_id01_data: fixture providing simulated
                experimental data and metadata.
        """
        from cdiutils.io.id01 import ID01Loader

        experiment_file = simulated_id01_data["experiment_file"]
        sample_name = simulated_id01_data["sample_name"]
        scan_number = simulated_id01_data["scan_number"]
        detector_name = simulated_id01_data["detector_name"]

        loader = ID01Loader(
            experiment_file_path=str(experiment_file),
            sample_name=sample_name,
            scan=scan_number,
            detector_name=detector_name,
        )

        # load motor positions
        motor_positions = loader.load_motor_positions()

        # verify motor positions dictionary
        assert motor_positions is not None
        assert isinstance(motor_positions, dict)

        # verify expected motor names are present (standardized names)
        expected_motors = [
            "sample_outofplane_angle",
            "sample_inplane_angle",
            "detector_outofplane_angle",
            "detector_inplane_angle",
        ]
        for motor in expected_motors:
            assert motor in motor_positions

    def test_detector_calibration_loading(
        self,
        simulated_id01_data,
    ):
        """
        Test that detector calibration can be loaded from simulated
        data.

        Args:
            simulated_id01_data: fixture providing simulated
                experimental data and metadata.
        """
        from cdiutils.io.id01 import ID01Loader

        experiment_file = simulated_id01_data["experiment_file"]
        sample_name = simulated_id01_data["sample_name"]
        scan_number = simulated_id01_data["scan_number"]
        detector_name = simulated_id01_data["detector_name"]

        loader = ID01Loader(
            experiment_file_path=str(experiment_file),
            sample_name=sample_name,
            scan=scan_number,
            detector_name=detector_name,
        )

        # load detector calibration parameters
        det_calib = loader.load_det_calib_params()

        # verify calibration parameters
        assert det_calib is not None
        assert isinstance(det_calib, dict)

        # verify expected calibration parameters are present
        expected_params = [
            "cch1",
            "cch2",
            "pwidth1",
            "pwidth2",
            "distance",
        ]
        for param in expected_params:
            assert param in det_calib
