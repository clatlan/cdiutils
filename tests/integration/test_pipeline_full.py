"""
Full pipeline integration tests.

These tests verify the complete BCDI pipeline workflow from preprocessing
through phase retrieval to postprocessing. GPU tests are marked separately.
"""

import pytest
import numpy as np
from pathlib import Path

cdiutils = pytest.importorskip("cdiutils", reason="cdiutils not installed")
from cdiutils.pipeline import BcdiPipeline  # noqa: E402


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.sixs
class TestFullPipelineSIXS:
    """Full pipeline tests for SIXS beamline."""

    def test_full_preprocessing_workflow(self, sixs_2019_params):
        """
        Test complete preprocessing workflow for SIXS.

        This is the main integration test that mirrors the manual
        testing script for SIXS 2019 data.
        """
        # initialise pipeline
        pipeline = BcdiPipeline(params=sixs_2019_params)

        # run preprocessing
        pipeline.preprocess()

        # verify all preprocessing outputs
        dump_dir = Path(sixs_2019_params["dump_dir"])
        scan = sixs_2019_params["scan"]

        # check output files exist
        assert (dump_dir / f"S{scan}_preprocessed_data.cxi").exists()
        assert (dump_dir / "pynx_phasing").exists()
        assert (
            dump_dir / "pynx_phasing" / f"S{scan}_pynx_input_data.npz"
        ).exists()
        assert (
            dump_dir / "pynx_phasing" / f"S{scan}_pynx_input_mask.npz"
        ).exists()

        # verify data structures
        assert pipeline.cropped_detector_data is not None
        assert pipeline.mask is not None
        assert pipeline.converter is not None
        assert pipeline.q_lab_pos["ref"] is not None

        # verify VOI structure
        assert "full" in pipeline.voi
        assert "cropped" in pipeline.voi


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.id01
class TestFullPipelineID01:
    """Full pipeline tests for ID01 beamline."""

    def test_full_preprocessing_workflow_spec(self, id01_core_shell_params):
        """
        Test complete preprocessing for ID01 with SPEC format.

        Uses light loading for faster execution in CI.
        """
        pipeline = BcdiPipeline(params=id01_core_shell_params)

        # run preprocessing with light loading
        pipeline.preprocess(
            preprocess_shape=(250, 250, 250),
            voxel_reference_methods=(147, 381, 348),
            light_loading=True,
        )

        # verify outputs
        assert pipeline.cropped_detector_data is not None
        assert pipeline.mask is not None

        dump_dir = Path(id01_core_shell_params["dump_dir"])
        scan = id01_core_shell_params["scan"]

        # check output files
        assert (dump_dir / f"S{scan}_preprocessed_data.cxi").exists()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.cristal
class TestFullPipelineCristal:
    """Full pipeline tests for Cristal beamline."""

    def test_full_preprocessing_workflow(self, cristal_params):
        """Test complete preprocessing for Cristal beamline."""
        pipeline = BcdiPipeline(params=cristal_params)

        # run preprocessing
        pipeline.preprocess()

        # verify outputs
        assert pipeline.cropped_detector_data is not None
        assert pipeline.mask is not None

        # verify hot pixel filter was applied
        if cristal_params["hot_pixel_filter"]:
            # check no extreme values
            assert np.isfinite(pipeline.cropped_detector_data).all()


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.requires_pynx
class TestPipelineWithPhaseRetrieval:
    """
    Tests that require PyNX and GPU for phase retrieval.

    These tests are skipped in standard CI and run only on
    GPU-enabled machines (e.g., ESRF GitLab CI).
    """

    def test_phase_retrieval_setup(self, sixs_2019_params):
        """Test phase retrieval configuration (without running)."""
        pytest.importorskip("pynx", reason="PyNX not installed")

        pipeline = BcdiPipeline(params=sixs_2019_params)
        pipeline.preprocess()

        # verify PyNX input file creation (without running phase retrieval)
        pynx_dir = Path(sixs_2019_params["dump_dir"]) / "pynx_phasing"

        # check that data and mask files exist
        scan = sixs_2019_params["scan"]
        assert (pynx_dir / f"S{scan}_pynx_input_data.npz").exists()
        assert (pynx_dir / f"S{scan}_pynx_input_mask.npz").exists()

    @pytest.mark.skipif(
        "not config.getoption('--run-gpu-tests', default=False)",
        reason="GPU tests disabled by default",
    )
    def test_full_pipeline_with_phasing(self, sixs_2019_params):
        """
        Test full pipeline including phase retrieval.

        This test is VERY slow and requires GPU + PyNX.
        Only run on ESRF GitLab CI or with --run-gpu-tests flag.
        """
        pytest.importorskip("pynx", reason="PyNX not installed")

        pipeline = BcdiPipeline(params=sixs_2019_params)

        # preprocessing
        pipeline.preprocess()

        # phase retrieval with minimal iterations for testing
        pipeline.phase_retrieval(
            nb_run=2,  # minimal runs for testing
            nb_raar=10,  # minimal iterations
            nb_hio=10,
            nb_er=10,
        )

        # verify phasing results were created
        pynx_dir = Path(sixs_2019_params["dump_dir"]) / "pynx_phasing"
        result_files = list(pynx_dir.glob("*Run*.cxi"))
        assert len(result_files) > 0

        # analyse results
        pipeline.analyse_phasing_results()

        # verify analysis outputs
        assert hasattr(pipeline, "result_analyser")
        assert pipeline.result_analyser is not None


@pytest.mark.integration
class TestPipelineReload:
    """Test pipeline state saving and loading."""

    def test_reload_from_cxi(self, sixs_2019_params):
        """Test reloading pipeline state from CXI file."""
        # run preprocessing
        pipeline1 = BcdiPipeline(params=sixs_2019_params)
        pipeline1.preprocess()

        # create new pipeline instance
        pipeline2 = BcdiPipeline(params=sixs_2019_params)

        # load preprocessed data
        dump_dir = Path(sixs_2019_params["dump_dir"])
        scan = sixs_2019_params["scan"]
        preprocessed_file = dump_dir / f"S{scan}_preprocessed_data.cxi"

        pipeline2.update_from_file(str(preprocessed_file))

        # verify state was restored
        assert pipeline2.converter is not None
        assert pipeline2.q_lab_pos["ref"] is not None

    def test_from_file_classmethod(self, sixs_2019_params):
        """Test creating pipeline from CXI file using classmethod."""
        # run preprocessing first
        pipeline1 = BcdiPipeline(params=sixs_2019_params)
        pipeline1.preprocess()

        dump_dir = Path(sixs_2019_params["dump_dir"])
        scan = sixs_2019_params["scan"]
        preprocessed_file = dump_dir / f"S{scan}_preprocessed_data.cxi"

        # create new pipeline from file
        pipeline2 = BcdiPipeline.from_file(str(preprocessed_file))

        # verify pipeline was created
        assert pipeline2 is not None
        assert hasattr(pipeline2, "converter")


@pytest.mark.integration
@pytest.mark.slow
class TestMultiBeamlinePipeline:
    """
    Test pipeline works across multiple beamlines.

    This ensures the pipeline is robust to different data formats.
    """

    @pytest.mark.parametrize(
        "beamline_fixture",
        [
            "sixs_2019_params",
            "cristal_params",
        ],
    )
    def test_preprocessing_multiple_beamlines(self, beamline_fixture, request):
        """Test preprocessing works for different beamlines."""
        params = request.getfixturevalue(beamline_fixture)

        pipeline = BcdiPipeline(params=params)

        # run preprocessing (might use light loading for some)
        if "id01" in beamline_fixture:
            pipeline.preprocess(light_loading=True)
        else:
            pipeline.preprocess()

        # verify basic outputs
        assert pipeline.cropped_detector_data is not None
        assert pipeline.mask is not None


@pytest.mark.unit
class TestPipelineParameterManagement:
    """Test parameter validation and management."""

    def test_parameter_validation(self, mock_pipeline_params):
        """Test parameter validation on initialization."""
        # create pipeline with minimal params
        pipeline = BcdiPipeline(params=mock_pipeline_params)

        # verify params were processed
        assert pipeline.params is not None
        assert "beamline_setup" in pipeline.params
        assert "scan" in pipeline.params

    def test_parameter_update(self, mock_pipeline_params):
        """Test parameter updates during pipeline execution."""
        pipeline = BcdiPipeline(params=mock_pipeline_params)

        # update parameters
        new_params = {"preprocess_shape": (200, 200, 200)}
        pipeline.params.update(new_params)

        # verify update
        assert pipeline.params["preprocess_shape"] == (200, 200, 200)

    def test_yaml_parameter_file(self, temp_output_dir, sixs_2019_params):
        """Test loading parameters from YAML file."""
        import yaml

        # create YAML parameter file
        yaml_file = temp_output_dir / "params.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(sixs_2019_params, f)

        # load pipeline from YAML
        pipeline = BcdiPipeline(param_file_path=str(yaml_file))

        # verify parameters were loaded
        assert pipeline.params["scan"] == sixs_2019_params["scan"]
        assert (
            pipeline.params["beamline_setup"]
            == sixs_2019_params["beamline_setup"]
        )


@pytest.mark.integration
class TestPipelineErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_required_parameters(self):
        """Test that missing required parameters raise errors."""
        # create pipeline with incomplete params
        with pytest.raises((ValueError, KeyError)):
            BcdiPipeline(params={})

    def test_invalid_beamline_setup(self):
        """Test handling of invalid beamline setup."""
        params = {
            "beamline_setup": "invalid_beamline",
            "scan": 1,
            "sample_name": "test",
        }

        # should either raise error or handle gracefully
        with pytest.raises((ValueError, KeyError)):
            pipeline = BcdiPipeline(params=params)
            pipeline.preprocess()

    def test_nonexistent_data_file(
        self, mock_pipeline_params, temp_output_dir
    ):
        """Test handling of nonexistent data files."""
        params = mock_pipeline_params.copy()
        params["experiment_file_path"] = "/nonexistent/file.nxs"
        params["dump_dir"] = str(temp_output_dir)

        pipeline = BcdiPipeline(params=params)

        with pytest.raises((FileNotFoundError, OSError)):
            pipeline.preprocess()


def pytest_addoption(parser):
    """Add custom command line options for GPU tests."""
    parser.addoption(
        "--run-gpu-tests",
        action="store_true",
        default=False,
        help="Run GPU-intensive tests (requires PyNX and GPU)",
    )
