"""
Full pipeline integration tests.

These tests verify the complete BCDI pipeline workflow from preprocessing
through phase retrieval to postprocessing for ID01 beamline.
GPU tests are marked separately.
"""

import os
from pathlib import Path

import pytest

from cdiutils.pipeline import BcdiPipeline


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.id01
class TestFullPipelineID01:
    """Full pipeline tests for ID01 beamline with real data."""

    def test_full_pipeline_real_data(
        self, id01_bliss_params: dict, tmp_path_factory
    ) -> None:
        """Run complete BCDI pipeline on real ID01 dislocation data.

        This test runs the full pipeline including GPU phase
        retrieval, and saves results to a location that can be
        reused by other tests.

        Workflow:
        - Preprocessing of real Bliss HDF5 data
        - PyNX phase retrieval on GPU
        - Phasing result analysis and candidate selection
        - Mode decomposition
        - Postprocessing and orthogonalisation

        Results are saved to $CI_PROJECT_DIR/gpu_test_results/ (CI)
        or to a temporary directory (local testing).

        Note: This is an expensive test that requires GPU and PyNX.

        Args:
            id01_bliss_params: Pipeline parameters for ID01 beamline.
            tmp_path_factory: Pytest fixture for creating temporary
                directories.
        """
        # use CI_PROJECT_DIR if in CI, otherwise use tmp_path
        ci_project_dir = os.environ.get("CI_PROJECT_DIR")
        if ci_project_dir:
            # in CI: save to persistent location for artifacts
            results_base = Path(ci_project_dir) / "gpu_test_results"
            results_base.mkdir(exist_ok=True)
        else:
            # local testing: use temporary directory
            results_base = tmp_path_factory.mktemp("gpu_results")

        # update params to use our controlled output directory
        params = id01_bliss_params.copy()
        params["dump_dir"] = str(results_base / "dump")
        params["save_dir"] = str(results_base / "save")

        # initialise pipeline
        pipeline = BcdiPipeline(params=params)

        # preprocessing
        pipeline.preprocess(
            preprocess_shape=(150, 150),
            voxel_reference_methods=["max", "com", "com"],
            voxel_size=12,
        )

        # verify preprocessing outputs
        dump_dir = Path(params["dump_dir"])
        scan: int = params["scan"]
        preprocessed_file = dump_dir / f"S{scan}_preprocessed_data.cxi"
        assert preprocessed_file.exists(), "Preprocessed CXI file not created"

        # phase retrieval - use environment variable for PyNX path
        pynx_bin = os.environ.get("PYNX_BIN", "")
        pynx_prefix: str = f"{pynx_bin}/" if pynx_bin else ""

        pipeline.phase_retrieval(
            cmd=f"{pynx_prefix}pynx-cdi-id01 pynx-cdi-inputs.txt",
            nb_run=20,  # more runs for real data
            nb_run_keep=10,
            clear_former_results=True,
        )

        # verify phasing results exist
        pynx_dir = dump_dir / "pynx_phasing"
        result_files = list(pynx_dir.glob("*Run*.cxi"))
        assert len(result_files) > 0, "No phasing result files created"
        assert len(result_files) <= 10, (
            f"Too many results kept: {len(result_files)}"
        )

        # analyse phasing results
        pipeline.analyse_phasing_results(
            sorting_criterion="mean_to_max",
            plot=False,  # don't create plots in CI
        )

        # select best candidates
        pipeline.select_best_candidates(nb_of_best_sorted_runs=2)

        # verify candidate selection
        candidate_files = list(pynx_dir.glob("candidate_*.cxi"))
        assert len(candidate_files) == 2, (
            f"Expected 2 candidates, got {len(candidate_files)}"
        )

        # mode decomposition
        pipeline.mode_decomposition(
            cmd=(
                f"{pynx_prefix}pynx-cdi-analysis candidate_*.cxi "
                "--modes 1 --modes_output mode.h5"
            )
        )

        # verify mode file exists
        mode_file = pynx_dir / "mode.h5"
        assert mode_file.exists(), "Mode decomposition output not created"

        # postprocessing
        pipeline.postprocess(
            isosurface=0.3,
            voxel_size=(10, 10, 10),
            handle_defects=True,
        )

        # verify final outputs
        final_file = dump_dir / f"S{scan}_postprocessed_data.cxi"
        assert final_file.exists(), "Final postprocessed file not created"

        # verify parameter file was saved
        param_file = dump_dir / f"S{scan}_parameters.yml"
        assert param_file.exists(), "Parameter file not saved"

        # save marker file with metadata for downstream tests
        marker_file = results_base / "pipeline_complete.txt"
        with open(marker_file, "w") as file:
            file.write(f"scan={scan}\\n")
            file.write(f"dump_dir={dump_dir}\\n")
            file.write(f"pynx_dir={pynx_dir}\\n")
            file.write(f"pynx_prefix={pynx_prefix}\\n")

        print("\\n=== GPU Pipeline Results Saved ===")
        print(f"Results location: {results_base}")
        print(f"Marker file: {marker_file}")
        print("These results can be reused by postprocessing tests")
        print("==================================\\n")


@pytest.mark.integration
class TestPipelineReload:
    """Test pipeline state saving and loading."""

    def test_reload_from_cxi(self, id01_bliss_params):
        """Test reloading pipeline state from CXI file."""
        # run preprocessing
        pipeline1 = BcdiPipeline(params=id01_bliss_params)
        pipeline1.preprocess()

        # create new pipeline instance
        pipeline2 = BcdiPipeline(params=id01_bliss_params)

        # load preprocessed data
        dump_dir = Path(id01_bliss_params["dump_dir"])
        scan = id01_bliss_params["scan"]
        preprocessed_file = dump_dir / f"S{scan}_preprocessed_data.cxi"

        pipeline2.update_from_file(str(preprocessed_file))

        # verify state was restored
        assert pipeline2.converter is not None
        assert pipeline2.q_lab_pos["ref"] is not None

    def test_from_file_classmethod(self, id01_bliss_params):
        """Test creating pipeline from CXI file using classmethod."""
        # run preprocessing first
        pipeline1 = BcdiPipeline(params=id01_bliss_params)
        pipeline1.preprocess()

        dump_dir = Path(id01_bliss_params["dump_dir"])
        scan = id01_bliss_params["scan"]
        preprocessed_file = dump_dir / f"S{scan}_preprocessed_data.cxi"

        # create new pipeline from file
        pipeline2 = BcdiPipeline.from_file(str(preprocessed_file))

        # verify pipeline was created
        assert pipeline2 is not None
        assert hasattr(pipeline2, "converter")


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

    def test_yaml_parameter_file(self, tmp_path, id01_bliss_params):
        """Test loading parameters from YAML file."""
        import yaml

        # create YAML parameter file
        yaml_file = tmp_path / "params.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(id01_bliss_params, f)

        # load pipeline from YAML
        pipeline = BcdiPipeline(param_file_path=str(yaml_file))

        # verify parameters were loaded
        assert pipeline.params["scan"] == id01_bliss_params["scan"]
        assert (
            pipeline.params["beamline_setup"]
            == id01_bliss_params["beamline_setup"]
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

    def test_nonexistent_data_file(self, mock_pipeline_params, tmp_path):
        """Test handling of nonexistent data files."""
        params = mock_pipeline_params.copy()
        params["experiment_file_path"] = "/nonexistent/file.nxs"
        params["dump_dir"] = str(tmp_path)

        pipeline = BcdiPipeline(params=params)

        with pytest.raises((FileNotFoundError, OSError)):
            pipeline.preprocess()
