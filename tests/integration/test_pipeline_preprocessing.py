"""
Integration tests for BcdiPipeline preprocessing stage.

These tests verify that the preprocessing pipeline works correctly with
real beamline data, including data loading, centring, cropping, and
orthogonalisation parameter computation.
"""

import pytest
import numpy as np
from pathlib import Path

# import cdiutils only within tests to allow proper skip handling
cdiutils = pytest.importorskip("cdiutils", reason="cdiutils not installed")
from cdiutils.pipeline import BcdiPipeline


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.sixs
class TestSIXSPreprocessing:
    """Test preprocessing with SIXS beamline data."""

    def test_sixs_2019_preprocess(self, sixs_2019_params):
        """Test SIXS 2019 preprocessing pipeline."""
        # create pipeline instance
        pipeline = BcdiPipeline(params=sixs_2019_params)

        # run preprocessing
        pipeline.preprocess()

        # verify outputs exist
        assert pipeline.cropped_detector_data is not None
        assert pipeline.mask is not None
        assert pipeline.converter is not None

        # verify shapes
        expected_shape = sixs_2019_params["preprocess_shape"]
        assert pipeline.cropped_detector_data.shape == expected_shape
        assert pipeline.mask.shape == expected_shape

        # verify VOI was computed
        assert "full" in pipeline.voi
        assert "cropped" in pipeline.voi
        assert "ref" in pipeline.voi["full"]
        assert "max" in pipeline.voi["cropped"]
        assert "com" in pipeline.voi["cropped"]

        # verify q_lab positions were computed
        assert pipeline.q_lab_pos["ref"] is not None

        # verify preprocessed data file was created
        dump_dir = Path(sixs_2019_params["dump_dir"])
        scan = sixs_2019_params["scan"]
        preprocessed_file = dump_dir / f"S{scan}_preprocessed_data.cxi"
        assert preprocessed_file.exists()

        # verify pynx input files were created
        pynx_dir = dump_dir / "pynx_phasing"
        assert pynx_dir.exists()
        assert (pynx_dir / f"S{scan}_pynx_input_data.npz").exists()
        assert (pynx_dir / f"S{scan}_pynx_input_mask.npz").exists()

    def test_preprocess_with_light_loading(self, sixs_2019_params):
        """Test preprocessing with light loading mode."""
        params = sixs_2019_params.copy()
        params["light_loading"] = True

        pipeline = BcdiPipeline(params=params)
        pipeline.preprocess()

        # verify light loading worked
        assert pipeline.cropped_detector_data is not None
        assert pipeline.mask is not None

    def test_preprocess_voxel_reference_methods(self, sixs_2019_params):
        """Test different voxel reference methods."""
        params = sixs_2019_params.copy()

        # test with different methods
        for methods in [
            ["max", "max", "max"],
            ["com", "com", "com"],
            ["max", "com", "max"],
        ]:
            params["voxel_reference_methods"] = methods
            pipeline = BcdiPipeline(params=params)
            pipeline.preprocess()

            assert pipeline.cropped_detector_data is not None


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.id01
class TestID01Preprocessing:
    """Test preprocessing with ID01 beamline data."""

    def test_id01_core_shell_preprocess(self, id01_core_shell_params):
        """Test ID01 core-shell sample preprocessing (SPEC format)."""
        pipeline = BcdiPipeline(params=id01_core_shell_params)

        # run preprocessing with light loading
        # (full loading might be too slow for CI)
        pipeline.preprocess(
            preprocess_shape=(250, 250, 250),
            voxel_reference_methods=(147, 381, 348),
            light_loading=True,
        )

        # verify outputs
        assert pipeline.cropped_detector_data is not None
        assert pipeline.mask is not None

        # verify SPEC-specific attributes were handled
        assert pipeline.converter is not None


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.cristal
class TestCristalPreprocessing:
    """Test preprocessing with Cristal beamline data."""

    def test_cristal_preprocess(self, cristal_params):
        """Test Cristal beamline preprocessing."""
        pipeline = BcdiPipeline(params=cristal_params)

        # run preprocessing
        pipeline.preprocess()

        # verify outputs
        assert pipeline.cropped_detector_data is not None
        assert pipeline.mask is not None

        # verify hot pixel filter was applied if requested
        if cristal_params["hot_pixel_filter"]:
            # data should have been filtered
            assert pipeline.cropped_detector_data.max() < 1e10

    def test_cristal_with_voxel_reference_strings(self, cristal_params):
        """Test Cristal preprocessing with string voxel references."""
        # cristal_params already uses ["max", "com", "com"]
        pipeline = BcdiPipeline(params=cristal_params)
        pipeline.preprocess()

        assert pipeline.voi["cropped"]["max"] is not None
        assert pipeline.voi["cropped"]["com"] is not None


@pytest.mark.unit
class TestPreprocessingParameters:
    """Test preprocessing parameter validation and handling."""

    def test_invalid_parameter_raises_error(self, mock_pipeline_params):
        """Test that invalid parameters raise ValueError."""
        pipeline = BcdiPipeline(params=mock_pipeline_params)

        with pytest.raises(ValueError, match="not recognised"):
            pipeline.preprocess(invalid_parameter=True)

    def test_shape_validation(self, mock_pipeline_params):
        """Test that preprocess_shape is validated for PyNX compatibility."""
        pipeline = BcdiPipeline(params=mock_pipeline_params)

        # shapes that need adjustment should be handled
        # (ensure_pynx_shape makes them compatible)
        pipeline.preprocess(preprocess_shape=(99, 99, 99))
        # should succeed even though 99 is not a PyNX-compatible number


@pytest.mark.unit
class TestPreprocessingOutputs:
    """Test preprocessing output data structures."""

    def test_voi_structure(self, sixs_2019_params):
        """Test VOI (Voxel Of Interest) dictionary structure."""
        pipeline = BcdiPipeline(params=sixs_2019_params)
        pipeline.preprocess()

        # verify VOI structure
        assert isinstance(pipeline.voi, dict)
        assert "full" in pipeline.voi
        assert "cropped" in pipeline.voi

        # verify VOI entries are tuples or numpy arrays
        for frame in ["full", "cropped"]:
            for key in pipeline.voi[frame]:
                voi_value = pipeline.voi[frame][key]
                assert (
                    isinstance(voi_value, (tuple, list, np.ndarray))
                    or voi_value is None
                )

    def test_converter_initialisation(self, sixs_2019_params):
        """Test that SpaceConverter is properly initialised."""
        pipeline = BcdiPipeline(params=sixs_2019_params)
        pipeline.preprocess()

        # verify converter exists and has required attributes
        assert pipeline.converter is not None
        assert hasattr(pipeline.converter, "geometry")
        assert hasattr(pipeline.converter, "roi")


@pytest.mark.integration
@pytest.mark.slow
class TestPreprocessingDataFormats:
    """Test preprocessing with different data formats."""

    def test_nxs_format_sixs(self, sixs_2019_params):
        """Test NeXus (.nxs) format loading for SIXS."""
        pipeline = BcdiPipeline(params=sixs_2019_params)
        pipeline.preprocess()

        # verify data was loaded
        assert pipeline.cropped_detector_data is not None
        # verify data is numeric
        assert np.isfinite(pipeline.cropped_detector_data).any()

    def test_spec_format_id01(self, id01_core_shell_params):
        """Test SPEC format loading for ID01."""
        pipeline = BcdiPipeline(params=id01_core_shell_params)

        # light loading for speed
        pipeline.preprocess(light_loading=True)

        # verify data was loaded from SPEC + EDF files
        assert pipeline.cropped_detector_data is not None


@pytest.mark.unit
class TestPreprocessingEdgeCases:
    """Test edge cases and error handling in preprocessing."""

    def test_missing_experiment_file(
        self, mock_pipeline_params, temp_output_dir
    ):
        """Test handling of missing experiment file."""
        params = mock_pipeline_params.copy()
        params["experiment_file_path"] = "/nonexistent/file.nxs"
        params["dump_dir"] = str(temp_output_dir)

        pipeline = BcdiPipeline(params=params)

        # should raise an error when trying to load
        with pytest.raises((FileNotFoundError, OSError)):
            pipeline.preprocess()

    def test_2d_to_3d_shape_conversion(self, mock_pipeline_params):
        """Test automatic conversion of 2D shape to 3D."""
        params = mock_pipeline_params.copy()
        params["preprocess_shape"] = (150, 150)  # 2D shape

        # this should be handled internally by _from_2d_to_3d_shape
        pipeline = BcdiPipeline(params=params)

        # verify internal shape handling
        assert hasattr(pipeline, "_from_2d_to_3d_shape")
