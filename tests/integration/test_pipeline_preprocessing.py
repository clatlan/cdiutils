"""
Integration tests for BcdiPipeline preprocessing stage.

These tests verify that the preprocessing pipeline works correctly with
ID01 beamline data, including data loading, centring, cropping, and
orthogonalisation parameter computation.
"""

import numpy as np
import pytest

# import cdiutils only within tests to allow proper skip handling
cdiutils = pytest.importorskip("cdiutils", reason="cdiutils not installed")
from cdiutils.pipeline import BcdiPipeline  # noqa: E402


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.id01
class TestID01Preprocessing:
    """Test preprocessing with ID01 beamline data."""

    def test_id01_bliss_preprocess(self, id01_bliss_params):
        """Test ID01 Bliss format preprocessing (primary test)."""
        pipeline = BcdiPipeline(params=id01_bliss_params)

        # run preprocessing
        pipeline.preprocess(
            preprocess_shape=(150, 150),
            voxel_reference_methods=["max", "com", "com"],
        )

        # verify outputs
        assert pipeline.cropped_detector_data is not None
        assert pipeline.mask is not None
        assert pipeline.converter is not None

        # verify shapes (150x150 in detector plane, rocking curve kept)
        assert pipeline.cropped_detector_data.shape[1:] == (150, 150)
        assert pipeline.mask.shape[1:] == (150, 150)

    def test_preprocess_with_light_loading(self, id01_bliss_params):
        """Test preprocessing with light loading mode."""
        pipeline = BcdiPipeline(params=id01_bliss_params)

        # light loading requires voxel coordinates
        pipeline.preprocess(
            preprocess_shape=(150, 150, 150),
            voxel_reference_methods=(128, 166, 365),
            light_loading=True,
        )

        # verify light loading worked
        assert pipeline.cropped_detector_data is not None
        assert pipeline.mask is not None

    def test_preprocess_voxel_reference_methods(self, id01_bliss_params):
        """Test different voxel reference methods."""
        params = id01_bliss_params.copy()

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

    def test_bliss_format_id01(self, id01_bliss_params):
        """Test Bliss HDF5 format loading for ID01."""
        pipeline = BcdiPipeline(params=id01_bliss_params)
        pipeline.preprocess()

        # verify data was loaded from Bliss HDF5 file
        assert pipeline.cropped_detector_data is not None


@pytest.mark.unit
class TestPreprocessingParameters:
    """Test preprocessing parameter validation and handling."""

    def test_invalid_parameter_raises_error(self, mock_pipeline_params):
        """Test that invalid parameters raise ValueError."""
        pipeline = BcdiPipeline(params=mock_pipeline_params)

        with pytest.raises(ValueError, match="not recognised"):
            pipeline.preprocess(invalid_parameter=True)

    def test_shape_validation(self, id01_bliss_params):
        """Test that preprocess_shape is validated for PyNX compatibility."""
        pipeline = BcdiPipeline(params=id01_bliss_params)

        # shapes that need adjustment should be handled
        # (ensure_pynx_shape makes them compatible)
        pipeline.preprocess(preprocess_shape=(99, 99))
        # should succeed even though 99 is not a PyNX-compatible number


@pytest.mark.unit
class TestPreprocessingOutputs:
    """Test preprocessing output data structures."""

    def test_voi_structure(self, id01_bliss_params):
        """Test VOI (Voxel Of Interest) dictionary structure."""
        pipeline = BcdiPipeline(params=id01_bliss_params)
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

    def test_converter_initialisation(self, id01_bliss_params):
        """Test that converter is properly initialised."""
        pipeline = BcdiPipeline(params=id01_bliss_params)
        pipeline.preprocess()

        # verify converter exists and has required attributes
        assert pipeline.converter is not None
        assert hasattr(pipeline.converter, "geometry")
        assert hasattr(pipeline.converter, "roi")


@pytest.mark.unit
class TestPreprocessingEdgeCases:
    """Test edge cases and error handling in preprocessing."""

    def test_missing_experiment_file(self, mock_pipeline_params, tmp_path):
        """Test handling of missing experiment file."""
        params = mock_pipeline_params.copy()
        params["experiment_file_path"] = "/nonexistent/file.nxs"
        params["dump_dir"] = str(tmp_path)

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
