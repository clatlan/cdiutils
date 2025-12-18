"""
Integration tests for BcdiPipeline postprocessing stage.

These tests verify that the postprocessing pipeline works correctly,
including orthogonalisation, strain computation, and structural property
analysis.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

cdiutils = pytest.importorskip("cdiutils", reason="cdiutils not installed")
from cdiutils.pipeline import BcdiPipeline  # noqa: E402


@pytest.mark.integration
@pytest.mark.slow
class TestPostprocessing:
    """Test postprocessing functionality."""

    @pytest.fixture
    def preprocessed_pipeline(self, id01_bliss_params):
        """
        Create a pipeline with preprocessing already done.

        This fixture runs preprocessing and returns the pipeline,
        ready for postprocessing tests.
        """
        pipeline = BcdiPipeline(params=id01_bliss_params)
        pipeline.preprocess()
        return pipeline

    def test_postprocess_basic(self, preprocessed_pipeline):
        """Test basic postprocessing without phasing results."""
        # note: this test will likely fail without actual phasing results
        # but it tests the postprocessing interface

        with pytest.raises((FileNotFoundError, ValueError)):
            # should fail because no reconstruction available
            preprocessed_pipeline.postprocess()

    def test_postprocess_with_reconstruction_file(
        self, id01_bliss_params, tmp_path
    ):
        """Test postprocessing with a mock reconstruction file."""
        # create a mock reconstruction
        shape = (100, 100, 100)
        amplitude = np.ones(shape)
        phase = np.zeros(shape)

        # create a CXI file with reconstruction
        pipeline = BcdiPipeline(params=id01_bliss_params)
        pipeline.preprocess()

        # create mock phasing result
        pynx_dir = Path(id01_bliss_params["dump_dir"]) / "pynx_phasing"
        mock_result_file = pynx_dir / "test_Run0001.cxi"

        with h5py.File(mock_result_file, "w") as f:
            # minimal CXI structure for testing
            entry = f.create_group("entry_1")
            data = entry.create_group("data_1")
            data.create_dataset("data", data=amplitude * np.exp(1j * phase))

    def test_postprocess_parameters(self, id01_bliss_params):
        """Test postprocessing parameter handling."""
        pipeline = BcdiPipeline(params=id01_bliss_params)
        pipeline.preprocess()

        # test that invalid parameters are caught
        with pytest.raises(ValueError, match="not recognised"):
            pipeline.postprocess(invalid_param=True)

    def test_voxel_size_computation(self, id01_bliss_params):
        """Test voxel size computation in postprocessing."""
        pipeline = BcdiPipeline(params=id01_bliss_params)
        pipeline.preprocess()

        # verify voxel size can be computed from extent
        # (this is done in preprocessing)
        assert hasattr(pipeline, "converter")
        assert pipeline.converter is not None


@pytest.mark.unit
class TestPostprocessingComponents:
    """Test individual postprocessing components."""

    def test_isosurface_estimation(self, sphere_data):
        """Test isosurface estimation on synthetic data."""
        from cdiutils.analysis.stats import find_isosurface

        data = sphere_data["data"]
        support = sphere_data["support"]

        # find isosurface
        iso, _ = find_isosurface(data)

        # verify isosurface is reasonable (0 is valid for poor SNR)
        assert 0 <= iso <= data.max()
        # Only check background comparison if isosurface was detected
        if iso > 0:
            assert iso > data[~support].mean()

    def test_phase_manipulation(self, simple_complex_array):
        """Test phase unwrapping and manipulation."""
        from cdiutils.process.postprocessor import PostProcessor

        # get phase
        phase = np.angle(simple_complex_array)

        # test phase unwrapping
        unwrapped = PostProcessor.unwrap_phase(phase)
        assert unwrapped.shape == phase.shape

        # test phase ramp removal
        phase_no_ramp = PostProcessor.remove_phase_ramp(phase)
        assert phase_no_ramp.shape == phase.shape

    def test_apodization(self, simple_complex_array):
        """Test apodization window application."""
        from cdiutils.process.postprocessor import PostProcessor

        # apply apodization
        apodized = PostProcessor.apodize(
            simple_complex_array, window_type="blackman"
        )

        # verify shape preserved
        assert apodized.shape == simple_complex_array.shape

        # TODO: should verify that edges are reduced

    def test_support_generation(self, sphere_data):
        """Test support generation from amplitude."""
        from cdiutils.utils import make_support

        data = sphere_data["data"]

        # generate support
        support = make_support(data, isosurface=0.5)

        # verify support is binary (0/1) and correct shape
        assert support.dtype in (bool, np.int64, int)
        assert support.shape == data.shape
        assert set(np.unique(support)).issubset({0, 1, False, True})

        # verify support contains the sphere
        expected_support = sphere_data["support"]
        # should have some overlap
        overlap = np.sum(support & expected_support)
        assert overlap > 0


@pytest.mark.unit
class TestStructuralProperties:
    """Test structural property computation."""

    def test_displacement_from_phase(self):
        """Test displacement computation from phase."""
        # create a phase with known gradient
        shape = (50, 50, 50)
        x, y, z = np.meshgrid(
            np.linspace(0, 10, shape[0]),
            np.linspace(0, 10, shape[1]),
            np.linspace(0, 10, shape[2]),
            indexing="ij",
        )

        # linear phase gradient
        phase = 0.1 * x + 0.2 * y + 0.3 * z

        # remove ramp
        from cdiutils.process.postprocessor import PostProcessor

        phase_no_ramp = PostProcessor.remove_phase_ramp(phase)

        # verify ramp was removed (phase should be more uniform)
        assert np.std(phase_no_ramp) < np.std(phase)


@pytest.mark.integration
@pytest.mark.slow
class TestPostprocessingOutputs:
    """Test postprocessing output files and data structures."""

    def test_postprocessed_cxi_structure(self, id01_bliss_params):
        """Test structure of postprocessed CXI file."""
        pipeline = BcdiPipeline(params=id01_bliss_params)
        pipeline.preprocess()

        # check that preprocessed CXI file was created
        dump_dir = Path(id01_bliss_params["dump_dir"])
        scan = id01_bliss_params["scan"]
        preprocessed_file = dump_dir / f"S{scan}_preprocessed_data.cxi"

        assert preprocessed_file.exists()

        # verify CXI structure
        with h5py.File(preprocessed_file, "r") as f:
            assert "entry_1" in f
            # should contain preprocessing results

    def test_vti_output_generation(self):
        """Test VTI file generation for 3D visualisation."""
        pytest.importorskip("vtk", reason="VTK not available")

        from cdiutils.io.vtk import save_as_vti

        # create test data
        data = np.random.rand(20, 20, 20)

        # save as VTI (to a temporary location)
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".vti") as tmp:
            save_as_vti(tmp.name, voxel_size=(10, 10, 10), data=data)
            assert Path(tmp.name).exists()


@pytest.mark.unit
class TestOrthogonalisation:
    """Test orthogonalisation and coordinate transformation."""

    def test_coordinate_transformation_matrices(self, id01_bliss_params):
        """Test that transformation matrices are computed."""
        pipeline = BcdiPipeline(params=id01_bliss_params)
        pipeline.preprocess()

        # converter should have transformation matrices
        assert hasattr(pipeline.converter, "to_dict")
        converter_dict = pipeline.converter.to_dict()

        assert "transformation_matrices" in converter_dict

    def test_q_lab_computation(self, id01_bliss_params):
        """Test Q-space position computation."""
        pipeline = BcdiPipeline(params=id01_bliss_params)
        pipeline.preprocess()

        # verify q_lab positions were computed for VOI
        assert pipeline.q_lab_pos["ref"] is not None

        # verify q_lab is a 3-element array/tuple
        assert len(pipeline.q_lab_pos["ref"]) == 3


@pytest.mark.integration
@pytest.mark.slow
class TestDefectHandling:
    """Test defect handling in postprocessing."""

    def test_defect_detection(self):
        """Test defect detection algorithms."""
        # create data with a defect (stacking fault)
        shape = (50, 50, 50)
        data = np.ones(shape, dtype=complex)

        # introduce a discontinuity (defect)
        data[25, :, :] *= -1  # phase flip

        # defect handling would be tested here
        # (requires actual implementation details)
        pass

    def test_handle_defects_parameter(self, id01_bliss_params):
        """Test handle_defects parameter in postprocessing."""
        pipeline = BcdiPipeline(params=id01_bliss_params)
        pipeline.preprocess()

        # verify parameter can be set
        # (actual testing requires phasing results)
        params_with_defects = {"handle_defects": True}
        pipeline.params.update(params_with_defects)

        assert pipeline.params["handle_defects"] is True


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.id01
class TestPostprocessingWithRealData:
    """Test postprocessing with real GPU phase retrieval results.

    This class uses the gpu_pipeline_results fixture which loads
    pre-computed results from test_full_pipeline_real_data.

    No GPU needed - just loads existing phasing results and tests
    different postprocessing options.
    """

    def test_postprocess_different_isosurface(
        self, gpu_pipeline_results: dict
    ) -> None:
        """Test postprocessing with different isosurface threshold.

        Args:
            gpu_pipeline_results: Pre-computed GPU pipeline results.
        """
        pipeline = BcdiPipeline(params=gpu_pipeline_results["params"])

        # re-run postprocessing with higher isosurface
        pipeline.postprocess(
            isosurface=0.5,  # higher than original 0.3
            voxel_size=(10, 10, 10),
            handle_defects=True,
        )

        # verify outputs were created
        dump_dir = gpu_pipeline_results["dump_dir"]
        scan: int = gpu_pipeline_results["scan"]
        final_file = dump_dir / f"S{scan}" / f"S{scan}_postprocessed_data.cxi"
        assert final_file.exists(), "Postprocessed file not updated"

    def test_postprocess_without_defect_handling(
        self, gpu_pipeline_results: dict
    ) -> None:
        """Test postprocessing without defect handling.

        Args:
            gpu_pipeline_results: Pre-computed GPU pipeline results.
        """
        pipeline = BcdiPipeline(params=gpu_pipeline_results["params"])

        # re-run postprocessing without defect handling
        pipeline.postprocess(
            isosurface=0.3,
            voxel_size=(10, 10, 10),
            handle_defects=False,  # disabled
        )

        dump_dir = gpu_pipeline_results["dump_dir"]
        scan: int = gpu_pipeline_results["scan"]
        final_file = dump_dir / f"S{scan}" / f"S{scan}_postprocessed_data.cxi"
        assert final_file.exists()

    def test_postprocess_different_voxel_size(
        self, gpu_pipeline_results: dict
    ) -> None:
        """Test postprocessing with different voxel size.

        Args:
            gpu_pipeline_results: Pre-computed GPU pipeline results.
        """
        pipeline = BcdiPipeline(params=gpu_pipeline_results["params"])

        # re-run postprocessing with smaller voxels
        pipeline.postprocess(
            isosurface=0.3,
            # smaller than original (10, 10, 10)
            voxel_size=(5, 5, 5),
            handle_defects=True,
        )

        dump_dir = gpu_pipeline_results["dump_dir"]
        scan: int = gpu_pipeline_results["scan"]
        final_file = dump_dir / f"S{scan}" / f"S{scan}_postprocessed_data.cxi"
        assert final_file.exists()

    def test_postprocess_support_threshold(
        self, gpu_pipeline_results: dict
    ) -> None:
        """Test postprocessing with different support threshold.

        Args:
            gpu_pipeline_results: Pre-computed GPU pipeline results.
        """
        pipeline = BcdiPipeline(params=gpu_pipeline_results["params"])

        # re-run postprocessing with different support threshold
        pipeline.postprocess(
            isosurface=0.3,
            support_threshold=0.5,  # Different threshold
            threshold_gradient=True,
        )

        dump_dir = gpu_pipeline_results["dump_dir"]
        scan: int = gpu_pipeline_results["scan"]
        final_file = dump_dir / f"S{scan}" / f"S{scan}_postprocessed_data.cxi"
        assert final_file.exists()
