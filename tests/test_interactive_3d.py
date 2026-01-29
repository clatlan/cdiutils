"""
Tests for interactive 3D visualisation tools.

This module tests the ThreeDViewer and plot_3d_isosurface functionality,
including widget creation, data handling, and NaN management.
"""

import numpy as np
import pytest

# check if interactive dependencies are available
pytest.importorskip("plotly", reason="plotly not installed")
pytest.importorskip("skimage", reason="scikit-image not installed")
pytest.importorskip("scipy", reason="scipy not installed")
pytest.importorskip("ipywidgets", reason="ipywidgets not installed")

from cdiutils.interactive import ThreeDViewer, plot_3d_isosurface  # noqa: E402
from cdiutils.interactive.volume import (  # noqa: E402
    _extract_isosurface_with_values,
    colorcet_to_plotly,
)


class TestThreeDViewer:
    """Test suite for the ThreeDViewer class."""

    @pytest.fixture
    def complex_data(self):
        """Create a simple 3D complex array for testing."""
        shape = (20, 20, 20)
        amplitude = np.ones(shape)
        # create a sphere in the centre
        x, y, z = np.ogrid[-10:10, -10:10, -10:10]
        mask = x**2 + y**2 + z**2 <= 7**2
        amplitude[mask] = 2.0
        phase = np.random.uniform(-np.pi, np.pi, shape)
        return amplitude * np.exp(1j * phase)

    @pytest.fixture
    def complex_data_with_nan(self):
        """Create complex data with NaN values."""
        shape = (20, 20, 20)
        amplitude = np.ones(shape)
        x, y, z = np.ogrid[-10:10, -10:10, -10:10]
        mask = x**2 + y**2 + z**2 <= 7**2
        amplitude[mask] = 2.0
        # introduce some NaN values
        amplitude[5:8, 5:8, 5:8] = np.nan
        phase = np.random.uniform(-np.pi, np.pi, shape)
        phase[10:12, 10:12, 10:12] = np.nan
        return amplitude * np.exp(1j * phase)

    def test_viewer_initialisation(self, complex_data):
        """Test that ThreeDViewer initialises correctly."""
        viewer = ThreeDViewer(complex_data)

        # check that the viewer was created
        assert viewer is not None
        assert viewer.data is not None

        # check that widgets were created
        assert viewer.threshold is not None
        assert viewer.toggle_phase is not None
        assert viewer.colormap is not None
        assert viewer.auto_scale is not None
        assert viewer.symmetric_scale is not None
        assert viewer.replace_nan is not None

        # check default values
        assert viewer.threshold.min >= 0.0
        assert viewer.threshold.max > 0.0  # depends on data
        assert viewer.toggle_phase.value == "Phase"
        assert viewer.auto_scale.value is True
        assert viewer.symmetric_scale.value is False
        assert viewer.replace_nan.value is False

    def test_viewer_voxel_size(self, complex_data):
        """Test voxel size handling."""
        voxel_size = (0.5, 0.5, 0.5)
        viewer = ThreeDViewer(complex_data, voxel_size=voxel_size)

        np.testing.assert_array_equal(viewer.voxel_size, voxel_size)

    def test_viewer_with_nan_data(self, complex_data_with_nan):
        """Test viewer with NaN values in data."""
        viewer = ThreeDViewer(complex_data_with_nan)

        # viewer should initialise without errors
        assert viewer is not None

        # check that NaN handling checkbox exists
        assert viewer.replace_nan is not None
        assert viewer.replace_nan.value is False

    def test_set_data(self, complex_data):
        """Test the set_data method."""
        viewer = ThreeDViewer()
        viewer.set_data(complex_data, threshold=0.5)

        assert viewer.data is not None
        assert viewer.threshold.value == 0.5

        # check that interpolator was created
        assert viewer.rgi is not None

    def test_colormap_options(self, complex_data):
        """Test that colormap options are correctly set."""
        viewer = ThreeDViewer(complex_data)

        # check that colormap options are available
        assert len(viewer.cmap_options) > 0
        assert "turbo" in viewer.cmap_options
        assert "viridis" in viewer.cmap_options
        assert "RdBu" in viewer.cmap_options
        assert "twilight" in viewer.cmap_options


class TestPlot3DIsosurface:
    """Test suite for the plot_3d_isosurface function."""

    @pytest.fixture
    def reconstruction_data(self):
        """Create sample reconstruction data."""
        shape = (20, 20, 20)
        amplitude = np.ones(shape)
        x, y, z = np.ogrid[-10:10, -10:10, -10:10]
        mask = x**2 + y**2 + z**2 <= 7**2
        amplitude[mask] = 2.0

        phase = np.random.uniform(-np.pi, np.pi, shape)
        strain = np.random.uniform(-0.01, 0.01, shape)
        support = np.zeros(shape)
        support[mask] = 1.0

        return {
            "amplitude": amplitude,
            "phase": phase,
            "het_strain": strain,
            "support": support,
        }

    @pytest.fixture
    def reconstruction_data_with_nan(self):
        """Create sample reconstruction data with NaN values."""
        shape = (20, 20, 20)
        amplitude = np.ones(shape)
        x, y, z = np.ogrid[-10:10, -10:10, -10:10]
        mask = x**2 + y**2 + z**2 <= 7**2
        amplitude[mask] = 2.0

        phase = np.random.uniform(-np.pi, np.pi, shape)
        strain = np.random.uniform(-0.01, 0.01, shape)
        # introduce NaN values
        strain[5:8, 5:8, 5:8] = np.nan

        support = np.zeros(shape)
        support[mask] = 1.0

        return {
            "amplitude": amplitude,
            "phase": phase,
            "het_strain": strain,
            "support": support,
        }

    def test_basic_plot_creation(self, reconstruction_data):
        """Test basic plot creation without errors."""
        # this should not raise any exceptions
        widget = plot_3d_isosurface(
            reconstruction_data["amplitude"],
            reconstruction_data,
        )

        # check that widget was created
        assert widget is not None

    def test_plot_with_voxel_size(self, reconstruction_data):
        """Test plot creation with voxel size."""
        voxel_size = (0.5, 0.5, 0.5)
        widget = plot_3d_isosurface(
            reconstruction_data["amplitude"],
            reconstruction_data,
            voxel_size=voxel_size,
        )

        assert widget is not None

    def test_plot_with_cmaps(self, reconstruction_data):
        """Test plot creation with custom colormaps."""
        cmaps = {
            "amplitude": "turbo",
            "het_strain": "RdBu",
            "phase": "twilight",
        }
        widget = plot_3d_isosurface(
            reconstruction_data["amplitude"],
            reconstruction_data,
            cmaps=cmaps,
        )

        assert widget is not None

    def test_plot_with_nan_data(self, reconstruction_data_with_nan):
        """Test plot creation with NaN values."""
        widget = plot_3d_isosurface(
            reconstruction_data_with_nan["amplitude"],
            reconstruction_data_with_nan,
        )

        # should handle NaN gracefully
        assert widget is not None


class TestHelperFunctions:
    """Test suite for helper functions in the volume module."""

    @pytest.fixture
    def simple_3d_data(self):
        """Create simple 3D data for testing."""
        shape = (15, 15, 15)
        data = np.ones(shape)
        x, y, z = np.ogrid[-7:8, -7:8, -7:8]
        mask = x**2 + y**2 + z**2 <= 5**2
        data[mask] = 2.0
        return data

    def test_extract_isosurface_basic(self, simple_3d_data):
        """Test basic isosurface extraction."""
        verts, faces, vals = _extract_isosurface_with_values(
            simple_3d_data,
            simple_3d_data,
            isosurface_level=1.5,
        )

        # check that outputs have correct types
        assert isinstance(verts, np.ndarray)
        assert isinstance(faces, np.ndarray)
        assert isinstance(vals, np.ndarray)

        # check shapes
        assert verts.ndim == 2
        assert verts.shape[1] == 3
        assert faces.ndim == 2
        assert faces.shape[1] == 3
        assert vals.ndim == 1
        assert len(vals) == len(verts)

    def test_extract_isosurface_with_voxel_size(self, simple_3d_data):
        """Test isosurface extraction with voxel size."""
        voxel_size = (0.5, 0.5, 0.5)
        verts, faces, vals = _extract_isosurface_with_values(
            simple_3d_data,
            simple_3d_data,
            isosurface_level=1.5,
            voxel_size=voxel_size,
        )

        # vertices should be scaled by voxel size
        assert verts.max() <= simple_3d_data.shape[0] * voxel_size[0]

    def test_extract_isosurface_complex_data(self, simple_3d_data):
        """Test isosurface extraction with complex quantity."""
        # create complex data
        phase = np.random.uniform(-np.pi, np.pi, simple_3d_data.shape)
        complex_data = simple_3d_data * np.exp(1j * phase)

        verts, faces, vals = _extract_isosurface_with_values(
            simple_3d_data,
            complex_data,
            isosurface_level=1.5,
            use_interpolator=True,
        )

        # vals should be complex
        assert np.iscomplexobj(vals)
        assert len(vals) == len(verts)

    def test_colorcet_to_plotly(self):
        """Test colormap conversion to Plotly format."""
        # test with matplotlib colormap
        colorscale = colorcet_to_plotly("viridis", n_colors=10)

        # check format
        assert isinstance(colorscale, list)
        assert len(colorscale) == 10

        # check that each entry is [position, color]
        for entry in colorscale:
            assert len(entry) == 2
            assert 0 <= entry[0] <= 1
            assert entry[1].startswith("rgb(")

    def test_colorcet_to_plotly_invalid_cmap(self):
        """Test colormap conversion with invalid colormap name."""
        with pytest.raises(ValueError):
            colorcet_to_plotly("not_a_real_colormap")


class TestNaNHandling:
    """Test suite specifically for NaN handling in visualisations."""

    @pytest.fixture
    def data_with_nan(self):
        """Create data with strategic NaN placement."""
        shape = (20, 20, 20)
        data = np.ones(shape)
        x, y, z = np.ogrid[-10:10, -10:10, -10:10]
        mask = x**2 + y**2 + z**2 <= 7**2
        data[mask] = 2.0

        # add NaN values in different locations
        data[5:8, 5:8, 5:8] = np.nan
        data[15, 15, 15] = np.nan

        return data

    def test_nan_in_data(self, data_with_nan):
        """Test that NaN values are present in test data."""
        assert np.any(np.isnan(data_with_nan))

    def test_viewer_with_nan_replacement(self, data_with_nan):
        """Test ThreeDViewer with NaN replacement enabled."""
        complex_data = data_with_nan * np.exp(
            1j * np.random.uniform(-np.pi, np.pi, data_with_nan.shape)
        )

        viewer = ThreeDViewer(complex_data)

        # enable NaN replacement
        viewer.replace_nan.value = True

        # viewer should handle this without errors
        assert viewer.replace_nan.value is True

    def test_nanmean_calculation(self, data_with_nan):
        """Test that np.nanmean works as expected."""
        mean_val = np.nanmean(data_with_nan)

        # mean should ignore NaN values
        assert not np.isnan(mean_val)
        assert mean_val > 0  # we know data is positive

    def test_nan_replacement_logic(self, data_with_nan):
        """Test the NaN replacement logic."""
        # simulate what happens in the viewer
        if np.any(np.isnan(data_with_nan)):
            mean_val = np.nanmean(data_with_nan)
            data_replaced = np.where(
                np.isnan(data_with_nan),
                mean_val,
                data_with_nan,
            )

            # check that no NaN values remain
            assert not np.any(np.isnan(data_replaced))

            # check that non-NaN values are preserved
            mask = ~np.isnan(data_with_nan)
            np.testing.assert_array_equal(
                data_replaced[mask], data_with_nan[mask]
            )


class TestWidgetInteraction:
    """Test suite for widget interactions and state management."""

    @pytest.fixture
    def viewer_with_data(self):
        """Create a viewer with data for interaction tests."""
        shape = (20, 20, 20)
        amplitude = np.ones(shape)
        x, y, z = np.ogrid[-10:10, -10:10, -10:10]
        mask = x**2 + y**2 + z**2 <= 7**2
        amplitude[mask] = 2.0
        phase = np.random.uniform(-np.pi, np.pi, shape)
        complex_data = amplitude * np.exp(1j * phase)

        return ThreeDViewer(complex_data)

    def test_threshold_slider_range(self, viewer_with_data):
        """Test threshold slider has valid range."""
        assert viewer_with_data.threshold.min >= 0.0
        assert viewer_with_data.threshold.max > 0.0
        assert viewer_with_data.threshold.min < viewer_with_data.threshold.max

    def test_toggle_phase_options(self, viewer_with_data):
        """Test phase/amplitude toggle has correct options."""
        assert "Phase" in viewer_with_data.toggle_phase.options
        assert "Amplitude" in viewer_with_data.toggle_phase.options

    def test_colormap_dropdown_options(self, viewer_with_data):
        """Test colormap dropdown has valid options."""
        assert len(viewer_with_data.colormap.options) > 0

        # check for some expected colormaps
        options = viewer_with_data.colormap.options
        assert "turbo" in options
        assert "viridis" in options
        assert "RdBu" in options

    def test_checkbox_states(self, viewer_with_data):
        """Test that checkboxes can be toggled."""
        viewer = viewer_with_data

        # test auto_scale
        initial_auto = viewer.auto_scale.value
        viewer.auto_scale.value = not initial_auto
        assert viewer.auto_scale.value != initial_auto

        # test symmetric_scale
        initial_sym = viewer.symmetric_scale.value
        viewer.symmetric_scale.value = not initial_sym
        assert viewer.symmetric_scale.value != initial_sym

        # test replace_nan
        initial_nan = viewer.replace_nan.value
        viewer.replace_nan.value = not initial_nan
        assert viewer.replace_nan.value != initial_nan


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_empty_data(self):
        """Test with empty/very small data."""
        small_data = np.ones((3, 3, 3))

        # should not raise error
        viewer = ThreeDViewer(small_data)
        assert viewer is not None

    def test_all_nan_data(self):
        """Test with data that is all NaN."""
        nan_data = np.full((10, 10, 10), np.nan, dtype=complex)

        # should initialise, though might not show useful plot
        viewer = ThreeDViewer(nan_data)
        assert viewer is not None

    def test_zero_data(self):
        """Test with all-zero data."""
        zero_data = np.zeros((10, 10, 10), dtype=complex)

        viewer = ThreeDViewer(zero_data)
        assert viewer is not None

    def test_invalid_voxel_size(self):
        """Test with invalid voxel size."""
        data = np.ones((10, 10, 10), dtype=complex)

        # wrong length tuple should be handled
        viewer = ThreeDViewer(data, voxel_size=(1, 1))
        assert viewer is not None

    def test_very_high_threshold(self):
        """Test with threshold higher than data values."""
        data = np.ones((10, 10, 10), dtype=complex)
        viewer = ThreeDViewer(data)

        # set threshold above maximum
        viewer.threshold.max = 10.0
        viewer.threshold.value = 5.0

        # should not crash
        assert viewer.threshold.value == 5.0
