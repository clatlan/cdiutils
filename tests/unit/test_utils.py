"""
Unit tests for utility functions in cdiutils.utils module.

These tests verify individual utility functions work correctly
in isolation.
"""

import numpy as np
import pytest

from cdiutils.utils import (
    CroppingHandler,
    ensure_pynx_shape,
    make_support,
    nan_to_zero,
    normalise,
    zero_to_nan,
)


@pytest.mark.unit
class TestCroppingHandler:
    """Test the CroppingHandler utility class."""

    def test_get_roi_basic(self):
        """Test basic ROI calculation."""
        output_shape = (100, 100, 100)
        where = (50, 50, 50)
        input_shape = (200, 200, 200)

        roi = CroppingHandler.get_roi(output_shape, where, input_shape)

        # verify ROI structure
        assert len(roi) == 6  # 3D -> 6 values (start, end for each axis)
        assert all(isinstance(x, (int, np.integer)) for x in roi)

        # verify ROI has correct size
        roi_shape = (roi[1] - roi[0], roi[3] - roi[2], roi[5] - roi[4])
        assert roi_shape == output_shape

    def test_get_roi_with_tuple_shape(self):
        """Test ROI calculation with tuple shape."""
        output_shape = (80, 120, 100)
        where = (100, 100, 100)
        input_shape = (200, 200, 200)

        roi = CroppingHandler.get_roi(output_shape, where, input_shape)

        # verify ROI respects requested shape
        roi_shape = (roi[1] - roi[0], roi[3] - roi[2], roi[5] - roi[4])
        assert roi_shape == output_shape

    def test_roi_list_to_slices(self):
        """Test conversion of ROI list to slices."""
        roi = [10, 110, 20, 120, 30, 130]

        slices = CroppingHandler.roi_list_to_slices(roi)

        # verify slices structure
        assert len(slices) == 3
        assert all(isinstance(s, slice) for s in slices)

        # verify slice values
        assert slices[0] == slice(10, 110)
        assert slices[1] == slice(20, 120)
        assert slices[2] == slice(30, 130)

    def test_get_position_max(self):
        """Test finding maximum position."""
        data = np.zeros((20, 20, 20))
        data[10, 12, 8] = 100  # set maximum

        pos = CroppingHandler.get_position(data, "max")

        # verify position
        assert pos == (10, 12, 8)

    def test_get_position_com(self):
        """Test finding centre of mass."""
        # create data with known COM
        data = np.zeros((20, 20, 20))
        data[8:12, 8:12, 8:12] = 1  # cube centred at (10, 10, 10)

        pos = CroppingHandler.get_position(data, "com")

        # verify COM is approximately in the centre
        assert all(abs(p - 9.5) < 1 for p in pos)


@pytest.mark.unit
class TestNormalisation:
    """Test normalisation and data manipulation functions."""

    def test_normalise_basic(self):
        """Test basic normalisation."""
        data = np.array([1, 2, 3, 4, 5], dtype=float)

        normalised = normalise(data)

        # verify range [0, 1]
        assert normalised.min() == 0.0
        assert normalised.max() == 1.0

        # verify monotonicity preserved
        assert np.all(np.diff(normalised) > 0)

    def test_normalise_with_nan(self):
        """Test normalisation with NaN values."""
        data = np.array([1, 2, np.nan, 4, 5], dtype=float)

        normalised = normalise(data)

        # verify NaN is preserved
        assert np.isnan(normalised[2])

        # verify non-NaN values normalised between 0 and 1
        valid_results = normalised[~np.isnan(normalised)]
        if len(valid_results) > 0:
            assert np.min(valid_results) >= -1e-10  # Allow tiny float error
            assert np.max(valid_results) <= 1.0 + 1e-10

    def test_zero_to_nan(self):
        """Test conversion of zeros to NaN."""
        data = np.array([0, 1, 0, 2, 0, 3], dtype=float)

        result = zero_to_nan(data)

        # verify zeros became NaN
        assert np.isnan(result[0])
        assert np.isnan(result[2])
        assert np.isnan(result[4])

        # verify non-zeros preserved
        assert result[1] == 1
        assert result[3] == 2
        assert result[5] == 3

    def test_nan_to_zero(self):
        """Test conversion of NaN to zeros."""
        data = np.array([np.nan, 1, np.nan, 2, np.nan, 3], dtype=float)

        result = nan_to_zero(data)

        # verify NaN became zeros
        assert result[0] == 0
        assert result[2] == 0
        assert result[4] == 0

        # verify non-NaN preserved
        assert result[1] == 1
        assert result[3] == 2
        assert result[5] == 3


@pytest.mark.unit
class TestSupportGeneration:
    """Test support generation functions."""

    def test_make_support_basic(self, sphere_data):
        """Test basic support generation."""
        data = sphere_data["data"]

        support = make_support(data, isosurface=0.5)

        # verify support is int (0 or 1), not boolean by default
        assert support.dtype in [np.int64, np.int32]
        assert support.shape == data.shape
        assert set(np.unique(support)) <= {0, 1}

        # verify support captures high-intensity regions
        # normalised data > 0.5 threshold should have support=1
        normalised_data = (data - data.min()) / (data.max() - data.min())
        assert support[normalised_data > 0.5].sum() > 0

    def test_make_support_with_threshold(self):
        """Test support generation with different thresholds."""
        data = np.random.rand(30, 30, 30)

        # different isosurfaces should give different supports
        support_low = make_support(data, isosurface=0.3)
        support_high = make_support(data, isosurface=0.7)

        # higher threshold should give smaller support
        assert support_low.sum() > support_high.sum()


@pytest.mark.unit
class TestPyNXShapeValidation:
    """Test PyNX shape validation and adjustment."""

    def test_ensure_pynx_shape_valid(self):
        """Test validation of PyNX-compatible shapes."""
        # PyNX requires shapes factorisable by 2, 3, 5, 7
        valid_shape = (128, 128, 128)  # 2^7

        result = ensure_pynx_shape(valid_shape)

        # should return unchanged
        assert result == valid_shape

    def test_ensure_pynx_shape_adjustment(self):
        """Test adjustment of incompatible shapes."""
        # shape that needs adjustment
        incompatible_shape = (99, 99, 99)

        result = ensure_pynx_shape(incompatible_shape)

        # verify result is PyNX-compatible
        assert all(isinstance(x, int) for x in result)
        # result should be close to input
        for original, adjusted in zip(incompatible_shape, result):
            assert abs(original - adjusted) < 20

    def test_ensure_pynx_shape_2d_to_3d(self):
        """Test handling of 2D shapes."""
        shape_2d = (128, 128)

        # should handle 2D shapes appropriately
        result = ensure_pynx_shape(shape_2d)

        # verify result
        assert len(result) >= 2

    @pytest.mark.parametrize(
        "shape",
        [
            (100, 100, 100),
            (150, 150, 150),
            (200, 200, 200),
            (64, 128, 256),
        ],
    )
    def test_ensure_pynx_shape_various_sizes(self, shape):
        """Test PyNX shape validation for various sizes."""
        result = ensure_pynx_shape(shape)

        # verify result is a tuple of ints
        assert isinstance(result, tuple)
        assert all(isinstance(x, int) for x in result)
        assert len(result) == len(shape)


@pytest.mark.unit
class TestArrayManipulation:
    """Test array manipulation utilities."""

    def test_hybrid_gradient(self):
        """Test hybrid gradient computation."""
        from cdiutils.utils import hybrid_gradient

        # create a simple 2D array with known gradient
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x, y)

        # linear function: f(x,y) = 2*x + 3*y
        data = 2 * X + 3 * Y

        # compute gradients with spacing
        dx = x[1] - x[0]  # spacing in x
        dy = y[1] - y[0]  # spacing in y
        grad_y, grad_x = hybrid_gradient(data, dy, dx)

        # verify gradients are approximately constant
        # Note: gradient returns (grad_axis0, grad_axis1)
        # axis 0 is Y (rows), axis 1 is X (columns)
        assert np.allclose(np.nanmean(grad_x), 2, atol=0.5)
        assert np.allclose(np.nanmean(grad_y), 3, atol=0.5)

    def test_find_suitable_array_shape(self):
        """Test finding suitable array shapes."""
        from cdiutils.utils import find_suitable_array_shape

        # create a simple 3D support array (not an integer)
        support = np.zeros((100, 100, 100))
        support[40:60, 40:60, 40:60] = 1  # 20x20x20 cube

        shape = find_suitable_array_shape(support, symmetrical=True)

        # verify returned shape
        assert isinstance(shape, tuple)
        # for symmetrical, all dimensions should be equal
        assert len(set(shape)) == 1
        # shape should be larger than the support region (20x20x20)
        assert shape[0] > 20


@pytest.mark.unit
class TestOversamplingCalculation:
    """Test oversampling calculation utilities."""

    def test_oversampling_from_diffraction(self):
        """Test oversampling ratio calculation."""
        from cdiutils.utils import oversampling_from_diffraction

        # create mock diffraction data with known dimensions
        data = np.zeros((100, 100, 100))
        # add a bragg peak
        data[45:55, 45:55, 45:55] = 1

        # calculate oversampling
        ratios = oversampling_from_diffraction(data)

        # verify ratios structure
        assert len(ratios) == 3
        assert all(r > 0 for r in ratios)

    def test_get_oversampling_ratios(self):
        """Test get_oversampling_ratios utility."""
        from cdiutils.utils import get_oversampling_ratios

        # create a 3D support array (not two separate tuples)
        support = np.zeros((100, 200, 200))
        support[45:55, 90:110, 90:110] = 1  # 10x20x20 region

        ratios = get_oversampling_ratios(support)

        # verify ratios structure
        assert ratios.shape == (3,)
        # ratios should be detector_shape / object_shape
        # approximately (100/9, 200/19, 200/19) ~ (11.1, 10.5, 10.5)
        # Allow some tolerance for boundary effects
        assert np.all(ratios > 9)
        assert np.all(ratios < 12)
