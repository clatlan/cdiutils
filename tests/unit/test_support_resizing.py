"""
Unit tests for the resizing of supports (crop and/or pad at the centre).
"""

import h5py
import numpy as np
import pytest

from cdiutils.utils import crop_at_center, resize_at_center


@pytest.mark.unit
class TestResizeAtCenter:
    """Test resize_at_center keeps the centre and the voxel size."""

    @pytest.mark.parametrize(
        "input_shape, output_shape",
        [
            ((200, 200, 200), (160, 180, 180)),
            ((160, 180, 180), (200, 200, 200)),
            ((200, 200, 200), (160, 220, 201)),
            ((199, 200, 201), (200, 199, 200)),
        ],
    )
    def test_centre_voxel_is_kept(self, input_shape, output_shape):
        """Test the n // 2 voxel lands at the output n // 2 voxel."""
        data = np.zeros(input_shape, dtype=np.int8)
        data[tuple(s // 2 for s in input_shape)] = 1
        resized = resize_at_center(data, output_shape)
        assert resized.shape == output_shape
        assert resized.dtype == data.dtype
        assert resized[tuple(s // 2 for s in output_shape)] == 1
        assert resized.sum() == 1

    def test_same_as_crop_at_center(self):
        """Test cropping matches the existing crop_at_center."""
        data = np.random.default_rng(0).random((20, 21, 22))
        np.testing.assert_array_equal(
            resize_at_center(data, (11, 12, 13)),
            crop_at_center(data, (11, 12, 13)),
        )

    def test_crop_then_pad_back(self):
        """Test a centred object survives a crop and a pad."""
        support = np.zeros((200, 200, 200), dtype=np.int8)
        support[80:120, 90:110, 70:130] = 1
        resized = resize_at_center(support, (160, 180, 180))
        assert resized.sum() == support.sum()
        np.testing.assert_array_equal(
            resize_at_center(resized, support.shape), support
        )

    def test_padding_value(self):
        """Test the padded voxels take the given value."""
        resized = resize_at_center(np.zeros((2, 2)), (4, 4), values=-1)
        assert (resized == -1).sum() == 12

    @pytest.mark.parametrize("output_shape", [(10, 10), (10, 0, 10)])
    def test_invalid_shape(self, output_shape):
        """Test a shape with a wrong length or a null size is rejected."""
        with pytest.raises(ValueError):
            resize_at_center(np.zeros((4, 4, 4)), output_shape)


@pytest.mark.unit
class TestSupportTools:
    """Test the loading and resizing of a support file."""

    @pytest.fixture
    def support(self):
        """Return a small centred support."""
        support = np.zeros((20, 20, 20), dtype=np.int8)
        support[7:13, 6:14, 5:15] = 1
        return support

    @pytest.fixture
    def support_tools(self):
        """Return the SupportTools class, skipped if not available."""
        pytest.importorskip("tables")
        from cdiutils.process.support_tools import SupportTools

        return SupportTools

    @pytest.mark.parametrize("key", ["support", "data", "mask", "other"])
    def test_load_npz(self, tmp_path, support, support_tools, key):
        """Test the keys accepted in a npz file."""
        path = str(tmp_path / "support.npz")
        np.savez(path, **{key: support})
        np.testing.assert_array_equal(
            support_tools.load_support(path), support
        )

    def test_load_cxi(self, tmp_path, support, support_tools):
        """Test the support is read at the PyNX location in a cxi file."""
        path = str(tmp_path / "Run0001.cxi")
        with h5py.File(path, "w") as file:
            file["entry_1/image_1/mask"] = support
        np.testing.assert_array_equal(
            support_tools.load_support(path), support
        )

    def test_resize_support(self, tmp_path, support, support_tools):
        """Test the resized support is saved under the "support" key."""
        path = tmp_path / "support.npz"
        np.savez(path, support=support)
        support_tools(path_to_support=str(path)).resize_support(
            "(16, 18, 24)"
        )
        with np.load(tmp_path / "support_16_18_24.npz") as file:
            resized = file["support"]
        assert resized.shape == (16, 18, 24)
        assert resized.sum() == support.sum()
