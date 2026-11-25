"""
Unit tests for the object loading of the phase retrieval engine.
"""

import h5py
import numpy as np
import pytest

from cdiutils.interactive.phase_retrieval.engine import load_object


@pytest.fixture
def obj():
    """Return a small complex object."""
    return (np.arange(24).reshape(2, 3, 4) * np.exp(1j)).astype(np.complex64)


@pytest.mark.unit
class TestLoadObject:
    """Test the loading of the object from the supported file formats."""

    def test_cxi(self, tmp_path, obj):
        """Test the object is read at the PyNX location in a cxi file."""
        path = tmp_path / "Run0001.cxi"
        with h5py.File(path, "w") as file:
            file["entry_1/image_1/data"] = obj
        np.testing.assert_array_equal(load_object(str(path)), obj)

    @pytest.mark.parametrize("key", ["object", "obj", "data"])
    def test_npz(self, tmp_path, obj, key):
        """Test the keys accepted in a npz file."""
        path = tmp_path / "object.npz"
        np.savez(path, **{key: obj})
        np.testing.assert_array_equal(load_object(str(path)), obj)

    def test_npy(self, tmp_path, obj):
        """Test loading from a npy file."""
        path = tmp_path / "object.npy"
        np.save(path, obj)
        np.testing.assert_array_equal(load_object(str(path)), obj)

    def test_cxi_without_object(self, tmp_path):
        """Test a cxi file that holds no object is rejected."""
        path = tmp_path / "input.cxi"
        with h5py.File(path, "w") as file:
            file["entry_1/data_1/data"] = np.zeros((2, 3, 4))
        with pytest.raises(ValueError, match="image_1"):
            load_object(str(path))

    def test_npz_without_known_key(self, tmp_path, obj):
        """Test a npz file without any known key is rejected."""
        path = tmp_path / "object.npz"
        np.savez(path, other=obj)
        with pytest.raises(ValueError, match="keys"):
            load_object(str(path))

    def test_unsupported_format(self, tmp_path):
        """Test an unknown extension is rejected."""
        with pytest.raises(ValueError, match="Unsupported"):
            load_object(str(tmp_path / "object.tif"))
