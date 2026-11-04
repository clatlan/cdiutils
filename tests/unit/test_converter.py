"""
Unit tests for cdiutils.converter.SpaceConverter.
"""

import numpy as np
import pytest

from cdiutils.converter import SpaceConverter


@pytest.mark.unit
def test_get_transformation_matrix_no_int32_overflow(monkeypatch):
    """Regression test for clatlan/cdiutils#109.

    np.indices defaults to int32 on some platforms (e.g. Windows with
    NumPy < 2), which made the Gram matrix overflow for large shapes.
    """
    original_indices = np.indices

    def indices_int32(dimensions, dtype=np.int32, sparse=False):
        return original_indices(dimensions, dtype=dtype, sparse=sparse)

    monkeypatch.setattr(np, "indices", indices_int32)

    # large enough for sum(i**2) to exceed the int32 range
    shape = (128, 128, 128)
    expected = np.array(
        [[1e-3, 2e-4, 0.0], [0.0, 1.5e-3, 3e-4], [5e-4, 0.0, 2e-3]]
    )
    grid = original_indices(shape, dtype=np.float64).reshape(3, -1)

    converter = object.__new__(SpaceConverter)
    converter._shape = shape
    converter._q_space_transitions = np.dot(expected, grid).reshape(
        (3,) + shape
    )

    matrix = converter.get_transformation_matrix()

    np.testing.assert_allclose(matrix, expected, rtol=1e-8, atol=1e-12)
