import numpy as np
import pytest

from cdiutils.geometry import Geometry


class TestGeometry:
    """Test suite for the Geometry class"""

    @pytest.fixture
    def geometry(self):
        """Create a default geometry instance for tests"""
        return Geometry()

    def test_swap_convention(self):
        """
        Test the swap_convention static method for converting between
        CXI and XU conventions.
        """
        # test with lists
        assert Geometry.swap_convention([1, 2, 3]) == [1, 3, 2]

        # test with tuples
        assert Geometry.swap_convention((1, 2, 3)) == (1, 3, 2)

        # test with numpy arrays (vector)
        vec = np.array([1, 2, 3])
        result = Geometry.swap_convention(vec)
        np.testing.assert_array_equal(result, np.array([1, 3, 2]))

        # test with 3D array
        arr = np.ones((3, 4, 5))
        arr[1, 2, 3] = 5  # mark a specific position
        result = Geometry.swap_convention(arr)
        assert result[1, 3, 2] == 5  # check that y,z axes were swapped

        # test edge case - empty array
        with pytest.raises(TypeError):
            Geometry.swap_convention("not an array")

    def test_sample_orientation(self, geometry):
        """Test the sample_orientation property and setter"""
        # test default (horizontal)
        assert geometry.sample_orientation == "horizontal"

        # test setter with 'vertical'
        geometry.sample_orientation = "vertical"
        assert geometry.sample_orientation == "vertical"
        np.testing.assert_array_equal(
            geometry.sample_surface_normal, [0, 0, 1]
        )

        # test setter with abbreviation
        geometry.sample_orientation = "h"
        assert geometry.sample_orientation == "horizontal"
        np.testing.assert_array_equal(
            geometry.sample_surface_normal, [0, 1, 0]
        )

        # test with custom vector (primarily y-axis)
        geometry.sample_surface_normal = [0, 0.8, 0.2]
        assert geometry.sample_orientation == "horizontal"

        # test with custom vector (primarily z-axis)
        geometry.sample_surface_normal = [0.1, 0.2, 0.9]
        assert geometry.sample_orientation == "vertical"

        # test with XU convention
        geometry.is_cxi = False
        geometry.sample_surface_normal = [0.1, 0.1, 0.9]  # primarily Zxu
        assert geometry.sample_orientation == "horizontal"

        # test invalid orientation
        with pytest.raises(ValueError):
            geometry.sample_orientation = "diagonal"

    def test_cxi_to_xu(self):
        """Test the conversion from CXI to XU convention"""
        # create a geometry instance with CXI convention
        geometry = Geometry(
            sample_circles=["x+", "y-"],
            detector_circles=["y+", "x-"],
            sample_surface_normal=[0, 1, 0],  # y-axis in CXI (horizontal)
            is_cxi=True,
        )

        # convert to XU
        geometry.cxi_to_xu()

        # check that is_cxi is now False
        assert not geometry.is_cxi

        # check that sample_surface_normal was swapped correctly
        np.testing.assert_array_equal(
            geometry.sample_surface_normal,
            [0, 0, 1],  # Zxu
        )

        # check that sample circles were converted correctly
        assert geometry.sample_circles == ["y+", "z-"]

        # check that detector circles were converted correctly
        assert geometry.detector_circles == ["z+", "y-"]
