"""
Pytest configuration and fixtures for CDIutils tests.

This module provides shared fixtures and configuration for testing the
CDIutils package, including test data management, temporary directories,
and common test objects.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ensure cdiutils is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# Test data paths
TEST_ROOT = Path(__file__).parent
CDIUTILS_ROOT = TEST_ROOT.parent / "src" / "cdiutils"

# test data directory (outside the package)
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DATA_ROOT = PROJECT_ROOT / "tests" / "raw_data"


@pytest.fixture(scope="session")
def test_data_path():
    """
    Provide the path to the test data directory.

    Returns:
        Path: path to tests/raw_data directory
    """
    if not TEST_DATA_ROOT.exists():
        pytest.skip(
            f"Test data directory not found: {TEST_DATA_ROOT}\n"
            "Real data tests require the tests/raw_data directory."
        )
    return TEST_DATA_ROOT


@pytest.fixture(scope="session")
def sixs_data_path(test_data_path):
    """Path to SIXS test data."""
    path = test_data_path / "sixs"
    if not path.exists():
        pytest.skip("SIXS test data not available")
    return path


@pytest.fixture(scope="session")
def id01_data_path(test_data_path):
    """Path to ID01 test data."""
    path = test_data_path / "id01"
    if not path.exists():
        pytest.skip("ID01 test data not available")
    return path


@pytest.fixture(scope="session")
def id27_data_path(test_data_path):
    """Path to ID27 test data."""
    path = test_data_path / "id27"
    if not path.exists():
        pytest.skip("ID27 test data not available")
    return path


@pytest.fixture(scope="session")
def cristal_data_path(test_data_path):
    """Path to Cristal test data."""
    path = test_data_path / "cristal"
    if not path.exists():
        pytest.skip("Cristal test data not available")
    return path


@pytest.fixture
def temp_output_dir():
    """
    Create a temporary directory for test outputs.

    Yields:
        Path: temporary directory path

    Cleanup:
        Directory is removed after test completes
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="cdiutils_test_"))
    yield temp_dir
    # cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_pipeline_params():
    """
    Provide a minimal set of parameters for BcdiPipeline testing.

    Returns:
        dict: minimal parameter dictionary
    """
    return {
        "beamline_setup": "sixs2019",
        "scan": 457,
        "sample_name": "test_sample",
        "dump_dir": None,  # will be set by test
    }


@pytest.fixture
def simple_3d_array():
    """
    Create a simple 3D numpy array for testing.

    Returns:
        np.ndarray: 3D array of shape (20, 20, 20)
    """
    return np.random.rand(20, 20, 20)


@pytest.fixture
def simple_complex_array():
    """
    Create a simple 3D complex array for testing.

    Returns:
        np.ndarray: complex 3D array
    """
    shape = (20, 20, 20)
    amplitude = np.random.rand(*shape)
    phase = np.random.uniform(-np.pi, np.pi, shape)
    return amplitude * np.exp(1j * phase)


@pytest.fixture
def sphere_data():
    """
    Create a 3D array with a spherical object in the centre.

    Useful for testing support generation, isosurface detection, etc.

    Returns:
        dict: containing 'data' and 'support' arrays
    """
    shape = (40, 40, 40)
    data = np.zeros(shape)

    # create a sphere in the centre
    centre = np.array(shape) // 2
    radius = 10

    x, y, z = np.ogrid[: shape[0], : shape[1], : shape[2]]

    # distance from centre
    dist_from_centre = np.sqrt(
        (x - centre[0]) ** 2 + (y - centre[1]) ** 2 + (z - centre[2]) ** 2
    )

    # create sphere
    support = dist_from_centre <= radius
    data[support] = 1.0

    # add some noise
    data += np.random.normal(0, 0.1, shape)
    data = np.maximum(data, 0)

    return {
        "data": data,
        "support": support,
        "centre": centre,
        "radius": radius,
    }


@pytest.fixture
def mock_detector_data():
    """
    Create mock detector data for testing preprocessing.

    Returns:
        dict: containing detector data and associated metadata
    """
    shape = (100, 256, 256)  # (frames, height, width)

    # create a Bragg peak
    data = np.zeros(shape)

    # position of peak
    peak_pos = (50, 128, 128)
    peak_width = 10

    # gaussian peak
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt(
                    (i - peak_pos[0]) ** 2
                    + (j - peak_pos[1]) ** 2
                    + (k - peak_pos[2]) ** 2
                )
                data[i, j, k] = 1000 * np.exp(-(dist**2) / (2 * peak_width**2))

    # add some background noise
    data += np.random.poisson(5, shape)

    return {"data": data, "peak_position": peak_pos, "shape": shape}


@pytest.fixture(scope="session")
def sixs_2019_params(sixs_data_path, tmp_path_factory):
    """
    Parameters for SIXS 2019 beamline test.

    Args:
        sixs_data_path: fixture providing SIXS data path
        tmp_path_factory: pytest fixture for temporary paths

    Returns:
        dict: complete parameter dictionary for BcdiPipeline
    """
    dump_dir = tmp_path_factory.mktemp("sixs_2019_results")

    nxs_file = sixs_data_path / "align_ascan_mu_00457.nxs"
    if not nxs_file.exists():
        pytest.skip(f"SIXS test file not found: {nxs_file}")

    return {
        "beamline_setup": "sixs2019",
        "scan": 457,
        "sample_name": "sixs_2019_test",
        "dump_dir": str(dump_dir),
        "experiment_file_path": str(nxs_file),
        "preprocess_shape": (100, 150, 150),
        "voxel_reference_methods": (128, 157, 145),
        "det_calib_params": {
            "distance": 1.0,
            "cch1": 250,
            "cch2": 250,
            "pwidth1": 55e-6,
            "pwidth2": 55e-6,
            "tiltazimuth": 0,
            "detrot": 0,
            "tilt": 0,
        },
    }


@pytest.fixture(scope="session")
def id01_core_shell_params(id01_data_path, tmp_path_factory):
    """
    Parameters for ID01 core-shell sample test (SPEC format).

    Args:
        id01_data_path: fixture providing ID01 data path
        tmp_path_factory: pytest fixture for temporary paths

    Returns:
        dict: complete parameter dictionary for BcdiPipeline
    """
    dump_dir = tmp_path_factory.mktemp("id01_core_shell_results")

    core_shell_path = id01_data_path / "core_shell"
    spec_file = core_shell_path / "BCDI_2020_11_06_012505.spec"
    detector_path = core_shell_path / "detector"

    if not spec_file.exists():
        pytest.skip(f"ID01 SPEC file not found: {spec_file}")
    if not detector_path.exists():
        pytest.skip(f"ID01 detector data not found: {detector_path}")

    return {
        "beamline_setup": "id01spec",
        "sample_name": "core_shell_spec_test",
        "scan": 163,
        "dump_dir": str(dump_dir),
        "experiment_file_path": str(spec_file),
        "detector_data_path": str(detector_path),
        "detector_name": "mpx4inr",
        "edf_file_template": "data_mpx4_%05d.edf.gz",
        "energy": 8994,
        "det_calib_params": {
            "cch1": 347.39340712173976,
            "cch2": 71.59340095141222,
            "pwidth1": 5.5e-05,
            "pwidth2": 5.5e-05,
            "distance": 0.8422249700427996,
            "tiltazimuth": 229.341164999341,
            "tilt": 2.6199557735894548,
            "detrot": -0.417948717948718,
            "outerangle_offset": 0.0,
        },
    }


@pytest.fixture(scope="session")
def cristal_params(cristal_data_path, tmp_path_factory):
    """
    Parameters for Cristal beamline test.

    Args:
        cristal_data_path: fixture providing Cristal data path
        tmp_path_factory: pytest fixture for temporary paths

    Returns:
        dict: complete parameter dictionary for BcdiPipeline
    """
    dump_dir = tmp_path_factory.mktemp("cristal_results")

    nxs_file = cristal_data_path / "mgomega-2022-11-20_22-37-18_0825.nxs"
    if not nxs_file.exists():
        pytest.skip(f"Cristal test file not found: {nxs_file}")

    return {
        "beamline_setup": "cristal",
        "sample_name": "cristal_test",
        "scan": 825,
        "dump_dir": str(dump_dir),
        "experiment_file_path": str(nxs_file),
        "voxel_reference_methods": ["max", "com", "com"],
        "hot_pixel_filter": True,
        "det_calib_params": {
            "cch1": 0,
            "cch2": 0,
            "pwidth1": 5.5e-05,
            "pwidth2": 5.5e-05,
            "distance": 1,
        },
    }


def pytest_configure(config):
    """
    Pytest hook for configuration.

    This adds custom markers and configures the test environment.
    """
    # add custom markers programmatically if needed
    config.addinivalue_line(
        "markers", "requires_pynx: tests that require PyNX to be installed"
    )


def pytest_collection_modifyitems(config, items):
    """
    Pytest hook to modify test collection.

    This automatically marks tests based on their location and imports.
    """
    for item in items:
        # mark tests in integration/ as integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)

        # mark tests in beamlines/ as beamline tests
        if "beamlines" in str(item.fspath):
            item.add_marker(pytest.mark.beamline)

        # mark GPU tests
        if "gpu" in item.nodeid.lower() or "pynx" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
            item.add_marker(pytest.mark.slow)
