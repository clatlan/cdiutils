"""
Pytest configuration and fixtures for CDIutils tests.

This module provides shared fixtures and configuration for testing the
CDIutils package, including test data management, temporary directories,
and common test objects.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import yaml

# use non-interactive backend for tests (prevents plot windows in CI)
matplotlib.use("Agg")

# import simulation utilities
from fixtures.simulate_id01_data import create_id01_experiment_file

# Test data paths
TEST_ROOT = Path(__file__).parent
CDIUTILS_ROOT = TEST_ROOT.parent / "src" / "cdiutils"

# test data directory - use environment variable or fallback to default
TEST_DATA_ROOT = Path(
    os.environ.get("CDIUTILS_TEST_DATA", "/scisoft/cdiutils_test_data")
)

# load test configuration
TEST_CONFIG_FILE = TEST_ROOT / "test_config.yaml"


def load_test_config():
    """
    Load test configuration from YAML file.

    Returns:
        dict: test configuration or empty dict if file doesn't exist
    """
    if TEST_CONFIG_FILE.exists():
        with open(TEST_CONFIG_FILE, "r") as f:
            return yaml.safe_load(f)
    return {}


@pytest.fixture(scope="session")
def test_data_path():
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
def mock_pipeline_params(tmp_path):
    """
    Provide a minimal set of parameters for BcdiPipeline testing.

    Uses pytest's tmp_path fixture to provide a temporary dump_dir
    that is automatically cleaned up after the test.

    Args:
        tmp_path: pytest fixture providing temporary directory

    Returns:
        dict: minimal parameter dictionary with valid dump_dir
    """
    return {
        "beamline_setup": "id01",
        "scan": 1,
        "sample_name": "test_sample",
        "dump_dir": str(tmp_path / "test_output"),
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

    # create sphere with good signal-to-noise ratio
    support = dist_from_centre <= radius
    data[support] = 100.0  # Strong signal

    # add realistic noise (much smaller than signal)
    data += np.random.normal(0, 5.0, shape)
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
def id01_bliss_params(tmp_path_factory):
    """
    Parameters for ID01 Bliss format test (primary test dataset).

    Uses the dislocation dataset at the fixed test data location.

    Args:
        tmp_path_factory: pytest fixture for temporary paths

    Returns:
        dict: complete parameter dictionary for BcdiPipeline
    """
    dump_dir = tmp_path_factory.mktemp("id01_bliss_results")

    experiment_file = (
        TEST_DATA_ROOT / "id01" / "dislocation" / "ihhc3936_id01.h5"
    )

    if not experiment_file.exists():
        pytest.skip(
            f"ID01 Bliss test file not found: {experiment_file}\n"
            f"Expected location: {TEST_DATA_ROOT}"
        )

    return {
        "beamline_setup": "id01",
        "sample_name": "PtYSZ_0001",
        "scan": 54,
        "dump_dir": str(dump_dir),
        "experiment_file_path": str(experiment_file),
        "det_calib_params": None,
        "voxel_size": 12,
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
    config.addinivalue_line(
        "markers",
        "simulation: tests using simulated experimental data",
    )


@pytest.fixture(scope="session")
def simulated_id01_data(tmp_path_factory):
    """
    Fixture providing complete simulated ID01 experiment for testing.

    This fixture generates realistic detector data using BCDISimulator
    and creates a full ID01 HDF5 file structure. The simulated data
    includes a synthetic object with known properties, allowing
    verification of the pipeline's performance.

    Args:
        tmp_path_factory: pytest fixture for temporary paths.

    Returns:
        dict: containing paths and metadata for the simulated
            experiment:
            - experiment_file: Path to master HDF5 file
            - sample_name: Name of the sample
            - dataset_name: Name of the dataset
            - scan_number: Scan number
            - detector_name: Detector name
            - dump_dir: Directory for pipeline outputs
            - num_frames: Number of frames in rocking curve
            - detector_shape: Detector dimensions
            - simulation_params: Parameters used for simulation
    """
    # lazy import to avoid circular dependencies
    cdiutils = pytest.importorskip(
        "cdiutils", reason="cdiutils required for simulation"
    )

    # simulation parameters (matching the notebook)
    num_frames = 256
    rocking_range = 1.2  # degrees
    lattice_parameter = 3.9236566724954263e-10  # metres
    energy = 8999.999342027731
    hkl = [1, 1, 1]
    eta_shift = 0.37821834  # degrees

    # target experimental Bragg peak position
    target_peak_position = (166, 365)  # vertical, horizontal pixels

    # detector geometry
    detector_shape = (516, 516)
    detector_name = "mpx1x4"

    det_calib_params = {
        "cch1": 183.668,  # direct beam vertical position
        "cch2": 239.286,  # direct beam horizontal position
        "pwidth1": 5.5e-05,
        "pwidth2": 5.5e-05,
        "distance": 0.904102,
    }

    # create geometry
    geometry = cdiutils.geometry.Geometry.from_setup("id01")

    # create simulator
    simulator = cdiutils.simulation.BCDISimulator(
        geometry,
        energy=energy,
        det_calib_params=det_calib_params,
        target_peak_position=target_peak_position,
        detector_name="maxipix",
        num_frames=num_frames,
    )

    # simulate object
    simulator.simulate_object(
        (350, 350, 350),
        voxel_size=10e-9 * 1 / 4,
        geometric_shape="cylinder",
        geometric_shape_params={
            "radius": 45,
            "height": 100,
            "axis": 2,
        },
        phase_type="random",
        phase_params={"phase_std": 0.5, "correlation_length": 15},
    )

    # compute diffractometer angles
    bragg_angle = simulator.lattice_parameter_to_bragg_angle(lattice_parameter)
    detector_angles = simulator.get_detector_angles(
        scattering_angle=bragg_angle * 2,
    )

    # set measurement parameters
    simulator.set_measurement_params(
        bragg_angle=bragg_angle,
        rocking_range=rocking_range,
        detector_angles=detector_angles,
    )

    # generate detector frame intensity
    detector_frame_intensity = simulator.to_detector_frame(
        method="matrix_transform",
        output_shape=(num_frames, 750, 750),
    )

    # add realistic noise
    realistic_detector_data = simulator.get_realistic_detector_data(
        detector_frame_intensity,
        photon_budget=2e7,
        shift=True,
        noise_params=[
            dict(gaussian_mean=0.0, gaussian_std=0.0),
            dict(poisson_statistics=True),
        ],
    )

    # create output directory
    output_dir = tmp_path_factory.mktemp("simulated_id01_data")
    dump_dir = tmp_path_factory.mktemp("simulated_results")

    # experiment metadata
    experiment_name = "test_exp"
    sample_name = "SimSample_0001"
    dataset_name = "SimSample_0001"
    scan_number = 1

    # create ID01 file structure
    experiment_file = create_id01_experiment_file(
        output_dir=str(output_dir),
        experiment_name=experiment_name,
        sample_name=sample_name,
        dataset_name=dataset_name,
        scan_number=scan_number,
        detector_name=detector_name,
        detector_data=realistic_detector_data,
        num_frames=num_frames,
        energy=energy,
        det_calib_params=det_calib_params,
        motor_positions={
            "eta": (
                simulator.diffractometer_angles["sample_outofplane_angle"]
                + eta_shift
            ),
            "phi": simulator.diffractometer_angles["sample_inplane_angle"],
            "delta": simulator.diffractometer_angles[
                "detector_outofplane_angle"
            ],
            "nu": simulator.diffractometer_angles["detector_inplane_angle"],
        },
    )

    return {
        "experiment_file": experiment_file,
        "sample_name": sample_name,
        "dataset_name": dataset_name,
        "scan_number": scan_number,
        "detector_name": detector_name,
        "dump_dir": dump_dir,
        "num_frames": num_frames,
        "detector_shape": detector_shape,
        "simulation_params": {
            "energy": energy,
            "lattice_parameter": lattice_parameter,
            "hkl": hkl,
            "rocking_range": rocking_range,
            "bragg_angle": bragg_angle,
            "target_peak_position": target_peak_position,
        },
        "simulator": simulator,  # for ground truth comparison
    }


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
