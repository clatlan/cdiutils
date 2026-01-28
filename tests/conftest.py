"""
Pytest configuration and fixtures for CDIutils tests.

This module provides shared fixtures and configuration for testing the
CDIutils package, including test data management, temporary directories,
and common test objects.
"""

import os
import shutil
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

# test data paths
TEST_ROOT = Path(__file__).parent
CDIUTILS_ROOT = TEST_ROOT.parent / "src" / "cdiutils"

# test data directory - use environment variable or fallback to default
TEST_DATA_ROOT = Path(
    os.environ.get("CDIUTILS_TEST_DATA", "/scisoft/cdiutils_test_data")
)

# load test configuration (dataset metadata, NOT paths)
TEST_CONFIG_FILE = TEST_ROOT / "test_config.yaml"


def load_test_config():
    """
    Load test configuration from YAML file.

    This provides dataset metadata (scan numbers, detector names, etc.)
    but NOT file paths (those come from TEST_DATA_ROOT).

    Returns:
        dict: test configuration or empty dict if file doesn't exist
    """
    if TEST_CONFIG_FILE.exists():
        with open(TEST_CONFIG_FILE, "r") as f:
            return yaml.safe_load(f)
    return {}


# load config at module level
TEST_CONFIG = load_test_config()


def get_dataset_params(dataset_name: str, tmp_path_factory) -> dict:
    """
    Factory function to create pipeline parameters from test config.

    This reads dataset metadata from test_config.yaml and combines it
    with the TEST_DATA_ROOT path to build complete pipeline parameters.

    Args:
        dataset_name: Key from test_config.yaml datasets section
        tmp_path_factory: pytest fixture for temporary directories

    Returns:
        dict: Complete BcdiPipeline parameters

    Raises:
        pytest.skip: If dataset not in config or files not found
    """
    if "datasets" not in TEST_CONFIG:
        pytest.skip(
            f"No datasets defined in {TEST_CONFIG_FILE}. "
            "Cannot create test parameters."
        )

    if dataset_name not in TEST_CONFIG["datasets"]:
        pytest.skip(
            f"Dataset '{dataset_name}' not found in test_config.yaml. "
            f"Available: {list(TEST_CONFIG['datasets'].keys())}"
        )

    dataset = TEST_CONFIG["datasets"][dataset_name]

    # construct file paths directly from TEST_DATA_ROOT
    # (no intermediate "directory" layer)
    if "experiment_data_dir_path" in dataset:
        # P10 uses directory path, not file path
        experiment_path = TEST_DATA_ROOT / dataset["experiment_data_dir_path"]

        if not experiment_path.exists():
            pytest.skip(
                f"Test data directory not found: {experiment_path}\n"
                f"Set CDIUTILS_TEST_DATA environment variable or "
                f"place data at:\n  {TEST_DATA_ROOT}"
            )
        experiment_file = None
    else:
        # most beamlines use experiment_file_path
        experiment_file = TEST_DATA_ROOT / dataset["experiment_file_path"]
        experiment_path = experiment_file

        if not experiment_file.exists():
            pytest.skip(
                f"Test data not found: {experiment_file}\n"
                f"Set CDIUTILS_TEST_DATA environment variable or "
                f"place data at:\n  {TEST_DATA_ROOT}"
            )

    # create output directory
    dump_dir = tmp_path_factory.mktemp(f"{dataset_name}_results")

    # build parameters
    params = {
        "beamline_setup": dataset["beamline_setup"],
        "sample_name": dataset["sample_name"],
        "scan": dataset["scan"],
        "dump_dir": str(dump_dir),
    }

    # add path parameter (P10 uses experiment_data_dir_path)
    if "experiment_data_dir_path" in dataset:
        params["experiment_data_dir_path"] = str(experiment_path)
    else:
        params["experiment_file_path"] = str(experiment_file)

    # pass all other parameters from config to pipeline
    # exclude internal metadata keys used only for test organization
    exclude_keys = {
        "experiment_file_path",
        "experiment_data_dir_path",
        "description",
        "beamline_setup",
        "sample_name",
        "scan",
        "detector_data_path",  # handled separately for SPEC format
    }

    for key, value in dataset.items():
        if key not in exclude_keys and key not in params:
            # handle special case: detector_data_path for SPEC format
            if key == "detector_data_path":
                params["detector_data_path"] = str(TEST_DATA_ROOT / value)
            else:
                # pass parameter directly to pipeline
                params[key] = value

    return params


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
    Create a 3D complex array using simulation objects.

    Uses cdiutils.simulation.objects to create a small cylinder with
    correlated phase distribution, representative of real reconstructions.

    Returns:
        np.ndarray: complex 3D array
    """
    pytest.importorskip("cdiutils", reason="cdiutils required for simulation")
    from cdiutils.simulation.objects import add_random_phase, make_cylinder

    # create small cylinder
    shape = (20, 20, 20)
    cylinder_mask = make_cylinder(
        shape,
        radius=8,
        height=15,
        axis=2,
        value=1.0,
    )

    # add random phase with correlation
    obj_with_phase = add_random_phase(
        cylinder_mask,
        phase_std=0.3,
        correlation_length=5,
    )

    return obj_with_phase


@pytest.fixture
def sphere_data():
    """
    Create a 3D complex array with a spherical object using simulation.

    Uses cdiutils.simulation.objects functions to create a physically
    realistic sphere with correlated phase distribution.

    Returns:
        dict: containing 'data' (amplitude), 'complex_data', 'support',
            and metadata
    """
    pytest.importorskip("cdiutils", reason="cdiutils required for simulation")
    from cdiutils.simulation.objects import (
        add_random_phase,
        make_ellipsoid,
    )

    # create spherical object
    shape = (40, 40, 40)
    radius = 10

    # make sphere (ellipsoid with equal radii)
    sphere_mask = make_ellipsoid(shape, radii=radius, value=1.0)

    # add random phase with correlation to create realistic phase distribution
    obj_with_phase = add_random_phase(
        sphere_mask,
        phase_std=0.1,
        correlation_length=10,
    )

    # extract amplitude and support
    amplitude = np.abs(obj_with_phase)
    support = amplitude > 0
    centre = np.array(shape) // 2

    # add some realistic noise to amplitude for KDE compatibility
    amplitude[support] += np.random.normal(
        0, amplitude[support].mean() * 0.05, support.sum()
    )
    amplitude = np.maximum(amplitude, 0)  # ensure non-negative

    return {
        "data": amplitude,  # amplitude with variation
        "complex_data": obj_with_phase,  # complex data
        "support": support,
        "centre": centre,
        "radius": radius,
    }


@pytest.fixture
def mock_detector_data():
    """
    Create realistic detector data using BCDISimulator.

    Generates a small sphere and projects it to detector space with
    realistic noise.

    Returns:
        dict: containing detector data and associated metadata
    """
    cdiutils = pytest.importorskip(
        "cdiutils", reason="cdiutils required for simulation"
    )

    # create geometry and simulator
    geometry = cdiutils.geometry.Geometry.from_setup("id01")

    num_frames = 100

    simulator = cdiutils.simulation.BCDISimulator(
        geometry=geometry,
        det_calib_params={
            "cch1": 128.0,
            "cch2": 128.0,
            "pwidth1": 5.5e-05,
            "pwidth2": 5.5e-05,
            "distance": 1.0,
        },
        energuy=8000.0,
        detector_name="maxipix",
        num_frames=num_frames,
    )

    # create small sphere
    simulator.simulate_object(
        (60, 60, 60),
        voxel_size=10e-9,
        geometric_shape="ellipsoid",
        geometric_shape_params={"radii": (10, 10, 10)},
        phase_type="constant",
        phase_params={"phase_value": 0.0},
    )

    # set measurement parameters
    bragg_angle = simulator.lattice_parameter_to_bragg_angle(3.92e-10)
    detector_angles = simulator.get_detector_angles(
        scattering_angle=bragg_angle * 2
    )

    simulator.set_measurement_params(
        bragg_angle=bragg_angle,
        rocking_range=1.0,
        detector_angles=detector_angles,
    )

    # generate detector data
    detector_intensity = simulator.to_detector_frame(
        method="matrix_transform",
        output_shape=(num_frames, 400, 400),
    )

    # add realistic noise
    realistic_data = simulator.get_realistic_detector_data(
        detector_intensity,
        photon_budget=1e7,
        shift=True,
        noise_params=[
            dict(gaussian_mean=0.0, gaussian_std=0.1),
            dict(poisson_statistics=True),
        ],
    )

    # find peak position
    peak_frame = num_frames // 2
    peak_2d = realistic_data[peak_frame]
    peak_pos_2d = np.unravel_index(peak_2d.argmax(), peak_2d.shape)
    peak_pos = (peak_frame, peak_pos_2d[0], peak_pos_2d[1])

    return {
        "data": realistic_data,
        "peak_position": peak_pos,
        "shape": realistic_data.shape,
        "simulator": simulator,
    }


@pytest.fixture(scope="session")
def id01_bliss_params(tmp_path_factory):
    """
    Parameters for ID01 Bliss format test (primary test dataset).

    Loads from test_config.yaml::datasets::id01_bliss_dislocation
    """
    return get_dataset_params("id01_bliss_dislocation", tmp_path_factory)


@pytest.fixture(scope="session")
def id01_spec_params(tmp_path_factory):
    """
    Parameters for ID01 SPEC format test (legacy format).

    Loads from test_config.yaml::datasets::id01_spec_core_shell
    """
    return get_dataset_params("id01_spec_core_shell", tmp_path_factory)


@pytest.fixture(scope="session")
def nanomax_params(tmp_path_factory):
    """
    Parameters for NanoMAX beamline test.

    Loads from test_config.yaml::datasets::nanomax_sample
    """
    return get_dataset_params("nanomax_sample", tmp_path_factory)


@pytest.fixture(scope="session")
def p10_params(tmp_path_factory):
    """
    Parameters for P10 (PETRA III) beamline test.

    Loads from test_config.yaml::datasets::p10_sample
    """
    return get_dataset_params("p10_sample", tmp_path_factory)


@pytest.fixture(scope="session")
def sixs_params(tmp_path_factory):
    """
    Parameters for SIXS beamline test.

    Loads from test_config.yaml::datasets::sixs_2019
    """
    return get_dataset_params("sixs_2019", tmp_path_factory)


@pytest.fixture(scope="session")
def cristal_params(tmp_path_factory):
    """
    Parameters for Cristal beamline test.

    Loads from test_config.yaml::datasets::cristal_sample
    """
    return get_dataset_params("cristal_sample", tmp_path_factory)


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
        "requires_real_data: tests that require real beamline data",
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


@pytest.fixture(scope="session")
def gpu_pipeline_results(id01_bliss_params: dict) -> dict:
    """Load pre-computed GPU pipeline results for testing.

    This fixture loads results from the full GPU pipeline test
    (test_full_pipeline_real_data) without re-running it.
    It finds the saved parameters YAML file path.

    In CI:
    - GPU test runs first and saves to
      $CI_PROJECT_DIR/gpu_test_results/
    - This job uses artifacts from the GPU test job
    - Fast - just loads existing files

    Locally:
    - Looks for results in standard locations
    - Falls back to TEST_DATA_ROOT/gpu_pipeline_results/
    - Skips if not found

    Args:
        id01_bliss_params: Base pipeline parameters (not used,
            kept for compatibility).

    Returns:
        Dictionary containing:
            - param_file_path: Path to S*_parameters.yml file
            - dump_dir: Path to dump directory
            - pynx_dir: Path to phasing results
            - scan: Scan number (extracted from filename)
            - results_base: Base directory for results

    Raises:
        pytest.skip: If GPU pipeline results are not available.
    """
    # try multiple locations for results
    search_paths: list[Path] = []

    # 1. CI artifacts location
    ci_project_dir = os.environ.get("CI_PROJECT_DIR")
    if ci_project_dir:
        search_paths.append(Path(ci_project_dir) / "gpu_test_results")

    # 2. golden data location
    search_paths.append(TEST_DATA_ROOT / "gpu_pipeline_results")

    # 3. local test output (if test was run locally)
    search_paths.append(Path("/tmp") / "gpu_test_results")

    results_base: Path | None = None
    param_file: Path | None = None

    # search for parameter YAML file
    for path in search_paths:
        dump_dir = path / "dump"
        if dump_dir.exists():
            # find parameter file: S*_parameters.yml
            param_files = list(dump_dir.glob("S*_parameters.yml"))
            if param_files:
                results_base = path
                param_file = param_files[0]
                break

    if results_base is None or param_file is None:
        pytest.skip(
            "GPU pipeline results not found. Run "
            "test_full_pipeline_real_data first.\n"
            f"Searched locations: {[str(p) for p in search_paths]}"
        )

    # extract scan number from filename: S54_parameters.yml -> 54
    scan_str = param_file.stem.split("_")[0][1:]  # remove 'S' prefix
    scan: int = int(scan_str)

    # infer paths from parameter file location
    dump_dir = param_file.parent
    pynx_dir = dump_dir / "pynx_phasing"

    # find preprocessed CXI file
    preprocessed_cxi = dump_dir / f"S{scan}_preprocessed_data.cxi"

    # verify key files exist
    if not pynx_dir.exists():
        pytest.skip(f"GPU pipeline pynx_dir not found: {pynx_dir}")

    if not preprocessed_cxi.exists():
        pytest.skip(f"Preprocessed CXI file not found: {preprocessed_cxi}")

    return {
        "param_file_path": str(param_file),
        "preprocessed_cxi_path": str(preprocessed_cxi),
        "dump_dir": dump_dir,
        "pynx_dir": pynx_dir,
        "scan": scan,
        "results_base": results_base,
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
