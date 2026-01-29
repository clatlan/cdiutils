# CDIutils Testing Guide

## Quick Start

```bash
# Run all tests (requires test data)
pytest tests/

# Run only simulation tests (no external data needed)
pytest tests/ -m simulation -v

# Run specific test category
pytest tests/unit -v                    # Unit tests only
pytest tests/integration -m id01 -v     # ID01 integration tests
```

## Test Data

Real beamline data location: `/scisoft/cdiutils_test_data/`

Override with: `export CDIUTILS_TEST_DATA=/path/to/data`

## Test Categories

| Type | Location | Requirements | Markers |
|------|----------|--------------|---------|
| **Unit** | `tests/unit/` | None | `@pytest.mark.unit` |
| **Simulation** | `tests/integration/test_pipeline_simulated.py` | cdiutils only | `@pytest.mark.simulation` |
| **ID01 Bliss** | `tests/integration/test_pipeline_*.py` | Real data | `@pytest.mark.id01` |
| **GPU/PyNX** | Any with GPU processing | PyNX + GPU | `@pytest.mark.gpu` |

## Simulation-Based Testing

The `simulated_id01_data` fixture (in `conftest.py`) generates realistic detector data:
- Complete ID01 HDF5 file structure
- 256-frame rocking curve
- Cylindrical nanoparticle with known properties
- No external data required

**Example usage:**
```python
@pytest.mark.simulation
def test_preprocessing(simulated_id01_data):
    params = {
        "beamline_setup": "id01",
        "experiment_file_path": str(simulated_id01_data["experiment_file"]),
        "sample_name": simulated_id01_data["sample_name"],
        "scan": simulated_id01_data["scan_number"],
    }
    pipeline = BcdiPipeline(params=params)
    pipeline.preprocess()
```

## Key Fixtures

| Fixture | Purpose | Scope |
|---------|---------|-------|
| `simulated_id01_data` | Complete simulated ID01 experiment | session |
| `id01_bliss_params` | Parameters for ID01 Bliss format test data | session |
| `tmp_path` | Temporary directory (pytest built-in) | function |
| `sphere_data` | 3D sphere for testing support/isosurface | function |
| `mock_detector_data` | Simple Gaussian Bragg peak | function |

## Common Test Commands

```bash
# Fast tests (no slow integration tests)
pytest tests/ -m "not slow" -v

# Simulation tests only (no external data)
pytest tests/ -m simulation -v

# All except GPU tests
pytest tests/ -m "not gpu" -v

# Specific test file
pytest tests/integration/test_pipeline_preprocessing.py -v

# Single test
pytest tests/unit/test_utils.py::test_specific_function -v
```

## Adding New Tests

1. **Testing a utility function?** → `tests/unit/test_*.py`
2. **Testing pipeline with known ground truth?** → `tests/integration/test_pipeline_simulated.py`
3. **Testing with real beamline data?** → `tests/integration/test_pipeline_*.py`

Mark tests appropriately: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.simulation`

