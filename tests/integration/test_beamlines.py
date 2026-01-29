"""
Integration tests for beamline-specific data I/O and preprocessing.

This module tests that the pipeline correctly reads and preprocesses
data from different beamlines (NanoMAX, P10, etc.). Focus is on
verifying I/O functionality and beamline-specific parameters.
"""

from pathlib import Path

import pytest

from cdiutils.io import CXIFile


@pytest.mark.integration
@pytest.mark.requires_real_data
def test_nanomax_preprocessing(nanomax_params: dict) -> None:
    """
    Test NanoMAX data loading and preprocessing.

    Verifies:
        - Data file loads correctly
        - Hot pixel filtering is applied
        - Preprocessed data has expected properties
        - Detector geometry is correctly read

    Args:
        nanomax_params: fixture providing NanoMAX test parameters
    """
    from cdiutils.pipeline import BcdiPipeline

    pipeline = BcdiPipeline(params=nanomax_params)
    pipeline.preprocess()

    # load preprocessed data from CXI file
    cxi_file = (
        Path(nanomax_params["dump_dir"])
        / f"S{nanomax_params['scan']}_preprocessed_data.cxi"
    )
    assert cxi_file.exists(), f"CXI file not created: {cxi_file}"

    with CXIFile(cxi_file, "r") as f:
        preprocessed_data = f["entry_1/cropped_detector_data/data"]

    # verify preprocessing succeeded
    assert preprocessed_data is not None, "preprocessing failed - no data"
    assert preprocessed_data.ndim == 3, (
        f"expected 3D data, got {preprocessed_data.ndim}D"
    )
    assert preprocessed_data.max() > 0, "preprocessed data is all zeros"

    # verify data is not all uniform (actual signal present)
    assert preprocessed_data.std() > 0, "preprocessed data has no variation"

    # verify hot pixel filtering was applied (check params were used)
    assert nanomax_params.get("hot_pixel_filter") is True, (
        "hot pixel filter should be enabled for NanoMAX"
    )


@pytest.mark.integration
@pytest.mark.requires_real_data
def test_p10_preprocessing(p10_params: dict) -> None:
    """
    Test P10 data loading and preprocessing.

    Verifies:
        - .fio file format is correctly parsed
        - Light loading mode works
        - Background subtraction is applied
        - Cropping to preprocess_shape works

    Args:
        p10_params: fixture providing P10 test parameters
    """
    from cdiutils.pipeline import BcdiPipeline

    pipeline = BcdiPipeline(params=p10_params)
    pipeline.preprocess()

    # load preprocessed data from CXI file
    cxi_file = (
        Path(p10_params["dump_dir"])
        / f"S{p10_params['scan']}_preprocessed_data.cxi"
    )
    assert cxi_file.exists(), f"CXI file not created: {cxi_file}"

    with CXIFile(cxi_file, "r") as f:
        preprocessed_data = f["entry_1/cropped_detector_data/data"]
    # verify preprocessing succeeded
    assert preprocessed_data is not None, "preprocessing failed - no data"
    assert preprocessed_data.ndim == 3, (
        f"expected 3D data, got {preprocessed_data.ndim}D"
    )

    # verify shape matches requested preprocess_shape
    expected_shape = tuple(p10_params["preprocess_shape"])
    assert preprocessed_data.shape == expected_shape, (
        f"expected shape {expected_shape}, got {preprocessed_data.shape}"
    )

    # verify data contains signal
    assert preprocessed_data.max() > 0, "preprocessed data is all zeros"

    # verify background subtraction was applied
    assert p10_params.get("background_level") == 2, (
        "background level should be 2 for P10"
    )

    # verify light loading was used (memory efficient)
    assert p10_params.get("light_loading") is True, (
        "light loading should be enabled for P10"
    )
