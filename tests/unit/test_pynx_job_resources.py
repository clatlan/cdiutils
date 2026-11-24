"""
Unit tests for the memory and time of the PyNX SLURM jobs.
"""

import numpy as np
import pytest

from cdiutils.pipeline.bcdi import BcdiPipeline

CYCLES = {"nb_hio": 300, "nb_raar": 600, "nb_er": 400, "nb_ml": 0}


@pytest.mark.unit
class TestPynxJobResources:
    """Test the estimates from the shape of the data."""

    def test_read_shape_npz_and_npy(self, tmp_path):
        """Test the shape is read from the header of npz and npy files."""
        data = np.zeros((6, 5, 4), dtype=np.float32)
        np.savez(tmp_path / "data.npz", data=data)
        np.save(tmp_path / "data.npy", data)
        for name in ("data.npz", "data.npy"):
            shape = BcdiPipeline._read_array_shape(str(tmp_path / name))
            assert tuple(shape) == (6, 5, 4)

    def test_large_data(self, tmp_path, monkeypatch):
        """Test the CuAg_BCDI_0058 case: 600 x 500 x 500 voxels."""
        monkeypatch.setattr(
            BcdiPipeline,
            "_read_array_shape",
            staticmethod(lambda path: (600, 500, 500)),
        )
        memory, run_seconds = BcdiPipeline._estimate_pynx_job_resources(
            "data.npz", dict(CYCLES, rebin="1, 1, 1")
        )
        # Measured: 6.5 GB peak and about 2.5 min per run on an L40S
        assert memory == "15G"
        assert run_seconds == pytest.approx(150)

    def test_small_data_and_rebin(self, tmp_path, monkeypatch):
        """Test the minimum memory and the effect of rebinning."""
        monkeypatch.setattr(
            BcdiPipeline,
            "_read_array_shape",
            staticmethod(lambda path: (600, 500, 500)),
        )
        memory, run_seconds = BcdiPipeline._estimate_pynx_job_resources(
            "data.npz", dict(CYCLES, rebin="2, 2, 2")
        )
        assert memory == "8G"
        assert run_seconds == pytest.approx(150 / 8)

    def test_unreadable_data(self):
        """Test a missing file gives a safe default."""
        memory, run_seconds = BcdiPipeline._estimate_pynx_job_resources(
            "/no/such/file.npz", CYCLES
        )
        assert (memory, run_seconds) == ("16G", 0.0)

    @pytest.mark.parametrize(
        "time_limit, seconds",
        [
            ("01:00:00", 3600),
            ("45:00", 2700),
            ("30", 1800),
            ("1-02:00:00", 93600),
        ],
    )
    def test_time_to_seconds(self, time_limit, seconds):
        """Test the SLURM time formats."""
        assert BcdiPipeline._time_to_seconds(time_limit) == seconds
