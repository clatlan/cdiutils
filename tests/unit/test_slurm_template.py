"""
Unit tests for the PyNX SLURM template of the BCDI pipeline.
"""

import os
import subprocess
from string import Template

import pytest

import cdiutils.pipeline

TEMPLATE_PATH = os.path.join(
    os.path.dirname(cdiutils.pipeline.__file__),
    "pynx-id01-cdi_template.slurm",
)


def render(**values) -> str:
    """
    Fill the template as BcdiPipeline._make_slurm_file does.

    Args:
        **values: values overriding the defaults used here.

    Returns:
        The text of the SLURM script.
    """
    mapping = {
        "number_of_nodes": 2,
        "data_path": "/data/pynx_phasing/",
        "partition": "gpu,p9gpu",
        "pynx_env_x86_64": "",
        "pynx_env_ppc64le": "",
        "SLURM_JOBID": "$SLURM_JOBID",
        "SLURM_NTASKS": "$SLURM_NTASKS",
    }
    mapping.update(values)
    with open(TEMPLATE_PATH, "r", encoding="utf8") as file:
        return Template(file.read()).substitute(mapping)


@pytest.mark.unit
class TestSlurmTemplate:
    """Test the template is filled into a valid SLURM script."""

    def test_values_are_filled(self):
        """Test the cdiutils values end up in the script."""
        text = render(partition="p9gpu", pynx_env_ppc64le="/envs/p9_pynx")
        assert "#SBATCH --partition=p9gpu" in text
        assert 'ppc64le) pynx_env="/envs/p9_pynx"' in text
        assert 'cd "/data/pynx_phasing/"' in text

    def test_clean_environment(self):
        """Test the job does not inherit the submission environment."""
        assert "#SBATCH --export=NONE" in render()

    def test_bash_variables_are_kept(self):
        """Test the bash variables are not taken as placeholders."""
        text = render()
        assert "arch=$(uname -m)" in text
        assert 'case "$arch" in' in text

    def test_bash_syntax(self, tmp_path):
        """Test the filled script is valid bash."""
        path = tmp_path / "pynx-id01-cdi.slurm"
        path.write_text(render())
        subprocess.run(["bash", "-n", str(path)], check=True)
