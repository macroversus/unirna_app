import pytest
import subprocess
from pathlib import Path
from rna_app.bohr.rna_ss_bohr import ModelTypeOptions
from rna_app.core.utils import CHEKPOINTS


@pytest.mark.parametrize("mission", list(CHEKPOINTS.keys()))
def test_runner(mission):
    repo_root = Path(__file__).parent.parent
    cmd = [
        "rna_app_infer",
    ]
    if mission.startswith("ss_"):
        cmd.extend(
            [
                "--in_data",
                f"{repo_root}/example/unirna_ss/input.fasta",
                "--mission", "rna_ss",
                "--output_dir",
                f"{repo_root}/example/unirna_ss/core/outputs/{mission}",
                "--model_type",
                (
                    "unirna"
                    if mission == "ss_unirna"
                    else "archiveii"
                ),
            ]
        )
    else:
        cmd.extend(
            [
                "--in_data",
                f"{repo_root}/example/{mission}/input.fasta",
                "--mission", mission,
                "--output_dir",
                f"{repo_root}/example/{mission}/core/outputs",
            ]
        )
    ret = subprocess.run(
        cmd,
    )
    assert ret.returncode == 0, f"Failed to run {cmd}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
