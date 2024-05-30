import pytest
import subprocess
from pathlib import Path
from rna_app.core.utils import CHEKPOINTS

@pytest.mark.parametrize("mission", list(CHEKPOINTS.keys()))
def test_runner(mission):
    repo_root = Path(__file__).parent.parent
    cmd = [
            "python",
            f"{repo_root}/rna_app/bohr/{mission}_bohr.py",
        ]
    if mission.startswith("ss_"):
        cmd.extend(
            [
                "--input_data", f"{repo_root}/example/unirna_ss/input.fasta",
                "--output_dir", f"{repo_root}/example/unirna_ss/outputs/{mission}",
                "--model_type", "unirna" if mission == "ss_unirna" else "archiveii",
            ]
        )
    else:
        cmd.extend(
            [
                "--input_data", f"{repo_root}/example/{mission}/input.fasta",
                "--output_dir", f"{repo_root}/example/{mission}/outputs",
            ]
        )
    ret = subprocess.run(
        cmd,
    )
    assert ret.returncode == 0, f"Failed to run {cmd}"

if __name__ == '__main__':
    pytest.main(['-v', __file__])