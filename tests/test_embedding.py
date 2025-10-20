import pytest
import subprocess
from pathlib import Path
from rna_app.core.utils import PRETRAINED


@pytest.mark.parametrize("pretrained", list(PRETRAINED.values()))
def test_runner(pretrained: str):
    repo_root = Path(__file__).parent.parent
    cmd = [
        "python",
        f"{repo_root}/rna_app/bohr/extract_embedding_bohr.py",
        "--input_data",
        f"{repo_root}/example/extract_embedding/input.fasta",
        "--output_dir",
        f"{repo_root}/example/extract_embedding/outputs/{pretrained}",
        "--pretrained",
        Path(pretrained).name,
    ]
    ret = subprocess.run(
        cmd,
    )
    assert ret.returncode == 0, f"Failed to run {cmd}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
