import subprocess
import sys
from pathlib import Path


def test_pipeline_smoke_cli():
    root = Path(__file__).resolve().parents[1]
    py = sys.executable

    for cmd in (
        [py, "main.py", "preprocess"],
        [py, "main.py", "train"],
        [py, "main.py", "evaluate"],
        [py, "main.py", "predict"],
    ):
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        assert (
            proc.returncode == 0
        ), f"failed: {' '.join(cmd)}\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
