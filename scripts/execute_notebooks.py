"""Execute all walkthroughs using this Python, without modifying global Jupyter kernels."""

import json
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    jupyter = ROOT / ".local" / "jupyter"
    kernel = jupyter / "kernels" / "learning-lab"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Learning Lab validation",
                "language": "python",
            }
        )
    )
    os.environ["JUPYTER_PATH"] = str(jupyter) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        nb = NotebookClient(
            nb, timeout=180, kernel_name="learning-lab", resources={"metadata": {"path": str(ROOT)}}
        ).execute()
        # Public metadata stays portable rather than requiring the local validation kernel.
        nb.metadata.kernelspec = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        nbformat.write(nb, path)
        print("Executed", path.name, flush=True)


if __name__ == "__main__":
    main()
