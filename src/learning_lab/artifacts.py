"""Experiment records with configs, versions, elapsed time and dataset fingerprints."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


def fingerprint(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str((a.shape, a.dtype)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def write_json(path, data):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value))

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Refuse NaN instead of silently publishing invalid metrics.
    p.write_text(json.dumps(data, indent=2, default=convert, allow_nan=False) + "\n")


class Run:
    def __init__(self, output, config):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.start = time.monotonic()
        versions = {}
        for package in ["numpy", "scipy", "pandas", "scikit-learn", "torch", "torchvision"]:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        self.record = {
            "config": config,
            "versions": versions,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": git.stdout.strip() or None,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        source = Path(__file__).resolve().parent
        self.record["package_source_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.glob("*.py"))
        }
        if config.get("device", "").startswith("cuda"):
            import torch

            self.record["accelerator"] = {
                "name": torch.cuda.get_device_name(config["device"]),
                "cuda_runtime": torch.version.cuda,
            }
        write_json(self.output / "run.json", self.record | {"status": "running"})

    def finish(self, results):
        write_json(self.output / "metrics.json", results)
        write_json(
            self.output / "run.json",
            self.record | {"status": "complete", "elapsed_seconds": time.monotonic() - self.start},
        )
