"""Run the thirteen foundational studies with bounded CPU/GPU resource use."""

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def group(studies, quick, config_dir, device):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
        env[variable] = "4"
    for name in studies:
        command = [sys.executable, "-m", "learning_lab.cli", name]
        command += ["--config", str(config_dir / f"{name}.json")]
        if device:
            command += ["--device", device]
        if quick:
            command.append("--quick")
        path = ROOT / ".local" / "logs" / (("smoke_" if quick else "reference_") + name + ".log")
        path.parent.mkdir(parents=True, exist_ok=True)
        print("Starting", name, "log:", path, flush=True)
        with path.open("w") as log:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError(f"{name} failed. See {path}")
        print("Completed", name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    groups = [
        ["probabilistic", "generalization", "kernels", "course_mixtures", "inverse"],
        ["classification", "regression", "signals", "denoising"],
        ["vision", "super_resolution", "colorization", "ddcolor"],
    ]
    parser.add_argument(
        "--studies", nargs="+", choices=[name for group in groups for name in group]
    )
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--device", help="Optional device override for every selected study")
    args = parser.parse_args()
    if args.studies:
        groups = [[s for s in g if s in args.studies] for g in groups]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(group, g, args.quick, args.config_dir, args.device) for g in groups if g
        ]
        for f in futures:
            f.result()


if __name__ == "__main__":
    main()
