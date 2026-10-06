"""Run independent studies concurrently with bounded CPU/GPU resource use."""

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def group(studies, quick):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    for name in studies:
        command = [sys.executable, "-m", "learning_lab.cli", name]
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
    parser.add_argument("--studies", nargs="+")
    args = parser.parse_args()
    groups = [
        ["probabilistic", "generalization", "kernels"],
        ["classification", "regression", "signals"],
        ["vision"],
    ]
    if args.studies:
        groups = [[s for s in g if s in args.studies] for g in groups]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(group, g, args.quick) for g in groups if g]
        for f in futures:
            f.result()


if __name__ == "__main__":
    main()
