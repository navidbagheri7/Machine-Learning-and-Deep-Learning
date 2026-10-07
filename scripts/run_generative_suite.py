"""Run independent restoration comparisons on three GPUs, with epoch-level resume."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILURE = threading.Event()


def worker(index, methods, devices, config_dir, output_root, data_root):
    for task in ["colorization", "super_resolution"]:
        for method in methods:
            config = json.loads((config_dir / f"generative_{task}.json").read_text())
            if task == "colorization" and method == "cvae":
                config = json.loads((config_dir / "generative_colorization_cvae.json").read_text())
                config["mean_checkpoint"] = str(
                    output_root / "generative_colorization/l1/checkpoints/l1_0.pt"
                )
                while not (ROOT / config["mean_checkpoint"]).exists():
                    if FAILURE.is_set():
                        raise RuntimeError("Mean-model training failed; cannot initialize CVAE")
                    time.sleep(10)
            config.update(
                methods=[method],
                device=devices[index],
                output=str(output_root / f"generative_{task}/{method}"),
            )
            if data_root:
                config["data_root"] = str(data_root)
                config["pretrained_weights"] = str(
                    data_root / "models/torch/hub/checkpoints/resnet18-f37072fd.pth"
                )
            output = ROOT / config["output"]
            output.mkdir(parents=True, exist_ok=True)
            record = output / "run.json"
            if record.exists():
                old = json.loads(record.read_text())
                checkpoints_present = method == "baseline" or all(
                    (output / "checkpoints" / f"{method}_{seed}.pt").exists()
                    for seed in config["seeds"]
                )
                if (
                    old.get("status") == "complete"
                    and old["config"] == config
                    and checkpoints_present
                ):
                    print("Already complete", task, method, flush=True)
                    continue
            path = output / "config.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
            environment = os.environ | {
                "OPENBLAS_NUM_THREADS": "4",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "PYTHONPATH": str(ROOT / "src")
                + os.pathsep
                + str(ROOT / ".local" / "upgrade_deps"),
            }
            with (output / "training.log").open("a") as log:
                completed = subprocess.run(
                    [sys.executable, "-m", "learning_lab.generative", "--config", str(path)],
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                if completed.returncode:
                    FAILURE.set()
                    raise RuntimeError(f"{task}/{method} failed; inspect {output / 'training.log'}")
            print("Complete", task, method, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs=3, default=["cuda:0", "cuda:1", "cuda:2"])
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output-root", type=Path, default=Path("results/reference"))
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(3) as executor:
        futures = [
            executor.submit(
                worker, i, methods, args.devices, args.config_dir, args.output_root, args.data_root
            )
            for i, methods in enumerate([["baseline", "l1", "gan"], ["cvae"], ["diffusion"]])
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
