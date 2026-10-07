"""Run the complete GAN objective study on three explicitly selected devices."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adversarial_variants.json")
    parser.add_argument("--devices", nargs=3, default=["cuda:0", "cuda:1", "cuda:2"])
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    tasks = queue.Queue()
    resumed = {device: queue.Queue() for device in args.devices}
    root = Path(__file__).resolve().parents[1]
    for task in config["tasks"]:
        for method in config["methods"]:
            for seed in config["seeds"]:
                directory = root / config["output"] / task / method / str(seed)
                record_path = directory / "run.json"
                if record_path.exists():
                    record = json.loads(record_path.read_text())
                    protocol = record["config"]
                    assert protocol["benchmark"] == config
                    assert (protocol["task"], protocol["method"], protocol["seed"]) == (
                        task,
                        method,
                        seed,
                    )
                    for name, value in record["fitting_source_sha256"].items():
                        assert (
                            hashlib.sha256(
                                (root / "src/learning_lab" / name).read_bytes()
                            ).hexdigest()
                            == value
                        )
                    if record["status"] == "complete":
                        print("Preserved complete fit", task, method, seed, flush=True)
                        continue
                    if protocol["device"] not in resumed:
                        raise ValueError(f"Resume device {protocol['device']} must be in --devices")
                    resumed[protocol["device"]].put((task, method, seed))
                else:
                    tasks.put((task, method, seed))
    failed = threading.Event()
    env = os.environ | dict(
        PYTHONPATH=str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        OPENBLAS_NUM_THREADS="2",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
    )

    def worker(device):
        while not failed.is_set():
            active = resumed[device] if not resumed[device].empty() else tasks
            try:
                task, method, seed = active.get_nowait()
            except queue.Empty:
                return
            directory = Path(config["output"]) / task / method / str(seed)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "training.log").open("a") as log:
                code = subprocess.run(
                    [
                        sys.executable,
                        "scripts/train_adversarial.py",
                        "--config",
                        args.config,
                        "--task",
                        task,
                        "--method",
                        method,
                        "--seed",
                        str(seed),
                        "--device",
                        device,
                    ],
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ).returncode
            print(task, method, seed, device, "exit", code, flush=True)
            active.task_done()
            if code:
                failed.set()
                raise RuntimeError(f"Failed {directory}; inspect its training.log")

    with concurrent.futures.ThreadPoolExecutor(3) as executor:
        futures = [executor.submit(worker, device) for device in args.devices]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    assert tasks.empty(), "Study not complete"
    assert all(q.empty() for q in resumed.values()), "Resume study not complete"
    print("All twenty adversarial restoration fits completed", flush=True)


if __name__ == "__main__":
    main()
