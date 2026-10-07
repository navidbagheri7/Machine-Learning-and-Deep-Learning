"""Score all fixed final EMA endpoints to expose adversarial changes without test selection."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from learning_lab.artifacts import Run, write_json
from learning_lab.generative import build_model, evaluate, save_arrays, save_checkpoint


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adversarial_endpoints.json")
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    endpoint = json.loads(Path(args.config).read_text())
    source_config = json.loads(Path(endpoint["source_config"]).read_text())
    torch.set_num_threads(2)
    torch.hub.set_dir(str(Path(source_config["data_root"]) / "models/torch/hub"))
    sources = {
        name: digest(name)
        for name in [
            "scripts/evaluate_adversarial_endpoints.py",
            "src/learning_lab/generative.py",
            "src/learning_lab/restoration.py",
            "src/learning_lab/artifacts.py",
        ]
    }
    public = Path("assets/source_data/adversarial_endpoints")
    public.mkdir(parents=True, exist_ok=True)
    for task in source_config["tasks"]:
        for method in source_config["methods"]:
            for seed in source_config["seeds"]:
                training = Path(source_config["output"]) / task / method / str(seed)
                record = json.loads((training / "run.json").read_text())
                assert (
                    record["status"] == "complete"
                    and record["config"]["benchmark"] == source_config
                )
                for name, value in record["fitting_source_sha256"].items():
                    assert digest(Path("src/learning_lab") / name) == value
                metrics = json.loads((training / "metrics.json").read_text())
                directory = Path(endpoint["output"]) / task / method / str(seed)
                protocol = dict(
                    endpoint_config=endpoint,
                    training_config=source_config,
                    task=task,
                    method=method,
                    seed=seed,
                    device=args.device,
                )
                path = directory / "run.json"
                if path.exists():
                    old = json.loads(path.read_text())
                    assert old["config"] == protocol and old["evaluation_source_sha256"] == sources
                    if old["status"] == "complete":
                        print("Preserved endpoint", task, method, seed, flush=True)
                        continue
                state_path = training / "checkpoints/training.pt"
                state = torch.load(state_path, map_location="cpu", weights_only=True)
                assert state["protocol"] == record["config"]
                assert len(state["history"]) == source_config["epochs"] + 1
                assert state["history"][-1]["epoch"] == source_config["epochs"] == 20
                model = (
                    build_model("l1", task, source_config, initialize_pretrained=False)
                    .to(args.device)
                    .eval()
                )
                model.load_state_dict(state["ema"], strict=True)
                model.requires_grad_(False)
                run = Run(directory, protocol)
                run.record["evaluation_source_sha256"] = sources
                write_json(path, run.record | dict(status="running"))
                checkpoint = directory / "checkpoints/final_epoch.pt"
                checkpoint.parent.mkdir(exist_ok=True)
                save_checkpoint(
                    dict(
                        state_dict=state["ema"],
                        config=source_config | dict(task=task),
                        method=method,
                        seed=seed,
                        best_epoch=20,
                        checkpoint_policy=endpoint["checkpoint_policy"],
                    ),
                    checkpoint,
                )
                alias = method + "_final_epoch"
                with torch.no_grad():
                    rows, scores, gallery = evaluate(
                        model, alias, seed, source_config | dict(task=task, device=args.device)
                    )
                stem = f"{task}_{method}_{seed}"
                save_arrays(public / (stem + "_scores.npz"), scores)
                display = {
                    key: (values.clip(0, 1) * 255).round().astype(np.uint8)
                    for key, values in gallery.items()
                }
                save_arrays(public / (stem + "_gallery.npz"), display)
                run.finish(
                    dict(
                        rows=rows,
                        final_epoch=20,
                        validation_loss=metrics["history"][-1]["validation_loss"],
                        selected_epoch_in_training=metrics["best_epoch"],
                        checkpoint_sha256=digest(checkpoint),
                        training_state_sha256=digest(state_path),
                        training_metrics_sha256=digest(training / "metrics.json"),
                        warm_start_sha256=record["warm_start_sha256"],
                        numeric_sha256={
                            suffix: digest(public / (stem + suffix + ".npz"))
                            for suffix in ["_scores", "_gallery"]
                        },
                        protocol=endpoint["scope"],
                    )
                )
                print(
                    "Scored fixed endpoint",
                    task,
                    method,
                    seed,
                    rows[0]["psnr"]["mean"],
                    rows[0]["lpips"]["mean"],
                    flush=True,
                )
                del state, model


if __name__ == "__main__":
    main()
