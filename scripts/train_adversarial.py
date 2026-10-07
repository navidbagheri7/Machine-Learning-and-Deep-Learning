"""Train one controlled GAN restoration variant with epoch-level resume."""

import argparse
import hashlib
import json
from pathlib import Path

from learning_lab.adversarial import METHODS, train
from learning_lab.artifacts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adversarial_variants.json")
    parser.add_argument("--task", choices=["colorization", "super_resolution"], required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.data_root:
        config["data_root"] = args.data_root
        config["pretrained_weights"] = str(
            Path(args.data_root) / "models/torch/hub/checkpoints/resnet18-f37072fd.pth"
        )
    if args.quick:
        config |= dict(
            quick=True,
            epochs=1,
            train_images=32,
            validation_images=8,
            test_images=8,
            crop=32,
            batch_size=8,
            workers=0,
            output="results/smoke/adversarial_variants",
        )
    record_path = Path(config["output"]) / args.task / args.method / str(args.seed) / "run.json"
    previous = json.loads(record_path.read_text()) if record_path.exists() else None
    if previous is not None and previous["status"] != "complete":
        assert previous["config"] == dict(
            benchmark=config, task=args.task, method=args.method, seed=args.seed, device=args.device
        )
        for name, value in previous["fitting_source_sha256"].items():
            source = Path(__file__).resolve().parents[1] / "src/learning_lab" / name
            assert hashlib.sha256(source.read_bytes()).hexdigest() == value
    try:
        train(config, args.task, args.method, args.seed, args.device)
    finally:
        if previous is not None and previous["status"] != "complete" and record_path.exists():
            current = json.loads(record_path.read_text())
            if current["config"] == previous["config"]:
                current["prior_attempts"] = previous.get("prior_attempts", []) + [
                    {key: value for key, value in previous.items() if key != "prior_attempts"}
                ]
                current["timing_scope"] = (
                    "elapsed_seconds covers the latest invocation; epoch histories include earlier completed epochs"
                )
                write_json(record_path, current)


if __name__ == "__main__":
    main()
