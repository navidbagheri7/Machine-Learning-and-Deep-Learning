"""Single experiment entry point; JSON configs are copied into every run record."""

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study",
        choices=[
            "classification",
            "regression",
            "probabilistic",
            "generalization",
            "kernels",
            "signals",
            "vision",
        ],
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small integration run; saved separately from reference results",
    )
    args = parser.parse_args()
    path = args.config or Path("configs") / (args.study + ".json")
    config = json.loads(path.read_text())
    config["study"] = args.study
    if args.data_root:
        config["data_root"] = args.data_root
    if args.device:
        config["device"] = args.device
    if args.quick:
        config.update(
            quick=True,
            seeds=[0],
            epochs=2,
            ssl_epochs=1,
            full_epochs=2,
            samples=1000,
            repeats=2,
            gd_steps=200,
            train_blocks=128,
            test_blocks=64,
            label_budget=200,
        )
        if args.study == "generalization":
            config["samples"] = 40
        config["output"] = f"results/smoke/{args.study}"
    if args.output:
        config["output"] = str(args.output)
    if args.study in ["classification", "regression", "signals", "vision"]:
        import torch

        torch.set_num_threads(4)
        if config["device"].startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable. Use --device cpu or execute with GPU access."
            )
    with threadpool_limits(limits=4):
        if args.study in ["classification", "regression"]:
            from learning_lab import tabular

            getattr(tabular, args.study)(config)
        elif args.study in ["probabilistic", "generalization", "kernels"]:
            from learning_lab import studies

            getattr(studies, args.study)(config)
        elif args.study == "signals":
            from learning_lab.signals import experiment

            experiment(config)
        else:
            from learning_lab.vision import experiment

            experiment(config)


if __name__ == "__main__":
    main()
