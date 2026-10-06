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
            "course_mixtures",
            "inverse",
            "denoising",
            "super_resolution",
            "colorization",
            "ddcolor",
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
        if args.study in ["inverse", "course_mixtures"]:
            config.update(
                train_images=100,
                validation_images=4,
                test_images=8,
                patch_sizes=[4, 8],
                components=[3],
                patch_samples=1000,
                em_iterations=10,
                course_validation=4,
                course_calibration=2,
                noise_candidates=[30],
            )
        if args.study in ["denoising", "super_resolution", "colorization"]:
            config.update(train_images=256, validation_images=32, test_images=64)
        config["output"] = f"results/smoke/{args.study}"
    if args.output:
        config["output"] = str(args.output)
    if args.study in [
        "classification",
        "regression",
        "signals",
        "vision",
        "denoising",
        "super_resolution",
        "colorization",
        "ddcolor",
    ]:
        import torch

        torch.set_num_threads(4)
        if config["device"].startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable. Use --device cpu or execute with GPU access."
            )
    # Import first so threadpoolctl sees every dynamically loaded BLAS library.
    if args.study in ["classification", "regression"]:
        from learning_lab import tabular

        experiment = getattr(tabular, args.study)
    elif args.study in ["probabilistic", "generalization", "kernels"]:
        from learning_lab import studies

        experiment = getattr(studies, args.study)
    elif args.study == "signals":
        from learning_lab.signals import experiment
    elif args.study in ["course_mixtures", "inverse"]:
        from learning_lab import project_studies

        experiment = getattr(project_studies, args.study)
    elif args.study in ["denoising", "super_resolution", "colorization"]:
        from learning_lab import restoration

        experiment = getattr(restoration, args.study)
    elif args.study == "ddcolor":
        from learning_lab.pretrained import experiment
    else:
        from learning_lab.vision import experiment
    with threadpool_limits(limits=4):
        experiment(config)


if __name__ == "__main__":
    main()
