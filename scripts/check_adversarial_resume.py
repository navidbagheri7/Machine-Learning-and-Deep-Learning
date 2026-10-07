"""Interrupt saved GAN epochs and compare complete resumed optimizer/RNG state."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from learning_lab import adversarial
from learning_lab.artifacts import write_json


class InterruptedEpoch(RuntimeError):
    pass


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/smoke/adversarial_resume_cpu"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh smoke output to preserve recovery evidence")
    config = json.loads(Path("configs/adversarial_variants.json").read_text()) | dict(
        quick=True,
        epochs=2,
        train_images=8,
        validation_images=8,
        test_images=8,
        crop=32,
        batch_size=4,
        workers=0,
        r1_interval=2,
    )
    checks = []
    for method in ["non_saturating_r1", "wasserstein_gp"]:
        normal = config | dict(output=str(args.output / method / "continuous"))
        resumed = config | dict(output=str(args.output / method / "resumed"))
        adversarial.train(copy.deepcopy(normal), "super_resolution", method, 0, args.device)
        original_save = adversarial.save_checkpoint

        def interrupt_after_epoch(value, path):
            original_save(value, path)
            if Path(path).name == "training.pt" and value["history"][-1]["epoch"] == 1:
                raise InterruptedEpoch("Injected interruption after atomic epoch1 save")

        adversarial.save_checkpoint = interrupt_after_epoch
        try:
            try:
                adversarial.train(
                    copy.deepcopy(resumed), "super_resolution", method, 0, args.device
                )
            except InterruptedEpoch:
                pass
            else:
                raise AssertionError("Interruption did not run")
        finally:
            adversarial.save_checkpoint = original_save
        adversarial.train(copy.deepcopy(resumed), "super_resolution", method, 0, args.device)
        left = Path(normal["output"]) / "super_resolution" / method / "0"
        right = Path(resumed["output"]) / "super_resolution" / method / "0"
        a = torch.load(left / "checkpoints/training.pt", map_location="cpu", weights_only=True)
        b = torch.load(right / "checkpoints/training.pt", map_location="cpu", weights_only=True)
        fields = [
            "model",
            "ema",
            "critic",
            "optimizer",
            "dopt",
            "scheduler",
            "updates",
            "rng",
            "loader_rng",
            "penalty_rng",
            "cuda_rng",
            "best_state",
            "best_epoch",
            "best",
        ]
        for key in fields:
            equal(a[key], b[key])
        for first, second in zip(a["history"], b["history"]):
            equal(
                {k: v for k, v in first.items() if k != "elapsed_seconds"},
                {k: v for k, v in second.items() if k != "elapsed_seconds"},
            )
        assert len(a["history"]) == len(b["history"]) == 3 and a["updates"] == 4
        checks.append(
            dict(
                method=method,
                seed=0,
                epochs=2,
                updates=4,
                compared_fields=fields,
                final_train_state_and_rng_bitwise_equal=True,
                losses_bitwise_equal=True,
            )
        )
        print("Verified interrupted epoch recovery", method, flush=True)
    write_json(
        "assets/adversarial_resume_audit.json",
        dict(
            status="passed",
            checks=checks,
            device=args.device,
            workers=0,
            scope="Small real-image integration on the declared device; final trainable weights, EMA, critic, Adam states, schedule and independent RNG streams compared, including a post-resume R1 update. Epoch timings excluded. This checks the declared device and worker0 path, not GPU bitwise or cross-device/worker-count identity.",
            source_sha256={
                name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in [
                    "scripts/check_adversarial_resume.py",
                    "src/learning_lab/adversarial.py",
                    "src/learning_lab/generative.py",
                    "src/learning_lab/artifacts.py",
                ]
            },
        ),
    )


if __name__ == "__main__":
    main()
