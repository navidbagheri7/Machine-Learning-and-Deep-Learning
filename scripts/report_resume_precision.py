"""Preserve the unsuccessful GPU bitwise probe separately from CPU recovery evidence."""

import hashlib
from pathlib import Path

import torch

from learning_lab.artifacts import write_json


def main():
    root = Path("results/smoke/adversarial_resume/non_saturating_r1")
    paths = [
        root / mode / "super_resolution/non_saturating_r1/0/checkpoints/training.pt"
        for mode in ["continuous", "resumed"]
    ]
    states = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    a, b = states
    deltas = {}
    for field in ["model", "ema", "critic"]:
        arrays = [
            (a[field][key], b[field][key]) for key in a[field] if a[field][key].is_floating_point()
        ]
        deltas[field] = dict(
            changed_tensors=sum(not torch.equal(x, y) for x, y in arrays),
            maximum_absolute_difference=max(float((x - y).abs().max()) for x, y in arrays),
        )

    def same(x, y):
        if isinstance(x, torch.Tensor):
            return torch.equal(x, y)
        if isinstance(x, (list, tuple)):
            return len(x) == len(y) and all(same(left, right) for left, right in zip(x, y))
        return x == y

    first = [
        {key: value for key, value in state["history"][1].items() if key != "elapsed_seconds"}
        for state in states
    ]
    assert first[0] != first[1], "This report documents observed pre-interruption variation"
    write_json(
        "assets/adversarial_gpu_precision_probe.json",
        dict(
            status="bitwise_check_failed",
            method="non_saturating_r1",
            scope="Two-epoch, eight-image, worker-zero CUDA probe. First-epoch differences occur before the injected interruption; this does not establish a recovery defect or GPU bitwise reproducibility.",
            first_epoch=first,
            final_weight_differences=deltas,
            final_rng_equal={
                key: same(a[key], b[key])
                for key in ["rng", "loader_rng", "penalty_rng", "cuda_rng"]
            },
            private_checkpoint_sha256=[
                hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
            ],
            reporting_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            successful_recovery_evidence="assets/adversarial_resume_audit.json",
            claim="CPU recovery is bitwise verified for R1 and WGAN-GP; GPU bitwise recovery is not claimed.",
        ),
    )
    print("Preserved GPU bitwise probe and its pre-interruption differences")


if __name__ == "__main__":
    main()
