"""Check every selected foundational restoration checkpoint through its image CLI."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from learning_lab.artifacts import write_json  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/reference")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".local/foundation_inference")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = args.output_dir.resolve()
    photo = np.load(args.data_root / "restoration256/val.npy", mmap_mode="r")[400]
    with np.load(args.data_root / "projects/mnist.npz") as data:
        digit = data["test"][0, :27, :25].astype(float)
    digit += np.random.default_rng(810).normal(0, 30, digit.shape)
    import torch
    from torch.nn import functional as F

    image = torch.tensor(photo.copy(), dtype=torch.float32).permute(2, 0, 1)[None] / 255
    low = F.interpolate(
        image, scale_factor=0.25, mode="bicubic", align_corners=False, antialias=True
    )
    low = (low[0].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
    inputs = {}
    for task, pixels in [
        ("denoising", digit.round().clip(0, 255).astype(np.uint8)),
        ("super_resolution", low[:61, :59]),
        ("colorization", np.asarray(photo[:251, :249])),
    ]:
        inputs[task] = args.output_dir / f"{task}_input.png"
        Image.fromarray(pixels).save(inputs[task])
    checks = []
    environment = os.environ | {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    for task, dimensions, mode in [
        ("denoising", (25, 27), "L"),
        ("super_resolution", (236, 244), "RGB"),
        ("colorization", (249, 251), "RGB"),
    ]:
        checkpoints = sorted((args.results_root / task / "checkpoints").glob("*.pt"))
        assert len(checkpoints) == (4 if task == "super_resolution" else 6)
        for checkpoint in checkpoints:
            repeats = []
            for repeat in [0, 1]:
                output = args.output_dir / f"{task}_{checkpoint.stem}_repeat{repeat}.png"
                with output.with_suffix(".log").open("w") as handle:
                    subprocess.run(
                        [
                            sys.executable,
                            "scripts/restore_image.py",
                            "--checkpoint",
                            str(checkpoint.resolve()),
                            "--input",
                            str(inputs[task]),
                            "--output",
                            str(output),
                            "--sigma",
                            "30",
                        ],
                        cwd=ROOT,
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                with Image.open(output) as result:
                    assert result.mode == mode and result.size == dimensions
                metadata = json.loads(output.with_suffix(".json").read_text())
                assert metadata["checkpoint_sha256"] == digest(checkpoint)
                assert metadata["input_sha256"] == digest(inputs[task])
                repeats.append(digest(output))
            assert repeats[0] == repeats[1]
            checks.append(
                {
                    "task": task,
                    "checkpoint": checkpoint.name,
                    "checkpoint_sha256": digest(checkpoint),
                    "output_dimensions": dimensions,
                    "repeated_output_sha256": repeats[0],
                }
            )
            print("Verified foundational inference", task, checkpoint.stem, flush=True)
    # Inject an invalid raw field into the same deployed inference path. Clipping
    # must not turn it into an apparently valid image.
    sys.path.insert(0, str(ROOT / "scripts"))
    import restore_image

    checkpoint = args.results_root / "denoising/checkpoints/autoencoder_0.pt"
    with Image.open(inputs["denoising"]) as image:
        for sigma in [-1.0, float("nan"), float("inf")]:
            try:
                restore_image.restore(checkpoint, image, sigma)
            except ValueError:
                continue
            raise AssertionError("Invalid noise setting accepted")
        original = restore_image.Autoencoder.forward
        try:
            restore_image.Autoencoder.forward = lambda self, x: torch.full_like(x, float("inf"))
            try:
                restore_image.restore(checkpoint, image)
            except FloatingPointError:
                pass
            else:
                raise AssertionError("Infinite raw prediction accepted")
        finally:
            restore_image.Autoencoder.forward = original
    write_json(
        args.output_dir / "audit.json",
        {
            "status": "passed",
            "device": "cpu",
            "checks": checks,
            "invalid_noise_settings_rejected": True,
            "infinite_raw_field_rejected": True,
            "script_sha256": digest(Path(__file__)),
            "inference_source_sha256": digest(ROOT / "scripts/restore_image.py"),
            "protocol": "All 16 selected foundational restoration checkpoints. Non-square first fixed COCO test photograph and first official MNIST test digit. PNG bytes reproduce in fresh CPU processes. Noise is deliberately quantized/clipped for the image CLI fixture; this is a deployment integration, not a recomputation of benchmark quality scores from unclipped floating observations. Checkpoint/input hashes, exact output dimensions and image modes checked. Invalid noise settings and injected infinite raw field rejected.",
        },
    )


if __name__ == "__main__":
    main()
