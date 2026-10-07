"""Exercise standalone restoration commands with real, non-square observations.

Run each model in a separate process, then repeat identical seeded inference.
This checks deployment plumbing and reproducibility, not restoration quality.
"""

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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tasks", nargs="+", choices=["colorization", "super_resolution"])
    parser.add_argument("--output-dir", type=Path, default=Path(".local/inference_audit"))
    args = parser.parse_args()
    tasks = args.tasks or ["colorization", "super_resolution"]
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_root = args.data_root.resolve()
    args.results_root = args.results_root.resolve()
    pixels = np.load(args.data_root / "restoration256/val.npy", mmap_mode="r")
    photo = np.asarray(pixels[400])
    color = args.output_dir / "native_lightness_source.png"
    low = args.output_dir / "actual_low_resolution.png"
    # Nonmultiples of eight exercise the padding/cropping deployment paths.
    Image.fromarray(photo[:251, :249]).save(color)
    import torch
    from torch.nn import functional as F

    tensor = torch.tensor(photo.copy(), dtype=torch.float32).permute(2, 0, 1)[None] / 255
    lr = F.interpolate(
        tensor, scale_factor=0.25, mode="bicubic", align_corners=False, antialias=True
    )
    lr_pixels = (lr[0].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(lr_pixels[:61, :59]).save(low)
    definitions = []
    for task in tasks:
        for method in ["l1", "gan", "cvae", "diffusion"]:
            definitions.append((task, method, "local"))
        methods = (
            ["ddcolor", "ct2", "bigcolor"] if task == "colorization" else ["swinir", "realesrgan"]
        )
        definitions.extend((task, method, "external") for method in methods)
    checks = []
    environment = os.environ | {"OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4"}
    for task, method, exposure in definitions:
        inp = color if task == "colorization" else low
        dimensions = (249, 251) if task == "colorization" else (59 * 4, 61 * 4)
        samples = 2 if method in ["cvae", "diffusion", "bigcolor"] else 1
        if exposure == "local":
            checkpoint = args.results_root / f"generative_{task}/{method}/checkpoints/{method}_0.pt"
            command = ["scripts/restore_generative.py", "--checkpoint", str(checkpoint)]
        else:
            command = ["scripts/restore_external.py", method, "--data-root", str(args.data_root)]
        attempts = []
        for repeat in [0, 1]:
            output = args.output_dir / f"{task}_{method}_repeat{repeat}.png"
            log = output.with_suffix(".log")
            arguments = [
                *command,
                "--device",
                args.device,
                "--input",
                str(inp),
                "--output",
                str(output),
                "--samples",
                str(samples),
                "--seed",
                "7000",
            ]
            with log.open("w") as handle:
                subprocess.run(
                    [sys.executable, *arguments],
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            sidecar = json.loads(output.with_suffix(".json").read_text())
            assert len(sidecar["outputs"]) == samples
            hashes = []
            for path in sidecar["outputs"]:
                with Image.open(path) as image:
                    assert image.mode == "RGB" and image.size == dimensions
                hashes.append(digest(path))
            assert sidecar["input_sha256"] == digest(inp)
            attempts.append(hashes)
        assert attempts[0] == attempts[1], f"Seeded output changed for {task}/{method}"
        checks.append(
            {
                "task": task,
                "method": method,
                "exposure": exposure,
                "samples": samples,
                "output_dimensions": dimensions,
                "input_sha256": digest(inp),
                "repeated_output_sha256": attempts[0],
                "checkpoint_sha256": digest(checkpoint) if exposure == "local" else None,
            }
        )
        print("Verified standalone inference", task, method, flush=True)
    manifest = json.loads((args.data_root / "restoration256/manifest.json").read_text())
    write_json(
        args.output_dir / "audit.json",
        {
            "status": "passed",
            "device": args.device,
            "tasks": tasks,
            "checks": checks,
            "test_photo_id": manifest["splits"]["test"][0]["id"],
            "script_sha256": digest(Path(__file__)),
            "source_sha256": {
                name: digest(ROOT / name)
                for name in [
                    "scripts/restore_generative.py",
                    "scripts/restore_external.py",
                    "src/learning_lab/generative.py",
                    "src/learning_lab/restoration.py",
                    "src/learning_lab/pretrained.py",
                    "src/learning_lab/external_restoration.py",
                ]
            },
            "protocol": "First fixed test photograph, native colorization crop251x249 and actual bicubic LR crop61x59. Fresh subprocesses reproduce PNG bytes under identical seeds, including two draws for stochastic models. Exact dimensions, RGB encoding and sidecar input hashes checked. Model commands reject nonfinite tensors before quantization. This is a deployment integration, not an additional quality evaluation.",
        },
    )


if __name__ == "__main__":
    main()
