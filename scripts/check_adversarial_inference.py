"""Check selected and fixed-final GAN generators on real standalone observations."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from learning_lab.artifacts import write_json
from learning_lab.generative import Images, build_model, observation, reconstruct


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path(".local/adversarial_inference"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path("configs/adversarial_variants.json").read_text())
    torch.set_num_threads(2)
    rgb = torch.stack([Images(config["data_root"], "test", 8)[i] for i in range(8)]).to(args.device)
    checks = []
    for task in config["tasks"]:
        if task == "colorization":
            condition = observation(rgb[:1], task)[0]
            gray = reconstruct(condition, torch.zeros(1, 2, 256, 256, device=args.device), task)
            pixels = (
                (gray[0].permute(1, 2, 0).cpu().numpy()[:251, :249] * 255)
                .round()
                .clip(0, 255)
                .astype(np.uint8)
            )
        else:
            low = F.interpolate(
                rgb[:1], scale_factor=0.25, mode="bicubic", align_corners=False, antialias=True
            )
            pixels = (
                (low[0].permute(1, 2, 0).cpu().numpy()[:61, :59] * 255)
                .round()
                .clip(0, 255)
                .astype(np.uint8)
            )
        input_path = args.output / (task + "_observation.png")
        Image.fromarray(pixels).save(input_path)
        for family in ["adversarial_variants", "adversarial_endpoints"]:
            for method in config["methods"]:
                for seed in config["seeds"]:
                    directory = Path("results/reference") / family / task / method / str(seed)
                    metrics = json.loads((directory / "metrics.json").read_text())
                    fixed = family == "adversarial_endpoints"
                    alias = method + ("_final_epoch" if fixed else "")
                    checkpoint = (
                        directory / "checkpoints" / ("final_epoch.pt" if fixed else "selected.pt")
                    )
                    assert digest(checkpoint) == metrics["checkpoint_sha256"]
                    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    assert saved["method"] == method and saved["config"]["task"] == task
                    model = (
                        build_model(method, task, saved["config"], initialize_pretrained=False)
                        .to(args.device)
                        .eval()
                    )
                    model.load_state_dict(saved["state_dict"], strict=True)
                    output = []
                    with torch.no_grad():
                        for batch in rgb.split(config["evaluation_batch_size"]):
                            condition, _ = observation(batch, task)
                            output.append(reconstruct(condition, model(condition), task))
                    predicted = torch.cat(output)[[0, 1, 3, 5]].cpu().numpy()
                    with np.load(
                        f"assets/source_data/{family}/{task}_{method}_{seed}_gallery.npz",
                        allow_pickle=False,
                    ) as gallery:
                        expected = gallery[f"{alias}_{seed}_sample0"]
                        if fixed:
                            observed = (predicted.clip(0, 1) * 255).round().astype(np.uint8)
                            error = float(abs(observed.astype(int) - expected.astype(int)).max())
                            assert error <= 1, "Float32 kernels changed more than one display level"
                        else:
                            np.testing.assert_allclose(predicted, expected, atol=1e-5, rtol=0)
                            error = float(abs(predicted - expected).max())
                    del model
                    paths = []
                    for attempt in [0, 1]:
                        path = args.output / f"{task}_{alias}_{seed}_{attempt}.png"
                        subprocess.run(
                            [
                                sys.executable,
                                "scripts/restore_generative.py",
                                "--checkpoint",
                                str(checkpoint),
                                "--input",
                                str(input_path),
                                "--output",
                                str(path),
                                "--device",
                                args.device,
                            ],
                            check=True,
                        )
                        sidecar = json.loads(path.with_suffix(".json").read_text())
                        assert sidecar["method"] == method and sidecar["task"] == task
                        assert sidecar["checkpoint_policy"] == (
                            "fixed_final_epoch_ema" if fixed else "validation_selected"
                        )
                        assert sidecar["selected_epoch"] == (20 if fixed else metrics["best_epoch"])
                        assert sidecar["input_sha256"] == digest(input_path)
                        assert sidecar["checkpoint_sha256"] == digest(checkpoint)
                        with Image.open(path) as image:
                            assert image.mode == "RGB"
                            assert image.size == (
                                (249, 251) if task == "colorization" else (236, 244)
                            )
                        paths.append(digest(path))
                    assert paths[0] == paths[1]
                    checks.append(
                        dict(
                            task=task,
                            method=method,
                            policy="fixed_final_epoch_ema" if fixed else "validation_selected",
                            gallery_tolerance_unit="display level" if fixed else "float intensity",
                            seed=seed,
                            checkpoint_sha256=digest(checkpoint),
                            fixed_gallery_max_absolute_error=error,
                            repeated_png_sha256=paths[0],
                            input_sha256=digest(input_path),
                        )
                    )
                    print(
                        "Verified variant inference",
                        task,
                        method,
                        seed,
                        "gallery error",
                        error,
                        flush=True,
                    )
    assert len(checks) == 40
    write_json(
        "assets/adversarial_inference_audit.json",
        dict(
            status="passed",
            checks=checks,
            device=args.device,
            numeric_absolute_tolerance=1e-5,
            fixed_endpoint_display_tolerance=1,
            gallery_protocol="Original eight test inputs, original evaluation batch size, fixed gallery indices0/1/3/5; float32 generator inference without metric model; fixed endpoints compare original uint8 displays within one level",
            standalone_protocol="Actual quantized lightness-only251x249 observation or bicubic61x59 LR; two fresh subprocesses per checkpoint; deterministic PNG bytes, task/weight hashes and output dimensions",
            source_sha256={
                name: digest(name)
                for name in [
                    "scripts/check_adversarial_inference.py",
                    "scripts/restore_generative.py",
                    "src/learning_lab/generative.py",
                    "src/learning_lab/restoration.py",
                ]
            },
        ),
    )


if __name__ == "__main__":
    main()
