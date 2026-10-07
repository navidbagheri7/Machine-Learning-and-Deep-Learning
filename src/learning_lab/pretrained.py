"""Frozen modern colorization reference with pinned external source and weights."""

import hashlib
import json
import subprocess
import sys
import time
from importlib.resources import files
from pathlib import Path

import torch
from torch.nn import functional as F

from .artifacts import Run
from .restoration import lab_to_rgb, measured, natural_data, pack_scores, rgb_to_lab, summarize


def load_ddcolor(config):
    source = Path(config["source_root"])
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    ).stdout.strip()
    if revision != config["source_revision"]:
        raise ValueError("DDColor source revision mismatch; use scripts/prepare_ddcolor.py")
    clean = subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD"])
    if clean.returncode:
        raise ValueError("Pinned DDColor source contains local tracked changes")
    path = Path(config["model_root"]) / "pytorch_model.bin"
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != config["weight_sha256"]:
        raise ValueError("DDColor weight checksum mismatch")
    sys.path.insert(0, str(source))
    from ddcolor import DDColor

    metadata_bytes = (Path(config["model_root"]) / "config.json").read_bytes()
    pins = json.loads((files("learning_lab") / "resources/ddcolor_manifest.json").read_text())
    if hashlib.sha256(metadata_bytes).hexdigest() != pins["model_config_sha256"]:
        raise ValueError("DDColor model configuration checksum mismatch")
    metadata = json.loads(metadata_bytes)
    metadata["input_size"] = [config["input_size"]] * 2
    model = DDColor(**metadata)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(config["device"]).eval(), checksum


@torch.no_grad()
def colorize_ddcolor(model, lightness, size=512, batch_size=2):
    outputs = []
    for light in lightness.split(batch_size):
        neutral = torch.cat([light, torch.zeros_like(light).expand(-1, 2, -1, -1)], 1)
        gray = lab_to_rgb(neutral)
        # Only native-resolution lightness enters the network; no RGB-before-resize chroma leakage.
        inp = F.interpolate(gray, size=(size, size), mode="bilinear", align_corners=False)
        ab = model(inp).float() / 110.0
        outputs.append(F.interpolate(ab, size=light.shape[-2:], mode="nearest"))
    return torch.cat(outputs)


def experiment(config):
    import numpy as np

    run = Run(config["output"], config)
    model, checksum = load_ddcolor(config)
    _, _, test_rgb = natural_data(config)
    lab = torch.cat([rgb_to_lab(x) for x in test_rgb.split(128)])
    # Warm up separately; latency excludes loading and metrics, includes CPU/GPU synchronization.
    colorize_ddcolor(model, lab[:1, :1], config["input_size"], config["batch_size"])
    if str(config["device"]).startswith("cuda"):
        torch.cuda.synchronize(config["device"])
    start = time.monotonic()
    ab = colorize_ddcolor(model, lab[:, :1], config["input_size"], config["batch_size"])
    if str(config["device"]).startswith("cuda"):
        torch.cuda.synchronize(config["device"])
    elapsed = time.monotonic() - start
    prediction = lab_to_rgb(torch.cat([lab[:, :1], ab], 1))
    row = {
        "model": "ddcolor_modelscope_frozen",
        **measured(test_rgb, prediction),
        "delta_e76": summarize(
            (((ab - lab[:, 1:]) * 110) ** 2).sum(1).sqrt().mean((1, 2)).cpu().numpy()
        ),
        "mean_chroma": summarize((ab**2).sum(1).sqrt().mean((1, 2)).cpu().numpy() * 110),
        "true_mean_chroma": float((lab[:, 1:] ** 2).sum(1).sqrt().mean() * 110),
    }
    pack_scores(run, [row])
    np.savez_compressed(
        run.output / "gallery.npz",
        clean=test_rgb[:8].cpu().numpy(),
        pretrained=prediction[:8].cpu().numpy(),
    )
    run.finish(
        {
            "rows": [row],
            "weight_sha256": checksum,
            "parameters": sum(p.numel() for p in model.parameters()),
            "evaluation_seconds": elapsed,
            "images_per_second": len(test_rgb) / elapsed,
            "protocol": "Frozen official DDColor-ModelScope (ICCV2023), ImageNet pretraining with authors' cleaning scheme. No local fitting or test-based selection. Same 400 COCO test crops and lightness-only input as local colorizers; 512px bilinear gray upsampling, nearest chroma downsampling. External training overlap with COCO was not audited; compare exposure explicitly.",
        }
    )
