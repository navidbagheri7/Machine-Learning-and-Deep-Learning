"""Apply a locally trained restoration checkpoint to a user image.

Example: python scripts/restore_image.py --checkpoint results/reference/colorization/
checkpoints/transfer_l2_0.pt --input photo.png --output .local/colorized.png
"""

import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from learning_lab.artifacts import write_json  # noqa: E402
from learning_lab.restoration import (  # noqa: E402
    Autoencoder,
    PixelShuffleSR,
    ResidualDenoiser,
    TransferColorizer,
    lab_to_rgb,
    rgb_to_lab,
)


def restore(checkpoint, image, sigma=30, device="cpu"):
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    meta = bundle["metadata"]
    task, architecture = meta["task"], meta["architecture"]
    supported = {
        "denoising": {"autoencoder", "residual_denoiser"},
        "super_resolution": {"autoencoder", "pixelshuffle_residual"},
        "colorization": {"autoencoder_l2", "transfer_l2", "transfer_balanced_bins"},
    }
    if task not in supported or architecture not in supported[task]:
        raise ValueError("Unsupported restoration checkpoint architecture")
    if task == "denoising" and (not math.isfinite(sigma) or sigma < 0):
        raise ValueError("Noise SD must be finite and nonnegative")
    data = (
        np.asarray(
            ImageOps.exif_transpose(image).convert("L" if task == "denoising" else "RGB"),
            dtype=np.float32,
        )
        / 255.0
    )
    if data.ndim == 2:
        data = data[..., None]
    x = torch.from_numpy(data.transpose(2, 0, 1).copy())[None].to(device)
    if task == "denoising":
        model = Autoencoder() if architecture == "autoencoder" else ResidualDenoiser()
        inp = (
            x
            if architecture == "autoencoder"
            else torch.cat([x, torch.full_like(x, sigma / 255.0)], 1)
        )
    elif task == "super_resolution":
        model = Autoencoder(3, 3) if architecture == "autoencoder" else PixelShuffleSR()
        inp = (
            F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False)
            if architecture == "autoencoder"
            else x
        )
    elif task == "colorization":
        model = (
            Autoencoder(1, 2)
            if architecture == "autoencoder_l2"
            else TransferColorizer(len(meta["centers"]) if architecture.endswith("bins") else 2)
        )
        lab = rgb_to_lab(x)
        inp = lab[:, :1]
    else:
        raise ValueError("Unknown checkpoint task")
    if isinstance(model, Autoencoder) and min(inp.shape[-2:]) < 4:
        raise ValueError("Autoencoder input must be at least four pixels on each side")
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.to(device).eval()
    with torch.no_grad():
        output = model(inp).float()
        if not torch.isfinite(output).all():
            raise FloatingPointError("Saved model produced a nonfinite restoration field")
        if task == "colorization":
            if architecture.endswith("bins"):
                centers = torch.tensor(meta["centers"], device=device)
                output = torch.einsum(
                    "nkhw,kc->nchw", F.softmax(output / meta["temperature"], 1), centers
                )
            output = lab_to_rgb(torch.cat([lab[:, :1], output], 1))
        if not torch.isfinite(output).all():
            raise FloatingPointError("Saved model produced a nonfinite image")
    pixels = (output[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(pixels[:, :, 0] if pixels.shape[-1] == 1 else pixels)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--sigma", type=float, default=30, help="Noise SD in 0..255 units, denoising only"
    )
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(args.input) as image:
        restore(args.checkpoint, image, args.sigma, args.device).save(args.output)
    write_json(
        args.output.with_suffix(".json"),
        {
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "noise_sigma_0_255": args.sigma,
            "output": str(args.output),
        },
    )
    print("Saved", args.output)
