"""Colorize a photograph or upscale actual LR using a pinned external reference."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from learning_lab.artifacts import write_json  # noqa: E402
from learning_lab.external_restoration import (  # noqa: E402
    BigColorReference,
    CT2Reference,
    DDColorReference,
    RealESRGANReference,
    SwinIRReference,
)
from learning_lab.restoration import lab_to_rgb, rgb_to_lab  # noqa: E402


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["ddcolor", "ct2", "bigcolor", "swinir", "realesrgan"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", default=1, type=int)
    parser.add_argument("--seed", default=7000, type=int)
    args = parser.parse_args()
    if args.samples < 1 or (args.samples != 1 and args.method != "bigcolor"):
        parser.error("Multiple positive samples apply only to stochastic BigColor")
    config_path = args.config or ROOT / "configs" / f"external_{args.method}.json"
    config = json.loads(config_path.read_text()) | {"device": args.device}
    if args.data_root:
        config["data_root"] = str(args.data_root)
        if args.method == "ddcolor":
            config["source_root"] = str(args.data_root / "models/DDColor")
            config["model_root"] = str(args.data_root / "models/ddcolor_modelscope")
        if "classifier_weights" in config:
            config["classifier_weights"] = str(
                args.data_root / "models/torch/hub/checkpoints/resnet50-11ad3fa6.pth"
            )
    torch.set_num_threads(4)
    torch.hub.set_dir(str(Path(config["data_root"]) / "models/torch/hub"))
    model = {
        "ddcolor": DDColorReference,
        "ct2": CT2Reference,
        "bigcolor": BigColorReference,
        "swinir": SwinIRReference,
        "realesrgan": RealESRGANReference,
    }[args.method](config)
    model.to(args.device).eval().requires_grad_(False)
    with Image.open(args.input) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB")).copy()
    x = torch.tensor(rgb, dtype=torch.float32, device=args.device).permute(2, 0, 1)[None] / 255
    height, width = x.shape[-2:]
    if args.method not in ["swinir", "realesrgan"]:
        light = rgb_to_lab(x)[:, :1]
        observed = (
            F.interpolate(light, size=(256, 256), mode="bilinear", align_corners=False)
            if args.method in ["ct2", "bigcolor"]
            else light
        )
    else:
        # SwinIR attention windows need multiples of eight. Replicate-padding works
        # even for tiny observations; crop to exactly four times the original size.
        observed = (
            F.pad(x, (0, (-width) % 8, 0, (-height) % 8), mode="replicate")
            if args.method == "swinir"
            else x
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for sample in range(args.samples):
        if args.method in ["swinir", "realesrgan"]:
            result = model.super_resolve(observed)
            if not torch.isfinite(result).all():
                raise FloatingPointError("External model produced a nonfinite raw RGB image")
            result = result[:, :, : height * 4, : width * 4].clamp(0, 1)
        else:
            if args.method == "bigcolor":
                model.generator = torch.Generator(device=args.device).manual_seed(
                    args.seed + sample
                )
            chroma = model(observed)
            if not torch.isfinite(chroma).all():
                raise FloatingPointError("External model produced nonfinite chroma")
            chroma = F.interpolate(
                chroma, size=(height, width), mode="bilinear", align_corners=False
            )
            result = lab_to_rgb(torch.cat([light, chroma], 1))
        if not torch.isfinite(result).all():
            raise FloatingPointError("External model produced a nonfinite image")
        pixels = (
            (result[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
        )
        path = (
            args.output
            if args.samples == 1
            else args.output.with_name(args.output.stem + f"_{sample}" + args.output.suffix)
        )
        Image.fromarray(pixels).save(path)
        outputs.append(str(path))
    write_json(
        args.output.with_suffix(".json"),
        {
            "reference": args.method,
            "samples": args.samples,
            "sampling_seed": args.seed,
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "outputs": outputs,
            "protocol": "Only native lightness enters colorization; preserve it after resizing predicted chroma. SR takes actual LR, window-pads and crops 4x output.",
            "config": config,
        },
    )
    print("Saved", *outputs)


if __name__ == "__main__":
    main()
