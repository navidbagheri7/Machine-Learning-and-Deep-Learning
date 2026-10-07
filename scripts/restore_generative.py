"""Use a local trained GAN, CVAE or diffusion checkpoint without downloading weights."""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from learning_lab.artifacts import write_json  # noqa: E402
from learning_lab.generative import build_model, reconstruct  # noqa: E402
from learning_lab.restoration import rgb_to_lab  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7000)
    args = parser.parse_args()
    if args.samples < 1 or not 2 <= args.steps <= 1000:
        parser.error("Positive sample count and 2..1000 diffusion steps required")
    torch.set_num_threads(4)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    method, config = state["method"], state["config"]
    if method not in ["cvae", "diffusion"] and args.samples != 1:
        parser.error("Deterministic checkpoints produce one output; use --samples 1")
    model = (
        build_model(method, config["task"], config, initialize_pretrained=False)
        .to(args.device)
        .eval()
    )
    model.load_state_dict(state["state_dict"], strict=True)
    with Image.open(args.input) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB")).copy()
    x = torch.tensor(rgb, device=args.device, dtype=torch.float32).permute(2, 0, 1)[None] / 255
    if config["task"] == "colorization":
        condition = rgb_to_lab(x)[:, :1]
    else:
        condition = F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False)
    h, w = condition.shape[-2:]
    condition = F.pad(condition, (0, (-w) % 8, 0, (-h) % 8), mode="replicate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    with torch.no_grad():
        for sample in range(args.samples):
            rng = torch.Generator(device=args.device).manual_seed(args.seed + sample)
            if method == "diffusion":
                field = model.sample(condition, args.steps, rng)
            elif method == "cvae":
                field = model(condition, generator=rng)[0]
            else:
                field = model(condition)
            if not torch.isfinite(field).all():
                raise FloatingPointError("Saved model produced a nonfinite restoration field")
            output = reconstruct(condition, field, config["task"])[0, :, :h, :w]
            if not torch.isfinite(output).all():
                raise FloatingPointError("Saved model produced a nonfinite image")
            pixels = (
                (output.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
            )
            path = (
                args.output
                if args.samples == 1
                else args.output.with_name(args.output.stem + f"_{sample}" + args.output.suffix)
            )
            Image.fromarray(pixels).save(path)
            paths.append(str(path))
    write_json(
        args.output.with_suffix(".json"),
        {
            "method": method,
            "task": config["task"],
            "seed": args.seed,
            "diffusion_steps": args.steps if method == "diffusion" else None,
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "outputs": paths,
        },
    )
    print("Saved", *paths)


if __name__ == "__main__":
    main()
