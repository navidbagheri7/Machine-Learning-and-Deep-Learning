"""Colorize a user photograph using the pinned, explicitly prepared DDColor model."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from learning_lab.pretrained import colorize_ddcolor, load_ddcolor  # noqa: E402
from learning_lab.restoration import lab_to_rgb, rgb_to_lab  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/ddcolor.json"))
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if args.device:
        config["device"] = args.device
    torch.set_num_threads(4)
    with Image.open(args.input) as image:
        x = (
            torch.tensor(
                np.asarray(image.convert("RGB")).transpose(2, 0, 1).copy(),
                dtype=torch.float32,
                device=config["device"],
            )[None]
            / 255.0
        )
    model, _ = load_ddcolor(config)
    with torch.no_grad():
        lightness = rgb_to_lab(x)[:, :1]
        ab = colorize_ddcolor(model, lightness, config["input_size"], 1)
        out = lab_to_rgb(torch.cat([lightness, ab], 1))[0].permute(1, 2, 0).cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((out * 255).round().astype(np.uint8)).save(args.output)
    print("Saved", args.output)
