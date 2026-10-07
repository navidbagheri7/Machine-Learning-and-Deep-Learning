"""Test frozen SR references under controlled blur/noise/compression mismatch."""

import argparse
import io
import json
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .artifacts import Run
from .external_restoration import RealESRGANReference, SwinIRReference
from .restoration import per_image_metrics, summarize


def degradation(target, name, noise):
    source = target
    if name == "blur":
        coordinate = torch.arange(9, device=target.device) - 4
        kernel = torch.exp(-coordinate.square() / (2 * 1.2**2))
        kernel = kernel / kernel.sum()
        weight = (kernel[:, None] * kernel[None, :]).expand(3, 1, 9, 9)
        source = F.conv2d(F.pad(target, (4, 4, 4, 4), mode="reflect"), weight, groups=3)
    low = F.interpolate(
        source, scale_factor=0.25, mode="bicubic", align_corners=False, antialias=True
    )
    if name == "noise_jpeg":
        pixels = (
            ((low.cpu() + noise * (3 / 255)).clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255)
            .round()
            .astype(np.uint8)
        )
        decoded = []
        for im in pixels:
            buffer = io.BytesIO()
            Image.fromarray(im).save(buffer, format="JPEG", quality=60, subsampling=2)
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                decoded.append(np.asarray(compressed.convert("RGB")).copy())
        low = (
            torch.tensor(np.stack(decoded), device=target.device, dtype=torch.float32).permute(
                0, 3, 1, 2
            )
            / 255
        )
    elif name not in ["bicubic", "blur"]:
        raise ValueError("Unknown degradation")
    return low


@torch.no_grad()
def experiment(config):
    torch.set_num_threads(4)
    torch.hub.set_dir(str(Path(config["data_root"]) / "models/torch/hub"))
    if not (Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth").exists():
        raise FileNotFoundError("Prepare LPIPS weights explicitly")
    run = Run(config["output"], config)
    device = config["device"]
    cache = np.load(Path(config["data_root"]) / "restoration256/val.npy", mmap_mode="r")
    clean = (
        torch.tensor(
            cache[400 : 400 + config["test_images"]].copy(), device=device, dtype=torch.float32
        ).permute(0, 3, 1, 2)
        / 255
    )
    noise = torch.randn((len(clean), 3, 64, 64), generator=torch.Generator().manual_seed(810))
    models = {
        "swinir": SwinIRReference(config).to(device).eval(),
        "realesrgan": RealESRGANReference(config).to(device).eval(),
    }
    metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    rows, arrays, gallery = [], {}, {"original": clean[[0, 1, 3, 5]].cpu().numpy()}
    for name in config["degradations"]:
        low = degradation(clean, name, noise)
        values = {
            method: {key: [] for key in ["psnr", "ssim", "lpips", "data_consistency_l1"]}
            for method in ["bicubic", *models]
        }
        pictures = {method: [] for method in values}
        for start in range(0, len(clean), config["batch_size"]):
            observed = low[start : start + config["batch_size"]]
            target = clean[start : start + config["batch_size"]]
            predictions = {
                "bicubic": F.interpolate(
                    observed, scale_factor=4, mode="bicubic", align_corners=False
                ),
                **{method: model.super_resolve(observed) for method, model in models.items()},
            }
            for method, predicted in predictions.items():
                if not torch.isfinite(predicted).all():
                    raise FloatingPointError(f"{method} produced a nonfinite SR stress-test output")
                predicted = predicted.clamp(0, 1)
                scores = per_image_metrics(target, predicted, border=4)
                scores["lpips"] = metric(target * 2 - 1, predicted * 2 - 1).flatten().cpu().numpy()
                reprojection = F.interpolate(
                    predicted,
                    scale_factor=0.25,
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
                scores["data_consistency_l1"] = (
                    (reprojection - observed).abs().mean((1, 2, 3)).cpu().numpy()
                )
                for key, value in scores.items():
                    values[method][key].extend(value.tolist())
                for i in range(len(target)):
                    if start + i in [0, 1, 3, 5]:
                        pictures[method].append(predicted[i].cpu().numpy())
        for method, scores in values.items():
            rows.append(
                {
                    "method": method,
                    "degradation": name,
                    **{key: summarize(value) for key, value in scores.items()},
                }
            )
            arrays.update(
                {f"{method}_{name}_{key}": np.asarray(value) for key, value in scores.items()}
            )
            gallery[f"{method}_{name}"] = np.stack(pictures[method])
        print(
            name,
            [(method, round(np.mean(score["psnr"]), 3)) for method, score in values.items()],
            flush=True,
        )
    np.savez_compressed(run.output / "per_image_scores.npz", **arrays)
    np.savez_compressed(run.output / "gallery.npz", **gallery)
    run.finish(
        {
            "rows": rows,
            "protocol": "Same 400 original 256px COCO test crops, no fitting or test selection. Frozen SwinIR and Real-ESRGAN versus bicubic. Matched antialiased bicubic4; HR Gaussian blur sigma1.2, 9px reflect kernel before downsampling; or bicubic LR with Gaussian sigma3/255 then clipping, uint8 conversion and JPEG quality60 chroma subsampling2. Fixed common noise seed810. RGB PSNR/SSIM with 4px shave, full-image LPIPS; consistency is reprojection to actual degraded LR, not a guarantee for mismatched degradation. These are controlled stressors, not a real-camera dataset or exact reproduction of either model's training degradation.",
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    experiment(json.loads(parser.parse_args().config.read_text()))
