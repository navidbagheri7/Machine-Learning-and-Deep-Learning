"""Natural-image Gaussian denoising: wavelet baseline versus frozen SwinIR."""

import argparse
import importlib.util
import json
from pathlib import Path

import lpips
import numpy as np
import torch
from skimage.restoration import denoise_wavelet

from .artifacts import Run
from .external_restoration import verified_source
from .restoration import per_image_metrics, summarize


def load_model(config):
    source, _ = verified_source(config, "SwinIR")
    spec = importlib.util.spec_from_file_location(
        "official_swinir_dn", source / "models" / "network_swinir.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.SwinIR(
        upscale=1,
        in_chans=3,
        img_size=128,
        window_size=8,
        img_range=1.0,
        depths=[6] * 6,
        embed_dim=180,
        num_heads=[6] * 6,
        mlp_ratio=2,
        upsampler="",
        resi_connection="1conv",
    )
    checkpoint = source / "005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth"
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)["params"], strict=True
    )
    return model.to(config["device"]).eval().requires_grad_(False)


@torch.no_grad()
def experiment(config):
    torch.set_num_threads(4)
    torch.hub.set_dir(str(Path(config["data_root"]) / "models" / "torch" / "hub"))
    if not (Path(torch.hub.get_dir()) / "checkpoints" / "alexnet-owt-7be5be79.pth").exists():
        raise FileNotFoundError("Prepare metric weights explicitly")
    run = Run(config["output"], config)
    model = load_model(config)
    metric = lpips.LPIPS(net="alex", verbose=False).to(config["device"]).eval()
    data = np.load(Path(config["data_root"]) / "restoration256" / "val.npy", mmap_mode="r")
    clean = data[400 : 400 + config["test_images"]].astype(np.float32) / 255
    rng = np.random.default_rng(814)
    noise = rng.normal(size=clean.shape).astype(np.float32)
    rows, arrays, gallery = [], {}, {}
    gallery_ids = [0, 1, 3, 5]
    gallery["original"] = clean[gallery_ids].transpose(0, 3, 1, 2)
    for sigma in config["sigmas"]:
        observed = clean + noise * (sigma / 255)
        score = {
            name: {"psnr": [], "ssim": [], "lpips": []}
            for name in ["noisy", "wavelet", "swinir_noise25"]
        }
        gallery[f"input_sigma{sigma}"] = observed[gallery_ids].transpose(0, 3, 1, 2).clip(0, 1)
        for start in range(0, len(clean), config["batch_size"]):
            target = torch.tensor(
                clean[start : start + config["batch_size"]].transpose(0, 3, 1, 2),
                device=config["device"],
            )
            x = torch.tensor(
                observed[start : start + config["batch_size"]].transpose(0, 3, 1, 2),
                device=config["device"],
            )
            wavelet = np.stack(
                [
                    denoise_wavelet(
                        im,
                        sigma=sigma / 255,
                        method="BayesShrink",
                        mode="soft",
                        convert2ycbcr=True,
                        rescale_sigma=True,
                        channel_axis=-1,
                    )
                    for im in observed[start : start + config["batch_size"]]
                ]
            )
            predictions = {
                "noisy": x,
                "wavelet": torch.tensor(
                    wavelet.transpose(0, 3, 1, 2), device=config["device"], dtype=torch.float32
                ),
                "swinir_noise25": model(x),
            }
            for name, pred in predictions.items():
                if not torch.isfinite(pred).all():
                    raise FloatingPointError(f"{name} produced a nonfinite denoising output")
                pred = pred.clamp(0, 1)
                values = per_image_metrics(target, pred)
                values["lpips"] = metric(target * 2 - 1, pred * 2 - 1).flatten().cpu().numpy()
                for key, v in values.items():
                    score[name][key].extend(v.tolist())
                for i in range(len(target)):
                    if start + i in gallery_ids:
                        gallery.setdefault(f"{name}_sigma{sigma}", []).append(pred[i].cpu().numpy())
        for name, values in score.items():
            rows.append(
                {
                    "model": name,
                    "sigma": sigma,
                    "matched_training_noise": name != "swinir_noise25" or sigma == 25,
                    **{k: summarize(v) for k, v in values.items()},
                }
            )
            arrays.update({f"{name}_sigma{sigma}_{k}": np.array(v) for k, v in values.items()})
        print(
            "Noise",
            sigma,
            [(name, round(np.mean(value["psnr"]), 3)) for name, value in score.items()],
            flush=True,
        )
    np.savez_compressed(run.output / "per_image_scores.npz", **arrays)
    np.savez_compressed(
        run.output / "gallery.npz", **{k: np.asarray(v) for k, v in gallery.items()}
    )
    run.finish(
        {
            "rows": rows,
            "protocol": "Same disjoint 400 COCO test images at 256px. Whole-image unclipped additive Gaussian noise, fixed common independent noise draw across methods; sigma in 0..255 intensity units. Wavelet BayesShrink uses known sigma; frozen SwinIR is trained externally at sigma25, so sigma15/50 are unadapted noise shifts. Predictions clipped only for metrics/display. No local fitting or test tuning; external data exposure/overlap not audited.",
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    experiment(json.loads(parser.parse_args().config.read_text()))
