"""Audit the trained VAE KL arithmetic against independent float64 calculations."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from learning_lab.artifacts import write_json
from learning_lab.generative import Images, build_model, gaussian_kl, observation

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/reference")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/vae_precision_audit.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This audit requires CUDA autocast with bfloat16")
    torch.set_num_threads(4)
    rows = []
    for task in ["colorization", "super_resolution"]:
        for seed in [0, 1]:
            path = args.results_root / f"generative_{task}/cvae/checkpoints/cvae_{seed}.pt"
            saved = torch.load(path, map_location="cpu", weights_only=True)
            model = build_model("cvae", task, saved["config"], initialize_pretrained=False)
            model.load_state_dict(saved["state_dict"], strict=True)
            model.to(device).eval()
            actual_values, reference_values = [], []
            for rgb in DataLoader(Images(args.data_root, "validation", 128), batch_size=8):
                condition, target = observation(rgb.to(device), task)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pm, pl = model.prior(condition)
                    qm, ql = model.posterior(torch.cat([condition, target], 1))
                    actual = gaussian_kl(qm, ql, pm, pl)
                    exponential_dtype = str(ql.exp().dtype)
                qmd, qld, pmd, pld = [v.double() for v in [qm, ql, pm, pl]]
                reference = (
                    (
                        0.5
                        * (
                            (pld - qld)
                            + (qld - pld).exp()
                            + (qmd - pmd).square() * (-pld).exp()
                            - 1
                        )
                    )
                    .sum(1)
                    .mean()
                )
                actual_values.append(actual.item())
                reference_values.append(reference.item())
            error = max(abs(a - b) for a, b in zip(actual_values, reference_values))
            assert all(v >= 0 for v in reference_values)
            assert all(v >= -1e-6 for v in actual_values)
            assert error < 1e-5, (task, seed, error)
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "validation_images": 128,
                    "encoder_dtype": str(qm.dtype),
                    "autocast_exp_dtype": exponential_dtype,
                    "amp_kl": actual_values,
                    "float64_kl_on_same_encoder_outputs": reference_values,
                    "max_absolute_error": error,
                    "maximum_weighted_objective_error": error * saved["config"]["kl_weight"],
                    "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    manifest = json.loads((args.data_root / "restoration256/manifest.json").read_text())
    write_json(
        args.output,
        {
            "status": "passed",
            "checks": rows,
            "validation_photo_ids": [p["id"] for p in manifest["splits"]["validation"][:128]],
            "source_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ["scripts/check_vae_precision.py", "src/learning_lab/generative.py"]
            },
            "protocol": "All 128 selection-validation photographs and four selected VAE checkpoints. Independent float64 closed-form Gaussian KL uses the same quantized encoder outputs; it does not run a float64 encoder. The absolute reporting tolerance is 1e-5 nats per image after summing the 32 latent dimensions and averaging each eight-image batch; exact errors and their maximum contribution to the weighted objective remain visible. An exploratory 32-image check understated the full-partition maximum; the first full audit exceeded its provisional 2e-6 threshold. This is a trained-checkpoint arithmetic audit, not a guarantee for arbitrary parameters, a gradient-equivalence test, or evidence of informative latents. No fitting, model selection or test evaluation changes.",
            "reference": "https://docs.pytorch.org/docs/2.7/amp.html#cuda-ops-that-can-autocast-to-float32",
        },
    )
    print(
        "VAE arithmetic audit passed; maximum absolute KL error",
        max(r["max_absolute_error"] for r in rows),
    )


if __name__ == "__main__":
    main()
