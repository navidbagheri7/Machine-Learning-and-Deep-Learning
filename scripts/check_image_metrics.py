"""Compare restoration metrics with independent scientific-library calculations."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import convolve2d
from skimage.color import deltaE_cie76, rgb2lab

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from learning_lab.artifacts import write_json  # noqa: E402
from learning_lab.restoration import per_image_metrics, rgb_to_lab  # noqa: E402


def reference_scores(target, prediction, border):
    """Float64 valid convolution, independently of PyTorch's grouped Conv2d."""
    if border:
        target = target[:, border:-border, border:-border]
        prediction = prediction[:, border:-border, border:-border]
    target, prediction = target.astype(np.float64), prediction.astype(np.float64)
    psnr = -10 * np.log10(np.maximum(np.mean((target - prediction) ** 2, axis=(1, 2, 3)), 1e-12))
    coordinates = np.mgrid[-3:4, -3:4]
    weights = np.exp(-np.sum(coordinates**2, axis=0) / (2 * 1.5**2))
    weights /= weights.sum()
    values = []
    for x, y in zip(target, prediction):
        channels = []
        for channel in range(x.shape[-1]):
            a, b = x[..., channel], y[..., channel]
            ma, mb = [convolve2d(z, weights, mode="valid") for z in [a, b]]
            va = convolve2d(a * a, weights, mode="valid") - ma**2
            vb = convolve2d(b * b, weights, mode="valid") - mb**2
            cov = convolve2d(a * b, weights, mode="valid") - ma * mb
            channels.append(
                np.mean(
                    ((2 * ma * mb + 0.01**2) * (2 * cov + 0.03**2))
                    / ((ma**2 + mb**2 + 0.01**2) * (va + vb + 0.03**2))
                )
            )
        values.append(np.mean(channels))
    return {"psnr": psnr, "ssim": np.asarray(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/metric_reference_audit.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    pixels = np.load(args.data_root / "restoration256/val.npy", mmap_mode="r")[400:416]
    target = pixels.astype(np.float32) / 255
    rng = np.random.default_rng(810)
    shifted = np.clip(target + rng.normal(0, 0.07, target.shape).astype(np.float32), 0, 1)
    checks = []
    for channels in [1, 3]:
        x, y = target[..., :channels], shifted[..., :channels]
        for border in [0, 4]:
            for name, prediction in [("identity", x), ("perturbed", y)]:
                expected = reference_scores(x, prediction, border)
                actual = per_image_metrics(
                    torch.tensor(x).permute(0, 3, 1, 2),
                    torch.tensor(prediction).permute(0, 3, 1, 2),
                    border,
                )
                errors = {}
                for metric, tolerance in [("psnr", 1e-4), ("ssim", 2e-5)]:
                    error = float(np.max(np.abs(expected[metric] - actual[metric])))
                    assert error < tolerance, (channels, border, name, metric, error)
                    errors[metric] = error
                checks.append({"channels": channels, "border": border, "case": name, **errors})
    lab_x, lab_y = [
        rgb_to_lab(torch.tensor(rgb).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()
        * [100, 110, 110]
        for rgb in [target, shifted]
    ]
    expected_x, expected_y = [
        rgb2lab(rgb, illuminant="D65", observer="2") for rgb in [target, shifted]
    ]
    lab_error = float(max(np.max(np.abs(lab_x - expected_x)), np.max(np.abs(lab_y - expected_y))))
    delta_e_error = float(
        np.max(
            np.abs(
                np.linalg.norm(lab_x - lab_y, axis=-1).mean((1, 2))
                - deltaE_cie76(expected_x, expected_y).mean((1, 2))
            )
        )
    )
    # Both implementations use D65 but round their RGB/XYZ matrices differently.
    assert lab_error < 0.025 and delta_e_error < 0.01, (lab_error, delta_e_error)
    manifest = json.loads((args.data_root / "restoration256/manifest.json").read_text())
    write_json(
        args.output,
        {
            "status": "passed",
            "checks": checks,
            "lab_max_absolute_difference": lab_error,
            "delta_e76_max_image_mean_difference": delta_e_error,
            "test_photo_ids": [p["id"] for p in manifest["splits"]["test"][:16]],
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "metric_source_sha256": hashlib.sha256(
                (ROOT / "src/learning_lab/restoration.py").read_bytes()
            ).hexdigest(),
            "protocol": "First 16 fixed held-out photographs, identity and seeded RGB perturbations. NumPy float64 PSNR and SciPy float64 valid 7x7 Gaussian convolution SSIM compare with the deployed PyTorch metrics, with and without 4px shaving, for one and three channels. D65/2-degree Lab and final-RGB DeltaE76 compare with scikit-image. Tolerances allow float32 arithmetic and rounded color matrices. The declared 7x7 SSIM window is not scikit-image's default 11x11 Gaussian SSIM. This audit neither changes predictions nor selects a model.",
            "references": [
                "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html",
                "https://scikit-image.org/docs/0.24.x/api/skimage.color.html#skimage.color.rgb2lab",
            ],
        },
    )
    print("Independent metric checks passed; Lab error", lab_error, "DeltaE error", delta_e_error)


if __name__ == "__main__":
    main()
