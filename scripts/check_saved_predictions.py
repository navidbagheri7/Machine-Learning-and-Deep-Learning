"""Independently reproduce saved predictions from all modern fitted checkpoints.

Checks fixed test subsets without fitting, calibrating or selecting anything.
CPU ConvNeXt probabilities are reported separately from reference CUDA bfloat16.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from learning_lab.artifacts import write_json  # noqa: E402
from learning_lab.data import adult, housing  # noqa: E402
from learning_lab.modern_inference import (  # noqa: E402
    adult_predictions,
    cifar_probabilities,
    housing_predictions,
)
from learning_lab.vision import load_cifar  # noqa: E402


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--results-root", default=ROOT / "results/reference", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=Path(".local/saved_prediction_check.json"), type=Path)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    (_, _), (features, _) = adult(args.data_root)
    hx, _ = housing(args.data_root)
    (_, _), (images, _) = load_cifar(args.image_root)
    tabular, vision = args.results_root / "modern_tabular", args.results_root / "modern_vision"
    metrics = json.loads((vision / "metrics.json").read_text())
    checks = []
    for seed in [0, 1, 2]:
        checkpoint = tabular / "checkpoints" / f"adult_{seed}.cbm"
        actual = adult_predictions(
            checkpoint, tabular / f"adult_inference_{seed}.json", features.iloc[:16]
        ).probability_income_above_50k.to_numpy()
        with np.load(tabular / f"adult_predictions_{seed}.npz", allow_pickle=False) as saved:
            np.testing.assert_allclose(actual, saved["p"][:16], atol=1e-12, rtol=0)
            difference = float(abs(actual - saved["p"][:16]).max())
        checks.append(
            {
                "task": "Adult",
                "seed": seed,
                "n": 16,
                "maximum_difference": difference,
                "checkpoint_sha256": sha256(checkpoint),
            }
        )
        for protocol in ["random", "geographic"]:
            checkpoint = tabular / "checkpoints" / f"housing_{protocol}_{seed}.cbm"
            indices = json.loads(
                (tabular / f"regression_split_{protocol}_{seed}.json").read_text()
            )["test"][:16]
            actual = housing_predictions(
                checkpoint, hx.iloc[indices]
            ).median_house_value_100000_usd.to_numpy()
            with np.load(
                tabular / f"housing_predictions_{protocol}_{seed}.npz", allow_pickle=False
            ) as saved:
                np.testing.assert_allclose(actual, saved["p"][:16], atol=1e-12, rtol=0)
                difference = float(abs(actual - saved["p"][:16]).max())
            checks.append(
                {
                    "task": "Housing " + protocol,
                    "seed": seed,
                    "n": 16,
                    "maximum_difference": difference,
                    "checkpoint_sha256": sha256(checkpoint),
                }
            )
        checkpoint = vision / "checkpoints" / f"convnext_{seed}.pt"
        temperature = next(r["temperature"] for r in metrics["rows"] if r["seed"] == seed)
        with np.load(vision / f"predictions_{seed}.npz", allow_pickle=False) as saved:
            expected = saved["p"][:64]
        for device in dict.fromkeys([args.device, "cpu"]):
            actual = cifar_probabilities(checkpoint, images[:64], temperature, device)
            difference = float(abs(actual - expected).max())
            if device.startswith("cuda"):
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
            checks.append(
                {
                    "task": "CIFAR",
                    "seed": seed,
                    "n": 64,
                    "device": device,
                    "maximum_probability_difference": difference,
                    "argmax_agreement": float((actual.argmax(1) == expected.argmax(1)).mean()),
                    "checkpoint_sha256": sha256(checkpoint),
                }
            )
    write_json(
        args.output,
        {
            "status": "passed",
            "checks": checks,
            "protocol": "Independent saved-model inference on fixed first16 Adult and Housing test rows and first64 CIFAR test images for all three seeds. CUDA probability agreement required within1e-6. CPU float32 versus reference CUDA bfloat16 differences are descriptive, not assumed identical. No model or calibration fitting.",
            "script_sha256": sha256(Path(__file__)),
        },
    )
    print("Saved predictions independently checked for all three fitting seeds.")


if __name__ == "__main__":
    main()
