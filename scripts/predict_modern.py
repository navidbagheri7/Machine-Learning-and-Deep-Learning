"""Batch inference from locally fitted CatBoost or ConvNeXt checkpoints."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from learning_lab.modern_inference import (  # noqa: E402
    adult_predictions,
    cifar_predictions,
    cifar_probabilities,
    housing_predictions,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["adult", "housing", "cifar"])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, help="Saved calibration/inference JSON")
    parser.add_argument(
        "--quantile-checkpoint", type=Path, help="Housing quantile model for CQR intervals"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Saved probability-calibration temperature for CIFAR",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.task == "cifar":
        if args.metadata is not None:
            if args.temperature is not None:
                parser.error("CIFAR metadata fixes temperature; omit --temperature")
            cifar_predictions(
                args.checkpoint, np.load(args.input, allow_pickle=False), args.metadata, args.device
            ).to_csv(args.output, index=False)
            print("Saved", args.output)
            return
        prediction = cifar_probabilities(
            args.checkpoint,
            np.load(args.input, allow_pickle=False),
            1.0 if args.temperature is None else args.temperature,
            args.device,
        )
        pd.DataFrame(
            prediction,
            columns=[
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ],
        ).to_csv(args.output, index=False)
    else:
        features = pd.read_csv(args.input, na_values="?")
        if args.task == "adult":
            if args.metadata is None:
                parser.error("Adult requires --metadata with the saved temperature and threshold")
            prediction = adult_predictions(args.checkpoint, args.metadata, features)
        else:
            prediction = housing_predictions(
                args.checkpoint, features, args.metadata, args.quantile_checkpoint
            )
        prediction.to_csv(args.output, index=False)
    print("Saved", args.output)


if __name__ == "__main__":
    main()
