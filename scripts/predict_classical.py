"""Predict CSV observations using any selected classical model and its calibration."""

import argparse
from pathlib import Path

import pandas as pd

from learning_lab.classical_inference import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = predict(args.checkpoint, args.metadata, pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print("Predicted", len(result), "observations")


if __name__ == "__main__":
    main()
