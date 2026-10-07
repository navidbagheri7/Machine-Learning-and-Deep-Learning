"""Decode received 16-QAM blocks from a saved local MLP/CNN receiver checkpoint."""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from learning_lab.artifacts import write_json  # noqa: E402
from learning_lab.receiver_inference import receive  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Float N x 2 x 64 NumPy array")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--disable-tf32", action="store_true", help="Disable CUDA convolution TF32")
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("Output must have .npz extension")
    torch.set_num_threads(4)
    predictions = receive(
        args.checkpoint,
        np.load(args.input, allow_pickle=False),
        args.device,
        args.batch_size,
        allow_tf32=not args.disable_tf32,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **predictions)
    temporary.replace(args.output)
    write_json(
        args.output.with_suffix(".json"),
        {
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "blocks": len(predictions["symbol_classes"]),
            "cuda_convolution_tf32_allowed": not args.disable_tf32,
            "cuda_matmul_tf32_allowed": False,
            "output": str(args.output),
            "protocol": "Independent circular 64-symbol blocks; channel0 real, channel1 imaginary; original unit-energy 16-QAM receiver convention. No target, SNR or gamma supplied. Class softmax probabilities are uncalibrated. Gray bits follow axis labels00,01,11,10, real-axis bits first. This is the discrete benchmark receiver, not a physical streaming-link implementation.",
        },
    )
    print("Decoded", len(predictions["symbol_classes"]), "blocks into", args.output)


if __name__ == "__main__":
    main()
