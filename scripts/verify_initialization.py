"""Verify that every refined CVAE checkpoint retains the selected frozen semantic mean."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from learning_lab.artifacts import write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=ROOT / "results/reference/generative_colorization/cvae"
    )
    parser.add_argument("--mean-checkpoint", type=Path)
    args = parser.parse_args()
    output = args.run
    config = json.loads((output / "run.json").read_text())["config"]
    path = args.mean_checkpoint or ROOT / config["mean_checkpoint"]
    mean = torch.load(path, map_location="cpu", weights_only=True)
    expected = {k.removeprefix("model."): v for k, v in mean["state_dict"].items()}
    checks = []
    for seed in config["seeds"]:
        checkpoint = output / "checkpoints" / f"cvae_{seed}.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        actual = {
            k.removeprefix("semantic."): v
            for k, v in state["state_dict"].items()
            if k.startswith("semantic.")
        }
        assert actual.keys() == expected.keys()
        assert all(torch.equal(actual[k], expected[k]) for k in actual)
        checks.append(
            {
                "seed": seed,
                "semantic_tensors_verified": len(actual),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        )
    write_json(
        output / "initialization_audit.json",
        {
            "status": "passed",
            "mean_checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mean_selected_epoch": mean["best_epoch"],
            "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checks": checks,
            "protocol": "Post-fit verification that all saved semantic parameters and BN buffers exactly equal the validation-selected direct predictor. These parameters are frozen during latent fitting. Both latent seeds share this fitted mean; no independent end-to-end seed uncertainty is claimed.",
        },
    )
    print("Frozen semantic initialization verified for all latent seeds.")


if __name__ == "__main__":
    main()
