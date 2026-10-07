"""Export device-specific ConvNeXt temperature and conformal calibration.

Weights, partitions and original CUDA results remain unchanged. CPU calibration
uses the same reserved probability/conformal partitions with its float32 predictor.
Official test labels only verify outcomes and never choose a cutoff.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import log_loss
from torchvision.models import convnext_tiny

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from learning_lab.artifacts import write_json  # noqa: E402
from learning_lab.modern_inference import CIFAR_CLASSES, cifar_predictions  # noqa: E402
from learning_lab.modern_vision import logits  # noqa: E402
from learning_lab.uncertainty import conformal_quantile, fit_temperature  # noqa: E402
from learning_lab.vision import load_cifar  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--results-root", type=Path, default=ROOT / "results/reference/modern_vision"
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA reference inference device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scratch-dir", type=Path, default=ROOT / ".local/cifar_calibration")
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        parser.error("Reference device must be CUDA; CPU profile is evaluated separately")
    output = args.output_dir or args.results_root
    output.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    (images, labels), (tests, truth) = load_cifar(args.image_root)
    record = json.loads((args.results_root / "run.json").read_text())
    assert record["status"] == "complete" and not record["config"].get("quick")
    original = json.loads((args.results_root / "metrics.json").read_text())
    rows, commands = [], []
    test_file = args.scratch_dir / "test_subset.npy"
    np.save(test_file, tests[:128])
    environment = os.environ | {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    for seed in [0, 1, 2]:
        split = json.loads((args.results_root / f"split_{seed}.json").read_text())
        prob, conf = split["probability_calibration"], split["conformal_calibration"]
        assert not set(prob) & set(conf)
        checkpoint = args.results_root / f"checkpoints/convnext_{seed}.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = convnext_tiny(weights=None)
        model.classifier[-1] = torch.nn.Linear(768, 10)
        model.load_state_dict(state["state_dict"], strict=True)
        model.eval().requires_grad_(False)
        baseline = next(r for r in original["rows"] if r["seed"] == seed)
        for profile, device in [("cuda_bfloat16", args.device), ("cpu_float32", "cpu")]:
            model.to(device)
            arrays = [
                torch.tensor(values, dtype=torch.float32, device=device) / 255
                for values in [images[prob], images[conf], tests]
            ]
            raw_prob, raw_conf, raw_test = [logits(model, a, state["config"]) for a in arrays]
            del arrays
            temperature = (
                baseline["temperature"]
                if profile == "cuda_bfloat16"
                else fit_temperature(raw_prob, labels[prob])
            )
            cal_p, p = [
                softmax(np.asarray(a, dtype=np.float64) / temperature, axis=1)
                for a in [raw_conf, raw_test]
            ]
            scores = 1 - cal_p[np.arange(len(conf)), labels[conf]]
            cutoff = conformal_quantile(scores, alpha=0.1)
            sets = 1 - p <= cutoff
            size = sets.sum(1)
            coverage = float(sets[np.arange(len(truth)), truth].mean())
            if profile == "cuda_bfloat16":
                with np.load(args.results_root / f"predictions_{seed}.npz") as saved:
                    np.testing.assert_allclose(p, saved["p"], atol=1e-6, rtol=0)
                assert coverage == baseline["set_coverage"]
                assert float(size.mean()) == baseline["set_size"]
            metadata = output / f"cifar_inference_{profile}_{seed}.json"
            write_json(
                metadata,
                {
                    "checkpoint_sha256": digest(checkpoint),
                    "class_names": CIFAR_CLASSES,
                    "evaluation_profile": profile,
                    "temperature": temperature,
                    "alpha": 0.1,
                    "conformal_score_cutoff": cutoff,
                    "probability_calibration_n": len(prob),
                    "conformal_calibration_n": len(conf),
                    "split_sha256": digest(args.results_root / f"split_{seed}.json"),
                    "protocol": "Fixed independent probability and conformal partitions. CUDA retains its original temperature; CPU temperature is fitted using only probability-calibration labels. No test selection. Metadata applies only to the declared inference precision profile.",
                },
            )
            np.savez_compressed(
                output / f"cifar_calibration_{profile}_{seed}.npz",
                p=p,
                y=truth,
                cal_scores=scores,
                conformal_indices=conf,
            )
            singleton = size == 1
            rows.append(
                {
                    "seed": seed,
                    "profile": profile,
                    "test_images": len(truth),
                    "temperature": temperature,
                    "cutoff": cutoff,
                    "accuracy": float((p.argmax(1) == truth).mean()),
                    "nll": float(log_loss(truth, p)),
                    "set_coverage": coverage,
                    "set_size": float(size.mean()),
                    "empty_set_rate": float((size == 0).mean()),
                    "singleton_fraction": float(singleton.mean()),
                    "singleton_accuracy": float(
                        (p.argmax(1)[singleton] == truth[singleton]).mean()
                    ),
                    "calibration_n": len(conf),
                    "checkpoint_sha256": digest(checkpoint),
                }
            )
            api = cifar_predictions(checkpoint, tests[:128], metadata, device)
            np.testing.assert_allclose(api[CIFAR_CLASSES], p[:128], atol=1e-6, rtol=0)
            np.testing.assert_array_equal(api.set_size, size[:128])
            csv = args.scratch_dir / f"{profile}_{seed}.csv"
            with csv.with_suffix(".log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/predict_modern.py",
                        "cifar",
                        "--checkpoint",
                        str(checkpoint.resolve()),
                        "--metadata",
                        str(metadata.resolve()),
                        "--input",
                        str(test_file.resolve()),
                        "--output",
                        str(csv.resolve()),
                        "--device",
                        device,
                    ],
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            deployed = pd.read_csv(csv)
            np.testing.assert_allclose(deployed[CIFAR_CLASSES], p[:128], atol=1e-6, rtol=0)
            np.testing.assert_array_equal(deployed.set_empty, size[:128] == 0)
            commands.append({"seed": seed, "profile": profile, "images": 128, "status": "passed"})
            print(
                "Verified CIFAR calibration",
                seed,
                profile,
                coverage,
                "empty",
                float((size == 0).mean()),
                flush=True,
            )
        del model, state
    write_json(
        output / "cifar_calibration_audit.json",
        {
            "status": "passed",
            "rows": rows,
            "commands": commands,
            "script_sha256": digest(Path(__file__)),
            "source_sha256": {
                name: digest(ROOT / name)
                for name in ["src/learning_lab/modern_inference.py", "scripts/predict_modern.py"]
            },
            "protocol": "Unchanged three fitted ConvNeXt checkpoints, original disjoint calibration partitions and all10,000 official test images. CUDA float64 scoring reproduces original probabilities within1e-6 and prediction-set coverage/size exactly. CPU uses its own reserved-partition temperature/conformal calibration. All six metadata-bound image commands verified on128 test images. Empty sets and singleton conditional accuracy retained; no claim of guaranteed conditional coverage. No test labels select a temperature, cutoff or model.",
        },
    )


if __name__ == "__main__":
    main()
