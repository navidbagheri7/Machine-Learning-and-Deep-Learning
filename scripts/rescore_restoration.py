"""Refresh final-output color metrics without retraining or changing checkpoints.

The training record and validation curves are preserved. A separate evaluation
audit records the scoring source hash and timestamp. Only completed runs enter.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from learning_lab.artifacts import write_json  # noqa: E402
from learning_lab.external_restoration import (  # noqa: E402
    BigColorReference,
    CT2Reference,
    DDColorReference,
    RealESRGANReference,
    SwinIRReference,
)
from learning_lab.generative import build_model, evaluate, save_arrays  # noqa: E402

SCORING_SOURCE_SHA256 = hashlib.sha256(
    (ROOT / "src/learning_lab/generative.py").read_bytes()
).hexdigest()
ADAPTER_SOURCE_SHA256 = hashlib.sha256(
    (ROOT / "src/learning_lab/external_restoration.py").read_bytes()
).hexdigest()

STUDIES = [
    f"generative_{task}/{method}"
    for task in ["colorization", "super_resolution"]
    for method in ["baseline", "l1", "gan", "cvae", "diffusion"]
] + ["external_" + method for method in ["ddcolor", "ct2", "bigcolor", "swinir", "realesrgan"]]


def score(study, device, force=False):
    output = ROOT / "results/reference" / study
    if not (output / "metrics.json").exists():
        return False
    record = json.loads((output / "run.json").read_text())
    if record["status"] != "complete":
        return False
    metrics = json.loads((output / "metrics.json").read_text())
    config = record["config"] | {"device": device}
    torch.hub.set_dir(str(Path(config["data_root"]) / "models/torch/hub"))
    rows, scores, galleries, checkpoint_hashes = [], {}, {}, {}
    method = config.get("reference", config.get("methods", [None])[0])
    seeds = [0] if study.startswith("external_") or method == "baseline" else config["seeds"]
    samples = config["samples"] if method in ["cvae", "diffusion", "bigcolor"] else 1
    expected = {(seed, sample) for seed in seeds for sample in range(samples)}
    if (
        not force
        and all(row.get("metric_protocol_version") == 3 for row in metrics["rows"])
        and {(row["seed"], row["sample"]) for row in metrics["rows"]} == expected
    ):
        return True
    for seed in seeds:
        if study.startswith("external_"):
            model = {
                "ddcolor": DDColorReference,
                "ct2": CT2Reference,
                "bigcolor": BigColorReference,
                "swinir": SwinIRReference,
                "realesrgan": RealESRGANReference,
            }[method](config)
        elif method == "baseline":
            model = None
        else:
            path = output / "checkpoints" / f"{method}_{seed}.pt"
            checkpoint_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            state = torch.load(path, map_location="cpu", weights_only=True)
            model = build_model(
                method, config["task"], state["config"], initialize_pretrained=False
            )
            model.load_state_dict(state["state_dict"], strict=True)
        if model is not None:
            model.to(device).eval().requires_grad_(False)
        values, arrays, pictures = evaluate(model, method, seed, config)
        rows.extend(values)
        scores.update(arrays)
        galleries.update(pictures)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    save_arrays(output / "per_image_scores.npz", scores)
    save_arrays(output / "gallery.npz", galleries)
    metrics["rows"] = rows
    if study == "external_bigcolor":
        metrics["protocol"] = metrics["protocol"].replace(
            "one fixed random latent", "all four fixed random latent draws"
        )
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "evaluation_audit.json",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "metric_protocol_version": 3,
            "definition": "Final RGB-derived three-channel DeltaE76 and chroma. Stochastic RGB L1 dispersion from first draw on every test image; sample 0 excluded from dispersion summaries. Other metrics unchanged.",
            "scoring_source_sha256": SCORING_SOURCE_SHA256,
            "adapter_source_sha256": ADAPTER_SOURCE_SHA256,
            "evaluation_seeds": seeds,
            "checkpoint_sha256": checkpoint_hashes,
        },
    )
    print("Rescored", study, flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wait", action="store_true", help="Wait for pending training runs")
    parser.add_argument(
        "--force", action="store_true", help="Refresh even already-current metric records"
    )
    parser.add_argument("--studies", nargs="+", default=STUDIES)
    args = parser.parse_args()
    torch.set_num_threads(4)
    pending = list(args.studies)
    while pending:
        for study in pending.copy():
            if score(study, args.device, args.force):
                pending.remove(study)
        if not args.wait:
            break
        if pending:
            time.sleep(30)
    print("Pending", pending, flush=True)


if __name__ == "__main__":
    main()
