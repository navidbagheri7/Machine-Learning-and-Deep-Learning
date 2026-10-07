"""Post-fit error and calibration diagnostics; no model selection or refitting."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from learning_lab.artifacts import write_json  # noqa: E402
from learning_lab.data import adult  # noqa: E402
from learning_lab.modern_tabular import categorical_frame  # noqa: E402
from learning_lab.uncertainty import (  # noqa: E402
    binary_metrics,
    binary_temperature,
    paired_bootstrap,
    prediction_sets,
    reliability,
)


def classification(data_root):
    from catboost import CatBoostClassifier

    output = ROOT / "results/reference/modern_tabular"
    (x, y), (xt, yt) = adult(data_root)
    metadata = json.loads((output / "adult_inference_0.json").read_text())
    split = json.loads((output / "classification_split_0.json").read_text())
    artifact = output / "checkpoints/adult_0.cbm"
    model = CatBoostClassifier()
    model.load_model(str(artifact))
    frame = categorical_frame(xt)
    p = binary_temperature(model.predict_proba(frame)[:, 1], metadata["temperature"])
    stored = np.load(output / "adult_predictions_0.npz", allow_pickle=False)
    np.testing.assert_array_equal(yt, stored["y"])
    np.testing.assert_allclose(p, stored["p"], atol=1e-12)
    calibration = np.array(split["conformal_calibration"])
    cp = binary_temperature(
        model.predict_proba(categorical_frame(x.iloc[calibration]))[:, 1], metadata["temperature"]
    )
    sets = prediction_sets(np.stack([1 - p, p], 1), np.stack([1 - cp, cp], 1), y[calibration])
    groups = []
    for feature in ["sex", "race"]:
        for group in sorted(xt[feature].dropna().unique()):
            mask = xt[feature].eq(group).to_numpy()
            # Retain counts for tiny groups rather than silently omitting them.
            enough = mask.sum() >= 100 and len(np.unique(yt[mask])) == 2
            row = {
                "feature": feature,
                "group": group,
                "n": int(mask.sum()),
                "metrics_supported": enough,
            }
            if enough:
                row.update(
                    binary_metrics(yt[mask], p[mask]),
                    set_coverage=float(sets[mask, yt[mask]].mean()),
                    tpr=float(np.mean(p[mask & (yt == 1)] >= metadata["threshold"])),
                    fpr=float(np.mean(p[mask & (yt == 0)] >= metadata["threshold"])),
                )
            groups.append(row)

    def losses(target, probability):
        probability = np.clip(probability, 1e-7, 1 - 1e-7)
        return -(target * np.log(probability) + (1 - target) * np.log(1 - probability))

    probe = frame.iloc[:2500].copy()
    truth = yt[:2500]
    original = losses(truth, p[:2500]).mean()
    rng = np.random.default_rng(834)
    importance = []
    for feature in frame.columns:
        differences = []
        for _ in range(3):
            shuffled = probe.copy()
            shuffled[feature] = shuffled[feature].to_numpy()[rng.permutation(len(probe))]
            pp = binary_temperature(model.predict_proba(shuffled)[:, 1], metadata["temperature"])
            differences.append(float(losses(truth, pp).mean() - original))
        importance.append(
            {
                "feature": feature,
                "NLL increase": float(np.mean(differences)),
                "repeat SD": float(np.std(differences, ddof=1)),
            }
        )
    return {
        "groups_seed0": groups,
        "reliability_seed0": reliability(yt, p),
        "permutation_seed0": importance,
        "checkpoint_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "protocol": "Post-fit official Adult test error analysis, never tuning. Groups with fewer than100 examples or a single observed class retain counts but omit unstable AUROC/TPR/FPR. Three feature permutations on the fixed first2500 test rows use calibrated NLL; importances are associative, not causal. Threshold, temperature and conformal cutoff remain training/validation/calibration selected.",
    }


def vision():
    output = ROOT / "results/reference/modern_vision"
    rows, paired = [], []
    for seed in [0, 1, 2]:
        data = np.load(output / f"predictions_{seed}.npz", allow_pickle=False)
        y, p = data["y"], data["p"]
        prediction = p.argmax(1)
        confusion = np.zeros((10, 10), dtype=int)
        np.add.at(confusion, (y, prediction), 1)
        confidence = p.max(1)
        correct = prediction == y
        bins = []
        for low, high in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
            keep = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
            if keep.any():
                bins.append(
                    {
                        "count": int(keep.sum()),
                        "confidence": float(confidence[keep].mean()),
                        "accuracy": float(correct[keep].mean()),
                    }
                )
        ece = sum(r["count"] / len(y) * abs(r["confidence"] - r["accuracy"]) for r in bins)
        rows.append(
            {
                "seed": seed,
                "confusion": confusion,
                "calibrated_top_label_ece": ece,
                "reliability": bins,
            }
        )
        for name in ["imagenet_resnet18", "supervised_lowlabel", "contrastive_encoder"]:
            baseline = ROOT / "results/reference/vision" / f"predictions_{name}_{seed}.npz"
            old = np.load(baseline, allow_pickle=False)
            np.testing.assert_array_equal(old["y"], y)

            def loss(probability):
                return -np.log(np.clip(probability[np.arange(len(y)), y], 1e-7, 1))

            paired.append(
                {
                    "seed": seed,
                    "comparison": "ConvNeXt fine-tuning minus " + name + " NLL",
                    **paired_bootstrap(loss(p), loss(old["p"]), seed=seed),
                }
            )
    return {
        "rows": rows,
        "paired": paired,
        "protocol": "Post-fit official CIFAR test confusion and calibrated top-label reliability. No test-based changes to models or calibration. External pretraining, architecture and fine-tuning differ from frozen-encoder baselines; this is not an equal-resource algorithm comparison.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("/data/navid.bagheri-shouraki/ML_data")
    )
    args = parser.parse_args()
    write_json(
        ROOT / "results/reference/modern_tabular/diagnostics.json", classification(args.data_root)
    )
    write_json(ROOT / "results/reference/modern_vision/diagnostics.json", vision())
    print("Saved post-fit subgroup, permutation and reliability diagnostics.")


if __name__ == "__main__":
    main()
