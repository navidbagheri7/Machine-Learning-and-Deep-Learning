"""Export deployment calibration from existing fitting/calibration partitions.

No models are fitted. The fixed conformal partitions reproduce previously measured
coverage exactly; official test labels only verify exported results.
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
from learning_lab.modern_inference import housing_predictions  # noqa: E402
from learning_lab.modern_tabular import categorical_frame  # noqa: E402
from learning_lab.uncertainty import binary_temperature, conformal_quantile  # noqa: E402


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from catboost import CatBoostClassifier, CatBoostRegressor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--results-root", type=Path, default=ROOT / "results/reference/modern_tabular"
    )
    args = parser.parse_args()
    output = args.results_root
    (x, y), _ = adult(args.data_root)
    metrics = json.loads((output / "metrics.json").read_text())
    checks = []
    for seed in [0, 1, 2]:
        indices = json.loads((output / f"classification_split_{seed}.json").read_text())[
            "conformal_calibration"
        ]
        path = output / f"adult_inference_{seed}.json"
        metadata = json.loads(path.read_text())
        checkpoint = output / "checkpoints" / f"adult_{seed}.cbm"
        model = CatBoostClassifier()
        model.load_model(str(checkpoint))
        p = binary_temperature(
            model.predict_proba(categorical_frame(x.iloc[indices]))[:, 1], metadata["temperature"]
        )
        cutoff = conformal_quantile(
            1 - np.stack([1 - p, p], 1)[np.arange(len(indices)), y[indices]]
        )
        metadata.update(
            alpha=0.1, conformal_score_cutoff=cutoff, checkpoint_sha256=sha256(checkpoint)
        )
        write_json(path, metadata)
        with np.load(output / f"adult_predictions_{seed}.npz", allow_pickle=False) as data:
            coverage = np.mean(
                1 - np.stack([1 - data["p"], data["p"]], 1)[np.arange(len(data["y"])), data["y"]]
                <= cutoff
            )
        expected = next(r["set_coverage"] for r in metrics["classification"] if r["seed"] == seed)
        assert coverage == expected
        checks.append(
            {
                "task": "Adult",
                "seed": seed,
                "calibration_n": len(indices),
                "test_coverage": float(coverage),
                "checkpoint_sha256": sha256(checkpoint),
            }
        )
    x, y = housing(args.data_root)
    for protocol in ["random", "geographic"]:
        for seed in [0, 1, 2]:
            split = json.loads((output / f"regression_split_{protocol}_{seed}.json").read_text())
            cal, test = split["calibration"], split["test"]
            checkpoint = output / "checkpoints" / f"housing_{protocol}_{seed}.cbm"
            quantile_checkpoint = output / "checkpoints" / f"housing_quantile_{protocol}_{seed}.cbm"
            model, quantile = CatBoostRegressor(), CatBoostRegressor()
            model.load_model(str(checkpoint))
            quantile.load_model(str(quantile_checkpoint))
            point, bounds = (
                model.predict(x.iloc[cal]),
                np.sort(quantile.predict(x.iloc[cal]), axis=1),
            )
            path = output / f"housing_inference_{protocol}_{seed}.json"
            write_json(
                path,
                {
                    "features": list(x.columns),
                    "alpha": 0.1,
                    "protocol": protocol,
                    "target_units": "$100,000",
                    "absolute_residual_cutoff": conformal_quantile(abs(y[cal] - point)),
                    "cqr_cutoff": conformal_quantile(
                        np.maximum(bounds[:, 0] - y[cal], y[cal] - bounds[:, 1])
                    ),
                    "checkpoint_sha256": sha256(checkpoint),
                    "quantile_checkpoint_sha256": sha256(quantile_checkpoint),
                },
            )
            actual = housing_predictions(checkpoint, x.iloc[test], path, quantile_checkpoint)
            with np.load(
                output / f"housing_predictions_{protocol}_{seed}.npz", allow_pickle=False
            ) as data:
                for field, key in [
                    ("split_conformal_lower", "lo"),
                    ("split_conformal_upper", "hi"),
                    ("cqr_lower", "cqr_lo"),
                    ("cqr_upper", "cqr_hi"),
                ]:
                    np.testing.assert_allclose(actual[field], data[key], atol=1e-12, rtol=0)
            checks.append(
                {
                    "task": "Housing " + protocol,
                    "seed": seed,
                    "calibration_n": len(cal),
                    "test_n": len(test),
                    "all_test_intervals_verified": True,
                    "checkpoint_sha256": sha256(checkpoint),
                    "quantile_checkpoint_sha256": sha256(quantile_checkpoint),
                }
            )
    write_json(
        output / "deployment_calibration_audit.json",
        {
            "status": "passed",
            "checks": checks,
            "script_sha256": sha256(Path(__file__)),
            "source_sha256": {
                name: sha256(ROOT / name)
                for name in [
                    "scripts/export_uncertainty.py",
                    "src/learning_lab/modern_inference.py",
                ]
            },
            "protocol": "Fixed training conformal partitions, unchanged fitted models and temperatures. All prior Adult coverage and every previously saved Housing test endpoint reproduced. Test labels do not choose cutoffs. Geographic interval validity is not guaranteed.",
        },
    )
    print("Exported uncertainty metadata and verified every saved Housing test interval.")


if __name__ == "__main__":
    main()
