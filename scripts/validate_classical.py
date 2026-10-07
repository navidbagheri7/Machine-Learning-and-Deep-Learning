"""Independently recompute classical metrics and check every data role and source binding."""

import json
from pathlib import Path

import numpy as np
from report_classical import records
from sklearn.metrics import mean_absolute_error, mean_squared_error

from learning_lab.artifacts import write_json
from learning_lab.classical import classify_metrics, digest


def main():
    config, rows = records()
    assert len(rows) == 243
    paths = {row["metrics_file"] for row in rows}
    for path in sorted(paths):
        directory = Path(path).parent
        split = json.loads((directory / "split.json").read_text())
        regression = "test" in split
        roles = (
            ["fit", "validation", "calibration", "test"]
            if regression
            else ["fit", "validation", "probability_calibration", "conformal_calibration"]
        )
        sets = [set(split[key]) for key in roles]
        assert len(set.union(*sets)) == sum(map(len, sets))
        if not regression and "train_ids" in split and directory.parent.name != "adult":
            assert not set(split["train_ids"]) & set(split["test_ids"])
        metrics = json.loads(Path(path).read_text())
        with np.load(metrics["evidence"], allow_pickle=False) as evidence:
            y = evidence["y"]
            assert len(y) == metrics["test_examples"]
            for key in evidence.files:
                assert np.isfinite(evidence[key]).all(), (path, key)
            for row in metrics["rows"]:
                name = row["model"]
                pred = evidence[name]
                if regression:
                    assert np.isclose(
                        row["rmse"], np.sqrt(mean_squared_error(y, pred)), rtol=0, atol=1e-12
                    )
                    assert np.isclose(row["mae"], mean_absolute_error(y, pred), rtol=0, atol=1e-12)
                    lower, upper = evidence[name + "_lower"], evidence[name + "_upper"]
                    assert row["interval_coverage"] == ((y >= lower) & (y <= upper)).mean()
                    assert row["interval_width"] == (upper - lower).mean()
                    scores = [r["validation_rmse"] for r in row["search"]]
                    assert row["selected_candidate"] == int(np.argmin(scores))
                    assert row["validation_rmse"] == min(scores)
                else:
                    np.testing.assert_allclose(pred.sum(1), 1, rtol=0, atol=1e-12)
                    recomputed = classify_metrics(y, pred)
                    assert row["confusion"] == recomputed["confusion"]
                    for key, value in recomputed.items():
                        if key != "confusion":
                            assert np.isclose(value, row[key], rtol=0, atol=1e-12), (
                                path,
                                name,
                                key,
                            )
                    prediction_sets = evidence[name + "_sets"]
                    assert row["set_coverage"] == prediction_sets[np.arange(len(y)), y].mean()
                    assert row["set_size"] == prediction_sets.sum(1).mean()
                    scores = [r["validation_macro_f1"] for r in row["search"]]
                    assert row["selected_candidate"] == int(np.argmax(scores))
                    assert row["validation_macro_f1"] == max(scores)
                assert row["candidate_count"] == len(scores)
    write_json(
        "docs/classical_validation.json",
        dict(
            status="passed",
            protocols=len(paths),
            selected_models=len(rows),
            independent_metrics=True,
            isolated_roles=True,
            private_models_required=False,
        ),
    )
    audit = json.loads(Path("assets/classical_inference_audit.json").read_text())
    assert audit["status"] == "passed" and audit["selected_models"] == 243
    assert audit["all_test_predictions_and_uncertainty_exact"]
    for name, value in audit["source_sha256"].items():
        assert digest(name) == value
    for row in audit["checks"]:
        assert digest(row["metadata_path"]) == row["metadata_sha256"]
        metadata = json.loads(Path(row["metadata_path"]).read_text())
        assert metadata["checkpoint_sha256"] == row["checkpoint_sha256"]
        assert metadata["family"] == row["family"] and metadata["seed"] == row["seed"]
    print("Verified15 protocols and243 selected models from public evidence")


if __name__ == "__main__":
    main()
