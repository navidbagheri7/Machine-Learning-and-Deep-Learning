"""Reconstruct calibration from its isolated role and audit every deployed prediction."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.datasets import load_breast_cancer, load_digits
from threadpoolctl import threadpool_limits

from learning_lab.artifacts import write_json
from learning_lab.classical import dataset_split, logits, take
from learning_lab.classical_inference import predict
from learning_lab.data import CATEGORICAL, housing
from learning_lab.uncertainty import conformal_quantile


def main():
    config = json.loads(Path("configs/classical.json").read_text())
    checks = []
    public = Path("assets/classical_calibration")
    public.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(config["threads"]):
        for dataset in config["classification_datasets"] + [
            "housing_" + p for p in config["regression_protocols"]
        ]:
            regression = dataset.startswith("housing")
            if regression:
                x, y = housing(config["data_root"])
                features = list(x.columns)
            else:
                x, y, xt, _, _, _ = dataset_split(dataset, config)
                data = (
                    load_digits()
                    if dataset == "digits"
                    else load_breast_cancer()
                    if dataset == "breast_cancer"
                    else None
                )
                features = list(x.columns) if data is None else list(data.feature_names)
                class_labels = (
                    ["at most $50,000", "above $50,000"]
                    if data is None
                    else [str(v) for v in data.target_names]
                )
            for seed in config["seeds"]:
                directory = Path(config["output"]) / dataset / str(seed)
                split = json.loads((directory / "split.json").read_text())
                metrics = json.loads((directory / "metrics.json").read_text())
                test = x.iloc[split["test"]] if regression else xt
                inputs = test if hasattr(test, "columns") else pd.DataFrame(test, columns=features)
                with np.load(metrics["evidence"], allow_pickle=False) as evidence:
                    for row in metrics["rows"]:
                        name = row["model"]
                        checkpoint = directory / "checkpoints" / (name + ".joblib")
                        assert (
                            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                            == row["checkpoint_sha256"]
                        )
                        stored = joblib.load(checkpoint)
                        metadata = dict(
                            task="regression" if regression else "classification",
                            dataset=dataset,
                            family=name,
                            seed=seed,
                            features=features,
                            categorical=CATEGORICAL if dataset == "adult" else [],
                            checkpoint_sha256=row["checkpoint_sha256"],
                            alpha=config["conformal_alpha"],
                            metrics_path=str(directory / "metrics.json"),
                        )
                        if regression:
                            cal = split["calibration"]
                            cutoff = conformal_quantile(
                                abs(y[cal] - stored.predict(x.iloc[cal])), config["conformal_alpha"]
                            )
                            metadata["absolute_residual_cutoff"] = cutoff
                        else:
                            ids = split["conformal_calibration"]
                            p = softmax(
                                logits(stored["model"], take(x, ids)) / stored["temperature"],
                                axis=1,
                            )
                            cutoff = conformal_quantile(
                                1 - p[np.arange(len(ids)), y[ids]], config["conformal_alpha"]
                            )
                            metadata.update(
                                temperature=stored["temperature"],
                                conformal_cutoff=cutoff,
                                classes=stored["model"].classes_.tolist(),
                                class_labels=class_labels,
                            )
                        output = public / f"{dataset}_{name}_{seed}.json"
                        write_json(output, metadata)
                        actual = predict(checkpoint, output, inputs)
                        if regression:
                            np.testing.assert_array_equal(
                                actual.prediction_100000_usd, evidence[name]
                            )
                            np.testing.assert_array_equal(actual.lower, evidence[name + "_lower"])
                            np.testing.assert_array_equal(actual.upper, evidence[name + "_upper"])
                        else:
                            columns = [f"probability_class_{k}" for k in metadata["classes"]]
                            np.testing.assert_array_equal(
                                actual[columns].to_numpy(), evidence[name]
                            )
                            columns = [f"set_contains_class_{k}" for k in metadata["classes"]]
                            np.testing.assert_array_equal(
                                actual[columns].to_numpy(), evidence[name + "_sets"]
                            )
                        checks.append(
                            dict(
                                dataset=dataset,
                                seed=seed,
                                family=name,
                                predictions=len(actual),
                                metadata_path=str(output),
                                metadata_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                                checkpoint_sha256=row["checkpoint_sha256"],
                            )
                        )
                print("Audited deployment", dataset, seed, flush=True)
    assert len(checks) == 243
    write_json(
        "assets/classical_inference_audit.json",
        dict(
            status="passed",
            selected_models=243,
            all_test_predictions_and_uncertainty_exact=True,
            checks=checks,
            source_sha256={
                name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in [
                    "src/learning_lab/classical_inference.py",
                    "src/learning_lab/classical.py",
                    "src/learning_lab/uncertainty.py",
                    "scripts/predict_classical.py",
                    "scripts/export_classical_inference.py",
                ]
            },
        ),
    )


if __name__ == "__main__":
    main()
