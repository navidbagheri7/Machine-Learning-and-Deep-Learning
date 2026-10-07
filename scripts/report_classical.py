"""Regenerate comparative classical figures and tables from complete numeric evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from learning_lab.artifacts import write_json
from learning_lab.classical import classifier_candidates, digest, regressor_candidates


def records():
    config = json.loads(Path("configs/classical.json").read_text())
    rows = []
    for dataset in config["classification_datasets"] + [
        "housing_" + p for p in config["regression_protocols"]
    ]:
        for seed in config["seeds"]:
            root = Path(config["output"]) / dataset / str(seed)
            record = json.loads((root / "run.json").read_text())
            assert record["status"] == "complete" and not record["config"]["config"].get(
                "quick", False
            )
            assert record["config"]["config"] == config
            metrics = json.loads((root / "metrics.json").read_text())
            assert digest(metrics["evidence"]) == metrics["evidence_sha256"]
            methods = (
                regressor_candidates(seed, config)
                if dataset.startswith("housing")
                else classifier_candidates(dataset, seed, config)
            )
            assert {r["model"] for r in metrics["rows"]} == set(methods)
            for name, value in record["fitting_source_sha256"].items():
                assert digest(Path("src/learning_lab") / name) == value
            rows.extend(
                dict(case=dataset, metrics_file=str(root / "metrics.json"), **r)
                for r in metrics["rows"]
            )
    return config, rows


def main():
    config, rows = records()
    assert len(rows) == 243
    frame = pd.DataFrame(rows)
    classification = frame[~frame.case.str.startswith("housing")]
    regression = frame[frame.case.str.startswith("housing")]
    classification.groupby(["case", "model"])[
        [
            "accuracy",
            "macro_f1",
            "log_loss",
            "roc_auc",
            "set_coverage",
            "set_size",
            "seconds_including_selection",
        ]
    ].agg(["mean", "std"]).to_csv("docs/classical_classification.csv")
    regression.groupby(["case", "model"])[
        ["rmse", "mae", "r2", "interval_coverage", "interval_width", "seconds_including_selection"]
    ].agg(["mean", "std"]).to_csv("docs/classical_regression.csv")
    for dataset in frame.case.unique():
        subset = frame[frame.case == dataset]
        key = "rmse" if dataset.startswith("housing") else "log_loss"
        grouped = subset.groupby("model")[key].agg(["mean", "std"]).sort_values("mean")
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
        ax.barh(grouped.index, grouped["mean"], xerr=grouped["std"], color="#087b94")
        ax.invert_yaxis()
        ax.set(
            xlabel=key.replace("_", " ") + " (lower is better)",
            title=dataset.replace("_", " ") + " — three independently fitted seeds",
        )
        ax.grid(axis="x", alpha=0.2)
        for suffix in ["png", "pdf"]:
            fig.savefig(f"assets/classical_{dataset}.{suffix}", dpi=170, bbox_inches="tight")
        plt.close(fig)
    columns = [
        "case",
        "model",
        "seed",
        "accuracy",
        "macro_f1",
        "log_loss",
        "roc_auc",
        "rmse",
        "mae",
        "interval_coverage",
        "interval_width",
        "set_coverage",
        "set_size",
        "seconds_including_selection",
        "test_prediction_seconds",
        "candidate_count",
        "selected_candidate",
        "metrics_file",
    ]
    frame[columns].to_csv("docs/classical_individual_fits.csv", index=False)
    lines = [
        "# Classical prediction comparisons",
        "",
        "Fifteen full protocols contain 243 selected models across three fitting seeds. All methods share case-specific fitting, selection and calibration partitions; stacking refits preprocessing inside its inner folds. Candidate counts and compute differ by family and are recorded. Test data do not choose hyperparameters. Historical studies remain available separately.",
        "",
        "## Classification",
        "",
        "Validation macro-F1 selects presets. Probability temperature and prediction-set cutoffs use two separate calibration partitions. The Adult official test is preserved; numeric datasets have a common fixed stratified test. RBF SVM runs on the numeric datasets, alongside the existing label-budget kernel study, and is explicitly excluded from full Adult.",
    ]
    for dataset in config["classification_datasets"]:
        grouped = (
            classification[classification.case == dataset]
            .groupby("model")[["accuracy", "log_loss", "macro_f1", "set_coverage"]]
            .agg(["mean", "std"])
            .sort_values(("log_loss", "mean"))
        )
        lines.extend(
            [
                "",
                f"### {dataset.replace('_', ' ').title()}",
                "",
                "| Model | Accuracy | NLL, mean ± fit SD | Macro-F1 | Set coverage |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, row in grouped.iterrows():
            lines.append(
                f"| {name} | {row[('accuracy', 'mean')]:.4f} | {row[('log_loss', 'mean')]:.4f} ± {row[('log_loss', 'std')]:.4f} | {row[('macro_f1', 'mean')]:.4f} | {row[('set_coverage', 'mean')]:.4f} |"
            )
        lines.extend(["", f"![{dataset} comparison](../assets/classical_{dataset}.png)"])
    lines.extend(
        [
            "",
            "## Regression",
            "",
            "California Housing uses the historical random and geographic partitions. Absolute-residual conformal intervals target 90% coverage under exchangeability; geographic coverage failure is retained. Housing target units are $100,000. These are declared practical presets with different tuning and compute budgets.",
        ]
    )
    for protocol in config["regression_protocols"]:
        case = "housing_" + protocol
        grouped = (
            regression[regression.case == case]
            .groupby("model")[["rmse", "mae", "interval_coverage", "interval_width"]]
            .agg(["mean", "std"])
            .sort_values(("rmse", "mean"))
        )
        lines.extend(
            [
                "",
                f"### {protocol.title()}",
                "",
                "| Model | RMSE, mean ± fit SD | MAE | Coverage | Mean width |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, row in grouped.iterrows():
            lines.append(
                f"| {name} | {row[('rmse', 'mean')]:.4f} ± {row[('rmse', 'std')]:.4f} | {row[('mae', 'mean')]:.4f} | {row[('interval_coverage', 'mean')]:.4f} | {row[('interval_width', 'mean')]:.4f} |"
            )
        lines.extend(["", f"![{case} comparison](../assets/classical_{case}.png)"])
    lines.extend(
        [
            "",
            "## Interpretation and reproduction",
            "",
            "LDA/QDA Gaussian class assumptions, KNN geometry, linear penalties and ensemble diversity are explicit modeling choices. QDA rank warnings and convergence warnings are recorded per candidate; regularized covariances are used. CatBoost here receives the common transformed numeric features; the native categorical extension remains a separately labeled benchmark.",
            "",
            "Three-seed SD is fitting variability; example-wise uncertainty conditional on a fitted model is a different quantity. The fixed numeric test is excluded from fitting, preprocessing, selection and calibration. Bundled Breast Cancer and Digits datasets are educational benchmarks; conclusions do not establish clinical deployment or large-image performance.",
            "",
            "```bash",
            "python scripts/run_classical.py --dataset adult --seed 0",
            "python scripts/run_classical.py --dataset housing_random --seed 0",
            "python scripts/report_classical.py",
            "python scripts/validate_classical.py",
            "```",
            "",
            "Repeat seeds 0, 1, 2 and all five declared cases. `--quick` writes separate smoke outputs and uses fewer examples/trees. Saved selected sklearn models remain private; public predictions support independent metric verification. Full candidate parameters, timings, warnings and source digests are retained in each protocol. See [individual fits](classical_individual_fits.csv), [classification aggregates](classical_classification.csv) and [regression aggregates](classical_regression.csv).",
            "",
            "Primary references: [scikit-learn LDA/QDA](https://scikit-learn.org/stable/modules/lda_qda.html), [neighbors](https://scikit-learn.org/stable/modules/neighbors.html), [ensembles](https://scikit-learn.org/stable/modules/ensemble.html), [linear models](https://scikit-learn.org/stable/modules/linear_model.html), and [support vector machines](https://scikit-learn.org/stable/modules/svm.html).",
        ]
    )
    Path("docs/classical_results.md").write_text("\n".join(lines) + "\n")
    write_json(
        "docs/classical_completion.json",
        dict(
            status="complete",
            protocols=15,
            fitted_models=len(rows),
            seeds=config["seeds"],
            scope="Three classical classification datasets, two Housing evaluation protocols; modern native boosting and deep studies preserved separately",
        ),
    )
    print("Reported243 fitted classical models across15 complete protocols")


if __name__ == "__main__":
    main()
