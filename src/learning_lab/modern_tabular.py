"""Native categorical boosting and quantile uncertainty on the existing held-out protocols."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from .artifacts import Run, write_json
from .data import CATEGORICAL, adult, housing
from .tabular import _interval_metrics, _split
from .uncertainty import (
    binary_metrics,
    binary_temperature,
    conformal_quantile,
    cqr_interval,
    fit_temperature,
    paired_bootstrap,
    prediction_sets,
    regression_interval,
)


def categorical_frame(frame):
    frame = frame.copy()
    for column in CATEGORICAL:
        frame[column] = frame[column].fillna("__missing__").astype(str)
    return frame


def experiment(config):
    from catboost import CatBoostClassifier, CatBoostRegressor

    run = Run(config["output"], config)
    (x, y), (xt, yt) = adult(config["data_root"])
    x, xt = categorical_frame(x), categorical_frame(xt)
    classification, regression, selections, comparisons = [], [], [], []
    checkpoints = run.output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    for seed in config["seeds"]:
        start = time.monotonic()
        fit, val, prob, conf = _split(np.arange(len(y)), y, seed)
        candidates = []
        for depth in config["depths"]:
            model = CatBoostClassifier(
                iterations=config["iterations"],
                depth=depth,
                learning_rate=0.05,
                loss_function="Logloss",
                random_seed=seed,
                thread_count=4,
                cat_features=CATEGORICAL,
                allow_writing_files=False,
                verbose=False,
            )
            model.fit(
                x.iloc[fit], y[fit], eval_set=(x.iloc[val], y[val]), early_stopping_rounds=100
            )
            nll = binary_metrics(y[val], model.predict_proba(x.iloc[val])[:, 1])["log_loss"]
            candidates.append((nll, model, depth))
        _, model, depth = min(candidates, key=lambda v: v[0])
        rawprob, rawconf, rawtest, rawval = [
            model.predict_proba(z)[:, 1] for z in [x.iloc[prob], x.iloc[conf], xt, x.iloc[val]]
        ]
        temperature = fit_temperature(
            np.stack([np.zeros(len(prob)), logit(np.clip(rawprob, 1e-7, 1 - 1e-7))], 1), y[prob]
        )
        pc, pt, pv = [binary_temperature(p, temperature) for p in [rawconf, rawtest, rawval]]
        sets = prediction_sets(np.stack([1 - pt, pt], 1), np.stack([1 - pc, pc], 1), y[conf])
        thresholds = np.linspace(0.05, 0.95, 91)

        def cost(truth, p, t):
            return np.mean(2 * ((p < t) & (truth == 1)) + ((p >= t) & (truth == 0)))

        threshold = float(min(thresholds, key=lambda t: cost(y[val], pv, t)))
        row = {
            "model": "catboost_native",
            "seed": seed,
            **binary_metrics(yt, pt),
            "temperature": temperature,
            "set_coverage": float(sets[np.arange(len(yt)), yt].mean()),
            "set_size": float(sets.sum(1).mean()),
            "decision_threshold": threshold,
            "decision_cost": float(cost(yt, pt, threshold)),
            "selected_depth": depth,
            "tree_count": model.tree_count_,
            "seconds_including_selection": time.monotonic() - start,
        }
        classification.append(row)
        selections.append(
            {
                "task": "classification",
                "seed": seed,
                "candidates": [
                    {"depth": d, "validation_nll": v, "trees": m.tree_count_}
                    for v, m, d in candidates
                ],
            }
        )
        write_json(
            run.output / f"classification_split_{seed}.json",
            {
                "fit": fit,
                "validation": val,
                "probability_calibration": prob,
                "conformal_calibration": conf,
            },
        )
        np.savez_compressed(run.output / f"adult_predictions_{seed}.npz", y=yt, p=pt)
        model.save_model(str(checkpoints / f"adult_{seed}.cbm"))
        write_json(
            run.output / f"adult_inference_{seed}.json",
            {
                "temperature": temperature,
                "threshold": threshold,
                "features": list(x.columns),
                "categorical": CATEGORICAL,
                "checkpoint_sha256": hashlib.sha256(
                    (checkpoints / f"adult_{seed}.cbm").read_bytes()
                ).hexdigest(),
                "alpha": 0.1,
                "conformal_score_cutoff": conformal_quantile(
                    1 - np.stack([1 - pc, pc], 1)[np.arange(len(conf)), y[conf]]
                ),
            },
        )
        original = Path(config["baseline_root"]) / "classification" / f"predictions_{seed}.npz"
        if original.exists():
            baseline = np.load(original, allow_pickle=False)
            assert np.array_equal(yt, baseline["y"])

            def loss(p):
                return -(
                    yt * np.log(np.clip(p, 1e-7, 1 - 1e-7))
                    + (1 - yt) * np.log(np.clip(1 - p, 1e-7, 1 - 1e-7))
                )

            comparisons.append(
                {
                    "task": "classification",
                    "seed": seed,
                    "comparison": "native CatBoost minus histogram boosting NLL",
                    **paired_bootstrap(loss(pt), loss(baseline["hist_boosting"]), seed=seed),
                }
            )
        print("Adult", seed, row["roc_auc"], row["log_loss"], flush=True)
    x, y = housing(config["data_root"])
    for protocol in ["random", "geographic"]:
        for seed in config["seeds"]:
            start = time.monotonic()
            indices = np.arange(len(y))
            if protocol == "random":
                train, test = train_test_split(indices, test_size=0.2, random_state=seed)
            else:
                cutoff = x.Longitude.quantile(0.8)
                train, test = indices[x.Longitude < cutoff], indices[x.Longitude >= cutoff]
            a, b, c, d = _split(np.arange(len(train)), None, seed)
            fit, val, cal = train[a], train[b], train[np.concatenate([c, d])]
            candidates = []
            for depth in config["depths"]:
                model = CatBoostRegressor(
                    iterations=config["iterations"],
                    depth=depth,
                    learning_rate=0.05,
                    loss_function="RMSE",
                    random_seed=seed,
                    thread_count=4,
                    allow_writing_files=False,
                    verbose=False,
                )
                model.fit(
                    x.iloc[fit], y[fit], eval_set=(x.iloc[val], y[val]), early_stopping_rounds=100
                )
                candidates.append(
                    (mean_squared_error(y[val], model.predict(x.iloc[val])), model, depth)
                )
            _, model, depth = min(candidates, key=lambda v: v[0])
            pc, pt = [model.predict(x.iloc[i]) for i in [cal, test]]
            lo, hi = regression_interval(pt, pc, y[cal])
            quantiles = CatBoostRegressor(
                iterations=config["iterations"],
                depth=depth,
                learning_rate=0.05,
                loss_function="MultiQuantile:alpha=0.05,0.95",
                random_seed=seed,
                thread_count=4,
                allow_writing_files=False,
                verbose=False,
            )
            quantiles.fit(
                x.iloc[fit], y[fit], eval_set=(x.iloc[val], y[val]), early_stopping_rounds=100
            )
            qc, qt = [quantiles.predict(x.iloc[i]) for i in [cal, test]]
            qlo, qhi = cqr_interval(qt[:, 0], qt[:, 1], qc[:, 0], qc[:, 1], y[cal])
            row = {
                "model": "catboost_native",
                "protocol": protocol,
                "seed": seed,
                "rmse": float(np.sqrt(mean_squared_error(y[test], pt))),
                "mae": float(mean_absolute_error(y[test], pt)),
                "split_conformal": _interval_metrics(y[test], lo, hi),
                "cqr": _interval_metrics(y[test], qlo, qhi),
                "selected_depth": depth,
                "tree_count": model.tree_count_,
                "seconds_including_selection": time.monotonic() - start,
            }
            regression.append(row)
            selections.append(
                {
                    "task": "regression",
                    "protocol": protocol,
                    "seed": seed,
                    "candidates": [
                        {"depth": d, "validation_rmse": float(np.sqrt(v)), "trees": m.tree_count_}
                        for v, m, d in candidates
                    ],
                }
            )
            write_json(
                run.output / f"regression_split_{protocol}_{seed}.json",
                {"fit": fit, "validation": val, "calibration": cal, "test": test},
            )
            np.savez_compressed(
                run.output / f"housing_predictions_{protocol}_{seed}.npz",
                y=y[test],
                p=pt,
                lo=lo,
                hi=hi,
                cqr_lo=qlo,
                cqr_hi=qhi,
            )
            model.save_model(str(checkpoints / f"housing_{protocol}_{seed}.cbm"))
            quantiles.save_model(str(checkpoints / f"housing_quantile_{protocol}_{seed}.cbm"))
            write_json(
                run.output / f"housing_inference_{protocol}_{seed}.json",
                {
                    "features": list(x.columns),
                    "alpha": 0.1,
                    "absolute_residual_cutoff": conformal_quantile(abs(y[cal] - pc)),
                    "cqr_cutoff": conformal_quantile(
                        np.maximum(np.min(qc, axis=1) - y[cal], y[cal] - np.max(qc, axis=1))
                    ),
                    "target_units": "$100,000",
                    "protocol": protocol,
                    "checkpoint_sha256": hashlib.sha256(
                        (checkpoints / f"housing_{protocol}_{seed}.cbm").read_bytes()
                    ).hexdigest(),
                    "quantile_checkpoint_sha256": hashlib.sha256(
                        (checkpoints / f"housing_quantile_{protocol}_{seed}.cbm").read_bytes()
                    ).hexdigest(),
                },
            )
            print(
                "Housing",
                protocol,
                seed,
                row["rmse"],
                row["split_conformal"]["coverage"],
                flush=True,
            )
    run.finish(
        {
            "classification": classification,
            "regression": regression,
            "selection": selections,
            "paired": comparisons,
            "protocol": "Existing official Adult test and exact 70/10/10/10 training splits; native missing/category handling. Identical Housing random/geographic holdouts and fit/selection/calibration partitions. Depth and early stopping validation-only; temperature fit on separate probability partition; conformal cutoff on separate calibration. CPU CatBoost, no external pretraining or test tuning. Frozen historical predictions provide paired Adult NLL comparisons.",
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    experiment(json.loads(parser.parse_args().config.read_text()))
