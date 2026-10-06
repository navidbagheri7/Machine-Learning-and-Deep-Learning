"""Adult decision study and California Housing uncertainty under geographic shift."""

import time

import joblib
import numpy as np
from scipy.special import logit
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import BayesianRidge, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from learning_lab.artifacts import Run, fingerprint, write_json
from learning_lab.data import CATEGORICAL, NUMERIC, adult, housing
from learning_lab.uncertainty import (
    binary_metrics,
    binary_temperature,
    cqr_interval,
    fit_temperature,
    paired_bootstrap,
    prediction_sets,
    regression_interval,
    reliability,
)


def _split(indices, y, seed):
    fit, reserve = train_test_split(indices, test_size=0.3, random_state=seed, stratify=y)
    val, cal = train_test_split(
        reserve, test_size=2 / 3, random_state=seed, stratify=y[reserve] if y is not None else None
    )
    prob, conf = train_test_split(
        cal, test_size=0.5, random_state=seed, stratify=y[cal] if y is not None else None
    )
    return fit, val, prob, conf


def _adult_preprocessor():
    return ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
            (
                "category",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )


def classification(config):
    run = Run(config["output"], config)
    (x, y), (xt, yt) = adult(config["data_root"])
    if config.get("quick"):
        ix, _ = train_test_split(np.arange(len(y)), train_size=2500, stratify=y, random_state=0)
        x, y = x.iloc[ix].reset_index(drop=True), y[ix]
        xt, yt = xt.iloc[:1500], yt[:1500]
    all_rows, diagnostics, paired = [], {}, []
    for seed in config["seeds"]:
        fit, val, prob, conf = _split(np.arange(len(y)), y, seed)
        prep = _adult_preprocessor().fit(x.iloc[fit])
        arrays = [
            prep.transform(z).astype(np.float32)
            for z in [x.iloc[fit], x.iloc[val], x.iloc[prob], x.iloc[conf], xt]
        ]
        xf, xv, xp, xc, test = arrays
        models = {
            "logistic": [LogisticRegression(C=c, max_iter=1000) for c in [0.1, 1, 10]],
            "hist_boosting": [
                HistGradientBoostingClassifier(
                    max_iter=250,
                    learning_rate=0.08,
                    max_leaf_nodes=leaves,
                    l2_regularization=2,
                    random_state=seed,
                )
                for leaves in [15, 31, 63]
            ],
        }
        predictions, raw_predictions = {}, {}
        for name in [*models, "tabular_resnet"]:
            start = time.monotonic()
            if name == "tabular_resnet":
                from learning_lab.neural import neural_predict, train_tabular

                model, history = train_tabular(
                    xf,
                    y[fit],
                    xv,
                    y[val],
                    seed=seed,
                    epochs=config["epochs"],
                    device=config["device"],
                )
                write_json(run.output / f"history_{name}_{seed}.json", history)

                def predict(a):
                    return neural_predict(model, a, config["device"], True)

                selected = {
                    "best_epoch": int(np.argmin([h["validation_loss"] for h in history]) + 1)
                }
            else:
                candidates = []
                for candidate in models[name]:
                    candidate.fit(xf, y[fit])
                    score = binary_metrics(y[val], candidate.predict_proba(xv)[:, 1])["log_loss"]
                    candidates.append((score, candidate))
                _, model = min(candidates, key=lambda z: z[0])

                def predict(a):
                    return model.predict_proba(a)[:, 1]

                selected = model.get_params()
            pprob, pconf, ptest, pval = [predict(a) for a in [xp, xc, test, xv]]
            temperature = fit_temperature(
                np.column_stack([np.zeros(len(pprob)), logit(np.clip(pprob, 1e-7, 1 - 1e-7))]),
                y[prob],
            )
            calibrated = [binary_temperature(p, temperature) for p in [pconf, ptest, pval]]
            pc, pt, pv = calibrated
            thresholds = np.linspace(0.05, 0.95, 91)
            # Illustrative utility: a false negative costs twice a false positive.
            costs = [
                np.mean(2 * ((pv < t) & (y[val] == 1)) + ((pv >= t) & (y[val] == 0)))
                for t in thresholds
            ]
            threshold = float(thresholds[np.argmin(costs)])
            sets = prediction_sets(
                np.column_stack([1 - pt, pt]), np.column_stack([1 - pc, pc]), y[conf]
            )
            row = {
                "model": name,
                "seed": seed,
                **binary_metrics(yt, pt),
                "raw_log_loss": binary_metrics(yt, ptest)["log_loss"],
                "temperature": temperature,
                "decision_threshold": threshold,
                "decision_cost": float(
                    np.mean(2 * ((pt < threshold) & (yt == 1)) + ((pt >= threshold) & (yt == 0)))
                ),
                "set_coverage": float(sets[np.arange(len(yt)), yt].mean()),
                "set_size": float(sets.sum(axis=1).mean()),
                "seconds_including_selection": time.monotonic() - start,
                "selected": selected,
            }
            all_rows.append(row)
            print(
                name,
                seed,
                "AUC",
                round(row["roc_auc"], 4),
                "NLL",
                round(row["log_loss"], 4),
                flush=True,
            )
            predictions[name], raw_predictions[name] = pt, ptest
            if seed == config["seeds"][0]:
                groups = []
                for feature in ["sex", "race"]:
                    for group in sorted(xt[feature].dropna().unique()):
                        mask = xt[feature].eq(group).to_numpy()
                        if mask.sum() < 100 or len(np.unique(yt[mask])) < 2:
                            continue
                        groups.append(
                            {
                                "feature": feature,
                                "group": group,
                                "n": int(mask.sum()),
                                **binary_metrics(yt[mask], pt[mask]),
                                "set_coverage": float(sets[mask, yt[mask]].mean()),
                                "tpr": float(np.mean(pt[mask & (yt == 1)] >= threshold)),
                                "fpr": float(np.mean(pt[mask & (yt == 0)] >= threshold)),
                            }
                        )
                diagnostics[name] = {"groups": groups, "reliability": reliability(yt, pt)}
                if name != "tabular_resnet":
                    pipeline = make_pipeline(prep, model)
                    artifact = {
                        "pipeline": pipeline,
                        "temperature": temperature,
                        "threshold": threshold,
                        "features": list(x.columns),
                        "model": name,
                    }
                    joblib.dump(artifact, run.output / f"{name}.joblib")
                    perm = permutation_importance(
                        pipeline,
                        xt.iloc[:2500],
                        yt[:2500],
                        n_repeats=3,
                        random_state=seed,
                        scoring="neg_log_loss",
                        n_jobs=1,
                    )
                    diagnostics[name]["permutation_importance"] = dict(
                        zip(x.columns, perm.importances_mean.tolist())
                    )
        np.savez_compressed(run.output / f"predictions_{seed}.npz", y=yt, **predictions)
        if seed == config["seeds"][0]:

            def losses(p):
                p = np.clip(p, 1e-7, 1 - 1e-7)
                return -(yt * np.log(p) + (1 - yt) * np.log(1 - p))

            for name in ["hist_boosting", "tabular_resnet"]:
                paired.append(
                    {
                        "comparison": name + " minus logistic NLL",
                        **paired_bootstrap(
                            losses(predictions[name]), losses(predictions["logistic"])
                        ),
                    }
                )
        write_json(
            run.output / f"split_{seed}.json",
            {
                "fit": fit,
                "validation": val,
                "probability_calibration": prob,
                "conformal_calibration": conf,
                "test": "official adult.test; never used for selection",
            },
        )
    run.finish(
        {
            "rows": all_rows,
            "diagnostics_seed0": diagnostics,
            "paired_seed0": paired,
            "data": {
                "train": len(y),
                "test": len(yt),
                "target": "income >50K",
                "test_y_hash": fingerprint(yt),
            },
            "protocol": "70/10/10/10 training split, plus official test; no survey weighting",
        }
    )


def _interval_metrics(y, lo, hi):
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width": float(np.mean(np.maximum(hi - lo, 0))),
        "empty_rate": float(np.mean(hi < lo)),
    }


def regression(config):
    run = Run(config["output"], config)
    x, y = housing(config["data_root"])
    if config.get("quick"):
        x, y = x.iloc[:2500].reset_index(drop=True), y[:2500]
    rows, diagnostics = [], {}
    for protocol in ["random", "geographic"]:
        for seed in config["seeds"]:
            ix = np.arange(len(y))
            if protocol == "random":
                train, test = train_test_split(ix, test_size=0.2, random_state=seed)
            else:
                # Hold out the easternmost 20%; labels do not define this partition.
                cutoff = x.Longitude.quantile(0.8)
                test, train = ix[x.Longitude >= cutoff], ix[x.Longitude < cutoff]
            fit, val, _, cal = _split(np.arange(len(train)), None, seed)
            # Merge the two calibration halves for regression (no temperature fitting).
            _, _, cal_a, cal_b = _split(np.arange(len(train)), None, seed)
            fit, val, cal = train[fit], train[val], train[np.concatenate([cal_a, cal_b])]
            scaler = StandardScaler().fit(x.iloc[fit])
            xf, xv, xc, xt = [
                scaler.transform(x.iloc[j]).astype(np.float32) for j in [fit, val, cal, test]
            ]
            for name in ["ridge", "bayesian_ridge", "hist_boosting", "feature_transformer"]:
                start = time.monotonic()
                if name == "feature_transformer":
                    from learning_lab.neural import neural_predict, train_tabular

                    mean, scale = float(y[fit].mean()), float(y[fit].std())
                    model, history = train_tabular(
                        xf,
                        (y[fit] - mean) / scale,
                        xv,
                        (y[val] - mean) / scale,
                        kind="transformer",
                        task="regression",
                        seed=seed,
                        epochs=config["epochs"],
                        device=config["device"],
                    )
                    write_json(run.output / f"history_{protocol}_{seed}.json", history)
                    pc, pt = [
                        neural_predict(model, z, config["device"]) * scale + mean for z in [xc, xt]
                    ]
                else:
                    candidates = {
                        "ridge": [Ridge(alpha=a) for a in [0.1, 10, 100]],
                        "bayesian_ridge": [BayesianRidge()],
                        "hist_boosting": [
                            HistGradientBoostingRegressor(
                                max_iter=300,
                                max_leaf_nodes=leaves,
                                learning_rate=0.07,
                                l2_regularization=2,
                                random_state=seed,
                            )
                            for leaves in [15, 31, 63]
                        ],
                    }[name]
                    scores = []
                    for m in candidates:
                        m.fit(xf, y[fit])
                        scores.append(mean_squared_error(y[val], m.predict(xv)))
                    model = candidates[int(np.argmin(scores))]
                    pc, pt = model.predict(xc), model.predict(xt)
                lo, hi = regression_interval(pt, pc, y[cal])
                row = {
                    "protocol": protocol,
                    "model": name,
                    "seed": seed,
                    "rmse": float(np.sqrt(mean_squared_error(y[test], pt))),
                    "mae": mean_absolute_error(y[test], pt),
                    "split_conformal": _interval_metrics(y[test], lo, hi),
                    "seconds": time.monotonic() - start,
                }
                if name == "bayesian_ridge":
                    _, sd = model.predict(xt, return_std=True)
                    row["gaussian_posterior"] = _interval_metrics(
                        y[test], pt - 1.644854 * sd, pt + 1.644854 * sd
                    )
                if name == "hist_boosting":
                    lower = clone(model).set_params(loss="quantile", quantile=0.05).fit(xf, y[fit])
                    upper = clone(model).set_params(loss="quantile", quantile=0.95).fit(xf, y[fit])
                    qlo, qhi = cqr_interval(
                        lower.predict(xt),
                        upper.predict(xt),
                        lower.predict(xc),
                        upper.predict(xc),
                        y[cal],
                    )
                    row["cqr"] = _interval_metrics(y[test], qlo, qhi)
                    np.savez_compressed(
                        run.output / f"intervals_{protocol}_{seed}.npz",
                        y=y[test],
                        pred=pt,
                        lower=qlo,
                        upper=qhi,
                        longitude=x.Longitude.iloc[test].to_numpy(),
                    )
                    if seed == config["seeds"][0]:
                        # Diagnose capped targets and large-occupancy rows after freezing models.
                        diagnostics[protocol] = {
                            "test_n": len(test),
                            "fit_n": len(fit),
                            "capped_target_fraction": float(np.mean(y[test] >= 5)),
                            "capped_mae": (
                                float(np.mean(np.abs(y[test][y[test] >= 5] - pt[y[test] >= 5])))
                                if np.any(y[test] >= 5)
                                else None
                            ),
                        }
                rows.append(row)
                print(protocol, name, seed, "RMSE", round(row["rmse"], 4), flush=True)
            write_json(
                run.output / f"split_{protocol}_{seed}.json",
                {"fit": fit, "validation": val, "calibration": cal, "test": test},
            )
    run.finish(
        {
            "rows": rows,
            "diagnostics": diagnostics,
            "data": {
                "n": len(y),
                "features": list(x.columns),
                "target_units": "$100,000",
                "target_hash": fingerprint(y),
            },
            "alpha": 0.1,
            "geographic_guarantee": False,
        }
    )
