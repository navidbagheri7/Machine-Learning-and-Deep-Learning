"""Classical prediction families with isolated selection/calibration and nested stacking.

The extension uses existing Adult/Housing splits and bundled numeric datasets.
Every ensemble base learner owns its preprocessing, including within stacking CV.
Validation chooses a declared preset; subsequent probability and conformal partitions
have distinct roles. Test data never enter fitting or model selection.
"""

import hashlib
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from .artifacts import Run, fingerprint, write_json
from .data import adult, housing
from .tabular import _adult_preprocessor, _split
from .uncertainty import fit_temperature, prediction_sets, regression_interval


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def take(x, indices):
    return x.iloc[indices] if hasattr(x, "iloc") else x[indices]


def preprocessing(dataset):
    if dataset == "adult":
        return _adult_preprocessor()
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


def classifier_candidates(dataset, seed, config):
    from catboost import CatBoostClassifier

    trees = config["trees"]

    def pipe(model):
        return make_pipeline(preprocessing(dataset), model)

    models = {
        "logistic": [LogisticRegression(C=c, max_iter=5000) for c in [0.1, 1, 10]],
        "lda": [LinearDiscriminantAnalysis(solver="lsqr", shrinkage=a) for a in ["auto", 0.1, 0.5]],
        "qda": [QuadraticDiscriminantAnalysis(reg_param=a) for a in [0.05, 0.2, 0.5]],
        "naive_bayes": [GaussianNB(var_smoothing=a) for a in [1e-9, 1e-5, 1e-2]],
        "knn": [
            KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=1) for k in [5, 15, 35]
        ],
        "tree": [
            DecisionTreeClassifier(max_depth=d, min_samples_leaf=5, random_state=seed)
            for d in [4, 8, None]
        ],
        "bagging": [
            BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=d, min_samples_leaf=5),
                n_estimators=trees,
                max_samples=0.8,
                random_state=seed,
                n_jobs=1,
            )
            for d in [5, 10, None]
        ],
        "random_forest": [
            RandomForestClassifier(
                n_estimators=trees, max_depth=d, min_samples_leaf=2, random_state=seed, n_jobs=1
            )
            for d in [8, 16, None]
        ],
        "extra_trees": [
            ExtraTreesClassifier(
                n_estimators=trees, max_depth=d, min_samples_leaf=2, random_state=seed, n_jobs=1
            )
            for d in [8, 16, None]
        ],
        "adaboost": [
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2),
                n_estimators=trees,
                learning_rate=a,
                algorithm="SAMME",
                random_state=seed,
            )
            for a in [0.1, 0.5, 1.0]
        ],
        "hist_boosting": [
            HistGradientBoostingClassifier(
                max_iter=250,
                max_leaf_nodes=n,
                learning_rate=0.08,
                l2_regularization=2,
                random_state=seed,
            )
            for n in [15, 31, 63]
        ],
        "catboost": [
            CatBoostClassifier(
                iterations=500,
                depth=d,
                learning_rate=0.05,
                verbose=False,
                allow_writing_files=False,
                random_seed=seed,
                thread_count=1,
            )
            for d in [4, 6, 8]
        ],
        "linear_svm": [
            LinearSVC(C=c, dual="auto", max_iter=20000, random_state=seed) for c in [0.1, 1, 10]
        ],
        "rbf_svm": [
            SVC(C=c, gamma="scale", cache_size=512, decision_function_shape="ovr")
            for c in [0.1, 1, 10]
        ],
    }
    models = {name: [pipe(model) for model in candidates] for name, candidates in models.items()}
    bases = [
        ("linear", pipe(LogisticRegression(C=1, max_iter=5000))),
        (
            "forest",
            pipe(
                RandomForestClassifier(
                    n_estimators=trees,
                    max_depth=16,
                    min_samples_leaf=2,
                    random_state=seed,
                    n_jobs=1,
                )
            ),
        ),
        (
            "boosting",
            pipe(
                HistGradientBoostingClassifier(max_iter=150, l2_regularization=2, random_state=seed)
            ),
        ),
    ]
    models["voting"] = [
        VotingClassifier(estimators=bases, voting="soft", weights=w, n_jobs=1)
        for w in [[1, 1, 1], [1, 1, 2]]
    ]
    models["stacking"] = [
        StackingClassifier(
            estimators=bases,
            final_estimator=LogisticRegression(C=c, max_iter=5000),
            cv=StratifiedKFold(config["stacking_folds"], shuffle=True, random_state=seed),
            n_jobs=1,
        )
        for c in [0.1, 1]
    ]
    if dataset == "adult":
        models = {k: v for k, v in models.items() if k not in config["adult_excluded_methods"]}
    return models


def regressor_candidates(seed, config):
    from catboost import CatBoostRegressor

    trees = config["trees"]

    def pipe(model):
        return make_pipeline(preprocessing("housing"), model)

    models = {
        "ols": [LinearRegression()],
        "ridge": [Ridge(alpha=a) for a in [0.1, 10, 100]],
        "lasso": [Lasso(alpha=a, max_iter=20000) for a in [0.001, 0.01, 0.1]],
        "elastic_net": [
            ElasticNet(alpha=a, l1_ratio=0.5, max_iter=20000) for a in [0.001, 0.01, 0.1]
        ],
        "bayesian_ridge": [BayesianRidge()],
        "huber": [HuberRegressor(epsilon=e, max_iter=2000) for e in [1.2, 1.5, 2]],
        "knn": [
            KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=1) for k in [5, 15, 35]
        ],
        "tree": [
            DecisionTreeRegressor(max_depth=d, min_samples_leaf=5, random_state=seed)
            for d in [4, 8, None]
        ],
        "bagging": [
            BaggingRegressor(
                estimator=DecisionTreeRegressor(max_depth=d, min_samples_leaf=5),
                n_estimators=trees,
                max_samples=0.8,
                random_state=seed,
                n_jobs=1,
            )
            for d in [5, 10, None]
        ],
        "random_forest": [
            RandomForestRegressor(
                n_estimators=trees, max_depth=d, min_samples_leaf=2, random_state=seed, n_jobs=1
            )
            for d in [8, 16, None]
        ],
        "extra_trees": [
            ExtraTreesRegressor(
                n_estimators=trees, max_depth=d, min_samples_leaf=2, random_state=seed, n_jobs=1
            )
            for d in [8, 16, None]
        ],
        "adaboost": [
            AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=4),
                n_estimators=trees,
                learning_rate=a,
                random_state=seed,
            )
            for a in [0.1, 0.5, 1]
        ],
        "hist_boosting": [
            HistGradientBoostingRegressor(
                max_iter=250,
                max_leaf_nodes=n,
                learning_rate=0.08,
                l2_regularization=2,
                random_state=seed,
            )
            for n in [15, 31, 63]
        ],
        "catboost": [
            CatBoostRegressor(
                iterations=500,
                depth=d,
                learning_rate=0.05,
                verbose=False,
                allow_writing_files=False,
                random_seed=seed,
                thread_count=1,
            )
            for d in [4, 6, 8]
        ],
        "rbf_svr": [SVR(C=c, epsilon=0.1, gamma="scale", cache_size=512) for c in [1, 10, 100]],
    }
    models = {name: [pipe(model) for model in candidates] for name, candidates in models.items()}
    bases = [
        ("linear", pipe(Ridge(alpha=10))),
        (
            "forest",
            pipe(
                RandomForestRegressor(
                    n_estimators=trees,
                    max_depth=16,
                    min_samples_leaf=2,
                    random_state=seed,
                    n_jobs=1,
                )
            ),
        ),
        (
            "boosting",
            pipe(
                HistGradientBoostingRegressor(max_iter=150, l2_regularization=2, random_state=seed)
            ),
        ),
    ]
    models["voting"] = [
        VotingRegressor(estimators=bases, weights=w, n_jobs=1) for w in [[1, 1, 1], [1, 1, 2]]
    ]
    models["stacking"] = [
        StackingRegressor(
            estimators=bases,
            final_estimator=Ridge(alpha=a),
            cv=KFold(config["stacking_folds"], shuffle=True, random_state=seed),
            n_jobs=1,
        )
        for a in [1, 10]
    ]
    return {k: v for k, v in models.items() if k not in config["housing_excluded_methods"]}


def logits(model, x):
    if hasattr(model, "predict_proba"):
        return np.log(np.clip(model.predict_proba(x), 1e-12, 1))
    scores = model.decision_function(x)
    return np.column_stack([np.zeros(len(scores)), scores]) if scores.ndim == 1 else scores


def classify_metrics(y, probabilities):
    k = probabilities.shape[1]
    predicted = probabilities.argmax(1)
    auc = (
        roc_auc_score(y, probabilities[:, 1])
        if k == 2
        else roc_auc_score(y, probabilities, multi_class="ovr", average="macro")
    )
    return dict(
        accuracy=float(accuracy_score(y, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
        macro_f1=float(f1_score(y, predicted, average="macro", zero_division=0)),
        log_loss=float(log_loss(y, probabilities, labels=np.arange(k))),
        roc_auc=float(auc),
        brier_multiclass=float(np.square(probabilities - np.eye(k)[y]).sum(1).mean()),
        confusion=confusion_matrix(y, predicted, labels=np.arange(k)).tolist(),
    )


def dataset_split(dataset, config):
    if dataset == "adult":
        (x, y), (xt, yt) = adult(config["data_root"])
        return x, y, xt, yt, np.arange(len(y)), np.arange(len(yt))
    data = load_digits() if dataset == "digits" else load_breast_cancer()
    fit, test = train_test_split(
        np.arange(len(data.target)),
        test_size=0.2,
        stratify=data.target,
        random_state=config["test_seed"],
    )
    return data.data[fit], data.target[fit], data.data[test], data.target[test], fit, test


def source_bound(run):
    run.record["fitting_source_sha256"] = {
        name: digest(Path(__file__).parent / name)
        for name in ["classical.py", "tabular.py", "data.py", "uncertainty.py", "artifacts.py"]
    }
    write_json(run.output / "run.json", run.record | {"status": "running"})


def fit_classification(config, dataset, seed):
    directory = Path(config["output"]) / dataset / str(seed)
    if (directory / "run.json").exists():
        existing = json.loads((directory / "run.json").read_text())
        if existing["status"] == "complete" and existing["config"] == dict(
            config=config, dataset=dataset, seed=seed
        ):
            for name, value in existing["fitting_source_sha256"].items():
                assert digest(Path(__file__).parent / name) == value
            print("Preserved complete fit", dataset, seed, flush=True)
            return
        raise FileExistsError(f"Preserve incomplete/different protocol at {directory}")
    run = Run(directory, dict(config=config, dataset=dataset, seed=seed))
    source_bound(run)
    x, y, xt, yt, train_ids, test_ids = dataset_split(dataset, config)
    if config.get("quick"):
        ix, _ = train_test_split(
            np.arange(len(y)), train_size=min(300, len(y) - 1), stratify=y, random_state=seed
        )
        x, y, train_ids = take(x, ix), y[ix], train_ids[ix]
        xt, yt, test_ids = take(xt, np.arange(min(100, len(yt)))), yt[:100], test_ids[:100]
    fit, val, prob, conf = _split(np.arange(len(y)), y, seed)
    x_digest = (
        hashlib.sha256(x.to_csv(index=False).encode()).hexdigest()
        if hasattr(x, "to_csv")
        else fingerprint(x)
    )
    write_json(
        directory / "split.json",
        dict(
            fit=fit,
            validation=val,
            probability_calibration=prob,
            conformal_calibration=conf,
            train_ids=train_ids,
            test_ids=test_ids,
            x_fingerprint=x_digest,
            y_fingerprint=fingerprint(y),
            test_y_fingerprint=fingerprint(yt),
        ),
    )
    families = classifier_candidates(dataset, seed, config)
    rows = []
    public = Path(
        "assets/source_data/classical"
        if not config.get("quick")
        else "results/smoke/classical_evidence"
    )
    public.mkdir(parents=True, exist_ok=True)
    all_predictions = dict(y=yt, test_ids=test_ids)
    for name, candidates in families.items():
        started = time.monotonic()
        selected, best, search = None, -np.inf, []
        for index, candidate in enumerate(candidates):
            t = time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                candidate.fit(take(x, fit), y[fit])
            score = f1_score(
                y[val], candidate.predict(take(x, val)), average="macro", zero_division=0
            )
            search.append(
                dict(
                    candidate=index,
                    validation_macro_f1=float(score),
                    seconds=time.monotonic() - t,
                    warnings=sorted({str(w.message) for w in caught}),
                    parameters={
                        key: str(value)
                        for key, value in candidate.get_params().items()
                        if "__" in key or key in ["weights", "cv"]
                    },
                )
            )
            if score > best:
                best, selected = score, candidate
        temperature = fit_temperature(logits(selected, take(x, prob)), y[prob])
        calibration = softmax(logits(selected, take(x, conf)) / temperature, axis=1)
        predict_started = time.monotonic()
        prediction = softmax(logits(selected, xt) / temperature, axis=1)
        predict_seconds = time.monotonic() - predict_started
        sets = prediction_sets(prediction, calibration, y[conf], config["conformal_alpha"])
        checkpoint = directory / "checkpoints" / f"{name}.joblib"
        checkpoint.parent.mkdir(exist_ok=True)
        joblib.dump(dict(model=selected, temperature=temperature, dataset=dataset), checkpoint)
        restored = joblib.load(checkpoint)
        np.testing.assert_array_equal(
            logits(restored["model"], take(xt, np.arange(min(32, len(yt))))),
            logits(selected, take(xt, np.arange(min(32, len(yt))))),
        )
        all_predictions[name] = prediction
        all_predictions[f"{name}_sets"] = sets
        rows.append(
            dict(
                model=name,
                seed=seed,
                dataset=dataset,
                **classify_metrics(yt, prediction),
                set_coverage=float(sets[np.arange(len(yt)), yt].mean()),
                set_size=float(sets.sum(1).mean()),
                temperature=temperature,
                selected_candidate=int(np.argmax([r["validation_macro_f1"] for r in search])),
                validation_macro_f1=float(best),
                candidate_count=len(search),
                search=search,
                seconds_including_selection=time.monotonic() - started,
                test_prediction_seconds=predict_seconds,
                checkpoint_sha256=digest(checkpoint),
            )
        )
        write_json(directory / "partial_metrics.json", dict(rows=rows))
        print(
            dataset,
            seed,
            name,
            "accuracy",
            round(rows[-1]["accuracy"], 4),
            "NLL",
            round(rows[-1]["log_loss"], 4),
            flush=True,
        )
    evidence = public / f"{dataset}_{seed}.npz"
    np.savez_compressed(evidence, **all_predictions)
    run.finish(
        dict(
            rows=rows,
            evidence=str(evidence),
            evidence_sha256=digest(evidence),
            fit_examples=len(fit),
            test_examples=len(yt),
            deployment_roundtrip="Exact raw scores on first32 test examples, every selected model",
            protocol="Shared fixed test; isolated fit/selection/probability/conformal roles. Ensemble base preprocessing refitted inside stacking folds. Macro-F1 selection; separate temperature calibration; no test selection.",
        )
    )


def fit_regression(config, protocol, seed):
    directory = Path(config["output"]) / f"housing_{protocol}" / str(seed)
    if (directory / "run.json").exists():
        existing = json.loads((directory / "run.json").read_text())
        if existing["status"] == "complete" and existing["config"] == dict(
            config=config, protocol=protocol, seed=seed
        ):
            for name, value in existing["fitting_source_sha256"].items():
                assert digest(Path(__file__).parent / name) == value
            print("Preserved complete fit", protocol, seed, flush=True)
            return
        raise FileExistsError(f"Preserve incomplete/different protocol at {directory}")
    run = Run(directory, dict(config=config, protocol=protocol, seed=seed))
    source_bound(run)
    x, y = housing(config["data_root"])
    if config.get("quick"):
        x, y = x.iloc[:500].reset_index(drop=True), y[:500]
    ix = np.arange(len(y))
    if protocol == "random":
        train, test = train_test_split(ix, test_size=0.2, random_state=seed)
    else:
        cutoff = x.Longitude.quantile(0.8)
        train, test = ix[x.Longitude < cutoff], ix[x.Longitude >= cutoff]
    f, v, a, b = _split(np.arange(len(train)), None, seed)
    fit, val, cal = train[f], train[v], train[np.concatenate([a, b])]
    write_json(
        directory / "split.json",
        dict(
            fit=fit,
            validation=val,
            calibration=cal,
            test=test,
            x_fingerprint=fingerprint(x.to_numpy()),
            y_fingerprint=fingerprint(y),
        ),
    )
    rows = []
    public = Path(
        "assets/source_data/classical"
        if not config.get("quick")
        else "results/smoke/classical_evidence"
    )
    public.mkdir(parents=True, exist_ok=True)
    predictions = dict(y=y[test], test_ids=test, longitude=x.Longitude.iloc[test].to_numpy())
    for name, candidates in regressor_candidates(seed, config).items():
        started = time.monotonic()
        selected, best, search = None, np.inf, []
        for index, candidate in enumerate(candidates):
            t = time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                candidate.fit(x.iloc[fit], y[fit])
            score = float(np.sqrt(mean_squared_error(y[val], candidate.predict(x.iloc[val]))))
            search.append(
                dict(
                    candidate=index,
                    validation_rmse=score,
                    seconds=time.monotonic() - t,
                    warnings=sorted({str(w.message) for w in caught}),
                    parameters={
                        key: str(value)
                        for key, value in candidate.get_params().items()
                        if "__" in key or key in ["weights", "cv"]
                    },
                )
            )
            if score < best:
                best, selected = score, candidate
        predict_started = time.monotonic()
        pred = selected.predict(x.iloc[test])
        prediction_seconds = time.monotonic() - predict_started
        lower, upper = regression_interval(
            pred, selected.predict(x.iloc[cal]), y[cal], config["conformal_alpha"]
        )
        checkpoint = directory / "checkpoints" / f"{name}.joblib"
        checkpoint.parent.mkdir(exist_ok=True)
        joblib.dump(selected, checkpoint)
        np.testing.assert_array_equal(
            joblib.load(checkpoint).predict(x.iloc[test[:32]]), selected.predict(x.iloc[test[:32]])
        )
        predictions[name], predictions[f"{name}_lower"], predictions[f"{name}_upper"] = (
            pred,
            lower,
            upper,
        )
        rows.append(
            dict(
                model=name,
                seed=seed,
                dataset="housing",
                protocol=protocol,
                rmse=float(np.sqrt(mean_squared_error(y[test], pred))),
                mae=float(mean_absolute_error(y[test], pred)),
                r2=float(r2_score(y[test], pred)),
                interval_coverage=float(((y[test] >= lower) & (y[test] <= upper)).mean()),
                interval_width=float((upper - lower).mean()),
                validation_rmse=best,
                selected_candidate=int(np.argmin([r["validation_rmse"] for r in search])),
                candidate_count=len(search),
                search=search,
                seconds_including_selection=time.monotonic() - started,
                test_prediction_seconds=prediction_seconds,
                checkpoint_sha256=digest(checkpoint),
            )
        )
        write_json(directory / "partial_metrics.json", dict(rows=rows))
        print("housing", protocol, seed, name, "RMSE", round(rows[-1]["rmse"], 4), flush=True)
    evidence = public / f"housing_{protocol}_{seed}.npz"
    np.savez_compressed(evidence, **predictions)
    run.finish(
        dict(
            rows=rows,
            evidence=str(evidence),
            evidence_sha256=digest(evidence),
            fit_examples=len(fit),
            test_examples=len(test),
            deployment_roundtrip="Exact predictions on first32 test examples, every selected model",
            protocol="Shared original Housing random/geographic roles; independent regression calibration. Stack base preprocessing fitted within training folds. RMSE selection on validation only. Intervals need exchangeability; geographic failures retained.",
        )
    )


def experiment(config):
    with threadpool_limits(config["threads"]):
        for seed in config["seeds"]:
            for dataset in config["classification_datasets"]:
                fit_classification(config, dataset, seed)
            for protocol in config["regression_protocols"]:
                fit_regression(config, protocol, seed)
