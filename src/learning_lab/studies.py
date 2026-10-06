"""Controlled generalization, kernel learning curves, and probabilistic clustering."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, load_wine
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from learning_lab.artifacts import Run
from learning_lab.probabilistic import BayesianLinearRegression, GaussianMixtureEM


def generalization(config):
    run = Run(config["output"], config)
    n = config["samples"]
    widths = sorted(
        set([int(n * r) for r in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.25, 1.5, 2, 3, 5]])
    )
    rows = []
    for seed in range(config["repeats"]):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, max(widths)))
        beta = np.zeros(max(widths))
        beta[:10] = 1 / np.sqrt(10)
        y = x @ beta + rng.normal(scale=0.5, size=n)
        for p in widths:
            a = x[:, :p]
            u, s, vt = np.linalg.svd(a, full_matrices=False)
            for penalty in [0, 1, 10]:
                factors = 1 / s if penalty == 0 else s / (s * s + penalty)
                w = vt.T @ (factors * (u.T @ y))
                full = np.zeros_like(beta)
                full[:p] = w
                rows.append(
                    {
                        "seed": seed,
                        "features": p,
                        "ratio": p / n,
                        "penalty": penalty,
                        "train_mse": float(np.mean((a @ w - y) ** 2)),
                        "population_excess_mse": float(np.sum((full - beta) ** 2)),
                        "population_mse": float(np.sum((full - beta) ** 2) + 0.25),
                        "minimum_singular_value": float(s.min()),
                    }
                )
    # Logistic GD on linearly separable data: compare direction with hard margin QP.
    rng = np.random.default_rng(19)
    x = rng.normal(size=(300, 2)) * [1, 2]
    y = np.where(x @ np.array([1.0, 0.4]) > 0, 1, -1)
    keep = np.abs(x @ np.array([1.0, 0.4])) > 0.4
    x, y = x[keep], y[keep]
    qp = minimize(
        lambda w: 0.5 * w @ w,
        np.ones(2),
        jac=lambda w: w,
        constraints={
            "type": "ineq",
            "fun": lambda w: y * (x @ w) - 1,
            "jac": lambda w: y[:, None] * x,
        },
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    if not qp.success:
        raise RuntimeError(qp.message)
    target = qp.x / np.linalg.norm(qp.x)
    w, implicit = np.zeros(2), []
    steps = set(np.unique(np.geomspace(1, config["gd_steps"], 100).astype(int)))
    for t in range(1, config["gd_steps"] + 1):
        w += 0.1 * x.T @ (y * expit(-y * (x @ w))) / len(y)
        if t in steps:
            implicit.append(
                {
                    "step": t,
                    "loss": float(np.logaddexp(0, -y * (x @ w)).mean()),
                    "norm": float(np.linalg.norm(w)),
                    "direction_cosine": float((w / np.linalg.norm(w)) @ target),
                    "normalized_margin": float(np.min(y * (x @ w)) / np.linalg.norm(w)),
                }
            )
    run.finish(
        {
            "double_descent": rows,
            "implicit_bias": implicit,
            "hard_margin_direction": target,
            "noise_variance": 0.25,
            "risk_protocol": "exact population risk for isotropic Gaussian covariates; fixed 10-feature signal",
            "penalty_convention": "||Xw-y||^2 + lambda ||w||^2",
        }
    )


def probabilistic(config):
    run = Run(config["output"], config)
    rng = np.random.default_rng(7)
    means = np.array([[-3, 0], [2, 2], [1, -3]])
    covs = np.array([[[2, 0.8], [0.8, 0.5]], [[0.4, -0.6], [-0.6, 2]], [[1.2, 0], [0, 0.25]]])
    labels = rng.choice(3, size=config["samples"], p=[0.4, 0.35, 0.25])
    x = np.array([rng.multivariate_normal(means[k], covs[k]) for k in labels])
    train, test = train_test_split(np.arange(len(x)), test_size=0.3, random_state=0)
    candidates = []
    for k in range(1, 7):
        model = GaussianMixture(k, n_init=5, random_state=0).fit(x[train])
        candidates.append(
            {"k": k, "bic": float(model.bic(x[train])), "heldout_nll": float(-model.score(x[test]))}
        )
    custom = GaussianMixtureEM(3, seed=0).fit(x[train])
    reference = GaussianMixture(3, n_init=5, random_state=0).fit(x[train])
    km = KMeans(3, n_init=10, random_state=0).fit(x[train])
    wine = load_wine()
    # Unsupervised geometry only: wine class labels are reserved for post-fit ARI.
    z = StandardScaler().fit_transform(wine.data)
    stability = []
    for seed in range(config["repeats"]):
        ix = np.random.default_rng(seed).choice(len(z), len(z), replace=True)
        a = KMeans(3, n_init=10, random_state=seed).fit(z[ix]).predict(z)
        b = KMeans(3, n_init=10, random_state=0).fit_predict(z)
        stability.append(float(adjusted_rand_score(a, b)))
    xx = np.sort(rng.uniform(-1, 1, 40))
    yy = np.sin(3 * xx) + rng.normal(scale=0.15, size=len(xx))
    grid = np.linspace(-2, 2, 300)

    def basis(a):
        return np.polynomial.legendre.legvander(a, 5)

    posterior = BayesianLinearRegression(prior_variance=1, noise_variance=0.15**2).fit(
        basis(xx), yy
    )
    pred, sd = posterior.predict(basis(grid))
    np.savez_compressed(
        run.output / "geometry.npz",
        x=x,
        labels=labels,
        means=custom.means_,
        covariances=custom.covariances_,
        grid=grid,
        posterior_mean=pred,
        posterior_sd=sd,
        regression_x=xx,
        regression_y=yy,
    )
    run.finish(
        {
            "mixture_selection": candidates,
            "em_history": custom.history_,
            "heldout_nll": {
                "custom_em": float(-custom.score_samples(x[test]).mean()),
                "sklearn_em": float(-reference.score(x[test])),
            },
            "ari": {
                "gmm": adjusted_rand_score(labels[test], custom.predict(x[test])),
                "kmeans": adjusted_rand_score(labels[test], km.predict(x[test])),
            },
            "wine": {
                "bootstrap_stability_ari": stability,
                "silhouette": silhouette_score(z, b),
                "class_ari": adjusted_rand_score(wine.target, b),
            },
            "posterior": {
                "in_range_sd": float(sd[np.abs(grid) < 1].mean()),
                "extrapolation_sd": float(sd[np.abs(grid) > 1.5].mean()),
            },
        }
    )


def kernels(config):
    run = Run(config["output"], config)
    # Bundled digits gives an offline kernel study; image experiments use full CIFAR.
    data = load_digits()
    x, y = data.data / 16, data.target
    train, test = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=17)
    rows = []
    for seed in config["seeds"]:
        fit, val = train_test_split(train, test_size=0.2, stratify=y[train], random_state=seed)
        for size in [100, 300, 1000]:
            ix, _ = train_test_split(fit, train_size=size, stratify=y[fit], random_state=seed)
            for name in ["logistic", "rbf_svm", "nystroem"]:
                candidates = []
                for c in [0.1, 1, 10]:
                    if name == "logistic":
                        model = make_pipeline(
                            StandardScaler(), LogisticRegression(C=c, max_iter=2000)
                        )
                    elif name == "rbf_svm":
                        model = make_pipeline(StandardScaler(), SVC(C=c, gamma="scale"))
                    else:
                        model = make_pipeline(
                            StandardScaler(),
                            Nystroem(gamma=0.01, n_components=min(200, size), random_state=seed),
                            LogisticRegression(C=c, max_iter=2000),
                        )
                    model.fit(x[ix], y[ix])
                    candidates.append((accuracy_score(y[val], model.predict(x[val])), model, c))
                _, model, c = max(candidates, key=lambda a: a[0])
                rows.append(
                    {
                        "seed": seed,
                        "train_size": size,
                        "model": name,
                        "selected_C": c,
                        "accuracy": accuracy_score(y[test], model.predict(x[test])),
                    }
                )
    run.finish(
        {
            "rows": rows,
            "test_n": len(test),
            "protocol": "fixed stratified test, validation-only C selection",
        }
    )
