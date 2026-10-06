"""Assignment-grounded mixture selection and Bayesian image reconstruction studies."""

import json
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.model_selection import train_test_split

from .artifacts import Run, fingerprint, write_json
from .categorical import CategoricalMixtureEM
from .inverse import MixtureInverse, extract_patches, restore_images
from .probabilistic import GaussianMixtureEM


def cache(config):
    root = Path(config["data_root"]) / "projects"
    if not (root / "audit.json").exists():
        raise FileNotFoundError(
            "Prepare datasets with scripts/prepare_projects.py; see docs/10_course_projects.md"
        )
    return root


def mixture_parameters(model, truth):
    cost = ((model.means_[:, None] - truth[None]) ** 2).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    return {
        "weights": model.weights_[rows[np.argsort(cols)]],
        "means": model.means_[rows[np.argsort(cols)]],
        "covariances": model.covariances_[rows[np.argsort(cols)]],
    }


def course_mixtures(config):
    run = Run(config["output"], config)
    directory = cache(config)
    rows, selection, imputation = [], [], []
    datasets = {}
    for number in [1, 2]:
        x = np.load(directory / f"mixture_{number}.npy")
        labels = np.repeat(np.arange(3), 200)
        truth = np.stack([x[labels == k].mean(0) for k in range(3)])
        datasets[str(number)] = {
            "sha256": fingerprint(x),
            "empirical_means": truth,
            "empirical_covariances": [np.cov(x[labels == k].T) for k in range(3)],
        }
        for seed in config["seeds"]:
            fit, other = train_test_split(
                np.arange(len(x)), test_size=0.4, random_state=seed, stratify=labels
            )
            val, test = train_test_split(
                other, test_size=0.5, random_state=seed, stratify=labels[other]
            )
            candidates = []
            for covariance in ["spherical", "diag", "full"]:
                for k in config["components"]:
                    m = GaussianMixture(
                        k, covariance_type=covariance, n_init=5, random_state=seed, reg_covar=1e-5
                    ).fit(x[fit])
                    row = {
                        "dataset": number,
                        "seed": seed,
                        "k": k,
                        "covariance": covariance,
                        "validation_nll": -m.score(x[val]),
                        "bic_fit": m.bic(x[fit]),
                    }
                    selection.append(row)
                    candidates.append((row["validation_nll"], m))
            selected = min(candidates, key=lambda t: t[0])[1]
            reference = GaussianMixture(3, n_init=5, random_state=seed).fit(x[fit])
            custom = max(
                [GaussianMixtureEM(3, seed=seed * 10 + j).fit(x[fit]) for j in range(3)],
                key=lambda m: m.score_samples(x[fit]).mean(),
            )
            variational = BayesianGaussianMixture(
                n_components=8, n_init=3, max_iter=500, random_state=seed
            ).fit(x[fit])
            km = KMeans(3, n_init=10, random_state=seed).fit(x[fit])
            for name, model in [
                ("custom_em_k3", custom),
                ("sklearn_em_k3", reference),
                ("selected_gmm", selected),
                ("variational_gmm", variational),
            ]:
                rows.append(
                    {
                        "dataset": number,
                        "seed": seed,
                        "model": name,
                        "test_nll": -model.score_samples(x[test]).mean(),
                        "test_ari": adjusted_rand_score(labels[test], model.predict(x[test])),
                    }
                )
            rows.append(
                {
                    "dataset": number,
                    "seed": seed,
                    "model": "kmeans",
                    "test_ari": adjusted_rand_score(labels[test], km.predict(x[test])),
                }
            )
            # Hold out one coordinate at test time; fit/select the prior on complete fit/validation only.
            coordinate = 1
            conditioned = MixtureInverse(
                reference.weights_, reference.means_, reference.covariances_, [[1.0, 0.0]], 1e-8
            )
            posterior = conditioned.condition(x[test, :1], return_covariance=True)
            prediction = posterior["mmse"][:, coordinate]
            baseline = x[fit, coordinate].mean()
            imputation.append(
                {
                    "dataset": number,
                    "seed": seed,
                    "conditional_gmm_rmse": np.sqrt(
                        np.mean((prediction - x[test, coordinate]) ** 2)
                    ),
                    "mean_imputation_rmse": np.sqrt(np.mean((baseline - x[test, coordinate]) ** 2)),
                    "mean_posterior_std": np.sqrt(
                        posterior["covariance"][:, coordinate, coordinate]
                    ).mean(),
                }
            )
            if seed == config["seeds"][0]:
                datasets[str(number)]["custom_k3_parameters"] = mixture_parameters(custom, truth)
                datasets[str(number)]["em_log_likelihood"] = custom.history_
                datasets[str(number)]["selected_k"] = selected.n_components
                datasets[str(number)]["selected_covariance"] = selected.covariance_type
                np.savez_compressed(
                    run.output / f"mixture_{number}.npz",
                    x=x,
                    labels=labels,
                    assignment=custom.predict(x),
                    means=custom.means_,
                    covariances=custom.covariances_,
                )
            write_json(
                run.output / f"split_{number}_{seed}.json",
                {"fit": fit, "validation": val, "test": test},
            )
    rng = np.random.default_rng(42)
    true_probabilities = np.full((3, 8, 3), 0.05)
    for k in range(3):
        for j in range(8):
            true_probabilities[k, j, (k + j) % 3] = 0.9
    latent = rng.choice(3, size=4000, p=[0.6, 0.3, 0.1])
    categorical = np.stack(
        [np.array([rng.choice(3, p=true_probabilities[k, j]) for k in latent]) for j in range(8)],
        axis=1,
    )
    masked = categorical.copy()
    missing = rng.random(masked.shape) < 0.25
    masked[missing] = -1
    categorical_model = max(
        [CategoricalMixtureEM(3, 3, seed=s).fit(masked[:3000]) for s in range(5)],
        key=lambda m: m.score_samples(masked[:3000]).mean(),
    )
    predictions = categorical_model.impute_probabilities(masked[3000:]).argmax(2)
    majority = np.array(
        [np.bincount(categorical[:3000, j], minlength=3).argmax() for j in range(8)]
    )
    cmm_results = {
        "test_nll_observed": -categorical_model.score_samples(masked[3000:]).mean(),
        "latent_test_ari": adjusted_rand_score(
            latent[3000:], categorical_model.predict_proba(masked[3000:]).argmax(1)
        ),
        "missing_accuracy": np.mean(
            predictions[missing[3000:]] == categorical[3000:][missing[3000:]]
        ),
        "majority_missing_accuracy": np.mean(
            np.broadcast_to(majority, predictions.shape)[missing[3000:]]
            == categorical[3000:][missing[3000:]]
        ),
        "learned_weights": categorical_model.weights_,
        "history": categorical_model.history_,
        "protocol": "Synthetic identifiable 8-feature categorical mixture; 25% MCAR entries; first 3000 fit, last 1000 test, training-likelihood-only initialization selection",
    }
    run.finish(
        {
            "datasets": datasets,
            "rows": rows,
            "selection": selection,
            "imputation": imputation,
            "categorical_mixture": cmm_results,
            "label_assumption": "Three contiguous 200-point groups, specified by the assignment; never used to fit/select models",
        }
    )


def fit_patch_prior(images, size, components, samples, seed, max_iter=60):
    """Full ambient covariances from a reduced-rank Gaussian mixture + isotropic floor.

    Patches are sampled uniformly over image and spatial origin without materializing
    the full set. PCA/GMM see training images only. This is a resource-bounded prior,
    not full-dimensional EM when size**2 exceeds the retained rank.
    """
    rng = np.random.default_rng(seed)
    ids = rng.integers(len(images), size=samples)
    a = rng.integers(images.shape[1] - size + 1, size=samples)
    b = rng.integers(images.shape[2] - size + 1, size=samples)
    x = np.stack([images[i, u : u + size, v : v + size].ravel() for i, u, v in zip(ids, a, b)])
    rank = min(32, size * size, len(x) - 1)
    pca = PCA(rank, svd_solver="randomized", random_state=seed).fit(x)
    features = pca.transform(x)
    gmm = GaussianMixture(
        components, n_init=2, max_iter=max_iter, reg_covar=1e-4, random_state=seed
    ).fit(features)
    residual = max(1e-4, np.mean((x - pca.inverse_transform(features)) ** 2))
    v = pca.components_
    covs = np.stack([v.T @ c @ v + residual * np.eye(size * size) for c in gmm.covariances_])
    means = pca.inverse_transform(gmm.means_)
    return {
        "weights": gmm.weights_,
        "means": means,
        "covariances": covs,
        "rank": rank,
        "residual_variance": residual,
        "converged": bool(gmm.converged_),
    }


def psnr(clean, predicted):
    mse = np.mean(
        (np.asarray(clean) - np.asarray(predicted)) ** 2, axis=tuple(range(1, np.ndim(clean)))
    )
    return -10 * np.log10(np.maximum(mse, 1e-12))


def inverse(config):
    run = Run(config["output"], config)
    directory = cache(config)
    mnist = np.load(directory / "mnist.npz")
    course_hashes = set()
    for path in directory.glob("course_m*.npz"):
        for image in np.load(path)["clean"]:
            course_hashes.add(fingerprint(image))
    # Course originals can originate from training MNIST: remove all exact matches.
    candidates = np.array(
        [i for i, x in enumerate(mnist["train"]) if fingerprint(x) not in course_hashes]
    )
    rng = np.random.default_rng(19)
    train_ids = rng.choice(candidates, config["train_images"], replace=False)
    train = mnist["train"][train_ids] / 255.0
    validation = (
        mnist["train"][candidates[~np.isin(candidates, train_ids)][: config["validation_images"]]]
        / 255.0
    )
    test = mnist["test"][: config["test_images"]] / 255.0
    sigma = config["controlled_sigma"] / 255.0
    noisy_val = validation + rng.normal(0, sigma, validation.shape)
    noisy_test = test + rng.normal(0, sigma, test.shape)
    priors, tuning, controlled = {}, [], []
    for m in config["patch_sizes"]:
        for k in config["components"]:
            print("Fitting prior", m, k, flush=True)
            prior = fit_patch_prior(
                train, m, k, config["patch_samples"], seed=0, max_iter=config["em_iterations"]
            )
            priors[m, k] = prior
            model = MixtureInverse(
                prior["weights"], prior["means"], prior["covariances"], np.eye(m * m), sigma**2
            )
            for method in ["mmse", "hard"]:
                val = restore_images(noisy_val, m, model, method)
                tuning.append(
                    {
                        "m": m,
                        "k": k,
                        "method": method,
                        "validation_psnr": psnr(validation, val).mean(),
                    }
                )
        chosen = max(
            [r for r in tuning if r["m"] == m and r["method"] == "mmse"],
            key=lambda r: r["validation_psnr"],
        )
        prior = priors[m, chosen["k"]]
        model = MixtureInverse(
            prior["weights"], prior["means"], prior["covariances"], np.eye(m * m), sigma**2
        )
        for method in ["mmse", "hard"]:
            prediction = restore_images(noisy_test, m, model, method)
            controlled.append(
                {
                    "m": m,
                    "k": chosen["k"],
                    "method": method,
                    "test_psnr": psnr(test, prediction).mean(),
                    "per_image_psnr": psnr(test, prediction),
                }
            )
            if m == 8 and method == "mmse":
                np.savez_compressed(
                    run.output / "controlled_gallery.npz",
                    clean=test[:8],
                    observed=np.clip(noisy_test[:8], 0, 1),
                    restored=prediction[:8],
                )
    selected = max([r for r in tuning if r["method"] == "mmse"], key=lambda r: r["validation_psnr"])
    supplied = []
    archive_audit = json.loads((directory / "audit.json").read_text())["archives"]
    for archive in archive_audit:
        m = archive["actual_patch_size"]
        if m not in config["patch_sizes"]:
            continue
        data = np.load(directory / f"course_m{m}.npz")
        clean, observed, ids, operator = (
            data["clean"] / 255.0,
            data["observed"] / 255.0,
            data["ids"],
            data["operator"],
        )
        order = np.random.default_rng(17).permutation(len(ids))
        val_ids, test_ids = (
            order[: config["course_validation"]],
            order[config["course_validation"] :],
        )
        calibration_ids = val_ids[: config["course_calibration"]]
        selection_ids = val_ids[config["course_calibration"] :]
        cp_cal = extract_patches(clean[calibration_ids], m) @ operator.T
        op_cal = extract_patches(observed[calibration_ids], m)
        gain = float(np.mean((cp_cal - cp_cal.mean()) * (op_cal - op_cal.mean())) / cp_cal.var())
        offset = float(op_cal.mean() - gain * cp_cal.mean())
        if gain <= 0:
            raise ValueError("Nonpositive calibrated forward gain")
        if config.get("quick"):
            test_ids = test_ids[:8]
        candidates_course = []
        for k in config["components"]:
            prior = priors[m, k]
            for noise in config["noise_candidates"]:
                model = MixtureInverse(
                    prior["weights"],
                    prior["means"],
                    prior["covariances"],
                    operator,
                    (noise / 255.0) ** 2,
                )
                pred = restore_images(observed[selection_ids], m, model)
                candidates_course.append(
                    {
                        "k": k,
                        "sigma_255": noise,
                        "validation_psnr": psnr(clean[selection_ids], pred).mean(),
                    }
                )
        chosen = max(candidates_course, key=lambda r: r["validation_psnr"])
        prior = priors[m, chosen["k"]]
        model = MixtureInverse(
            prior["weights"],
            prior["means"],
            prior["covariances"],
            operator,
            (chosen["sigma_255"] / 255.0) ** 2,
        )
        soft, hard = (
            restore_images(observed[test_ids], m, model),
            restore_images(observed[test_ids], m, model, "hard"),
        )
        calibrated_candidates = []
        for k in config["components"]:
            prior = priors[m, k]
            for noise in config["noise_candidates"]:
                corrected = MixtureInverse(
                    prior["weights"],
                    prior["means"],
                    prior["covariances"],
                    gain * operator,
                    (noise / 255.0) ** 2,
                    offset=offset,
                )
                pred = restore_images(observed[selection_ids], m, corrected)
                calibrated_candidates.append(
                    {
                        "k": k,
                        "sigma_255": noise,
                        "validation_psnr": psnr(clean[selection_ids], pred).mean(),
                    }
                )
        corrected_choice = max(calibrated_candidates, key=lambda r: r["validation_psnr"])
        prior = priors[m, corrected_choice["k"]]
        corrected = MixtureInverse(
            prior["weights"],
            prior["means"],
            prior["covariances"],
            gain * operator,
            (corrected_choice["sigma_255"] / 255.0) ** 2,
            offset=offset,
        )
        calibrated = restore_images(observed[test_ids], m, corrected)
        cp, op = extract_patches(clean[val_ids], m), extract_patches(observed[val_ids], m)
        residual = op - cp @ operator.T
        row = {
            "archive": archive["file"],
            "actual_m": m,
            "selected": chosen,
            "candidates": candidates_course,
            "validation_ids": ids[val_ids],
            "calibration_ids": ids[calibration_ids],
            "selection_ids": ids[selection_ids],
            "test_ids": ids[test_ids],
            "test_raw_psnr": psnr(clean[test_ids], observed[test_ids]).mean(),
            "test_mmse_psnr": psnr(clean[test_ids], soft).mean(),
            "test_hard_psnr": psnr(clean[test_ids], hard).mean(),
            "test_calibrated_psnr": psnr(clean[test_ids], calibrated).mean(),
            "per_image_calibrated_psnr": psnr(clean[test_ids], calibrated),
            "calibrated_forward": {
                "gain": gain,
                "offset_255": offset * 255,
                "calibration_residual_std_255": np.std(op_cal - gain * cp_cal - offset) * 255,
                "selected": corrected_choice,
                "candidates": calibrated_candidates,
            },
            "per_image_mmse_psnr": psnr(clean[test_ids], soft),
            "observation_residual_mean_255": residual.mean() * 255,
            "observation_residual_std_255": residual.std() * 255,
            "clipped_observation_fraction": np.mean((op == 0) | (op == 1)),
        }
        supplied.append(row)
        np.savez_compressed(
            run.output / f"course_gallery_m{m}.npz",
            clean=clean[test_ids[:6]],
            observed=observed[test_ids[:6]],
            restored=soft[:6],
            hard=hard[:6],
            calibrated=calibrated[:6],
            ids=ids[test_ids[:6]],
        )
        print("Archive result", m, row["test_raw_psnr"], row["test_mmse_psnr"], flush=True)
    joblib.dump(priors[selected["m"], selected["k"]], run.output / "selected_prior.joblib")
    write_json(
        run.output / "split.json",
        {
            "train_indices": train_ids,
            "excluded_training_duplicates": len(mnist["train"]) - len(candidates),
        },
    )
    run.finish(
        {
            "controlled": controlled,
            "tuning": tuning,
            "selected": selected,
            "controlled_raw_psnr": psnr(test, np.clip(noisy_test, 0, 1)).mean(),
            "supplied": supplied,
            "archive_audit": archive_audit,
            "prior_diagnostics": [
                {
                    "m": m,
                    "k": k,
                    "rank": p["rank"],
                    "residual_variance": p["residual_variance"],
                    "converged": p["converged"],
                }
                for (m, k), p in priors.items()
            ],
            "protocol": "Controlled AWGN: m and k selected on independent MNIST validation images. Supplied archives: split paired IDs once; select k/sigma on validation only. Dimensions inferred from W, never filenames.",
        }
    )
