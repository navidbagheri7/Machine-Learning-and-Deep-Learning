"""Compare centroid, mixture, graph and adaptive-density clustering with noise coverage."""

import hashlib
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.datasets import load_digits, load_wine, make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .artifacts import Run, fingerprint, write_json


def experiment(config):
    run = Run(config["output"], config)
    run.record["fitting_source_sha256"] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ["clustering.py", "artifacts.py"]
    }
    write_json(run.output / "run.json", run.record | dict(status="running"))
    rows = []
    public = Path("assets/source_data/clustering")
    public.mkdir(parents=True, exist_ok=True)
    for case in config["cases"]:
        if case == "variable_density":
            raw, truth = make_blobs(
                n_samples=[250, 350, 400],
                centers=[[-4, 0], [0, 3], [4, 0]],
                cluster_std=[0.15, 0.5, 1.4],
                random_state=183,
            )
            count, eps, minimum = 3, 0.16, 25
            origin = "Procedural unequal-density three-component Gaussian example, fixed seed183"
        else:
            data = load_wine() if case == "wine" else load_digits()
            raw, truth = data.data, data.target
            count, eps, minimum = (3, 1.8, 15) if case == "wine" else (10, 3.2, 25)
            origin = f"Bundled scikit-learn {case}; labels withheld from all fitting"
        x = StandardScaler().fit_transform(raw)
        if x.shape[1] > 30:
            x = PCA(30, svd_solver="full").fit_transform(x)
        projection = PCA(2, svd_solver="full").fit_transform(x)
        values = dict(projection=projection, true_labels=truth)
        for seed in config["seeds"]:
            models = {
                "kmeans": KMeans(count, n_init=20, random_state=seed),
                "gmm": GaussianMixture(count, n_init=5, random_state=seed),
                "variational_gmm": BayesianGaussianMixture(
                    n_components=2 * count,
                    n_init=3,
                    max_iter=500,
                    weight_concentration_prior=0.1,
                    random_state=seed,
                ),
                "agglomerative": AgglomerativeClustering(count, linkage="ward"),
                "spectral": SpectralClustering(
                    count,
                    affinity="nearest_neighbors",
                    n_neighbors=12,
                    assign_labels="kmeans",
                    random_state=seed,
                    n_jobs=1,
                ),
                "dbscan": DBSCAN(eps=eps, min_samples=8, n_jobs=1),
                "hdbscan": HDBSCAN(min_cluster_size=minimum, min_samples=8, n_jobs=1),
            }
            for name in config["methods"]:
                model = models[name]
                with threadpool_limits(1), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if hasattr(model, "fit_predict"):
                        labels = model.fit_predict(x)
                    else:
                        labels = model.fit(x).predict(x)
                covered = labels >= 0
                clusters = len(set(labels[covered]))
                silhouette = (
                    float(
                        silhouette_score(
                            x[covered],
                            labels[covered],
                            sample_size=min(1000, int(covered.sum())),
                            random_state=seed,
                        )
                    )
                    if 1 < clusters < covered.sum()
                    else None
                )
                row = dict(
                    case=case,
                    method=name,
                    seed=seed,
                    ari_all_points=float(adjusted_rand_score(truth, labels)),
                    nonnoise_fraction=float(covered.mean()),
                    clusters=clusters,
                    silhouette_nonnoise=silhouette,
                    warnings=sorted({str(w.message) for w in caught}),
                    parameters={key: str(value) for key, value in model.get_params().items()},
                    origin=origin,
                    representation_fingerprint=fingerprint(x),
                    protocol=config["scope"],
                )
                rows.append(row)
                values[f"{name}_{seed}"] = labels
                print(
                    case,
                    name,
                    seed,
                    "ARI",
                    round(row["ari_all_points"], 3),
                    "coverage",
                    round(row["nonnoise_fraction"], 3),
                    flush=True,
                )
        np.savez_compressed(public / (case + ".npz"), **values)
    run.finish(
        dict(
            rows=rows,
            source_data_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in public.glob("*.npz")
            },
            protocol=config["scope"],
        )
    )
