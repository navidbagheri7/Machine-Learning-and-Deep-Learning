"""Validate public clustering evidence and regenerate density/coverage comparisons."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_wine, make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from learning_lab.artifacts import fingerprint, write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate():
    config = json.loads(Path("configs/unsupervised.json").read_text())
    directory = Path(config["output"])
    record = json.loads((directory / "run.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    assert record["status"] == "complete" and record["config"] == config
    for name, value in record["fitting_source_sha256"].items():
        assert digest(Path("src/learning_lab") / name) == value
    expected = {
        (case, method, seed)
        for case in config["cases"]
        for method in config["methods"]
        for seed in config["seeds"]
    }
    actual = [(r["case"], r["method"], r["seed"]) for r in metrics["rows"]]
    assert len(actual) == len(set(actual)) == 63 and set(actual) == expected
    for case in config["cases"]:
        path = Path("assets/source_data/clustering") / (case + ".npz")
        assert digest(path) == metrics["source_data_sha256"][path.name]
        if case == "variable_density":
            raw, truth = make_blobs(
                n_samples=[250, 350, 400],
                centers=[[-4, 0], [0, 3], [4, 0]],
                cluster_std=[0.15, 0.5, 1.4],
                random_state=183,
            )
        else:
            data = load_wine() if case == "wine" else load_digits()
            raw, truth = data.data, data.target
        with threadpool_limits(1):
            x = StandardScaler().fit_transform(raw)
            reduced = x.shape[1] > 30
            if reduced:
                x = PCA(30, svd_solver="full").fit_transform(x)
        representation = Path("assets/source_data/clustering_representations") / (case + ".npz")
        with np.load(representation, allow_pickle=False) as stored:
            assert stored["x"].shape == x.shape
            if reduced:
                # Singular vectors may flip signs across valid LAPACK builds.
                # Align orientation only; the coordinate checks stay strict.
                x *= np.where((x * stored["x"]).sum(axis=0) < 0, -1, 1)
            np.testing.assert_allclose(x, stored["x"], atol=1e-8, rtol=1e-8)
            x = stored["x"]
        with threadpool_limits(1):
            projection = PCA(2, svd_solver="full").fit_transform(x)
        with np.load(path, allow_pickle=False) as arrays:
            np.testing.assert_array_equal(truth, arrays["true_labels"])
            projection *= np.where((projection * arrays["projection"]).sum(axis=0) < 0, -1, 1)
            np.testing.assert_allclose(projection, arrays["projection"], atol=1e-8, rtol=1e-8)
            for row in (r for r in metrics["rows"] if r["case"] == case):
                labels = arrays[f"{row['method']}_{row['seed']}"]
                assert labels.dtype.kind == "i" and labels.shape == truth.shape
                assert row["representation_fingerprint"] == fingerprint(x)
                assert row["ari_all_points"] == adjusted_rand_score(truth, labels)
                covered = labels >= 0
                clusters = len(set(labels[covered]))
                assert row["nonnoise_fraction"] == covered.mean()
                assert row["clusters"] == clusters
                silhouette = (
                    silhouette_score(
                        x[covered],
                        labels[covered],
                        sample_size=min(1000, int(covered.sum())),
                        random_state=row["seed"],
                    )
                    if 1 < clusters < covered.sum()
                    else None
                )
                if silhouette is None:
                    assert row["silhouette_nonnoise"] is None
                else:
                    assert np.isclose(row["silhouette_nonnoise"], silhouette, atol=1e-10, rtol=0)
    write_json(
        "docs/clustering_validation.json",
        dict(
            status="passed",
            fitted_models=63,
            independently_recomputed_metrics=True,
            source_and_data_hashes=True,
        ),
    )
    return config, metrics["rows"]


def main():
    config, rows = validate()
    frame = pd.DataFrame(rows)
    frame.to_csv("docs/clustering_individual_fits.csv", index=False)
    aggregate = frame.groupby(["case", "method"])[
        ["ari_all_points", "nonnoise_fraction", "clusters", "silhouette_nonnoise"]
    ].agg(["mean", "std"])
    aggregate.to_csv("docs/clustering_aggregates.csv")
    lines = [
        "# Clustering geometry and noise coverage",
        "",
        config["scope"],
        "",
        "Seven families, three datasets and three seeds give 63 fitted results. Deterministic families repeat the same fit; their zero seed variance is not evidence of robustness to new datasets. Cluster-count priors differ from density methods, and presets are fixed before label evaluation.",
        "",
        "Standardization uses all unlabeled examples. Digits uses 30 principal components; the two-dimensional projection is only a display. ARI includes noise points, and non-noise silhouette is explicitly conditional on accepted points. A visually clean tiny accepted subset can still have poor full-data recovery.",
    ]
    for case in config["cases"]:
        data = aggregate.loc[case].sort_values(("ari_all_points", "mean"), ascending=False)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for ax, key, title in zip(
            axes,
            ["ari_all_points", "nonnoise_fraction"],
            ["ARI on all points", "Non-noise coverage"],
        ):
            ax.barh(data.index, data[(key, "mean")], xerr=data[(key, "std")], color="#087b94")
            ax.invert_yaxis()
            ax.set(xlabel=title, title=case.replace("_", " "))
            ax.grid(axis="x", alpha=0.2)
        for extension in ["png", "pdf"]:
            fig.savefig(f"assets/clustering_{case}.{extension}", dpi=170)
        plt.close(fig)
        lines.extend(
            [
                "",
                "## " + case.replace("_", " ").title(),
                "",
                "| Method | ARI, mean ± fit SD | Coverage | Clusters | Conditional silhouette |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, row in data.iterrows():
            lines.append(
                f"| {name} | {row[('ari_all_points', 'mean')]:.4f} ± {row[('ari_all_points', 'std')]:.4f} | {row[('nonnoise_fraction', 'mean')]:.3f} | {row[('clusters', 'mean')]:.1f} | {row[('silhouette_nonnoise', 'mean')]:.3f} |"
            )
        lines.extend(["", f"![{case} clustering](../assets/clustering_{case}.png)"])
    lines.extend(
        [
            "",
            "## What the comparison shows",
            "",
            "A mixture recovers the variable-density Gaussian construction well. Adaptive density does not guarantee recovery of semantic image classes: HDBSCAN rejects most Digits examples, and its full-data ARI is substantially weaker than the mixture reference. This is a retained limitation, not a hand-selected illustration of success.",
            "",
            "Run `python scripts/run_clustering.py`, then `python scripts/report_clustering.py`. The second command verifies labels, representation fingerprints, source hashes and every metric without refitting. [Individual fits](clustering_individual_fits.csv) and [aggregates](clustering_aggregates.csv) retain all methods and seeds. Explore them in [the browser demo](explorer.html).",
            "",
            "Primary references: [scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html), [mixture models](https://scikit-learn.org/stable/modules/mixture.html), and [HDBSCAN API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html).",
        ]
    )
    Path("docs/clustering_results.md").write_text("\n".join(lines) + "\n")
    print("Verified and reported all 63 clustering fits")


if __name__ == "__main__":
    main()
