"""Export fitted-space reference bytes only after matching the recorded fingerprints."""

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits, load_wine, make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from learning_lab.artifacts import fingerprint


def main():
    metrics = json.loads(Path("results/reference/unsupervised/metrics.json").read_text())
    output = Path("assets/source_data/clustering_representations")
    output.mkdir(parents=True, exist_ok=True)
    for case in ["variable_density", "wine", "digits"]:
        if case == "variable_density":
            raw, _ = make_blobs(
                n_samples=[250, 350, 400],
                centers=[[-4, 0], [0, 3], [4, 0]],
                cluster_std=[0.15, 0.5, 1.4],
                random_state=183,
            )
        else:
            raw = (load_wine() if case == "wine" else load_digits()).data
        with threadpool_limits(1):
            x = StandardScaler().fit_transform(raw)
            if x.shape[1] > 30:
                x = PCA(30, svd_solver="full").fit_transform(x)
        hashes = {r["representation_fingerprint"] for r in metrics["rows"] if r["case"] == case}
        assert hashes == {fingerprint(x)}, "Exact fit environment required for initial export"
        target = output / (case + ".npz")
        if target.exists():
            with np.load(target, allow_pickle=False) as values:
                np.testing.assert_array_equal(values["x"], x)
        else:
            np.savez_compressed(target, x=x)
    print("Exported three fingerprint-verified clustering representations")


if __name__ == "__main__":
    main()
