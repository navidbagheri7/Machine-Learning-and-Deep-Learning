"""Latent-class EM for conditionally independent categorical observations."""

import numpy as np
from scipy.special import logsumexp


class CategoricalMixtureEM:
    """Categorical product mixtures with Dirichlet pseudo-counts and missing=-1.

    A mixture over just one categorical variable is not identifiable; repeated
    features give the latent-class model observable dependence. Unobserved entries
    contribute zero to the log likelihood and no categorical counts in the M-step.
    """

    def __init__(self, components, categories, smoothing=0.01, max_iter=300, tol=1e-7, seed=0):
        self.components, self.categories = components, categories
        self.smoothing, self.max_iter, self.tol, self.seed = smoothing, max_iter, tol, seed
        if components < 1 or categories < 2 or smoothing <= 0:
            raise ValueError("Positive components/smoothing and at least two categories required")

    def _validate(self, x):
        x = np.asarray(x)
        if (
            x.ndim != 2
            or not np.isfinite(x).all()
            or not np.equal(x, np.floor(x)).all()
            or np.any((x < -1) | (x >= self.categories))
        ):
            raise ValueError("Expected integer category matrix with -1 for missing")
        return x.astype(int)

    def _joint(self, x):
        x = self._validate(x)
        out = np.broadcast_to(np.log(self.weights_), (len(x), self.components)).copy()
        for j in range(x.shape[1]):
            observed = x[:, j] >= 0
            out[observed] += np.log(self.probabilities_[:, j, x[observed, j]].T)
        return out

    def fit(self, x):
        x = self._validate(x)
        if not len(x):
            raise ValueError("Empty training matrix")
        rng = np.random.default_rng(self.seed)
        self.weights_ = np.full(self.components, 1 / self.components)
        self.probabilities_ = rng.dirichlet(
            np.ones(self.categories), size=(self.components, x.shape[1])
        )
        self.history_ = []
        for _ in range(self.max_iter):
            joint = self._joint(x)
            r = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
            nk = r.sum(0) + self.smoothing
            self.weights_ = nk / nk.sum()
            for j in range(x.shape[1]):
                counts = (
                    np.stack(
                        [r[x[:, j] == value].sum(0) for value in range(self.categories)], axis=1
                    )
                    + self.smoothing
                )
                self.probabilities_[:, j] = counts / counts.sum(1, keepdims=True)
            value = float(self.score_samples(x).mean())
            self.history_.append(value)
            if len(self.history_) > 1 and abs(value - self.history_[-2]) < self.tol:
                break
        return self

    def score_samples(self, x):
        return logsumexp(self._joint(x), axis=1)

    def predict_proba(self, x):
        j = self._joint(x)
        return np.exp(j - logsumexp(j, axis=1, keepdims=True))

    def impute_probabilities(self, x):
        return np.einsum("nk,kdc->ndc", self.predict_proba(x), self.probabilities_)
