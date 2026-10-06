"""Gaussian posterior regression and full-covariance Gaussian mixture EM."""

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans


class BayesianLinearRegression:
    """w ~ N(0, prior_variance I), y|w ~ N(Xw, noise_variance I)."""

    def __init__(self, prior_variance=1.0, noise_variance=0.1):
        if prior_variance <= 0 or noise_variance <= 0:
            raise ValueError("Variances must be positive")
        self.prior_variance, self.noise_variance = prior_variance, noise_variance

    def fit(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        precision = np.eye(x.shape[1]) / self.prior_variance + x.T @ x / self.noise_variance
        self.covariance_ = np.linalg.solve(precision, np.eye(x.shape[1]))
        self.mean_ = np.linalg.solve(precision, x.T @ y / self.noise_variance)
        return self

    def predict(self, x, include_noise=True):
        x = np.asarray(x)
        variance = np.einsum("ij,jk,ik->i", x, self.covariance_, x)
        if include_noise:
            variance += self.noise_variance
        return x @ self.mean_, np.sqrt(np.maximum(variance, 0))


class GaussianMixtureEM:
    """EM with log-domain responsibilities and covariance flooring."""

    def __init__(self, components=3, max_iter=200, tol=1e-6, reg=1e-6, seed=0):
        self.components, self.max_iter, self.tol = components, max_iter, tol
        self.reg, self.seed = reg, seed

    def _joint(self, x):
        terms = []
        d = x.shape[1]
        for w, mu, cov in zip(self.weights_, self.means_, self.covariances_):
            chol = np.linalg.cholesky(cov)
            z = np.linalg.solve(chol, (x - mu).T).T
            terms.append(
                np.log(w)
                - 0.5
                * (d * np.log(2 * np.pi) + 2 * np.log(np.diag(chol)).sum() + (z * z).sum(axis=1))
            )
        return np.stack(terms, axis=1)

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        if not 1 <= self.components <= len(x):
            raise ValueError("Invalid component count")
        self.means_ = (
            KMeans(self.components, n_init=3, random_state=self.seed).fit(x).cluster_centers_
        )
        self.weights_ = np.full(self.components, 1 / self.components)
        self.covariances_ = np.repeat(
            (np.atleast_2d(np.cov(x.T)) + self.reg * np.eye(x.shape[1]))[None],
            self.components,
            axis=0,
        )
        self.history_ = [float(self.score_samples(x).mean())]
        for _ in range(self.max_iter):
            joint = self._joint(x)
            r = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
            nk = r.sum(axis=0)
            if np.any(nk < 1e-8):
                raise FloatingPointError("Collapsed component: retry another initialization")
            self.weights_ = nk / len(x)
            self.means_ = r.T @ x / nk[:, None]
            for k in range(self.components):
                diff = x - self.means_[k]
                self.covariances_[k] = (diff.T * r[:, k]) @ diff / nk[k] + self.reg * np.eye(
                    x.shape[1]
                )
            self.history_.append(float(self.score_samples(x).mean()))
            if abs(self.history_[-1] - self.history_[-2]) < self.tol:
                break
        return self

    def score_samples(self, x):
        return logsumexp(self._joint(np.asarray(x)), axis=1)

    def predict_proba(self, x):
        j = self._joint(np.asarray(x))
        return np.exp(j - logsumexp(j, axis=1, keepdims=True))

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)
