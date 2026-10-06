"""Stable Gaussian-mixture conditioning for linear inverse problems.

This module computes observation evidence, posterior component means and MMSE.
Choosing the most probable component mean is explicitly a heuristic, not mixture MAP.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp


class MixtureInverse:
    """p(z)=sum pi N(mu,C); y=Az+b+epsilon, epsilon~N(0,R).

    R may be a scalar variance or a full covariance. Solves in observation space
    avoid inverting singular/ill-conditioned prior covariance matrices.
    """

    def __init__(self, weights, means, covariances, operator, noise, offset=None):
        self.weights = np.asarray(weights, dtype=float)
        self.means = np.asarray(means, dtype=float)
        self.covariances = np.asarray(covariances, dtype=float)
        self.operator = np.asarray(operator, dtype=float)
        k, d = self.means.shape
        q = self.operator.shape[0]
        if (
            self.operator.shape[1] != d
            or self.covariances.shape != (k, d, d)
            or self.weights.shape != (k,)
            or np.any(self.weights <= 0)
            or not np.isclose(self.weights.sum(), 1)
        ):
            raise ValueError("Invalid mixture or operator dimensions/weights")
        self.offset = np.zeros(q) if offset is None else np.broadcast_to(offset, (q,))
        r = np.eye(q) * noise if np.ndim(noise) == 0 else np.asarray(noise, float)
        if r.shape != (q, q) or np.linalg.eigvalsh(r).min() <= 0:
            raise ValueError("Noise covariance must be positive definite")
        self.observation_means = self.means @ self.operator.T + self.offset
        self.factors, self.gains, self.posterior_covariances = [], [], []
        for c in self.covariances:
            cross = c @ self.operator.T
            s = self.operator @ cross + r
            factor = cho_factor(s, lower=True)
            gain = cho_solve(factor, cross.T).T
            post = c - gain @ cross.T
            self.factors.append(factor)
            self.gains.append(gain)
            self.posterior_covariances.append((post + post.T) / 2)
        self.posterior_covariances = np.asarray(self.posterior_covariances)

    def condition(self, observations, return_covariance=False):
        y = np.atleast_2d(np.asarray(observations, dtype=float))
        if y.shape[1] != self.operator.shape[0] or not np.isfinite(y).all():
            raise ValueError("Invalid observations")
        joint, means = [], []
        for w, mu, oy, factor, gain in zip(
            self.weights, self.means, self.observation_means, self.factors, self.gains
        ):
            diff = y - oy
            quad = (diff * cho_solve(factor, diff.T).T).sum(axis=1)
            logdet = 2 * np.log(np.diag(factor[0])).sum()
            joint.append(np.log(w) - 0.5 * (y.shape[1] * np.log(2 * np.pi) + logdet + quad))
            means.append(mu + diff @ gain.T)
        joint = np.stack(joint, axis=1)
        log_evidence = logsumexp(joint, axis=1)
        probabilities = np.exp(joint - log_evidence[:, None])
        means = np.stack(means, axis=1)
        mmse = np.einsum("nk,nkd->nd", probabilities, means)
        hard = means[np.arange(len(y)), probabilities.argmax(axis=1)]
        result = {
            "mmse": mmse,
            "hard": hard,
            "weights": probabilities,
            "component_means": means,
            "log_evidence": log_evidence,
        }
        if return_covariance:
            delta = means - mmse[:, None]
            result["covariance"] = np.einsum(
                "nk,kij->nij", probabilities, self.posterior_covariances
            ) + np.einsum("nk,nki,nkj->nij", probabilities, delta, delta)
        return result


def extract_patches(images, size):
    x = np.asarray(images)
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3 or not 1 <= size <= min(x.shape[-2:]):
        raise ValueError("Invalid image or patch dimensions")
    return (
        np.lib.stride_tricks.sliding_window_view(x, (size, size), axis=(-2, -1))
        .reshape(-1, size * size)
        .copy()
    )


def overlap_average(patches, shape, size):
    n, h, w = shape
    p = np.asarray(patches).reshape(n, h - size + 1, w - size + 1, size, size)
    result, counts = np.zeros(shape), np.zeros((h, w))
    for i in range(size):
        for j in range(size):
            result[:, i : i + h - size + 1, j : j + w - size + 1] += p[..., i, j]
            counts[i : i + h - size + 1, j : j + w - size + 1] += 1
    return result / counts


def restore_images(images, size, model, method="mmse", batch_size=1024):
    patches = extract_patches(images, size)
    predictions = [
        model.condition(patches[i : i + batch_size])[method]
        for i in range(0, len(patches), batch_size)
    ]
    return np.clip(overlap_average(np.concatenate(predictions), images.shape, size), 0, 1)
