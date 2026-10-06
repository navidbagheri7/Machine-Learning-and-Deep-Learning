import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from learning_lab.inverse import MixtureInverse, extract_patches, overlap_average


def test_linear_conditioning_rectangular_operator_and_correlated_noise():
    mean = np.array([[0.4, -0.2, 0.8]])
    cov = np.array([[[1.0, 0.2, 0.1], [0.2, 0.8, 0.3], [0.1, 0.3, 0.6]]])
    operator = np.array([[1.0, 0.4, 0.0], [0.0, 1.0, 1.0]])
    noise = np.array([[0.2, 0.03], [0.03, 0.1]])
    offset = np.array([0.2, -0.1])
    observed = np.array([[0.7, 1.0]])
    model = MixtureInverse([1.0], mean, cov, operator, noise, offset)
    result = model.condition(observed, return_covariance=True)
    s = operator @ cov[0] @ operator.T + noise
    gain = cov[0] @ operator.T @ np.linalg.inv(s)
    np.testing.assert_allclose(
        result["mmse"], mean + (observed - mean @ operator.T - offset) @ gain.T
    )
    np.testing.assert_allclose(
        result["covariance"][0], cov[0] - gain @ operator @ cov[0], atol=1e-12
    )
    assert np.linalg.eigvalsh(result["covariance"][0]).min() > 0


def test_component_weights_use_observation_evidence_and_total_covariance():
    weights, means, covs = (
        np.array([0.8, 0.2]),
        np.array([[-1.0], [2.0]]),
        np.array([[[0.5]], [[2.0]]]),
    )
    model = MixtureInverse(weights, means, covs, [[0.7]], 0.2)
    y = np.array([[-0.3], [1.5]])
    result = model.condition(y, return_covariance=True)
    joint = np.column_stack(
        [
            np.log(w) + multivariate_normal.logpdf(y, 0.7 * mu, 0.49 * cov + 0.2)
            for w, mu, cov in zip(weights, means, covs)
        ]
    )
    r = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
    np.testing.assert_allclose(result["weights"], r, atol=1e-12)
    expected_mean = (r[:, :, None] * result["component_means"]).sum(1)
    expected_variance = (
        r
        * (
            model.posterior_covariances[:, 0, 0][None]
            + (result["component_means"][:, :, 0] - expected_mean) ** 2
        )
    ).sum(1)
    np.testing.assert_allclose(result["mmse"], expected_mean)
    np.testing.assert_allclose(result["covariance"][:, 0, 0], expected_variance)
    # The incorrect prior-only weight formula must disagree with Bayes' rule.
    assert not np.allclose(r, weights[None])


def test_patch_roundtrip_including_single_pixel_edges_and_full_image():
    x = np.random.default_rng(3).normal(size=(3, 28, 28))
    for m in [1, 4, 8, 12, 16, 20, 28]:
        patches = extract_patches(x, m)
        np.testing.assert_allclose(overlap_average(patches, x.shape, m), x, atol=1e-13)
    with pytest.raises(ValueError):
        extract_patches(x, 29)


def test_singular_prior_and_extreme_observation_remain_finite():
    model = MixtureInverse(
        [0.5, 0.5], [[0.0, 0.0], [1.0, 1.0]], np.zeros((2, 2, 2)), np.eye(2), 0.01
    )
    result = model.condition([[1e4, 1e4]], return_covariance=True)
    assert np.isfinite(result["weights"]).all()
    np.testing.assert_allclose(result["weights"].sum(1), 1)
    np.testing.assert_allclose(result["mmse"], [[1.0, 1.0]])
    with pytest.raises(ValueError):
        MixtureInverse([1.0], [[0.0]], [[[1.0]]], [[1.0]], 0.0)
