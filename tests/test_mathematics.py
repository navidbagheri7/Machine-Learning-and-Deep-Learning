import numpy as np
import pytest
from scipy.special import softmax
from sklearn.linear_model import Ridge

from learning_lab.probabilistic import BayesianLinearRegression, GaussianMixtureEM
from learning_lab.tabular import _split
from learning_lab.uncertainty import (
    conformal_quantile,
    cqr_interval,
    fit_temperature,
    prediction_sets,
    regression_interval,
)


def test_posterior_mean_equals_matching_ridge_and_variance_contracts():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(100, 8))
    y = rng.normal(size=100)
    posterior = BayesianLinearRegression(prior_variance=2, noise_variance=0.3).fit(x, y)
    reference = Ridge(alpha=0.3 / 2, fit_intercept=False).fit(x, y)
    np.testing.assert_allclose(posterior.mean_, reference.coef_, atol=1e-10)
    _, prior_sd = BayesianLinearRegression(2, 0.3).fit(x[:5], y[:5]).predict(x, include_noise=False)
    _, posterior_sd = posterior.predict(x, include_noise=False)
    assert np.all(posterior_sd <= prior_sd + 1e-10)


def test_em_density_responsibilities_and_monotonicity():
    rng = np.random.default_rng(0)
    x = np.concatenate(
        [rng.normal([-3, 0], 0.4, size=(200, 2)), rng.normal([3, 1], 0.5, size=(200, 2))]
    )
    model = GaussianMixtureEM(2).fit(x)
    np.testing.assert_allclose(model.predict_proba(x).sum(1), 1)
    assert np.min(np.diff(model.history_)) >= -1e-6
    grid = np.linspace(-7, 7, 400)
    xx, yy = np.meshgrid(grid, grid)
    density = np.exp(model.score_samples(np.column_stack([xx.ravel(), yy.ravel()])))
    assert abs(density.sum() * (grid[1] - grid[0]) ** 2 - 1) < 0.01


def test_conformal_finite_sample_rank_not_interpolated_quantile():
    assert conformal_quantile(np.arange(9), 0.2) == 7
    assert np.isinf(conformal_quantile([1, 2], 0.1))
    with pytest.raises(ValueError):
        conformal_quantile([np.nan], 0.1)


def test_conformal_empirical_marginal_coverage():
    rng = np.random.default_rng(0)
    covered = []
    for _ in range(500):
        calibration = rng.normal(size=100)
        test = rng.normal(size=100)
        lo, hi = regression_interval(np.zeros(100), np.zeros(100), calibration)
        covered.append(np.mean((test >= lo) & (test <= hi)))
    assert 0.89 <= np.mean(covered) <= 0.93


def test_cqr_repair_crossing_and_class_sets():
    lo, hi = cqr_interval(np.ones(5), -np.ones(5), np.ones(20), -np.ones(20), np.zeros(20))
    np.testing.assert_allclose(lo, 0)
    np.testing.assert_allclose(hi, 0)
    sets = prediction_sets(
        np.array([[0.9, 0.1], [0.4, 0.6]]), np.tile([0.8, 0.2], (100, 1)), np.zeros(100, dtype=int)
    )
    assert sets.tolist() == [[True, False], [False, False]]


def test_probability_calibration_does_not_increase_objective():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(1000, 3))
    probabilities = softmax(logits, axis=1)
    y = np.array([rng.choice(3, p=p) for p in probabilities])
    inflated = 4 * logits
    temperature = fit_temperature(inflated, y)
    assert 2.5 < temperature < 5.5


def test_selection_and_calibration_partitions_are_disjoint():
    y = np.tile([0, 1], 500)
    partitions = _split(np.arange(len(y)), y, 0)
    assert len(np.concatenate(partitions)) == len(y)
    assert len(np.unique(np.concatenate(partitions))) == len(y)
