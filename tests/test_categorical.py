import numpy as np
import pytest

from learning_lab.categorical import CategoricalMixtureEM


def test_categorical_missing_data_normalization_and_imputation():
    x = np.array([[0, 0], [0, 1], [1, 1], [1, -1], [-1, 0]])
    model = CategoricalMixtureEM(2, 2, seed=3).fit(x)
    np.testing.assert_allclose(model.probabilities_.sum(-1), 1)
    np.testing.assert_allclose(model.weights_.sum(), 1)
    np.testing.assert_allclose(model.predict_proba([[-1, -1]]), model.weights_[None])
    np.testing.assert_allclose(model.score_samples([[-1, -1]]), 0, atol=1e-12)
    np.testing.assert_allclose(model.impute_probabilities(x).sum(-1), 1)
    assert np.isfinite(model.score_samples(x)).all()
    with pytest.raises(ValueError):
        model.score_samples([[0, 2]])


def test_single_component_m_step_matches_observed_counts():
    x = np.array([[0, 1], [0, 1], [1, -1]])
    alpha = 0.1
    m = CategoricalMixtureEM(1, 2, smoothing=alpha).fit(x)
    np.testing.assert_allclose(m.probabilities_[0, 0], (np.array([2, 1]) + alpha) / (3 + 2 * alpha))
    np.testing.assert_allclose(m.probabilities_[0, 1], (np.array([0, 2]) + alpha) / (2 + 2 * alpha))


def test_imputation_conditions_on_known_values_and_preserves_them():
    model = CategoricalMixtureEM(2, 2)
    model.weights_ = np.array([0.25, 0.75])
    model.probabilities_ = np.array([[[0.8, 0.2], [0.9, 0.1]], [[0.2, 0.8], [0.1, 0.9]]])
    # Observing first feature=0 gives posterior component weights 4/7 and 3/7.
    actual = model.impute_probabilities([[0, -1], [-1, -1], [1, 0]])
    np.testing.assert_allclose(actual[0], [[1, 0], [3.9 / 7, 3.1 / 7]])
    np.testing.assert_allclose(actual[1], [[0.35, 0.65], [0.3, 0.7]])
    np.testing.assert_array_equal(actual[2], [[0, 1], [1, 0]])
