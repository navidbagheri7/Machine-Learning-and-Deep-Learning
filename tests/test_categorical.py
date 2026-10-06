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
