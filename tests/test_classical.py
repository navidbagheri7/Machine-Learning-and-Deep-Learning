"""Check the extension's data roles and probability/decision contracts."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from learning_lab.classical import classify_metrics, dataset_split, logits
from learning_lab.tabular import _split
from learning_lab.uncertainty import fit_temperature


def test_fixed_numeric_test_is_separate_from_every_fitting_partition():
    x, y, xt, yt, train_ids, test_ids = dataset_split("digits", dict(test_seed=9183))
    assert not set(train_ids) & set(test_ids)
    assert len(x) + len(xt) == 1797
    for seed in [0, 1, 2]:
        partitions = _split(np.arange(len(y)), y, seed)
        assert sum(len(p) for p in partitions) == len(y)
        assert len(set(np.concatenate(partitions))) == len(y)
        assert all(set(y[p]) == set(yt) for p in partitions)


def test_binary_margin_and_probability_paths_keep_class_order_and_calibration():
    from scipy.special import softmax

    x, y = make_classification(n_samples=120, n_features=5, random_state=3)
    for model in [LogisticRegression(), LinearSVC(dual="auto")]:
        model.fit(x[:80], y[:80])
        scores = logits(model, x[80:])
        assert scores.shape == (40, 2)
        np.testing.assert_array_equal(scores.argmax(1), model.predict(x[80:]))
        temperature = fit_temperature(scores[:20], y[80:100])
        probabilities = softmax(scores[20:] / temperature, axis=1)
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(1), 1)
        metrics = classify_metrics(y[100:], probabilities)
        assert sum(map(sum, metrics["confusion"])) == 20
