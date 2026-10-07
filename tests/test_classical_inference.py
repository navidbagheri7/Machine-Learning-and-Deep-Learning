"""Check calibration identity, input schemas and uncertainty on deployed models."""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from learning_lab.classical_inference import predict


def bundle(tmp_path):
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    model = LogisticRegression().fit(x, [0, 0, 1, 1])
    checkpoint = tmp_path / "classifier.joblib"
    joblib.dump(dict(model=model, temperature=1.0, dataset="breast_cancer"), checkpoint)
    metadata = dict(
        task="classification",
        dataset="breast_cancer",
        features=["a", "b"],
        categorical=[],
        temperature=1.0,
        conformal_cutoff=0.9,
        classes=[0, 1],
        class_labels=["malignant", "benign"],
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(metadata))
    return checkpoint, path, metadata, pd.DataFrame(x, columns=["a", "b"])


def test_reordered_columns_and_prediction_sets_preserve_class_semantics(tmp_path):
    checkpoint, path, _, x = bundle(tmp_path)
    output = predict(checkpoint, path, x[["b", "a"]].assign(unused=12))
    np.testing.assert_array_equal(output, predict(checkpoint, path, x))
    p = output[["probability_class_0", "probability_class_1"]].to_numpy()
    np.testing.assert_allclose(p.sum(1), 1)
    np.testing.assert_array_equal(output.set_size, (1 - p <= 0.9).sum(1))
    assert list(output.predicted_label) == ["malignant", "malignant", "benign", "benign"]


def test_checkpoint_identity_and_invalid_inputs_are_rejected(tmp_path):
    checkpoint, path, metadata, x = bundle(tmp_path)
    with pytest.raises(ValueError, match="Missing"):
        predict(checkpoint, path, x.drop(columns="a"))
    with pytest.raises(ValueError, match="Infinite"):
        predict(checkpoint, path, x.assign(a=np.inf))
    metadata["temperature"] = 2
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="identity"):
        predict(checkpoint, path, x)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="differs"):
        predict(checkpoint, path, x)
