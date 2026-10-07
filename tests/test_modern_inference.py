import json
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from learning_lab.data import CATEGORICAL, NUMERIC
from learning_lab.modern_inference import (
    adult_predictions,
    cifar_probabilities,
    housing_predictions,
)
from learning_lab.uncertainty import binary_temperature


def test_native_model_reload_handles_missing_and_unseen_categories(tmp_path):
    catboost = pytest.importorskip("catboost")
    frame = pd.DataFrame({name: np.arange(40, dtype=float) for name in NUMERIC})
    for name in CATEGORICAL:
        frame[name] = np.where(np.arange(40) % 2, "a", "b")
    model = catboost.CatBoostClassifier(
        iterations=8,
        depth=2,
        cat_features=CATEGORICAL,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(frame, np.arange(40) % 2)
    checkpoint = tmp_path / "model.cbm"
    model.save_model(str(checkpoint))
    metadata = tmp_path / "inference.json"
    metadata.write_text(
        json.dumps(
            {
                "features": list(frame.columns),
                "temperature": 2.0,
                "threshold": 0.3,
                "conformal_score_cutoff": 0.7,
                "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
            }
        )
    )
    probe = frame.iloc[:4].copy()
    probe.loc[0, "occupation"] = None
    probe.loc[1, "occupation"] = "unseen"
    prediction = adult_predictions(checkpoint, metadata, probe)
    expected = probe.copy()
    expected["occupation"] = expected["occupation"].fillna("__missing__")
    np.testing.assert_allclose(
        prediction.iloc[:, 0], binary_temperature(model.predict_proba(expected)[:, 1], 2.0)
    )
    assert np.isfinite(prediction.to_numpy(dtype=float)).all()
    np.testing.assert_array_equal(prediction.set_contains_at_most_50k, prediction.iloc[:, 0] <= 0.7)
    np.testing.assert_array_equal(prediction.set_contains_above_50k, prediction.iloc[:, 0] >= 0.3)
    with pytest.raises(ValueError, match="Missing input"):
        adult_predictions(checkpoint, metadata, probe.drop(columns="age"))
    probe.loc[0, "age"] = np.inf
    with pytest.raises(ValueError, match="Infinite input"):
        adult_predictions(checkpoint, metadata, probe)
    value = json.loads(metadata.read_text())
    value.pop("checkpoint_sha256")
    metadata.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="must contain checkpoint_sha256"):
        adult_predictions(checkpoint, metadata, frame.iloc[:4])


def test_saved_housing_uncertainty_preserves_units_and_empty_cqr(tmp_path):
    catboost = pytest.importorskip("catboost")
    frame = pd.DataFrame({"a": np.linspace(0, 1, 40), "b": np.linspace(-1, 1, 40)})
    target = frame.a + frame.b
    point = catboost.CatBoostRegressor(
        iterations=8, depth=2, thread_count=2, verbose=False, allow_writing_files=False
    ).fit(frame, target)
    quantile = catboost.CatBoostRegressor(
        iterations=8,
        depth=2,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
        loss_function="MultiQuantile:alpha=0.05,0.95",
    ).fit(frame, target)
    path, quantile_path = tmp_path / "point.cbm", tmp_path / "quantile.cbm"
    point.save_model(str(path))
    quantile.save_model(str(quantile_path))
    metadata = tmp_path / "calibration.json"
    metadata.write_text(
        json.dumps(
            {
                "features": list(frame.columns),
                "absolute_residual_cutoff": 0.25,
                "cqr_cutoff": -10.0,
                "checkpoint_sha256": sha256(path.read_bytes()).hexdigest(),
                "quantile_checkpoint_sha256": sha256(quantile_path.read_bytes()).hexdigest(),
            }
        )
    )
    result = housing_predictions(path, frame.iloc[:4], metadata, quantile_path)
    np.testing.assert_allclose(result.median_house_value_100000_usd, point.predict(frame.iloc[:4]))
    np.testing.assert_allclose(result.split_conformal_upper - result.split_conformal_lower, 0.5)
    assert result.cqr_empty.all() and (result.cqr_lower > result.cqr_upper).all()
    with pytest.raises(ValueError, match="calibration metadata"):
        housing_predictions(path, frame.iloc[:4], quantile_checkpoint=quantile_path)
    missing = json.loads(metadata.read_text())
    missing.pop("quantile_checkpoint_sha256")
    metadata.write_text(json.dumps(missing))
    with pytest.raises(ValueError, match="must contain quantile_checkpoint_sha256"):
        housing_predictions(path, frame.iloc[:4], metadata, quantile_path)
    value = json.loads(metadata.read_text()) | {"checkpoint_sha256": "0" * 64}
    metadata.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Checkpoint differs"):
        housing_predictions(path, frame.iloc[:4], metadata)


def test_vision_inference_rejects_wrong_shape_before_loading_checkpoint():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="uint8 CIFAR"):
        cifar_probabilities("/missing/checkpoint.pt", np.zeros((1, 32, 32, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="Temperature"):
        cifar_probabilities("/missing/checkpoint.pt", np.zeros((1, 3, 32, 32), dtype=np.uint8), 0.0)
    with pytest.raises(ValueError, match="uint8 CIFAR"):
        cifar_probabilities("/missing/checkpoint.pt", np.zeros((0, 3, 32, 32), dtype=np.uint8))


def test_cifar_metadata_retains_empty_sets_and_binds_device_and_checkpoint(tmp_path, monkeypatch):
    from learning_lab import modern_inference

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixed-model")
    metadata = tmp_path / "calibration.json"
    record = {
        "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "class_names": modern_inference.CIFAR_CLASSES,
        "evaluation_profile": "cpu_float32",
        "temperature": 0.7,
        "conformal_score_cutoff": 0.05,
    }
    metadata.write_text(json.dumps(record))
    probability = np.array([[0.91] + [0.01] * 9, [0.99] + [0.01 / 9] * 9])
    monkeypatch.setattr(modern_inference, "cifar_probabilities", lambda *args: probability)
    result = modern_inference.cifar_predictions(checkpoint, None, metadata)
    np.testing.assert_array_equal(result.set_empty, [True, False])
    np.testing.assert_array_equal(result.set_size, [0, 1])
    np.testing.assert_array_equal(result.set_contains_airplane, [False, True])
    assert list(result.predicted_class) == ["airplane", "airplane"]
    with pytest.raises(ValueError, match="precision profile"):
        modern_inference.cifar_predictions(checkpoint, None, metadata, "cuda:0")
    checkpoint.write_bytes(b"another-model")
    with pytest.raises(ValueError, match="Checkpoint differs"):
        modern_inference.cifar_predictions(checkpoint, None, metadata)
