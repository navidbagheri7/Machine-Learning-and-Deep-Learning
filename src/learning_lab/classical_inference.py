"""Prediction from source-bound classical checkpoints and saved calibration metadata."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax

from .classical import logits


def predict(checkpoint, metadata_path, frame):
    metadata = json.loads(Path(metadata_path).read_text())
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from its calibration metadata")
    if metadata["task"] not in ["classification", "regression"]:
        raise ValueError("Unknown prediction task")
    if not len(frame) or not frame.columns.is_unique:
        raise ValueError("Nonempty input with unique feature columns required")
    features = metadata["features"]
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input features: {sorted(missing)}")
    x = frame[features].copy()
    for column in features:
        if column in metadata["categorical"]:
            x[column] = x[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            x[column] = x[column].replace("?", np.nan)
        else:
            x[column] = pd.to_numeric(x[column], errors="raise")
            if np.isinf(x[column].to_numpy(dtype=float)).any():
                raise ValueError("Infinite feature values")
    saved = joblib.load(checkpoint)
    values = (
        x
        if metadata["dataset"] == "adult" or metadata["task"] == "regression"
        else np.ascontiguousarray(x.to_numpy(dtype=float))
    )
    if metadata["task"] == "classification":
        temperature = metadata["temperature"]
        cutoff = metadata["conformal_cutoff"]
        if not np.isfinite([temperature, cutoff]).all() or temperature <= 0 or not 0 <= cutoff <= 1:
            raise ValueError("Invalid calibration parameters")
        if saved["dataset"] != metadata["dataset"] or saved["temperature"] != temperature:
            raise ValueError("Calibration identity differs from the checkpoint")
        model = saved["model"]
        if list(model.classes_) != metadata["classes"]:
            raise ValueError("Class order differs from calibration")
        p = softmax(logits(model, values) / temperature, axis=1)
        if not np.isfinite(p).all():
            raise FloatingPointError("Nonfinite saved-model probabilities")
        output = pd.DataFrame(p, columns=[f"probability_class_{k}" for k in metadata["classes"]])
        output["predicted_class"] = np.asarray(metadata["classes"])[p.argmax(1)]
        output["predicted_label"] = np.asarray(metadata["class_labels"])[p.argmax(1)]
        sets = 1 - p <= cutoff
        for i, value in enumerate(metadata["classes"]):
            output[f"set_contains_class_{value}"] = sets[:, i]
        output["set_size"] = sets.sum(1)
        output["set_empty"] = ~sets.any(1)
    else:
        correction = metadata["absolute_residual_cutoff"]
        if not np.isfinite(correction) or correction < 0:
            raise ValueError("Invalid interval correction")
        point = saved.predict(values)
        if not np.isfinite(point).all():
            raise FloatingPointError("Nonfinite saved-model predictions")
        output = pd.DataFrame(
            dict(prediction_100000_usd=point, lower=point - correction, upper=point + correction)
        )
    return output
