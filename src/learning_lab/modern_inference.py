"""Inference from native boosting and ConvNeXt artifacts, without training data."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from .data import CATEGORICAL, NUMERIC
from .uncertainty import binary_temperature

CIFAR_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def verify_checkpoint(path, metadata, key="checkpoint_sha256"):
    if key not in metadata:
        raise ValueError(f"Calibration metadata must contain {key}")
    if sha256(Path(path).read_bytes()).hexdigest() != metadata[key]:
        raise ValueError("Checkpoint differs from the saved calibration metadata")


def adult_predictions(checkpoint, metadata_path, frame):
    from catboost import CatBoostClassifier

    metadata = json.loads(Path(metadata_path).read_text())
    verify_checkpoint(checkpoint, metadata)
    if not np.isfinite(metadata["temperature"]) or metadata["temperature"] <= 0:
        raise ValueError("Temperature must be positive and finite")
    if not np.isfinite(metadata["threshold"]) or not 0 <= metadata["threshold"] <= 1:
        raise ValueError("Decision threshold must be in [0,1]")
    missing = set(metadata["features"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    frame = frame[metadata["features"]].copy()
    for name in NUMERIC:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if np.isinf(frame[name].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite input in {name}")
    for name in CATEGORICAL:
        frame[name] = frame[name].map(lambda v: v.strip() if isinstance(v, str) else v)
        frame[name] = frame[name].replace("?", np.nan).fillna("__missing__").astype(str)
    model = CatBoostClassifier()
    model.load_model(str(checkpoint))
    probability = binary_temperature(model.predict_proba(frame)[:, 1], metadata["temperature"])
    output = pd.DataFrame(
        {
            "probability_income_above_50k": probability,
            "decision": (probability >= metadata["threshold"]).astype(int),
        }
    )
    if "conformal_score_cutoff" in metadata:
        cutoff = metadata["conformal_score_cutoff"]
        if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
            raise ValueError("Conformal probability cutoff must be in [0,1]")
        output["set_contains_at_most_50k"] = probability <= cutoff
        output["set_contains_above_50k"] = 1 - probability <= cutoff
    return output


def housing_predictions(checkpoint, frame, metadata_path=None, quantile_checkpoint=None):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor()
    model.load_model(str(checkpoint))
    missing = set(model.feature_names_) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    features = frame[model.feature_names_].apply(pd.to_numeric, errors="raise")
    if np.isinf(features.to_numpy()).any():
        raise ValueError("Infinite numeric input")
    # California housing target units are $100,000; these are benchmark predictions.
    point = model.predict(features)
    output = pd.DataFrame({"median_house_value_100000_usd": point})
    if quantile_checkpoint and metadata_path is None:
        raise ValueError("Quantile intervals require saved calibration metadata")
    if metadata_path is not None:
        metadata = json.loads(Path(metadata_path).read_text())
        verify_checkpoint(checkpoint, metadata)
        if metadata["features"] != model.feature_names_:
            raise ValueError("Calibration feature schema differs from the fitted model")
        correction = metadata["absolute_residual_cutoff"]
        if not np.isfinite(correction) or correction < 0 or not np.isfinite(metadata["cqr_cutoff"]):
            raise ValueError("Invalid saved interval corrections")
        output["split_conformal_lower"] = point - correction
        output["split_conformal_upper"] = point + correction
        if quantile_checkpoint:
            verify_checkpoint(quantile_checkpoint, metadata, "quantile_checkpoint_sha256")
            quantile = CatBoostRegressor()
            quantile.load_model(str(quantile_checkpoint))
            if quantile.feature_names_ != model.feature_names_:
                raise ValueError("Quantile model feature schema differs")
            bounds = np.sort(quantile.predict(features), axis=1)
            correction = metadata["cqr_cutoff"]
            output["cqr_lower"] = bounds[:, 0] - correction
            output["cqr_upper"] = bounds[:, 1] + correction
            output["cqr_empty"] = output.cqr_lower > output.cqr_upper
    return output


def cifar_probabilities(checkpoint, images, temperature=1.0, device="cpu"):
    import torch
    from torchvision.models import convnext_tiny

    from .modern_vision import logits

    images = np.asarray(images)
    if (
        images.ndim != 4
        or images.shape[1:] != (3, 32, 32)
        or not len(images)
        or images.dtype != np.uint8
    ):
        raise ValueError("Expected uint8 CIFAR-like N x 3 x 32 x 32 images")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = convnext_tiny(weights=None)
    model.classifier[-1] = torch.nn.Linear(768, 10)
    model.load_state_dict(state["state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    rgb = torch.tensor(images, dtype=torch.float32, device=device) / 255
    prediction = logits(model, rgb, state["config"])
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Saved vision model produced nonfinite logits")
    # Match the reference experiment's float64 temperature/softmax convention.
    return softmax(np.asarray(prediction, dtype=np.float64) / temperature, axis=1)


def cifar_predictions(checkpoint, images, metadata_path, device="cpu"):
    metadata = json.loads(Path(metadata_path).read_text())
    if "checkpoint_sha256" not in metadata or metadata["class_names"] != CIFAR_CLASSES:
        raise ValueError("CIFAR calibration needs a checkpoint hash and the fixed class schema")
    verify_checkpoint(checkpoint, metadata)
    expected_profile = "cuda_bfloat16" if device.startswith("cuda") else "cpu_float32"
    if metadata["evaluation_profile"] != expected_profile:
        raise ValueError("Calibration precision profile differs from the requested device")
    cutoff = metadata["conformal_score_cutoff"]
    if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("Conformal probability cutoff must be in [0,1]")
    probability = cifar_probabilities(checkpoint, images, metadata["temperature"], device)
    output = pd.DataFrame(probability, columns=CIFAR_CLASSES)
    output["predicted_class"] = np.asarray(CIFAR_CLASSES)[probability.argmax(1)]
    sets = 1 - probability <= cutoff
    for column, name in enumerate(CIFAR_CLASSES):
        output["set_contains_" + name] = sets[:, column]
    output["set_size"] = sets.sum(1)
    output["set_empty"] = ~sets.any(1)
    return output
