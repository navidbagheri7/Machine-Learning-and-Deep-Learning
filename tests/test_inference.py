import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from learning_lab.data import CATEGORICAL, NUMERIC
from learning_lab.tabular import _adult_preprocessor

spec = importlib.util.spec_from_file_location(
    "predict_script", Path(__file__).resolve().parents[1] / "scripts" / "predict.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_inference_schema_unknown_categories_and_calibrated_probabilities():
    frame = pd.DataFrame({name: np.arange(50, dtype=float) for name in NUMERIC})
    for name in CATEGORICAL:
        frame[name] = np.where(np.arange(50) % 2, "group_a", "group_b")
    y = np.arange(50) % 2
    pipe = make_pipeline(_adult_preprocessor(), LogisticRegression()).fit(frame, y)
    bundle = {
        "features": list(frame.columns),
        "pipeline": pipe,
        "temperature": 2.0,
        "threshold": 0.3,
    }
    unknown = frame.iloc[:5].copy()
    unknown.loc[:, "occupation"] = "new_unseen_category"
    values = module.predict(bundle, unknown)
    assert np.isfinite(values.to_numpy()).all()
    assert values.probability_income_above_50k.between(0, 1).all()
    assert np.array_equal(values.decision, (values.probability_income_above_50k >= 0.3).astype(int))
    with pytest.raises(ValueError, match="Missing required"):
        module.predict(bundle, unknown.drop(columns="age"))
    unknown["age"] = "invalid_number"
    with pytest.raises(ValueError):
        module.predict(bundle, unknown)
