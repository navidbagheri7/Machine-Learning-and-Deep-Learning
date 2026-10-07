"""Verify evaluation and resume guards on a completed integration checkpoint.

No model fitting occurs. This checks exact scoring reproduction and preservation
of the original fitting record and histories. Use a small real-data integration.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from learning_lab import generative  # noqa: E402
from learning_lab.artifacts import write_json  # noqa: E402
from learning_lab.generative import experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(".local/evaluation_check.json"))
    args = parser.parse_args()
    record_path = args.run / "run.json"
    original_record = record_path.read_bytes()
    config = json.loads(original_record)["config"]
    if not config.get("quick") or config["test_images"] > 32:
        parser.error("Use a completed --quick integration with at most32 test images")
    original_metrics = json.loads((args.run / "metrics.json").read_text())
    with np.load(args.run / "per_image_scores.npz", allow_pickle=False) as saved:
        expected = {k: saved[k].copy() for k in saved.files}
    try:
        experiment(config | {"learning_rate": config["learning_rate"] * 2})
    except ValueError as error:
        assert "Resume configuration differs" in str(error)
    else:
        raise AssertionError("A mismatched resume was not rejected")
    assert record_path.read_bytes() == original_record
    experiment(config | {"evaluate_only": True})
    assert record_path.read_bytes() == original_record
    actual_metrics = json.loads((args.run / "metrics.json").read_text())
    assert actual_metrics["curves"] == original_metrics["curves"]
    build_model = generative.build_model

    def broken_model(method, task, *arguments, **keywords):
        model = build_model(method, task, *arguments, **keywords)

        def nonfinite(condition, *unused, **extra):
            channels = 2 if task == "colorization" else 3
            field = condition.new_full((len(condition), channels, *condition.shape[-2:]), torch.inf)
            return (field, condition.new_zeros(())) if method == "cvae" else field

        if method == "diffusion":
            model.sample = nonfinite
        else:
            model.forward = nonfinite
        return model

    generative.build_model = broken_model
    try:
        experiment(config | {"evaluate_only": True})
    except FloatingPointError as error:
        assert "nonfinite restoration field" in str(error)
    else:
        raise AssertionError("Nonfinite inference was not rejected before gamut clipping")
    finally:
        generative.build_model = build_model
    assert record_path.read_bytes() == original_record
    assert json.loads((args.run / "metrics.json").read_text()) == actual_metrics
    with np.load(args.run / "per_image_scores.npz", allow_pickle=False) as actual:
        assert set(actual.files) == set(expected)
        for key, value in expected.items():
            np.testing.assert_array_equal(actual[key], value)
    write_json(
        args.output,
        {
            "status": "passed",
            "task": config["task"],
            "methods": config["methods"],
            "test_images": config["test_images"],
            "score_arrays_verified": len(expected),
            "original_fitting_record_preserved": True,
            "selected_histories_preserved": True,
            "mismatched_resume_record_preserved": True,
            "nonfinite_re_evaluation_preserved_evidence": True,
            "nonfinite_guard_before_gamut_clipping": True,
            "evaluation_source_sha256": hashlib.sha256(
                (ROOT / "src/learning_lab/generative.py").read_bytes()
            ).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "protocol": "Exact score-array equality on a completed quick integration, not a quality benchmark. Evaluation changes neither fitting provenance nor selected curves. A conflicting resume fails before overwriting the fitting record.",
        },
    )
    print("Evaluation reproduction and training-record preservation verified.")


if __name__ == "__main__":
    main()
