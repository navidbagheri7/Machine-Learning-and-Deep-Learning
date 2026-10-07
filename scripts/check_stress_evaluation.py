"""Reproduce complete noise/degradation studies and reject invalid outputs before clipping."""

import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from learning_lab import natural_denoising, restoration_shift
from learning_lab.artifacts import write_json

ROOT = Path(__file__).resolve().parents[1]


class ZeroMetric(torch.nn.Module):
    def forward(self, target, prediction):
        return target.new_zeros((len(target), 1, 1, 1))


class InvalidOutput(torch.nn.Module):
    def __init__(self, value, upscale=1):
        super().__init__()
        self.value, self.upscale = value, upscale

    def forward(self, image):
        return image.new_full(
            (len(image), 3, image.shape[-2] * self.upscale, image.shape[-1] * self.upscale),
            self.value,
        )

    super_resolve = forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/reference")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--work-dir", type=Path, default=ROOT / ".local/stress_reproduction")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/stress_evaluation_audit.json")
    args = parser.parse_args()
    checks, guards = [], []
    for study, module in [
        ("natural_denoising", natural_denoising),
        ("sr_shift", restoration_shift),
    ]:
        original = args.results_root / study
        record_bytes = (original / "run.json").read_bytes()
        config = json.loads(record_bytes)["config"] | {
            "data_root": str(args.data_root),
            "device": args.device,
            "output": str(args.work_dir / study),
        }
        print("Reproducing", study, flush=True)
        module.experiment(config)
        errors = {}
        for filename in ["per_image_scores.npz", "gallery.npz"]:
            with (
                np.load(original / filename, allow_pickle=False) as expected,
                np.load(Path(config["output"]) / filename, allow_pickle=False) as actual,
            ):
                assert set(expected.files) == set(actual.files)
                for key in expected.files:
                    np.testing.assert_allclose(actual[key], expected[key], atol=1e-5, rtol=1e-6)
                errors[filename] = max(
                    float(np.max(np.abs(actual[key] - expected[key]))) for key in expected.files
                )
        assert (original / "run.json").read_bytes() == record_bytes
        checks.append({"study": study, "test_images": config["test_images"], "errors": errors})
        for name, value in [
            ("positive_infinity", torch.inf),
            ("negative_infinity", -torch.inf),
            ("nan", torch.nan),
        ]:
            failed_output = args.work_dir / f"{study}_{name}"
            tiny = config | {"test_images": 8, "output": str(failed_output)}
            with patch.object(module.lpips, "LPIPS", return_value=ZeroMetric()):
                if study == "natural_denoising":
                    model_patch = patch.object(
                        module, "load_model", return_value=InvalidOutput(value)
                    )
                    second_patch = nullcontext()
                else:
                    model_patch = patch.object(
                        module, "SwinIRReference", return_value=InvalidOutput(value, 4)
                    )
                    second_patch = patch.object(
                        module, "RealESRGANReference", return_value=InvalidOutput(value, 4)
                    )
                with model_patch, second_patch:
                    try:
                        module.experiment(tiny)
                    except FloatingPointError as error:
                        assert "nonfinite" in str(error)
                    else:
                        raise AssertionError("Invalid raw output was hidden by clipping")
            assert not (failed_output / "metrics.json").exists()
            assert not (failed_output / "per_image_scores.npz").exists()
            assert (original / "run.json").read_bytes() == record_bytes
            guards.append({"study": study, "case": name, "rejected_before_clipping": True})
        print("Passed", study, errors, flush=True)
    write_json(
        args.output,
        {
            "status": "passed",
            "checks": checks,
            "guards": guards,
            "original_run_records_preserved": True,
            "source_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in [
                    "scripts/check_stress_evaluation.py",
                    "src/learning_lab/natural_denoising.py",
                    "src/learning_lab/restoration_shift.py",
                ]
            },
            "protocol": "Separate full re-evaluation of all 400 fixed test images at three noise levels and three SR degradations, using original configs, data and frozen weights. Every saved score and gallery array is compared with the original numerical evidence, allowing atol1e-5/rtol1e-6 across L40S devices. Three injected invalid raw outputs per study use eight real images and a dummy metric only for failure-path testing, never quality evaluation. Invalid outputs must raise before clipping and publishing score files; original evidence is untouched.",
        },
    )
    print("Full stress-study reproduction and failure-path checks passed.", flush=True)


if __name__ == "__main__":
    main()
