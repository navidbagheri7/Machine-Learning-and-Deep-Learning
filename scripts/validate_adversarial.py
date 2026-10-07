"""Validate full budgets, selection and image-wise GAN-variant evidence without weights."""

import hashlib
import json
from pathlib import Path

import numpy as np

from learning_lab.artifacts import write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records():
    config = json.loads(Path("configs/adversarial_variants.json").read_text())
    rows = []
    for task in config["tasks"]:
        for seed in config["seeds"]:
            originals, inputs, warm = None, None, None
            for method in config["methods"]:
                directory = Path(config["output"]) / task / method / str(seed)
                record = json.loads((directory / "run.json").read_text())
                metrics = json.loads((directory / "metrics.json").read_text())
                assert record["status"] == "complete"
                assert record["config"]["benchmark"] == config and not config.get("quick")
                assert (
                    record["config"]["task"],
                    record["config"]["method"],
                    record["config"]["seed"],
                ) == (task, method, seed)
                for name, value in record["fitting_source_sha256"].items():
                    assert digest(Path("src/learning_lab") / name) == value
                if warm is None:
                    warm = record["warm_start_sha256"]
                assert record["warm_start_sha256"] == warm
                history = metrics["history"]
                assert [r["epoch"] for r in history] == list(range(config["epochs"] + 1))
                validation = [r["validation_loss"] for r in history]
                assert np.isfinite(validation).all()
                assert metrics["best_epoch"] == int(np.argmin(validation))
                assert metrics["selection_objective"] == min(validation)
                updates = config["epochs"] * int(
                    np.ceil(config["train_images"] / config["batch_size"])
                )
                assert metrics["generator_updates"] == updates
                assert metrics["critic_updates"] == (
                    0 if method == "direct_finetune" else updates * config["critic_steps"]
                )
                stem = Path("assets/source_data/adversarial_variants") / f"{task}_{method}_{seed}"
                for suffix in ["_scores", "_gallery"]:
                    assert digest(str(stem) + suffix + ".npz") == metrics["numeric_sha256"][suffix]
                assert len(metrics["rows"]) == 1
                row = metrics["rows"][0]
                assert (
                    row["method"],
                    row["seed"],
                    row["sample"],
                    row["metric_protocol_version"],
                ) == (method, seed, 0, 3)
                with np.load(str(stem) + "_scores.npz", allow_pickle=False) as scores:
                    prefix = f"{method}_{seed}_sample0_"
                    for key in ["psnr", "ssim", "lpips", "data_consistency_l1"] + (
                        ["delta_e76", "chroma"] if task == "colorization" else []
                    ):
                        values = scores[prefix + key]
                        assert (
                            values.shape == (config["test_images"],) and np.isfinite(values).all()
                        )
                        assert np.isclose(values.mean(), row[key]["mean"], atol=1e-12, rtol=0)
                        lower, upper = row[key]["image_bootstrap_ci95"]
                        assert np.isfinite([lower, upper]).all() and lower <= upper
                with np.load(str(stem) + "_gallery.npz", allow_pickle=False) as gallery:
                    assert set(gallery.files) == {"original", "input", f"{method}_{seed}_sample0"}
                    for key in gallery.files:
                        assert gallery[key].shape == (4, 3, 256, 256)
                        assert np.isfinite(gallery[key]).all()
                        assert gallery[key].min() >= 0 and gallery[key].max() <= 1
                    if originals is None:
                        originals, inputs = gallery["original"].copy(), gallery["input"].copy()
                    np.testing.assert_array_equal(originals, gallery["original"])
                    np.testing.assert_array_equal(inputs, gallery["input"])
                rows.append(
                    dict(task=task, method=method, seed=seed, directory=str(directory), **metrics)
                )
    assert len(rows) == 20
    write_json(
        "docs/adversarial_validation.json",
        dict(
            status="passed",
            protocols=20,
            independently_recomputed_image_means=True,
            all_epochs_completed=True,
            selection_includes_initial_model=True,
            shared_seed_specific_warm_starts=True,
            source_and_numeric_hashes=True,
            private_checkpoints_required=False,
        ),
    )
    return config, rows


if __name__ == "__main__":
    records()
    print("Verified all 20 adversarial-variant protocols")
