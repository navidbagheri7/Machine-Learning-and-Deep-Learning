"""Check evidence completion, English notebooks, local links and release contents."""

import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDIES = [
    "classification",
    "regression",
    "probabilistic",
    "generalization",
    "kernels",
    "signals",
    "vision",
    "course_mixtures",
    "inverse",
    "denoising",
    "super_resolution",
    "colorization",
    "ddcolor",
    "modern_tabular",
    "modern_vision",
    "natural_denoising",
    "sr_shift",
    "external_ddcolor",
    "external_ct2",
    "external_bigcolor",
    "external_swinir",
    "external_realesrgan",
]
STUDIES += [
    f"generative_{task}/{method}"
    for task in ["colorization", "super_resolution"]
    for method in ["baseline", "l1", "gan", "cvae", "diffusion"]
]


def expected_config(study):
    if "/" not in study:
        return json.loads((ROOT / "configs" / (study + ".json")).read_text())
    group, method = study.split("/")
    name = "generative_colorization_cvae" if study == "generative_colorization/cvae" else group
    config = json.loads((ROOT / "configs" / (name + ".json")).read_text())
    device = "cuda:1" if method == "cvae" else "cuda:2" if method == "diffusion" else "cuda:0"
    return config | {"methods": [method], "output": f"results/reference/{study}", "device": device}


def disjoint_splits(record, names):
    groups = [set(record[name]) for name in names]
    assert sum(len(g) for g in groups) == len(set.union(*groups)), "Split overlap"


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or np.isfinite(value)


def main():
    issues = []
    results = {}
    locations = json.loads((ROOT / "assets" / "data_locations.json").read_text())["relocations"]
    for study in STUDIES:
        try:
            directory = ROOT / "results" / "reference" / study
            record = json.loads((directory / "run.json").read_text())
            metrics = json.loads((directory / "metrics.json").read_text())
            config = expected_config(study)
            assert record["status"] == "complete"
            assert not record["config"].get("quick", False)
            assert finite(metrics)
            assert all(
                (
                    locations.get(record["config"][k], record["config"][k])
                    if isinstance(record["config"][k], str)
                    else record["config"][k]
                )
                == v
                for k, v in config.items()
            )
            results[study] = {
                "status": record["status"],
                "elapsed_seconds": record["elapsed_seconds"],
            }
            budgets = {"classification": 3, "regression": 8, "kernels": 9, "signals": 50}
            if study in budgets:
                assert len(metrics["rows"]) == budgets[study] * len(config["seeds"])
                assert {r["seed"] for r in metrics["rows"]} == set(config["seeds"])
            if study in ["classification", "regression"]:
                for seed in config["seeds"]:
                    names = (
                        ["fit", "validation", "probability_calibration", "conformal_calibration"]
                        if study == "classification"
                        else ["fit", "validation", "calibration", "test"]
                    )
                    protocols = [None] if study == "classification" else ["random", "geographic"]
                    for protocol in protocols:
                        filename = (
                            f"split_{seed}.json"
                            if protocol is None
                            else f"split_{protocol}_{seed}.json"
                        )
                        disjoint_splits(json.loads((directory / filename).read_text()), names)
            if study == "signals":
                assert all(
                    r["blocks"] == config["test_blocks"]
                    and r["bits"] == config["test_blocks"] * 64 * 4
                    for r in metrics["rows"]
                )
            if study == "generalization":
                assert {r["seed"] for r in metrics["double_descent"]} == set(
                    range(config["repeats"])
                )
                assert metrics["implicit_bias"][-1]["step"] == config["gd_steps"]
            if study == "course_mixtures":
                cmm = metrics["categorical_mixture"]
                assert len(cmm["majority_modes"]) == 8
                assert len(cmm["majority_observed_fit_counts"]) == 8
                assert all(0 < n <= 3000 for n in cmm["majority_observed_fit_counts"])
                assert "masked fitting rows only" in cmm["majority_protocol"]
            if study in ["denoising", "super_resolution", "colorization", "ddcolor"]:
                with np.load(
                    ROOT / "assets" / "source_data" / f"{study}_per_image_scores.npz",
                    allow_pickle=False,
                ) as scores:
                    for row in metrics["rows"]:
                        values = scores[row["per_image_psnr_key"]]
                        assert len(values) == config["test_images"]
                        assert np.isfinite(values).all()
                        assert np.isclose(values.mean(), row["psnr"]["mean"], atol=1e-8)
                for curve in metrics.get("curves", []):
                    history = curve["history"]
                    assert len(history) == config["epochs"]
                    best = min(history, key=lambda r: r["validation_loss"])
                    assert curve["best_epoch"] == best["epoch"]
                    assert curve["best_validation_loss"] == best["validation_loss"]
            if study == "inverse":
                for row in metrics["supplied"]:
                    calibration, selection, test = [
                        set(row[k]) for k in ["calibration_ids", "selection_ids", "test_ids"]
                    ]
                    assert not (calibration & selection or calibration & test or selection & test)
                    assert calibration | selection == set(row["validation_ids"])
            if study.startswith(("generative_", "external_")):
                method = config.get("reference", config.get("methods", [None])[0])
                stochastic = method in ["cvae", "diffusion", "bigcolor"]
                seeds = config.get("seeds", [0]) if method != "baseline" else [0]
                count = config["samples"] if stochastic else 1
                assert len(metrics["rows"]) == len(seeds) * count
                assert {(r["seed"], r["sample"]) for r in metrics["rows"]} == {
                    (seed, sample) for seed in seeds for sample in range(count)
                }
                score_path = (
                    ROOT
                    / "assets/source_data"
                    / (study.replace("/", "_") + "_per_image_scores.npz")
                )
                with np.load(score_path, allow_pickle=False) as scores:
                    for row in metrics["rows"]:
                        assert row["metric_protocol_version"] == 3
                        for name in [
                            "psnr",
                            "ssim",
                            "lpips",
                            "data_consistency_l1",
                            "delta_e76",
                            "chroma",
                            "sample_rgb_l1_from_first",
                        ]:
                            if name not in row:
                                continue
                            values = scores[f"{method}_{row['seed']}_sample{row['sample']}_{name}"]
                            assert values.shape == (config["test_images"],)
                            assert np.isfinite(values).all()
                            assert np.isclose(values.mean(), row[name]["mean"], atol=1e-8)
                            if name == "sample_rgb_l1_from_first" and row["sample"] == 0:
                                assert (values == 0).all()
                with np.load(
                    ROOT / "assets/source_data" / (study.replace("/", "_") + "_gallery.npz"),
                    allow_pickle=False,
                ) as gallery:
                    assert all(
                        gallery[k].shape == (4, 3, 256, 256) and gallery[k].dtype == np.uint8
                        for k in gallery.files
                    )
                    assert set(gallery.files) == {"original", "input"} | {
                        f"{method}_{seed}_sample{sample}"
                        for seed in seeds
                        for sample in range(count)
                    }
                curves = metrics.get("curves", [])
                if study.startswith("generative_") and method != "baseline":
                    assert {c["seed"] for c in curves} == set(seeds)
                for curve in curves:
                    assert len(curve["history"]) == config["epochs"]
                    best = min(curve["history"], key=lambda r: r["validation_loss"])
                    assert best["epoch"] == curve["best_epoch"]
                    assert best["validation_loss"] == curve["selection_objective"]
                if study == "generative_colorization/cvae":
                    initialization = json.loads(
                        (directory / "initialization_audit.json").read_text()
                    )
                    assert initialization["status"] == "passed"
                    assert {r["seed"] for r in initialization["checks"]} == set(seeds)
                    assert all(r["semantic_tensors_verified"] > 0 for r in initialization["checks"])
            if study == "modern_vision":
                assert {r["seed"] for r in metrics["rows"]} == set(config["seeds"])
                calibration = json.loads((directory / "cifar_calibration_audit.json").read_text())
                assert calibration["status"] == "passed"
                expected_profiles = {
                    (s, p) for s in config["seeds"] for p in ["cuda_bfloat16", "cpu_float32"]
                }
                assert {(r["seed"], r["profile"]) for r in calibration["rows"]} == expected_profiles
                assert {
                    (r["seed"], r["profile"]) for r in calibration["commands"]
                } == expected_profiles
                assert all(
                    r["status"] == "passed" and r["images"] == 128 for r in calibration["commands"]
                )
                assert (
                    calibration["script_sha256"]
                    == hashlib.sha256(
                        (ROOT / "scripts/export_cifar_calibration.py").read_bytes()
                    ).hexdigest()
                )
                for name, digest in calibration["source_sha256"].items():
                    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
                for row in calibration["rows"]:
                    stem = f"{row['profile']}_{row['seed']}"
                    metadata = json.loads((directory / f"cifar_inference_{stem}.json").read_text())
                    assert metadata["evaluation_profile"] == row["profile"]
                    assert metadata["checkpoint_sha256"] == row["checkpoint_sha256"]
                    assert metadata["alpha"] == 0.1 and metadata["conformal_calibration_n"] == 1500
                    assert (
                        metadata["split_sha256"]
                        == hashlib.sha256(
                            (directory / f"split_{row['seed']}.json").read_bytes()
                        ).hexdigest()
                    )
                    with np.load(
                        ROOT / f"assets/source_data/modern_vision_cifar_calibration_{stem}.npz",
                        allow_pickle=False,
                    ) as saved:
                        assert saved["p"].shape == (10000, 10) and saved["cal_scores"].shape == (
                            1500,
                        )
                        q = np.sort(saved["cal_scores"])[int(np.ceil(1501 * 0.9)) - 1]
                        assert np.isclose(q, metadata["conformal_score_cutoff"], atol=1e-12)
                        sets = 1 - saved["p"] <= metadata["conformal_score_cutoff"]
                        size = sets.sum(1)
                        assert np.isclose(
                            sets[np.arange(10000), saved["y"]].mean(), row["set_coverage"]
                        )
                        assert np.isclose(size.mean(), row["set_size"])
                        assert np.isclose((size == 0).mean(), row["empty_set_rate"])
                for curve in metrics["curves"]:
                    assert len(curve["history"]) == config["epochs"]
                    assert (
                        min(curve["history"], key=lambda r: r["validation_nll"])["epoch"]
                        == curve["best_epoch"]
                    )
                for seed in config["seeds"]:
                    disjoint_splits(
                        json.loads((directory / f"split_{seed}.json").read_text()),
                        ["fit", "validation", "probability_calibration", "conformal_calibration"],
                    )
                    with np.load(
                        ROOT / "assets/source_data" / f"modern_vision_predictions_{seed}.npz",
                        allow_pickle=False,
                    ) as p:
                        assert p["p"].shape == (10000, 10)
                        assert np.isclose(
                            (p["p"].argmax(1) == p["y"]).mean(), metrics["rows"][seed]["accuracy"]
                        )
            if study == "modern_tabular":
                assert len(metrics["classification"]) == len(config["seeds"])
                assert len(metrics["regression"]) == len(config["seeds"]) * 2
                calibration_audit = json.loads(
                    (directory / "deployment_calibration_audit.json").read_text()
                )
                assert (
                    calibration_audit["status"] == "passed"
                    and len(calibration_audit["checks"]) == 9
                )
                for name, digest in calibration_audit["source_sha256"].items():
                    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
                for seed in config["seeds"]:
                    metadata = json.loads((directory / f"adult_inference_{seed}.json").read_text())
                    assert metadata["alpha"] == 0.1 and 0 <= metadata["conformal_score_cutoff"] <= 1
                    assert len(metadata["checkpoint_sha256"]) == 64
                    disjoint_splits(
                        json.loads((directory / f"classification_split_{seed}.json").read_text()),
                        ["fit", "validation", "probability_calibration", "conformal_calibration"],
                    )
                    for protocol in ["random", "geographic"]:
                        metadata = json.loads(
                            (directory / f"housing_inference_{protocol}_{seed}.json").read_text()
                        )
                        assert (
                            metadata["alpha"] == 0.1 and metadata["absolute_residual_cutoff"] >= 0
                        )
                        assert (
                            finite(metadata) and len(metadata["quantile_checkpoint_sha256"]) == 64
                        )
                        disjoint_splits(
                            json.loads(
                                (directory / f"regression_split_{protocol}_{seed}.json").read_text()
                            ),
                            ["fit", "validation", "calibration", "test"],
                        )
            if study == "natural_denoising":
                assert len(metrics["rows"]) == len(config["sigmas"]) * 3
                with np.load(
                    ROOT / "assets/source_data/natural_denoising_per_image_scores.npz",
                    allow_pickle=False,
                ) as scores:
                    for row in metrics["rows"]:
                        for name in ["psnr", "ssim", "lpips"]:
                            values = scores[f"{row['model']}_sigma{row['sigma']}_{name}"]
                            assert len(values) == config["test_images"]
                            assert np.isclose(values.mean(), row[name]["mean"], atol=1e-8)
            if study == "sr_shift":
                assert len(metrics["rows"]) == len(config["degradations"]) * 3
                with np.load(
                    ROOT / "assets/source_data/sr_shift_per_image_scores.npz", allow_pickle=False
                ) as scores:
                    for row in metrics["rows"]:
                        for name in ["psnr", "ssim", "lpips"]:
                            values = scores[f"{row['method']}_{row['degradation']}_{name}"]
                            assert len(values) == config["test_images"]
                            assert np.isclose(values.mean(), row[name]["mean"], atol=1e-8)
        except (OSError, ValueError, AssertionError, KeyError) as exc:
            issues.append(f"{study}: incomplete or inconsistent evidence ({exc})")
    notebooks = []
    try:
        summary = json.loads((ROOT / "assets/modern_summary.json").read_text())
        assert summary["status"] == "complete" and not summary["pending"]
        assert all(
            key in summary
            for key in [
                "modern_tabular_diagnostics",
                "modern_vision_diagnostics",
                "modern_vision_calibration",
            ]
        )
        for task in ["colorization", "super_resolution"]:
            comparisons = summary[task]
            methods = {row["method"] for row in comparisons["comparisons"]}
            assert {row["method"] for row in comparisons["resources"]} == methods
            latent = comparisons["latent_diagnostics"]
            assert {row["seed"] for row in latent} == {0, 1}
            metrics = json.loads(
                (ROOT / f"results/reference/generative_{task}/cvae/metrics.json").read_text()
            )
            for row in latent:
                curve = next(c for c in metrics["curves"] if c["seed"] == row["seed"])
                selected = next(r for r in curve["history"] if r["epoch"] == curve["best_epoch"])
                assert row["selected epoch"] == curve["best_epoch"]
                assert row["selected-epoch posterior KL"] == selected["train"]["kl"]
                assert row["final posterior KL"] == curve["history"][-1]["train"]["kl"]
                assert row["selected prior validation loss"] == curve["selection_objective"]
            for row in comparisons["image_distributions"]:
                method = row["method"]
                study = (
                    f"generative_{task}/{method}"
                    if method in ["baseline", "l1", "gan", "cvae", "diffusion"]
                    else f"external_{method}"
                )
                with np.load(
                    ROOT
                    / "assets/source_data"
                    / (study.replace("/", "_") + "_per_image_scores.npz"),
                    allow_pickle=False,
                ) as scores:
                    values = np.mean(
                        [scores[k] for k in scores.files if k.endswith("_psnr")], axis=0
                    )
                assert row["images"] == len(values) == 400
                assert np.isclose(row["PSNR median"], np.median(values))
                assert row["PSNR above 60dB"] == int((values > 60).sum())
        expected = json.loads((ROOT / "assets/external_restoration_manifest.json").read_text())
        packaged = json.loads(
            (ROOT / "src/learning_lab/resources/restoration_manifest.json").read_text()
        )
        assert expected == packaged
        assert json.loads((ROOT / "assets/pretrained_manifest.json").read_text()) == json.loads(
            (ROOT / "src/learning_lab/resources/ddcolor_manifest.json").read_text()
        )
        for filename in [
            "training_resume_audit.json",
            "evaluation_preservation_audit.json",
            "sr_evaluation_preservation_audit.json",
            "saved_prediction_audit.json",
            "integration_upgrade_audit.json",
            "ct2_device_audit.json",
            "foundation_inference_audit.json",
            "metric_reference_audit.json",
            "vae_precision_audit.json",
            "stress_evaluation_audit.json",
            "receiver_inference_audit.json",
        ]:
            assert json.loads((ROOT / "assets" / filename).read_text())["status"] == "passed"
        stress = json.loads((ROOT / "assets/stress_evaluation_audit.json").read_text())
        assert len(stress["checks"]) == 2 and len(stress["guards"]) == 6
        assert stress["original_run_records_preserved"]
        assert all(row["test_images"] == 400 for row in stress["checks"])
        assert all(row["rejected_before_clipping"] for row in stress["guards"])
        for name, digest in stress["source_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
        precision = json.loads((ROOT / "assets/vae_precision_audit.json").read_text())
        assert len(precision["checks"]) == 4 and len(precision["validation_photo_ids"]) == 128
        assert {(row["task"], row["seed"]) for row in precision["checks"]} == {
            (task, seed) for task in ["colorization", "super_resolution"] for seed in [0, 1]
        }
        for name, digest in precision["source_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
        for row in precision["checks"]:
            errors = np.abs(np.array(row["amp_kl"]) - row["float64_kl_on_same_encoder_outputs"])
            assert len(errors) == 16 and np.isclose(errors.max(), row["max_absolute_error"])
            assert row["max_absolute_error"] < 1e-5
        for task, count in [("colorization", 7), ("super_resolution", 6)]:
            audit = json.loads((ROOT / "assets" / f"{task}_inference_audit.json").read_text())
            assert audit["status"] == "passed" and audit["tasks"] == [task]
            assert len(audit["checks"]) == count
            assert (
                audit["script_sha256"]
                == hashlib.sha256(
                    (ROOT / "scripts/check_restoration_inference.py").read_bytes()
                ).hexdigest()
            )
            for name, expected_hash in audit["source_sha256"].items():
                assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected_hash
        foundation = json.loads((ROOT / "assets/foundation_inference_audit.json").read_text())
        assert len(foundation["checks"]) == 16
        assert foundation["invalid_noise_settings_rejected"]
        assert foundation["infinite_raw_field_rejected"]
        assert (
            foundation["inference_source_sha256"]
            == hashlib.sha256((ROOT / "scripts/restore_image.py").read_bytes()).hexdigest()
        )
        receiver = json.loads((ROOT / "assets/receiver_inference_audit.json").read_text())
        assert len(receiver["checks"]) == 60 and len(receiver["commands"]) == 12
        assert len(receiver["cpu_numerics"]) == 60
        assert all(
            row["blocks"] == 2000
            and row["bits"] == 512000
            and row["ber_absolute_difference"] < 1e-12
            and row["ser_absolute_difference"] < 1e-12
            for row in receiver["checks"]
        )
        assert all(
            row["different_symbol_classes"] / row["symbols"] <= 1e-3
            and row["maximum_probability_difference"] <= 0.02
            for row in receiver["cpu_numerics"]
        )
        for name, expected_hash in receiver["source_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected_hash
    except (OSError, AssertionError, KeyError) as exc:
        issues.append(f"Modern report or packaged provenance is incomplete ({exc})")
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        executed = 0
        for cell in nb.cells:
            if cell.cell_type == "markdown" and any(
                ord(char) < 32 and char != "\n" for char in cell.source
            ):
                issues.append(f"{path.name}: control character in Markdown/math source")
            if re.search(r"[\u0600-\u06ff]", cell.source):
                issues.append(f"{path.name}: untranslated source")
            if cell.cell_type == "code" and cell.source.strip():
                executed += 1
                if cell.execution_count is None:
                    issues.append(f"{path.name}: unexecuted code cell")
                for out in cell.get("outputs", []):
                    if out.output_type == "error":
                        issues.append(f"{path.name}: output error")
                    text = out.get("text", "") + " ".join(
                        str(v) for k, v in out.get("data", {}).items() if k.startswith("text/")
                    )
                    if re.search(r"[\u0600-\u06ff]", text):
                        issues.append(f"{path.name}: untranslated output")
        notebooks.append({"name": path.name, "executed_cells": executed})
    if len(notebooks) != 19:
        issues.append("Expected nineteen executed walkthroughs")
    public = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    public = sorted(set(public))
    for name in public:
        path = ROOT / name
        if name.startswith(("legacy/originals/", ".local/", "results/smoke/")):
            issues.append(f"Excluded local content is publishable: {name}")
        if (
            path.suffix in [".pt", ".pth", ".ckpt", ".cbm", ".rar", ".joblib", ".tar", ".gz"]
            or "checkpoints/" in name
        ):
            issues.append(f"Large model or data artifact: {name}")
        if path.stat().st_size > 10 * 1024 * 1024:
            issues.append(f"Unexpected file over 10 MB: {name}")
        if path.suffix in [".md", ".py", ".json", ".yml", ".toml", ".txt", ".cff"]:
            text = path.read_text()
            if re.search(r"[\u0600-\u06ff]", text):
                issues.append(f"Untranslated public text: {name}")
            token = re.search(r"(?:gh" + r"p_|github_" + r"pat_)[A-Za-z0-9_]{30,}", text)
            if token:
                issues.append(f"Credential-like content: {name}")
        if path.suffix == ".md":
            # Validate file targets, not remote websites or Markdown section anchors.
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
                target = target.split("#")[0].strip().split(' "')[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                if not (path.parent / unquote(target)).exists():
                    issues.append(f"Broken local link in {name}: {target}")
    for path in (ROOT / "assets" / "source_data").glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                assert data[key].dtype.kind != "O"
    manifest = json.loads((ROOT / "assets" / "project_data_manifest.json").read_text())
    train_ids = {r["id"] for r in manifest["natural_images"]["train"]}
    heldout_ids = [r["id"] for r in manifest["natural_images"]["val"]]
    assert len(train_ids) == 8000 and len(heldout_ids) == len(set(heldout_ids)) == 800
    assert not train_ids & set(heldout_ids)
    highres = json.loads((ROOT / "assets/restoration_data_manifest.json").read_text())
    # The larger benchmark preserves held-out ordering and the old training prefix.
    assert [
        r["id"] for r in highres["splits"]["validation"] + highres["splits"]["test"]
    ] == heldout_ids
    new_train = [r["id"] for r in highres["splits"]["train"]]
    assert len(new_train) == len(set(new_train)) == 16000
    assert not set(new_train) & set(heldout_ids)
    assert new_train[:8000] == [r["id"] for r in manifest["natural_images"]["train"]]
    source_audit = json.loads((ROOT / "assets/source_resolution_audit.json").read_text())
    assert source_audit["status"] == "passed"
    for row in source_audit["prepared_arrays"]:
        assert row["sha256"] == highres[row["split_file"].removesuffix(".npy") + "_sha256"]
    assert source_audit["splits"]["train"]["n"] == 16000
    assert source_audit["splits"]["test"]["n"] == 400
    credits = json.loads((ROOT / "assets" / "photo_credits.json").read_text())
    assert len(credits) == 4
    for credit in credits:
        assert credit["license"] == "CC BY 2.0"
        assert credit["coco_id"] == heldout_ids[400 + credit["test_index"]]
        assert credit["author"] and credit["source_url"].startswith("https://www.flickr.com/")
    external = json.loads((ROOT / "assets" / "pretrained_manifest.json").read_text())
    ddcolor = json.loads((ROOT / "results" / "reference" / "ddcolor" / "metrics.json").read_text())
    assert external["weight_sha256"] == ddcolor["weight_sha256"]
    source_hash = hashlib.sha256()
    for path in (
        sorted((ROOT / "src").rglob("*.py"))
        + sorted((ROOT / "src").rglob("*.json"))
        + sorted((ROOT / "configs").glob("*.json"))
    ):
        source_hash.update(str(path.relative_to(ROOT)).encode())
        source_hash.update(path.read_bytes())
    report = {
        "status": "passed" if not issues else "failed",
        "studies": results,
        "notebooks": notebooks,
        "publishable_files": len(public),
        "public_bytes": sum((ROOT / n).stat().st_size for n in public),
        "release_source_config_sha256": source_hash.hexdigest(),
        "issues": issues,
    }
    (ROOT / "docs" / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
