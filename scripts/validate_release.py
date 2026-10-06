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
]


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
            config = json.loads((ROOT / "configs" / (study + ".json")).read_text())
            assert record["status"] == "complete"
            assert not record["config"].get("quick", False)
            assert finite(metrics)
            assert all(
                (
                    locations.get(record["config"][k], record["config"][k])
                    if k == "data_root"
                    else record["config"][k]
                )
                == v
                for k, v in config.items()
            )
            results[study] = {
                "status": record["status"],
                "elapsed_seconds": record["elapsed_seconds"],
            }
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
        except (OSError, ValueError, AssertionError, KeyError) as exc:
            issues.append(f"{study}: incomplete or inconsistent evidence ({exc})")
    notebooks = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        executed = 0
        for cell in nb.cells:
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
    if len(notebooks) != len(STUDIES):
        issues.append("Expected thirteen executed walkthroughs")
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
        if path.suffix in [".pt", ".joblib", ".tar", ".gz"] or "checkpoints/" in name:
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
    for path in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "configs").glob("*.json")):
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
