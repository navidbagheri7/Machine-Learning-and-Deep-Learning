"""Validate published demo images and embedded measurements against public source arrays."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from learning_lab.artifacts import write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    page = root / "docs/explorer.html"
    manifest = json.loads((root / "docs/explorer_manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["missing_protocols"] == []
    assert digest(page) == manifest["html_sha256"]
    for field in ["source_sha256", "images_sha256"]:
        for name, value in manifest[field].items():
            assert digest(root / name) == value, name
    data = json.loads(
        re.search(
            r'<script id="evidence" type="application/json">(.*?)</script>', page.read_text(), re.S
        ).group(1)
    )
    assert data["completion"]["status"] == "complete"
    assert len(data["classification"]) == 141 and len(data["regression"]) == 102
    assert len(data["clustering"]["rows"]) == 63
    for entry in data["classification"] + data["regression"]:
        metrics = json.loads((page.parent / entry["metrics"]).read_text())
        row = next(
            r
            for r in metrics["rows"]
            if r["model"] == entry["model"] and r["seed"] == entry["seed"]
        )
        for key, value in row.items():
            if key not in ["search", "checkpoint_sha256"]:
                assert entry[key] == value, (entry["model"], key)
        with np.load(root / metrics["evidence"], allow_pickle=False) as values:
            if "points" in entry:
                points = entry["points"]
                assert len(points) == min(200, len(values["y"]))
                for i, point in enumerate(points):
                    assert [point[k] for k in ["x", "y", "lo", "hi"]] == [
                        values[k][i]
                        for k in [
                            "y",
                            entry["model"],
                            entry["model"] + "_lower",
                            entry["model"] + "_upper",
                        ]
                    ]
            else:
                assert sum(r["n"] for r in entry["reliability"]) == len(values["y"])
                for threshold in entry["thresholds"]:
                    p, y = values[entry["model"]][:, 1] >= threshold["threshold"], values["y"]
                    assert threshold["tp"] == int((p & (y == 1)).sum())
                    assert threshold["fp"] == int((p & (y == 0)).sum())
                    assert threshold["tn"] == int((~p & (y == 0)).sum())
                    assert threshold["fn"] == int((~p & (y == 1)).sum())
    image_checks = 0
    seen = set()
    for entry in data["restoration"]:
        identity = tuple(entry[k] for k in ["task", "method", "seed", "draw", "image_index"])
        assert identity not in seen
        seen.add(identity)
        source = (page.parent / entry["source"]).resolve()
        metrics = json.loads((page.parent / entry["metrics"]).read_text())
        if entry["task"] == "natural_denoising":
            method, sigma = entry["method"].rsplit("_sigma", 1)
            row = next(
                r for r in metrics["rows"] if r["model"] == method and str(r["sigma"]) == sigma
            )
            key, inp = entry["method"], "input_sigma" + sigma
        else:
            row = next(
                r
                for r in metrics["rows"]
                if r["method"] == entry["method"]
                and r["seed"] == entry["seed"]
                and r["sample"] == entry["draw"]
            )
            key = f"{entry['method']}_{entry['seed']}_sample{entry['draw']}"
            inp = "input"
        assert entry["aggregate"] == {
            k: v for k, v in row.items() if isinstance(v, dict) and "mean" in v
        }
        with np.load(source, allow_pickle=False) as values:
            for field, array_key in [("prediction", key), ("truth", "original"), ("input", inp)]:
                arr = values[array_key][[0, 1, 3, 5].index(entry["image_index"])].transpose(1, 2, 0)
                expected = (
                    arr.astype(np.uint8)
                    if arr.dtype.kind in "ui"
                    else (arr.clip(0, 1) * 255).round().astype(np.uint8)
                )
                with Image.open(page.parent / entry[field]) as image:
                    np.testing.assert_array_equal(np.asarray(image), expected)
                image_checks += 1
    browser = json.loads((root / "docs/explorer_browser_validation.json").read_text())
    assert browser["status"] == "passed" and browser["javascript_errors"] == []
    assert browser["sections"] == 8
    write_json(
        root / "docs/explorer_validation.json",
        dict(
            status="passed",
            source_bound=True,
            measured_models=243,
            clustering_fits=63,
            restoration_views=len(seen),
            image_array_checks=image_checks,
            browser_presets=browser["checked_presets"],
            html_sha256=digest(page),
        ),
    )
    print("Verified demo source, every prediction row and", image_checks, "PNG/array pairs")


if __name__ == "__main__":
    main()
