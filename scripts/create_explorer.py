"""Build a portable concept explorer from completed, hash-bound numerical evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from learning_lab.artifacts import write_json

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/explorer"
PHOTO_INDICES = [0, 1, 3, 5]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(study):
    root = ROOT / "results/reference" / study
    record = json.loads((root / "run.json").read_text())
    assert record["status"] == "complete", study
    return json.loads((root / "metrics.json").read_text())


def pixels(array):
    values = np.asarray(array)
    assert np.isfinite(values).all()
    return (
        values.astype(np.uint8)
        if np.issubdtype(values.dtype, np.integer)
        else np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
    )


def picture(array, name):
    values = np.asarray(array)
    if values.ndim == 3:
        values = values.transpose(1, 2, 0)
        if values.shape[-1] == 1:
            values = values[..., 0]
    assert np.isfinite(values).all()
    path = ASSETS / (name + ".png")
    Image.fromarray(pixels(values)).save(path)
    return "../" + str(path.relative_to(ROOT))


def probability_view(y, p):
    confidence = p.max(1)
    correct = p.argmax(1) == y
    bins = []
    for left in np.linspace(0, 1, 16)[:-1]:
        keep = (confidence >= left) & (
            confidence < left + 1 / 15 if left < 14 / 15 else confidence <= 1
        )
        if keep.any():
            bins.append(
                dict(
                    x=float(confidence[keep].mean()),
                    y=float(correct[keep].mean()),
                    n=int(keep.sum()),
                )
            )
    thresholds = []
    if p.shape[1] == 2:
        for t in np.linspace(0, 1, 101):
            positive = p[:, 1] >= t
            thresholds.append(
                dict(
                    threshold=float(t),
                    tp=int((positive & (y == 1)).sum()),
                    fp=int((positive & (y == 0)).sum()),
                    tn=int((~positive & (y == 0)).sum()),
                    fn=int((~positive & (y == 1)).sum()),
                )
            )
    return dict(reliability=bins, thresholds=thresholds)


def classical():
    classification, regression = [], []
    for path in sorted((ROOT / "results/reference/classical").glob("*/*/metrics.json")):
        record = json.loads(path.with_name("run.json").read_text())
        assert record["status"] == "complete" and not record["config"]["config"].get("quick")
        metrics = json.loads(path.read_text())
        assert sha(ROOT / metrics["evidence"]) == metrics["evidence_sha256"]
        with np.load(ROOT / metrics["evidence"], allow_pickle=False) as values:
            for row in metrics["rows"]:
                entry = {
                    key: value
                    for key, value in row.items()
                    if key not in ["search", "checkpoint_sha256"]
                }
                entry["metrics"] = "../" + str(path.relative_to(ROOT))
                entry["n"] = metrics["test_examples"]
                model = row["model"]
                if "protocol" in row:
                    entry["points"] = [
                        dict(x=float(y), y=float(p), lo=float(lo), hi=float(hi))
                        for y, p, lo, hi in zip(
                            values["y"][:200],
                            values[model][:200],
                            values[model + "_lower"][:200],
                            values[model + "_upper"][:200],
                        )
                    ]
                    regression.append(entry)
                else:
                    entry.update(probability_view(values["y"], values[model]))
                    classification.append(entry)
    assert len(classification) == 141 and len(regression) == 102
    return classification, regression


def restoration(allow_pending):
    entries = []
    references = {
        "colorization": [
            "baseline",
            "l1",
            "gan",
            "cvae",
            "diffusion",
            "ddcolor",
            "ct2",
            "bigcolor",
        ],
        "super_resolution": ["baseline", "l1", "gan", "cvae", "diffusion", "swinir", "realesrgan"],
    }
    truth_cache = {}
    sources = []
    for task, methods in references.items():
        for method in methods:
            group = (
                f"generative_{task}_{method}"
                if method in ["baseline", "l1", "gan", "cvae", "diffusion"]
                else "external_" + method
            )
            study = (
                f"generative_{task}/{method}"
                if group.startswith("generative")
                else "external_" + method
            )
            metrics = load(study)
            gallery_path = ROOT / "assets/source_data" / (group + "_gallery.npz")
            score_path = ROOT / "assets/source_data" / (group + "_per_image_scores.npz")
            sources.extend([gallery_path, score_path])
            with (
                np.load(gallery_path, allow_pickle=False) as gallery,
                np.load(score_path, allow_pickle=False) as scores,
            ):
                originals = gallery["original"]
                if task in truth_cache:
                    np.testing.assert_array_equal(pixels(originals), pixels(truth_cache[task]))
                truth_cache[task] = originals
                for row in metrics["rows"]:
                    key = f"{row['method']}_{row['seed']}_sample{row['sample']}"
                    if key not in gallery:
                        raise KeyError((gallery_path, key))
                    values = []
                    for i, index in enumerate(PHOTO_INDICES):
                        entry = dict(
                            task=task,
                            method=method,
                            seed=row["seed"],
                            draw=row["sample"],
                            image_index=index,
                            input=picture(gallery["input"][i], f"{task}_input_{index}"),
                            truth=picture(originals[i], f"{task}_truth_{index}"),
                            prediction=picture(
                                gallery[key][i],
                                f"{task}_{method}_{row['seed']}_{row['sample']}_{index}",
                            ),
                            aggregate={
                                name: value
                                for name, value in row.items()
                                if isinstance(value, dict) and "mean" in value
                            },
                            per_image={
                                name: float(scores[key + "_" + name][index])
                                for name in [
                                    "psnr",
                                    "ssim",
                                    "lpips",
                                    "data_consistency_l1",
                                    "delta_e76",
                                    "chroma",
                                ]
                                if key + "_" + name in scores
                            },
                            metrics="../results/reference/" + study + "/metrics.json",
                            source="../" + str(gallery_path.relative_to(ROOT)),
                            profile="Original matched 256px study; external pretraining differs by method",
                        )
                        values.append(entry)
                    entries.extend(values)
    configs = json.loads((ROOT / "configs/adversarial_variants.json").read_text())
    missing = []
    for family in ["adversarial_variants", "adversarial_endpoints"]:
        for task in configs["tasks"]:
            for method in configs["methods"]:
                for seed in configs["seeds"]:
                    path = ROOT / "results/reference" / family / task / method / str(seed)
                    if (
                        not path.joinpath("run.json").exists()
                        or json.loads(path.joinpath("run.json").read_text())["status"] != "complete"
                    ):
                        missing.append(str(path.relative_to(ROOT)))
                        continue
                    metrics = json.loads(path.joinpath("metrics.json").read_text())
                    stem = f"{task}_{method}_{seed}"
                    gp = ROOT / "assets/source_data" / family / (stem + "_gallery.npz")
                    sp = gp.with_name(stem + "_scores.npz")
                    sources.extend([gp, sp])
                    assert (
                        sha(gp) == metrics["numeric_sha256"]["_gallery"]
                        and sha(sp) == metrics["numeric_sha256"]["_scores"]
                    )
                    with (
                        np.load(gp, allow_pickle=False) as gallery,
                        np.load(sp, allow_pickle=False) as scores,
                    ):
                        np.testing.assert_array_equal(
                            pixels(gallery["original"]), pixels(truth_cache[task])
                        )
                        row = metrics["rows"][0]
                        alias = method + (
                            "_final_epoch" if family == "adversarial_endpoints" else ""
                        )
                        key = f"{alias}_{seed}_sample0"
                        for i, index in enumerate(PHOTO_INDICES):
                            entries.append(
                                dict(
                                    task=task,
                                    method=alias,
                                    seed=seed,
                                    draw=0,
                                    image_index=index,
                                    input=f"../assets/explorer/{task}_input_{index}.png",
                                    truth=f"../assets/explorer/{task}_truth_{index}.png",
                                    prediction=picture(
                                        gallery[key][i], f"{task}_{alias}_{seed}_0_{index}"
                                    ),
                                    aggregate={
                                        name: value
                                        for name, value in row.items()
                                        if isinstance(value, dict) and "mean" in value
                                    },
                                    per_image={
                                        name: float(scores[key + "_" + name][index])
                                        for name in [
                                            "psnr",
                                            "ssim",
                                            "lpips",
                                            "data_consistency_l1",
                                            "delta_e76",
                                            "chroma",
                                        ]
                                        if key + "_" + name in scores
                                    },
                                    metrics="../" + str(path.relative_to(ROOT)) + "/metrics.json",
                                    source="../" + str(gp.relative_to(ROOT)),
                                    profile=(
                                        "Fixed final epoch 20 EMA diagnostic; same warm start; deployment selection separate"
                                        if family == "adversarial_endpoints"
                                        else f"Controlled 20-epoch warm-start study; selected epoch {metrics['best_epoch']}; critic cost differs"
                                    ),
                                )
                            )
    assert allow_pending or not missing, missing
    metrics = load("natural_denoising")
    gp = ROOT / "assets/source_data/natural_denoising_gallery.npz"
    sources.append(gp)
    with np.load(gp, allow_pickle=False) as gallery:
        for row in metrics["rows"]:
            sigma = row["sigma"]
            method = row["model"]
            for i, index in enumerate(PHOTO_INDICES):
                entries.append(
                    dict(
                        task="natural_denoising",
                        method=f"{method}_sigma{sigma}",
                        seed=0,
                        draw=0,
                        image_index=index,
                        input=picture(
                            gallery[f"input_sigma{sigma}"][i], f"denoising_input_{sigma}_{index}"
                        ),
                        truth=picture(gallery["original"][i], f"denoising_truth_{index}"),
                        prediction=picture(
                            gallery[f"{method}_sigma{sigma}"][i],
                            f"denoising_{method}_{sigma}_{index}",
                        ),
                        aggregate={
                            name: value
                            for name, value in row.items()
                            if isinstance(value, dict) and "mean" in value
                        },
                        per_image={},
                        metrics="../results/reference/natural_denoising/metrics.json",
                        source="../assets/source_data/natural_denoising_gallery.npz",
                        profile=f"Frozen wavelet/SwinIR comparison; test noise sigma{sigma}/255; SwinIR trained for sigma25",
                    )
                )
    return entries, sources, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Development preview only; release validation requires complete fits",
    )
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    classification, regression = classical()
    images, sources, missing = restoration(args.allow_pending)
    clustering = load("unsupervised")
    cluster_data = {}
    for case in ["variable_density", "wine", "digits"]:
        path = ROOT / "assets/source_data/clustering" / (case + ".npz")
        sources.append(path)
        with np.load(path, allow_pickle=False) as values:
            cluster_data[case] = {key: values[key].tolist() for key in values.files}
    geometry = ROOT / "assets/source_data/probabilistic_geometry.npz"
    sources.append(geometry)
    for study in [
        "probabilistic",
        "generalization",
        "kernels",
        "vision",
        "modern_vision",
        "signals",
        "unsupervised",
    ]:
        sources.extend((ROOT / "results/reference" / study).glob("*.json"))
    sources.extend((ROOT / "results/reference/classical").glob("*/*/run.json"))
    sources.extend((ROOT / "results/reference/classical").glob("*/*/metrics.json"))
    sources.extend((ROOT / "assets/source_data/classical").glob("*.npz"))
    for row in images:
        sources.append((ROOT / "docs" / row["metrics"]).resolve())
    with np.load(geometry, allow_pickle=False) as values:
        order = np.random.default_rng(42).permutation(len(values["regression_x"]))
        bayes = dict(
            x=values["regression_x"][order].tolist(),
            y=values["regression_y"][order].tolist(),
            grid=values["grid"].tolist(),
            reference_mean=values["posterior_mean"].tolist(),
            reference_sd=values["posterior_sd"].tolist(),
            degree=5,
            prior_variance=1,
            noise_sd=0.15,
        )
    data = dict(
        classification=classification,
        regression=regression,
        restoration=images,
        clustering=dict(rows=clustering["rows"], data=cluster_data),
        bayes=bayes,
        generalization=load("generalization"),
        kernels=load("kernels"),
        representations=load("vision")["rows"] + load("modern_vision")["rows"],
        signals=load("signals"),
        completion=dict(status="development" if missing else "complete", missing=missing),
    )
    encoded = json.dumps(data, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    template = ROOT / "scripts/templates/ml_explorer.html"
    javascript = ROOT / "scripts/templates/ml_explorer.js"
    output = ROOT / "docs/explorer.html"
    output.write_text(
        template.read_text()
        .replace("__EVIDENCE__", encoded)
        .replace("__JAVASCRIPT__", javascript.read_text())
    )
    write_json(
        ROOT / "docs/explorer_manifest.json",
        dict(
            status="development" if missing else "complete",
            sections=8,
            classical_models=len(classification) + len(regression),
            restoration_views=len(images),
            clustering_fits=len(clustering["rows"]),
            missing_protocols=missing,
            html_sha256=sha(output),
            source_sha256={
                str(path.relative_to(ROOT)): sha(path)
                for path in [Path(__file__), template, javascript, *sources]
            },
            images_sha256={
                str(path.relative_to(ROOT)): sha(path) for path in sorted(ASSETS.glob("*.png"))
            },
            scope="Saved measured results with explicit numerical Bayesian illustration; PNG display quantization distinct from float-image metrics. No test-set model selection.",
        ),
    )
    print(
        "Generated explorer:",
        len(images),
        "restoration views;243 classical models;8 concepts",
        flush=True,
    )


if __name__ == "__main__":
    main()
