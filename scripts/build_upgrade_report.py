"""Regenerate modern-method comparisons from saved, auditable experiment evidence.

Scores retain full precision. Gallery-only arrays are quantized to uint8 to keep
the public repository small; these images never enter metric computation.
"""

import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS, ASSETS = ROOT / "results/reference", ROOT / "assets"
METHODS = ["baseline", "l1", "gan", "cvae", "diffusion"]
EXTERNAL = {
    "colorization": ["ddcolor", "ct2", "bigcolor"],
    "super_resolution": ["swinir", "realesrgan"],
}
METRICS = ["psnr", "ssim", "lpips", "data_consistency_l1", "delta_e76", "chroma"]
LABELS = {
    "baseline": "Baseline",
    "l1": "L1 + features",
    "gan": "cGAN",
    "cvae": "cVAE",
    "diffusion": "Diffusion",
    "ddcolor": "DDColor",
    "ct2": "CT2",
    "bigcolor": "BigColor",
    "swinir": "SwinIR",
    "realesrgan": "Real-ESRGAN",
}


def table(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return "No completed evidence yet."

    def format_value(value):
        if value is None or isinstance(value, float) and np.isnan(value):
            return "—"
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    cells = [[str(v) for v in frame.columns]] + [
        [format_value(v) for v in row] for row in frame.itertuples(index=False, name=None)
    ]
    lines = ["| " + " | ".join(cells[0]) + " |", "| " + " | ".join(["---"] * len(cells[0])) + " |"]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in cells[1:]])


def save(fig, name):
    fig.savefig(ASSETS / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(ASSETS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def evidence(study):
    path = RESULTS / study
    if not (path / "metrics.json").exists():
        return None
    if json.loads((path / "run.json").read_text())["status"] != "complete":
        return None
    return json.loads((path / "metrics.json").read_text())


def arrays(study, filename, gallery=False):
    target = ASSETS / "source_data" / f"{study.replace('/', '_')}_{filename}"
    source = RESULTS / study / filename
    if source.exists():
        if gallery:
            with np.load(source, allow_pickle=False) as data:
                np.savez_compressed(
                    target,
                    **{k: (data[k].clip(0, 1) * 255).round().astype(np.uint8) for k in data.files},
                )
        else:
            shutil.copyfile(source, target)
    return np.load(target, allow_pickle=False)


def gallery(study, method):
    with arrays(study, "gallery.npz", gallery=True) as data:
        predictions = sorted(k for k in data.files if k.startswith(method + "_"))
        keys = ["original", "input"] + predictions
        # Keep every sample visible while avoiding unreadably wide composite figures.
        for seed in sorted({k.split("_sample")[0] for k in predictions}):
            selected = keys[:2] + [k for k in predictions if k.startswith(seed + "_sample")]
            fig, axes = plt.subplots(
                4, len(selected), figsize=(2.0 * len(selected), 8), squeeze=False
            )
            for i in range(4):
                for j, key in enumerate(selected):
                    axes[i, j].imshow(data[key][i].transpose(1, 2, 0))
                    axes[i, j].axis("off")
                    if i == 0:
                        title = (
                            "Target RGB"
                            if key == "original"
                            else "Observed input"
                            if key == "input"
                            else f"{LABELS[method]} · seed {seed.rsplit('_', 1)[-1]}\nDraw {key.split('_sample')[-1]}"
                        )
                        axes[i, j].set_title(title, fontsize=9)
            fig.suptitle("Fixed held-out photographs · all samples shown · CC BY 2.0")
            save(fig, study.replace("/", "_") + "_" + seed + "_gallery")


def image_interval(values):
    values = np.asarray(values)
    rng = np.random.default_rng(704)
    means = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
    return float(values.mean()), *np.quantile(means, [0.025, 0.975]).tolist()


def diagnostics(study):
    path = RESULTS / study / "diagnostics.json"
    return json.loads(path.read_text()) if path.exists() else None


def reliability_plot(ax, rows, confidence="confidence", accuracy="accuracy", label=None):
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.plot([r[confidence] for r in rows], [r[accuracy] for r in rows], "o-", label=label)
    ax.set(
        xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed frequency"
    )
    if label:
        ax.legend(fontsize=8)


def restoration_scatter(ax, rows):
    offsets = {"l1": (4, -14), "cvae": (4, 8), "gan": (4, -14)}
    for row in rows:
        ax.scatter(
            row["psnr"],
            row["lpips"],
            marker="s" if row["pipeline"] == "external" else "o",
            s=50,
        )
        ax.annotate(
            LABELS[row["method"]],
            (row["psnr"], row["lpips"]),
            xytext=offsets.get(row["method"], (4, 4)),
            textcoords="offset points",
            fontsize=8,
        )
    ax.margins(x=0.16, y=0.12)
    ax.grid(alpha=0.2)


def restoration_panel(task, rows):
    """Show every pipeline on the same four fixed images, with seed0/draw0 fixed."""
    methods = [row["method"] for row in rows if row["method"] != "baseline"]
    pictures = {}
    for method in ["baseline", *methods]:
        study = f"generative_{task}/{method}" if method in METHODS else f"external_{method}"
        with np.load(
            ASSETS / "source_data" / (study.replace("/", "_") + "_gallery.npz"),
            allow_pickle=False,
        ) as data:
            pictures[method] = data[f"{method}_0_sample0"]
            if method == "baseline":
                original, observed = data["original"], data["input"]
    labels = ["Target RGB", "Observed input"] + [
        LABELS[method] + "\n" + next(r["pipeline"] for r in rows if r["method"] == method)
        for method in methods
    ]
    values = [original, observed] + [pictures[method] for method in methods]
    fig, axes = plt.subplots(4, len(values), figsize=(2 * len(values), 8), squeeze=False)
    for column, (label, value) in enumerate(zip(labels, values)):
        for index in range(4):
            axes[index, column].imshow(value[index].transpose(1, 2, 0))
            axes[index, column].axis("off")
        axes[0, column].set_title(label, fontsize=9)
    fig.suptitle(
        task.replace("_", " ").title()
        + " · same four fixed test photographs · seed0/draw0 · CC BY 2.0"
    )
    save(fig, task + "_comparison_gallery")


def restoration(task):
    comparisons, paired, paired_direct, distributions, missing = [], [], [], [], []
    budgets, latent_diagnostics = [], []
    baseline, direct = None, None
    records = [(f"generative_{task}/{m}", m, "local") for m in METHODS] + [
        ("external_" + m, m, "external") for m in EXTERNAL[task]
    ]
    for study, method, exposure in records:
        record = evidence(study)
        if record is None:
            missing.append(study)
            continue
        rows = record["rows"]
        curves = record.get("curves", [])
        config = json.loads((RESULTS / study / "run.json").read_text())["config"]
        parameters = curves[0]["parameter_count"] if curves else record.get("parameter_count", 0)
        budgets.append(
            {
                "method": method,
                "inference parameters (M)": parameters / 1e6,
                "epochs per fitted seed": config.get("epochs", 0) if curves else 0,
                "local training images": config.get("train_images", 0) if curves else 0,
                "selection images": config.get("validation_images", 0) if curves else 0,
                "training crop": config.get("crop") if curves else None,
                "summed epoch hours": sum(
                    r["elapsed_seconds"] for curve in curves for r in curve["history"]
                )
                / 3600
                if curves
                else None,
                "seconds per 400 images including metrics": float(
                    np.mean([r["inference_seconds_including_metrics"] for r in rows])
                ),
            }
        )
        row = {
            "method": method,
            "pipeline": exposure,
            "fitted checkpoints": 0 if method == "baseline" else len({r["seed"] for r in rows}),
            "samples per seed": len(rows) // len({r["seed"] for r in rows}),
        }
        # Averaging all draws and then seeds gives each training seed equal weight.
        for name in METRICS:
            if name in rows[0]:
                grouped = [
                    np.mean([r[name]["mean"] for r in rows if r["seed"] == seed])
                    for seed in sorted({r["seed"] for r in rows})
                ]
                row[name] = float(np.mean(grouped))
                row[name + " seed SD"] = (
                    float(np.std(grouped, ddof=1)) if len(grouped) > 1 else None
                )
        comparisons.append(row)
        if "sample_rgb_l1_from_first" in rows[0]:
            row["RGB draw dispersion"] = float(
                np.mean([r["sample_rgb_l1_from_first"]["mean"] for r in rows if r["sample"] > 0])
            )
        with arrays(study, "per_image_scores.npz") as data:
            per_image = {
                name: np.mean(
                    [data[f"{method}_{r['seed']}_sample{r['sample']}_{name}"] for r in rows], axis=0
                )
                for name in ["psnr", "lpips"]
            }
        distributions.append(
            {
                "method": method,
                "images": len(per_image["psnr"]),
                "PSNR median": float(np.median(per_image["psnr"])),
                "PSNR Q25": float(np.quantile(per_image["psnr"], 0.25)),
                "PSNR Q75": float(np.quantile(per_image["psnr"], 0.75)),
                "PSNR above 60dB": int((per_image["psnr"] > 60).sum()),
            }
        )
        if method == "baseline":
            baseline = per_image
        elif baseline is not None:
            for name in ["psnr", "lpips"]:
                mean, low, high = image_interval(per_image[name] - baseline[name])
                paired.append(
                    {
                        "method": method,
                        "metric": name,
                        "difference vs baseline": mean,
                        "image CI lower": low,
                        "image CI upper": high,
                    }
                )
        if method == "l1":
            direct = per_image
        elif direct is not None and method != "baseline":
            for name in ["psnr", "lpips"]:
                mean, low, high = image_interval(per_image[name] - direct[name])
                paired_direct.append(
                    {
                        "method": method,
                        "metric": name,
                        "difference vs L1 + features": mean,
                        "image CI lower": low,
                        "image CI upper": high,
                    }
                )
        gallery(study, method)
        for curve in record.get("curves", []):
            history = curve["history"]
            if method == "cvae":
                selected = next(r for r in history if r["epoch"] == curve["best_epoch"])
                latent_diagnostics.append(
                    {
                        "seed": curve["seed"],
                        "selected epoch": curve["best_epoch"],
                        "selected-epoch posterior KL": selected["train"]["kl"],
                        "final posterior KL": history[-1]["train"]["kl"],
                        "selected prior validation loss": curve["selection_objective"],
                        "test RGB draw dispersion": float(
                            np.mean(
                                [
                                    r["sample_rgb_l1_from_first"]["mean"]
                                    for r in rows
                                    if r["seed"] == curve["seed"] and r["sample"] > 0
                                ]
                            )
                        ),
                    }
                )
                fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
                epochs = [r["epoch"] for r in history]
                axes[0].plot(epochs, [max(r["train"]["kl"], 1e-12) for r in history])
                axes[0].set(yscale="log", ylabel="KL, summed over latent dimensions")
                axes[1].plot(epochs, [r["validation_loss"] for r in history])
                axes[1].set(ylabel="Prior-based selection loss")
                for axis in axes:
                    axis.axvline(curve["best_epoch"], color="gray", linestyle="--")
                    axis.set(xlabel="Epoch")
                    axis.grid(alpha=0.2)
                fig.suptitle(f"{task.replace('_', ' ').title()} CVAE · seed {curve['seed']}")
                save(fig, study.replace("/", "_") + f"_seed{curve['seed']}_latent")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(
                [r["epoch"] for r in history],
                [r["train"]["objective"] for r in history],
                label="Training objective",
            )
            ax.plot(
                [r["epoch"] for r in history],
                [r["validation_loss"] for r in history],
                label="Validation selection objective",
            )
            ax.axvline(curve["best_epoch"], color="gray", linestyle="--", label="Selected epoch")
            ax.set(
                title=f"{task}: {method}, seed {curve['seed']}",
                xlabel="Epoch",
                ylabel="Loss (objectives can differ)",
            )
            ax.legend(fontsize=8)
            save(fig, study.replace("/", "_") + f"_seed{curve['seed']}_training")
    return comparisons, paired, paired_direct, distributions, budgets, latent_diagnostics, missing


def comparison_table(rows):
    output = []
    for row in rows:
        compact = {
            k: row[k] for k in ["method", "pipeline", "fitted checkpoints", "samples per seed"]
        }
        for key in [
            "psnr",
            "ssim",
            "lpips",
            "delta_e76",
            "data_consistency_l1",
            "RGB draw dispersion",
        ]:
            if key not in row:
                continue
            sd = row.get(key + " seed SD")
            compact[key] = f"{row[key]:.4f} ± {sd:.4f}" if sd is not None else row[key]
        output.append(compact)
    return table(output)


def overview(summary):
    """Show fidelity/perception, shift and limited-label gains together."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, task in zip(axes[0], ["colorization", "super_resolution"]):
        restoration_scatter(ax, summary[task]["comparisons"])
        ax.set(
            title=task.replace("_", " ").title() + " · 400 shared test images",
            xlabel="RGB PSNR (dB) ↑",
            ylabel="LPIPS-Alex ↓",
        )
        ax.grid(alpha=0.2)
    old_housing = evidence("regression")
    housing_rows = summary["modern_tabular"]["regression"]
    for offset, (name, records) in enumerate(
        [("Histogram boosting", old_housing["rows"]), ("Native CatBoost", housing_rows)]
    ):
        selected = [r for r in records if r["model"] in ["hist_boosting", "catboost_native"]]
        means = [
            np.mean(
                [r["split_conformal"]["coverage"] for r in selected if r["protocol"] == protocol]
            )
            for protocol in ["random", "geographic"]
        ]
        axes[1, 0].bar(np.arange(2) + (offset - 0.5) * 0.34, means, width=0.34, label=name)
    axes[1, 0].axhline(0.9, color="gray", linestyle="--", label="Nominal 90%")
    axes[1, 0].set(
        xticks=[0, 1],
        xticklabels=["Random", "Geographic"],
        ylim=(0, 1),
        ylabel="Observed interval coverage",
        title="Stronger prediction does not repair shift",
    )
    axes[1, 0].legend(fontsize=8)
    old_vision = evidence("vision")
    names = [
        "supervised_lowlabel",
        "contrastive_encoder",
        "imagenet_resnet18",
        "convnext_tiny_finetuned",
    ]
    records = old_vision["rows"] + summary["modern_vision"]
    values = [[r["accuracy"] for r in records if r["model"] == name] for name in names]
    axes[1, 1].bar(
        np.arange(4),
        [np.mean(v) for v in values],
        yerr=[np.std(v, ddof=1) for v in values],
        capsize=4,
    )
    axes[1, 1].set(
        xticks=np.arange(4),
        xticklabels=[
            "Scratch",
            "Contrastive\nprobe",
            "ImageNet\nResNet probe",
            "ImageNet\nConvNeXt fit",
        ],
        ylim=(0, 1),
        ylabel="Official-test accuracy",
        title="5,000 fitting + 5,000 held-out labels\nExternal exposure differs",
    )
    fig.suptitle(
        "Measured outcomes and limitations · checkpoint-conditioned restoration, three-seed prediction"
    )
    fig.tight_layout()
    save(fig, "modern_overview")


def update_readme(summary, status):
    vision = summary.get("modern_vision")
    tabular = summary.get("modern_tabular")
    if not vision or not tabular:
        return
    accuracy = np.mean([r["accuracy"] for r in vision])
    auc = np.mean([r["roc_auc"] for r in tabular["classification"]])
    text = (
        f"**Completed modern measurements:** ConvNeXt fine-tuning reaches **{accuracy:.2%}** "
        f"CIFAR-10 accuracy with 5,000 fitting labels, 5,000 selection/calibration labels "
        f"and ImageNet initialization (three seeds); "
        f"native CatBoost reaches Adult AUROC **{auc:.4f}** (three seeds). "
    )
    sr = {r["method"]: r for r in summary["super_resolution"]["comparisons"]}
    if "swinir" in sr and "diffusion" in sr:
        text += (
            f"SwinIR achieves SR PSNR **{sr['swinir']['psnr']:.2f} dB**; "
            f"local diffusion achieves LPIPS **{sr['diffusion']['lpips']:.4f}** with "
            f"PSNR **{sr['diffusion']['psnr']:.2f} dB**, showing a fidelity/perception tradeoff. "
        )
    text += (
        "Geographic interval failure, noise mismatch, ambiguous colors and weak CVAE "
        "diversity remain visible in the [modern report](docs/modern_results.md)."
    )
    if status == "complete":
        text += (
            " The laboratory contains 32 completed reference protocols and 19 English walkthroughs."
        )
    path = ROOT / "README.md"
    source = path.read_text()
    source = re.sub(
        r"<!-- MODERN_RESULTS_START -->.*?<!-- MODERN_RESULTS_END -->",
        "<!-- MODERN_RESULTS_START -->\n" + text + "\n<!-- MODERN_RESULTS_END -->",
        source,
        flags=re.S,
    )
    if status == "complete":
        source = source.replace(
            "![Measured colorization comparisons](assets/generative_colorization_fidelity_perception.png)",
            "![Measured outcomes across restoration and prediction](assets/modern_overview.png)",
        )
        preview = (
            "<!-- RESTORATION_PREVIEW_START -->\n"
            "<details>\n<summary>Fixed visual comparisons across all restoration pipelines</summary>\n\n"
            "Four attributed held-out photographs, with seed0/draw0 fixed for every "
            "pipeline. Full galleries retain all fitted seeds and stochastic draws.\n\n"
            "![Colorization pipelines](assets/colorization_comparison_gallery.png)\n\n"
            "![Super-resolution pipelines](assets/super_resolution_comparison_gallery.png)\n\n"
            "</details>\n<!-- RESTORATION_PREVIEW_END -->\n\n"
        )
        if "<!-- RESTORATION_PREVIEW_START -->" in source:
            source = re.sub(
                r"<!-- RESTORATION_PREVIEW_START -->.*?<!-- RESTORATION_PREVIEW_END -->\n\n",
                preview,
                source,
                flags=re.S,
            )
        else:
            source = source.replace("<!-- RESULTS_START -->", preview + "<!-- RESULTS_START -->")
    path.write_text(source)


def main():
    (ASSETS / "source_data").mkdir(exist_ok=True)
    missing, summary = [], {}
    sections = [
        "# Modern methods: measured comparisons\n",
        "Generated by `scripts/build_upgrade_report.py` from saved run records. "
        "Full-precision per-image scores are public; galleries are display-only uint8 copies. "
        "Image intervals condition on the fitted checkpoints and fixed samples. "
        "Seed SD is descriptive across two training seeds, not a confidence interval. "
        "Local means fitted here: colorization's direct/GAN generators use ImageNet "
        "initialization, and its CVAE shares that fitted semantic mean. Feature-loss "
        "networks are also pretrained. External denotes a frozen released pipeline; "
        "neither label implies equal training exposure.\n",
    ]
    for task in ["colorization", "super_resolution"]:
        rows, paired, paired_direct, distributions, budgets, latent, pending = restoration(task)
        missing.extend(pending)
        summary[task] = {
            "comparisons": rows,
            "paired": paired,
            "paired_direct": paired_direct,
            "image_distributions": distributions,
            "resources": budgets,
            "latent_diagnostics": latent,
        }
        visual = []
        if not pending:
            restoration_panel(task, rows)
            visual.append(
                f"![All {task.replace('_', ' ')} pipelines on the same fixed photographs]"
                f"(../assets/{task}_comparison_gallery.png)\n\n"
                "This compact comparison fixes seed0/draw0 for every pipeline. "
                "The baseline is the observed input. All fitted seeds and stochastic "
                "draws remain available in the individual gallery figures and walkthroughs. "
                "No photograph or draw was chosen using model quality.\n"
            )
        sections += [
            f"## {task.replace('_', ' ').title()}\n",
            *visual,
            comparison_table(rows),
            "\nPaired differences preserve the same held-out images. Higher PSNR and lower LPIPS are preferable.\n",
            table(paired),
            "\nPaired differences against the trained direct predictor:\n",
            table(paired_direct),
            "\nDistribution of per-image PSNR after averaging the fixed checkpoints and draws:\n",
            table(distributions),
        ]
        gan = evidence(f"generative_{task}/gan")
        direct = evidence(f"generative_{task}/l1")
        if gan and direct:
            gan_psnr = np.mean([row["psnr"]["mean"] for row in gan["rows"]])
            direct_psnr = np.mean([row["psnr"]["mean"] for row in direct["rows"]])
            gan_lpips = np.mean([row["lpips"]["mean"] for row in gan["rows"]])
            direct_lpips = np.mean([row["lpips"]["mean"] for row in direct["rows"]])
            if gan_psnr < direct_psnr and gan_lpips > direct_lpips:
                selected = [curve["best_epoch"] for curve in gan["curves"]]
                sections.append(
                    "\nThe local GAN does not improve either mean PSNR or LPIPS over the "
                    f"direct predictor in this experiment. Validation selected epochs {selected}; "
                    "all configured training epochs were completed. Training curves and fixed "
                    "galleries retain this result rather than implying that adversarial learning "
                    "automatically improves appearance.\n"
                )
        if task == "colorization":
            sections += [
                "\nThe grayscale baseline preserves lightness exactly. Eleven nearly achromatic "
                "photographs exceed 60dB, including exact reconstructions at the numerical "
                "120dB ceiling from MSE clipping at 1e-12. These values raise its mean PSNR; "
                "the median and quartiles reveal the typical-image behavior. All methods "
                "use the same scoring convention, and no images were removed. Lower "
                "LPIPS and DeltaE support different conclusions from mean PSNR.\n"
            ]
        sections += [
            "\nMeasured resource context:\n",
            table(budgets),
            "\nParameter counts describe inference models and include frozen encoders; "
            "they exclude training-only feature networks and discriminators. Epoch hours "
            "sum training/validation loops over all fitted seeds, excluding initialization "
            "and checkpoint export. The colorization CVAE's latent epochs additionally "
            "depend on its shared 60-epoch direct predictor. Inference timings include "
            "metrics for one full 400-image draw; concurrent workloads and adapter "
            "preprocessing differ, so these are run context rather than controlled latency benchmarks.\n",
        ]
        if latent:
            sections += [
                "\n### Does the CVAE use its latent variable?\n",
                table(latent),
                "\nKL values are training-batch averages from the posterior, summed over all "
                "latent dimensions. Prior validation uses the EMA checkpoint and observes no "
                "target in its encoder. Small KL together with negligible RGB draw dispersion "
                "is consistent with a nearly deterministic decoder; a low reconstruction loss "
                "alone does not establish useful conditional generation. These diagnostics "
                "do not estimate mutual information or prove posterior collapse.\n",
            ]
            for row in latent:
                stem = f"generative_{task}_cvae_seed{row['seed']}_latent"
                sections.append(
                    f"![CVAE latent diagnostics, seed {row['seed']}](../assets/{stem}.png)"
                )
        if rows:
            fig, ax = plt.subplots(figsize=(8, 5))
            restoration_scatter(ax, rows)
            ax.set(
                xlabel="RGB PSNR (dB), higher is preferable",
                ylabel="LPIPS-Alex, lower is preferable",
                title=task.replace("_", " ").title() + " · circles: local; squares: external",
            )
            ax.grid(alpha=0.2)
            save(fig, f"generative_{task}_fidelity_perception")
            sections.append(
                f"![Fidelity and perceptual metric comparison]"
                f"(../assets/generative_{task}_fidelity_perception.png)"
            )
    modern = evidence("modern_tabular")
    if modern:
        for path in (RESULTS / "modern_tabular").glob("*predictions*.npz"):
            arrays("modern_tabular", path.name).close()
        summary["modern_tabular"] = modern
        sections += [
            "\n## Native categorical boosting\n",
            table(
                [
                    {
                        k: row[k]
                        for k in [
                            "seed",
                            "roc_auc",
                            "log_loss",
                            "ece",
                            "set_coverage",
                            "set_size",
                            "decision_cost",
                            "selected_depth",
                        ]
                    }
                    for row in modern["classification"]
                ]
            ),
            "\nHousing random versus geographic holdout:\n",
            table(
                [
                    {k: r[k] for k in ["protocol", "seed", "rmse", "mae"]}
                    | {
                        "split coverage": r["split_conformal"]["coverage"],
                        "split width": r["split_conformal"]["mean_width"],
                        "CQR coverage": r["cqr"]["coverage"],
                        "CQR width": r["cqr"]["mean_width"],
                        "CQR empty rate": r["cqr"]["empty_rate"],
                    }
                    for r in modern["regression"]
                ]
            ),
            "\nBoosting improves the random split but does not solve geographic shift. "
            "Conformal coverage assumptions fail under that shift; report the observed coverage.\n",
        ]
        analysis = diagnostics("modern_tabular")
        if analysis:
            summary["modern_tabular_diagnostics"] = analysis
            groups = [
                {
                    k: r.get(k)
                    for k in [
                        "feature",
                        "group",
                        "n",
                        "roc_auc",
                        "ece",
                        "set_coverage",
                        "tpr",
                        "fpr",
                    ]
                }
                for r in analysis["groups_seed0"]
            ]
            sections += [
                "\nPost-fit subgroup diagnostics, seed 0:\n",
                table(groups),
                "\nMarginal conformal coverage does not guarantee group coverage. These descriptive test diagnostics do not establish fairness or causal effects and did not change the model. Small groups retain their counts.\n",
            ]
            importance = sorted(analysis["permutation_seed0"], key=lambda r: r["NLL increase"])
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            reliability_plot(
                axes[0], analysis["reliability_seed0"]["bins"], "probability", "frequency"
            )
            axes[0].set_title("Adult calibrated reliability · seed 0")
            axes[1].barh(
                [r["feature"] for r in importance],
                [r["NLL increase"] for r in importance],
                xerr=[r["repeat SD"] for r in importance],
            )
            axes[1].set(
                xlabel="Calibrated NLL increase (3 permutations)",
                title="Fixed first 2,500 test rows · associative",
            )
            fig.tight_layout()
            save(fig, "modern_tabular_diagnostics")
            sections += [
                "\n![Adult reliability and permutation diagnostics](../assets/modern_tabular_diagnostics.png)\n"
            ]
        else:
            missing.append("modern_tabular/diagnostics")
    else:
        missing.append("modern_tabular")
    vision = evidence("modern_vision")
    if vision:
        for path in (RESULTS / "modern_vision").glob("predictions_*.npz"):
            arrays("modern_vision", path.name).close()
        summary["modern_vision"] = vision["rows"]
        sections += [
            "\n## Limited-label ConvNeXt transfer\n",
            "Each seed uses 5,000 fitting labels plus 2,000 validation, 1,500 probability-"
            "calibration and 1,500 conformal-calibration labels. These 10,000 local labels "
            "have disjoint roles. Stratified benchmark sampling uses the available class "
            "annotations; this is a fitting-label budget, not an annotation-acquisition claim.\n\n",
            table(
                [{k: v for k, v in r.items() if not isinstance(v, list)} for r in vision["rows"]]
            ),
            "\nThe input is the same 32px CIFAR information resized to 128px. ImageNet pretraining "
            "is additional exposure. Comparison with scratch or self-supervised learning is a "
            "practical systems comparison; external-data overlap was not audited. Gaussian-noise "
            "accuracy reveals a substantial remaining robustness limitation.\n",
        ]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for r in vision["curves"]:
            for ax, key in zip(axes, ["train_loss", "validation_nll"]):
                ax.plot(
                    [v["epoch"] for v in r["history"]],
                    [v[key] for v in r["history"]],
                    label=f"Seed {r['seed']}",
                )
                ax.set(xlabel="Epoch", ylabel=key.replace("_", " "))
                ax.legend()
        save(fig, "modern_vision_training")
        analysis = diagnostics("modern_vision")
        if analysis:
            summary["modern_vision_diagnostics"] = analysis
            paired_rows = [
                {
                    "seed": r["seed"],
                    "comparison": r["comparison"],
                    "NLL difference": r["difference"],
                    "image CI lower": r["ci95"][0],
                    "image CI upper": r["ci95"][1],
                }
                for r in analysis["paired"]
            ]
            sections += [
                "\nPaired held-out NLL differences against earlier pipelines:\n",
                table(paired_rows),
                "\nIntervals condition on fitted checkpoints; architecture, pretraining and fine-tuning resources differ.\n",
            ]
            classes = [
                "plane",
                "car",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for ax, row in zip(axes, analysis["rows"]):
                counts = np.asarray(row["confusion"])
                image = ax.imshow(
                    counts / counts.sum(1, keepdims=True), vmin=0, vmax=1, cmap="Blues"
                )
                ax.set(
                    xticks=np.arange(10),
                    yticks=np.arange(10),
                    xticklabels=classes,
                    yticklabels=classes,
                    xlabel="Predicted class",
                    ylabel="True class",
                    title=f"Seed {row['seed']} · row-normalized",
                )
                ax.tick_params(axis="x", labelrotation=90)
            fig.colorbar(image, ax=list(axes), fraction=0.02, pad=0.02)
            save(fig, "modern_vision_confusion")
            fig, ax = plt.subplots(figsize=(5, 5))
            for row in analysis["rows"]:
                reliability_plot(
                    ax,
                    row["reliability"],
                    label=f"Seed {row['seed']}, ECE {row['calibrated_top_label_ece']:.3f}",
                )
            ax.set_title("CIFAR calibrated top-label reliability")
            save(fig, "modern_vision_reliability")
            sections += [
                "\n![Per-class errors across seeds](../assets/modern_vision_confusion.png)\n",
                "\n![Calibrated confidence](../assets/modern_vision_reliability.png)\n",
            ]
        else:
            missing.append("modern_vision/diagnostics")
        calibration_path = RESULTS / "modern_vision/cifar_calibration_audit.json"
        calibration = (
            json.loads(calibration_path.read_text()) if calibration_path.exists() else None
        )
        if calibration and calibration["status"] == "passed":
            summary["modern_vision_calibration"] = calibration
            for path in (RESULTS / "modern_vision").glob("cifar_calibration_*.npz"):
                arrays("modern_vision", path.name).close()
            fields = [
                "seed",
                "profile",
                "temperature",
                "accuracy",
                "nll",
                "set_coverage",
                "set_size",
                "empty_set_rate",
                "singleton_fraction",
                "singleton_accuracy",
            ]
            sections += [
                "\n### Deployment prediction sets\n",
                table([{key: row[key] for key in fields} for row in calibration["rows"]]),
                "\nEach profile evaluates all 10,000 official test images. CUDA preserves the "
                "original temperature and prediction sets; CPU temperature and cutoffs use only "
                "the reserved independent calibration partitions. Metadata binds the checkpoint "
                "and precision profile. Empty sets remain explicit. Singleton accuracy is "
                "descriptive and has no conditional coverage guarantee.\n",
            ]
        else:
            missing.append("modern_vision/calibration_export")
    else:
        missing.append("modern_vision")
    denoise = evidence("natural_denoising")
    if denoise:
        rows = [
            {
                "model": r["model"],
                "noise sigma": r["sigma"],
                **{k: r[k]["mean"] for k in ["psnr", "ssim", "lpips"]},
            }
            for r in denoise["rows"]
        ]
        sections += [
            "\n## Natural-image denoising and noise shift\n",
            table(rows),
            "\nSwinIR was trained externally at sigma 25. Sigma 15 and 50 test noise mismatch "
            "without adaptation. Wavelet BayesShrink uses the known test noise level. "
            "At sigma 50 the learned reference degrades strongly; this failure stays in the report.\n",
        ]
        summary["natural_denoising"] = rows
        arrays("natural_denoising", "per_image_scores.npz").close()
        with arrays("natural_denoising", "gallery.npz", gallery=True) as data:
            for sigma in [15, 25, 50]:
                keys = [
                    "original",
                    f"input_sigma{sigma}",
                    f"wavelet_sigma{sigma}",
                    f"swinir_noise25_sigma{sigma}",
                ]
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                for i in range(4):
                    for j, key in enumerate(keys):
                        axes[i, j].imshow(data[key][i].transpose(1, 2, 0))
                        axes[i, j].axis("off")
                        if i == 0:
                            axes[i, j].set_title(key.replace("_", "\n"), fontsize=8)
                save(fig, f"natural_denoising_sigma{sigma}_gallery")
    else:
        missing.append("natural_denoising")
    shifted = evidence("sr_shift")
    if shifted:
        shift_rows = [
            {
                "method": r["method"],
                "degradation": r["degradation"],
                **{k: r[k]["mean"] for k in ["psnr", "ssim", "lpips"]},
            }
            for r in shifted["rows"]
        ]
        summary["sr_shift"] = shift_rows
        sections += [
            "\n## SR degradation stress tests\n",
            table(shift_rows),
            "\nThe same fixed photographs receive bicubic downsampling, pre-downsampling "
            "Gaussian blur, or LR Gaussian noise followed by JPEG compression. "
            "Both external models remain frozen. These controlled stressors do not "
            "reproduce real camera acquisition or either paper's full degradation distribution.\n",
        ]
        arrays("sr_shift", "per_image_scores.npz").close()
        with arrays("sr_shift", "gallery.npz", gallery=True) as data:
            for degradation in ["bicubic", "blur", "noise_jpeg"]:
                keys = ["original"] + [
                    f"{method}_{degradation}" for method in ["bicubic", "swinir", "realesrgan"]
                ]
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                for i in range(4):
                    for j, key in enumerate(keys):
                        axes[i, j].imshow(data[key][i].transpose(1, 2, 0))
                        axes[i, j].axis("off")
                        if i == 0:
                            axes[i, j].set_title(key.replace("_", "\n"), fontsize=8)
                fig.suptitle("Fixed SR stress test · " + degradation.replace("_", " "))
                save(fig, f"sr_shift_{degradation}_gallery")
    else:
        missing.append("sr_shift")
    sections += [
        "\n## Interpretation and reproducibility\n",
        "[Protocol and model equations](12_generative_restoration.md). "
        "[Attribution for every displayed photograph](photo_attributions.md). "
        "Local GANs and diffusion models have limited data and compute; a modern architecture "
        "alone does not guarantee a stronger output. All stochastic samples are included, "
        "with no closest-to-target selection. The CVAE uses its conditional prior at test "
        "time; only the colorization CVAE uses a shared frozen validation-selected semantic "
        "mean. External models have "
        "different training resources and possible unaudited overlap.\n",
    ]
    status = "complete" if not missing else "in progress"
    if status == "complete":
        overview(summary)
    update_readme(summary, status)
    sections.insert(
        1,
        f"**Evidence status: {status}.**"
        + (" Pending: " + ", ".join(missing) if missing else "")
        + "\n",
    )
    (ROOT / "docs/modern_results.md").write_text("\n\n".join(sections) + "\n")
    (ASSETS / "modern_summary.json").write_text(
        json.dumps({"status": status, "pending": missing, **summary}, indent=2) + "\n"
    )
    print("Report", status, "pending", missing)


if __name__ == "__main__":
    main()
