"""Build assignment-grounded figures and a measured report from saved evidence."""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parents[1]
RESULTS, ASSETS = ROOT / "results" / "reference", ROOT / "assets"
STUDIES = ["course_mixtures", "inverse", "denoising", "super_resolution", "colorization", "ddcolor"]


def save(fig, name):
    for extension in ["png", "pdf"]:
        fig.savefig(ASSETS / f"{name}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def array(study, filename):
    source = RESULTS / study / filename
    target = ASSETS / "source_data" / f"{study}_{filename}"
    if source.exists():
        if study in ["super_resolution", "colorization", "ddcolor"] and filename == "gallery.npz":
            indices = [
                row["test_index"] for row in json.loads((ASSETS / "photo_credits.json").read_text())
            ]
            with np.load(source, allow_pickle=False) as values:
                np.savez_compressed(target, **{k: values[k][indices] for k in values.files})
        else:
            shutil.copyfile(source, target)
    return np.load(target, allow_pickle=False)


def gallery(study, keys, labels, name, count=4):
    data = array(study, "gallery.npz" if study not in ["inverse"] else "controlled_gallery.npz")
    count = min(count, len(data[keys[0]]))
    fig, axes = plt.subplots(
        count, len(keys), figsize=(2.2 * len(keys), 2.2 * count), squeeze=False
    )
    for i in range(count):
        for j, key in enumerate(keys):
            pixels = data[key][i]
            if pixels.ndim == 3:
                pixels = pixels.transpose(1, 2, 0)
            axes[i, j].imshow(pixels, cmap="gray", vmin=0, vmax=1)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(labels[j])
    fig.suptitle("Held-out images · fixed ordering and attribution", y=1.01)
    save(fig, name)


def curves(data, study, name):
    models = sorted({r["model"] for r in data[study]["curves"]})
    fig, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 3.5), squeeze=False)
    for ax, model in zip(axes[0], models):
        for record in [r for r in data[study]["curves"] if r["model"] == model]:
            frame = pd.DataFrame(record["history"])
            ax.plot(frame.epoch, frame.train_loss, alpha=0.7, label=f"train seed {record['seed']}")
            ax.plot(
                frame.epoch,
                frame.validation_loss,
                "--",
                alpha=0.9,
                label=f"val seed {record['seed']}",
            )
            ax.axvline(record["best_epoch"], color="gray", alpha=0.15)
        ax.set(title=model.replace("_", " "), xlabel="Epoch", ylabel="Objective loss")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, name)


def figures(data):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for i in [1, 2]:
        arrays = array("course_mixtures", f"mixture_{i}.npz")
        for j, key in enumerate(["labels", "assignment"]):
            ax = axes[i - 1, j]
            ax.scatter(*arrays["x"].T, c=arrays[key], cmap="viridis", s=8, alpha=0.6)
            for mu, cov in zip(arrays["means"], arrays["covariances"]):
                eigen, vectors = np.linalg.eigh(cov)
                direction = vectors[:, eigen.argmax()]
                ellipse = Ellipse(
                    mu,
                    4 * np.sqrt(eigen.max()),
                    4 * np.sqrt(eigen.min()),
                    angle=np.degrees(np.arctan2(direction[1], direction[0])),
                    fill=False,
                    edgecolor="#ef4444",
                )
                ax.add_patch(ellipse)
            ax.set(
                title=f"Image {i}: {'source groups' if j == 0 else 'EM assignments'}",
                xlabel="Coordinate 1",
                ylabel="Coordinate 2",
            )
            ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "course_mixture_geometry")
    fig, ax = plt.subplots(figsize=(7, 4))
    inv = pd.DataFrame(data["inverse"]["controlled"])
    for method, frame in inv.groupby("method"):
        ax.plot(frame.m, frame.test_psnr, "o-", label=method)
    ax.axhline(data["inverse"]["controlled_raw_psnr"], color="gray", linestyle="--", label="noisy")
    ax.set(
        xlabel="Patch width (m)",
        ylabel="Mean test PSNR (dB)",
        title="One common AWGN observation across patch sizes",
    )
    ax.legend()
    save(fig, "inverse_patch_sweep")
    gallery(
        "inverse",
        ["clean", "observed", "restored"],
        ["Clean", "Noisy σ=30", "GMM MMSE, m=8"],
        "inverse_gallery",
    )
    # Supplied archives remain a separate figure/evaluation population.
    if data["inverse"]["supplied"]:
        arr = array("inverse", "course_gallery_m4.npz")
        fig, axes = plt.subplots(4, 5, figsize=(10, 8))
        for i in range(4):
            for j, key in enumerate(["clean", "observed", "hard", "restored", "calibrated"]):
                axes[i, j].imshow(arr[key][i], cmap="gray", vmin=0, vmax=1)
                axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(
                        [
                            "Clean",
                            "Supplied PNG",
                            "Hard heuristic",
                            "Nominal MMSE",
                            "Calibrated MMSE",
                        ][j]
                    )
        save(fig, "course_inverse_gallery")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in ["noisy", "pca_projection", "pca_wiener", "autoencoder", "residual_denoiser"]:
        rows = [r for r in data["denoising"]["rows"] if r["model"] == model and "sigma_255" in r]
        frame = pd.DataFrame([{"noise": r["sigma_255"], "psnr": r["psnr"]["mean"]} for r in rows])
        mean, std = frame.groupby("noise").psnr.mean(), frame.groupby("noise").psnr.std().fillna(0)
        ax.errorbar(
            mean.index, mean, yerr=std, marker="o", capsize=3, label=model.replace("_", " ")
        )
    ax.axvspan(51, 75, color="#f59e0b", alpha=0.1, label="Outside training noise range")
    ax.set(
        xlabel="Noise SD (0–255 units)",
        ylabel="Mean test PSNR (dB)",
        title="Denoising across noise levels; bars = seed SD",
    )
    ax.legend(fontsize=8)
    save(fig, "denoising_noise_sweep")
    gallery(
        "denoising",
        ["clean", "observed", "autoencoder", "residual_denoiser"],
        ["Clean", "Noisy σ=30", "Autoencoder", "Residual CNN"],
        "denoising_gallery",
    )
    gallery(
        "super_resolution",
        ["clean", "bicubic", "autoencoder", "pixelshuffle_residual"],
        ["Target 96px", "Bicubic 4×", "Autoencoder", "Residual pixel shuffle"],
        "super_resolution_gallery",
    )
    gallery(
        "colorization",
        ["clean", "neutral_gray", "autoencoder_l2", "transfer_l2", "transfer_balanced_bins"],
        ["Original", "Lightness only", "Autoencoder L2", "Transfer L2", "Balanced bins"],
        "colorization_gallery",
    )
    for study in ["denoising", "super_resolution", "colorization"]:
        curves(data, study, study + "_training")
        array(study, "per_image_scores.npz")
    colors = array("colorization", "gallery.npz")
    pretrained = array("ddcolor", "gallery.npz")
    fig, axes = plt.subplots(
        len(pretrained["clean"]), 3, figsize=(7, 2.2 * len(pretrained["clean"])), squeeze=False
    )
    for i in range(len(pretrained["clean"])):
        for j, pixels in enumerate(
            [pretrained["clean"][i], colors["transfer_l2"][i], pretrained["pretrained"][i]]
        ):
            axes[i, j].imshow(pixels.transpose(1, 2, 0))
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(["Original", "Local transfer L2", "Frozen DDColor"][j])
    save(fig, "ddcolor_gallery")
    array("ddcolor", "per_image_scores.npz")
    fig, axes = plt.subplots(1, len(colors["entropy"]), figsize=(9, 2.8))
    for i, ax in enumerate(axes):
        plot = ax.imshow(
            colors["entropy"][i],
            cmap="magma",
            vmin=0,
            vmax=np.log(data["colorization"]["palette_centers"].__len__()),
        )
        ax.axis("off")
    fig.colorbar(plot, ax=axes, fraction=0.025, label="Palette entropy (nats)")
    save(fig, "colorization_entropy")


def report(data):
    lines = [
        "# Measured course-project extensions",
        "",
        "These are completed reference runs, separate from quick pipeline checks. See [protocols and derivations](10_course_projects.md) and [restoration design](11_image_restoration.md). Galleries use fixed held-out ordering; natural-image publication uses the first four test-order photographs with verified creator attribution. Photos retain their [source attribution](photo_attributions.md).",
        "",
        "## Mixture geometry and missing data",
        "",
        "![Course mixture geometry](../assets/course_mixture_geometry.png)",
        "",
        "| Dataset | Model | Test ARI, mean ± seed SD | Test NLL, mean |",
        "|---|---|---:|---:|",
    ]
    frame = pd.DataFrame(data["course_mixtures"]["rows"])
    for (dataset, model), group in frame.groupby(["dataset", "model"]):
        nll = f"{group.test_nll.mean():.4f}" if group.test_nll.notna().any() else "—"
        lines.append(
            f"| Image {dataset} | {model} | {group.test_ari.mean():.4f} ± {group.test_ari.std():.4f} | {nll} |"
        )
    lines += [
        "",
        "Image 2 centers nearly coincide: overlapping component labels cannot be recovered perfectly. Validation density chooses three full-covariance components in the first partition for both datasets. NLL describes density estimation; ARI describes agreement with supplied source groups. Labels do not enter fitting or selection.",
        "",
        "| Missing-coordinate example | Conditional GMM RMSE | Mean-imputation RMSE |",
        "|---|---:|---:|",
    ]
    for dataset, group in pd.DataFrame(data["course_mixtures"]["imputation"]).groupby("dataset"):
        lines.append(
            f"| Image {dataset} | {group.conditional_gmm_rmse.mean():.4f} | {group.mean_imputation_rmse.mean():.4f} |"
        )
    cmm = data["course_mixtures"]["categorical_mixture"]
    lines += [
        "",
        f"The synthetic categorical mixture imputes **{cmm['missing_accuracy']:.2%}** of missing test entries correctly, versus **{cmm['majority_missing_accuracy']:.2%}** for per-feature majority fitted only on observed training entries. Held-out latent ARI is {cmm['latent_test_ari']:.4f}; this is a controlled identifiable model, not an external real-data benchmark.",
        "",
        "## Bayesian patch reconstruction",
        "",
        "![Controlled patch sweep](../assets/inverse_patch_sweep.png)",
        "",
        "![Controlled reconstruction gallery](../assets/inverse_gallery.png)",
        "",
        "| Patch width | Validation-selected K | MMSE test PSNR | Hard-component test PSNR |",
        "|---|---:|---:|---:|",
    ]
    inv = data["inverse"]
    for row in inv["controlled"]:
        if row["method"] == "mmse":
            hard = next(
                r for r in inv["controlled"] if r["method"] == "hard" and r["m"] == row["m"]
            )
            lines.append(
                f"| {row['m']} | {row['k']} | {row['test_psnr']:.3f} | {hard['test_psnr']:.3f} |"
            )
    lines += [
        "",
        f"The common noisy input scores {inv['controlled_raw_psnr']:.3f} dB. Validation selects m={inv['selected']['m']}, K={inv['selected']['k']}; the test curve was not used to choose them. Prior convergence flags and PCA rank limits are in the metric record.",
        "",
        "### Supplied archives: a separate, imperfect observation model",
        "",
        "| Actual m | Raw PSNR | Hard PSNR | Nominal MMSE | Calibrated MMSE | Validation residual mean (0–255) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in inv["supplied"]:
        lines.append(
            f"| {row['actual_m']} | {row['test_raw_psnr']:.3f} | {row['test_hard_psnr']:.3f} | {row['test_mmse_psnr']:.3f} | {row['test_calibrated_psnr']:.3f} | {row['observation_residual_mean_255']:.2f} |"
        )
    if inv["supplied"]:
        lines += [
            "",
            "![Supplied archive reconstruction](../assets/course_inverse_gallery.png)",
            "",
            "Clipped observations, mismatched filenames and nonzero empirical residual means limit the nominal independent Gaussian patch model. These results are not presented as controlled denoising scores or a common-data patch-size comparison.",
        ]
    data["colorization"]["rows"] += data["ddcolor"]["rows"]
    for study, title, condition in [
        ("denoising", "Neural denoising at σ=30", lambda r: r.get("sigma_255") == 30),
        (
            "super_resolution",
            "4× natural-image super-resolution",
            lambda r: r.get("degradation") == "bicubic",
        ),
        ("colorization", "Color reconstruction and uncertainty", lambda r: True),
    ]:
        lines += [
            "",
            "## " + title,
            "",
            "| Model | Test PSNR, mean ± seed SD | Test SSIM, mean |"
            + (" DeltaE76 | Mean chroma |" if study == "colorization" else ""),
            "|---|---:|---:|" + ("---:|---:|" if study == "colorization" else ""),
        ]
        models = sorted({r["model"] for r in data[study]["rows"] if condition(r)})
        for model in models:
            rows = [r for r in data[study]["rows"] if condition(r) and r["model"] == model]
            values = [r["psnr"]["mean"] for r in rows]
            sd = np.std(values, ddof=1) if len(values) > 1 else 0.0
            line = f"| {model} | {np.mean(values):.3f} ± {sd:.3f} | {np.mean([r['ssim']['mean'] for r in rows]):.4f} |"
            if study == "colorization":
                line += f" {np.mean([r['delta_e76']['mean'] for r in rows]):.3f} | {np.mean([r['mean_chroma']['mean'] for r in rows]):.3f} |"
            lines.append(line)
        lines += [
            "",
            f"![{title} gallery](../assets/{study}_gallery.png)",
            "",
            f"![{title} training curves](../assets/{study}_training.png)",
            "",
            "Per-image bootstrap intervals are in the metrics; the table's ± value is training-seed SD. Single deterministic baselines have zero seed SD, which does not imply zero test-image uncertainty.",
        ]
    lines += [
        "",
        "![Denoising noise sweep](../assets/denoising_noise_sweep.png)",
        "",
        "![Color uncertainty](../assets/colorization_entropy.png)",
        "",
        "Colorization is underdetermined. A stronger chroma output or higher palette entropy is not evidence that original colors were recovered. Compare neutral gray, L2 regression, and balanced classification jointly using errors and the fixed galleries.",
        "",
        "![Modern pretrained reference](../assets/ddcolor_gallery.png)",
        "",
        f"The frozen official DDColor-ModelScope model has {data['ddcolor']['parameters']:,} parameters and evaluates {data['ddcolor']['images_per_second']:.2f} images/s here. It uses externally trained ImageNet representations and color decoders; none of its parameters was fit in this laboratory. No COCO/pretraining overlap audit was possible. Its 512px internal input is an upsampled version of the same 96px lightness evidence, not additional high-resolution color input.",
        "",
        "## Full evidence and reproduction",
        "",
        "| Study | Complete run time (s) | Evidence |",
        "|---|---:|---|",
    ]
    for study in STUDIES:
        record = json.loads((RESULTS / study / "run.json").read_text())
        lines.append(
            f"| {study} | {record['elapsed_seconds']:.1f} | [metrics](../results/reference/{study}/metrics.json), [run](../results/reference/{study}/run.json) |"
        )
    lines += [
        "",
        "Full per-image PSNR arrays are losslessly compressed under `assets/source_data/<study>_per_image_scores.npz`; JSON row keys identify their arrays. This preserves paired comparisons without large repeated JSON lists. Gallery pixels and mixture figure inputs are also included, so every figure can be regenerated in a clean checkout. Model checkpoints, raw archives and unrelated original notebooks stay local.",
        "",
        "Run `python scripts/build_report.py` to rebuild both reports and all figures. Read the [reproduction guide](07_reproduction.md) for caches, devices and clean-checkout checks. These budgets establish useful baselines and failure analysis; they do not establish state-of-the-art restoration performance.",
        "",
    ]
    (ROOT / "docs" / "project_results.md").write_text("\n".join(lines))
    selected = json.loads((ASSETS / "photo_credits.json").read_text())
    credits = [
        "# Photo attributions",
        "",
        "Natural-image figures and their compact source arrays use the first four test-order photographs with verifiable live creator attribution. Selection depends on attribution availability, not model quality. Evaluation still includes all 400 fixed test images. Each image is center-cropped and resized to 96×96; predictions, grayscale versions and uncertainty maps derive from it. The code licence does not replace the photograph licence.",
        "",
        "All these source photographs carry [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) in both official COCO metadata and the verified Flickr metadata. Creator names and source links are preserved in `assets/photo_credits.json`.",
        "",
        "| Test order | COCO ID | Photographer's source page | Modification |",
        "|---|---:|---|---|",
    ]
    for row in selected:
        credits.append(
            f"| {row['test_index'] + 1} | {row['coco_id']} | [{row['author']}]({row['source_url']}) | Crop, resize; restoration/color predictions |"
        )
    credits += [
        "",
        "MNIST galleries derive from the public MNIST digit benchmark; see [dataset references](references.md). The coordinate figure depicts the supplied synthetic numerical examples, not medical imagery. Original handouts and ZIP archives are not redistributed.",
        "",
    ]
    (ROOT / "docs" / "photo_attributions.md").write_text("\n".join(credits))


def main():
    data = {s: json.loads((RESULTS / s / "metrics.json").read_text()) for s in STUDIES}
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    )
    figures(data)
    report(data)
    print("Rebuilt six project studies and thirteen additional figure sets.")


if __name__ == "__main__":
    main()
