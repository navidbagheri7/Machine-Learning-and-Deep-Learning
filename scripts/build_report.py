"""Regenerate publication figures and the measured report from completed reference records."""

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
RESULTS = ROOT / "results" / "reference"
ASSETS = ROOT / "assets"
COLORS = ["#2563eb", "#0d9488", "#e89225", "#a855f7", "#e45756"]
NAMES = {
    "logistic": "Logistic",
    "hist_boosting": "Boosting",
    "tabular_resnet": "Tabular ResNet",
    "ridge": "Ridge",
    "bayesian_ridge": "Bayesian ridge",
    "feature_transformer": "Feature transformer",
    "random_encoder": "Random features",
    "contrastive_encoder": "Contrastive features",
    "imagenet_resnet18": "ImageNet transfer",
    "supervised_lowlabel": "Supervised 5k",
    "supervised_full": "Supervised 45k",
    "nearest_raw": "Raw detector",
    "ridge_window": "Linear window",
    "known_channel_inverse": "Known-channel inverse",
    "mlp": "Window MLP",
    "cnn": "Temporal CNN",
    "rbf_svm": "RBF SVM",
    "nystroem": "Nyström",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "#f8fafc",
        "axes.prop_cycle": plt.cycler(color=COLORS),
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    }
)


def load(study):
    record = json.loads((RESULTS / study / "run.json").read_text())
    if record["status"] != "complete" or record["config"].get("quick"):
        raise ValueError(f"Cannot publish incomplete or quick {study} run")
    return json.loads((RESULTS / study / "metrics.json").read_text())


def source(study, filename):
    dest = ASSETS / "source_data" / (study + "_" + filename)
    dest.parent.mkdir(parents=True, exist_ok=True)
    original = RESULTS / study / filename
    if original.exists():
        shutil.copyfile(original, dest)
    return np.load(dest, allow_pickle=False)


def save(fig, name):
    fig.savefig(ASSETS / (name + ".png"), dpi=180)
    fig.savefig(ASSETS / (name + ".pdf"))
    plt.close(fig)


def summarize(frame, key, group="model"):
    return frame.groupby(group, sort=False)[key].agg(["mean", "std", "count"]).fillna(0)


def bar(ax, frame, key, title, percent=False):
    stats = summarize(frame, key)
    scale = 100 if percent else 1
    names = [NAMES.get(n, n) for n in stats.index]
    ax.barh(
        names,
        stats["mean"] * scale,
        xerr=stats["std"] * scale,
        color=COLORS[: len(names)],
        alpha=0.9,
    )
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(key.replace("_", " ") + (" (%)" if percent else ""))
    ax.grid(axis="x", alpha=0.15)


def tables():
    data = {
        s: load(s)
        for s in [
            "classification",
            "regression",
            "probabilistic",
            "generalization",
            "kernels",
            "signals",
            "vision",
        ]
    }
    adult = pd.DataFrame(data["classification"]["rows"])
    housing = pd.DataFrame(data["regression"]["rows"])
    vision = pd.DataFrame(data["vision"]["rows"])
    signal = pd.DataFrame(data["signals"]["rows"])
    return data, adult, housing, vision, signal


def figures(data, adult, housing, vision, signal):
    # The shift walkthrough uses this compact input even though the main interval panel is random.
    with source("regression", "intervals_geographic_0.npz"):
        pass
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    bar(axs[0], adult, "roc_auc", "Adult: discrimination")
    bar(axs[1], adult, "log_loss", "Adult: calibrated probability loss")
    for name, d in data["classification"]["diagnostics_seed0"].items():
        bins = d["reliability"]["bins"]
        axs[2].plot(
            [b["probability"] for b in bins],
            [b["frequency"] for b in bins],
            "o-",
            label=NAMES[name],
            markersize=4,
        )
    axs[2].plot([0, 1], [0, 1], "--", color="gray")
    axs[2].set(
        xlabel="Predicted probability",
        ylabel="Observed positive frequency",
        title="Reliability, seed 0",
    )
    axs[2].legend(fontsize=8)
    save(fig, "classification")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    groups = data["classification"]["diagnostics_seed0"]["hist_boosting"]["groups"]
    labels = [r["group"] + f" (n={r['n']:,})" for r in groups]
    axs[0].barh(labels, [r["set_coverage"] * 100 for r in groups], color=COLORS[1])
    axs[0].axvline(90, ls="--", color="gray")
    axs[0].set(xlabel="Prediction-set coverage (%)", title="Boosting subgroup diagnostics, seed 0")
    importance = data["classification"]["diagnostics_seed0"]["hist_boosting"][
        "permutation_importance"
    ]
    ranked = sorted(importance, key=importance.get, reverse=True)
    axs[1].barh(ranked, [importance[k] for k in ranked], color=COLORS[0])
    axs[1].invert_yaxis()
    axs[1].set(xlabel="Increase in test log loss", title="Raw-feature permutation importance")
    save(fig, "tabular_diagnostics")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    names = list(housing.model.unique())
    for j, protocol in enumerate(["random", "geographic"]):
        frame = housing[housing.protocol == protocol]
        stats = summarize(frame, "rmse").reindex(names)
        axs[0].bar(
            np.arange(len(names)) + (j - 0.5) * 0.35,
            stats["mean"],
            0.35,
            label=protocol,
            color=COLORS[j],
        )
        coverage = frame.copy()
        coverage["coverage"] = coverage.split_conformal.map(lambda x: x["coverage"])
        stats = summarize(coverage, "coverage").reindex(names)
        axs[1].bar(
            np.arange(len(names)) + (j - 0.5) * 0.35,
            stats["mean"] * 100,
            0.35,
            label=protocol,
            color=COLORS[j],
        )
    for ax in axs[:2]:
        ax.set_xticks(np.arange(len(names)), [NAMES[n] for n in names], rotation=30, ha="right")
        ax.legend(fontsize=8)
    axs[0].set(title="Point prediction across splits", ylabel="RMSE ($100,000)")
    axs[1].set(title="Residual conformal coverage", ylabel="Coverage (%)")
    axs[1].axhline(90, color="gray", ls="--")
    intervals = source("regression", "intervals_random_0.npz")
    order = np.argsort(intervals["pred"])[:120]
    axs[2].fill_between(
        np.arange(len(order)),
        intervals["lower"][order],
        intervals["upper"][order],
        alpha=0.25,
        label="CQR 90%",
    )
    axs[2].scatter(np.arange(len(order)), intervals["y"][order], s=8, label="Observed")
    axs[2].plot(intervals["pred"][order], color=COLORS[1], label="Predicted")
    axs[2].set(
        title="First 120 by prediction rank, seed 0",
        ylabel="Housing value ($100,000)",
        xlabel="Sorted test example",
    )
    axs[2].legend(fontsize=8)
    save(fig, "regression")

    geometry = source("probabilistic", "geometry.npz")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    axs[0].scatter(
        geometry["x"][:, 0],
        geometry["x"][:, 1],
        c=geometry["labels"],
        cmap="viridis",
        s=4,
        alpha=0.3,
    )
    for mean, cov in zip(geometry["means"], geometry["covariances"]):
        eig, vec = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vec[1, -1], vec[0, -1]))
        axs[0].add_patch(
            Ellipse(
                mean,
                4 * np.sqrt(eig[-1]),
                4 * np.sqrt(eig[0]),
                angle=angle,
                fill=False,
                edgecolor="#0f172a",
                lw=2,
            )
        )
    axs[0].set(title="Anisotropic mixture and fitted 2-SD contours", xlabel="x1", ylabel="x2")
    history = data["probabilistic"]["em_history"]
    axs[1].plot(history, "o-")
    axs[1].set(title="EM optimization", xlabel="EM iteration", ylabel="Mean fitting log density")
    g = geometry["grid"]
    axs[2].fill_between(
        g,
        geometry["posterior_mean"] - 1.645 * geometry["posterior_sd"],
        geometry["posterior_mean"] + 1.645 * geometry["posterior_sd"],
        alpha=0.25,
        label="90% model interval",
    )
    axs[2].plot(g, geometry["posterior_mean"], label="Posterior mean")
    axs[2].plot(g, np.sin(3 * g), "--", color="gray", label="True mean")
    axs[2].scatter(geometry["regression_x"], geometry["regression_y"], s=15, color=COLORS[1])
    axs[2].set(
        title="Polynomial posterior beyond fitting range", xlabel="x", ylabel="y", ylim=(-5, 5)
    )
    axs[2].legend(fontsize=8)
    save(fig, "probabilistic")

    descent = pd.DataFrame(data["generalization"]["double_descent"])
    implicit = pd.DataFrame(data["generalization"]["implicit_bias"])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for penalty, frame in descent.groupby("penalty"):
        stats = frame.groupby("ratio").population_mse.quantile([0.25, 0.5, 0.75]).unstack()
        axs[0].plot(stats.index, stats[0.5], label=f"penalty {penalty}")
        axs[0].fill_between(stats.index, stats[0.25], stats[0.75], alpha=0.15)
    axs[0].axvline(1, ls="--", color="gray")
    axs[0].set(
        yscale="log",
        title="Exact population risk, 30 repeats",
        xlabel="Features / observations",
        ylabel="Median MSE and IQR",
    )
    axs[0].legend(fontsize=8)
    axs[1].plot(implicit.step, implicit.loss, label="Logistic loss")
    axs[1].set(
        xscale="log",
        yscale="log",
        title="Loss keeps decreasing",
        xlabel="Gradient updates",
        ylabel="Logistic loss",
    )
    axs[2].plot(implicit.step, 1 - implicit.direction_cosine)
    axs[2].set(
        xscale="log",
        yscale="log",
        title="Direction approaches hard margin",
        xlabel="Gradient updates",
        ylabel="1 - direction cosine",
    )
    save(fig, "generalization")
    kernels = pd.DataFrame(data["kernels"]["rows"])
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for name, frame in kernels.groupby("model", sort=False):
        stats = frame.groupby("train_size").accuracy.agg(["mean", "std"])
        ax.errorbar(
            stats.index,
            stats["mean"] * 100,
            yerr=stats["std"] * 100,
            marker="o",
            capsize=4,
            label=NAMES[name],
        )
    ax.set(
        title="Nonlinear features with limited labels",
        xlabel="Training labels",
        ylabel="Test accuracy (%)",
    )
    ax.legend()
    save(fig, "kernels")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, gamma in zip(axs, [0.2, 0.4]):
        for name, frame in signal[signal.gamma == gamma].groupby("model", sort=False):
            stats = frame.groupby("snr_db").ber.agg(["mean", "std"])
            # Plotting floor is labeled; all exact zero values remain in metrics.
            ax.semilogy(stats.index, np.maximum(stats["mean"], 1e-6), "o-", label=NAMES[name])
        ax.set(
            title=f"Nonlinearity {gamma}"
            + (" (training channel)" if gamma == 0.2 else " (unadapted shift)"),
            xlabel="Es/N0 (dB)",
            ylabel="Mean BER, 3 seeds",
            ylim=(1e-6, 0.3),
        )
        ax.grid(alpha=0.2)
    axs[1].legend(fontsize=8)
    fig.text(
        0.5,
        -0.02,
        "Display floor 1e-6; the known-channel inverse uses the test channel parameters.",
        ha="center",
        fontsize=9,
    )
    save(fig, "signals")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    bar(axs[0], vision, "accuracy", "CIFAR-10: information budgets", True)
    bar(axs[1], vision, "nll", "Calibrated test probability loss")
    for seed, frame in vision[vision.model == "supervised_lowlabel"].iterrows():
        axs[2].plot(
            [0, 1],
            [frame.accuracy * 100, frame.gaussian_noise_accuracy * 100],
            "o-",
            label=f"seed {frame.seed}",
        )
    axs[2].set_xticks([0, 1], ["Clean", "Gaussian noise 0.1"])
    axs[2].set(title="Frozen low-label CNN stress test", ylabel="Accuracy (%)")
    axs[2].legend(fontsize=8)
    save(fig, "vision")
    errors = source("vision", "errors.npz")
    classes = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    fig, axs = plt.subplots(4, 8, figsize=(15, 8), layout="constrained")
    for i, ax in enumerate(axs.flat):
        ax.axis("off")
        if i < len(errors["truth"]):
            ax.imshow(errors["images"][i].transpose(1, 2, 0))
            ax.set_title(
                f"True: {classes[errors['truth'][i]]}\nPred: {classes[errors['pred'][i]]}\nConfidence {errors['confidence'][i]:.2f}",
                fontsize=8,
            )
    fig.suptitle("First 32 full-supervised mistakes in official test order", fontsize=14)
    save(fig, "vision_errors")

    fig, axs = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    bar(axs[0, 0], adult, "roc_auc", "Tabular discrimination")
    for protocol, frame in housing[housing.model == "hist_boosting"].groupby(
        "protocol", sort=False
    ):
        coverage = np.mean([r["cqr"]["coverage"] for r in frame.to_dict("records")])
        axs[0, 1].bar(protocol, coverage * 100, color=COLORS[0 if protocol == "random" else 4])
    axs[0, 1].axhline(90, color="gray", ls="--")
    axs[0, 1].set(title="Quantile intervals under shift", ylabel="Coverage (%)")
    for penalty, frame in descent.groupby("penalty"):
        stats = frame.groupby("ratio").population_mse.median()
        axs[0, 2].plot(stats.index, stats, label=f"penalty {penalty}")
    axs[0, 2].set(
        yscale="log",
        title="Interpolation and regularization",
        xlabel="Features / observations",
        ylabel="Population MSE",
    )
    axs[0, 2].legend(fontsize=8)
    bar(axs[1, 0], vision, "accuracy", "Image representation learning", True)
    for name, frame in signal[signal.gamma == 0.2].groupby("model", sort=False):
        stats = frame.groupby("snr_db").ber.mean()
        axs[1, 1].semilogy(stats.index, np.maximum(stats, 1e-6), label=NAMES[name])
    axs[1, 1].set(
        title="Learned nonlinear equalization",
        xlabel="Es/N0 (dB)",
        ylabel="BER (display floor 1e-6)",
    )
    axs[1, 1].legend(fontsize=7)
    axs[1, 2].scatter(
        geometry["x"][:, 0],
        geometry["x"][:, 1],
        c=geometry["labels"],
        cmap="viridis",
        s=3,
        alpha=0.3,
    )
    axs[1, 2].set(title="Probabilistic latent structure", xlabel="x1", ylabel="x2")
    fig.suptitle(
        "Learning & Generalization Lab | measured reference experiments",
        fontsize=19,
        fontweight="bold",
    )
    save(fig, "overview")


def format_stat(frame, key, digits=4):
    values = np.asarray(frame[key], dtype=float)
    mean = float(values.mean())
    if len(values) == 1:
        return f"{mean:.{digits}f} (one run)"
    return f"{mean:.{digits}f} ± {values.std(ddof=1):.{digits}f}"


def report(data, adult, housing, vision, signal):
    lines = [
        "# Measured reference results",
        "",
        "Generated by `python scripts/build_report.py` from completed reference records.",
        "Every number below comes from a finished configured run. ± is the sample standard deviation across three seeds unless specified.",
        "",
        "## Adult: probability quality and decisions",
        "",
        "Official test: 16,281 examples. The natural, unweighted test distribution is preserved.",
        "",
        "| Model | AUROC ↑ | AP ↑ | Calibrated NLL ↓ | Accuracy | Set coverage | Set size |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, frame in adult.groupby("model", sort=False):
        lines.append(
            "| "
            + NAMES[name]
            + " | "
            + " | ".join(
                format_stat(frame, k)
                for k in [
                    "roc_auc",
                    "average_precision",
                    "log_loss",
                    "accuracy",
                    "set_coverage",
                    "set_size",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Boosting leads the declared Adult comparison. The residual network improves on logistic probability loss in some partitions but does not displace the tree baseline. These results favor choosing models from validation evidence rather than assuming a neural architecture must win.",
            "",
            "Paired test-example bootstrap for seed zero (1,000 repeats); differences are model NLL minus logistic NLL. These intervals are conditional on the fitted models:",
            "",
        ]
    )
    for result in data["classification"]["paired_seed0"]:
        lines.append(
            f"- {result['comparison']}: {result['difference']:.5f}, 95% interval [{result['ci95'][0]:.5f}, {result['ci95'][1]:.5f}]."
        )
    lines.extend(
        [
            "",
            "![Calibration diagnostics](../assets/tabular_diagnostics.png)",
            "",
            "## Housing: prediction and interval reliability",
            "",
            "Target units: $100,000. Requested interval coverage: 90%. Geographic rows hold out the easternmost longitude region; their calibration examples remain in the fitting geography.",
            "",
            "| Split | Model | RMSE ↓ | MAE ↓ | Residual conformal coverage | Mean width |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for (protocol, name), frame in housing.groupby(["protocol", "model"], sort=False):
        values = frame.copy()
        values["coverage"] = values.split_conformal.map(lambda z: z["coverage"])
        values["width"] = values.split_conformal.map(lambda z: z["mean_width"])
        lines.append(
            f"| {protocol} | {NAMES[name]} | "
            + " | ".join(format_stat(values, k) for k in ["rmse", "mae", "coverage", "width"])
            + " |"
        )
    lines.extend(
        [
            "",
            "CQR for boosting (separate quantile fits, with finite-sample calibration):",
            "",
            "| Split | Coverage | Mean width | Empty interval rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for protocol, frame in housing[housing.model == "hist_boosting"].groupby(
        "protocol", sort=False
    ):
        vals = pd.DataFrame(frame.cqr.tolist())
        lines.append(
            f"| {protocol} | "
            + " | ".join(format_stat(vals, k) for k in ["coverage", "mean_width", "empty_rate"])
            + " |"
        )
    lines.extend(
        [
            "",
            "Boosting predicts best under the random protocol but extrapolates poorly across the chosen geography. Its interval coverage drops sharply under that shift, while linear models remain broader and perform better on this holdout. Calibration alone cannot recover an exchangeability assumption that the test protocol violates. Both fitted population and test population change between protocols, so RMSE differences are not solely an estimate of a single model's degradation.",
            "",
            "## Image representations",
            "",
            "Official test: 10,000 images. The frozen encoders use three 5,000-label subsets; only the scratch low-label network is retrained for each seed. Contrastive pretraining has one seed and the full-label supervised reference has one seed.",
            "",
            "| Model / information budget | Accuracy ↑ | Calibrated NLL ↓ | Set coverage | Set size |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, frame in vision.groupby("model", sort=False):
        lines.append(
            "| "
            + NAMES[name]
            + " | "
            + " | ".join(
                format_stat(frame, k) for k in ["accuracy", "nll", "set_coverage", "set_size"]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Contrastive features substantially improve on a frozen random encoder, but this 40-epoch pretraining budget trails scratch supervision at 5,000 labels. External ImageNet transfer supplies a strong reference with extra pretraining data and a different architecture. These information budgets are explicitly unequal.",
            "",
            "Observed prediction-set coverage is empirical on the official test file. A nominal marginal guarantee averages over exchangeable calibration and test draws; it does not force exactly 90% coverage for this fixed calibration realization. In particular, contrastive prediction sets under-cover here, and the actual values are retained without test-driven cutoff adjustment.",
            "",
            "Low-label scratch accuracy under Gaussian pixel noise, standard deviation 0.1 (no adaptation):",
            "",
        ]
    )
    for row in vision[vision.model == "supervised_lowlabel"].to_dict("records"):
        lines.append(
            f"- Seed {row['seed']}: clean {row['accuracy'] * 100:.2f}%, corrupted {row['gaussian_noise_accuracy'] * 100:.2f}%."
        )
    lines.extend(
        [
            "",
            "![Deterministic error gallery](../assets/vision_errors.png)",
            "",
            "## Nonlinear communication channel",
            "",
            "The table fixes SNR at 18 dB. Three seeds, 2,000 independent 64-symbol test blocks per seed and condition (512,000 bits). Other SNR values and block-bootstrap intervals are in the raw metrics.",
            "",
            "| Test gamma | Receiver | BER ↓ | SER ↓ |",
            "|---|---|---:|---:|",
        ]
    )
    for (gamma, name), frame in signal[signal.snr_db == 18].groupby(["gamma", "model"], sort=False):
        lines.append(
            f"| {gamma} | {NAMES[name]} | {format_stat(frame, 'ber', 6)} | {format_stat(frame, 'ser', 6)} |"
        )
    lines.extend(
        [
            "",
            "Neural receivers improve on the fitted linear-window receiver. The known-channel inverse is still stronger under this particular channel because it knows the test nonlinearity and FIR. Learned receiver error increases under gamma=0.4 mismatch. Gamma changes signal amplitude as well as distortion under the declared Es/N0 convention, so a known-channel receiver can benefit from the stronger amplitude.",
            "",
            "## Density, generalization and kernels",
            "",
        ]
    )
    p = data["probabilistic"]
    lines.extend(
        [
            f"- Mixture held-out NLL: custom EM {p['heldout_nll']['custom_em']:.6f}; reference EM {p['heldout_nll']['sklearn_em']:.6f}.",
            f"- Synthetic-label ARI: Gaussian mixture {p['ari']['gmm']:.4f}; K-means {p['ari']['kmeans']:.4f}. BIC chooses {min(p['mixture_selection'], key=lambda r: r['bic'])['k']} components.",
            f"- Wine bootstrap cluster stability: mean ARI {np.mean(p['wine']['bootstrap_stability_ari']):.4f}, descriptive resampling of 178 rows.",
            "- Double descent: 100 observations, 14 nested feature budgets, three penalties, 30 repeats; exact Gaussian population MSE, with median/IQR plots.",
            f"- Logistic GD: final hard-margin direction cosine {data['generalization']['implicit_bias'][-1]['direction_cosine']:.8f} after 30,000 steps.",
            "",
            "| Kernel study | Training labels | Test accuracy |",
            "|---|---:|---:|",
        ]
    )
    for (name, size), frame in pd.DataFrame(data["kernels"]["rows"]).groupby(
        ["model", "train_size"], sort=False
    ):
        lines.append(f"| {NAMES[name]} | {size} | {format_stat(frame, 'accuracy')} |")
    lines.extend(
        [
            "",
            "## Completed budgets and provenance",
            "",
            "Wall time includes selection and evaluation, with studies running concurrently. It is not a hardware-normalized algorithm ranking.",
            "",
            "| Study | Completion | Wall seconds | Raw evidence |",
            "|---|---|---:|---|",
        ]
    )
    for name in data:
        record = json.loads((RESULTS / name / "run.json").read_text())
        lines.append(
            f"| {name} | {record['status']} | {record['elapsed_seconds']:.1f} | [metrics](../results/reference/{name}/metrics.json), [run](../results/reference/{name}/run.json) |"
        )
    lines.extend(
        [
            "",
            "Reference configs, versions, dates and exact partitions accompany the records. Original datasets and model checkpoints are excluded from Git. Compact figure inputs are included under `assets/source_data`, so the report can be regenerated without repeating training.",
            "",
            "The benchmark is intentionally finite: three partitions are not a broad multi-dataset benchmark; the synthetic channel is not a physical fiber model; one contrastive pretraining seed does not estimate pretraining variance; and geographic stress testing is not a distribution-shift guarantee. See [model cards](09_model_cards.md) and [reproduction](07_reproduction.md).",
            "",
        ]
    )
    (ROOT / "docs" / "results.md").write_text("\n".join(lines))
    a = adult[adult.model == "hist_boosting"].roc_auc.mean()
    v = vision[vision.model == "supervised_full"].accuracy.iloc[0]
    c = signal[(signal.model == "cnn") & (signal.gamma == 0.2) & (signal.snr_db == 18)].ber.mean()
    block = (
        "<!-- RESULTS_START -->\n"
        f"**Completed reference results:** Adult boosting AUROC **{a:.4f}** (three seeds); "
        f"CIFAR-10 full supervised accuracy **{v * 100:.2f}%** (one run); "
        f"temporal receiver BER **{c:.6f}** at 18 dB (three seeds). "
        "The housing study documents pronounced coverage loss under geographic shift. "
        "Read the [full report](docs/results.md) for protocols and limitations.\n"
        "<!-- RESULTS_END -->"
    )
    path = ROOT / "README.md"
    text = path.read_text()
    start, end = (
        text.index("<!-- RESULTS_START -->"),
        text.index("<!-- RESULTS_END -->") + len("<!-- RESULTS_END -->"),
    )
    path.write_text(text[:start] + block + text[end:])


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    data, adult, housing, vision, signal = tables()
    figures(data, adult, housing, vision, signal)
    report(data, adult, housing, vision, signal)
    print("Rebuilt measured report and ten figure sets.")
