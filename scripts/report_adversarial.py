"""Regenerate controlled GAN-variant comparisons, including validation-retained starts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from validate_adversarial import records

from learning_lab.artifacts import write_json


def paired_interval(values):
    rng = np.random.default_rng(9183)
    draws = values[rng.integers(0, len(values), (2000, len(values)))].mean(1)
    return float(values.mean()), np.quantile(draws, [0.025, 0.975]).tolist()


def main():
    config, evidence = records()
    rows, paired = [], []
    for record in evidence:
        row = record["rows"][0]
        rows.append(
            dict(
                task=record["task"],
                method=record["method"],
                seed=record["seed"],
                selected_epoch=record["best_epoch"],
                validation_loss=record["selection_objective"],
                fit_seconds=sum(r["elapsed_seconds"] for r in record["history"]),
                **{key: value["mean"] for key, value in row.items() if isinstance(value, dict)},
            )
        )
    frame = pd.DataFrame(rows)
    frame.to_csv("docs/adversarial_individual_fits.csv", index=False)
    grouped = frame.groupby(["task", "method"])[["psnr", "ssim", "lpips", "fit_seconds"]].agg(
        ["mean", "std"]
    )
    grouped.to_csv("docs/adversarial_aggregates.csv")
    public = Path("assets/source_data/adversarial_variants")
    for task in config["tasks"]:
        baseline = []
        for seed in config["seeds"]:
            with np.load(
                public / f"{task}_direct_finetune_{seed}_scores.npz", allow_pickle=False
            ) as values:
                baseline.append(
                    {
                        key: values[f"direct_finetune_{seed}_sample0_{key}"]
                        for key in ["psnr", "ssim", "lpips"]
                    }
                )
        for method in config["methods"]:
            if method == "direct_finetune":
                continue
            differences = {key: [] for key in ["psnr", "ssim", "lpips"]}
            for seed in config["seeds"]:
                with np.load(
                    public / f"{task}_{method}_{seed}_scores.npz", allow_pickle=False
                ) as values:
                    for key in differences:
                        differences[key].append(
                            values[f"{method}_{seed}_sample0_{key}"] - baseline[seed][key]
                        )
            for key, arrays in differences.items():
                mean, interval = paired_interval(np.mean(arrays, axis=0))
                paired.append(
                    dict(
                        task=task,
                        method=method,
                        metric=key,
                        mean_delta=mean,
                        image_bootstrap_ci95=interval,
                        paired_images=config["test_images"],
                        scope="Paired image resampling after averaging two fitted seeds; conditional on those fits, not seed uncertainty",
                    )
                )
    write_json("docs/adversarial_paired_comparisons.json", dict(rows=paired))
    lines = [
        "# Controlled adversarial restoration variants",
        "",
        "Twenty full fits compare two tasks, five objectives and two seeds. All generators start from the same seed-specific validation-selected direct checkpoint. The direct fine-tuning control receives the same additional epochs, supervised losses, architecture, data order, learning rate and EMA schedule. Critic objective and regularization differ; their compute costs differ too.",
        "",
        "Every fit completes 20 epochs and 10,000 generator updates using 16,000 training photographs. Selection uses 128 independent validation photographs and includes epoch 0. Scoring uses the same 400 held-out 256px photographs as the earlier restoration studies, with metric protocol 3. The initial model remains selected if all later checkpoints worsen validation. No test metric selects checkpoints, objectives or photographs.",
        "",
        "## Objectives and scope",
        "",
        "| Variant | Critic loss | Generator adversarial loss | Stabilization |",
        "|---|---|---|---|",
        "| direct_finetune | No critic | None | Supervised and frozen-feature control |",
        "| hinge_sn | Half mean hinge loss | Negative mean fake score | Spectral normalization |",
        "| non_saturating_r1 | Logistic softplus | Softplus of negative fake score | Spectral normalization and lazy real-image R1 |",
        "| least_squares | Half squared real/fake targets | Half squared real-target loss | Spectral normalization |",
        "| wasserstein_gp | Mean fake minus mean real | Negative mean fake score | RGB-interpolant gradient penalty, fixed condition |",
        "",
        "Critics return mean patch scores per image. Gradient penalties differentiate only RGB, use float32 including second derivatives, and keep the observed condition fixed. WGAN-GP uses one critic update per generator; this controlled conditional adaptation does not reproduce the original paper's full training schedule. R1 uses gamma 10 every 16 updates with interval compensation. Generator objectives also include field L1 and 0.15 frozen ImageNet feature L1; adversarial weight is 0.02.",
    ]
    for task in config["tasks"]:
        data = grouped.loc[task].reindex(config["methods"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for ax, metric, title in zip(axes, ["psnr", "lpips"], ["PSNR ↑", "LPIPS ↓"]):
            ax.barh(data.index, data[(metric, "mean")], xerr=data[(metric, "std")], color="#087b94")
            ax.invert_yaxis()
            ax.set(xlabel=title, title=task.replace("_", " "))
            ax.grid(axis="x", alpha=0.2)
        for extension in ["png", "pdf"]:
            fig.savefig(f"assets/adversarial_{task}.{extension}", dpi=170)
        plt.close(fig)
        lines.extend(
            [
                "",
                "## " + task.replace("_", " ").title(),
                "",
                "| Variant | PSNR, mean ± fit SD | SSIM | LPIPS, mean ± fit SD | Selected epochs |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for method, row in data.iterrows():
            epochs = (
                frame[(frame.task == task) & (frame.method == method)]
                .sort_values("seed")
                .selected_epoch.astype(int)
                .tolist()
            )
            lines.append(
                f"| {method} | {row[('psnr', 'mean')]:.3f} ± {row[('psnr', 'std')]:.3f} | {row[('ssim', 'mean')]:.4f} | {row[('lpips', 'mean')]:.4f} ± {row[('lpips', 'std')]:.4f} | {epochs} |"
            )
        lines.extend(
            [
                "",
                f"![{task} objectives](../assets/adversarial_{task}.png)",
                "",
                "### Paired deltas against direct fine-tuning",
                "",
                "| Variant | ΔPSNR, image CI | ΔLPIPS, image CI |",
                "|---|---|---|",
            ]
        )
        for method in config["methods"][1:]:
            selected = {
                r["metric"]: r for r in paired if r["task"] == task and r["method"] == method
            }
            strings = []
            for key in ["psnr", "lpips"]:
                item = selected[key]
                a, b = item["image_bootstrap_ci95"]
                strings.append(f"{item['mean_delta']:+.4f} [{a:+.4f}, {b:+.4f}]")
            lines.append(f"| {method} | {' | '.join(strings)} |")
    retained = int((frame.selected_epoch == 0).sum())
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            f"Validation retains the warm start in {retained} of 20 fits. Completion means the declared budgets finished, including unsuccessful updates. Small objective differences, weak chroma or identical selected images are retained. Two-seed SD describes fitting variation; paired image intervals condition on those two fits and average their per-image differences before resampling.",
            "",
            "These conditional deterministic generators do not measure unconditional GAN diversity, mode coverage or state-of-the-art text-to-image generation. The earlier [local/external modern study](modern_results.md) supplies CVAE/diffusion draws and larger pretrained references. A variant's higher PSNR need not imply lower LPIPS or better exact chroma; no single metric establishes visual realism.",
            "",
            "```bash",
            "python scripts/run_adversarial_suite.py --devices cuda:0 cuda:1 cuda:2",
            "python scripts/report_adversarial.py",
            "python scripts/validate_adversarial.py",
            "python scripts/restore_generative.py --checkpoint results/reference/adversarial_variants/colorization/wasserstein_gp/0/checkpoints/selected.pt --input gray.png --output color.png",
            "```",
            "",
            "Prepare the original L1 warm-start checkpoints first. The saved model is the selected EMA generator; critic weights are training state and are unnecessary for inference. Use [portable configs](../scripts/make_portable_configs.py) for your cache/results roots. All checkpoints remain private; numeric arrays permit independent release validation.",
            "",
            "Primary references: [WGAN-GP](https://arxiv.org/abs/1704.00028), [spectral normalization](https://arxiv.org/abs/1802.05957), [R1 and GAN convergence](https://arxiv.org/abs/1801.04406), and [least-squares GAN](https://arxiv.org/abs/1611.04076). Explore all fixed photographs, seeds and objectives in [the local demo](explorer.html).",
        ]
    )
    Path("docs/adversarial_results.md").write_text("\n".join(lines) + "\n")
    print("Reported all 20 controlled fits; validation retained", retained, "warm starts")


if __name__ == "__main__":
    main()
