"""Validate and report all fixed final-epoch adversarial diagnostics."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from learning_lab.artifacts import write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config = json.loads(Path("configs/adversarial_endpoints.json").read_text())
    training_config = json.loads(Path(config["source_config"]).read_text())
    rows = []
    for task in training_config["tasks"]:
        for method in training_config["methods"]:
            for seed in training_config["seeds"]:
                directory = Path(config["output"]) / task / method / str(seed)
                record = json.loads((directory / "run.json").read_text())
                metrics = json.loads((directory / "metrics.json").read_text())
                assert record["status"] == "complete"
                assert (
                    record["config"]["endpoint_config"] == config
                    and record["config"]["training_config"] == training_config
                )
                assert (
                    record["config"]["task"],
                    record["config"]["method"],
                    record["config"]["seed"],
                ) == (task, method, seed)
                for name, value in record["evaluation_source_sha256"].items():
                    assert digest(name) == value
                training = Path(training_config["output"]) / task / method / str(seed)
                assert digest(training / "metrics.json") == metrics["training_metrics_sha256"]
                source = json.loads((training / "metrics.json").read_text())
                assert metrics["final_epoch"] == source["history"][-1]["epoch"] == 20
                assert metrics["validation_loss"] == source["history"][-1]["validation_loss"]
                assert metrics["selected_epoch_in_training"] == source["best_epoch"]
                assert len(metrics["rows"]) == 1
                row = metrics["rows"][0]
                alias = method + "_final_epoch"
                assert (
                    row["method"],
                    row["seed"],
                    row["sample"],
                    row["metric_protocol_version"],
                ) == (alias, seed, 0, 3)
                stem = Path("assets/source_data/adversarial_endpoints") / f"{task}_{method}_{seed}"
                for suffix in ["_scores", "_gallery"]:
                    assert digest(str(stem) + suffix + ".npz") == metrics["numeric_sha256"][suffix]
                with np.load(str(stem) + "_scores.npz", allow_pickle=False) as scores:
                    for key, value in row.items():
                        if isinstance(value, dict) and "mean" in value:
                            array = scores[f"{alias}_{seed}_sample0_{key}"]
                            assert len(array) == training_config["test_images"] == 400
                            assert np.isfinite(array).all()
                            assert np.isclose(array.mean(), value["mean"], atol=1e-12, rtol=0)
                with np.load(str(stem) + "_gallery.npz", allow_pickle=False) as gallery:
                    assert set(gallery.files) == {"original", "input", f"{alias}_{seed}_sample0"}
                    assert all(
                        gallery[key].dtype == np.uint8 and gallery[key].shape == (4, 3, 256, 256)
                        for key in gallery.files
                    )
                rows.append(
                    dict(
                        task=task,
                        method=method,
                        seed=seed,
                        policy="fixed_final_epoch_ema",
                        validation_loss=metrics["validation_loss"],
                        validation_selected_epoch=source["best_epoch"],
                        **{
                            key: value["mean"]
                            for key, value in row.items()
                            if isinstance(value, dict) and "mean" in value
                        },
                    )
                )
    assert len(rows) == 20
    frame = pd.DataFrame(rows)
    frame.to_csv("docs/adversarial_endpoints_individual.csv", index=False)
    selected = pd.read_csv("docs/adversarial_individual_fits.csv")
    lines = [
        "# What the final adversarial updates changed",
        "",
        config["scope"],
        "",
        "All 20 fixed endpoints are evaluated. This follow-up was designed after observing validation-retained warm starts, before choosing or scoring any final endpoint by test performance. It adds no fitting and uses no best-of-test selection. The deployment study remains [validation-selected](adversarial_results.md). The endpoint comparison reveals changes and damage that selection may otherwise hide.",
        "",
        "Tables average the two fitted seeds. The 400 photographs, observation process and metric protocol are unchanged. Endpoint gallery arrays are quantized for display; all metrics use full float images.",
    ]
    for task in training_config["tasks"]:
        end = frame[frame.task == task].groupby("method")[["psnr", "ssim", "lpips"]].mean()
        validation = (
            selected[selected.task == task].groupby("method")[["psnr", "ssim", "lpips"]].mean()
        )
        lines.extend(
            [
                "",
                "## " + task.replace("_", " ").title(),
                "",
                "| Variant | Selected PSNR | Final PSNR | Selected LPIPS | Final LPIPS |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method in training_config["methods"]:
            lines.append(
                f"| {method} | {validation.loc[method, 'psnr']:.3f} | {end.loc[method, 'psnr']:.3f} | {validation.loc[method, 'lpips']:.4f} | {end.loc[method, 'lpips']:.4f} |"
            )
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for ax, metric in zip(axes, ["psnr", "lpips"]):
            for i, method in enumerate(training_config["methods"]):
                ax.plot(
                    [validation.loc[method, metric], end.loc[method, metric]],
                    [i, i],
                    color="#8999a4",
                )
            ax.scatter(
                validation.loc[training_config["methods"], metric],
                np.arange(5),
                label="Validation-selected",
                color="#087b94",
            )
            ax.scatter(
                end.loc[training_config["methods"], metric],
                np.arange(5),
                label="Fixed final epoch20",
                color="#d48216",
            )
            ax.set(
                yticks=np.arange(5),
                yticklabels=training_config["methods"],
                xlabel=metric.upper(),
                title=task.replace("_", " "),
            )
            ax.invert_yaxis()
            ax.legend()
            ax.grid(axis="x", alpha=0.2)
        for suffix in ["png", "pdf"]:
            fig.savefig(f"assets/adversarial_endpoints_{task}.{suffix}", dpi=170)
        plt.close(fig)
        lines.extend(
            ["", f"![{task} endpoint changes](../assets/adversarial_endpoints_{task}.png)"]
        )
    lines.extend(
        [
            "",
            "## Use the comparison",
            "",
            "In [the explorer](explorer.html), each controlled family has a separate **fixed epoch20** entry. It preserves both seeds and the same four predetermined photographs. Vary the method while keeping photograph/seed fixed to see whether updates add plausible color/detail, distort the observation, or erase fidelity. Neither visual vividness nor a GAN label establishes accuracy or realism.",
            "",
            "```bash",
            "python scripts/evaluate_adversarial_endpoints.py --device cuda:0",
            "python scripts/report_adversarial_endpoints.py",
            "python scripts/restore_generative.py --checkpoint results/reference/adversarial_endpoints/colorization/wasserstein_gp/0/checkpoints/final_epoch.pt --input gray.png --output final_color.png",
            "```",
            "",
            "The sidecar identifies the fixed-final policy and epoch, while the checkpoint retains the base generator family. Read [all individual endpoints](adversarial_endpoints_individual.csv) and [validation](adversarial_endpoints_validation.json).",
        ]
    )
    Path("docs/adversarial_endpoints.md").write_text("\n".join(lines) + "\n")
    write_json(
        "docs/adversarial_endpoints_validation.json",
        dict(
            status="passed",
            endpoint_evaluations=20,
            fixed_epoch=20,
            independent_image_means=True,
            original_training_preserved=True,
            source_and_numeric_hashes=True,
            new_training_updates=0,
        ),
    )
    print("Verified and reported all 20 fixed endpoints")


if __name__ == "__main__":
    main()
