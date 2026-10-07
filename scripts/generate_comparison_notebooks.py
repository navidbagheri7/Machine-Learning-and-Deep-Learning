"""Generate focused English walkthroughs for prediction, density and GAN comparisons."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = """from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
root = Path.cwd() if Path("configs").exists() else Path.cwd().parent
sys.path.insert(0, str(root / "src"))
"""


def save(filename, cells):
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}
        },
    )
    nbf.write(notebook, ROOT / "notebooks" / filename)


def main():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    save(
        "20_classical_prediction.ipynb",
        [
            md(
                "# Classical prediction, modern ensembles and honest uncertainty\n\nLDA/QDA, neighbors, kernels and linear penalties remain useful modeling choices. Compare them with bagging, extra trees, adaptive/histogram/native boosting, voting and cross-validated stacking. The 243 selected models share isolated fitting, selection and calibration roles; candidate and compute budgets differ. Regression includes geographic shift."
            ),
            code(SETUP),
            md(
                "## Probability scores and data roles\n\nA binary linear SVM returns a margin; use two ordered logits before temperature calibration. Temperature and conformal scores need distinct calibration data. A test threshold slider explores outcomes; it cannot retrospectively select a deployable threshold."
            ),
            code("""from scipy.special import softmax
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from learning_lab.classical import logits
x, y = make_classification(n_samples=120, n_features=5, random_state=8)
svm = LinearSVC(dual="auto").fit(x[:80], y[:80])
scores = logits(svm, x[80:])
assert np.array_equal(scores.argmax(1), svm.predict(x[80:]))
pd.DataFrame(softmax(scores[:5], axis=1), columns=["class0", "class1"])"""),
            md(
                "## All fitted families\n\nLog-loss measures probability quality, macro-F1 weights classes equally, and accuracy depends on the decision rule. Rankings need not agree. The bundled diagnostic datasets do not establish clinical readiness."
            ),
            code("""fits = pd.read_csv(root / "docs/classical_individual_fits.csv")
assert len(fits) == 243
fits.groupby(["case", "model"])[["accuracy", "log_loss", "rmse"]].mean().dropna(how="all").round(4)"""),
            md(
                "## Geographic uncertainty\n\nAbsolute-residual conformal intervals target 90% coverage under exchangeability. The geographic test can violate that assumption. Report coverage and width together, and preserve failures."
            ),
            code("""housing = fits[fits.case.str.startswith("housing")]
summary = housing.groupby(["case", "model"])[["rmse", "interval_coverage", "interval_width"]].mean()
summary.loc[(slice(None), ["ridge", "catboost", "stacking"]), :]"""),
            md(
                "## Stacking without preprocessing leakage\n\nEach base estimator owns its preprocessing pipeline. The outer fitting partition contains the inner folds. Scaling and imputation refit inside those folds; out-of-fold predictions train the final estimator. Selection labels and calibration labels are never inner-fold fitting data."
            ),
            code("""protocol = json.loads((root / "results/reference/classical/digits/0/metrics.json").read_text())
stack = next(row for row in protocol["rows"] if row["model"] == "stacking")
{key: stack[key] for key in ["candidate_count", "selected_candidate", "validation_macro_f1", "seconds_including_selection"]}"""),
            md(
                "Reproduce with `python scripts/run_classical.py --dataset digits --seed 0`; verify every public prediction with `python scripts/validate_classical.py`. Read [classical results](../docs/classical_results.md), and open [the explorer](../docs/explorer.html) for confusion matrices, threshold controls and interval plots."
            ),
        ],
    )
    save(
        "21_clustering_geometry.ipynb",
        [
            md(
                "# Clustering geometry, priors and rejected points\n\nCompare centroid, full-covariance/variational mixture, Ward, spectral, DBSCAN and HDBSCAN clustering. Seven methods × three datasets × three seeds produce 63 fits. This is transductive analysis: full-data standardization/PCA uses all unlabeled examples. Labels are diagnostic only."
            ),
            code(SETUP),
            md(
                "## Count priors and density decisions\n\nKnown cluster counts 3/3/10 are explicit priors for count-constrained families. Variational mixtures use a larger component cap; density methods determine count and noise. This difference prevents an unqualified claim of equal information budgets."
            ),
            code("""result = json.loads((root / "results/reference/unsupervised/metrics.json").read_text())
fits = pd.DataFrame(result["rows"])
assert len(fits) == 63
fits.groupby(["case", "method"])[["ari_all_points", "nonnoise_fraction", "clusters"]].mean().round(4)"""),
            md(
                "## Silhouette can reward selective rejection\n\nSilhouette computed only on accepted points must be read with coverage. All-points ARI includes rejected points. PCA is a display, not the space in which Digits was clustered."
            ),
            code("""from sklearn.metrics import adjusted_rand_score
with np.load(root / "assets/source_data/clustering/digits.npz", allow_pickle=False) as values:
    truth = values["true_labels"]
    labels = values["hdbscan_0"]
    assert adjusted_rand_score(truth, labels) == fits.query("case == 'digits' and method == 'hdbscan' and seed == 0").ari_all_points.iloc[0]
    print("Accepted fraction:", (labels >= 0).mean(), "ARI:", adjusted_rand_score(truth, labels))"""),
            md(
                "## Display the complete variable-density construction\n\nThe example intentionally has unequal densities. Compare every method rather than only a favorable subset. True labels were not used to choose density presets."
            ),
            code("""with np.load(root / "assets/source_data/clustering/variable_density.npz", allow_pickle=False) as values:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, method in zip(axes, ["kmeans", "gmm", "hdbscan"]):
        ax.scatter(*values["projection"].T, c=values[method + "_0"], s=6, cmap="tab10")
        ax.set_title(method)
    plt.show()"""),
            md(
                "## Separate fitting randomness from data robustness\n\nDeterministic families repeat identical fits across seeds. A zero SD does not establish stability to resampling, feature noise, or a new population."
            ),
            code(
                """fits.groupby(["case", "method"])[["ari_all_points", "nonnoise_fraction"]].agg(["mean", "std"]).round(4)"""
            ),
            md(
                "Reproduce with `python scripts/run_clustering.py`; verify and report with `python scripts/report_clustering.py`. Read [clustering results](../docs/clustering_results.md) and [the interactive projection](../docs/explorer.html)."
            ),
        ],
    )
    save(
        "22_adversarial_objectives.ipynb",
        [
            md(
                "# Adversarial objectives and controlled restoration\n\nCompare hinge/SN, non-saturating/R1, least-squares and Wasserstein-GP with direct fine-tuning on colorization and 4× SR. All five share the same generator warm start within each seed. Selection includes epoch 0; unsuccessful changes remain visible. The experiment measures conditional deterministic restoration, not unconditional generation."
            ),
            code(
                SETUP
                + "\nimport torch\nfrom torch import nn\nfrom learning_lab.adversarial import gradient_penalty, r1_penalty, critic_objective, generator_objective\n"
            ),
            md(
                "## Critic and generator signs\n\nThe Wasserstein critic minimizes mean fake minus mean real. The generator minimizes negative mean fake. R1 differentiates real examples; WGAN-GP interpolates real/fake RGB while keeping the observed condition fixed."
            ),
            code("""real = torch.tensor([2.0, 3.0], requires_grad=True)
fake = torch.tensor([-1.0, 0.0], requires_grad=True)
pd.DataFrame([{ "method": method, "critic": float(critic_objective(real, fake, method).detach()), "generator": float(generator_objective(fake, method).detach()) }
    for method in ["hinge_sn", "non_saturating_r1", "least_squares", "wasserstein_gp"]])"""),
            md(
                "## An analytic gradient-penalty check\n\nA scalar linear critic with a unit RGB gradient has zero interpolant penalty. The condition contributes to its score but does not enter the RGB norm. Second derivatives must remain in the graph for trainable critics."
            ),
            code("""class LinearCritic(nn.Module):
    def forward(self, condition, rgb):
        return rgb.flatten(1).sum(1) / (rgb[0].numel() ** 0.5) + condition.flatten(1).sum(1)
rgb = torch.randn(3, 3, 4, 4)
condition = torch.randn(3, 1, 4, 4)
gp, norm = gradient_penalty(LinearCritic(), condition, rgb, torch.zeros_like(rgb))
assert float(gp) < 1e-12
print("Gradient norm:", float(norm), "Penalty:", float(gp))"""),
            md(
                "## All epochs and selected checkpoints\n\nValidation uses field L1 plus a frozen-feature loss. It is not a perceptual-realism oracle. If later adversarial updates worsen that criterion, the initial model remains selected after the full budget completes."
            ),
            code("""history = json.loads((root / "results/reference/adversarial_variants/colorization/hinge_sn/0/metrics.json").read_text())
assert history["best_epoch"] == int(np.argmin([row["validation_loss"] for row in history["history"]]))
pd.DataFrame(history["history"])[["epoch", "validation_loss"]].plot(x="epoch", y="validation_loss", title="Hinge/SN colorization, seed 0")
plt.show()"""),
            md(
                "## Fidelity, perception and uncertainty\n\nSeed SD describes fit variability. Image intervals resample the same paired photos after averaging two seeds; they condition on those fits. These are different sources of uncertainty."
            ),
            code("""fits = pd.read_csv(root / "docs/adversarial_individual_fits.csv")
assert len(fits) == 20
fits.groupby(["task", "method"])[["psnr", "ssim", "lpips", "selected_epoch"]].mean().round(4)"""),
            md(
                "Read [objectives and results](../docs/adversarial_results.md). Use `scripts/run_adversarial_suite.py` after preparing the original direct checkpoints. Restore your own observation with `scripts/restore_generative.py`. The [browser gallery](../docs/explorer.html) retains all four fixed photos, all methods, seeds and original stochastic draws."
            ),
        ],
    )
    print("Generated three comparison walkthroughs")


if __name__ == "__main__":
    main()
