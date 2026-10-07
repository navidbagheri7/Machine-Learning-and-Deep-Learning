"""Generate the English numerical walkthroughs; execute with execute_notebooks.py."""

import inspect
import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = """
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
root = Path.cwd()
while not (root / 'pyproject.toml').exists() and root != root.parent:
    root = root.parent
sys.path.insert(0, str(root / 'src'))
def evidence(study):
    return json.loads((root / 'results' / 'reference' / study / 'metrics.json').read_text())
plt.rcParams.update({'figure.figsize': (7, 4), 'axes.spines.top': False, 'axes.spines.right': False})
"""


def write(name, cells):
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(inspect.cleandoc(c))
        if kind == "m"
        else nbf.v4.new_code_cell(textwrap.dedent(c).strip())
        for kind, c in cells
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    nbf.write(notebook, ROOT / "notebooks" / name)


def main():
    write(
        "01_gaussian_posterior.ipynb",
        [
            (
                "m",
                r"""# Gaussian posterior: functions, assumptions and uncertainty
        Derive a conjugate posterior, verify its relationship with ridge, and inspect
        how uncertainty changes outside the observed region. Read the full derivation in
        [the probabilistic chapter](../docs/01_probabilistic_models.md).""",
            ),
            ("c", SETUP),
            (
                "m",
                r"""## Model and numerical solution
        With $w\sim N(0,\tau^2I)$ and $y|w\sim N(Xw,\sigma^2I)$, posterior precision
        is $\tau^{-2}I+\sigma^{-2}X^TX$. An intercept is explicitly a basis column;
        it is regularized under the same prior here.""",
            ),
            (
                "c",
                """from learning_lab.probabilistic import BayesianLinearRegression
from sklearn.linear_model import Ridge
rng = np.random.default_rng(8)
x = np.sort(rng.uniform(-1, 1, 60))
y = np.sin(3*x) + rng.normal(0, .15, len(x))
def basis(t):
    return np.polynomial.legendre.legvander(t, 5)
posterior = BayesianLinearRegression(1, .15**2).fit(basis(x), y)
ridge = Ridge(alpha=.15**2, fit_intercept=False).fit(basis(x), y)
np.testing.assert_allclose(posterior.mean_, ridge.coef_, atol=1e-10)
print('Posterior/ridge coefficient discrepancy:', np.max(abs(posterior.mean_-ridge.coef_)))""",
            ),
            (
                "m",
                """## Epistemic and observation variance
        The predictive variance adds observation noise to basis-dependent coefficient
        uncertainty. Inspect the function, not just a coefficient table.""",
            ),
            (
                "c",
                """grid = np.linspace(-2, 2, 300)
mean, total_sd = posterior.predict(basis(grid))
_, parameter_sd = posterior.predict(basis(grid), include_noise=False)
np.testing.assert_allclose(total_sd**2-parameter_sd**2, .15**2)
plt.fill_between(grid, mean-1.645*total_sd, mean+1.645*total_sd, alpha=.25, label='90% predictive interval')
plt.plot(grid, mean, label='Posterior mean')
plt.plot(grid, np.sin(3*grid), '--', label='True mean')
plt.scatter(x, y, s=12)
plt.ylim(-5, 5); plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.show()""",
            ),
            (
                "m",
                """## Data reduce posterior parameter uncertainty
        Compare the same prior with ten versus sixty observations. In this conjugate
        model extra data can only reduce the covariance in the positive-semidefinite order.""",
            ),
            (
                "c",
                """small = BayesianLinearRegression(1, .15**2).fit(basis(x[:10]), y[:10])
_, small_sd = small.predict(basis(grid), include_noise=False)
assert np.all(parameter_sd <= small_sd + 1e-9)
print(pd.Series(evidence('probabilistic')['posterior']))""",
            ),
            (
                "m",
                r"""## Exercise and worked answer
        Would doubling every basis value while keeping the coefficient prior unchanged
        preserve the prior over functions? No. The prior function variance is
        $\tau^2\phi(x)\phi(x)^T$, so doubling the basis multiplies it by four.
        Posterior confidence also depends on the feature family: polynomial uncertainty
        does not account for every possible nonpolynomial alternative.""",
            ),
        ],
    )
    write(
        "02_tabular_decisions.ipynb",
        [
            (
                "m",
                """# Tabular prediction: information boundaries and decisions
        Inspect the completed Adult benchmark, verify partition isolation, and derive an
        operating threshold from a declared utility. [Protocol](../docs/02_tabular_decisions.md).""",
            ),
            ("c", SETUP),
            (
                "c",
                """metrics = evidence('classification')
table = pd.DataFrame(metrics['rows'])
table.groupby('model')[['roc_auc', 'average_precision', 'log_loss', 'brier', 'set_coverage', 'set_size']].agg(['mean', 'std'])""",
            ),
            (
                "m",
                """## Confirm that four training partitions do not overlap
        The official test is a separate file. Preprocessing sees fitting rows only.
        Calibration is distinct from settings and threshold selection.""",
            ),
            (
                "c",
                """split = json.loads((root/'results/reference/classification/split_0.json').read_text())
parts = [np.array(split[k]) for k in ['fit', 'validation', 'probability_calibration', 'conformal_calibration']]
assert len(np.unique(np.concatenate(parts))) == sum(map(len, parts))
assert sum(map(len, parts)) == metrics['data']['train']
print({k: len(split[k]) for k in ['fit', 'validation', 'probability_calibration', 'conformal_calibration']})""",
            ),
            (
                "m",
                r"""## Decisions are not the same as probability ranking
        If a false positive costs 1 and a false negative costs 2, predicting positive
        has expected cost $1-p$ and predicting negative costs $2p$. The Bayes threshold
        is therefore $1/3$, under calibrated probabilities and those particular costs.""",
            ),
            (
                "c",
                """probabilities = np.linspace(0, 1, 200)
plt.plot(probabilities, 1-probabilities, label='Positive action cost')
plt.plot(probabilities, 2*probabilities, label='Negative action cost')
plt.axvline(1/3, color='gray', linestyle='--')
plt.xlabel('Calibrated positive probability'); plt.ylabel('Expected cost'); plt.legend(); plt.show()
table[['model', 'seed', 'decision_threshold', 'decision_cost']]""",
            ),
            (
                "m",
                """## Diagnostic subgroup behavior
        Marginal coverage does not guarantee every subgroup's coverage. Keep sample sizes
        next to scores and avoid turning descriptive differences into causal claims.""",
            ),
            (
                "c",
                """groups = pd.DataFrame(metrics['diagnostics_seed0']['hist_boosting']['groups'])
groups[['feature','group','n','roc_auc','log_loss','set_coverage','tpr','fpr']]""",
            ),
            ("c", """pd.DataFrame(metrics['paired_seed0'])"""),
            (
                "m",
                """## Exercise and worked answer
        Can model selection use the conformal-calibration labels and still rely on the
        simple split-conformal proof? No. Those scores need to come from a predictor fixed
        without fitting or selection on the calibration labels. A separate calibration
        partition makes the data dependence explicit. The actual income benchmark is
        historical; the costs above are an illustration, not a hiring recommendation.""",
            ),
        ],
    )
    write(
        "03_intervals_and_shift.ipynb",
        [
            (
                "m",
                """# Conformal ranks and geographic failure
        A finite-sample rank gives a precise marginal statement under exchangeability.
        A shifted test population need not satisfy it. [Chapter](../docs/03_uncertainty_and_shift.md).""",
            ),
            ("c", SETUP),
            (
                "c",
                """from learning_lab.uncertainty import conformal_quantile, regression_interval
scores = np.arange(9, dtype=float)
q = conformal_quantile(scores, alpha=.2)
assert q == 7  # ceil((9+1)*.8) is the eighth order statistic
assert np.isinf(conformal_quantile(np.array([.1, .2]), alpha=.1))
print('Correct finite-sample cutoff:', q)""",
            ),
            (
                "m",
                """## Repeated exchangeable draws versus one shifted population
        The following simulation reuses no fitting labels. Calibration residuals have
        zero mean; the shifted observations have a systematic mean error.""",
            ),
            (
                "c",
                """rng = np.random.default_rng(4)
cal = rng.normal(size=500)
test = rng.normal(size=5000)
shifted = test + 2
lower, upper = regression_interval(np.zeros(len(test)), np.zeros(len(cal)), cal)
print('Exchangeable empirical coverage:', np.mean((test>=lower)&(test<=upper)))
print('Shifted empirical coverage:', np.mean((shifted>=lower)&(shifted<=upper)))""",
            ),
            (
                "m",
                """## Inspect the real housing study
        The geographic holdout uses longitude, not target labels. Both protocols refit
        models; this changes the fitting population as well as the evaluation population.""",
            ),
            (
                "c",
                """rows = pd.DataFrame(evidence('regression')['rows'])
rows['coverage'] = rows.split_conformal.map(lambda z: z['coverage'])
rows['width'] = rows.split_conformal.map(lambda z: z['mean_width'])
rows.groupby(['protocol','model'])[['rmse','coverage','width']].mean()""",
            ),
            (
                "c",
                """intervals = np.load(root/'assets/source_data/regression_intervals_geographic_0.npz')
error = abs(intervals['y']-intervals['pred'])
covered = (intervals['y']>=intervals['lower']) & (intervals['y']<=intervals['upper'])
plt.scatter(intervals['longitude'], error, c=covered, cmap='coolwarm_r', s=5, alpha=.4)
plt.xlabel('Held-out longitude'); plt.ylabel('Absolute error ($100,000)'); plt.show()
print('Observed CQR coverage in the shifted region:', covered.mean())""",
            ),
            (
                "m",
                """## Exercise and worked answer
        Does a prediction interval that includes 90% of calibration labels necessarily
        include 90% of the shifted test labels? No. The rank argument requires compatible
        calibration and test draws. The measured geographic coverage is a stress result,
        not a shifted-data theorem. Interval width and capped-target behavior also matter.""",
            ),
        ],
    )
    write(
        "04_latent_structure.ipynb",
        [
            (
                "m",
                """# Latent structure: EM, responsibilities and model selection
        Implemented Gaussian mixture EM separates density estimation from descriptive
        clustering. [Derivation](../docs/01_probabilistic_models.md).""",
            ),
            ("c", SETUP),
            (
                "c",
                """from learning_lab.probabilistic import GaussianMixtureEM
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
rng = np.random.default_rng(0)
labels = np.repeat([0,1], 250)
x = np.concatenate([rng.multivariate_normal([-2,0], [[1,.8],[.8,1]], 250),
                    rng.multivariate_normal([2,1], [[.5,-.4],[-.4,1]], 250)])
model = GaussianMixtureEM(2).fit(x)
r = model.predict_proba(x)
np.testing.assert_allclose(r.sum(axis=1), 1)
assert np.min(np.diff(model.history_)) > -1e-6
print('EM ARI:', adjusted_rand_score(labels, r.argmax(1)))
print('K-means ARI:', adjusted_rand_score(labels, KMeans(2,n_init=10,random_state=0).fit_predict(x)))""",
            ),
            (
                "c",
                """fig, axes = plt.subplots(1,2,figsize=(11,4))
axes[0].scatter(x[:,0], x[:,1], c=r[:,0], s=10, cmap='viridis')
axes[0].set_title('Posterior component-0 responsibility')
axes[1].plot(model.history_, 'o-'); axes[1].set_title('EM fitting log density')
axes[1].set_xlabel('Iteration'); plt.show()""",
            ),
            (
                "m",
                """## Density and latent labels answer different questions
        BIC is a training likelihood penalty, held-out NLL measures density prediction,
        and ARI measures a partition's match with external labels. Inspect all three.""",
            ),
            (
                "c",
                """metrics = evidence('probabilistic')
display(pd.DataFrame(metrics['mixture_selection']))
display(pd.Series(metrics['heldout_nll']))
display(pd.Series(metrics['ari']))
print('Wine descriptive stability:', np.mean(metrics['wine']['bootstrap_stability_ari']))""",
            ),
            (
                "m",
                """## Exercise and worked answer
        Why should responsibilities sum to one even when densities are tiny? They are
        normalized conditional component probabilities. Log-domain normalization avoids
        underflow in the density calculation. A covariance floor keeps factorization
        defined, but multiple starts and held-out assessment are still needed for a
        nonconvex mixture. Component indices have no intrinsic ordering.""",
            ),
        ],
    )
    write(
        "05_generalization.ipynb",
        [
            (
                "m",
                """# Interpolation, exact risk and optimization bias
        Separate training fit from population prediction and parameter norm from
        parameter direction. [Chapter](../docs/04_generalization.md).""",
            ),
            ("c", SETUP),
            (
                "m",
                r"""## Verify the exact population-risk identity
        For an independent isotropic Gaussian $x$, expected squared coefficient error
        is $\|\hat w-\beta\|^2$. Add observation noise variance for prediction of $y$.""",
            ),
            (
                "c",
                """rng = np.random.default_rng(9)
beta = np.array([1., .5, 0., 0.])
w = np.array([.8, .7, -.2, .1])
x = rng.normal(size=(100000,4))
noise = rng.normal(scale=.5,size=len(x))
empirical = np.mean((x@beta + noise - x@w)**2)
exact = np.sum((beta-w)**2) + .25
print({'empirical_mse': empirical, 'exact_mse': exact})
assert abs(empirical-exact) < .01""",
            ),
            (
                "c",
                """metrics = evidence('generalization')
rows = pd.DataFrame(metrics['double_descent'])
for penalty, frame in rows.groupby('penalty'):
    stats = frame.groupby('ratio').population_mse.quantile([.25,.5,.75]).unstack()
    plt.semilogy(stats.index, stats[.5], label=f'Penalty {penalty}')
    plt.fill_between(stats.index, stats[.25], stats[.75], alpha=.15)
plt.axvline(1,color='gray',linestyle='--')
plt.xlabel('Features / observations'); plt.ylabel('Population MSE: median and IQR'); plt.legend(); plt.show()""",
            ),
            (
                "m",
                """## Optimization can select a direction while norm diverges
        Logistic loss on separable data has no finite minimizer. Inspect both quantities
        alongside the hard-margin direction, rather than claiming parameter convergence.""",
            ),
            (
                "c",
                """implicit = pd.DataFrame(metrics['implicit_bias'])
display(implicit.iloc[[0,-1]])
fig,axes = plt.subplots(1,2,figsize=(10,4))
axes[0].semilogx(implicit.step,implicit.norm); axes[0].set_title('Parameter norm')
axes[1].loglog(implicit.step,1-implicit.direction_cosine); axes[1].set_title('Direction discrepancy')
for ax in axes: ax.set_xlabel('Gradient updates')
plt.show()""",
            ),
            (
                "m",
                """## Nonlinear features at limited label budgets
        The offline digit study uses the same test set across all budgets. Validation
        selects regularization; test labels do not determine settings.""",
            ),
            (
                "c",
                """pd.DataFrame(evidence('kernels')['rows']).groupby(['model','train_size']).accuracy.agg(['mean','std'])""",
            ),
            (
                "m",
                """## Exercise and worked answer
        Would the interpolation curve have the same interpretation if widths below ten
        omitted true signal coordinates? No. Approximation error would then change with
        width. The reference experiment includes all ten signal features at every width,
        isolating noise amplification and regularization more directly. Its conclusions
        concern the declared Gaussian model, not every deep learning architecture.""",
            ),
        ],
    )
    write(
        "06_representation_learning.ipynb",
        [
            (
                "m",
                """# Representation learning and data exposure
        Check the contrastive objective, then read the completed CIFAR experiments by
        their information budgets. [Chapter](../docs/05_representation_learning.md).""",
            ),
            ("c", SETUP),
            (
                "c",
                """import torch
from learning_lab.vision import augment, contrastive_loss, ImageEncoder
torch.set_num_threads(2)
torch.manual_seed(0)
z = torch.eye(32)
correct = float(contrastive_loss(z,z))
permuted = float(contrastive_loss(z,torch.roll(z,1,0)))
assert correct < permuted
print({'correct_pairs_loss':correct,'permuted_pairs_loss':permuted})""",
            ),
            (
                "m",
                """## A view changes appearance without using class labels
        The encoder and projection objective are separate from the downstream linear
        probe. Avoid assuming every augmentation preserves every application label.""",
            ),
            (
                "c",
                """data = np.load(root/'assets/source_data/vision_errors.npz')
image = torch.tensor(data['images'][:1],dtype=torch.float32)/255
views = augment(image.repeat(6,1,1,1),contrastive=True)
mean = torch.tensor([.4914,.4822,.4465])[None,:,None,None]
std = torch.tensor([.247,.2435,.2616])[None,:,None,None]
display_images = (views*std+mean).clamp(0,1)
fig,axes = plt.subplots(1,6,figsize=(12,2))
for ax,v in zip(axes,display_images):
    ax.imshow(v.permute(1,2,0).numpy()); ax.axis('off')
plt.show()
assert ImageEncoder()(views).shape == (6,128)""",
            ),
            (
                "c",
                """rows = pd.DataFrame(evidence('vision')['rows'])
rows.groupby('model')[['accuracy','nll','set_coverage','set_size']].agg(['mean','std','count'])""",
            ),
            (
                "m",
                """## Full supervision and corruption
        External ImageNet transfer has more information and a different architecture.
        The full scratch reference has more CIFAR labels. Frozen random features isolate
        the value of fitting the shared contrastive architecture.""",
            ),
            (
                "c",
                """rows[rows.model=='supervised_lowlabel'][['seed','accuracy','gaussian_noise_accuracy']]
""",
            ),
            (
                "c",
                """from IPython.display import Image, display
display(Image(filename=str(root/'assets/vision_errors.png')))
""",
            ),
            (
                "m",
                """## Exercise and worked answer
        What variation does the contrastive row's three-seed standard deviation measure?
        It measures fitting-label subset variation in the downstream head; the encoder's
        pretraining seed is fixed. It does not establish the variance of contrastive
        pretraining. The error gallery follows official test order, so it does not hide
        mistakes through visual selection.""",
            ),
        ],
    )
    write(
        "07_learned_receivers.ipynb",
        [
            (
                "m",
                """# Learned receivers: verify the simulator before trusting BER
        Work from symbol energy and bit mapping to exact inverse checks and measured
        receiver comparisons. [Chapter](../docs/06_signal_equalization.md).""",
            ),
            ("c", SETUP),
            (
                "c",
                """from learning_lab.signals import constellation, channel, physics_inverse, error_metrics
points, bits = constellation()
assert np.isclose(np.mean(abs(points)**2),1)
grid = bits.reshape(4,4,4)
assert np.all(np.sum(grid[1:] != grid[:-1],axis=-1)==1)
plt.scatter(points.real,points.imag)
for p,b in zip(points,bits):
    plt.annotate(''.join(map(str,b)),(p.real,p.imag),xytext=(4,4),textcoords='offset points',fontsize=8)
plt.xlabel('Real'); plt.ylabel('Imaginary'); plt.title('Unit-power Gray-coded 16-QAM'); plt.show()""",
            ),
            (
                "m",
                """## Exact reconstruction in the noiseless limit
        Circular FIR memory gives a well-defined Fourier inverse. Newton iteration
        removes the declared amplitude nonlinearity before the linear inverse.""",
            ),
            (
                "c",
                """rng = np.random.default_rng(12)
labels = rng.integers(16,size=(100,64))
transmitted = points[labels]
received = channel(transmitted,np.inf,.2,rng)
recovered = physics_inverse(received,.2)
np.testing.assert_allclose(recovered,transmitted,atol=1e-10)
prediction = np.argmin(abs(recovered[:,:,None]-points),axis=-1)
print(error_metrics(labels,prediction))""",
            ),
            (
                "m",
                """## Channel knowledge is a resource
        The inverse knows test parameters, while learned receivers are fitted at one
        channel setting. Test data is generated as independent blocks.""",
            ),
            (
                "c",
                """rows = pd.DataFrame(evidence('signals')['rows'])
rows[rows.snr_db==18].groupby(['gamma','model'])[['ber','ser']].agg(['mean','std'])""",
            ),
            (
                "c",
                """fig,axes = plt.subplots(1,2,figsize=(12,4))
for ax,gamma in zip(axes,[.2,.4]):
    for model,frame in rows[rows.gamma==gamma].groupby('model',sort=False):
        curve = frame.groupby('snr_db').ber.mean()
        ax.semilogy(curve.index,np.maximum(curve,1e-6),'o-',label=model)
    ax.set_title(f'Gamma {gamma}'); ax.set_xlabel('Es/N0 (dB)'); ax.set_ylabel('BER, display floor 1e-6')
axes[1].legend(fontsize=8); plt.show()""",
            ),
            (
                "m",
                """## Saved-model deployment and numerical precision
The receiver command takes observed circular blocks and uses no target, SNR or
nonlinearity. Its probabilities are uncalibrated. The audit reproduces every CUDA
benchmark row and records CPU differences separately: a fixed seed does not fix
cross-device arithmetic. Reference convolution allows TF32; matrix TF32 is disabled.""",
            ),
            (
                "c",
                """deployment = json.loads((root/'assets/receiver_inference_audit.json').read_text())
assert deployment['status'] == 'passed'
assert len(deployment['checks']) == 60 and len(deployment['commands']) == 12
numerics = pd.DataFrame(deployment['cpu_numerics'])
numerics.groupby('model').agg(
    evaluated_symbol_predictions=('symbols','sum'),
    changed_cpu_classes=('different_symbol_classes','sum'),
    maximum_probability_difference=('maximum_probability_difference','max'))""",
            ),
            (
                "m",
                """## Exercise and worked answer
        Why bootstrap blocks instead of individual bits? Context receivers and channel
        memory correlate errors within a block. Resampling whole independent blocks
        preserves that structure. A measured zero BER is an observation over a finite
        number of bits, not a proof of a perfect receiver. This discrete simulator does
        not establish performance on a physical optical link.""",
            ),
        ],
    )


if __name__ == "__main__":
    main()
    from generate_project_notebooks import main as projects

    projects()
    from generate_upgrade_notebooks import main as upgrades

    upgrades()
