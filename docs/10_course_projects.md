# From mixture assignments to probabilistic reconstruction

The two project phases ask for EM on supplied two-dimensional data, categorical
mixture derivations, and reconstruction of blurred/noisy MNIST patches. This chapter
completes those topics and extends them with model selection, missing-data prediction,
stable Bayesian conditioning, and an explicit audit of the supplied observations.
The original handouts and student notebooks remain private. The public implementation
is independent of notebook execution state and uses English throughout.

## What changed, and why

| Source issue | Consequence | Reconstruction |
|---|---|---|
| Euclidean nearest-center assignment described as an E-step | Conflates hard k-means and probabilistic EM | Compare k-means, weighted full-covariance EM, selected covariance families, and variational mixtures |
| Original code omitted mixture weights in responsibilities | Incorrect inference when component frequencies differ | Log-domain weighted responsibilities; check against scikit-learn and an imbalanced categorical example |
| Independent unsorted glob lists for original/noisy images | Metrics can compare different digits | Pair by numeric filename and report every unmatched ID |
| Prior patches in 0–1, observations in 0–255 | Noise and prior precision operate in incompatible units | Everything in 0–1; convert noise SD once from 0–255 |
| Component scoring evaluated the observed patch under a clean-patch posterior | Chooses components using the wrong probability | Score observation evidence under `A C Aᵀ + R` |
| Imagewise rescaling by the maximum reconstruction | Hides photometric errors and changes contrast | Clip only to the physical 0–1 range; fixed-range PSNR/SSIM |
| Single-image random tuning and test inspection | Selected results are unreliable | Fixed validation IDs; untouched test IDs; full per-image metrics |
| Large patch arrays accumulated with repeated concatenation | Quadratic copying and excess memory | Sample patch origins directly; batch conditioning; overlap averaging |

## Phase 1: Gaussian mixtures and hard assignments

For a mixture with probabilities πₖ, means μₖ, covariances Cₖ and observations xᵢ,

\[
r_{ik}=\frac{\pi_k\mathcal N(x_i;\mu_k,C_k)}
 {\sum_j\pi_j\mathcal N(x_i;\mu_j,C_j)},\quad
N_k=\sum_i r_{ik},\quad \pi_k=N_k/N,
\]
\[
\mu_k=\frac{\sum_i r_{ik}x_i}{N_k},\qquad
C_k=\frac{\sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^T}{N_k}.
\]

The complete-data likelihood is the product over observations/components of
`[πₖ N(xᵢ; μₖ,Cₖ)]` raised to a one-hot latent assignment. EM replaces those
assignments with their conditional expectations. The implementation adds a small
covariance floor; this stabilizes density evaluation and is disclosed separately
from the exact unconstrained maximum-likelihood formulas.

The handout's Euclidean assignment is hard k-means: each point has one nearest
center. It corresponds to an equal-weight isotropic small-variance limit, not the
general Gaussian E-step. It cannot model elongated clusters well. The comparison
therefore preserves this baseline while implementing actual probabilistic EM.
Responsibilities have shape `N × K` in the package; the handout uses its transpose.

Each supplied CSV has 600 rows, described as three concatenated 200-point groups.
These group labels are used only for evaluation and plots. Three independent
60/20/20 partitions separate fitting, selection and testing. Component counts 1–6
and spherical/diagonal/full covariance compete on validation density. Fit-set BIC is
reported as a separate diagnostic. Three-start custom EM, five-start library EM,
k-means, and a variational Gaussian mixture provide complementary checks.

The two supplied files are synthetic coordinates. They are not MRI pixels, patient
records, or a tumor segmentation benchmark, despite the assignment's motivating story.
Empirical generating-group means/covariances and fitted parameters are recorded in
[metrics](../results/reference/course_mixtures/metrics.json). Missing-coordinate
prediction conditions the fitted mixture on the observed coordinate and compares
against mean imputation. A nearly independent coordinate offers little predictive
information; the experiment makes that failure case visible.

## Phase 1: categorical mixtures and the ELBO

For independent categorical features conditional on component k,

\[
p(x_i,z_i=k)=\pi_k\prod_{d\in O_i}\theta_{kd,x_{id}},\quad
r_{ik}\propto\pi_k\prod_{d\in O_i}\theta_{kd,x_{id}},
\]
\[
\theta_{kdc}=\frac{\sum_{i:d\in O_i}r_{ik}\mathbf 1[x_{id}=c]}
 {\sum_{i:d\in O_i}r_{ik}}.
\]

Oᵢ is the observed-feature set. Missing entries contribute no observed likelihood
and no counts. `CategoricalMixtureEM` adds symmetric positive pseudo-counts to these
counts and mixture counts, avoiding zero probabilities; its implemented estimator is
the disclosed smoothed update. Feature-imputation probabilities average component
categorical probabilities under the posterior component weights.

A mixture of **one** unconstrained categorical variable cannot identify individual
components: only the convex combination of their probability vectors is observable.
The executed extension uses eight conditionally independent three-valued features,
imbalanced component weights and 25% missing-at-random entries. This provides
observable dependence and a meaningful imputation evaluation. The latent-class model
is compared with per-feature majority imputation on 1,000 held-out synthetic rows.
Bernoulli mixtures are the two-category special case.

For any q(z), the ELBO is `E_q log p(x,z;θ) + H(q)` and the gap to log evidence is
`KL(q || p(z|x;θ))`. The exact E-step makes this gap zero; the M-step maximizes the
tight bound for fixed q. A tight lower bound gives monotone improvement, but the
surrogate need not be globally convex and EM does not guarantee the global optimum.
Variational inference instead restricts q or approximates its optimization; its bound
may remain non-tight. These distinctions answer the theoretical MM/EM questions
without treating a local optimum as global certainty.

## Phase 2: the correct posterior

Let z∼N(μ,C), y=Az+b+ε and ε∼N(0,R), independently. The joint Gaussian has mean
`[μ; Aμ+b]` and block covariance `[[C, CAᵀ], [AC, ACAᵀ+R]]`. Define

\[
S=ACA^T+R,\quad G=CA^TS^{-1},\quad
\mu_{z|y}=\mu+G(y-A\mu-b),\quad C_{z|y}=C-GAC.
\]

This equals the precision-form expression in the handout when C is invertible.
The implementation solves Cholesky systems in observation space instead of computing
inverse matrices; it also supports rectangular operators and correlated noise.

A Gaussian-mixture prior gives a **Gaussian-mixture posterior**, with component weights

\[
\alpha_k(y)=\frac{\pi_k\mathcal N(y;A\mu_k+b,AC_kA^T+R)}
 {\sum_j\pi_j\mathcal N(y;A\mu_j+b,AC_jA^T+R)}.
\]

Each component has the Gaussian posterior above. The squared-error optimal estimate
is the posterior mean `Σₖ αₖ μₖ|y`. The highest-weight component mean is a useful hard
heuristic, but **neither it nor the mean is the general mixture MAP**. Modes can lie
between component means, and component density peaks depend on covariance volume.
The handout's approximate MAP is evaluated as a heuristic alongside MMSE.

Total posterior covariance includes both the average within-component covariance
and the between-component mean covariance. Tests compare these moments with independent
Gaussian formulas and observation-density calculations. Under a correct probabilistic
model MMSE minimizes expected squared error; an individual held-out image or a misspecified
archive can still favor the hard estimator.

## The supplied archive audit

| Archive filename | Actual m from W | Matched image pairs |
|---|---:|---:|
| `MNIST-m=8.zip` | 4 | 278 |
| `MNISTm=16 (1).zip` | 12 | 296 |
| `MNIST-m=20 (1).zip` | 16 | 296 |
| `MNIST-m=28 (1).zip` | 20 | 296 |
| `MNIST-m=4.zip` | 28 | 295 |

No supplied operator has dimension 64×64: the m=8 case is evaluated under controlled
noise, not fabricated from an archive name. Unmatched IDs are excluded and listed in
[the data manifest](../assets/project_data_manifest.json). The held-out image identities
are shared across archives where possible, but the archives have different corruption
processes and missing IDs; their scores do not establish which patch size is best for
one common inverse task.

The controlled study uses the **same whole-image Gaussian observation** for every m
in `{4,8,12,16,20,28}`. Fit the prior from patches sampled across 1,000 independent
training images, tune 4/12 components on 40 independent validation images, and evaluate
150 official test images at σ=30. Retain at most 32 PCA directions, fit a full-covariance
mixture in that space, and add isotropic residual variance in ambient space. For large
patches this is a reduced-rank regularized prior, rather than full-dimensional EM.
Convergence flags are published. It avoids pretending that limited training data can
reliably estimate dozens of 784×784 unconstrained covariances.

For each supplied archive, match original/corrupted IDs, reserve 60 pairs for validation,
use the first 30 to estimate a scalar forward gain and offset, and use the remaining
30 to select K and σ from `{5,10,20,30,50}`. Both the nominal and calibrated Gaussian
models use the same selection/test identities. The calibrated model replaces W with
`gain × W` and adds the learned offset; it requires paired clean calibration data and
is not blind noise estimation. Evaluate all remaining pairs once. Exact
duplicates of course originals are excluded from MNIST training before learning priors.
Overlapping reconstructed patches are averaged with the actual coverage count at every
pixel. Clipped PNG observations and their empirical residual mean/variance are reported:
the nominal independent zero-mean Gaussian patch model is only an approximation to
these files. Their results are kept separate from the controlled benchmark.

This is one-pass patch reconstruction, not the iterative whole-image EPLL optimizer.
[Zoran and Weiss](https://people.csail.mit.edu/danielzoran/EPLLICCVCameraReady.pdf)
explain why patch averaging and optimizing a whole-image patch objective differ.

## Reproduce and investigate

```bash
python scripts/prepare_projects.py --data-root .local/data --download
python -m learning_lab.cli course_mixtures --data-root .local/data
python -m learning_lab.cli inverse --data-root .local/data
```

The public numeric coordinate examples are included; the private ZIPs are optional.
Place your own course archives in `legacy/originals/projects` or pass `--originals`.
Without them the controlled inverse benchmark still runs, with an empty supplied-archive
section. On the author's host preparation reads existing shared MNIST/COCO files and
writes only `/data/navid.bagheri-shouraki/ML_data/projects`.

Open [08_course_mixtures.ipynb](../notebooks/08_course_mixtures.ipynb) and
[09_mixture_inverse.ipynb](../notebooks/09_mixture_inverse.ipynb) for executed numerical
examples. Inspect [project results](project_results.md) for measured outcomes and plots.

**Worked exercise.** If every feature of a categorical row is missing, what is its
posterior component distribution? The product over observed features is empty and
equals one, so the posterior equals the learned mixture weights. An imputation is
then the mixture-averaged marginal distribution, not an arbitrarily selected component.
