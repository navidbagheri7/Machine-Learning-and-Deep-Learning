# Model and dataset cards

## Modern restoration, transfer and native boosting extension

**Purpose/data.** Compare practical restoration and prediction pipelines with explicit
observations, external exposure and failure cases. The new image benchmark uses
16,000 original-photograph 256px CC BY COCO training crops and the same disjoint
400/400 held-out identities. Every public gallery retains creator attribution.
Adult, California Housing and CIFAR keep the earlier evaluation protocols.

**Local restoration.** Direct and conditional-GAN predictors, conditional VAE and
cosine velocity diffusion share observations and held-out images. Most fit two seeds
for 60 epochs on 128px training crops. The refined colorization VAE fits two latent
seeds for 20 full-image epochs after freezing a 60-epoch validation-selected semantic
mean; the shared initialization is additional computation and limits the interpretation
of its seed variation. Four prior/DDIM samples per test image are included without
best-of-target selection. Validation and test use the conditional prior, not the
target-conditioned VAE posterior.

**External restoration.** DDColor, CT2, BigColor, SwinIR and Real-ESRGAN are frozen,
officially released models, not newly developed algorithms. Source pins, content
hashes, licenses and adapter differences are recorded. CT2 is noncommercial and is
not included in the repository's MIT code grant. BigColor's fixed ResNet50 selector
differs from its default classifier. Real-ESRGAN's blind-degradation training differs
from this controlled bicubic SR test. Possible external training overlap is unaudited.

**Natural-image denoising.** Sigma15/25/50 Gaussian observations are generated once
per test image and shared across methods. BayesShrink knows sigma; frozen SwinIR was
trained at sigma25. The sigma50 degradation is an observed limitation, not a tuned
model variant or a discarded outlier. These measurements do not cover real camera noise.

**Modern prediction.** Native categorical CatBoost adds validation-selected boosting
and quantile uncertainty. Better random-split accuracy does not eliminate geographic
shift or restore conformal guarantees. Three ConvNeXt-Tiny fine-tuning seeds use
5,000 CIFAR fitting labels each, ImageNet initialization and 128px interpolation of 32px input.
Each seed additionally uses 2,000 validation and 3,000 calibration labels, for 10,000
local labels in disjoint roles. Stratified benchmark sampling uses the available
class annotations; this is not a total annotation-acquisition claim.
The substantial pretraining resource differs from scratch and self-supervised controls.
Noise-corrupted accuracy and per-class errors remain part of the evidence.
Device-specific ConvNeXt calibration exports also reproduce all 10,000 test prediction
sets for each seed and precision profile. CPU calibration uses only the reserved
partitions. Empty class sets remain explicit; singleton accuracy is descriptive.

**Deployment uncertainty.** Native-boosting metadata exports conformal class sets,
absolute-residual intervals and CQR corrections using the original calibration
partitions. Every saved housing test interval is reproduced before export. Metadata
hashes bind calibration to fitted models; empty intervals remain explicit. Post-fit
subgroup coverage is descriptive and does not establish conditional coverage or fairness.

**Artifacts and limits.** Full-precision score/prediction arrays and exact split IDs
are public. Gallery source arrays are uint8 display copies and never used for scoring.
Large checkpoints, original sources and raw data remain outside the release. The
inference commands rebuild locally fitted models without downloading initialization
weights. Image fidelity, LPIPS, observation consistency and RGB sample dispersion
answer different questions; dispersion alone does not establish plausible diversity.
Read [methods](12_generative_restoration.md), [implementation audit](13_implementation_audit.md)
and [measured comparisons](modern_results.md).

## Added mixture and inverse studies

**Purpose/data.** Demonstrate latent-variable density estimation, missing-data inference
and probabilistic restoration. Two supplied synthetic coordinate datasets are not MRI
or clinical records. The categorical example is a controlled eight-feature synthetic
model with 25% missing-at-random entries. Image priors use independent MNIST training
images after excluding exact matches to supplied course originals.

**Evaluation.** Separate fitting/selection/test partitions; full posterior evidence
and covariance checks; six patch sizes under a common controlled AWGN observation.
Supplied archive scores use filename-matched pairs, matrix-derived patch sizes,
30 calibration/30 selection pairs and the remaining pairs for testing. Forward
gain/offset calibration uses paired clean data and is not a blind estimator.

**Limits/artifacts.** At most 32 latent PCA directions regularize large image priors.
Convergence flags are public. One-pass patch averaging is not a whole-image EPLL
optimizer; Gaussian posterior uncertainty is not empirically calibrated confidence.
Large priors and raw archives remain local; numerical examples, exact IDs, metrics
and fixed galleries are public. See [derivations](10_course_projects.md).

## Neural image restoration and frozen colorization

**Purpose/data.** Compare PCA, bottlenecks, residual learning, local transfer and
pretrained deployment. MNIST uses 20,000 training/2,000 validation images and the
official 10,000-image test. Natural-image methods share 8,000 CC BY COCO training
crops and disjoint 400/400 validation/test crops. Public gallery photos preserve
verified creator credits and licence; the MIT code licence does not replace them.

**Training/evaluation.** Three seeds/25 epochs for neural digit denoising; two seeds/
60 epochs for local SR and colorization. Validation loss selects checkpoints. Report
unit-range PSNR/SSIM, noise/degradation shifts, Lab DeltaE76, chroma and palette entropy,
with complete curves and per-image paired evidence. Image-bootstrap intervals condition
on a checkpoint; seed SD estimates training variation over the declared seeds only.

**External exposure.** Local transferred colorizers start from ImageNet ResNet18.
Frozen DDColor-ModelScope is an official externally trained 227.9-million-parameter
model: it is not trained or proposed here, and its COCO training overlap was not
audited. Source/model revisions, weight hash, licence and synchronized throughput
are recorded. Its 512px internal input upsamples the same observed lightness.

**Limits/artifacts.** 96px controlled SR is not a full-resolution DIV2K leaderboard.
AWGN is not camera noise. Exact color is underdetermined; PSNR, chroma and perceived
plausibility can disagree. No human preference, clinical or forensic evaluation was
run. Saved inference scripts load local checkpoints or the explicitly prepared
frozen model; raw data, large weights and third-party source remain in the personal cache.
Read [protocols](11_image_restoration.md) and [measured outcomes](project_results.md).

## Adult probability and decision study

**Purpose.** Demonstrate mixed-data preprocessing, comparative modeling, probability
calibration, subgroup analysis and cost-aware decisions. The target is a historical
census-derived income label; current salaries or hiring outcomes are not represented.

**Data.** Original UCI training/test files, with missing categorical values retained
for imputation. Survey weights are omitted from predictors and metrics. The code
uses race and sex features to expose and discuss subgroup behavior, not to recommend
their use in an employment decision.

**Evaluation.** Three declared training partitions, official test, AUROC/AP/NLL/Brier,
calibrated prediction sets, subgroup counts and operating-point TPR/FPR. Paired NLL
intervals are conditional on seed-zero fitted models. No causal or fairness certification
is inferred from these descriptive comparisons.

**Artifacts.** Local fitted logistic and boosting pipelines, calibration parameters,
feature schema and threshold. Public release includes metrics and split indices,
with a command to regenerate the models. Input validation checks are covered separately.

## California Housing regression

**Purpose.** Demonstrate point prediction, posterior versus conformal uncertainty,
adaptive intervals and geographic mismatch. This historical, capped-target dataset
does not provide a contemporary real-estate valuation product.

**Evaluation.** Separate random and eastern-geography holdouts. Target units are
$100,000. RMSE/MAE, observed interval coverage, width and empty-interval rate are reported.
Exchangeability is deliberately violated for the geographic stress test; no shifted
coverage guarantee is claimed.

**Limitations.** Longitude is only one possible holdout design. Target censoring,
spatial dependence and geographic population differences complicate broader conclusions.
The feature transformer is a compact numerical-token variant, not a complete tabular
foundation model or exact reproduction of a published benchmark.

## CIFAR representation study

**Purpose.** Compare limited supervision, contrastive fitting and external transfer
with explicit data exposure. This is a small-image educational benchmark, not a
production image recognition system.

**Evaluation.** Official 10,000-image test split; three low-label subsets; disjoint
selection, probability-calibration and conformal-calibration partitions. External
ResNet18 weights are identified by their source and SHA-256. Corruption means the
specific Gaussian pixel-noise protocol stated in the chapter.

**Limitations.** Pretrained ImageNet and compact CIFAR encoders differ in architecture
and data. One contrastive training seed and one full-label supervised seed do not
estimate training variability for those encoders. Contrastive set coverage falls below
the nominal target in the measured run; the reported cutoff is unchanged by test labels.
Dataset-pretraining overlap is not audited beyond known split handling.

## Nonlinear receiver study

**Purpose.** Evaluate local nonlinear receivers with a simulator whose symbol energy,
noise variance and boundary conditions are explicit and testable.

**Evaluation.** Independent fitting/selection/test blocks, BER/SER across SNR, repeated
seeds, block-bootstrap intervals, and unadapted nonlinearity shift. Known-channel inversion
uses knowledge not supplied to learned receivers and remains labeled separately.

**Limitations.** Circular baseband blocks omit physical stream overhead, synchronization,
phase recovery and fiber propagation. A low BER on this simulator does not establish
real-link performance. Zero observed errors would require an evaluation-count-dependent
bound rather than a literal zero-risk claim.

## Analytical and exploratory studies

Gaussian posterior and EM examples use declared synthetic distributions. The double-descent
study measures exact Gaussian population risk under its fixed ten-feature signal and
noise. Wine clustering is descriptive and fits scaling on the entire dataset. Digits
learning curves are a small offline benchmark. Their conclusions are scoped to these
designs and do not substitute for evaluation on a deployment population.
