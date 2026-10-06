# Model and dataset cards

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
