# Model and dataset cards

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
