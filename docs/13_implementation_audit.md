# Implementation and evidence audit

This audit concerns the portfolio upgrade requested on 2026-10-07. The original
source mapping remains in [the archive audit](08_source_audit.md). Mathematical
foundations are retained where they remain useful; modern methods extend the
experiments with measurable limitations and explicit resource differences.

| Area | Correctness contract | Evidence and extension |
|---|---|---|
| Gaussian regression | Prior/noise variances positive; precision solved rather than inverted; predictive variance separates noise and coefficient uncertainty | Independent ridge identity and covariance contraction tests; executable derivation |
| Full-covariance EM | Log-domain normalization, positive covariance floor and explicit component collapse | Responsibility normalization, density integration and likelihood checks; held-out density and synthetic identifiability |
| Categorical mixtures | Missing features contribute no counts; a fully missing row recovers the component prior | Analytic posterior and missing-data tests; eight-feature controlled example instead of unidentifiable single-variable claims |
| Mixture inverse | Observation evidence includes operator, offset and noise; total covariance includes between-component terms | Independent Gaussian conditioning and marginal-density comparisons; singular-prior and extreme-observation tests |
| Patch restoration | Overlap normalization includes every border pixel; matrix dimensions define patch size | Round trips across seven patch sizes; supplied archive pairs and bias measured; forward calibration isolated from selection/test |
| Generalization | Fixed signal/noise, train-only fit, declared exact population-risk calculation | Interpolation and implicit-bias experiments retain their scoped scientific purpose |
| Kernels | Fit scaling on fitting data; choose hyperparameters separately from evaluation; repeat label budgets | RBF and Nyström comparisons measure approximation and budget effects rather than claiming modern visual competitiveness |
| Adult classification | Separate fit, selection, probability calibration and conformal calibration; official test untouched | Native categorical CatBoost added to linear, histogram boosting and residual controls; saved-model predictions checked independently |
| Housing regression | Random and geographic protocols kept separate; quantile crossing repaired consistently; empty CQR intervals retained | CatBoost point and quantile models added; random improvement and geographic coverage failure both reported |
| CIFAR representation | Train subsets and calibration partitions disjoint; checkpoint selected by validation NLL | Three-seed ConvNeXt fine-tuning with mixup and label smoothing; official test and corruption probe; external pretraining stated |
| Nonlinear receivers | Symbol energy and noise units explicit; Gray coding and circular block memory tested | Linear/MLP/context-CNN and known-channel inverse controls; independent block evaluation and block bootstrap; physical-link claims excluded |
| Digit denoising | Unclipped Gaussian observation, output clipping declared, residual initialization equals observation | PCA, autoencoder and conditioned CNN retained; full official test and noise-shift probes |
| Natural-image denoising | All methods receive identical noise realizations; sigma25 checkpoint remains fixed at sigma15/50 | Wavelet and SwinIR references executed on 400 held-out 256px crops from original photographs; strong sigma50 failure retained |
| Color conversion | D65 units and normalization explicit; gradients finite at black and outside the RGB gamut | Fixed inactive fractional-power derivatives at zero; numerical round-trip and gradient tests |
| Conditional GAN | Discriminator update detaches generator output; generator update freezes discriminator | Direct and adversarial generators share architecture; hinge loss, spectral normalization and validation selection declared |
| Conditional VAE | Conditional prior at inference; posterior target allowed only in training; analytic KL checked independently | Two validation-only pilots preserved privately; refined frozen semantic mean with a latent residual; test sample dispersion exposes collapse or weak diversity |
| Conditional diffusion | Correct cosine cumulative alpha, velocity identity and target support | Seeded DDIM test; wider SR residual clamp fixed; multiple full-test draws without best-of-target selection |
| Super-resolution | Actual antialiased bicubic LR generated once; SwinIR receives that tensor directly | Local L1/GAN/CVAE/diffusion compared with bicubic and frozen transformer; RGB fidelity shave and full-image LPIPS specified |
| External adapters | Source revision and clean tracked content checked; released checkpoint hashes checked; strict state loading | DDColor, CT2, BigColor, SwinIR and Real-ESRGAN references stay outside the repository; package carries expected pins; adapters and differing licenses stated |
| Checkpoint deployment | Architecture rebuilt from saved metadata; initial pretrained encoder unnecessary | Saved restoration, native boosting and ConvNeXt inference commands; schema and numerical checks separate from training |
| Scientific artifacts | Figures generated from numerical evidence; fixed attributed gallery identities | Full-precision public score arrays; uint8 gallery arrays used only for display, never scoring; report regenerates without raw data or models |

## Validation refinements and metric versioning

The 256px benchmark fixes the main limitation of the earlier 96px demonstration:
detail is taken from original source photographs, not enlarged from small cached
crops. Its 16,000 training identities extend the original 8,000, while held-out
identities remain unchanged. Four publicly displayed photographs were fixed and
attributed before the new model fitting.

[The data integrity audit](../assets/source_resolution_audit.json) checks every
prepared-array and original-photograph SHA256 against the public manifest, and reads
resolution from the actual source image headers. It identifies the small subset of
originals below the target resolution rather than claiming that every 256px crop
contains native 256px detail. It changes neither selected identities nor benchmark pixels.

Two early CVAE configurations had weak **prior-based validation** reconstruction.
They are preserved under `.local` as pilot evidence and excluded from reference
claims. The refined colorization CVAE freezes a validation-selected direct predictor
and learns a conditional latent residual. Its 20 latent-training epochs follow the
mean predictor's 60 epochs. This additional initialization and shared fitted mean
are explicit, so its two latent seeds are not described as two independent complete
training runs. Test results did not choose this refinement.

[The pilot decision record](../assets/validation_decisions.json) identifies completed
epochs, prior-based validation values and saved-source/history hashes. Pilot
architecture details are recovered from checkpoint parameter keys, rather than
assuming that a feature-loss weight path implies a semantic decoder. Final CVAE
checkpoints are separately checked for exact equality of every frozen semantic
parameter and BN buffer with the selected direct predictor.

The new metric protocol measures three-channel DeltaE76 and chroma from the final
RGB prediction, after clipping, rather than raw pre-gamut chroma fields. The older
96px experiment's historical metric remains a pre-gamut chroma discrepancy and is
not numerically substituted for the new metric. Completed checkpoints scored under
the earlier extension protocol are rescored without changing learned parameters;
the separate `evaluation_audit.json` preserves scoring hash, timestamp and checkpoint
hashes alongside the original training record and curves.

Stochastic RGB L1 dispersion compares subsequent draws with the first draw on every
test image. The first draw's zero difference is excluded from aggregate dispersion.
This statistic detects nearly deterministic output, but cannot establish semantic
plausibility. All four draws remain visible and contribute to fidelity/perceptual
scores. Pixel-error variation across draws is not itself a measure of diversity.

[The VAE precision audit](../assets/vae_precision_audit.json) independently evaluates
Gaussian KL in float64 on the same bfloat16 encoder outputs for all 128 selection
images and four selected checkpoints. CUDA autocast evaluates the exponential terms
in float32, as specified in [PyTorch's operator reference](https://docs.pytorch.org/docs/2.7/amp.html#cuda-ops-that-can-autocast-to-float32).
The largest observed discrepancy is 3.32e-6 nats, contributing at most 3.32e-8 to
the weighted objective. A preliminary 32-image check understated this maximum;
the full audit retains the measured errors and declares a 1e-5 absolute tolerance.
It checks these trained states rather than arbitrary parameters or gradients,
and does not establish that their latents are informative.

[The stress-evaluation audit](../assets/stress_evaluation_audit.json) re-runs all
400 images at every denoising noise level and SR degradation in separate directories.
Every saved score and gallery value reproduces exactly on the checked L40S device.
Both pipelines now reject positive/negative infinity and NaN before clipping;
six injected-output checks verify that invalid predictions cannot publish score
files. The original measured evidence and run records remain intact.

## Reproducibility limits

Run records fingerprint the source used when fitting started. Later metric fixes
have separate scoring provenance; recorded fits are not silently attributed to a
new source hash. CUDA inference uses bfloat16 where specified. Saved ConvNeXt GPU
probabilities reproduce the reference to numerical precision; CPU float32 inference
can change calibrated probabilities and should be checked separately before relying
on an operating threshold. Hardware and dependency versions are recorded.

The release validator checks experiment completion, canonical configurations,
finite values, all seeds/draws, score/summary agreement, selected epochs, split
disjointness, English notebook outputs, attribution, links and public-file bounds.
Packaging and clean-checkout execution are separate checks. Hosted CI is only
claimed after an actual hosted run; local checks do not establish one.

The earlier tabular network used a small improvement tolerance for early stopping.
All nine recorded tabular histories were checked: its selected epochs agree exactly
with the strict validation minima in these reference runs. Future fitting now saves
every strict improvement, making the selection contract explicit for other datasets.
JSON records and newly written checkpoints use atomic replacement so readers do not
observe partially written artifacts. Evaluation-only restoration rebuilds its
architecture from the saved config and needs no original initialization checkpoint.

[The recovery audit](../assets/training_resume_audit.json) interrupts a two-epoch
CPU integration after saving the first epoch, then resumes through the full experiment
entry point. Selected weights and losses agree bitwise with uninterrupted fitting;
the completed record retains the earlier attempt and selected epoch. This establishes
the tested CPU recovery path, rather than claiming cross-hardware determinism.
[The evaluation audit](../assets/evaluation_preservation_audit.json) separately checks
exact reproduction of every score array while preserving the original fitting record
and selected histories. A conflicting resume configuration is rejected before writing
the run record.
[A separate SR integration](../assets/sr_evaluation_preservation_audit.json) checks
the same contract. Both audits inject an infinite output field and verify rejection
before gamut clipping, with the original metrics, score arrays and fitting provenance
preserved. This matters because clipping an infinite SR residual could otherwise
produce a finite, invalid image.

[The saved-prediction audit](../assets/saved_prediction_audit.json) checks native
boosting and ConvNeXt inference independently of their training loops. Housing's
[calibration export audit](../results/reference/modern_tabular/deployment_calibration_audit.json)
also reconstructs every held-out interval from the original calibration partition.
Exported calibration metadata identifies its model by SHA256, so an unrelated point
or quantile checkpoint cannot silently inherit its uncertainty estimates.
Native calibration inference now requires those hashes rather than treating them
as optional. Tests explicitly reject missing point or quantile identities as well
as mismatched identities; valid exported metadata keeps the measured predictions.

[The ConvNeXt calibration export audit](../results/reference/modern_vision/cifar_calibration_audit.json)
verifies all three checkpoints on 10,000 official test images in CUDA bfloat16 and
CPU float32 profiles. CUDA reproduces the original test probabilities, coverage and
set sizes; CPU recalibrates on the original reserved partitions. Six fresh commands
match their metadata-bound prediction sets. Empty sets remain explicit, and singleton
accuracy carries no conditional coverage guarantee.

Standalone CT2 inference on GPU2 exposed upstream helper tensors that remained on
default CUDA instead of following the model. The adapter constructs the same lookup
tables on CPU, registers them as nonpersistent buffers, and supplies the relocated
buffers to the unchanged helper computations. The released model's strict parameter
loading and source checkout remain intact. DDColor inference metadata and BigColor's
class-selection checkpoint are also pinned by content hash; BigColor EMA loading
requires every parameter's count and shape to match.

[The CT2 device audit](../assets/ct2_device_audit.json) records successful CPU inference
without CUDA access and successful GPU2 inference on the same non-square observation.
[The colorization command audit](../assets/colorization_inference_audit.json) repeats
all seven local/external pipelines in fresh processes with identical seeds and verifies
PNG bytes, dimensions and input hashes. These integration checks complement the
full held-out quality evaluation; they do not establish cross-device bitwise identity.

[The foundational command audit](../assets/foundation_inference_audit.json) checks
all 16 earlier restoration checkpoints through fresh CPU image commands. It covers
exact non-square dimensions, grayscale/RGB encoding, repeated PNG bytes and
checkpoint/input hashes. Those commands now honor EXIF orientation, validate
denoising sigma, and reject nonfinite raw fields before clipping or color conversion.
The audit also injects an infinite field and checks three invalid noise settings.
No trained parameters or recorded benchmark predictions were changed.

[The receiver deployment audit](../assets/receiver_inference_audit.json) reproduces
all 60 saved neural-receiver CUDA benchmark conditions: both architectures, three
fitting seeds, two nonlinearities and five SNRs. Each uses its original 2,000 blocks
and 512,000 bits. Independent integer symbol/Gray-bit error counts reproduce every
BER/SER exactly on GPU2. Twelve fresh-process commands reproduce each device's API
outputs on eight-block subsets. Neither targets nor channel settings enter inference.

Reference CUDA convolution allows TF32 while matrix TF32 is disabled; this is now
explicit in receiver deployment, with a `--disable-tf32` alternative. On the complete
CPU comparison, 63 of 7.68 million evaluated symbol predictions differ, with at most
nine differences in one condition. Maximum probability difference is 0.004995 and
maximum BER difference is 7.8125e-6. These results retain the numerical differences
and do not estimate a cross-platform accuracy guarantee. Backend flags are restored
after API inference. A fixed seed alone does not imply identical device arithmetic.

[The independent metric audit](../assets/metric_reference_audit.json) uses the first
16 fixed test photographs with identity and seeded perturbations, one/three channels,
and zero/four-pixel shaving. Float64 SciPy convolution agrees with the declared
PyTorch SSIM to within 8e-6, and NumPy PSNR agrees within 2.2e-6 dB. D65 Lab agrees
with scikit-image within 0.0049 coordinate units; image-mean DeltaE76 differs by less
than 0.00018. Rounded color matrices and float32 arithmetic explain the allowed
tolerances. The 7x7 SSIM convention is explicitly different from scikit-image's
default Gaussian window, rather than silently asserting equality to another metric.

Professional readiness requires all requested reference experiments and validation
to complete. The [work log](upgrade_worklog.md) tracks that acceptance boundary.
