# Reproduction and release engineering

Commands run from the repository root. The completed reference records identify
Python and library versions. Reference training used `simple_udm` and L40S GPUs.
Each config declares budgets, seeds, device and paths; none reads a hidden credential.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[deep,notebooks,dev]'
python -m pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
```

The scientific package can be installed without the `deep` extra to use Gaussian
inference, EM, kernels, generalization and uncertainty utilities. For the exact
reference software set use [requirements-tested.txt](../requirements-tested.txt)
in a separate environment after installing a suitable CUDA or CPU PyTorch build.
Do not install a CUDA wheel incompatible with your driver. The reference versions
are evidence of a tested environment, not a claim that all possible dependency
combinations have been validated.

## Dataset preparation

On the author's machine, existing CIFAR resides at `/data/tii/Image_data/cifar10/raw`
and tabular data at `/data/navid.bagheri-shouraki/ML_data`.

The tabular cache was moved after the reference runs. Their recorded execution paths
are preserved; [data_locations.json](../assets/data_locations.json) records the relocation.

The public commands can use local caches:

```bash
python scripts/prepare_data.py --data-root .local/data
python scripts/prepare_vision.py --data-root .local/cifar10 --weights .local/models/resnet18-f37072fd.pth
```

Adult preparation retrieves the official UCI archive. California Housing preparation
uses scikit-learn's verified download and cache. CIFAR preparation verifies the
official archive checksum and extracts only expected members. Model preparation
verifies the ResNet18 weight hash prefix used by torchvision. Training loaders
require existing files and do not trigger background downloads.

For portable image runs, copy `configs/vision.json` and edit `image_root`,
`pretrained_weights` and `device` to match these paths. Keep the edited config with
your experiment; it is copied into `run.json`. Set a new output directory when
changing budgets so reference results are preserved.

### Added course-project cache and modern model

```bash
python scripts/prepare_projects.py --data-root /data/navid.bagheri-shouraki/ML_data
python scripts/prepare_ddcolor.py --data-root /data/navid.bagheri-shouraki/ML_data
python scripts/run_suite.py --studies course_mixtures inverse denoising super_resolution colorization ddcolor
```

Project preparation reuses `/data/tii/Image_data` read-only and writes resized natural
images, MNIST, paired course arrays and an audit under the personal cache's `projects/`.
For a public clone on another machine, `prepare_projects.py --data-root YOUR_CACHE --download`
fetches official MNIST, COCO metadata and selected CC BY photographs explicitly.
Private course archives are optional; controlled inverse evaluation still runs without them.
The preparation tools do not publish the private handouts or student notebooks.

DDColor preparation downloads pinned third-party source and approximately 870 MB weights
under `YOUR_CACHE/models`; it emits an edited config there for portable inference.
Use `--config YOUR_CACHE/models/ddcolor_config.json --device YOUR_DEVICE`. Local
colorization training separately uses the checked torchvision ResNet18 weights declared
in its config. The two models have different external exposure and should not be conflated.

Offline CI uses explicitly labelled procedural fixtures from `prepare_smoke_data.py` for
the five trainable/new numerical pipelines. Those fixtures are never reference datasets.
The frozen external model requires prepared weights and is evaluated in the local full
reference run; its notebook, provenance and compact evidence remain usable offline.

The CLI imports numerical libraries before applying threadpool limits, and the suite
sets BLAS/OpenMP limits for subprocesses. This avoids oversubscribing shared CPU resources.
Run logs are local; configs, complete curves, selected checkpoints' cards, measured metrics,
exact IDs and source hashes are public. Large checkpoints remain local.

## Integration versus reference runs

### Modern extension on another machine

```bash
python -m pip install -e '.[deep,restoration,pretrained,boosting,notebooks,dev]'
python scripts/prepare_data.py --data-root /YOUR/CACHE
python scripts/prepare_vision.py --data-root /YOUR/CACHE/cifar10 --weights /YOUR/CACHE/models/torch/hub/checkpoints/resnet18-f37072fd.pth
python scripts/prepare_restoration.py --data-root /YOUR/CACHE --download
python scripts/prepare_external_restoration.py --data-root /YOUR/CACHE
python scripts/prepare_ddcolor.py --data-root /YOUR/CACHE
python scripts/make_portable_configs.py --data-root /YOUR/CACHE
python scripts/run_generative_suite.py --config-dir .local/portable_configs --output-root results/portable --data-root /YOUR/CACHE
python -m learning_lab.cli modern_tabular --config .local/portable_configs/modern_tabular.json
python scripts/export_uncertainty.py --data-root /YOUR/CACHE --results-root results/portable/modern_tabular
python -m learning_lab.cli modern_vision --config .local/portable_configs/modern_vision.json
python scripts/export_cifar_calibration.py --image-root /YOUR/CACHE/cifar10 --results-root results/portable/modern_vision --device cuda:2
python -m learning_lab.cli external_ddcolor --config .local/portable_configs/external_ddcolor.json
python -m learning_lab.cli external_ct2 --config .local/portable_configs/external_ct2.json
python -m learning_lab.cli external_bigcolor --config .local/portable_configs/external_bigcolor.json
python -m learning_lab.cli external_swinir --config .local/portable_configs/external_swinir.json
python -m learning_lab.cli external_realesrgan --config .local/portable_configs/external_realesrgan.json
python -m learning_lab.cli natural_denoising --config .local/portable_configs/natural_denoising.json
python -m learning_lab.cli sr_shift --config .local/portable_configs/sr_shift.json
```

Config generation preserves budgets and writes explicit cache/device/output paths;
it does not modify the published configs or download data. Use `--devices cpu cpu cpu`
for a CPU override, or declare other three-GPU assignments. Native boosting's paired
historical comparison is available when the chosen baseline directory contains its
earlier predictions. The three-GPU restoration suite resumes per-model epochs,
waits for the CVAE's semantic mean and refuses mismatched resume configurations.
The CT2 archive extractor needs a system libarchive library in addition to libarchive-c.

To regenerate the published figures from compact evidence, also run
`python scripts/build_project_report.py` and `python scripts/build_upgrade_report.py`.
These reporting commands work without original coursework, raw datasets, weights or
GPU access. Unit/interface checks and notebook evidence are offline; the new neural
reference workloads require explicitly prepared pretrained metric/feature weights.

```bash
python -m learning_lab.cli classification --quick --data-root .local/data --device cpu
python -m learning_lab.cli regression --quick --data-root .local/data --device cpu
python -m learning_lab.cli probabilistic --quick
python -m learning_lab.cli generalization --quick
python -m learning_lab.cli kernels --quick
python -m learning_lab.cli signals --quick --device cpu
python -m learning_lab.cli vision --quick --config configs/vision_portable.json --device cpu
```

Create `vision_portable.json` from `vision.json` with the paths described above.
`--quick` uses much smaller budgets and saves to `results/smoke/<study>`, keeping
those checks separate from reference evidence. CPU overrides work, but a complete
vision reference run is much faster on a GPU.

For the author's data layout, `python scripts/run_suite.py` concurrently runs the
CPU studies, a sequential tabular/signal group on GPU 1, and the vision study on GPU 2.
It caps numerical libraries at four CPU threads per process. On another machine
edit device values first, or run each study sequentially:

```bash
python -m learning_lab.cli classification --data-root .local/data --device cuda:0
python -m learning_lab.cli regression --data-root .local/data --device cuda:0
python -m learning_lab.cli probabilistic
python -m learning_lab.cli generalization
python -m learning_lab.cli kernels
python -m learning_lab.cli signals --device cuda:0
python -m learning_lab.cli vision --config configs/vision_portable.json --device cuda:0
```

NumPy and PyTorch seeds are fixed. CUDA floating-point kernels, library versions and
hardware can still change low-order numerical values; reference results should be
compared with meaningful tolerances rather than exact byte equality of checkpoints.

## Artifact contracts

Every run records its complete config, package versions, start time, completion status
and wall time. `metrics.json` uses strict JSON serialization: NaN metrics fail visibly.
Missing subgroup diagnostics have explicit null values where appropriate. Dataset
targets and pretrained weights have hashes; tabular and image partitions include exact
indices. The data preparation manifest records original tabular-file hashes.

Large prediction arrays, model checkpoints and joblib pipelines are local artifacts
and excluded from Git. Compact numerical figure inputs are exported to
`assets/source_data`; they permit regeneration from a fresh clone:

```bash
python scripts/build_report.py
python scripts/build_project_report.py
python scripts/build_upgrade_report.py
python scripts/execute_notebooks.py
python scripts/validate_release.py
python -m build
```

The executed notebooks use the bundled evidence and small numerical examples. They
do not launch the complete GPU training budget. Their outputs are checked for errors
and all published notebook sources and output text are English.

## Batch prediction

ConvNeXt exports separate CUDA bfloat16 and CPU float32 calibration profiles from
the original independent calibration partitions. The command reports probabilities,
point labels, class membership flags, set sizes and explicit empty sets:

```bash
python scripts/export_cifar_calibration.py --image-root /YOUR/CACHE/cifar10 --device cuda:0
python scripts/predict_modern.py cifar \
  --checkpoint results/reference/modern_vision/checkpoints/convnext_0.pt \
  --metadata results/reference/modern_vision/cifar_inference_cpu_float32_0.json \
  --input cifar_images_uint8_nchw.npy --output .local/cifar_with_sets.csv --device cpu
```

The metadata must match the inference device and precision profile. Omitting metadata
retains the probabilities-only interface with an optional manual temperature.

The modern native-boosting artifacts also expose calibrated prediction sets and
regression intervals. `export_uncertainty.py` reconstructs their cutoffs from the
fixed calibration partitions and verifies saved test evidence without refitting.
Deployment loads only the model files and exported scalar/schema metadata:

```bash
python scripts/predict_modern.py adult \
  --checkpoint results/reference/modern_tabular/checkpoints/adult_0.cbm \
  --metadata results/reference/modern_tabular/adult_inference_0.json \
  --input adult_features.csv --output .local/adult_with_sets.csv
python scripts/predict_modern.py housing \
  --checkpoint results/reference/modern_tabular/checkpoints/housing_random_0.cbm \
  --quantile-checkpoint results/reference/modern_tabular/checkpoints/housing_quantile_random_0.cbm \
  --metadata results/reference/modern_tabular/housing_inference_random_0.json \
  --input housing_features.csv --output .local/housing_with_intervals.csv
```

House-value predictions and interval endpoints use the benchmark's $100,000 units.
Empty CQR intervals remain explicit. The geographic holdout demonstrates that the
nominal level does not establish coverage under distribution shift. Calibration
metadata contains checkpoint hashes, so accidentally mixing fitted models is rejected.

Independent deployment and epoch-resumption checks are reproducible after preparing
the data and fitting the corresponding models:

```bash
python scripts/check_saved_predictions.py --data-root /YOUR/CACHE --image-root /YOUR/CACHE/cifar10
python scripts/check_training_resume.py --data-root /YOUR/CACHE
python scripts/check_restoration_data.py --data-root /YOUR/CACHE
python scripts/check_image_metrics.py --data-root /YOUR/CACHE
python scripts/check_vae_precision.py --data-root /YOUR/CACHE --results-root results/portable --device cuda:0
python scripts/check_stress_evaluation.py --data-root /YOUR/CACHE --results-root results/portable --device cuda:0
python scripts/check_foundation_inference.py --data-root /YOUR/CACHE --results-root results/portable
python scripts/check_receiver_inference.py --results-root results/portable/signals --device cuda:0 --compare-cpu
python scripts/verify_initialization.py --run results/portable/generative_colorization/cvae --mean-checkpoint results/portable/generative_colorization/l1/checkpoints/l1_0.pt
python scripts/check_restoration_inference.py --data-root /YOUR/CACHE --results-root results/portable --device cuda:0
```

The prediction check covers all three fitting seeds on fixed test subsets. CUDA
bfloat16 probabilities and CPU float32 results are reported separately. The resume
check uses a fresh temporary directory, two CPU epochs and 16 real images; it proves
resume behavior rather than restoration quality. The CVAE initialization check
verifies all frozen semantic tensors against its validation-selected mean.

The image-metric audit compares PSNR and the declared valid 7x7 Gaussian SSIM
against independent float64 SciPy calculations, and D65 Lab/DeltaE76 against
scikit-image. The foundational inference check covers all 16 saved digit-denoising,
96px-trained SR and colorization checkpoints on actual non-square observations.
It also checks rejection of invalid noise settings and injected infinite outputs.
Image CLI fixtures use quantized pixels; these deployment checks do not replace
the full-precision benchmark observations or quality scores.
The standalone restoration audit repeats local and external commands in fresh
processes on fixed non-square observations, checking seeded PNG bytes and exact
output dimensions. Each external adapter is isolated because upstream package names
can collide. CT2's fixed lookup buffers now follow CPU and nondefault GPU placement.

After running the Adult reference study, the saved pipeline includes fitting-only
preprocessing, the model, a probability temperature and a validation-selected threshold.
The input CSV needs the 12 feature columns listed in `learning_lab.data`.

```bash
python scripts/predict.py \
  --artifact results/reference/classification/hist_boosting.joblib \
  --input .local/adult_examples.csv \
  --output .local/adult_predictions.csv
```

Missing features and nonnumeric numeric-column values produce explicit errors.
Unseen categories follow the fitted encoder's unknown-category handling. These
historical income predictions are an educational artifact, not a supported production
service. Only load serialized Python artifacts from a trusted source.

## CI and publishing

The GitHub Actions workflow checks style, mathematical tests, wheel building, executed
notebooks and an offline integration study. Expensive training stays a declared local
reference workload. CI does not pretend to repeat the full GPU benchmark.

The repository is prepared locally for its author to publish. Original coursework and
third-party PDFs are ignored, and local checkpoints, data and logs are not staged.
Before uploading, run the release validator and inspect `git status` and `git diff --cached`.
The [release checklist](release_checklist.md) records the completed validation.
