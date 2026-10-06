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

## Integration versus reference runs

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
python scripts/execute_notebooks.py
python scripts/validate_release.py
python -m build
```

The executed notebooks use the bundled evidence and small numerical examples. They
do not launch the complete GPU training budget. Their outputs are checked for errors
and all published notebook sources and output text are English.

## Batch prediction

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
