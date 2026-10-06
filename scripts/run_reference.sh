#!/usr/bin/env bash
set -euo pipefail
# Run from repository root after installing .[deep,notebooks,dev].
python scripts/prepare_data.py --data-root /data/tii/ML_data
for study in probabilistic generalization kernels classification regression signals vision; do
    python -m learning_lab.cli "$study"
done
python scripts/build_report.py
python scripts/validate_release.py
