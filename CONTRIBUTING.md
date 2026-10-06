# Contributing

Prefer a focused question with a declared evaluation protocol over adding an untested
model name. New studies should state data exposure, selection/calibration/test boundaries,
seeds, budget, metrics and the scope of any uncertainty claim.

Install `.[deep,notebooks,dev]` in your own environment, then run:

```bash
ruff check src tests scripts
ruff format --check src tests scripts
python -m pytest -q
python scripts/validate_release.py
```

Add a meaningful mathematical or statistical check when changing a method's contract.
Do not require network downloads or GPU training in unit tests. Save trial experiments
to a new directory; keep quick integration results separate from the reference report.
When a new completed reference run is deliberately selected, regenerate plots and
notebooks and explain any protocol change alongside the updated results.

Keep explanations, comments and notebook outputs in English. Credit methods and dataset
sources. Do not commit raw datasets, private credentials, local paths in executable defaults
without a portable alternative, serialized Python models, or third-party paper PDFs.
The repository's author owns final publication decisions.
