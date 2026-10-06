# Local release validation — 2026-10-06

The repository is prepared for the author to publish. No GitHub repository was created
and no content was pushed, following the author's explicit preference.

- Seven full reference studies completed, with matching configs and strict finite metrics.
- Seven fresh-kernel notebooks executed successfully; 35 code cells in total.
- Thirteen mathematical, statistical, neural-interface and inference tests passed.
- Ruff lint and formatting checks passed.
- Ten scientific figure sets regenerated as PNG and PDF from measured results.
- Tabular dataset preparation and CIFAR/weight checksum verification completed.
- Editable installation and the installed experiment command verified in a separate local environment.
- Adult batch inference exercised on actual official-test feature rows.
- Source archive and wheel built successfully with the declared build backend.
- Source inventory retained locally; original coursework, paper PDFs, datasets and checkpoints excluded from Git.
- Public English text, executed outputs, local links and publishable file contents checked by
  [the release validator](../scripts/validate_release.py); its machine-readable record is
  [validation.json](validation.json).

## Publish from this directory

Use `learning-generalization-lab` as a suggested GitHub repository name. Suggested description:
“Probabilistic ML and deep learning experiments with calibration, distribution shift,
generalization, representation learning and nonlinear signal equalization.”

Create an empty repository in your GitHub account, then set its URL as the remote:

```bash
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
git push origin v1.0.0
```

The code and documentation use MIT licensing. Datasets and external pretrained weights
remain under their source terms, listed in [references](references.md). The repository
includes a CI workflow; its hosted status becomes available after you publish. Local
validation does not claim a GitHub Actions run has already occurred.
