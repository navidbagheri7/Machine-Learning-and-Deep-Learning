# Local release validation — 2026-10-06

The repository is prepared locally for the author to publish. No GitHub repository
was created and no content was pushed, following the author's preference.

- Thirteen full reference studies completed with matching configs and finite metrics.
- Thirteen fresh-kernel English notebooks executed successfully, containing 65 code cells.
- Twenty-two mathematical, statistical, neural-interface and inference tests passed.
- Ruff lint and formatting checks passed.
- Twenty-three scientific figure sets regenerated as PNG/PDF from measured evidence.
- Five new project pipelines passed offline procedural-fixture integration checks;
  these fixtures are clearly distinguished from actual reference datasets.
- Denoising, SR, local colorization and frozen DDColor image-inference commands exercised
  on actual project images; the earlier Adult inference check remains available.
- Supplied project files inventoried; archive dimensions, missing pairs and corruption
  bias audited. Calibration and model selection use disjoint held-out identities.
- Prepared official DDColor source/model revisions checked; weights stored only in
  `/data/navid.bagheri-shouraki/ML_data/models`, with public SHA-256 provenance.
- Gallery photographs carry verified creator attribution and CC BY 2.0.
- Source archive and wheel built, installed command tested, and clean local checkout
  validated without course originals, raw data or model checkpoints.
- English text, outputs, local links, paired-score arrays, config consistency and
  publishable contents checked by [the release validator](../scripts/validate_release.py);
  its machine-readable record is [validation.json](validation.json).

## Publish from this directory

Suggested repository name: `learning-generalization-lab`. Suggested description:
“Probabilistic ML and deep learning with uncertainty, distribution shift, latent-variable
inference, image restoration and reproducible pretrained-model evaluation.”

Create an empty repository in your account and set its URL as the remote:

```bash
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
git push origin v1.1.0
```

Code and original documentation use MIT licensing. Data and external models retain
their source terms, listed in [references](references.md) and [photo credits](photo_attributions.md).
The included CI workflow will run after publication; local validation does not claim
a hosted GitHub Actions run has already occurred.
