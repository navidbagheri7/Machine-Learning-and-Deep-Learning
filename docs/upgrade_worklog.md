# Portfolio upgrade work log

Requested on 2026-10-07. Publication remains local; the owner will publish.

## Acceptance criteria

- Audit original briefs and all current numerical, training, evaluation and inference code.
- Add conditional VAE, conditional GAN and conditional diffusion comparisons for
  colorization and 4x super-resolution, with reproducible training and held-out evaluation.
- Evaluate official DDColor, CT2 and BigColor references where obtainable, and a
  modern pretrained super-resolution reference. Record external exposure and provenance.
- Use genuine higher-resolution images, fixed disjoint splits, image-level uncertainty,
  perceptual and fidelity metrics, multiple stochastic samples, fixed galleries and failures.
- Review every existing study for correctness and weak evidence; update affected results.
- Execute notebooks, inference, meaningful tests, packaging and clean-checkout checks.
- After the ML/DL acceptance criteria are met, rebuild RL_project around its original
  slides and examples with evaluated planning, tabular and deep RL comparisons.
- After RL is completed and validated, rebuild the Convex optimization project around
  its original theory, examples and assignments with executed solver comparisons.

## Current state

The ML/DL upgrade is complete locally as version 1.2.0. All 32 reference protocols,
19 fresh-kernel notebooks (115 code cells), 38 tests, saved-model inference,
independent numerical audits and clean source-archive verification passed. The
[release checklist](release_checklist.md) records the final acceptance evidence.
The author publishes GitHub themselves. RL and optimization are separate sequential
follow-up repositories and are not claimed as completed by this ML release.

## Earlier progress records

- Existing v1.1.0 is preserved. Its compact 96px restoration experiments are baselines,
  not evidence that the requested visual quality has been achieved.
- Three L40S GPUs available through simple_udm; no unrelated jobs at initial inspection.
- Shared image sources read-only; downloads and derived caches belong under
  /data/navid.bagheri-shouraki/ML_data.
- RL_project has original Python code and slides and no Git repository yet. Detailed RL
  implementation starts after the ML/DL upgrade is validated.
- The 16,000/400/400 original-photograph 256px cache is prepared and hashed. All five
  external restoration references have completed full-test evaluation with current
  final-RGB metrics and stochastic dispersion.
- Native categorical/quantile boosting and three-seed ConvNeXt fitting completed.
  Natural-image denoising completed at sigma15/25/50, retaining noise-mismatch failures.
- All colorization models, SR direct, SR CVAE and SR diffusion completed and use the current scoring
  definitions. The SR GAN comparison remains in progress. Two validation-only
  CVAE pilots were superseded by a declared frozen-semantic-mean refinement; no test-based selection.
- Thirty-eight source tests and thirty-eight installed-wheel tests passed; the latter
  ran from a separate temporary directory without a source import path. Final wheel verification remains required. Native
  inference also has interval, empty-CQR
  and calibration/checkpoint compatibility checks. Saved Adult and housing predictions
  were checked for all three seeds; every Housing test interval reproduced on export.
  ConvNeXt GPU probabilities agree with reference values on 64 test images per seed;
  CPU float32 probabilities differ under the declared mixed-precision evaluation.
- Reports, six additional walkthroughs and expanded release checks are implemented.
  Eighteen notebooks passed fresh-kernel execution, including subgroup, permutation,
  confusion and reliability diagnostics. Epoch-resumption independently verified
  bitwise-identical final weights and losses on a declared CPU integration budget.
  All seven colorization commands reproduce seeded PNGs on non-square observations;
  CT2 also passes CPU and nondefault-GPU checks. Every source-photo and prepared-array
  checksum matches the public manifest. Wheel/source-archive preflight checks passed;
  final packaging, the remaining SR notebook, complete reference evidence and
  clean-checkout checks remain required. The upgraded ML/DL repository is not yet complete.
- All 16 foundational restoration checkpoints pass repeated fresh-process CPU image
  inference on actual non-square observations. Invalid noise and injected infinite
  raw outputs are rejected. Independent SciPy/NumPy/scikit-image metric calculations
  agree within the declared numerical tolerances. Final colorization commands pass
  with current command/module hashes; SR commands await all selected checkpoints.
- Saved MLP/CNN receivers have a schema-checked decoding API/command. All 60 original
  neural CUDA conditions reproduce BER/SER exactly; twelve CPU/CUDA commands reproduce
  their respective API outputs. The complete CPU comparison retains 63 changed class
  decisions across 7.68 million predictions and explicit convolution TF32 settings.
  Two additional CPU schema, Gray-bit and invalid-logit tests passed.

- All six ConvNeXt precision profiles export checkpoint-bound temperature and
  conformal metadata. Full official-test coverage and empty-set rates reproduce;
  six fresh commands match the corresponding deployment API. CPU calibration uses
  only the original reserved partitions, with no test-based parameter selection.
- The four selected CVAE checkpoints passed the full selection-partition arithmetic
  audit against independent float64 Gaussian KL on the same encoder outputs. The
  measured maximum is 3.32e-6 nats and its weighted-objective contribution is at most
  3.32e-8; the audit records the initial subset's underestimate and its scope.
- Nine offline CPU study commands passed from a separate publishable checkout using
  the installed wheel. Natural denoising and SR stress evaluation now reject invalid
  raw outputs before clipping; full independent reproduction of both 400-image,
  three-condition studies gives exact score/gallery agreement. Six injected-output
  checks pass while preserving the original numerical evidence.
- Native uncertainty inference rejects missing checkpoint identities. The targeted
  tests pass and every original Housing test interval reproduces with the required
  point/quantile hashes. All six full ConvNeXt precision-profile verifications pass
  with the current source binding after this interface correction.
- The ConvNeXt label budget now explicitly separates 5,000 fitting labels from
  2,000 validation and 3,000 calibration labels, and declares label-aware stratified
  benchmark sampling. Headline, figure and notebook-source wording now use this declared full budget.

- Corrected categorical imputation to preserve observed categories; checked missing-field probabilities against hand-derived Bayes posteriors. The majority baseline uses observed fitting entries only; the complete reference study was rerun, retaining the same missing-entry scores.

- The current installed wheel passed all nine offline CPU integrations from a new separate checkout, with procedural fixtures and no source import path, original data or pre-existing checkpoints. All nineteen generated notebook sources passed Markdown indentation, duplicate-heading and control-character checks; the Gaussian prior explanation now preserves its LaTeX escape correctly. Full fresh-kernel execution remains queued after the final GAN fit.
