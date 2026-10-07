# Learning path

The laboratory connects a method's assumptions to an implementation and an evaluation.
You can read it as a curriculum or use each study independently. Basic linear algebra,
derivatives, probability, Python and array operations are assumed. No GPU is required
for the first four chapters.

1. **Inference and density.** Work through [Gaussian posterior regression and EM](01_probabilistic_models.md).
   Derive the posterior precision, inspect the Cholesky-based mixture density, and run
   notebook 01. Explain why confident extrapolation can still be wrong.
2. **Prediction and decisions.** Read [the tabular study](02_tabular_decisions.md).
   Trace every partition of Adult and compare predictive loss with decision utility.
   Run notebook 02 and inspect subgroup sample counts.
3. **Uncertainty and shift.** Read [the conformal chapter](03_uncertainty_and_shift.md).
   Compute the finite-sample rank yourself, then inspect random and geographic holdouts
   in notebook 03. Explain which guarantee survives each change.
4. **Generalization and kernels.** Read [the generalization study](04_generalization.md).
   Separate approximation, variance and optimization bias. Use notebook 04 for EM
   and notebook 05 for interpolation and margin dynamics.
5. **Representation learning.** Read [the image study](05_representation_learning.md).
   Reconstruct the information available to each encoder. Notebook 06 checks the
   contrastive objective and examines full reference results.
6. **An end-to-end domain example.** Read [signal equalization](06_signal_equalization.md).
   Use notebook 07 to verify the simulator, Gray coding and inverse receiver before
   interpreting a neural BER curve.
7. **Repair an inverse model.** Read [the course-project reconstruction](10_course_projects.md).
   Use notebooks 08–09 to separate density modeling from group-label recovery and to
   condition a mixture correctly. Inspect the validation-only correction of the supplied
   corruption model before interpreting the restored digits.
8. **Restore and color natural images.** Read [image restoration](11_image_restoration.md).
   Use notebooks 10–13 to compare PCA, bottlenecks, residual networks, transfer learning,
   color distributions and a modern frozen model. Explain why PSNR and plausible color
   can disagree, and why external model exposure changes the comparison.

For an applied data scientist interview, begin with the Adult study: justify the
split, preprocessing, model selection metric, calibration and operating threshold.
Then explain why the housing experiment needs a geographic evaluation. For an ML
research interview, discuss the exact population-risk calculation, implicit bias,
the contrastive ablations and the limitations of a fixed-budget comparison.

## Reading a result

Every experiment has a JSON config, a `run.json` completion record, `metrics.json`,
and any needed histories or exact split indices. Plots do not substitute for metrics.
Seed standard deviations describe variation across the selected training partitions
or label subsets; example-wise bootstrap intervals answer a different question.

The report labels synthetic and real datasets, external pretraining, numerical checks,
and distribution shift. A reference run completing successfully establishes that its
configured protocol ran; it does not establish universal superiority of a method.

## A useful sequence of questions

What population is the target? Which information enters fitting? Which information
enters model selection? What loss corresponds to the intended action? What makes a
confidence claim defensible? Which failure cases change the conclusion? Answer these
before adding another model.

See [reproduction](07_reproduction.md) to rerun studies and [model cards](09_model_cards.md)
to assess their scope.

The [modern restoration chapter](12_generative_restoration.md) extends the course
autoencoders into conditional GAN/VAE/diffusion and pretrained-transformer comparisons.
Then inspect native boosting, ConvNeXt transfer and noise mismatch in the
[measured modern report](modern_results.md). The [implementation audit](13_implementation_audit.md)
connects each mathematical/evaluation contract to its code and verification.
