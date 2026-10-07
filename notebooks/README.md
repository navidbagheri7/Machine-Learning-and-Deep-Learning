# Executed walkthroughs

Run these from the repository root with the package installed. Each notebook discovers
the root from its working directory, then uses public package APIs and saved evidence.
They execute short numerical demonstrations and do not launch the full training suite.

| Notebook | Main task |
|---|---|
| [01 Gaussian posterior](01_gaussian_posterior.ipynb) | Verify posterior/ridge identity and predictive uncertainty |
| [02 Tabular decisions](02_tabular_decisions.ipynb) | Trace split isolation, calibration and expected decision cost |
| [03 Intervals and shift](03_intervals_and_shift.ipynb) | Derive finite-sample quantiles and inspect geographic coverage |
| [04 Latent structure](04_latent_structure.ipynb) | Fit EM, inspect responsibilities and compare density versus clusters |
| [05 Generalization](05_generalization.ipynb) | Verify exact risk, inspect interpolation and margin dynamics |
| [06 Representation learning](06_representation_learning.ipynb) | Check contrastive positive pairs and compare information budgets |
| [07 Learned receivers](07_learned_receivers.ipynb) | Verify Gray coding, channel inversion and measured BER |
| [08 Course mixtures](08_course_mixtures.ipynb) | Compare overlapping latent groups, density selection and categorical missing-data EM |
| [09 Mixture inverse](09_mixture_inverse.ipynb) | Verify posterior moments, patch overlap and observation-model calibration |
| [10 Denoising](10_denoising.ipynb) | Compare PCA, autoencoder and noise-conditioned residual learning |
| [11 Super-resolution](11_super_resolution.ipynb) | Inspect pixel shuffle, paired-image evidence and degradation mismatch |
| [12 Colorization](12_colorization.ipynb) | Compare chroma fidelity, uncertainty and L2 desaturation |
| [13 Pretrained reference](13_pretrained_reference.ipynb) | Audit frozen DDColor provenance, external exposure and paired deployment metrics |
| [14 Generative colorization](14_generative_colorization.ipynb) | Compare adversarial, conditional latent and diffusion outputs without best-of-test sampling |
| [15 Generative SR](15_generative_super_resolution.ipynb) | Verify target support, velocity prediction and fidelity/perception tradeoffs |
| [16 External restoration](16_external_restoration.ipynb) | Audit released sources, adapters, licensing and training exposure |
| [17 Native boosting](17_native_boosting.ipynb) | Compare categorical learning, calibrated decisions and geographic interval failure |
| [18 Modern transfer](18_modern_transfer.ipynb) | Inspect limited-label ConvNeXt selection, calibration and corruption |
| [19 Natural denoising](19_natural_denoising.ipynb) | Compare wavelets and a frozen transformer under noise mismatch |

Regenerate all outputs with `python scripts/execute_notebooks.py`. Fresh-kernel execution is required for every public notebook; output errors fail the release checks.
