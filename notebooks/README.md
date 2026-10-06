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

Regenerate all outputs with `python scripts/execute_notebooks.py`. Every public notebook
has been run from a fresh kernel; output errors fail the release checks.
