# Source audit and reconstruction

The initial directory contained ten ordinary notebooks, one notebook with a `.pdf`
extension, one PDF with an `.ipynb` extension, and research-paper PDFs. Several
notebooks depended on absent CSV/NumPy files, old course-only instructions, hosted
notebook links, or Persian explanations. They were not a runnable standalone package.

The originals are preserved without alteration under `legacy/originals`, ignored by
Git. [The inventory](../legacy/manifest.json) records original names, detected formats,
byte counts and SHA-256 values. Assignment handouts, student identifiers, embedded
course outputs and third-party paper PDFs are excluded from the public release.
The new implementation and explanations are English throughout.

| Original source | Topic retained | Completed reconstruction |
|---|---|---|
| `IML_CHW1_Q1.ipynb` | Python foundations | Replaced by reusable Python modules, tests, packaging and typed-by-contract numerical interfaces |
| `IML_CWH1_Q2.ipynb` | Arrays and exploratory tables | Dataset loaders, auditable preprocessing, group summaries and plotting pipeline |
| `Question1.ipynb` | Least squares and gradient optimization | Gaussian posterior derivation, ridge identity checks, real regression and uncertainty |
| `Question2.ipynb` | Polynomial fit and model selection | Posterior basis regression, explicit validation partitions and double-descent controls |
| `Question3.ipynb` | Logistic, naïve Bayes and LDA ideas | Probabilistic classification, calibrated linear and nonlinear baselines, decisions and diagnostics |
| `Support_Vector_Machines_1.ipynb` | Income prediction with an SVM | Official Adult splits, mixed-feature preprocessing, boosting/linear/residual comparison, subgroup and utility analysis |
| `Support_Vector_Machines_2.ipynb` | Kernel image classification | Offline RBF and Nyström learning curves with repeated budgets and isolated selection data |
| `Clustering.ipynb` | K-means and clustering | Full-covariance EM from scratch, held-out density, BIC, synthetic geometry and bootstrap stability |
| `HW_NN_Q1.ipynb` | Neural-network building blocks | Residual tabular network, tested forward/gradient contracts and reproducible training |
| `HW_NN_Q2.ipynb` | CIFAR CNN | Residual image encoder, low/full-label supervision, contrastive representations and external transfer |
| `Report.pdf` (actually notebook) | Neural receivers and nonlinear channels | Gray-coded block simulator, exact inverse checks, linear/MLP/CNN controls, BER/SER and channel mismatch |
| `Code.ipynb` (actually PDF) and paper collection | Generalization and project context | Original English derivations and primary-source citations; no redistribution of paper PDFs |

This mapping retains the intellectual topics rather than copying obsolete assignment
restrictions. The real building-energy CSV was not present in the source; California
Housing supplies a downloadable regression study with a meaningful geographic stress
test. The absent airline CSV is replaced by fully specified synthetic geometry and the
bundled Wine dataset. Those substitutions are declared, not presented as a reproduction
of unavailable original data.

The communication reconstruction is a discrete baseband benchmark with circular memory.
It does not claim to reproduce the original continuous propagation equations. Physical
model assumptions and knowledge supplied to each receiver are stated in the chapter.

## Added project briefs and data

The later `projects/` folder adds two project PDFs, two coordinate CSVs, five MNIST
inverse-problem archives, two student solution notebooks, and three empty deep-learning
templates. These sources are now included in the hash inventory. Their requirements
are reconstructed in [the mixture/inverse chapter](10_course_projects.md) and
[the restoration chapter](11_image_restoration.md), with [measured results](project_results.md).

| Added source | Completed work |
|---|---|
| Phase 1 PDF and solution notebook | Gaussian/categorical EM derivations, hard-versus-soft assignment distinction, missing-data latent-class EM, model selection and imputation |
| `Image1` / `Image2` numeric CSVs | Three independent fit/selection/test partitions; recovered parameters, held-out density, ARI and covariance geometry |
| Phase 2 PDF and solution notebook | Stable linear-Gaussian conditioning, observation evidence, mixture posterior moments, hard heuristic/MMSE and overlap averaging |
| Five corrupted MNIST archives | Filename pairing, matrix-derived dimensions, unmatched-ID reporting, controlled-versus-supplied separation and validation-only forward calibration |
| Denoising template | PCA projection/shrinkage, autoencoder and noise-conditioned residual CNN; full official-test evaluation and noise shift |
| Super-resolution template | Declared natural-image dataset substitution, 4× bicubic/autoencoder/pixel-shuffle comparison and blur stress test |
| Colorization template | Lab regression, transferred encoder/decoder, balanced chroma bins, uncertainty and frozen DDColor reference |

The reconstruction fixes missing mixture weights, incompatible intensity units,
incorrect posterior component scoring, unpaired image ordering, imagewise contrast
rescaling, and random single-image tuning. Archive filenames are inconsistent with
their matrices, and the supplied observations have positive residual bias; those
facts are measured and addressed explicitly rather than hidden by favorable plots.
Natural-image gallery publication uses fixed test ordering and verified CC BY creator
credits. The raw handouts, partner identifiers, archives and original outputs remain private.
