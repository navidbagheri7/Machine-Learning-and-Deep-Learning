# References and dataset sources

These primary sources motivate the implementations. The laboratory provides its own
code and explanations and does not redistribute textbooks or research-paper PDFs.

## Foundations

- Kevin P. Murphy. *Probabilistic Machine Learning: An Introduction*. MIT Press, 2022.
  [Author's book page](https://probml.github.io/book1).
- Kevin P. Murphy. *Probabilistic Machine Learning: Advanced Topics*. MIT Press, 2023.
  [Author's book page](https://probml.github.io/book2).
- Ian Goodfellow, Yoshua Bengio and Aaron Courville. *Deep Learning*. MIT Press, 2016.
  [Author-hosted book](https://www.deeplearningbook.org/).

## Implemented extensions and motivating research

- Gorishniy et al. *Revisiting Deep Learning Models for Tabular Data*, NeurIPS 2021.
  [Paper](https://arxiv.org/abs/2106.11959). Residual tabular models and feature-token attention;
  this repository uses declared compact variants.
- Chen et al. *A Simple Framework for Contrastive Learning of Visual Representations*, ICML 2020.
  [Paper](https://arxiv.org/abs/2002.05709). Contrastive objective and projection-head motivation;
  the CIFAR recipe differs from their full ImageNet experiment.
- He et al. *Deep Residual Learning for Image Recognition*, CVPR 2016.
  [Paper](https://arxiv.org/abs/1512.03385). Residual image architectures.
- Guo et al. *On Calibration of Modern Neural Networks*, ICML 2017.
  [Paper](https://proceedings.mlr.press/v70/guo17a.html). Temperature calibration.
- Romano, Patterson and Candès. *Conformalized Quantile Regression*, NeurIPS 2019.
  [Paper](https://arxiv.org/abs/1905.03222). Adaptive regression intervals.
- Angelopoulos and Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free
  Uncertainty Quantification*. [Paper](https://arxiv.org/abs/2107.07511).
- Belkin et al. *Reconciling modern machine-learning practice and the classical bias–variance
  trade-off*, PNAS 2019. [Paper](https://arxiv.org/abs/1812.11118).
- Hastie et al. *Surprises in High-Dimensional Ridgeless Least Squares Interpolation*.
  [Paper](https://arxiv.org/abs/1903.08560). Context for the exact-risk interpolation study.
- Soudry et al. *The Implicit Bias of Gradient Descent on Separable Data*, JMLR 2018.
  [Paper](https://arxiv.org/abs/1710.10345). Logistic direction convergence.

## Data and pretrained weights

- Becker and Kohavi. **Adult**, UCI Machine Learning Repository.
  [Dataset and terms](https://archive.ics.uci.edu/dataset/2/adult).
  The original archive supplies fixed training and test files; file hashes are recorded.
- Pace and Barry. **California Housing**, 1997 census block-group regression data.
  [Loader documentation and original-source attribution](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html).
- Krizhevsky. **CIFAR-10**, 2009. [Official dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
  Full image datasets are not included in the Git release; a small deterministic error
  gallery illustrates model behavior.
- **Wine** and **Optical Recognition of Handwritten Digits**, bundled in scikit-learn.
  [Wine loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html),
  [Digits loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
- **Torchvision ResNet18 ImageNet-1K V1**.
  [Official model card and preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
  The weight SHA-256 is included in the image run record.

For further reading, evaluate later tabular foundation models and large-scale visual
pretraining against explicit data exposure and deployment constraints. This repository's
measured scope is the implementations listed above; reading references do not imply
those additional methods have been implemented or evaluated here.
