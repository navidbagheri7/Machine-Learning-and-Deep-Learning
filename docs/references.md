# References and dataset sources

These primary sources motivate the implementations. The laboratory provides its own
code and explanations and does not redistribute textbooks or research-paper PDFs.

## Added project methods and images

- Zoran and Weiss (2011), *From Learning Models of Natural Image Patches to Whole
  Image Restoration*. [Author-hosted paper](https://people.csail.mit.edu/danielzoran/EPLLICCVCameraReady.pdf).
  Motivates the patch-prior connection; this laboratory's one-pass averaging is not the EPLL optimizer.
- Zhang et al. (2017), *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for
  Image Denoising*. [Paper](https://arxiv.org/abs/1608.03981). Our compact conditioned CNN is a different architecture.
- Lim et al. (2017), *Enhanced Deep Residual Networks for Single Image Super-Resolution*.
  [Paper](https://arxiv.org/abs/1707.02921). Motivates residual scaling, no normalization and learned upsampling.
- Zhang, Isola and Efros (2016), *Colorful Image Colorization*.
  [Authors' project](https://richzhang.github.io/colorization/). Motivates uncertainty and rebalanced chroma classification.
- Kang et al. (2023), *DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders*.
  [Paper](https://arxiv.org/abs/2212.11613), [official source](https://github.com/piddnad/DDColor),
  [official pretrained release](https://huggingface.co/piddnad/ddcolor_modelscope).
  Frozen source/model revisions and Apache-2.0 provenance are pinned in the public manifest.
- COCO 2017 photographs: [official download page](https://cocodataset.org/#download),
  [project](https://cocodataset.org/). Only CC BY 2.0 images are selected for the local
  restoration benchmark. Public galleries preserve [verified creator credits](photo_attributions.md).
- Supplied course numeric examples: two 600-point synthetic coordinate arrays, and
  five privately preserved corrupted-digit archives. Names/hashes are inventoried;
  handout and archive contents are excluded from publication.

## Conditional generation and modern reference methods

- Sohn, Lee and Yan (2015), *Learning Structured Output Representation using Deep
  Conditional Generative Models*. [Paper](https://proceedings.neurips.cc/paper_files/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf).
  Motivates the learned conditional prior; the local perceptual objective differs from an exact ELBO.
- Isola et al. (2017), *Image-to-Image Translation with Conditional Adversarial Networks*.
  [Authors' project](https://phillipi.github.io/pix2pix/). The local hinge/spectral-normalization recipe is a declared adaptation.
- Song, Meng and Ermon (2021), *Denoising Diffusion Implicit Models*.
  [Paper](https://arxiv.org/abs/2010.02502). Distinct initial noise drives multiple deterministic DDIM trajectories.
- Weng et al. (2022), *CT2: Colorization Transformer via Color Tokens*.
  [Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670001.pdf),
  [official implementation and noncommercial terms](https://github.com/shuchenweng/CT2).
- Kim et al. (2022), *BigColor: Colorization using a Generative Color Prior for Natural Images*.
  [Authors' project](https://kimgeonung.github.io/bigcolor/),
  [paper](https://arxiv.org/abs/2207.09685). Released encoder/generator and EMA are frozen; class-selection adaptation is declared.
- Liang et al. (2021), *SwinIR: Image Restoration Using Swin Transformer*.
  [Paper](https://arxiv.org/abs/2108.10257), [official implementation](https://github.com/JingyunLiang/SwinIR).
  Classical 4x SR and sigma25 color denoising references retain their published training exposure.
- Wang et al. (2021), *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data*.
  [Official source and model release](https://github.com/xinntao/Real-ESRGAN),
  [architecture source](https://github.com/XPixelGroup/BasicSR).
  Its blind-degradation training differs from this controlled bicubic test.
- Zhang et al. (2018), *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*.
  [Authors' project](https://richzhang.github.io/PerceptualSimilarity/).
  LPIPS is a learned proxy; no human preference study is claimed here.
- Blau and Michaeli (2018), *The Perception-Distortion Tradeoff*.
  [Paper](https://arxiv.org/abs/1711.06077). Its distributional notion of perception
  motivates separating appearance and distortion; our LPIPS comparisons do not
  independently verify that theorem or measure its distributional distance.
- He et al. (2019), *Lagging Inference Networks and Posterior Collapse in Variational Autoencoders*.
  [Paper](https://arxiv.org/abs/1901.05534). Context for latent-use diagnostics;
  its aggressive inference-training method is not implemented in this laboratory.
- Prokhorenkova et al. (2018), *CatBoost: unbiased boosting with categorical features*.
  [Paper](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html).
  Native boosting is evaluated through the maintained library, not claimed as a new implementation of its internals.
- Liu et al. (2022), *A ConvNet for the 2020s*.
  [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html).
  ConvNeXt-Tiny uses official ImageNet weights and a declared limited-label fine-tuning recipe.

## Textbook foundations

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
