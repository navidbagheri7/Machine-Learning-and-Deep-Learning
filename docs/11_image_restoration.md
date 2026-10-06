# Restoration, resolution, and ambiguous color

The added deep-learning templates ask for MNIST denoising with autoencoders/PCA and
natural-image colorization/super-resolution. The reconstruction keeps each requested
baseline, then tests whether residual architectures, transfer learning and probabilistic
color modeling improve it under an explicit budget. All reference runs train real models;
digit galleries use the first held-out IDs. Natural-image galleries use the first four
test-order photographs with verified creator attribution, independent of appearance
or model performance. The full evaluation still uses every declared test image.

## Denoising: represent the corruption explicitly

Images use intensities in 0–1. Training observations are `x + σ ε`, with independent
standard-normal ε and σ drawn uniformly from 10.2–51.0 in 0–255 units. Inputs are not
clipped; output intensities are clipped only for evaluation. This is the conventional
unclipped AWGN experiment, and differs from camera noise or clipped course PNGs.

The comparison includes a convolutional bottleneck autoencoder, a noise-conditioned
eight-layer residual CNN, ordinary PCA projection, and Gaussian-prior PCA shrinkage.
The residual network predicts noise and subtracts it from the observation; a constant
noise map supplies its known SD. This compact model borrows the residual-denoising idea
from [DnCNN](https://arxiv.org/abs/1608.03981), but has a different depth, no batch
normalization, and an explicit noise input. It is not a paper-reproduction claim.

For PCA directions V and variances λ, projection reconstructs retained coefficients
unchanged. Gaussian shrinkage multiplies each coefficient by `λ/(λ+σ²)`. Directions,
variances and mean are fit on 20,000 training images; ranks 16/32/64/128 are selected by
MSE on 2,000 disjoint validation images. The best rank can differ between the two methods.
Shrinkage is optimal under a Gaussian coefficient prior, not under an arbitrary digit
distribution. A measured ordinary projection win is evidence of that prior mismatch.

Train each neural model for 25 epochs across three seeds. Select checkpoints by
validation MSE at σ=30. Evaluate the official 10,000 test digits at σ=10/30/50/75,
including a σ=75 stress test beyond the training range. A Poisson-plus-Gaussian probe
tests a different noise mechanism. The noise map contains only the Gaussian component
in that probe; it does not reveal or estimate the signal-dependent Poisson variance.

## A shared natural-image benchmark

The templates name two unavailable hosted landscape/Unsplash datasets. Existing COCO
photographs supply an explicit substitute with stable IDs, disjoint splits and metadata.
This is not a reproduction of those unnamed hosted dataset versions.

Select only COCO photographs carrying **CC BY 2.0**. A fixed RNG chooses 8,000 training
images from `train2017` and 800 images from official `val2017`; the first 400 are
validation and the remaining 400 test. Center crop and resize each to 96×96 with Pillow
bicubic sampling; the cached uint8 pixels are shared between methods. Training may
horizontally flip images. Public image figures credit every source and the modification
in [photo attributions](photo_attributions.md). This licence filter changes the population;
findings describe this subset, not all COCO images.

The reference experiments use two independent training seeds and 60 epochs per model.
AdamW, cosine learning-rate decay, gradient clipping and CUDA mixed precision are
implemented in one trainer. Validation loss selects checkpoints, while test images are
used only for final evaluation. Autoencoder and transferred models receive the same
local image budget; ImageNet pretraining is additional external exposure, disclosed in
checkpoint cards rather than treated as free data.

## 4× super-resolution: preserve low-resolution evidence

Antialiased bicubic downsampling maps 96×96 RGB targets to 24×24 inputs. The controls
are bicubic upsampling and a bottleneck autoencoder operating on bicubic-upsampled
inputs. The residual model operates at low resolution, uses eight scaled residual
blocks at width 64, and applies two 2× pixel-shuffle stages with a bicubic global skip.
There is no batch normalization. These design choices are inspired by
[EDSR](https://arxiv.org/abs/1707.02921); the network and budget are much smaller.

Training/checkpoint selection minimize RGB L1 error. Evaluate RGB PSNR and SSIM after
shaving four pixels from each border. SSIM uses a valid 7×7 Gaussian window, SD 1.5,
unit-range constants and a mean over channels. This differs from published Y-channel,
full-resolution DIV2K/Set5 protocols; direct leaderboard comparisons would be invalid.
A second test applies a 5×5 binomial blur before downsampling, with no retraining.

PSNR measures fidelity, not human preference. No adversarial/perceptual loss or image
quality survey was run. The examples show what compact pixel-fidelity models can
recover, including their failure to restore information removed by downsampling.

## Colorization: predict plausible chroma, evaluate its uncertainty

Convert sRGB to D65 CIELAB. The input is normalized L only; networks predict a/110
and b/110. A numerical test checks known red/white/black Lab values and RGB roundtrips.
Predicted colors retain the observed lightness and are converted back to RGB with gamut
clipping. A neutral-gray baseline sets a=b=0.

Compare a bottleneck autoencoder trained with chroma L2, a ResNet18 encoder/U-Net-style
decoder with ImageNet initialization trained with the same L2, and that transferred
architecture trained to classify chroma bins. The classification palette uses 32 k-means
centers learned from training pixels. Inverse-frequency weights mix the empirical prior
with 50% uniform mass to prevent extreme rare-bin weights. Decode by a temperature-scaled
probability-weighted chroma mean; select among temperatures 0.38/0.7/1.0 on validation
DeltaE76. Palette frequencies, chosen temperatures and predictive entropy are recorded.

This is a compact adaptation of the uncertainty/class-rebalancing approach in
[Zhang, Isola and Efros](https://richzhang.github.io/colorization/), not their pretrained
colorization system. The external encoder is the official torchvision ImageNet ResNet18;
its checksum is verified. All task-specific parameters are trained here.

Report RGB PSNR/SSIM, mean DeltaE76 before gamut clipping, predicted/true chroma, and
palette entropy. Low PSNR may coexist with a plausible alternative color. L2 can produce
desaturation because it averages possible chroma. Balanced classification can increase
chroma while worsening exact-color error. Entropy measures uncertainty over this learned
palette, not a calibrated confidence probability or proof of perceptual realism.
Neither model is validated for forensic restoration or inferring historical true colors.

## Frozen DDColor reference

The [official DDColor-ModelScope release](https://github.com/piddnad/DDColor) adds a
modern dual-decoder reference from ICCV 2023, combining a ConvNeXt-L encoder with
learnable color queries and a transformer color decoder. This is a **frozen external
model**, not a model trained or proposed by this repository. Source Git revision and
Hugging Face model revision are pinned in `configs/ddcolor.json`; the evaluated weight
SHA-256, parameter count and synchronized GPU throughput are recorded with its results.
The upstream source and model are Apache-2.0 and remain in the personal model cache.

Feed the same native-resolution lightness evidence as local colorizers. Reconstruct
neutral gray from L, bilinearly upsample to the released 512px internal size, predict
chroma, and downsample chroma with the upstream nearest-neighbor convention. This
avoids extracting lightness from resized RGB, which could inadvertently let original
chroma affect the input. Retain the original L for the final RGB output and evaluate
the same PSNR/SSIM/DeltaE/chroma metrics on the same 400 images.

The authors describe this release as ImageNet-trained with a data-cleaning scheme.
Its training budget and possible overlap with COCO were not audited. It therefore
provides a practical deployment reference, not a fair equal-data training comparison.
No local fine-tuning, test-based model selection or human-preference experiment is used.

```bash
python -m pip install -e '.[deep,pretrained]'
python scripts/prepare_ddcolor.py --data-root .local/data
python -m learning_lab.cli ddcolor --config .local/data/models/ddcolor_config.json --device cuda:0
```

For the author's run, the newly downloaded source and approximately 870 MB weight file
are under `/data/navid.bagheri-shouraki/ML_data/models`. Other project image data reused
the shared cache. Neither the third-party source checkout nor weights are published
inside this repository. The public preparation script fetches the exact pinned revisions.

## Reproduction and saved inference

```bash
python scripts/prepare_projects.py --data-root .local/data --download
python scripts/prepare_vision.py --data-root .local/cifar10 --weights .local/models/resnet18-f37072fd.pth
python -m learning_lab.cli denoising --data-root .local/data --device cuda:0
python -m learning_lab.cli super_resolution --data-root .local/data --device cuda:0
```

For colorization, copy `configs/colorization.json`, point `pretrained_weights` to the
prepared file and change data/device/output paths. Pass it with `--config`. CPU quick
runs are supported for pipeline validation; they are not reference-quality training.
The preparation tool downloads official metadata, selected photos and MNIST only when
`--download` is explicit. On the author's machine no additional image/model download
was needed: existing shared files were reused read-only and caches are personal.

```bash
python scripts/restore_image.py \
  --checkpoint results/reference/colorization/checkpoints/transfer_l2_0.pt \
  --input photograph.png --output .local/colorized.png
```

The same tool loads denoising and SR checkpoints without loading external weights;
pass the known noise level with `--sigma` for a residual denoiser. SR input is already
low-resolution and output has four times its height/width. Model files stay local and
are excluded from Git. Regenerate them with the published configs.

[Project results](project_results.md) contain the measured comparisons, curves, galleries
and seed variation. Image bootstrap intervals quantify variability across held-out
images for one checkpoint; seed standard deviations quantify training variation. They
are different sources of uncertainty. The compact natural-image budgets and two seeds
limit conclusions about large-scale restoration performance.

**Worked exercise.** Why can an L2 colorizer predict gray for a bright object? If the
grayscale evidence supports opposite chroma values equally, their conditional mean
can be near zero. Gray minimizes expected squared error even when either saturated
alternative looks more natural. A diverse color distribution and a decision rule need
to be considered separately from exact-color fidelity.
