# Generative restoration: fidelity, appearance and ambiguity

The original colorization and SR briefs ask for deterministic autoencoders. The new
comparison retains those baselines and adds conditional latent-variable models,
adversarial training, diffusion and externally trained transformers. The 96px study
remains a separate experiment; its numbers cannot be substituted for the 256px study.

## Data and observations

Use 16,000 CC BY 2.0 photographs from COCO train2017, with 400 validation and 400 test
photographs from val2017. Exact image IDs, source hashes and preprocessing are recorded.
The original 8,000 training IDs are a prefix of the larger training list; validation
and test ordering is unchanged. Photographs are center-resized from their original
files to 256px, not enlarged from the old 96px cache. The fixed public gallery uses the
same four attributed test identities, selected before these experiments.

The prepared validation pool contains 400 photographs. The local generative configs
use its fixed first 128 for epoch selection, leaving the remaining 272 unused in this
comparison. All 400 test photographs are evaluated. Pool sizes and actually consumed
budgets are kept distinct in configs and resource tables.

The [source-resolution audit](../assets/source_resolution_audit.json) records 124
training, four validation and six test originals whose shorter side is below 256px;
these are enlarged by the declared resize. Most sources are larger, with median
shorter sides of 428, 454 and 435 pixels respectively. This benchmark uses original
photographs rather than enlarged 96px caches, but does not assert native 256px detail
for every source image. All identities remain in the evaluation.

Most local models train on random 128px crops with horizontal flips. The colorization
CVAE refinement uses full 256px images and a frozen, validation-selected direct
predictor after weak prior-based validation in two preserved pilots. Its 20-epoch
latent training follows the mean predictor's 60-epoch run; its two latent fitting seeds
share that fitted mean. This budget differs and is recorded in a separate complete
[config](../configs/generative_colorization_cvae.json). External references are frozen.

**Colorization:** the input is D65 CIELAB lightness L/100. Predict a/110 and b/110 and
retain the observed lightness when converting to sRGB. Neither original RGB, color
hints nor test chroma enter inference. Colors outside the RGB gamut are clipped;
the observed-lightness consistency metric detects the resulting small changes.

**4x SR:** antialiased PyTorch bicubic downsampling produces the actual LR image.
Local models condition on its bicubic enlargement and predict twice the RGB residual.
SwinIR receives the actual LR image directly. This interpolation implementation
differs from the MATLAB bicubic degradation used in some published SR benchmarks;
these are new measurements, not a reproduction of the authors' headline scores.

## Local models and objectives

The direct predictor and conditional GAN share a generator architecture. Colorization
uses an ImageNet ResNet18 encoder with skip-connected decoder; SR uses a residual
U-Net with GroupNorm and bottleneck attention. Direct training combines field L1 and
a frozen ResNet feature loss. The feature loss is distinct from evaluation LPIPS.

The conditional GAN adds a spectrally normalized PatchGAN discriminator. Its hinge
objective is E[max(0,1-D(c,y)) + max(0,1+D(c,G(c)))]/2. The generator adds -E[D(c,G(c))]
to its reconstruction/feature objective. Discriminator updates see detached generated
images; generator updates freeze the discriminator. This is a local pix2pix-style
adaptation, not an exact reproduction of the original pix2pix architecture or losses.

The CVAE learns q(z|c,y), p(z|c) and a decoder G(c,z). Both latent distributions are
diagonal Gaussians. Reparameterization uses z=mu+exp(logvar/2)*epsilon. The analytic
KL(q||p) is summed over latent dimensions and averaged over the batch, with a declared
annealed coefficient. Reconstruction L1 plus a feature term makes this a variational
inspired perceptual objective; the feature term is not a normalized observation
likelihood, so the complete objective must not be reported as an exact ELBO.

At validation and test time, z is sampled from the **conditional prior**, not the
posterior that has access to target colors. Colorization's semantic decoder provides
a learned conditional mean, with a latent-conditioned residual branch. Training KL,
prior-based validation error and sample variation should be considered together:
posterior reconstruction quality alone does not establish useful inference or diversity.

The report pairs each CVAE's posterior KL curve with prior-based validation and test
RGB draw dispersion. Training KL is a batch average summed over all 32 latent
dimensions, rather than per-dimension KL or an estimate of mutual information.
The final colorization runs have very small KL and nearly identical samples despite
strong reconstruction from their frozen semantic mean. These observations are
consistent with weak latent use; they do not establish a rigorous posterior-collapse
diagnosis. The model remains a CVAE, but these fitted checkpoints should not be
presented as evidence of meaningful color diversity.

Conditional diffusion predicts velocity v=sqrt(alpha_bar)*epsilon-
sqrt(1-alpha_bar)*y_tgt under a 1,000-step cosine noise schedule. Its denoiser observes
the corrupted target field, timestep embedding and task condition. The test sampler
uses 50 deterministic DDIM updates starting from separately seeded Gaussian noise.
The support clamp is [-1,1] for normalized chroma and [-3,3] for the wider SR residual;
using the chroma clamp for SR would incorrectly exclude valid residual targets.
These are compact task-trained diffusion baselines, not pretrained Stable Diffusion.

AdamW, cosine learning-rate decay, gradient clipping, bfloat16 CUDA autocast and
parameter EMA are explicit. Best EMA checkpoints use validation only: reconstruction
plus feature loss for direct/GAN/CVAE models, fixed-noise velocity MSE for diffusion.
This is a proxy for sample quality, not a perceptual optimum. Test results never
choose a checkpoint, temperature, sample or gallery image.

## External references and attribution

- [DDColor](https://github.com/piddnad/DDColor): official ModelScope release, ConvNeXt-L
  and dual decoders; native 256px grayscale evidence, 512px internal processing.
- [CT2](https://github.com/shuchenweng/CT2): official ViT-L checkpoint with 313 color
  tokens and lightness-dependent mask. Source is CC BY-NC 4.0 and stays outside this
  repository. The adapter never supplies actual chroma to its diagnostic target argument.
  Its fixed gamut lookup tensors are registered as buffers so CPU and nondefault-GPU
  inference use the requested device; upstream source files remain unchanged.
- [BigColor](https://github.com/KIMGEONUNG/BigColor): official encoder-generator and EMA
  parameters. This comparison uses a declared fixed ImageNet ResNet50 class selector
  rather than the authors' default EfficientNet-L2; it is an adapter comparison, not
  exact reproduction of their default pipeline. Four fixed random latents are evaluated.
- [SwinIR](https://github.com/JingyunLiang/SwinIR): official medium classical 4x SR model,
  trained on DF2K; no local fine-tuning. The natural-image denoising extension also
  evaluates its sigma25 color-denoising checkpoint against wavelet BayesShrink and the
  noisy input, with sigma15/50 as unadapted noise shifts.
  Every method receives the same unclipped noisy observation; predictions, including
  the noisy-input baseline, are clipped to [0,1] for scoring and display.
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): official x4plus EMA checkpoint,
  using the pinned BasicSR RRDBNet forward graph. Training uses synthetic blind
  degradations rather than only bicubic downsampling; this controlled test is a
  degradation mismatch. No face enhancement, tiling, test adaptation or extra scale
  interpolation is applied. Initialization and registration helpers are isolated
  from BasicSR's training dependencies; strict loading replaces every model parameter.

Published source revisions, weight SHA-256 values and licenses are in
[the external manifest](../assets/external_restoration_manifest.json). Raw sources,
archives and weights stay in the personal cache. External training exposure differs
from local COCO training; overlap with test photographs is not audited. A frozen
reference does not demonstrate that its algorithm was developed or trained here.

## Compare the actual outputs

Report RGB PSNR, channel-averaged SSIM, LPIPS-Alex, observation consistency and, for
colorization, three-channel DeltaE76 and mean chroma derived from the final displayed
RGB after gamut clipping. RGB L1 differences between the first and subsequent stochastic
draws measure sample dispersion on every test image; the identical first draw is excluded
from dispersion summaries. Near-zero dispersion suggests limited useful stochasticity,
while large differences alone do not establish plausible colors. PSNR/SSIM shave four HR pixels for
SR; LPIPS uses the full image and [-1,1] RGB normalization. Lower LPIPS is a learned
perceptual proxy, not a substitute for human judgment. No user-preference study was run.

All four stochastic samples per image are included. Aggregate samples within each
training seed before describing seed variation; four draws from one frozen model are
not four independently trained models. Paired image bootstrap intervals condition on
the fitted models. Avoid selecting the sample closest to the unavailable true colors
and then presenting it as ordinary deployment performance.

Use fixed galleries, training curves and measured tables together. Successful sky or
food colorization does not imply exact recovery of clothing, flower or paint colors.
For SR, hallucinated texture may improve a perceptual score while reducing fidelity.
The same model need not win all objectives or remain strongest under distribution shift.

Mean PSNR also needs a distribution check. The grayscale baseline reconstructs eleven
nearly achromatic test photographs above 60dB; some attain the numerical ceiling of
120dB from the common MSE floor of 1e-12. Its mean therefore exceeds its median by a
substantial margin. The report retains every image and presents median/quartile
diagnostics alongside ordinary mean scores, perceptual metrics and paired differences
against the trained direct predictor. This avoids mistaking an outlier-sensitive
mean for typical color recovery.

The `sr_shift` study evaluates frozen SwinIR and Real-ESRGAN against bicubic under
three shared conditions: matched bicubic4; HR Gaussian blur sigma1.2 with a 9px reflect
kernel before downsampling; and bicubic LR with sigma3/255 Gaussian noise followed
by clipping, uint8 conversion and JPEG quality60/subsampling2. Noise seed810 and all
400 test identities are fixed. Neither model is adapted. Its auxiliary consistency
score is **bicubic reprojection discrepancy to actual LR**, not a full forward-model
likelihood for blur or compression. These controlled probes are not a real-camera
test set and do not reproduce Real-ESRGAN's complete training degradation pipeline.

## Reproduce and use checkpoints

```bash
python -m pip install -e '.[deep,restoration,pretrained,boosting,notebooks,dev]'
python scripts/prepare_restoration.py --data-root /YOUR/CACHE --download
python scripts/prepare_external_restoration.py --data-root /YOUR/CACHE
python scripts/prepare_ddcolor.py --data-root /YOUR/CACHE
```

Copy the supplied configs and change cache, pretrained-weight, device and output
paths. Run one model with `learning-lab generative_colorization --config YOUR.json
--methods gan`; use `generative_super_resolution` for SR. The three-GPU suite records
per-model configs, training histories and epoch-level resumable checkpoints.
For the refined colorization CVAE, use `configs/generative_colorization_cvae.json`
after fitting the direct predictor; its saved semantic parameters are frozen during
latent training. The suite handles that dependency.

```bash
python scripts/restore_generative.py \
  --checkpoint results/reference/generative_colorization/cvae/checkpoints/cvae_0.pt \
  --input photograph.png --output .local/colorized.png --samples 4
```

For SR, provide an already low-resolution image. Inference reconstructs the entire
model from checkpoint metadata without downloading initial encoder weights. Multiple
outputs use declared seeds; a sidecar JSON records input/checkpoint hashes and sampling.

Frozen references also have a standalone command:

```bash
python scripts/restore_external.py ddcolor --input gray.png --output .local/ddcolor.png --device cuda:0
python scripts/restore_external.py realesrgan --input low.png --output .local/sr.png --device cuda:0
python scripts/predict_modern.py adult --checkpoint results/reference/modern_tabular/checkpoints/adult_0.cbm \
  --metadata results/reference/modern_tabular/adult_inference_0.json --input features.csv --output predictions.csv
```

Use `--data-root` or an edited config for another cache. CIFAR inference accepts a
uint8 N x 3 x 32 x 32 NumPy array and the saved probability-calibration temperature.
Housing inference takes named feature columns and reports benchmark target units.
Supplying its exported calibration metadata adds split-conformal intervals;
supplying the fitted quantile model also adds CQR intervals and explicit empty flags.
Adult metadata adds calibrated class-set membership. Metadata/checkpoint hashes
prevent mixing calibration with a different fitted model.

**Worked exercise.** A deterministic colorizer beats a generative one in DeltaE but
produces mostly gray clothing. Has it solved colorization better? Only for that exact
color-error objective. Examine appearance, chroma, possible alternative colors and
diversity under the same input, while preserving the ambiguity of the original colors.
