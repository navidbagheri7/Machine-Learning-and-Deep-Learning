"""Generate six English walkthroughs for the executed modern-method extension."""

import nbformat
from generate_notebooks import ROOT, write
from generate_notebooks import SETUP as BASE_SETUP

SETUP = (
    BASE_SETUP
    + """
from io import BytesIO
from PIL import Image as PILImage
from IPython.display import Image as NotebookImage, display
def preview(path):
    # Compressed display preview; full-resolution scientific figures remain in assets.
    with PILImage.open(path) as picture:
        picture.thumbnail((760,760))
        buffer = BytesIO()
        picture.convert('RGB').save(buffer,format='JPEG',quality=90)
    display(NotebookImage(data=buffer.getvalue()))
"""
)


def main():
    for number, task in [(14, "colorization"), (15, "super_resolution")]:
        write(
            f"{number}_generative_{task}.ipynb",
            [
                (
                    "m",
                    f"""# Conditional generative {task.replace("_", " ")}
Compare direct reconstruction, adversarial learning, a conditional VAE and diffusion
on identical 256px held-out photographs. [Measured results](../docs/modern_results.md)
and [complete protocol](../docs/12_generative_restoration.md). Every sample is included;
there is no closest-to-target selection.""",
                ),
                ("c", SETUP),
                (
                    "c",
                    f"""task = '{task}'
summary = json.loads((root/'assets/modern_summary.json').read_text())
expected_methods = {{'baseline','l1','gan','cvae','diffusion'}} | (
    {{'ddcolor','ct2','bigcolor'}} if task == 'colorization' else {{'swinir','realesrgan'}})
assert {{row['method'] for row in summary[task]['comparisons']}} == expected_methods
pd.DataFrame(summary[task]['comparisons'])""",
                ),
                (
                    "m",
                    """## Pair images, then distinguish samples from training seeds
An image bootstrap conditions on the fitted checkpoints. Averaging four samples
from one model does not produce four training seeds."""
                    + (
                        " The colorization CVAE seeds share a frozen semantic mean; their SD describes latent fitting, not end-to-end training."
                        if task == "colorization"
                        else " Both SR CVAE seeds fit their complete model independently."
                    ),
                ),
                ("c", "pd.DataFrame(summary[task]['paired'])"),
                (
                    "m",
                    """## Compare against a trained predictor and inspect score distributions
A generative method must also be compared with a strong direct predictor.
No examples are filtered out, and all methods use the same numerical scoring floor."""
                    + (
                        " A few almost achromatic photographs make the grayscale baseline unusually strong in mean PSNR; medians and quartiles reveal typical-image behavior."
                        if task == "colorization"
                        else " Report medians and quartiles as well as mean PSNR: image content and texture strongly affect reconstruction difficulty."
                    ),
                ),
                ("c", "pd.DataFrame(summary[task]['paired_direct'])"),
                ("c", "pd.DataFrame(summary[task]['image_distributions'])"),
                (
                    "m",
                    "## Resource context\nInference parameter counts include frozen encoders. Summed epoch times include validation and exclude initialization/export; the colorization CVAE additionally depends on its shared direct predictor. Metric-inclusive inference times come from concurrent reference workloads, not a controlled latency benchmark.",
                ),
                ("c", "pd.DataFrame(summary[task]['resources'])"),
                (
                    "m",
                    "## Does the latent variable affect the output?\nCompare training posterior KL, prior-based validation and held-out RGB draw dispersion. Tiny KL and negligible sample differences are consistent with weak latent use. They do not prove a mutual-information claim, and reconstruction alone does not demonstrate useful conditional generation.",
                ),
                ("c", "pd.DataFrame(summary[task]['latent_diagnostics'])"),
                (
                    "c",
                    "for path in sorted(root.glob(f'assets/generative_{task}_cvae_seed*_latent.png')):\n    preview(path)",
                ),
                (
                    "m",
                    """## Verify the Gaussian KL independently
The conditional prior is available at inference. The posterior observes the training
target and must never enter test inference. The feature term means the entire training
objective is not an exact evidence lower bound.""",
                ),
                (
                    "c",
                    """import torch
from learning_lab.generative import gaussian_kl
torch.set_num_threads(2)
qmean, pmean = torch.tensor([[1.,-1.]]), torch.zeros(1,2)
qlog, plog = torch.zeros(1,2), torch.zeros(1,2)
actual = gaussian_kl(qmean,qlog,pmean,plog)
expected = torch.distributions.kl_divergence(
    torch.distributions.Normal(qmean,(qlog/2).exp()),
    torch.distributions.Normal(pmean,(plog/2).exp())).sum(1).mean()
torch.testing.assert_close(actual,expected)
print('Independent analytic KL:',actual.item())""",
                ),
                (
                    "m",
                    """## Velocity prediction and the conditional observation
Under the cosine schedule, velocity and the noisy target form an invertible rotation.
The observation remains separate from the diffused field. SR conditions on an actual
bicubic LR observation; colorization conditions only on lightness.""",
                ),
                (
                    "c",
                    """from learning_lab.generative import alpha_schedule, observation, reconstruct
a = alpha_schedule()[500]
rng = torch.Generator().manual_seed(7)
clean = torch.randn(2,3,16,16,generator=rng)
noise = torch.randn(clean.shape,generator=rng)
noisy = a.sqrt()*clean + (1-a).sqrt()*noise
velocity = a.sqrt()*noise - (1-a).sqrt()*clean
torch.testing.assert_close(a.sqrt()*noisy-(1-a).sqrt()*velocity,clean)
rgb = torch.rand(2,3,32,32,generator=rng)
c, field = observation(rgb,task)
torch.testing.assert_close(reconstruct(c,field,task),rgb,atol=2e-5,rtol=0)
print('Velocity identity and observation reconstruction verified.')""",
                ),
                (
                    "c",
                    """from IPython.display import Image,display
figures = sorted(root.glob(f'assets/generative_{task}_*_*_gallery.png'))
references = ['ddcolor','ct2','bigcolor'] if task == 'colorization' else ['swinir','realesrgan']
for method in references:
    figures += sorted(root.glob(f'assets/external_{method}_*_gallery.png'))
assert figures
for path in figures:
    print(path.name)
    preview(path)""",
                ),
                (
                    "m",
                    """## Worked answer
Why can an adversarial or diffusion output look plausible but have lower PSNR? The
observation does not uniquely determine the target. Plausible alternatives can differ
from its pixels. This does not excuse weak outputs: inspect LPIPS, observation
consistency, fixed galleries and RGB sample dispersion together. Near-zero VAE
dispersion indicates limited stochastic behavior even when reconstruction is strong.
Neither these experiments nor their two local seeds establish a general method ranking.""",
                ),
            ],
        )
    path = ROOT / "notebooks/15_generative_super_resolution.ipynb"
    notebook = nbformat.read(path, as_version=4)
    notebook.cells.extend(
        [
            nbformat.v4.new_markdown_cell(
                "## Degradation mismatch\nFrozen transformer and adversarial references are also tested under shared blur and LR noise/JPEG compression. No test tuning or adaptation. Their bicubic reprojection metric is not a complete forward likelihood for these stressors."
            ),
            nbformat.v4.new_code_cell(
                "shifted = evidence('sr_shift')\npd.DataFrame([{'model':r['method'],'degradation':r['degradation'],**{k:r[k]['mean'] for k in ['psnr','ssim','lpips']}} for r in shifted['rows']])"
            ),
            nbformat.v4.new_code_cell(
                "for condition in ['bicubic','blur','noise_jpeg']:\n    preview(root/f'assets/sr_shift_{condition}_gallery.png')"
            ),
        ]
    )
    nbformat.write(notebook, path)
    write(
        "16_external_restoration.ipynb",
        [
            (
                "m",
                """# Frozen external references and deployment
Evaluate DDColor, CT2, BigColor, SwinIR and Real-ESRGAN on the same observations as the local models.
[Source pins, checkpoint hashes and licenses](../assets/external_restoration_manifest.json).
External sources and weights are not redistributed by this repository.""",
            ),
            ("c", SETUP),
            (
                "c",
                """manifest = json.loads((root/'assets/external_restoration_manifest.json').read_text())
pd.DataFrame(manifest['sources']).T""",
            ),
            (
                "c",
                """rows = []
for study in ['external_ddcolor','external_ct2','external_bigcolor','external_swinir','external_realesrgan']:
    m = evidence(study)
    rows.extend({'method':r['method'],'sample':r['sample'],
                 'psnr':r['psnr']['mean'],'lpips':r['lpips']['mean'],
                 'consistency':r['data_consistency_l1']['mean']} for r in m['rows'])
pd.DataFrame(rows)""",
            ),
            (
                "m",
                """## Exposure and adapters
BigColor uses the released encoder/generator and EMA with a fixed ResNet50 class
selector, differing from its default EfficientNet-L2. CT2 processes 256px lightness
and receives zero chroma in its diagnostic target argument. DDColor processes
512px internally, without extra observed detail. SwinIR receives actual LR rather
than a second downsampling of the enlarged image. Real-ESRGAN training uses blind degradations, differing from this controlled bicubic test. CT2 has a noncommercial license.""",
            ),
            (
                "c",
                """weights = pd.DataFrame(manifest['weights']).T
assert all(len(value)==64 for value in weights.sha256)
weights[['sha256','bytes']]""",
            ),
            (
                "c",
                """from IPython.display import Image,display
for method in ['ddcolor','ct2','bigcolor','swinir','realesrgan']:
    paths = sorted(root.glob(f'assets/external_{method}_*_gallery.png'))
    assert paths
    for path in paths:
        preview(path)""",
            ),
            (
                "m",
                """## Worked answer
Does a frozen model beating a local one establish a superior algorithm? This is a
practical pipeline comparison. Training data, architecture and compute all differ,
and external-data overlap has not been audited. State the adapter and exposure,
report measured values rather than quoting the paper's different benchmark numbers,
and provide a direct inference command with explicit preprocessing.""",
            ),
        ],
    )
    write(
        "17_native_boosting.ipynb",
        [
            (
                "m",
                """# Native categorical boosting, calibration and geographic shift
Extend Adult classification and California housing regression with CatBoost and
quantile uncertainty, preserving the earlier held-out protocols.
[Results](../docs/modern_results.md) and [split rationale](../docs/03_uncertainty_and_shift.md).""",
            ),
            ("c", SETUP),
            (
                "c",
                """m = evidence('modern_tabular')
pd.DataFrame(m['classification'])""",
            ),
            ("c", """pd.DataFrame(m['paired'])"""),
            (
                "m",
                "## Post-fit error analysis\nInspect calibration, subgroup coverage and permutation sensitivity using fixed checkpoints. This analysis never chooses a model or threshold; marginal coverage does not guarantee subgroup coverage.",
            ),
            (
                "c",
                "analysis = json.loads((root/'results/reference/modern_tabular/diagnostics.json').read_text())\npd.DataFrame(analysis['groups_seed0'])[['feature','group','n','roc_auc','ece','set_coverage','tpr','fpr']]",
            ),
            (
                "c",
                "preview(root/'assets/modern_tabular_diagnostics.png')\npd.DataFrame(analysis['permutation_seed0']).sort_values('NLL increase',ascending=False)",
            ),
            (
                "m",
                """## Keep tuning, probability calibration and coverage calibration separate
Depth and early stopping use validation labels. A separate probability partition
fits temperature. Another conformal partition chooses the cutoff. Decision costs
are an explicit illustrative choice, not a claim about income decisions in practice.""",
            ),
            (
                "c",
                """rows = []
for row in m['regression']:
    rows.append({k:v for k,v in row.items() if isinstance(v,(str,int,float))} |
                {'coverage':row['split_conformal']['coverage'],
                 'CQR coverage':row['cqr']['coverage']})
pd.DataFrame(rows)""",
            ),
            (
                "c",
                """from learning_lab.uncertainty import conformal_quantile
scores = np.arange(1.,10.)
q = conformal_quantile(scores,.2)
assert q == 8.
print('Finite-sample conformal order statistic:',q)
pd.DataFrame(m['selection']).head()""",
            ),
            (
                "m",
                """## Worked answer
Can stronger boosting repair the geographic coverage collapse? Better predictive
accuracy does not restore exchangeability between calibration and shifted test data.
The results retain this failure and compare random and geographic splits explicitly.
Native categories avoid fitting a global one-hot vocabulary; this does not make the
model immune to unseen categories, missingness shift or population changes.""",
            ),
        ],
    )
    write(
        "18_modern_transfer.ipynb",
        [
            (
                "m",
                """# Limited-label ConvNeXt transfer and robustness
Fine-tune a modern ImageNet encoder using the same 5,000 fitting labels per seed,
plus 2,000 validation and 3,000 calibration labels in disjoint partitions. The local
label-role total is 10,000. Stratified benchmark sampling uses available class
annotations; this is not a total annotation-acquisition claim.
[Measured comparisons](../docs/modern_results.md).
Resizing 32px data to 128px supplies no additional observed information.""",
            ),
            ("c", SETUP),
            (
                "c",
                """m = evidence('modern_vision')
pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,list)} for r in m['rows']])""",
            ),
            (
                "c",
                """accuracy = np.array([r['accuracy'] for r in m['rows']])
print('Accuracy mean and descriptive training-seed SD:',accuracy.mean(),accuracy.std(ddof=1))
pd.DataFrame([r['per_class_accuracy'] for r in m['rows']],index=[0,1,2],
             columns=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])""",
            ),
            (
                "m",
                """## Confirm checkpoint selection
Mixup and label smoothing affect the training objective, while unsmoothed validation
NLL chooses checkpoints. Temperature then uses separate probability-calibration
labels. Test accuracy never chooses an epoch or a temperature.""",
            ),
            (
                "c",
                """for curve in m['curves']:
    best = min(curve['history'],key=lambda r:r['validation_nll'])
    assert best['epoch']==curve['best_epoch']
print('Every selected epoch minimizes its recorded validation NLL.')""",
            ),
            (
                "c",
                """from IPython.display import Image,display
preview(root/'assets/modern_vision_training.png')
pd.DataFrame([{'seed':r['seed'],'clean':r['accuracy'],'Gaussian noise':r['gaussian_noise_accuracy']}
              for r in m['rows']]).set_index('seed').plot.bar(ylim=(0,1),ylabel='Accuracy')
plt.show()""",
            ),
            (
                "m",
                """## Worked answer
Why does high clean accuracy coexist with poor corrupted accuracy? Ordinary
fine-tuning optimizes the training distribution, and probability calibration does
not train invariance to corruption. This example establishes a strong practical
transfer baseline with a measurable weakness. It does not attribute all gains to
the ConvNeXt architecture, since ImageNet pretraining is additional exposure.""",
            ),
            (
                "m",
                "## Class errors and confidence\nPair predictions on identical official test examples. Intervals condition on the fitted models. Inspect all three confusion matrices rather than selecting the most favorable seed.",
            ),
            (
                "c",
                "analysis = json.loads((root/'results/reference/modern_vision/diagnostics.json').read_text())\npd.DataFrame(analysis['paired'])",
            ),
            (
                "c",
                "preview(root/'assets/modern_vision_confusion.png')\npreview(root/'assets/modern_vision_reliability.png')",
            ),
            (
                "m",
                "## Deployable prediction sets\nCUDA and CPU use separate precision-bound calibration metadata. All three checkpoints retain their original independent partitions. Empty sets are explicit; singleton accuracy is descriptive, with no conditional guarantee.",
            ),
            (
                "c",
                """calibration = json.loads((root/'results/reference/modern_vision/cifar_calibration_audit.json').read_text())
assert calibration['status']=='passed' and len(calibration['rows'])==6
pd.DataFrame(calibration['rows'])[['seed','profile','temperature','set_coverage','set_size','empty_set_rate','singleton_accuracy']]""",
            ),
            (
                "c",
                """for row in calibration['rows']:
    stem = f\"{row['profile']}_{row['seed']}\"
    with np.load(root/f'assets/source_data/modern_vision_cifar_calibration_{stem}.npz') as saved:
        q = np.sort(saved['cal_scores'])[int(np.ceil((len(saved['cal_scores'])+1)*0.9))-1]
        assert np.isclose(q,row['cutoff'])
        sets = 1-saved['p'] <= q
        assert np.isclose(sets[np.arange(len(saved['y'])),saved['y']].mean(),row['set_coverage'])
        assert np.isclose((sets.sum(1)==0).mean(),row['empty_set_rate'])
print('All six cutoffs, test coverages and empty-set rates independently reproduced.')""",
            ),
        ],
    )
    write(
        "19_natural_denoising.ipynb",
        [
            (
                "m",
                """# Natural-image denoising and noise mismatch
Compare BayesShrink wavelets, the noisy baseline and frozen SwinIR on the same 400
COCO test images. The model was trained externally at sigma 25; sigma 15 and 50 test
noise shift. [Full protocol](../docs/12_generative_restoration.md).""",
            ),
            ("c", SETUP),
            (
                "c",
                """m = evidence('natural_denoising')
pd.DataFrame([{'model':r['model'],'sigma':r['sigma'],
              **{k:r[k]['mean'] for k in ['psnr','ssim','lpips']}} for r in m['rows']])""",
            ),
            (
                "m",
                """## Shared noise and paired error
Whole-image Gaussian noise is generated once and scaled by sigma. It is shared
across methods and not clipped before inference. Predictions are clipped for metrics.
The wavelet method receives the known sigma, whereas SwinIR stays frozen at sigma25.""",
            ),
            (
                "c",
                """scores = np.load(root/'assets/source_data/natural_denoising_per_image_scores.npz')
rows = []
rng = np.random.default_rng(8)
for sigma in [15,25,50]:
    delta = scores[f'swinir_noise25_sigma{sigma}_psnr']-scores[f'wavelet_sigma{sigma}_psnr']
    sampled = delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(1)
    rows.append({'sigma':sigma,'paired PSNR difference':delta.mean(),
                 'CI lower':np.quantile(sampled,.025),'CI upper':np.quantile(sampled,.975)})
pd.DataFrame(rows)""",
            ),
            (
                "c",
                """from IPython.display import Image,display
for sigma in [15,25,50]:
    preview(root/f'assets/natural_denoising_sigma{sigma}_gallery.png')""",
            ),
            (
                "m",
                """## Worked answer
Why retain the sigma50 result where the wavelet baseline wins? It exposes the
learned model's sensitivity to its observation model. Removing it would hide a
deployment failure. A future remedy could condition on sigma or train across noise
levels, but neither remedy is claimed as an executed result in this study.""",
            ),
        ],
    )


if __name__ == "__main__":
    main()
