"""Generate the five English project walkthroughs, then execute in fresh kernels."""

from generate_notebooks import SETUP, write


def main():
    write(
        "13_pretrained_reference.ipynb",
        [
            (
                "m",
                """# Deploy a modern pretrained colorization reference
        Separate local training from externally trained models and compare the same test
        population. [DDColor protocol](../docs/11_image_restoration.md#frozen-ddcolor-reference)
        and [photo attribution](../docs/photo_attributions.md).""",
            ),
            ("c", SETUP),
            (
                "c",
                """external = evidence('ddcolor')
record = json.loads((root/'results/reference/ddcolor/run.json').read_text())
pd.Series({'source_revision':record['config']['source_revision'],
           'model_revision':record['config']['model_revision'],
           'weight_sha256':external['weight_sha256'],
           'parameters':external['parameters'],
           'images_per_second':external['images_per_second']})""",
            ),
            (
                "c",
                """local = evidence('colorization')
pd.DataFrame([{'model':r['model'],'seed':r.get('seed'),'psnr':r['psnr']['mean'],
              'delta_e76':r['delta_e76']['mean'],'chroma':r['mean_chroma']['mean']}
             for r in local['rows']+external['rows']])""",
            ),
            (
                "m",
                """## Compare paired errors without conflating exposure
        The same photographs enter both tests. ImageNet exposure and the released model's
        task-specific training are additional resources; it is not an equal-data comparison.
        Possible overlap with external training was not audited.""",
            ),
            (
                "c",
                """own_scores = np.load(root/'assets/source_data/colorization_per_image_scores.npz')
external_scores = np.load(root/'assets/source_data/ddcolor_per_image_scores.npz')
own = [r for r in local['rows'] if r['model']=='transfer_l2']
delta = external_scores[external['rows'][0]['per_image_psnr_key']] - np.mean([own_scores[r['per_image_psnr_key']] for r in own],axis=0)
print('Paired external-minus-local PSNR (dB):',delta.mean())
print('Fraction of images with higher external-model PSNR:',np.mean(delta>0))""",
            ),
            (
                "c",
                """from IPython.display import Image, display
display(Image(filename=str(root/'assets/ddcolor_gallery.png')))""",
            ),
            (
                "m",
                """## Worked answer
        Why upsample only lightness-derived gray for DDColor? Resizing RGB and then taking
        Lab lightness can let source chroma influence the input through a nonlinear color
        conversion. The adapter takes native-resolution L, reconstructs neutral gray and
        only then resizes. A 512px internal tensor does not supply additional observed detail.""",
            ),
        ],
    )
    write(
        "08_course_mixtures.ipynb",
        [
            (
                "m",
                """# Mixture geometry, latent classes and missing data
        Compare the supplied coordinate datasets with independent held-out selection.
        Read [the full derivations](../docs/10_course_projects.md) and [measured results](../docs/project_results.md).
        The synthetic coordinates are not clinical image data.""",
            ),
            ("c", SETUP),
            (
                "c",
                """m = evidence('course_mixtures')
rows = pd.DataFrame(m['rows'])
rows.groupby(['dataset','model'])[['test_ari','test_nll']].agg(['mean','std'])""",
            ),
            (
                "m",
                """## Geometry and identifiability
        The means separate in Image 1 and nearly coincide in Image 2. Covariance differences
        can help model density without making the generating label predictable. Labels
        below are used only to inspect the stated contiguous groups.""",
            ),
            (
                "c",
                """points = np.load(root / 'assets/source_data/assignment_points.npz')
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for i, ax in enumerate(axes, 1):
    x = points[f'image{i}']
    ax.scatter(*x.T, c=np.repeat(np.arange(3), 200), s=9, alpha=.6, cmap='viridis')
    ax.set(title=f'Image {i}', xlabel='Coordinate 1', ylabel='Coordinate 2')
    ax.set_aspect('equal')
plt.show()""",
            ),
            (
                "c",
                """pd.DataFrame(m['imputation']).groupby('dataset')[['conditional_gmm_rmse','mean_imputation_rmse']].mean()""",
            ),
            (
                "m",
                """## Categorical EM with missing features
        The posterior of a completely missing row equals the prior component weights.
        Smoothed updates count only observed entries. Inspect both model recovery and
        imputation rather than treating high training likelihood as sufficient evidence.""",
            ),
            (
                "c",
                """from learning_lab.categorical import CategoricalMixtureEM
x = np.array([[0,0],[0,1],[1,1],[1,-1],[-1,0]])
model = CategoricalMixtureEM(2,2,seed=3).fit(x)
np.testing.assert_allclose(model.predict_proba([[-1,-1]]),model.weights_[None])
pd.Series({key:value for key,value in m['categorical_mixture'].items() if key not in ['history','protocol','learned_weights']})""",
            ),
            (
                "m",
                """## Worked answer
        An unconstrained mixture of one categorical variable identifies only its weighted
        marginal, not each latent component. Multiple conditionally independent features
        supply observable dependence. The executed experiment uses eight features and 25%
        missing entries; its success is a controlled result, not a universal imputation guarantee.""",
            ),
        ],
    )
    write(
        "09_mixture_inverse.ipynb",
        [
            (
                "m",
                r"""# Linear inverse problems with a Gaussian-mixture prior
        The posterior remains a mixture. Observation evidence updates its weights,
        and total covariance includes within- and between-component uncertainty.
        [Derivation and archive audit](../docs/10_course_projects.md).""",
            ),
            ("c", SETUP),
            (
                "c",
                """from learning_lab.inverse import MixtureInverse, extract_patches, overlap_average
prior = MixtureInverse([.7,.3], [[-1.],[2.]], [[[.5]],[[1.]]], [[.8]], .3)
y = np.linspace(-2,3,150)[:,None]
p = prior.condition(y, return_covariance=True)
fig, ax = plt.subplots()
ax.plot(y[:,0],p['mmse'][:,0],label='MMSE')
ax.plot(y[:,0],p['hard'][:,0],label='Highest-weight component mean')
sd = np.sqrt(p['covariance'][:,0,0])
ax.fill_between(y[:,0],p['mmse'][:,0]-sd,p['mmse'][:,0]+sd,alpha=.15,label='One posterior SD')
ax.set(xlabel='Observation',ylabel='Estimated latent signal'); ax.legend(); plt.show()""",
            ),
            (
                "m",
                """The shaded band describes posterior standard deviation, not a calibrated
        confidence interval or a Gaussian approximation justified for every mixture.
        A highest-weight component mean is not the general mixture MAP.""",
            ),
            (
                "c",
                """x = np.random.default_rng(0).normal(size=(3,28,28))
for size in [4,8,12,16,20,28]:
    np.testing.assert_allclose(overlap_average(extract_patches(x,size), x.shape,size), x, atol=1e-13)
print('Patch overlap reconstruction verified, including all boundary pixels.')""",
            ),
            (
                "c",
                """m = evidence('inverse')
pd.DataFrame(m['controlled'])[['m','k','method','test_psnr']]""",
            ),
            (
                "c",
                """pd.DataFrame([{'actual_m':r['actual_m'],'raw_psnr':r['test_raw_psnr'],
              'nominal_mmse':r['test_mmse_psnr'],'calibrated_mmse':r['test_calibrated_psnr'],
              'residual_mean_255':r['observation_residual_mean_255']} for r in m['supplied']])""",
            ),
            (
                "m",
                """## Worked answer
        Why not score an observed patch under a clean-patch posterior? These distributions
        concern different variables. Component evidence is the marginal observation density
        with mean Aμ+b and covariance ACAᵀ+R. The supplied PNGs also violate the nominal
        zero-mean Gaussian model; calibration uses separate validation pairs, never test pairs.
        An individual test image may favor a hard estimator when the prior is misspecified.""",
            ),
        ],
    )
    write(
        "10_denoising.ipynb",
        [
            (
                "m",
                """# Denoising: linear priors, bottlenecks and residual learning
        Twenty-five epochs, three training seeds and 10,000 official test digits.
        [Protocols](../docs/11_image_restoration.md) explain units, clipping, selection and shift.""",
            ),
            ("c", SETUP),
            (
                "c",
                """m = evidence('denoising')
pd.DataFrame([{'model':r['model'],'seed':r.get('seed'),'psnr':r['psnr']['mean'],
              'ssim':r['ssim']['mean']} for r in m['rows'] if r.get('sigma_255')==30])""",
            ),
            (
                "m",
                """## Gaussian coefficient shrinkage
        For PCA coefficients with prior variance λ and isotropic noise variance σ²,
        the posterior-mean multiplier is λ/(λ+σ²). This shrinks weak directions most,
        but optimality depends on the prior. Ordinary projection can win on non-Gaussian digits.""",
            ),
            (
                "c",
                """variances = np.geomspace(1e-5,1,150)
for sigma in [10,30,50]:
    plt.semilogx(variances,variances/(variances+(sigma/255)**2),label=f'σ={sigma}')
plt.xlabel('PCA coefficient variance'); plt.ylabel('Shrinkage'); plt.legend(); plt.show()
pd.DataFrame(m['pca_selection'])""",
            ),
            (
                "c",
                """import torch
from learning_lab.restoration import ResidualDenoiser
torch.set_num_threads(2)
net = ResidualDenoiser()
inp = torch.rand(2,2,28,28)
torch.testing.assert_close(net(inp), inp[:,:1])
print('Zero-initialized residual head begins at the observation baseline.')""",
            ),
            (
                "c",
                """from IPython.display import Image, display
display(Image(filename=str(root/'assets/denoising_gallery.png')))
display(Image(filename=str(root/'assets/denoising_noise_sweep.png')))""",
            ),
            (
                "m",
                """## Worked answer
        Why avoid clipping the noisy input in the controlled test? Clipping produces a
        signal-dependent, non-Gaussian observation process near zero and one. Output
        clipping expresses the physical intensity range; it does not repair the noise model.
        The Poisson shift probe makes this assumption visible rather than changing the test
        protocol until every model looks strong.""",
            ),
        ],
    )
    write(
        "11_super_resolution.ipynb",
        [
            (
                "m",
                """# Super-resolution: evidence, degradation and paired comparisons
        Compare bicubic interpolation, a bottleneck autoencoder and a compact residual
        pixel-shuffle model under a shared 4× task. [Model design](../docs/11_image_restoration.md).
        Gallery photographs retain their [source attribution](../docs/photo_attributions.md).""",
            ),
            ("c", SETUP),
            (
                "c",
                """m = evidence('super_resolution')
pd.DataFrame([{'model':r['model'],'seed':r.get('seed'),'degradation':r['degradation'],
              'psnr':r['psnr']['mean'],'ssim':r['ssim']['mean']} for r in m['rows']])""",
            ),
            (
                "m",
                """## Pixel shuffle is a reshape with spatial semantics
        Four low-resolution feature channels become one channel at twice the resolution.
        The learned convolutions produce subpixel predictions; reshaping itself invents no information.""",
            ),
            (
                "c",
                """import torch
from torch.nn.functional import pixel_shuffle
features = torch.arange(16.).reshape(1,4,2,2)
print('Subpixel feature channels:',features[0].numpy())
print('Reshaped output:',pixel_shuffle(features,2)[0,0].numpy())""",
            ),
            (
                "c",
                """scores = np.load(root/'assets/source_data/super_resolution_per_image_scores.npz')
base = next(r for r in m['rows'] if r['model']=='bicubic' and r['degradation']=='bicubic')
learned = [r for r in m['rows'] if r['model']=='pixelshuffle_residual' and r['degradation']=='bicubic']
delta = np.mean([scores[r['per_image_psnr_key']] for r in learned],axis=0)-scores[base['per_image_psnr_key']]
rng = np.random.default_rng(7)
bootstrap = delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(1)
print('Mean paired improvement (dB):',delta.mean())
print('Image-bootstrap 95% interval:',np.quantile(bootstrap,[.025,.975]))""",
            ),
            (
                "c",
                """from IPython.display import Image, display
display(Image(filename=str(root/'assets/super_resolution_gallery.png')))""",
            ),
            (
                "m",
                """## Worked answer
        Why compare each test image with its own baseline? Texture and content affect PSNR
        substantially. Paired differences preserve that shared difficulty and isolate the
        model contrast. The interval above conditions on these two trained checkpoints;
        it does not estimate uncertainty over all possible training seeds or degradation models.""",
            ),
        ],
    )
    write(
        "12_colorization.ipynb",
        [
            (
                "m",
                """# Colorization: fidelity, diversity and ambiguous predictions
        Investigate grayscale/L2/transfer/balanced-bin predictions together.
        [Protocols and interpretation](../docs/11_image_restoration.md).
        Photos retain their [source attribution](../docs/photo_attributions.md).""",
            ),
            ("c", SETUP),
            (
                "c",
                """import torch
from learning_lab.restoration import rgb_to_lab, lab_to_rgb
rgb = torch.tensor([[[[1.]],[[0.]],[[0.]]]])
lab = rgb_to_lab(rgb)
print('D65 red CIELAB:',lab[0,:,0,0].numpy()*[100,110,110])
torch.testing.assert_close(lab_to_rgb(lab),rgb,atol=3e-6,rtol=0)""",
            ),
            (
                "m",
                """## A simple ambiguity example
        Two opposite chroma hypotheses with equal probability have a zero-chroma conditional
        mean. That prediction minimizes expected L2 error, even if it looks desaturated.""",
            ),
            (
                "c",
                """hypotheses = np.array([[40.,20.],[-40.,-20.]])
mean_chroma = hypotheses.mean(0)
print('Conditional mean:',mean_chroma)
print('Expected squared error of mean:',np.mean(np.sum((hypotheses-mean_chroma)**2,axis=1)))
print('Expected squared error of one hypothesis:',np.mean(np.sum((hypotheses-hypotheses[0])**2,axis=1)))""",
            ),
            (
                "c",
                """m = evidence('colorization')
pd.DataFrame([{'model':r['model'],'seed':r.get('seed'),'psnr':r['psnr']['mean'],
              'delta_e76':r['delta_e76']['mean'],'chroma':r['mean_chroma']['mean'],
              'true_chroma':r['true_mean_chroma']} for r in m['rows']])""",
            ),
            (
                "c",
                """from IPython.display import Image, display
display(Image(filename=str(root/'assets/colorization_gallery.png')))
display(Image(filename=str(root/'assets/colorization_entropy.png')))""",
            ),
            (
                "m",
                """## Worked answer
        Does high entropy mean the RGB color is wrong? No. Entropy is uncertainty over
        this model's palette and depends on temperature. The palette distribution is not
        calibrated to human plausibility or true-color frequency. Evaluate exact-color
        errors, chroma and fixed examples together, and disclose external ImageNet exposure.""",
            ),
        ],
    )


if __name__ == "__main__":
    main()
