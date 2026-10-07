import pytest

torch = pytest.importorskip("torch")

from learning_lab.generative import (  # noqa: E402
    ConditionalDiffusion,
    ConditionalVAE,
    alpha_schedule,
    build_model,
    gaussian_kl,
    observation,
    reconstruct,
)
from learning_lab.restoration import lab_to_rgb, rgb_to_lab  # noqa: E402


def test_lab_conversion_has_finite_gradients_outside_gamut():
    lab = torch.tensor([0.0, 1.5, -1.5]).reshape(1, 3, 1, 1).requires_grad_()
    lab_to_rgb(lab).sum().backward()
    assert torch.isfinite(lab.grad).all()
    black = torch.zeros(1, 3, 4, 4, requires_grad=True)
    rgb_to_lab(black).sum().backward()
    assert torch.isfinite(black.grad).all()


def test_observation_reconstruction_and_color_lightness():
    rgb = torch.rand(2, 3, 32, 32)
    for task in ["super_resolution", "colorization"]:
        condition, target = observation(rgb, task)
        restored = reconstruct(condition, target, task)
        assert torch.allclose(restored, rgb, atol=2e-5)
    light = rgb_to_lab(rgb)[:, :1]
    gray = reconstruct(light, torch.zeros(2, 2, 32, 32), "colorization")
    assert torch.allclose(rgb_to_lab(gray)[:, :1], light, atol=2e-5)


def test_conditional_gaussian_kl_matches_torch_distribution():
    qmean, pmean, qlog, plog = [torch.randn(4, 6) for _ in range(4)]
    q = torch.distributions.Normal(qmean, (qlog / 2).exp())
    p = torch.distributions.Normal(pmean, (plog / 2).exp())
    expected = torch.distributions.kl_divergence(q, p).sum(1).mean()
    assert torch.allclose(gaussian_kl(qmean, qlog, pmean, plog), expected)


def test_vae_prior_inference_does_not_require_target_and_samples_latent():
    torch.set_num_threads(2)
    model = ConditionalVAE("colorization", width=16, latent=8).eval()
    # Zero-initialized output needs a nonzero final layer to expose latent dependence.
    torch.nn.init.normal_(model.decoder.tail[-1].weight, std=0.03)
    condition = torch.rand(2, 1, 32, 32)
    a, kl = model(condition, generator=torch.Generator().manual_seed(7))
    b, _ = model(condition, generator=torch.Generator().manual_seed(8))
    assert a.shape == (2, 2, 32, 32)
    assert kl == 0
    assert not torch.allclose(a, b)
    _, posterior_kl = model(condition, torch.rand_like(a))
    assert torch.isfinite(posterior_kl) and posterior_kl >= 0


def test_diffusion_velocity_identity_and_seeded_sampling():
    torch.set_num_threads(2)
    schedule = alpha_schedule()
    assert (schedule > 0).all() and (schedule < 1).all()
    assert (schedule[1:] < schedule[:-1]).all()
    clean, noise = torch.randn(2, 2, 16, 16), torch.randn(2, 2, 16, 16)
    a = schedule[450]
    xt = a.sqrt() * clean + (1 - a).sqrt() * noise
    v = a.sqrt() * noise - (1 - a).sqrt() * clean
    assert torch.allclose(a.sqrt() * xt - (1 - a).sqrt() * v, clean, atol=1e-6)
    model = ConditionalDiffusion("colorization", width=16).eval()
    condition = torch.rand(2, 1, 16, 16)
    first = model.sample(condition, 4, torch.Generator().manual_seed(12))
    second = model.sample(condition, 4, torch.Generator().manual_seed(12))
    assert torch.equal(first, second)
    assert torch.isfinite(first).all() and first.abs().max() <= 1


def test_ddim_oracle_recovers_sr_targets_outside_chroma_support():
    class Oracle(ConditionalDiffusion):
        def forward(self, noisy, condition, timestep):
            a = self.alpha[timestep][:, None, None, None]
            return (a.sqrt() * noisy - 2.0) / (1 - a).sqrt()

    model = Oracle("super_resolution", width=16)
    condition = torch.zeros(1, 3, 16, 16)
    field = model.sample(condition, 8, torch.Generator().manual_seed(2))
    torch.testing.assert_close(field, torch.full_like(field, 2.0), atol=1e-5, rtol=0)


def test_frozen_vae_semantic_decoder_reloads_without_initialization_files():
    config = {
        "width": 16,
        "latent": 8,
        "semantic_decoder": True,
        "pretrained_weights": "/missing/encoder.pth",
        "mean_checkpoint": "/missing/mean.pt",
    }
    model = build_model("cvae", "colorization", config, initialize_pretrained=False)
    model.train()
    assert not model.semantic.training
    assert not any(p.requires_grad for p in model.semantic.parameters())
    before = model.semantic.encoder.bn1.running_mean.clone()
    model(torch.rand(2, 1, 32, 32), torch.rand(2, 2, 32, 32))
    torch.testing.assert_close(before, model.semantic.encoder.bn1.running_mean)
