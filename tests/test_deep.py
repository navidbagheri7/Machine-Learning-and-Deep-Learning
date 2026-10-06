import numpy as np
import pytest

torch = pytest.importorskip("torch")

from learning_lab.neural import FeatureTransformer, TabularResNet
from learning_lab.signals import (
    Equalizer,
    channel,
    constellation,
    error_metrics,
    physics_inverse,
    windows,
)
from learning_lab.vision import ImageEncoder, augment, contrastive_loss


def test_gray_code_unit_power_and_adjacent_bit_distance():
    points, bits = constellation()
    assert np.isclose(np.mean(abs(points) ** 2), 1)
    grid = bits.reshape(4, 4, 4)
    assert np.all(np.sum(grid[1:] != grid[:-1], axis=-1) == 1)
    assert np.all(np.sum(grid[:, 1:] != grid[:, :-1], axis=-1) == 1)


def test_known_channel_inverse_recovers_noiseless_symbols():
    rng = np.random.default_rng(0)
    points, _ = constellation()
    s = points[rng.integers(16, size=(10, 64))]
    for gamma in [0, 0.2, 0.4]:
        r = channel(s, np.inf, gamma, rng)
        np.testing.assert_allclose(physics_inverse(r, gamma), s, atol=1e-10)


def test_signal_context_respects_block_boundaries_and_ber():
    x = np.zeros((2, 2, 64), dtype=np.float32)
    x[1] = 5
    assert not windows(x)[0].any()
    for kind in ["mlp", "cnn"]:
        assert Equalizer(kind)(torch.from_numpy(x)).shape == (2, 64, 16)
    assert error_metrics(np.zeros((2, 64), dtype=int), np.zeros((2, 64), dtype=int))["ber"] == 0


def test_neural_contracts_and_gradients():
    for model in [TabularResNet(8), FeatureTransformer(8)]:
        pred = model(torch.randn(5, 8))
        assert pred.shape == (5,)
        pred.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_vision_augmentation_and_positive_pair_objective():
    torch.manual_seed(0)
    x = torch.rand(4, 3, 32, 32)
    for contrastive in [True, False]:
        a = augment(x, contrastive)
        assert a.shape == x.shape and torch.isfinite(a).all()
    assert ImageEncoder()(a).shape == (4, 128)
    z = torch.eye(32)
    assert contrastive_loss(z, z) < contrastive_loss(z, torch.roll(z, 1, 0))
