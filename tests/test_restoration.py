import numpy as np
import pytest

torch = pytest.importorskip("torch")

from learning_lab.restoration import (  # noqa: E402
    Autoencoder,
    PixelShuffleSR,
    ResidualDenoiser,
    TransferColorizer,
    lab_to_rgb,
    per_image_metrics,
    rgb_to_lab,
)


def test_cielab_d65_reference_colors_and_roundtrip():
    colors = torch.tensor(
        [[[[1.0]], [[0.0]], [[0.0]]], [[[1.0]], [[1.0]], [[1.0]]], [[[0.0]], [[0.0]], [[0.0]]]]
    )
    lab = rgb_to_lab(colors)
    np.testing.assert_allclose(
        lab[0, :, 0, 0].numpy() * [100, 110, 110], [53.2408, 80.0925, 67.2032], atol=0.005
    )
    np.testing.assert_allclose(lab[1, :, 0, 0].numpy() * [100, 110, 110], [100, 0, 0], atol=0.005)
    torch.testing.assert_close(lab_to_rgb(lab), colors, atol=3e-6, rtol=0)
    random = torch.rand(4, 3, 12, 12)
    torch.testing.assert_close(lab_to_rgb(rgb_to_lab(random)), random, atol=4e-6, rtol=0)


def test_restoration_models_shapes_and_backward():
    torch.set_num_threads(2)
    for model, x, shape in [
        (Autoencoder(), torch.rand(2, 1, 28, 28), (2, 1, 28, 28)),
        (ResidualDenoiser(), torch.rand(2, 2, 28, 28), (2, 1, 28, 28)),
        (PixelShuffleSR(width=16, blocks=2), torch.rand(2, 3, 24, 24), (2, 3, 96, 96)),
        (TransferColorizer(outputs=8), torch.rand(2, 1, 96, 96), (2, 8, 96, 96)),
    ]:
        result = model(x)
        assert result.shape == shape
        result.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_quality_metrics_identity_and_known_constant_error():
    target = torch.full((2, 3, 16, 16), 0.5)
    same = per_image_metrics(target, target)
    np.testing.assert_allclose(same["ssim"], 1, atol=1e-6)
    values = per_image_metrics(target, target + 0.1, border=2)
    np.testing.assert_allclose(values["psnr"], 20, atol=1e-5)
    assert (values["ssim"] < 1).all()
