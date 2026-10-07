"""Independent objective and gradient-regularization identities."""

import pytest

torch = pytest.importorskip("torch")
from learning_lab.adversarial import critic_objective, gradient_penalty, r1_penalty  # noqa: E402


def test_wasserstein_penalty_uses_scalar_critic_gradient_with_fixed_condition():
    class LinearCritic(torch.nn.Module):
        def forward(self, condition, rgb):
            return 2 * rgb.flatten(1).sum(1) + 100 * condition.flatten(1).sum(1)

    real = torch.zeros(3, 3, 2, 2)
    fake = torch.ones_like(real)
    condition = torch.ones(3, 1, 2, 2)
    penalty, norm = gradient_penalty(LinearCritic(), condition, real, fake)
    expected = 2 * (12**0.5)
    assert norm.item() == pytest.approx(expected)
    assert penalty.item() == pytest.approx((expected - 1) ** 2)
    real.requires_grad_(True)
    assert r1_penalty(LinearCritic()(condition, real), real).item() == pytest.approx(48)


def test_objective_signs_reward_high_real_and_low_fake_scores():
    for method in ["hinge_sn", "non_saturating_r1", "least_squares", "wasserstein_gp"]:
        real = torch.tensor([0.2], requires_grad=True)
        fake = torch.tensor([0.1], requires_grad=True)
        loss = critic_objective(real, fake, method)
        a, b = torch.autograd.grad(loss, (real, fake))
        assert a.item() < 0 and b.item() > 0
