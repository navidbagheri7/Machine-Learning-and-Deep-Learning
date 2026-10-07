import numpy as np
import pytest

torch = pytest.importorskip("torch")

from learning_lab.receiver_inference import receive  # noqa: E402
from learning_lab.signals import Equalizer  # noqa: E402


def test_receiver_schema_is_checked_before_checkpoint_access():
    for values in [
        np.zeros((2, 64, 2), dtype=np.float32),
        np.zeros((0, 2, 64), dtype=np.float32),
        np.zeros((2, 2, 64), dtype=np.uint8),
        np.full((2, 2, 64), np.nan),
    ]:
        with pytest.raises(ValueError, match="received blocks"):
            receive("/missing.pt", values)


def test_saved_receiver_gray_labels_batch_contract_and_invalid_logits(tmp_path):
    torch.set_num_threads(2)
    values = np.random.default_rng(9).normal(size=(3, 2, 64)).astype(np.float32)
    for kind in ["mlp", "cnn"]:
        model = Equalizer(kind)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.net[-1].bias[7] = 3
        checkpoint = tmp_path / f"{kind}.pt"
        torch.save({"state_dict": model.state_dict(), "kind": kind}, checkpoint)
        first = receive(checkpoint, values, batch_size=1)
        second = receive(checkpoint, values, batch_size=2)
        assert np.all(first["symbol_classes"] == 7)
        # Class7 is real-axis index1 (01), imaginary-axis index3 (10).
        np.testing.assert_array_equal(first["gray_bits"], np.tile([0, 1, 1, 0], (3, 64, 1)))
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        np.testing.assert_allclose(first["probabilities"].sum(-1), 1, atol=1e-6)
        assert np.all(first["probabilities"][..., 7] > 0.5)
        with torch.no_grad():
            model.net[-1].bias[7] = float("inf")
        torch.save({"state_dict": model.state_dict(), "kind": kind}, checkpoint)
        with pytest.raises(FloatingPointError, match="nonfinite logits"):
            receive(checkpoint, values)
